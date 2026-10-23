"""The encrypted cache: real sealing and opening, a fake Hub, no network."""

from __future__ import annotations

import hashlib
import io
import os
import random
from pathlib import Path

import pytest
from cache_support import FakeHub

from megatoken import cache

SEGMENT = 1024
PIECE = 4096


def keys(seed: int = 1) -> cache.Keys:
    return cache.Keys(hashlib.sha256(f"key-{seed}".encode()).digest())


def seal_bytes(data: bytes, key: cache.Keys, aad: bytes = b"a", segment: int = SEGMENT) -> bytes:
    sink = io.BytesIO()
    cache.encrypt_stream(io.BytesIO(data), sink, key, aad, segment)
    return sink.getvalue()


def open_all(blob: bytes, key: cache.Keys, aad: bytes = b"a") -> bytes:
    return b"".join(cache.decrypt_stream(io.BytesIO(blob), key, aad))


# ---- sealing ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "size", [0, 1, SEGMENT - 1, SEGMENT, SEGMENT + 1, 3 * SEGMENT, 3 * SEGMENT + 5]
)
def test_every_size_round_trips_to_the_size_cipher_size_predicts(size):
    data = random.Random(size).randbytes(size)
    blob = seal_bytes(data, keys())
    assert open_all(blob, keys()) == data
    assert len(blob) == cache.cipher_size(size, SEGMENT)
    if size >= 16:  # A shorter run of plaintext turns up in random ciphertext by chance.
        assert data[:64] not in blob


def test_equal_inputs_seal_differently_each_time():
    assert seal_bytes(b"x" * 100, keys()) != seal_bytes(b"x" * 100, keys())


@pytest.mark.parametrize("where", ["header", "first", "middle", "last"])
def test_a_changed_byte_is_detected(where):
    blob = bytearray(seal_bytes(random.Random(3).randbytes(3 * SEGMENT + 7), keys()))
    position = {
        "header": 10,
        "first": cache.HEADER_BYTES + 3,
        "middle": cache.HEADER_BYTES + SEGMENT + cache.TAG_BYTES + 5,
        "last": len(blob) - 1,
    }[where]
    blob[position] ^= 1
    with pytest.raises(cache.CacheError):
        open_all(bytes(blob), keys())


def test_cut_off_extended_reordered_and_foreign_files_do_not_open():
    data = random.Random(4).randbytes(3 * SEGMENT)
    blob = seal_bytes(data, keys())
    step = SEGMENT + cache.TAG_BYTES
    head = blob[: cache.HEADER_BYTES]
    with pytest.raises(cache.CacheError):  # cut exactly at a segment boundary
        open_all(blob[: cache.HEADER_BYTES + 2 * step], keys())
    with pytest.raises(cache.CacheError):
        open_all(blob[: cache.HEADER_BYTES], keys())
    with pytest.raises(cache.CacheError):
        open_all(blob + b"\0", keys())
    swapped = head + blob[cache.HEADER_BYTES + step : cache.HEADER_BYTES + 2 * step]
    swapped += blob[cache.HEADER_BYTES : cache.HEADER_BYTES + step] + blob[-step - 16 :]
    with pytest.raises(cache.CacheError):
        open_all(swapped[: len(blob)], keys())
    with pytest.raises(cache.CacheError, match="Not an encrypted"):
        open_all(b"plain text, not sealed at all" * 5, keys())


def test_the_wrong_key_and_another_files_name_do_not_open():
    blob = seal_bytes(b"secret rows" * 300, keys(1), aad=b"piece-a")
    with pytest.raises(cache.CacheError, match="key is wrong"):
        open_all(blob, keys(2), aad=b"piece-a")
    with pytest.raises(cache.CacheError):  # the same bytes under another file's name
        open_all(blob, keys(1), aad=b"piece-b")


def test_keygen_makes_a_private_key_and_never_replaces_it(tmp_path):
    path = cache.keygen(tmp_path / "config" / "cache.key")
    assert path.stat().st_mode & 0o777 == 0o600
    first = path.read_text()
    assert len(bytes.fromhex(first.strip())) == 32
    with pytest.raises(cache.CacheError, match="never replaced"):
        cache.keygen(path)
    assert path.read_text() == first
    assert cache.load_keys(path).label("x") == cache.load_keys(path).label("x")
    path.chmod(0o640)
    with pytest.raises(cache.CacheError, match="readable by others"):
        cache.load_keys(path)


def test_names_are_repeatable_per_key_and_reveal_nothing():
    a, b = keys(1), keys(2)
    assert a.label("piece/prepare-full-agentic/0/0") == a.label("piece/prepare-full-agentic/0/0")
    assert a.label("piece/x/0/0") != b.label("piece/x/0/0")
    folder = cache.folder_name(a, "prepare-full-agentic", 0)
    assert folder.startswith("encrypted/") and "agentic" not in folder
    assert folder != cache.folder_name(a, "prepare-full-agentic", 1)
    assert folder != cache.folder_name(b, "prepare-full-agentic", 0)


# ---- a part, stored and fetched -----------------------------------------------------------

REPO = "someone/data"


def make_source(root: Path, seed: int = 0) -> dict[str, bytes]:
    """Two prepared roots, one with a file that spans pieces, an empty file and a working database."""
    rng = random.Random(seed)
    files = {
        "00000/manifest.json": b'{"complete": true, "source": "secret-dataset-name"}',
        "00000/shard-000.jsonl": rng.randbytes(9000),
        "00000/index.sqlite": rng.randbytes(3000),
        "00000/rejections.json": b"",
        "00001/manifest.json": b'{"complete": true}',
        "00001/shard-000.jsonl": rng.randbytes(5000),
    }
    for name, data in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_bytes(data)
    (root / "00000/preparation.sqlite").write_bytes(b"working store" * 100)
    (root / "00001/digest.sqlite").write_bytes(b"digest")
    return files


def store_part(hub, tmp_path, key=None, *, part=0, source="src", work="work", **kwargs):
    source_dir = tmp_path / source
    source_dir.mkdir(parents=True, exist_ok=True)
    if not any(source_dir.iterdir()):
        make_source(source_dir)
    options = {"piece_bytes": PIECE, "segment": SEGMENT, "log": lambda _: None}
    return cache.store(
        keys=key or keys(),
        api=hub,
        repo=REPO,
        run="prepare-full-agentic",
        part=part,
        source=source_dir,
        work=tmp_path / work,
        **{**options, **kwargs},
    )


def fetch_part(hub, receipt, dest, key=None, *, revision=None, **kwargs):
    return cache.fetch_part(
        keys=key or keys(),
        api=hub,
        repo=receipt["repo"],
        revision=revision or hub.head[receipt["repo"]],
        run=receipt["run"],
        part=receipt["part"],
        folder=receipt["folder"],
        dest=dest,
        scratch=dest.with_name(dest.name + ".scratch"),
        log=lambda _: None,
        **kwargs,
    )


def read_tree(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def repo_files(hub):
    return hub.commits[REPO][hub.head[REPO]]


def test_a_stored_part_comes_back_identical_and_the_repo_reveals_nothing(tmp_path):
    hub = FakeHub()
    receipt = store_part(hub, tmp_path)
    expected = {
        name: data
        for name, data in read_tree(tmp_path / "src").items()
        if Path(name).name not in cache.SKIP_NAMES
    }
    assert receipt["files"] == len(expected)
    assert receipt["stream_bytes"] == sum(map(len, expected.values()))
    assert receipt["pieces"] == 5 and receipt["folder"].startswith("encrypted/")
    assert all(private is False for _, private in hub.created)
    # Public bytes: nothing but fixed-size ciphertext under opaque names, in the part's folder.
    files = repo_files(hub)
    assert len(files) == 6  # five pieces and the manifest
    for name, data in files.items():
        folder, _, leaf = name.rpartition("/")
        assert folder == receipt["folder"]
        assert leaf.endswith(".bin") and len(leaf) == 36  # 32 hex characters of HMAC
        assert b"secret-dataset-name" not in data and b"00000" not in data
        assert "agentic" not in name
    # Every piece has the same size; only the manifest (padded to a megabyte) differs.
    sizes = {len(data) for data in files.values()}
    assert len(sizes) == 2 and cache.cipher_size(PIECE, SEGMENT) in sizes
    target = tmp_path / "restored"
    summary = fetch_part(hub, receipt, target)
    assert summary["files"] == len(expected)
    assert read_tree(target) == expected
    assert not (tmp_path / "restored.scratch").exists() or not read_tree(
        tmp_path / "restored.scratch"
    )


def test_only_the_named_roots_are_stored(tmp_path):
    hub = FakeHub()
    receipt = store_part(hub, tmp_path, only=["00001"])
    target = tmp_path / "restored"
    fetch_part(hub, receipt, target)
    assert set(read_tree(target)) == {"00001/manifest.json", "00001/shard-000.jsonl"}


def test_parts_land_in_their_own_folders_and_restore_side_by_side(tmp_path):
    hub = FakeHub()
    first = store_part(hub, tmp_path, part=0, source="a", work="wa")
    (tmp_path / "b").mkdir()
    (tmp_path / "b/00007").mkdir()
    (tmp_path / "b/00007/shard.jsonl").write_bytes(random.Random(9).randbytes(6000))
    second = store_part(hub, tmp_path, part=1, source="b", work="wb")
    assert first["folder"] != second["folder"]
    target = tmp_path / "all"
    revision = hub.head[REPO]
    fetch_part(hub, first, target, revision=revision)
    fetch_part(hub, second, target, revision=revision)
    assert (target / "00007/shard.jsonl").read_bytes() == (
        tmp_path / "b/00007/shard.jsonl"
    ).read_bytes()
    assert (target / "00001/shard-000.jsonl").is_file()


def test_storing_again_reseals_nothing_unless_the_files_changed(tmp_path, monkeypatch):
    hub = FakeHub()
    calls = []
    real = cache.encrypt_stream
    monkeypatch.setattr(
        cache, "encrypt_stream", lambda *args, **kw: calls.append(args[3]) or real(*args, **kw)
    )
    first = store_part(hub, tmp_path)
    sealed = len(calls)
    assert sealed == first["pieces"] + 1  # every piece, then the manifest
    second = store_part(hub, tmp_path)
    assert len(calls) == sealed + 1  # only the manifest is written again
    assert second["folder"] == first["folder"]
    (tmp_path / "src/00001/shard-000.jsonl").write_bytes(random.Random(5).randbytes(5000))
    store_part(hub, tmp_path)
    assert len(calls) > sealed + 1 + first["pieces"] - 1  # the plan changed: all resealed


def test_a_part_with_nothing_to_store_is_refused(tmp_path):
    (tmp_path / "empty").mkdir()
    (tmp_path / "empty/preparation.sqlite").write_bytes(b"x")
    with pytest.raises(cache.CacheError, match="no files"):
        store_part(FakeHub(), tmp_path, source="empty")


def test_an_upload_that_did_not_arrive_intact_is_not_reported_as_stored(tmp_path):
    hub = FakeHub()
    real = hub.upload_large_folder

    def lossy(*, repo_id, folder_path, repo_type):
        real(repo_id=repo_id, folder_path=folder_path, repo_type=repo_type)
        files = repo_files(hub)
        name = next(iter(files))
        files[name] = files[name][:-1] + bytes([files[name][-1] ^ 1])

    hub.upload_large_folder = lossy
    with pytest.raises(cache.CacheError, match="differs|intact"):
        store_part(hub, tmp_path)


def test_a_commit_that_races_with_another_is_repeated(tmp_path, monkeypatch):
    from megatoken import hub as hub_module

    monkeypatch.setattr(hub_module.time, "sleep", lambda _: None)
    hub = FakeHub()
    hub.conflicts = 2
    receipt = store_part(hub, tmp_path)
    assert hub.refused == 2 and len(repo_files(hub)) == 6
    assert receipt["folder"] in next(iter(repo_files(hub)))


def stored(tmp_path):
    hub = FakeHub()
    return hub, store_part(hub, tmp_path)


def corrupt_a_piece(hub, receipt, *, drop=False):
    files = repo_files(hub)
    manifest = cache.manifest_file(keys(), receipt["run"], 0)
    name = sorted(n for n in files if not n.endswith(manifest))[0]
    if drop:
        del files[name]
    else:
        files[name] = files[name][:100] + bytes([files[name][100] ^ 1]) + files[name][101:]


def test_fetch_refuses_an_altered_or_missing_piece(tmp_path):
    hub, receipt = stored(tmp_path)
    corrupt_a_piece(hub, receipt)
    with pytest.raises(cache.CacheError, match="does not match its hash"):
        fetch_part(hub, receipt, tmp_path / "out")
    hub, receipt = stored(tmp_path / "again")
    corrupt_a_piece(hub, receipt, drop=True)
    with pytest.raises(cache.CacheError, match="is missing"):
        fetch_part(hub, receipt, tmp_path / "out2")


def test_fetch_with_the_wrong_key_fails_before_writing_anything(tmp_path):
    hub, receipt = stored(tmp_path)
    with pytest.raises(cache.CacheError, match="is missing"):  # another key names other files
        fetch_part(hub, receipt, tmp_path / "out", keys(2))
    assert read_tree(tmp_path / "out") == {}


def test_a_file_whose_decrypted_hash_differs_from_the_manifest_is_never_named(
    tmp_path, monkeypatch
):
    real = cache.scan

    def lying(source, only=None):
        entries = real(source, only)
        return [
            cache.Entry(e.path, e.offset, e.bytes, "0" * 64)
            if e.path.endswith("index.sqlite")
            else e
            for e in entries
        ]

    monkeypatch.setattr(cache, "scan", lying)
    hub, receipt = stored(tmp_path)
    with pytest.raises(cache.CacheError, match="recorded hash"):
        fetch_part(hub, receipt, tmp_path / "out")
    assert not (tmp_path / "out/00000/index.sqlite").exists()
    assert not list((tmp_path / "out").rglob("*.partial"))


def test_a_rerun_fetch_downloads_only_what_is_still_missing(tmp_path):
    hub, receipt = stored(tmp_path)
    target = tmp_path / "out"
    fetch_part(hub, receipt, target)
    hub.downloads.clear()
    fetch_part(hub, receipt, target)
    assert len(hub.downloads) == 1  # the manifest alone
    (target / "00001/shard-000.jsonl").unlink()
    (target / "00000/manifest.json").write_bytes(b"damaged")
    hub.downloads.clear()
    fetch_part(hub, receipt, target)
    expected = read_tree(tmp_path / "src")
    assert (target / "00001/shard-000.jsonl").read_bytes() == expected["00001/shard-000.jsonl"]
    assert (target / "00000/manifest.json").read_bytes() == expected["00000/manifest.json"]
    assert 1 < len(hub.downloads) < 8  # not every piece again


def test_the_unsafe_paths_of_a_forged_manifest_are_refused():
    for bad in ("../x", "/etc/passwd", "a/../../b", ""):
        with pytest.raises(cache.CacheError, match="Unsafe"):
            cache._safe_relative(bad)


def verify(hub, receipt, tmp_path, key=None, revision=None):
    return cache.verify_part(
        hub,
        key or keys(),
        repo=receipt["repo"],
        revision=revision or hub.head[receipt["repo"]],
        run=receipt["run"],
        part=receipt["part"],
        folder=receipt["folder"],
        scratch=tmp_path / "verify",
    )


def test_a_part_verifies_without_downloading_its_pieces(tmp_path):
    hub, receipt = stored(tmp_path)
    hub.downloads.clear()
    manifest = verify(hub, receipt, tmp_path)
    assert len(manifest["pieces"]) == 5
    # The manifest and the small pieces are read; nothing is restored.
    assert not (tmp_path / "out").exists()
    with pytest.raises(cache.CacheError):
        verify(hub, receipt, tmp_path, keys(2))
    corrupt_a_piece(hub, receipt)
    with pytest.raises(cache.CacheError, match="does not match"):
        verify(hub, receipt, tmp_path)
    hub, receipt = stored(tmp_path / "x")
    corrupt_a_piece(hub, receipt, drop=True)
    with pytest.raises(cache.CacheError, match="missing"):
        verify(hub, receipt, tmp_path)


def test_the_key_is_found_in_the_sandbox_secrets_or_the_local_file(tmp_path, monkeypatch):
    monkeypatch.delenv("MEGATOKEN_CACHE_KEY_FILE", raising=False)
    secrets = tmp_path / "secrets"
    secrets.mkdir()
    (secrets / cache.SECRET_NAME).write_text(os.urandom(32).hex())
    (secrets / cache.SECRET_NAME).chmod(0o644)  # a sandbox's own directory is private already
    monkeypatch.setenv("MEGATOKEN_CW_SECRETS", str(secrets))
    assert cache.key_path() == secrets / cache.SECRET_NAME
    assert len(cache.read_key_file()) == 32


def test_a_rebuilt_sandbox_seals_the_same_bytes_so_the_hub_skips_them(tmp_path):
    first, second = FakeHub(), FakeHub()
    store_part(first, tmp_path, work="w1")
    store_part(second, tmp_path, work="w2")  # a fresh work directory: nothing to resume
    manifest = cache.manifest_file(keys(), "prepare-full-agentic", 0)

    def pieces(hub):
        return {n: d for n, d in repo_files(hub).items() if not n.endswith(manifest)}

    assert pieces(first) == pieces(second) and pieces(first)


def test_the_calls_the_cache_makes_match_the_real_huggingface_api():
    import inspect

    hub = pytest.importorskip("huggingface_hub")
    shapes = {
        "create_repo": dict(repo_id="r", repo_type="dataset", private=False, exist_ok=True),
        "upload_large_folder": dict(repo_id="r", folder_path="f", repo_type="dataset"),
        "create_commit": dict(repo_id="r", operations=[], commit_message="m", repo_type="dataset"),
        "dataset_info": dict(repo_id="r", revision="x"),
        "list_repo_tree": dict(
            repo_id="r",
            path_in_repo="p",
            repo_type="dataset",
            revision="x",
            recursive=True,
            expand=True,
        ),
        "hf_hub_download": dict(
            repo_id="r", filename="n", repo_type="dataset", revision="x", local_dir="d"
        ),
    }
    for name, arguments in shapes.items():
        inspect.signature(getattr(hub.HfApi, name)).bind(None, **arguments)


def test_rotating_the_key_means_restoring_and_storing_again_under_the_new_one(tmp_path):
    hub = FakeHub()
    old = store_part(hub, tmp_path, keys(1))
    plain = tmp_path / "plain"
    fetch_part(hub, old, plain, keys(1))
    new = store_part(hub, tmp_path, keys(2), source="plain", work="again")
    assert old["folder"] != new["folder"]  # other key, other names: the old folder can be deleted
    restored = tmp_path / "restored"
    fetch_part(hub, new, restored, keys(2))
    assert read_tree(restored) == read_tree(plain)
    files = repo_files(hub)
    for name in [n for n in files if n.startswith(old["folder"])]:
        del files[name]  # once the old folder is deleted the old key opens nothing
    with pytest.raises(cache.CacheError):
        verify(hub, old, tmp_path, keys(1))
    assert verify(hub, new, tmp_path, keys(2))


def test_the_command_line_makes_a_key_once_and_never_prints_it(tmp_path, capsys):
    from megatoken import cli

    path = tmp_path / "cache.key"
    assert cli.main(["cache", "keygen", "--path", str(path)]) == 0
    shown = capsys.readouterr().out
    assert str(path) in shown and path.read_text().strip() not in shown
    with pytest.raises(SystemExit) as refused:
        cli.main(["cache", "keygen", "--path", str(path)])
    assert refused.value.code == 1
