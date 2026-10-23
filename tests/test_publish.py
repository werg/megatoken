"""Publishing prepared roots to one repository, and reading them back (a fake Hub, no network)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from cache_support import FakeHub

from megatoken import cache, publish
from megatoken.data import PreparedCorpus, prepare_sources, write_demo_source
from megatoken.generation.codec import ByteCodec

REPO = "someone/megatoken-data"
RUN = "unit"
KEY = cache.Keys(bytes(range(32)))


def make_root(tmp_path: Path, name: str, count: int = 7, source_id: str | None = None) -> Path:
    """A real prepared root (several shards) of demo episodes."""
    source = write_demo_source(tmp_path / f"{name}.jsonl", count=count)
    if source_id:
        from dataclasses import replace

        source = replace(source, source_id=source_id)
    root = tmp_path / "prepared" / name
    prepare_sources([source], root, ByteCodec.byte_fallback(), shard_bytes=1500).close()
    return root


def sources_of(*ids: str) -> dict[str, dict[str, str]]:
    return {
        source_id: {
            "source_id": source_id,
            "upstream": f"https://huggingface.co/datasets/example/{source_id}",
            "revision": "a" * 40,
            "license": "cc-by-4.0",
            "license_url": f"https://example.org/{source_id}",
        }
        for source_id in ids
    }


def tree(root: Path) -> dict[str, bytes]:
    return {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def listing(directory: Path) -> dict[str, Path]:
    return {
        p.relative_to(directory).as_posix(): p for p in sorted(directory.rglob("*")) if p.is_file()
    }


# ---- one root, as Parquet and back ---------------------------------------------------------


def test_a_root_becomes_readable_parquet_and_comes_back_byte_for_byte(tmp_path):
    root = make_root(tmp_path, "00000")
    manifest = json.loads((root / "manifest.json").read_text())
    assert len(manifest["shards"]) > 1
    out = tmp_path / "out"
    counts = publish.convert_root(root, out)
    (source_id,) = counts
    assert counts[source_id]["episodes"] == manifest["counts"]["*"]
    files = listing(out)
    assert "data-00000.parquet" in files and "root/manifest.json" in files
    assert not any("preparation.sqlite" in name for name in files)
    import pyarrow.parquet as pq

    table = pq.read_table(files["data-00000.parquet"])
    assert table.num_rows == manifest["counts"]["*"]
    row = table.slice(0, 1).to_pylist()[0]
    assert row["source_id"] == source_id and row["messages"] and row["messages"][0]["text"]
    assert json.loads(row["episode_json"])["episode_id"] == row["episode_id"]
    restored = tmp_path / "restored"
    assert publish.restore_root(files, restored) == table.num_rows
    original = {n: d for n, d in tree(root).items() if Path(n).name not in cache.SKIP_NAMES}
    assert tree(restored) == original
    with PreparedCorpus([restored], split=None) as corpus:
        assert len(corpus) == table.num_rows


def test_a_root_in_many_parquet_files_still_restores_in_order(tmp_path, monkeypatch):
    monkeypatch.setattr(publish, "ROW_GROUP_ROWS", 2)
    root = make_root(tmp_path, "00000", count=9)
    out = tmp_path / "out"
    publish.convert_root(root, out, target=1)  # every row group closes its file
    names = sorted(n for n in listing(out) if n.endswith(".parquet"))
    assert len(names) == 5
    restored = tmp_path / "restored"
    publish.restore_root(listing(out), restored)
    assert tree(restored) == {
        n: d for n, d in tree(root).items() if Path(n).name not in cache.SKIP_NAMES
    }


def test_a_damaged_published_root_is_not_restored(tmp_path, monkeypatch):
    monkeypatch.setattr(publish, "ROW_GROUP_ROWS", 2)
    root = make_root(tmp_path, "00000", count=9)
    out = tmp_path / "out"
    publish.convert_root(root, out, target=1)
    files = listing(out)
    missing = {n: p for n, p in files.items() if n != "data-00001.parquet"}
    with pytest.raises(publish.PublishError, match="does not match|do not cover"):
        publish.restore_root(missing, tmp_path / "a")
    assert not (tmp_path / "a/manifest.json").exists()  # the commit marker is written last
    (out / "root/index.sqlite").write_bytes(b"not the index")
    with pytest.raises(publish.PublishError, match="index"):
        publish.restore_root(listing(out), tmp_path / "b")
    with pytest.raises(publish.PublishError, match="no manifest"):
        publish.restore_root(
            {n: p for n, p in files.items() if "manifest" not in n}, tmp_path / "c"
        )


def test_an_episode_that_differs_from_its_index_is_refused(tmp_path):
    root = make_root(tmp_path, "00000")
    shard = root / json.loads((root / "manifest.json").read_text())["shards"][0]["path"]
    data = bytearray(shard.read_bytes())
    data[5] ^= 1
    shard.write_bytes(bytes(data))
    with pytest.raises(publish.PublishError, match="differs from its index"):
        publish.convert_root(root, tmp_path / "out")


def test_a_failed_repeat_restore_removes_the_old_completion_marker(tmp_path):
    root = make_root(tmp_path, "00000")
    out, restored = tmp_path / "out", tmp_path / "restored"
    publish.convert_root(root, out)
    publish.restore_root(listing(out), restored)
    (out / "root/index.sqlite").write_bytes(b"damaged")
    with pytest.raises(publish.PublishError, match="index"):
        publish.restore_root(listing(out), restored)
    assert not (restored / "manifest.json").exists()


def test_the_transcript_joins_the_parts_of_a_record_and_names_media():
    import base64

    def record(role, text, logical, media="text/plain; charset=utf-8", rid="r"):
        return {
            "record_id": rid,
            "role": role,
            "logical_id": logical,
            "media_type": media,
            "payload": base64.b64encode(text).decode(),
        }

    episode = {
        "records": [
            record("user", b"Hello, ", "u", rid="r1"),
            record("user", b"world", "u", rid="r2"),
            record("assistant", b"Hi", "a", rid="r3"),
            record("tool", b"\x00\x01\x02", "t", "image/png", rid="r4"),
        ]
    }
    assert publish.readable(episode) == [
        {"role": "user", "text": "Hello, world"},
        {"role": "assistant", "text": "Hi"},
        {"role": "tool", "text": "[image/png: 3 bytes]"},
    ]


# ---- a part, stored, pinned and read back --------------------------------------------------


def store(hub, tmp_path, *, part=0, names=("00000", "00001"), groups=None, run=RUN, **kwargs):
    prepared = tmp_path / f"prepared-{run}-{part}"
    for name in names:
        shutil.copytree(make_root(tmp_path, f"{run}-{part}-{name}"), prepared / name)
    groups = groups or {names[0]: "permissive", names[1]: "no"}
    ids = {publish.root_rows(prepared / name).__next__()["source_id"] for name in names}
    return publish.store_part(
        keys=KEY,
        api=hub,
        repo=REPO,
        run=run,
        part=part,
        prepared=prepared,
        work=tmp_path / f"work-{run}-{part}",
        groups=groups,
        sources=sources_of(*ids),
        piece_bytes=1 << 15,
        segment=1 << 12,
        log=lambda _: None,
        **kwargs,
    ), prepared


def finalize(hub, receipts, tmp_path, run=RUN, key=KEY):
    return publish.finalize(
        api=hub, keys=key, repo=REPO, run=run, receipts=receipts, scratch=tmp_path / "scratch"
    )


def test_a_part_publishes_its_public_roots_in_the_clear_and_encrypts_the_rest(tmp_path):
    hub = FakeHub()
    receipt, prepared = store(hub, tmp_path)
    files = hub.commits[REPO][hub.head[REPO]]
    folder = publish.part_dir("permissive", RUN, 0)
    assert any(n.startswith(f"{folder}/00000/") and n.endswith(".parquet") for n in files)
    assert f"{folder}/{publish.SOURCES_FILE}" in files
    assert not any(n.startswith("no/") for n in files)
    assert receipt["encrypted"].startswith("encrypted/") and receipt["roots"] == {
        "00000": "permissive",
        "00001": "no",
    }
    assert all(not n.startswith(("share_alike", "non_commercial")) for n in files)
    # The encrypted folder holds ciphertext only (the README apart): no episode text in the clear.
    for name, data in files.items():
        if name.startswith("encrypted/") and name != publish.ENCRYPTED_CARD:
            assert b"episode" not in data and b"messages" not in data and b"00001" not in data
    assert all(private is False for _, private in hub.created)
    listing_entry = json.loads(files[f"{folder}/{publish.SOURCES_FILE}"])
    assert listing_entry["group"] == "permissive" and listing_entry["sources"][0]["episodes"] > 0


def test_every_root_needs_a_group_and_every_source_an_attribution(tmp_path):
    hub = FakeHub()
    make_root(tmp_path, "00000")
    prepared = tmp_path / "prepared"
    common = dict(
        keys=KEY, api=hub, repo=REPO, run=RUN, part=0, prepared=prepared, work=tmp_path / "w"
    )
    with pytest.raises(publish.PublishError, match="No licence group"):
        publish.store_part(**common, groups={}, sources={})
    with pytest.raises(publish.PublishError, match="no source to attribute"):
        publish.store_part(**common, groups={"00000": "permissive"}, sources={})
    with pytest.raises(publish.PublishError, match="Unknown licence groups"):
        publish.store_part(**common, groups={"00000": "typo"}, sources={})


def test_the_pin_records_one_commit_and_the_whole_run_comes_back(tmp_path):
    hub = FakeHub()
    receipt, prepared = store(hub, tmp_path)
    pin = finalize(hub, [receipt], tmp_path)
    assert pin["revision"] == hub.head[REPO] and pin["repo"] == REPO
    assert (
        publish.prepared_corpus_metadata(pin)["fingerprint"]
        == pin["prepared_corpus"]["fingerprint"]
    )
    changed_summary = json.loads(json.dumps(pin))
    changed_summary["parts"][0]["prepared_roots"]["00000"]["fingerprint"] = "f" * 64
    with pytest.raises(publish.PublishError, match="summary differs"):
        publish.prepared_corpus_metadata(changed_summary)
    with pytest.raises(publish.PublishError, match="rebuild and republish"):
        publish.prepared_corpus_metadata({"parts": []})
    assert b"Zstandard-compressed" in hub.commits[REPO][pin["revision"]]["README.md"]
    path = tmp_path / "run.pin.json"
    path.write_text(json.dumps(pin))
    assert publish.decide(path, KEY, hub, tmp_path / "s").mode == "stored"
    restored = tmp_path / "restored"
    summary = publish.fetch(
        api=hub, keys=KEY, pin=publish.load_pin(path), dest=restored, log=lambda _: None
    )
    assert summary["public_roots"] == 1
    for name in ("00000", "00001"):
        assert tree(restored / name) == {
            n: d for n, d in tree(prepared / name).items() if Path(n).name not in cache.SKIP_NAMES
        }
    with PreparedCorpus(sorted(restored.iterdir()), split=None) as corpus:
        assert len(corpus) == 14
    assert not (tmp_path / "restored.fetching").exists()


def test_a_pin_of_public_roots_alone_needs_no_key(tmp_path):
    hub = FakeHub()
    receipt, prepared = store(
        hub,
        tmp_path,
        names=("00000", "00001"),
        groups={"00000": "permissive", "00001": "share_alike"},
    )
    assert receipt["encrypted"] is None
    pin = finalize(hub, [receipt], tmp_path, key=None)
    assert pin["parts"][0]["encrypted"] is None
    restored = tmp_path / "restored"
    publish.fetch(api=hub, keys=None, pin=pin, dest=restored, log=lambda _: None)
    assert sorted(p.name for p in restored.iterdir()) == ["00000", "00001"]


def test_rebuilding_a_part_removes_old_parquet_files_and_newly_restricted_roots(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(publish, "ROW_GROUP_ROWS", 2)
    hub = FakeHub()
    first, prepared = store(hub, tmp_path, parquet_bytes=1)
    prefix = publish.part_dir("permissive", RUN, 0)
    assert sum(n.endswith(".parquet") for n in hub.commits[REPO][hub.head[REPO]]) > 1
    ids = {
        row["source_id"] for name in first["roots"] for row in publish.root_rows(prepared / name)
    }
    common = dict(
        keys=KEY,
        api=hub,
        repo=REPO,
        run=RUN,
        part=0,
        prepared=prepared,
        work=tmp_path / "rebuild",
        sources=sources_of(*ids),
        piece_bytes=1 << 15,
        segment=1 << 12,
        log=lambda _: None,
    )
    second = publish.store_part(**common, groups=first["roots"])
    assert sum(n.endswith(".parquet") for n in hub.commits[REPO][hub.head[REPO]]) == 1
    pin = finalize(hub, [second], tmp_path)
    restored = tmp_path / "restored"
    publish.fetch(api=hub, keys=KEY, pin=pin, dest=restored, log=lambda _: None)
    assert tree(restored / "00000") == {
        n: d for n, d in tree(prepared / "00000").items() if Path(n).name not in cache.SKIP_NAMES
    }
    publish.store_part(**common, groups={"00000": "no", "00001": "no"})
    assert not any(n.startswith(prefix + "/") for n in hub.commits[REPO][hub.head[REPO]])


def decide(hub, pin, tmp_path, key=KEY, *, write=True):
    path = tmp_path / "decide.pin.json"
    path.unlink(missing_ok=True)
    if write and pin is not None:
        path.write_text(json.dumps(pin))
    return publish.decide(path, key, hub, tmp_path / "scratch")


def stored(tmp_path):
    hub = FakeHub()
    receipt, _ = store(hub, tmp_path)
    return hub, finalize(hub, [receipt], tmp_path)


def test_anything_that_does_not_verify_means_rebuild(tmp_path):
    hub, pin = stored(tmp_path)
    assert decide(hub, pin, tmp_path).mode == "stored"
    assert decide(hub, pin, tmp_path, write=False).mode == "rebuild"  # no pin recorded
    assert "no pin" in decide(hub, pin, tmp_path, write=False).reason
    assert decide(hub, pin, tmp_path, cache.Keys(bytes(32))).mode == "rebuild"  # the key was lost
    files = hub.commits[REPO][pin["revision"]]
    public = next(n for n in files if n.endswith(".parquet"))
    original = files[public]
    files[public] = original[:-1] + bytes([original[-1] ^ 1])  # a public file was altered
    assert decide(hub, pin, tmp_path).mode == "rebuild"
    del files[public]  # or removed
    assert decide(hub, pin, tmp_path).mode == "rebuild"
    files[public] = original
    assert decide(hub, pin, tmp_path).mode == "stored"
    piece = next(n for n in files if n.startswith("encrypted/") and n.endswith(".bin"))
    files[piece] = files[piece][:100] + bytes([files[piece][100] ^ 1]) + files[piece][101:]
    assert decide(hub, pin, tmp_path).mode == "rebuild"  # an encrypted piece was altered
    hub, pin = stored(tmp_path / "x")
    pin["revision"] = "f" * 40
    assert decide(hub, pin, tmp_path).mode == "rebuild"
    hub, pin = stored(tmp_path / "y")
    hub.commits.clear()
    hub.head.clear()
    assert decide(hub, pin, tmp_path).mode == "rebuild"  # the repository was deleted
    (tmp_path / "decide.pin.json").write_text("{not json")
    assert decide(FakeHub(), None, tmp_path, write=False).mode == "rebuild"


def test_a_network_failure_is_not_mistaken_for_a_missing_copy(tmp_path):
    hub, pin = stored(tmp_path)

    def down(*args, **kwargs):
        raise ConnectionError("no route")

    hub.dataset_info = hub.list_repo_tree = hub.hf_hub_download = down
    with pytest.raises(ConnectionError):
        decide(hub, pin, tmp_path)


def test_fetch_refuses_a_public_file_that_differs_from_the_pin(tmp_path):
    hub, pin = stored(tmp_path)
    files = hub.commits[REPO][pin["revision"]]
    public = next(n for n in files if n.endswith(".parquet"))
    files[public] = files[public][:-1] + bytes([files[public][-1] ^ 1])
    with pytest.raises(publish.PublishError, match="does not match the pin"):
        publish.fetch(api=hub, keys=KEY, pin=pin, dest=tmp_path / "out", log=lambda _: None)


# ---- several sandboxes, one repository ------------------------------------------------------


def test_sandboxes_write_their_own_folders_and_races_are_repeated(tmp_path, monkeypatch):
    from megatoken import hub as hub_module

    monkeypatch.setattr(hub_module.time, "sleep", lambda _: None)
    hub = FakeHub()
    first, _ = store(hub, tmp_path, part=0)
    hub.conflicts = 2  # the next sandbox's commits race with others and are repeated
    second, _ = store(
        hub,
        tmp_path,
        part=1,
        names=("00002", "00003"),
        groups={"00002": "permissive", "00003": "no"},
    )
    assert hub.refused == 2
    assert first["encrypted"] != second["encrypted"]
    paths = {item["path"] for item in first["public"]} & {item["path"] for item in second["public"]}
    assert not paths  # no path is written by two sandboxes
    pin = finalize(hub, [second, first], tmp_path)  # receipts in any order
    assert [part["part"] for part in pin["parts"]] == [0, 1]
    restored = tmp_path / "restored"
    publish.fetch(api=hub, keys=KEY, pin=pin, dest=restored, log=lambda _: None)
    assert sorted(p.name for p in restored.iterdir()) == ["00000", "00001", "00002", "00003"]
    with pytest.raises(publish.PublishError, match="parts 0..n-1"):
        publish.make_pin(RUN, REPO, "x", [first, first])
    with pytest.raises(publish.PublishError, match="different run"):
        publish.make_pin("another-run", REPO, "x", [first])
    mismatched = {**first, "prepared_roots": {"not-the-group": first["prepared_roots"]["00000"]}}
    with pytest.raises(publish.PublishError, match="Part roots differ"):
        publish.make_pin(RUN, REPO, "x", [mismatched])
    with pytest.raises(publish.PublishError, match="none of the requested parts"):
        publish.fetch(api=hub, keys=KEY, pin=pin, dest=tmp_path / "none", parts=[7])


def test_one_part_can_be_asked_for(tmp_path):
    hub = FakeHub()
    first, _ = store(hub, tmp_path, part=0)
    second, _ = store(
        hub,
        tmp_path,
        part=1,
        names=("00002", "00003"),
        groups={"00002": "permissive", "00003": "no"},
    )
    pin = finalize(hub, [first, second], tmp_path)
    restored = tmp_path / "restored"
    publish.fetch(api=hub, keys=KEY, pin=pin, dest=restored, parts=[1], log=lambda _: None)
    assert sorted(p.name for p in restored.iterdir()) == ["00002", "00003"]


def test_the_card_is_rebuilt_from_the_repository_across_runs(tmp_path):
    hub = FakeHub()
    first, _ = store(hub, tmp_path, run="run-a")
    finalize(hub, [first], tmp_path, run="run-a")
    second, _ = store(hub, tmp_path, run="run-b")
    pin = finalize(hub, [second], tmp_path, run="run-b")
    card = hub.commits[REPO][pin["revision"]]["README.md"].decode()
    # Both runs' sources are summed into one table; the demo source is in both.
    (row,) = [
        line for line in card.splitlines() if line.startswith("| demo") or "example/" in line
    ][:1]
    assert "permissive" in card and "2 compressed parts" in card
    assert str(2 * second["episodes"]["permissive"]) in row.replace(",", "")
    # Nothing changed: rebuilding again makes no new commit.
    before = hub.head[REPO]
    assert publish.refresh_card(hub, REPO, scratch=tmp_path / "again") == before
    assert hub.head[REPO] == before


def test_a_card_update_reloads_sources_when_another_run_uploads(tmp_path, monkeypatch):
    from megatoken import hub as hub_module

    monkeypatch.setattr(hub_module.time, "sleep", lambda _: None)
    hub = FakeHub()
    store(hub, tmp_path, run="run-a")
    original = hub.create_commit
    raced = False

    def commit(**kwargs):
        nonlocal raced
        if not raced:
            raced = True
            store(hub, tmp_path, run="run-b")
        return original(**kwargs)

    hub.create_commit = commit
    revision = publish.refresh_card(hub, REPO, scratch=tmp_path / "card")
    card = hub.commits[REPO][revision]["README.md"].decode()
    assert "2 compressed parts" in card
    assert "| 14 |" in card  # both runs' public episodes are described


# ---- the card -------------------------------------------------------------------------------

ENTRIES = {
    "permissive": [
        {
            "source_id": "alpha",
            "upstream": "https://huggingface.co/datasets/org/alpha",
            "revision": "b" * 40,
            "license": "apache-2.0",
            "license_url": "https://huggingface.co/datasets/org/alpha",
            "episodes": 1500,
            "tokens": 9,
        }
    ],
    "non_commercial": [
        {
            "source_id": "beta",
            "upstream": "",
            "revision": "c" * 40,
            "license": "cc-by-nc-sa-4.0",
            "license_url": "",
            "episodes": 3,
            "tokens": 1,
        }
    ],
}


def test_the_card_labels_compression_and_attributes_every_source():
    card = publish.build_card(
        repo=REPO,
        subsets=ENTRIES,
        encrypted={"parts": 4, "bytes": 5 << 30},
        contact="write to maintainer@example.org",
    )
    front, _, body = card.partition("\n---\n")
    assert front.startswith("---\nlicense: other")
    assert "config_name: permissive" in front and "permissive/**/*.parquet" in front
    assert "non_commercial/**/*.parquet" in front
    assert "share_alike" not in front and "encrypted" not in front  # not a loadable config
    notice = body.index("Zstandard-compressed training data")
    assert notice < body.index("## ") and "project's download tools" in body
    assert "encrypt" not in body.lower() and "private key" not in body.lower()
    assert "write to maintainer@example.org" in body
    assert "| `permissive` |" in body and "| `non_commercial` | " in body
    assert "CC-BY-NC-SA-4.0" in body  # a share-alike non-commercial source sets the subset licence
    assert (
        "| alpha | [org/alpha](https://huggingface.co/datasets/org/alpha) | `bbbbbbbbbbbb` | apache-2.0 |"
        in body
    )
    assert "1,500" in body and "local file" in body
    assert "4 compressed parts, 5.0 GiB" in body
    assert "### `permissive`" in body and "### `non_commercial`" in body
    assert "### `share_alike`" not in body
    assert body.index("### `permissive`") < body.index("### `non_commercial`")
    assert "Compressed" in publish.encrypted_card(
        "c"
    ) and "download tools" in publish.encrypted_card("c")
    assert "encrypt" not in publish.encrypted_card("c").lower()


def test_a_card_without_public_data_declares_no_subset():
    card = publish.build_card(repo=REPO, subsets={}, encrypted={"parts": 1, "bytes": 1})
    assert "configs:" not in card and "Zstandard-compressed" in card


def test_the_subset_licences():
    assert publish.subset_licence("share_alike", []) == "CC-BY-SA-4.0"
    assert publish.subset_licence("non_commercial", ENTRIES["non_commercial"][:0]) == "CC-BY-NC-4.0"
    assert publish.subset_licence("non_commercial", ENTRIES["non_commercial"]) == "CC-BY-NC-SA-4.0"
