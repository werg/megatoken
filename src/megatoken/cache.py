"""An encrypted, durable cache of prepared data on public Hugging Face dataset repos.

Sandboxes vanish within hours, so a finished build is copied out before they do. The copy
is encrypted: every file is sealed with AES-256-GCM under a key that lives only in a local
file (``megatoken cache keygen`` makes it) and reaches a sandbox only as a secret file.
What the public repos hold is ciphertext, with opaque names, and one short plaintext
README. Nothing about the sources, their licences or their sizes is readable.

What goes up, for one *part* (the roots one finishing sandbox wrote):

* The part's files are laid end to end as one byte stream and cut into *pieces* of the
  same size (the last is padded with zeros), so file and source sizes are not visible.
  Each piece is sealed on its own, in 4 MiB segments, so a multi-GB piece is never held
  in memory. A segment's nonce counts its position and marks the last one, so reordering
  and truncation are caught, and a piece is bound to its own file name.
* A manifest (file paths, offsets, sha256 of every plaintext file, sha256 of every piece's
  ciphertext) is sealed the same way and padded to a fixed granularity.
* File names are ``HMAC(key derived from the data key, logical name)``. Pieces go to
  as many repos as ``repo_bytes`` needs; the repo names are opaque too.

Reading back needs the key and a *pin* (repo names and commit hashes; it holds no secret and
may be committed). ``fetch`` downloads the pinned commits, checks every piece's ciphertext
hash against the manifest, decrypts, and checks every file's plaintext hash.
``decide`` is the rule a training run follows: a pin whose repos still verify means fetch,
anything else means rebuild (see ``megatoken.cloud_prepare ensure``).

  megatoken cache keygen [--path FILE]            create the key (never overwrites)
  python -m megatoken.cache store ...             (in a sandbox) encrypt and upload a part
  python -m megatoken.cache fetch --pin FILE --dest DIR
  python -m megatoken.cache verify --pin FILE     check the pinned repos without downloading
  python -m megatoken.cache pin RECEIPT... --run-id ID --output FILE   (used when rotating the key)
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import hmac
import json
import os
import shutil
import stat
import sys
import time
from collections import deque
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

from megatoken.io import canonical_json, content_hash, sha256_file

KEY_FILE = Path(
    os.environ.get("MEGATOKEN_CACHE_KEY_FILE", Path.home() / ".config/megatoken/cache.key")
)
SECRET_NAME = "cache-key"  # the file name a sandbox finds the key under, in its secrets directory
PIP = ["cryptography>=43,<51"]  # what a sandbox needs beside the project's own requirements

MAGIC = b"MTC1"
SALT_BYTES = 32
HEADER_BYTES = len(MAGIC) + 4 + SALT_BYTES
TAG_BYTES = 16
SEGMENT_BYTES = 4 << 20  # plaintext sealed at a time
MAX_SEGMENT_BYTES = 64 << 20
PIECE_BYTES = 1 << 30  # plaintext of every piece (the last is padded to it)
# Public dataset repos are asked to stay under about 300 GB; each of ours holds at most this.
REPO_BYTES = 250 << 30
MANIFEST_GRAIN = 1 << 20  # a manifest is padded (JSON allows trailing blanks) to a multiple of this
# Working files of the preparation that training never reads.
SKIP_NAMES = frozenset(
    {
        "preparation.sqlite",
        "preparation.sqlite-wal",
        "preparation.sqlite-shm",
        "digest.sqlite",
        "index.building.sqlite",
    }
)
PIN_SCHEMA = "megatoken.cache-pin/1"
MANIFEST_SCHEMA = "megatoken.cache-part/1"
RECEIPT_SCHEMA = "megatoken.cache-receipt/1"
README = (
    "# Encrypted cache\n\n"
    "Encrypted cache of preprocessed machine-learning training data for the megatoken "
    "project. The files are ciphertext and cannot be read without a private key.\n"
)
NOT_FOUND = frozenset(
    {
        "RepositoryNotFoundError",
        "RevisionNotFoundError",
        "EntryNotFoundError",
        "RemoteEntryNotFoundError",
        "LocalEntryNotFoundError",
    }
)


class CacheError(RuntimeError):
    pass


# ---- the key ------------------------------------------------------------------------------


def keygen(path: Path | None = None) -> Path:
    """Write a fresh random 256-bit key (hex, mode 600); refuses to replace an existing key,
    since everything stored under the old one would become unreadable."""
    path = Path(path or KEY_FILE).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        raise CacheError(
            f"{path} exists; it is never replaced (a new key cannot read what the old one "
            "sealed). Move it away yourself to rotate; see docs/data.md."
        ) from None
    with os.fdopen(descriptor, "w") as file:
        file.write(os.urandom(32).hex() + "\n")
    return path


class Keys:
    """Every secret derived from the one data key: file keys, the name key."""

    def __init__(self, master: bytes) -> None:
        if len(master) != 32:
            raise CacheError("The cache key must be 32 bytes (64 hex characters)")
        self._master = master
        self._names = self._derive(b"names")
        self._salts = self._derive(b"salts")

    def _derive(self, purpose: bytes, salt: bytes | None = None) -> bytes:
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.kdf.hkdf import HKDF

        return HKDF(hashes.SHA256(), 32, salt, b"megatoken-cache/v1/" + purpose).derive(
            self._master
        )

    def file_key(self, salt: bytes) -> bytes:
        return self._derive(b"file", salt)

    def salt(self, text: str) -> bytes:
        """A file's salt, repeatable from what the file holds (see ``encrypt_stream``)."""
        return hmac.new(self._salts, text.encode(), hashlib.sha256).digest()

    def label(self, text: str, length: int = 32) -> str:
        """An opaque, repeatable name for a logical one."""
        return hmac.new(self._names, text.encode(), hashlib.sha256).hexdigest()[:length]


def key_path() -> Path:
    """Where the key is: ``$MEGATOKEN_CACHE_KEY_FILE``, the sandbox's secret, or the default."""
    if os.environ.get("MEGATOKEN_CACHE_KEY_FILE"):
        return KEY_FILE
    secrets = os.environ.get("MEGATOKEN_CW_SECRETS")
    if secrets and (Path(secrets) / SECRET_NAME).is_file():
        return Path(secrets) / SECRET_NAME
    return KEY_FILE


def read_key_file(path: Path | None = None) -> bytes:
    """The key's raw bytes; a file other users can read is refused."""
    path = Path(path or key_path()).expanduser()
    if not path.is_file():
        raise CacheError(f"No cache key at {path}: create one with `megatoken cache keygen`")
    # A sandbox's secrets directory is already private (mode 700); a local file must be too.
    in_sandbox = os.environ.get("MEGATOKEN_CW_SECRETS") == str(path.parent)
    if not in_sandbox and path.stat().st_mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise CacheError(f"{path} is readable by others; chmod 600 it")
    try:
        return bytes.fromhex(path.read_text().strip())
    except ValueError:
        raise CacheError(f"{path} is not 64 hex characters") from None


def load_keys(path: Path | None = None) -> Keys:
    return Keys(read_key_file(path))


# ---- sealing and opening a stream ---------------------------------------------------------


def _nonce(index: int, final: bool) -> bytes:
    return index.to_bytes(11, "big") + (b"\x01" if final else b"\x00")


def _read_full(source: BinaryIO, size: int) -> bytes:
    parts, got = [], 0
    while got < size:
        block = source.read(size - got)
        if not block:
            break
        parts.append(block)
        got += len(block)
    return b"".join(parts)


def encrypt_stream(
    source: BinaryIO,
    sink: BinaryIO,
    keys: Keys,
    aad: bytes = b"",
    segment: int = SEGMENT_BYTES,
    salt: bytes | None = None,
) -> tuple[str, int]:
    """Seal ``source`` into ``sink`` segment by segment; returns the ciphertext's sha256 and size.

    The file key comes from a salt in the header, so no nonce repeats under a key. The salt
    is random unless the caller passes one that is a function of the plaintext (``Keys.salt``
    of its hash): then the same plaintext always seals to the same bytes, which lets a
    rebuilt sandbox repeat an upload the Hub already holds, and different plaintexts never
    share a key.
    the last segment's nonce says so, which is what makes a cut-off file fail to open.
    """
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    if not 0 < segment <= MAX_SEGMENT_BYTES:
        raise CacheError("Unsupported segment size")
    salt = os.urandom(SALT_BYTES) if salt is None else salt
    header = MAGIC + segment.to_bytes(4, "big") + salt
    aead = AESGCM(keys.file_key(salt))
    digest = hashlib.sha256(header)
    sink.write(header)
    total = len(header)
    current, index = _read_full(source, segment), 0
    while True:
        following = _read_full(source, segment) if len(current) == segment else b""
        final = not following
        block = aead.encrypt(_nonce(index, final), current, header + aad)
        digest.update(block)
        sink.write(block)
        total += len(block)
        if final:
            return digest.hexdigest(), total
        current, index = following, index + 1


def decrypt_stream(source: BinaryIO, keys: Keys, aad: bytes = b"") -> Iterator[bytes]:
    """Plaintext segments of a sealed stream. A tampered, reordered, cut-off or foreign file,
    or the wrong key, raises ``CacheError`` (possibly after earlier segments were yielded, so
    a consumer commits what it wrote only once the iterator ends)."""
    from cryptography.exceptions import InvalidTag
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    header = _read_full(source, HEADER_BYTES)
    if len(header) != HEADER_BYTES or header[: len(MAGIC)] != MAGIC:
        raise CacheError("Not an encrypted cache file")
    segment = int.from_bytes(header[len(MAGIC) : len(MAGIC) + 4], "big")
    if not 0 < segment <= MAX_SEGMENT_BYTES:
        raise CacheError("Unsupported segment size")
    aead = AESGCM(keys.file_key(header[len(MAGIC) + 4 :]))
    size = segment + TAG_BYTES
    current, index = _read_full(source, size), 0
    if not current:
        raise CacheError("Encrypted file is cut off")
    while True:
        following = _read_full(source, size) if len(current) == size else b""
        final = not following
        try:
            yield aead.decrypt(_nonce(index, final), current, header + aad)
        except InvalidTag:
            raise CacheError(
                "Decryption failed: the file was altered or cut off, or the key is wrong"
            ) from None
        if final:
            return
        current, index = following, index + 1


def cipher_size(plain: int, segment: int = SEGMENT_BYTES) -> int:
    """The size of ``plain`` bytes once sealed."""
    return HEADER_BYTES + plain + TAG_BYTES * max(1, -(-plain // segment))


def open_bytes(data: bytes, keys: Keys, aad: bytes) -> bytes:
    import io

    return b"".join(decrypt_stream(io.BytesIO(data), keys, aad))


# ---- the stream of a part's files ---------------------------------------------------------


@dataclass(frozen=True)
class Entry:
    path: str  # relative, with ``/``
    offset: int
    bytes: int
    sha256: str


def scan(source: Path) -> list[Entry]:
    """Every file under ``source`` (working databases left out) with its hash, in sorted order,
    placed end to end."""
    entries, offset = [], 0
    for path in sorted(p for p in source.rglob("*") if p.is_file() and p.name not in SKIP_NAMES):
        size = path.stat().st_size
        entries.append(Entry(path.relative_to(source).as_posix(), offset, size, sha256_file(path)))
        offset += size
    return entries


class _Slice:
    """A readable stretch of the files' end-to-end stream, zero-padded past its end."""

    def __init__(self, source: Path, entries: Sequence[Entry], start: int, length: int) -> None:
        self.source = source
        self.entries = [entry for entry in entries if entry.bytes]
        self.offsets = [entry.offset for entry in self.entries]
        self.total = self.entries[-1].offset + self.entries[-1].bytes if self.entries else 0
        self.position, self.end = start, start + length
        self.handle: BinaryIO | None = None
        self.open_index = -1

    def read(self, size: int) -> bytes:
        size = min(size, self.end - self.position)
        out = []
        while size > 0:
            if self.position >= self.total:
                out.append(bytes(size))
                self.position += size
                break
            index = bisect.bisect_right(self.offsets, self.position) - 1
            entry = self.entries[index]
            if self.open_index != index:
                self.close()
                self.handle = (self.source / entry.path).open("rb")
                self.open_index = index
            assert self.handle is not None
            take = min(size, entry.offset + entry.bytes - self.position)
            self.handle.seek(self.position - entry.offset)
            data = self.handle.read(take)
            if len(data) != take:
                raise CacheError(f"{entry.path} changed while it was being sealed")
            out.append(data)
            self.position += take
            size -= take
        return b"".join(out)

    def close(self) -> None:
        if self.handle is not None:
            self.handle.close()
            self.handle = None
            self.open_index = -1


# ---- the Hub, as far as the cache uses it ------------------------------------------------


def _missing(error: Exception) -> bool:
    status = getattr(getattr(error, "response", None), "status_code", None)
    return status == 404 or type(error).__name__ in NOT_FOUND


class Remote:
    """The few Hub calls the cache makes (``HfApi`` or anything shaped like it)."""

    def __init__(self, api: Any, owner: str | None = None) -> None:
        self.api, self.owner = api, owner

    def tree(self, repo: str, revision: str | None = None) -> dict[str, Any]:
        try:
            return {
                item.path: item
                for item in self.api.list_repo_tree(
                    repo, repo_type="dataset", revision=revision, recursive=True, expand=True
                )
            }
        except Exception as error:
            if _missing(error):
                raise CacheError(f"{repo} (revision {revision}) does not exist") from error
            raise

    def head(self, repo: str, revision: str | None = None) -> str:
        try:
            return self.api.dataset_info(repo, revision=revision).sha
        except Exception as error:
            if _missing(error):
                raise CacheError(f"{repo} (revision {revision}) does not exist") from error
            raise

    def download(self, repo: str, name: str, revision: str, directory: Path) -> Path:
        try:
            return Path(
                self.api.hf_hub_download(
                    repo_id=repo,
                    filename=name,
                    repo_type="dataset",
                    revision=revision,
                    local_dir=str(directory),
                )
            )
        except Exception as error:
            if _missing(error):
                raise CacheError(f"{repo}: {name} is missing at revision {revision}") from error
            raise

    def remote_sha256(self, repo: str, name: str, revision: str, item: Any, scratch: Path) -> str:
        """A stored file's sha256: the Hub records it for large files; a small file is read."""
        lfs = getattr(item, "lfs", None)
        sha = getattr(lfs, "sha256", None) if lfs is not None else None
        if sha:
            return sha
        path = self.download(repo, name, revision, scratch)
        try:
            return sha256_file(path)
        finally:
            path.unlink(missing_ok=True)


# ---- names and pins -----------------------------------------------------------------------


def repo_name(keys: Keys, owner: str, run: str, part: int, volume: int) -> str:
    return f"{owner}/{keys.label(f'repo/{run}/{part}/{volume}', 24)}"


def piece_file(keys: Keys, run: str, part: int, index: int) -> str:
    return keys.label(f"piece/{run}/{part}/{index}") + ".bin"


def manifest_file(keys: Keys, run: str, part: int) -> str:
    return keys.label(f"manifest/{run}/{part}") + ".bin"


def load_pin(path: Path) -> dict[str, Any]:
    pin = json.loads(Path(path).read_text())
    if pin.get("schema") != PIN_SCHEMA or not pin.get("run") or not pin.get("parts"):
        raise CacheError(f"{path} is not a cache pin")
    for part in pin["parts"]:
        if not part.get("repos") or not all(
            r.get("repo") and r.get("revision") for r in part["repos"]
        ):
            raise CacheError(f"{path}: a part has no pinned repos")
    return pin


def make_pin(run: str, receipts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The pin of a whole run from its parts' receipts: only repo names and commit hashes."""
    parts = sorted(receipts, key=lambda receipt: receipt["part"])
    if [receipt["part"] for receipt in parts] != list(range(len(parts))):
        raise CacheError("The receipts do not cover parts 0..n-1 exactly once")
    return {
        "schema": PIN_SCHEMA,
        "run": run,
        "parts": [
            {
                "part": receipt["part"],
                "repos": [
                    {"repo": item["repo"], "revision": item["revision"]}
                    for item in receipt["repos"]
                ],
            }
            for receipt in parts
        ],
    }


# ---- storing ------------------------------------------------------------------------------


def _write_state(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value))
    os.replace(temporary, path)


def store(
    *,
    keys: Keys,
    api: Any,
    owner: str,
    run: str,
    part: int,
    source: Path,
    work: Path,
    piece_bytes: int = PIECE_BYTES,
    repo_bytes: int = REPO_BYTES,
    segment: int = SEGMENT_BYTES,
    workers: int = 4,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Seal the files under ``source`` and upload them; returns the receipt (repos, commits).

    Resumable: ``work`` keeps every sealed piece with the plan that made it, so a rerun over
    the same files reseals nothing, and the large-folder upload skips what the Hub holds.
    """
    remote = Remote(api, owner)
    entries = scan(source)
    total = sum(entry.bytes for entry in entries)
    if not total:
        raise CacheError(f"{source} holds no files to store")
    plan = content_hash(
        {
            "run": run,
            "part": part,
            "piece_bytes": piece_bytes,
            "segment": segment,
            "files": [[e.path, e.bytes, e.sha256] for e in entries],
        }
    )
    plan_file = work / "plan.json"
    if not plan_file.exists() or json.loads(plan_file.read_text()).get("plan") != plan:
        shutil.rmtree(work / "upload", ignore_errors=True)
        shutil.rmtree(work / "state", ignore_errors=True)
        _write_state(plan_file, {"plan": plan})
    pieces = -(-total // piece_bytes)
    per_repo = max(1, repo_bytes // cipher_size(piece_bytes, segment))
    volumes = -(-pieces // per_repo)

    def seal(index: int) -> dict[str, Any]:
        name = piece_file(keys, run, part, index)
        volume = index // per_repo
        target = work / "upload" / str(volume) / name
        state = work / "state" / f"{name}.json"
        if target.exists() and state.exists():
            done = json.loads(state.read_text())
            if (
                done["plan"] == plan
                and target.stat().st_size == done["bytes"]
                and sha256_file(target) == done["sha256"]
            ):
                return {
                    "name": name,
                    "volume": volume,
                    "bytes": done["bytes"],
                    "sha256": done["sha256"],
                }
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = target.with_name(target.name + ".partial")
        reader = _Slice(source, entries, index * piece_bytes, piece_bytes)
        try:
            plain = hashlib.sha256()
            while block := reader.read(8 << 20):
                plain.update(block)
            reader.close()
            reader = _Slice(source, entries, index * piece_bytes, piece_bytes)
            with staging.open("wb") as sink:
                sha, size = encrypt_stream(
                    reader,
                    sink,
                    keys,
                    name.encode(),
                    segment,
                    keys.salt(f"{name}/{plain.hexdigest()}"),
                )
        finally:
            reader.close()
        os.replace(staging, target)
        _write_state(state, {"plan": plan, "bytes": size, "sha256": sha})
        return {"name": name, "volume": volume, "bytes": size, "sha256": sha}

    with ThreadPoolExecutor(max(1, min(workers, pieces))) as pool:
        sealed = list(pool.map(seal, range(pieces)))
    log(f"sealed {pieces} pieces in {volumes} repos")

    manifest = {
        "schema": MANIFEST_SCHEMA,
        "run": run,
        "part": part,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "piece_bytes": piece_bytes,
        "segment": segment,
        "stream_bytes": total,
        "volumes": volumes,
        "files": [
            {"path": e.path, "offset": e.offset, "bytes": e.bytes, "sha256": e.sha256}
            for e in entries
        ],
        "pieces": sealed,
    }
    body = canonical_json(manifest)
    body += b" " * (-len(body) % MANIFEST_GRAIN)
    manifest_name = manifest_file(keys, run, part)
    manifest_path = work / "upload" / "0" / manifest_name
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    import io

    with manifest_path.open("wb") as sink:
        encrypt_stream(
            io.BytesIO(body),
            sink,
            keys,
            manifest_name.encode(),
            segment,
            keys.salt(f"{manifest_name}/{hashlib.sha256(body).hexdigest()}"),
        )

    repos = []
    for volume in range(volumes):
        folder = work / "upload" / str(volume)
        (folder / "README.md").write_text(README)
        repo = repo_name(keys, owner, run, part, volume)
        api.create_repo(repo, repo_type="dataset", private=False, exist_ok=True)
        api.upload_large_folder(repo_id=repo, folder_path=str(folder), repo_type="dataset")
        head = remote.head(repo)
        tree = remote.tree(repo, head)
        for local in sorted(p for p in folder.iterdir() if p.is_file()):
            item = tree.get(local.name)
            if item is None or item.size != local.stat().st_size:
                raise CacheError(f"{repo}: {local.name} did not arrive intact")
            if remote.remote_sha256(repo, local.name, head, item, work / "verify") != sha256_file(
                local
            ):
                raise CacheError(f"{repo}: {local.name} differs from what was uploaded")
        repos.append({"repo": repo, "revision": head, "volume": volume})
        log(f"uploaded and verified {repo}")
    shutil.rmtree(work / "verify", ignore_errors=True)
    return {
        "schema": RECEIPT_SCHEMA,
        "run": run,
        "part": part,
        "files": len(entries),
        "stream_bytes": total,
        "pieces": pieces,
        "repos": repos,
    }


# ---- reading back -------------------------------------------------------------------------


def _safe_relative(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute() or ".." in candidate.parts or not candidate.parts:
        raise CacheError(f"Unsafe path in the manifest: {path!r}")
    return candidate


def read_manifest(
    remote: Remote, keys: Keys, pin: Mapping[str, Any], part: Mapping[str, Any], scratch: Path
) -> dict[str, Any]:
    """A part's decrypted manifest from its first pinned repo, checked against the pin."""
    run = pin["run"]
    name = manifest_file(keys, run, part["part"])
    first = part["repos"][0]
    path = remote.download(first["repo"], name, first["revision"], scratch)
    try:
        with path.open("rb") as handle:
            body = b"".join(decrypt_stream(handle, keys, name.encode()))
    finally:
        path.unlink(missing_ok=True)
    manifest = json.loads(body)
    if (
        manifest.get("schema") != MANIFEST_SCHEMA
        or manifest["run"] != run
        or manifest["part"] != part["part"]
    ):
        raise CacheError(f"Part {part['part']}: the manifest belongs to another run or part")
    if manifest["volumes"] != len(part["repos"]):
        raise CacheError(f"Part {part['part']}: the pin lists the wrong number of repos")
    for item in manifest["files"]:
        _safe_relative(item["path"])
    return manifest


def verify_remote(api: Any, keys: Keys, pin: Mapping[str, Any], scratch: Path) -> dict[int, Any]:
    """Check every pinned repo without downloading the data: the manifests decrypt, and every
    piece is stored at its exact size with the ciphertext hash the manifest records.
    Returns the manifests by part; raises ``CacheError`` for anything missing or altered."""
    remote = Remote(api)
    manifests = {}
    try:
        for part in pin["parts"]:
            manifest = read_manifest(remote, keys, pin, part, scratch)
            trees = {
                index: remote.tree(item["repo"], item["revision"])
                for index, item in enumerate(part["repos"])
            }
            for piece in manifest["pieces"]:
                repo = part["repos"][piece["volume"]]
                item = trees[piece["volume"]].get(piece["name"])
                if item is None or item.size != piece["bytes"]:
                    raise CacheError(f"Part {part['part']}: a piece is missing or the wrong size")
                found = remote.remote_sha256(
                    repo["repo"], piece["name"], repo["revision"], item, scratch
                )
                if found != piece["sha256"]:
                    raise CacheError(f"Part {part['part']}: a stored piece does not match its hash")
            manifests[part["part"]] = manifest
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    return manifests


class _Unpacker:
    """Writes a part's files back from consecutive stretches of the plaintext stream, checking
    each file's sha256 before it is moved to its name."""

    def __init__(self, manifest: Mapping[str, Any], dest: Path, done: set[str]) -> None:
        self.files = [item for item in manifest["files"] if item["bytes"]]
        self.offsets = [item["offset"] for item in self.files]
        self.total = manifest["stream_bytes"]
        self.dest, self.done = dest, done
        self.handle: BinaryIO | None = None
        self.hasher: Any = None
        self.current = -1
        self.position = 0

    def start(self, position: int) -> None:
        self.position = position

    def write(self, data: bytes) -> None:
        view = memoryview(data)
        while len(view) and self.position < self.total:
            index = bisect.bisect_right(self.offsets, self.position) - 1
            item = self.files[index]
            end = item["offset"] + item["bytes"]
            take = min(len(view), end - self.position, self.total - self.position)
            if item["path"] not in self.done:
                self._append(index, item, view[:take])
                if self.position + take == end:
                    self._close(item)
            self.position += take
            view = view[take:]

    def _append(self, index: int, item: Mapping[str, Any], block: memoryview) -> None:
        if self.current != index:
            if self.position != item["offset"]:
                raise CacheError(f"{item['path']}: a piece it needs was skipped")
            target = self.dest / _safe_relative(item["path"])
            target.parent.mkdir(parents=True, exist_ok=True)
            self.handle = target.with_name(target.name + ".partial").open("wb")
            self.hasher, self.current = hashlib.sha256(), index
        assert self.handle is not None
        self.handle.write(block)
        self.hasher.update(block)

    def _close(self, item: Mapping[str, Any]) -> None:
        assert self.handle is not None
        self.handle.close()
        target = self.dest / _safe_relative(item["path"])
        partial = target.with_name(target.name + ".partial")
        self.handle, self.current = None, -1
        if self.hasher.hexdigest() != item["sha256"]:
            partial.unlink(missing_ok=True)
            raise CacheError(f"{item['path']}: decrypted content does not match its recorded hash")
        os.replace(partial, target)
        self.done.add(item["path"])

    def abandon(self) -> None:
        if self.handle is not None:
            self.handle.close()
            self.handle, self.current = None, -1


def _already_there(manifest: Mapping[str, Any], dest: Path) -> set[str]:
    """Files of the manifest already in ``dest`` with the right size and hash (a resumed fetch)."""
    done = set()
    for item in manifest["files"]:
        target = dest / _safe_relative(item["path"])
        if (
            target.is_file()
            and target.stat().st_size == item["bytes"]
            and sha256_file(target) == item["sha256"]
        ):
            done.add(item["path"])
    return done


def _ahead(
    pool: ThreadPoolExecutor, items: Sequence[Any], call: Callable[[Any], Path], window: int
) -> Iterator[tuple[Any, Path]]:
    """``(item, call(item))`` in order, with up to ``window`` calls running ahead of the reader."""
    queue: deque[tuple[Any, Future[Path]]] = deque()
    upcoming = iter(items)
    try:
        while True:
            while len(queue) < max(1, window):
                item = next(upcoming, None)
                if item is None:
                    break
                queue.append((item, pool.submit(call, item)))
            if not queue:
                return
            item, future = queue.popleft()
            yield item, future.result()
    finally:
        for _, pending in queue:
            if not pending.cancel() and pending.exception() is None:
                pending.result().unlink(missing_ok=True)


def fetch(
    *,
    keys: Keys,
    api: Any,
    pin: Mapping[str, Any],
    dest: Path,
    parts: Sequence[int] | None = None,
    workers: int = 4,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Download the pinned commits into ``dest``: every piece's ciphertext hash is checked
    before it is opened, and every file's plaintext hash before it is given its name. A rerun
    skips files already there and verified. Raises ``CacheError`` on any mismatch."""
    remote = Remote(api)
    dest.mkdir(parents=True, exist_ok=True)
    scratch = dest.with_name(dest.name + ".fetching")
    chosen = [part for part in pin["parts"] if parts is None or part["part"] in parts]
    if not chosen:
        raise CacheError("The pin has none of the requested parts")
    files = byte_count = 0
    try:
        for part in chosen:
            manifest = read_manifest(remote, keys, pin, part, scratch)
            done = _already_there(manifest, dest)
            unpack = _Unpacker(manifest, dest, done)
            size = manifest["piece_bytes"]
            wanted = []
            for index, piece in enumerate(manifest["pieces"]):
                start, end = index * size, min((index + 1) * size, manifest["stream_bytes"])
                if any(
                    item["path"] not in done
                    for item in manifest["files"]
                    if item["bytes"]
                    and item["offset"] < end
                    and item["offset"] + item["bytes"] > start
                ):
                    wanted.append((index, piece))

            def download(
                item: tuple[int, Mapping[str, Any]], part: Mapping[str, Any] = part
            ) -> Path:
                repo = part["repos"][item[1]["volume"]]
                return remote.download(repo["repo"], item[1]["name"], repo["revision"], scratch)

            with ThreadPoolExecutor(max(1, workers)) as pool:
                for (index, piece), path in _ahead(pool, wanted, download, workers):
                    try:
                        if sha256_file(path) != piece["sha256"]:
                            raise CacheError(
                                f"Part {part['part']}: a downloaded piece does not match its hash"
                            )
                        unpack.start(index * size)
                        with path.open("rb") as handle:
                            for block in decrypt_stream(handle, keys, piece["name"].encode()):
                                unpack.write(block)
                    except BaseException:
                        unpack.abandon()
                        raise
                    finally:
                        path.unlink(missing_ok=True)
            missing = [
                item["path"]
                for item in manifest["files"]
                if item["bytes"] and item["path"] not in done
            ]
            if missing:
                raise CacheError(f"Part {part['part']}: {len(missing)} files were not restored")
            for item in manifest["files"]:
                if not item["bytes"]:
                    target = dest / _safe_relative(item["path"])
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(b"")
            files += len(manifest["files"])
            byte_count += manifest["stream_bytes"]
            log(f"part {part['part']}: {len(manifest['files'])} files restored")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    return {
        "run": pin["run"],
        "parts": [p["part"] for p in chosen],
        "files": files,
        "bytes": byte_count,
    }


# ---- the decision a run follows -----------------------------------------------------------


@dataclass(frozen=True)
class Decision:
    mode: str  # "stored": fetch the pin; "rebuild": prepare the data again
    reason: str


def decide(pin_path: Path | None, keys: Keys, api: Any, scratch: Path) -> Decision:
    """Fetch if the pin names repos that still verify, otherwise rebuild.

    A missing pin or repo or commit, a piece that is gone or altered, or a key that no longer
    opens the manifests all mean rebuild. A network failure is not an answer and propagates.
    """
    if pin_path is None or not Path(pin_path).is_file():
        return Decision("rebuild", "no pin has been recorded for this run")
    try:
        pin = load_pin(Path(pin_path))
        verify_remote(api, keys, pin, scratch)
    except (CacheError, KeyError, ValueError) as error:
        return Decision("rebuild", f"the stored copy does not verify: {error}")
    return Decision("stored", "every pinned repo verifies against its manifest")


# ---- command line -------------------------------------------------------------------------


def _api() -> Any:
    from huggingface_hub import HfApi

    from megatoken import hub

    hub.install()
    return HfApi()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="megatoken cache", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    item = commands.add_parser("keygen", help="create the data key (refuses to overwrite)")
    item.add_argument("--path", type=Path, default=KEY_FILE)
    item = commands.add_parser("store", help="(in a sandbox) encrypt a part and upload it")
    item.add_argument("--run-id", required=True)
    item.add_argument("--part", type=int, required=True)
    item.add_argument("--owner", required=True, help="the Hub account that owns the repos")
    item.add_argument("--source", type=Path, required=True)
    item.add_argument("--work", type=Path, required=True)
    item.add_argument("--receipt", type=Path, required=True)
    item.add_argument("--piece-bytes", type=int, default=PIECE_BYTES)
    item.add_argument("--repo-bytes", type=int, default=REPO_BYTES)
    item = commands.add_parser("fetch", help="download, verify and decrypt a pinned run")
    item.add_argument("--pin", type=Path, required=True)
    item.add_argument("--dest", type=Path, required=True)
    item.add_argument("--part", type=int, action="append", help="only this part (repeatable)")
    item.add_argument("--workers", type=int, default=4)
    item = commands.add_parser("verify", help="check the pinned repos without downloading data")
    item.add_argument("--pin", type=Path, required=True)
    item = commands.add_parser("pin", help="write a run's pin from its parts' store receipts")
    item.add_argument("receipts", nargs="+", type=Path)
    item.add_argument("--run-id", required=True)
    item.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "keygen":
            path = keygen(args.path)
            print(json.dumps({"key_file": str(path)}))
            return 0
        if args.command == "pin":
            pin = make_pin(args.run_id, [json.loads(path.read_text()) for path in args.receipts])
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(pin, indent=1))
            print(json.dumps({"pin": str(args.output), "parts": len(pin["parts"])}))
            return 0
        keys = load_keys()
        if args.command == "store":
            receipt = store(
                keys=keys,
                api=_api(),
                owner=args.owner,
                run=args.run_id,
                part=args.part,
                source=args.source,
                work=args.work,
                piece_bytes=args.piece_bytes,
                repo_bytes=args.repo_bytes,
            )
            args.receipt.parent.mkdir(parents=True, exist_ok=True)
            args.receipt.write_text(json.dumps(receipt, indent=1))
            result: Any = {key: receipt[key] for key in ("part", "files", "pieces")}
        elif args.command == "fetch":
            result = fetch(
                keys=keys,
                api=_api(),
                pin=load_pin(args.pin),
                dest=args.dest,
                parts=args.part,
                workers=args.workers,
            )
        else:
            scratch = Path(os.environ.get("TMPDIR", "/tmp")) / "megatoken-cache-verify"
            manifests = verify_remote(_api(), keys, load_pin(args.pin), scratch)
            result = {"verified_parts": sorted(manifests)}
    except CacheError as error:
        print(f"cache: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
