"""An encrypted, durable cache of the prepared data that may not be published in the clear.

Sandboxes vanish within hours, so a finished build is copied out before they do. Sources whose
licences do not clearly allow redistribution are copied encrypted: every file is sealed with
AES-256-GCM under a key that lives only in a local file (``megatoken cache keygen`` makes it)
and reaches a sandbox only as a secret file. What the public repository holds under
``encrypted/`` is ciphertext with opaque names. Nothing about the sources, their licences or
their sizes is readable there. Public descriptions label this as a compressed project cache.
The sources that may be published go to the same repository in the clear.

What goes up, for one *part* (the encrypted roots one finishing sandbox wrote):

* Each file is compressed with Zstandard first. Those frames are laid end to end and cut into *pieces* of the
  same size (the last is padded with zeros), so file and source sizes are not visible.
  Each piece is sealed on its own, in 4 MiB segments, so a multi-GB piece is never held
  in memory. A segment's nonce counts its position and marks the last one, so reordering
  and truncation are caught, and a piece is bound to its own file name.
* A manifest (file paths, offsets, sha256 of every plaintext file, sha256 of every piece's
  ciphertext) is sealed the same way and padded to a fixed granularity.
* File names are ``HMAC(key derived from the data key, logical name)``. A part owns one folder,
  ``encrypted/<opaque name>/``, so parts uploaded by different sandboxes at the same time never
  touch the same path.

Reading back needs the key and a *pin* (``megatoken.publish``: the repository and one commit of
it; it holds no secret and may be committed). ``fetch_part`` downloads the pinned commit's files,
checks every piece's ciphertext hash against the manifest, decrypts, decompresses, and checks
every file's original plaintext hash. Earlier uncompressed uploads can still be restored.

  megatoken cache keygen [--path FILE]            create the key (never overwrites)
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

from megatoken import hub
from megatoken.io import canonical_json, content_hash, sha256_file

KEY_FILE = Path(
    os.environ.get("MEGATOKEN_CACHE_KEY_FILE", Path.home() / ".config/megatoken/cache.key")
)
SECRET_NAME = "cache-key"  # the file name a sandbox finds the key under, in its secrets directory
PIP = ["cryptography>=43,<51", "zstandard>=0.23,<1"]

MAGIC = b"MTC1"
SALT_BYTES = 32
HEADER_BYTES = len(MAGIC) + 4 + SALT_BYTES
TAG_BYTES = 16
SEGMENT_BYTES = 4 << 20  # plaintext sealed at a time
MAX_SEGMENT_BYTES = 64 << 20
PIECE_BYTES = 1 << 30  # plaintext of every piece (the last is padded to it)
MANIFEST_GRAIN = 1 << 20  # a manifest is padded (JSON allows trailing blanks) to a multiple of this
ENCRYPTED_DIR = "encrypted"  # the folder of the shared repository that holds every part
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
MANIFEST_SCHEMA = "megatoken.cache-part/2"
RECEIPT_SCHEMA = "megatoken.cache-receipt/2"
COMPRESSION = "zstd"
COMPRESSION_LEVEL = 3
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


def scan(source: Path, only: Sequence[str] | None = None) -> list[Entry]:
    """Every file under ``source`` (working databases left out; with ``only``, just the folders of
    that name) with its hash, in sorted order, placed end to end."""
    entries, offset = [], 0
    for path in sorted(
        p
        for p in source.rglob("*")
        if p.is_file()
        and p.name not in SKIP_NAMES
        and (only is None or p.relative_to(source).parts[0] in only)
    ):
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

    def __init__(self, api: Any) -> None:
        self.api = api

    def tree(
        self, repo: str, revision: str | None = None, path: str | None = None
    ) -> dict[str, Any]:
        """Every file of the repo (or of the folder ``path``) at ``revision``, by path."""
        try:
            return {
                item.path: item
                for item in self.api.list_repo_tree(
                    repo,
                    path_in_repo=path,
                    repo_type="dataset",
                    revision=revision,
                    recursive=True,
                    expand=True,
                )
            }
        except Exception as error:
            if _missing(error):
                raise CacheError(
                    f"{repo}/{path or ''} (revision {revision}) does not exist"
                ) from error
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


# ---- names --------------------------------------------------------------------------------


def folder_name(keys: Keys, run: str, part: int) -> str:
    """The repo folder of one part: opaque, and its own, so concurrent uploads never collide."""
    return f"{ENCRYPTED_DIR}/{keys.label(f'folder/{run}/{part}', 20)}"


def piece_file(keys: Keys, run: str, part: int, index: int) -> str:
    return keys.label(f"piece/{run}/{part}/{index}") + ".bin"


def manifest_file(keys: Keys, run: str, part: int) -> str:
    return keys.label(f"manifest/{run}/{part}") + ".bin"


# ---- storing ------------------------------------------------------------------------------


def _write_state(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value))
    os.replace(temporary, path)


def _compress_files(source: Path, entries: Sequence[Entry], work: Path) -> list[Entry]:
    """Compress each file independently, retaining bounded memory and resumable downloads."""
    import zstandard

    plan = content_hash(
        {
            "compression": COMPRESSION,
            "level": COMPRESSION_LEVEL,
            "version": zstandard.ZSTD_VERSION,
            "files": [[e.path, e.bytes, e.sha256] for e in entries],
        }
    )
    state = work / "compressed.json"
    root = work / "compressed"
    if state.exists():
        saved = json.loads(state.read_text())
        if saved.get("plan") == plan:
            found = [Entry(**item) for item in saved["files"]]
            if all(
                (root / e.path).is_file()
                and (root / e.path).stat().st_size == e.bytes
                and sha256_file(root / e.path) == e.sha256
                for e in found
            ):
                return found
    shutil.rmtree(root, ignore_errors=True)
    compressed = []
    offset = 0
    compressor = zstandard.ZstdCompressor(level=COMPRESSION_LEVEL, write_checksum=True)
    for entry in entries:
        target = root / entry.path
        target.parent.mkdir(parents=True, exist_ok=True)
        with (source / entry.path).open("rb") as reader, target.open("wb") as sink:
            if entry.bytes:
                compressor.copy_stream(reader, sink, read_size=8 << 20, write_size=1 << 20)
        size = target.stat().st_size
        compressed.append(Entry(entry.path, offset, size, sha256_file(target)))
        offset += size
    _write_state(state, {"plan": plan, "files": [vars(e) for e in compressed]})
    return compressed


def store(
    *,
    keys: Keys,
    api: Any,
    repo: str,
    run: str,
    part: int,
    source: Path,
    work: Path,
    only: Sequence[str] | None = None,
    piece_bytes: int = PIECE_BYTES,
    segment: int = SEGMENT_BYTES,
    workers: int = 4,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Seal the files under ``source`` (just the folders named in ``only``, if given), upload
    them into the part's own folder of ``repo`` and return the receipt.

    Resumable: ``work`` keeps every sealed piece with the plan that made it, so a rerun over
    the same files reseals nothing, and the large-folder upload skips what the Hub holds.
    """
    remote = Remote(api)
    entries = scan(source, only)
    total = sum(entry.bytes for entry in entries)
    if not total:
        raise CacheError(f"{source} holds no files to store")
    packed = _compress_files(source, entries, work)
    compressed_total = sum(e.bytes for e in packed)
    plan = content_hash(
        {
            "compression": COMPRESSION,
            "run": run,
            "part": part,
            "piece_bytes": piece_bytes,
            "segment": segment,
            "files": [[e.path, e.bytes, e.sha256] for e in packed],
        }
    )
    plan_file = work / "plan.json"
    if not plan_file.exists() or json.loads(plan_file.read_text()).get("plan") != plan:
        shutil.rmtree(work / "upload", ignore_errors=True)
        shutil.rmtree(work / "state", ignore_errors=True)
        _write_state(plan_file, {"plan": plan})
    pieces = -(-compressed_total // piece_bytes)
    folder = folder_name(keys, run, part)
    local = work / "upload" / folder

    def seal(index: int) -> dict[str, Any]:
        name = piece_file(keys, run, part, index)
        target = local / name
        state = work / "state" / f"{name}.json"
        if target.exists() and state.exists():
            done = json.loads(state.read_text())
            if (
                done["plan"] == plan
                and target.stat().st_size == done["bytes"]
                and sha256_file(target) == done["sha256"]
            ):
                return {"name": name, "bytes": done["bytes"], "sha256": done["sha256"]}
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = target.with_name(target.name + ".partial")
        reader = _Slice(work / "compressed", packed, index * piece_bytes, piece_bytes)
        try:
            plain = hashlib.sha256()
            while block := reader.read(8 << 20):
                plain.update(block)
            reader.close()
            reader = _Slice(work / "compressed", packed, index * piece_bytes, piece_bytes)
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
        return {"name": name, "bytes": size, "sha256": sha}

    with ThreadPoolExecutor(max(1, min(workers, pieces))) as pool:
        sealed = list(pool.map(seal, range(pieces)))
    log(f"sealed {pieces} pieces")

    manifest = {
        "schema": MANIFEST_SCHEMA,
        "run": run,
        "part": part,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "piece_bytes": piece_bytes,
        "segment": segment,
        "compression": COMPRESSION,
        "stream_bytes": compressed_total,
        "uncompressed_bytes": total,
        "files": [
            {
                "path": e.path,
                "offset": c.offset,
                "bytes": e.bytes,
                "sha256": e.sha256,
                "compressed_bytes": c.bytes,
            }
            for e, c in zip(entries, packed, strict=True)
        ],
        "pieces": sealed,
    }
    body = canonical_json(manifest)
    body += b" " * (-len(body) % MANIFEST_GRAIN)
    manifest_name = manifest_file(keys, run, part)
    manifest_path = local / manifest_name
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

    api.create_repo(repo, repo_type="dataset", private=False, exist_ok=True)
    hub.retry_conflicts(
        lambda: api.upload_large_folder(
            repo_id=repo, folder_path=str(work / "upload"), repo_type="dataset"
        )
    )
    head = remote.head(repo)
    tree = remote.tree(repo, head, folder)
    for path in sorted(p for p in local.iterdir() if p.is_file()):
        name = f"{folder}/{path.name}"
        item = tree.get(name)
        if item is None or item.size != path.stat().st_size:
            raise CacheError(f"{repo}: {name} did not arrive intact")
        if remote.remote_sha256(repo, name, head, item, work / "verify") != sha256_file(path):
            raise CacheError(f"{repo}: {name} differs from what was uploaded")
    shutil.rmtree(work / "verify", ignore_errors=True)
    # A more compressible rebuild may need fewer pieces than the previous upload.
    from huggingface_hub import CommitOperationDelete

    wanted = {f"{folder}/{p.name}" for p in local.iterdir() if p.is_file()}

    def prune() -> bool:
        head = remote.head(repo)
        stale = sorted(
            name
            for name, item in remote.tree(repo, head, folder).items()
            if name not in wanted and getattr(item, "size", None) is not None
        )
        if not stale:
            return False
        api.create_commit(
            repo_id=repo,
            repo_type="dataset",
            parent_commit=head,
            operations=[CommitOperationDelete(path_in_repo=p) for p in stale[:100]],
            commit_message="Remove superseded compressed cache pieces",
        )
        return True

    while hub.retry_conflicts(prune):
        pass
    log(f"uploaded and verified {folder}")
    return {
        "schema": RECEIPT_SCHEMA,
        "run": run,
        "part": part,
        "files": len(entries),
        "stream_bytes": total,
        "compression": COMPRESSION,
        "compressed_bytes": compressed_total,
        "pieces": pieces,
        "repo": repo,
        "folder": folder,
    }


# ---- reading back -------------------------------------------------------------------------


def _safe_relative(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute() or ".." in candidate.parts or not candidate.parts:
        raise CacheError(f"Unsafe path in the manifest: {path!r}")
    return candidate


def read_manifest(
    remote: Remote,
    keys: Keys,
    *,
    repo: str,
    revision: str,
    run: str,
    part: int,
    folder: str,
    scratch: Path,
) -> dict[str, Any]:
    """A part's decrypted manifest from its folder at the pinned commit."""
    name = manifest_file(keys, run, part)
    path = remote.download(repo, f"{folder}/{name}", revision, scratch)
    try:
        with path.open("rb") as handle:
            body = b"".join(decrypt_stream(handle, keys, name.encode()))
    finally:
        path.unlink(missing_ok=True)
    manifest = json.loads(body)
    if (
        manifest.get("schema") != MANIFEST_SCHEMA
        or manifest["run"] != run
        or manifest["part"] != part
    ):
        raise CacheError(f"Part {part}: the manifest belongs to another run or part")
    for item in manifest["files"]:
        _safe_relative(item["path"])
    if manifest.get("compression") not in (None, COMPRESSION):
        raise CacheError(f"Unsupported cache compression: {manifest['compression']}")
    if manifest.get("compression") == COMPRESSION:
        if any("compressed_bytes" not in item for item in manifest["files"]):
            raise CacheError("Compressed cache is missing file lengths")
    return manifest


def verify_part(
    api: Any,
    keys: Keys,
    *,
    repo: str,
    revision: str,
    run: str,
    part: int,
    folder: str,
    scratch: Path,
) -> dict[str, Any]:
    """Check one part without downloading its data: the manifest decrypts, and every piece is
    stored at its exact size with the ciphertext hash the manifest records. Returns the
    manifest; raises ``CacheError`` for anything missing or altered."""
    remote = Remote(api)
    manifest = read_manifest(
        remote,
        keys,
        repo=repo,
        revision=revision,
        run=run,
        part=part,
        folder=folder,
        scratch=scratch,
    )
    tree = remote.tree(repo, revision, folder)
    for piece in manifest["pieces"]:
        name = f"{folder}/{piece['name']}"
        item = tree.get(name)
        if item is None or item.size != piece["bytes"]:
            raise CacheError(f"Part {part}: a piece is missing or the wrong size")
        if remote.remote_sha256(repo, name, revision, item, scratch) != piece["sha256"]:
            raise CacheError(f"Part {part}: a stored piece does not match its hash")
    return manifest


class _Unpacker:
    """Writes a part's files back from consecutive stretches of the plaintext stream, checking
    each file's sha256 before it is moved to its name."""

    def __init__(self, manifest: Mapping[str, Any], dest: Path, done: set[str]) -> None:
        self.files = [item for item in manifest["files"] if item["bytes"]]
        self.offsets = [item["offset"] for item in self.files]
        self.total = manifest["stream_bytes"]
        self.compressed = manifest.get("compression") == COMPRESSION
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
            end = item["offset"] + item.get("compressed_bytes", item["bytes"])
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
        if not self.compressed:
            self.hasher.update(block)

    def _close(self, item: Mapping[str, Any]) -> None:
        assert self.handle is not None
        self.handle.close()
        target = self.dest / _safe_relative(item["path"])
        partial = target.with_name(target.name + ".partial")
        self.handle, self.current = None, -1
        if self.compressed:
            import zstandard

            packed = partial.with_name(partial.name + ".zst")
            partial.replace(packed)
            self.hasher = hashlib.sha256()
            size = 0
            try:
                with (
                    packed.open("rb") as source,
                    zstandard.ZstdDecompressor().stream_reader(source) as reader,
                    partial.open("wb") as sink,
                ):
                    while block := reader.read(8 << 20):
                        size += len(block)
                        if size > item["bytes"]:
                            raise CacheError(f"{item['path']}: decompressed file exceeds its size")
                        sink.write(block)
                        self.hasher.update(block)
                if size != item["bytes"]:
                    raise CacheError(f"{item['path']}: decompressed file has the wrong size")
            except BaseException as error:
                partial.unlink(missing_ok=True)
                if isinstance(error, zstandard.ZstdError):
                    raise CacheError(f"{item['path']}: invalid compressed data") from error
                raise
            finally:
                packed.unlink(missing_ok=True)
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


def fetch_part(
    *,
    keys: Keys,
    api: Any,
    repo: str,
    revision: str,
    run: str,
    part: int,
    folder: str,
    dest: Path,
    scratch: Path,
    workers: int = 4,
    log: Callable[[str], None] = print,
) -> dict[str, int]:
    """Download one part of the pinned commit into ``dest``: every piece's ciphertext hash is
    checked before it is opened, and every file's plaintext hash before it is given its name.
    A rerun skips files already there and verified. Raises ``CacheError`` on any mismatch."""
    remote = Remote(api)
    dest.mkdir(parents=True, exist_ok=True)
    manifest = read_manifest(
        remote,
        keys,
        repo=repo,
        revision=revision,
        run=run,
        part=part,
        folder=folder,
        scratch=scratch,
    )
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
            and item["offset"] + item.get("compressed_bytes", item["bytes"]) > start
        ):
            wanted.append((index, piece))

    def download(item: tuple[int, Mapping[str, Any]]) -> Path:
        return remote.download(repo, f"{folder}/{item[1]['name']}", revision, scratch)

    with ThreadPoolExecutor(max(1, workers)) as pool:
        for (index, piece), path in _ahead(pool, wanted, download, workers):
            try:
                if sha256_file(path) != piece["sha256"]:
                    raise CacheError(f"Part {part}: a downloaded piece does not match its hash")
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
        item["path"] for item in manifest["files"] if item["bytes"] and item["path"] not in done
    ]
    if missing:
        raise CacheError(f"Part {part}: {len(missing)} files were not restored")
    for item in manifest["files"]:
        if not item["bytes"]:
            target = dest / _safe_relative(item["path"])
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"")
    log(f"part {part}: {len(manifest['files'])} cached files restored")
    return {
        "files": len(manifest["files"]),
        "bytes": manifest.get("uncompressed_bytes", manifest["stream_bytes"]),
    }


# ---- command line -------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="megatoken cache", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    item = commands.add_parser("keygen", help="create the data key (refuses to overwrite)")
    item.add_argument("--path", type=Path, default=KEY_FILE)
    args = parser.parse_args(argv)
    try:
        path = keygen(args.path)
    except CacheError as error:
        print(f"cache: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"key_file": str(path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
