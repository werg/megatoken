"""Immutable, bounded file-backed storage for rendered media bytes."""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from threading import RLock


class FileOutputStore:
    """Content-addressed media store. Handles are scoped to their owner session."""

    def __init__(self, root: str | Path | None = None, *, max_bytes: int = 512 << 20):
        if type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError("Output store byte budget must be positive")
        self._temporary = (
            tempfile.TemporaryDirectory(prefix="megatoken-output-") if root is None else None
        )
        self.root = Path(self._temporary.name if self._temporary else root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_bytes
        self._entries: dict[str, tuple[set[str], int, str]] = {}
        self._sizes: dict[str, int] = {}
        self._reservations: dict[str, int] = {}
        self._lock = RLock()

    def put(self, owner: str, data: bytes) -> dict:
        if not data:
            raise ValueError("Stored media payloads must be nonempty")
        digest = hashlib.sha256(data).hexdigest()
        handle = hashlib.sha256(owner.encode() + b"\0" + digest.encode()).hexdigest()
        with self._lock:
            prior = self._entries.get(handle)
            if prior is not None:
                prior[0].add(owner)
                return {"handle": handle, "sha256": digest, "byte_length": len(data)}
            path = self.root / digest
            reserved = self._reservations.get(owner, 0)
            increment = 0 if digest in self._sizes else len(data)
            projected = (
                sum(self._sizes.values())
                + sum(self._reservations.values())
                - min(reserved, len(data))
                + increment
            )
            if projected > self.max_bytes:
                raise ValueError("Stored media byte budget exhausted")
            if digest not in self._sizes:
                fd, tmp = tempfile.mkstemp(dir=self.root, prefix=".pending-")
                try:
                    with os.fdopen(fd, "wb") as stream:
                        stream.write(data)
                        stream.flush()
                        os.fsync(stream.fileno())
                    os.replace(tmp, path)
                    path.chmod(0o444)
                finally:
                    if os.path.exists(tmp):
                        os.unlink(tmp)
                self._sizes[digest] = len(data)
            if reserved:
                self._reservations[owner] = max(0, reserved - len(data))
                if not self._reservations[owner]:
                    self._reservations.pop(owner, None)
            self._entries[handle] = ({owner}, len(data), digest)
        return {"handle": handle, "sha256": digest, "byte_length": len(data)}

    def reserve(self, owner: str, byte_count: int) -> None:
        if not owner or type(byte_count) is not int or byte_count < 0:
            raise ValueError("Invalid output storage reservation")
        with self._lock:
            old = self._reservations.get(owner, 0)
            reserved_total = sum(self._reservations.values()) - old
            if sum(self._sizes.values()) + reserved_total + byte_count > self.max_bytes:
                raise ValueError("Stored media byte budget exhausted before generation")
            if byte_count:
                self._reservations[owner] = byte_count
            else:
                self._reservations.pop(owner, None)

    def release_reservation(self, owner: str) -> None:
        with self._lock:
            self._reservations.pop(owner, None)

    def read(self, owner: str, handle: str, offset: int, length: int) -> bytes:
        if (
            type(offset) is not int
            or offset < 0
            or type(length) is not int
            or not 1 <= length <= 1 << 20
        ):
            raise ValueError("Media reads need a nonnegative offset and a 1 MiB maximum chunk")
        with self._lock:
            entry = self._entries.get(handle)
            if entry is None or owner not in entry[0]:
                raise ValueError("Unknown media reference for this session")
            size = entry[1]
            if offset >= size:
                raise ValueError("Media read offset is outside the stored payload")
            path = self.root / entry[2]
            if path.is_symlink():
                raise ValueError("Stored media failed its SHA256 integrity check")
            with path.open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != entry[2]:
                    raise ValueError("Stored media failed its SHA256 integrity check")
                stream.seek(offset)
                return stream.read(min(length, size - offset))

    def delete_owner(self, owner: str) -> None:
        with self._lock:
            self._reservations.pop(owner, None)
            for handle, (owners, _, digest) in tuple(self._entries.items()):
                if owner in owners:
                    owners.discard(owner)
                if not owners:
                    self._entries.pop(handle, None)
                    if not any(entry[2] == digest for entry in self._entries.values()):
                        (self.root / digest).unlink(missing_ok=True)
                        self._sizes.pop(digest, None)

    def close(self) -> None:
        with self._lock:
            for digest in tuple(self._sizes):
                (self.root / digest).unlink(missing_ok=True)
            self._entries.clear()
            self._sizes.clear()
            self._reservations.clear()
            if self._temporary:
                self._temporary.cleanup()
