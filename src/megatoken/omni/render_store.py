"""Immutable content-addressed rendered media, separate from selected raw group tapes.

A full 81-frame 720p video must not be base64 embedded in a control/replay JSON
tape. The raw selected action remains [32,D] per event; only physical media
bytes are stored here, never exposed to P/C as another learned content path.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from megatoken.io import atomic_binary_writer

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True, slots=True)
class RenderReceipt:
    sha256: str
    byte_count: int

    def __post_init__(self):
        if not _SHA256.fullmatch(self.sha256) or type(self.byte_count) is not int or self.byte_count < 1:
            raise ValueError("Rendered output needs SHA256 and positive byte count")


class RenderedOutputStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def _path(self, sha256: str):
        if not _SHA256.fullmatch(sha256):
            raise ValueError("Invalid immutable media output digest")
        return self.root / "rendered" / (sha256 + ".bin")

    def put(self, payload: bytes) -> RenderReceipt:
        if not isinstance(payload, bytes) or not payload:
            raise ValueError("Cannot store an empty media output")
        digest = hashlib.sha256(payload).hexdigest()
        receipt = RenderReceipt(digest, len(payload))
        path = self._path(digest)
        if path.exists():
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError("Stored rendered artifact differs from its SHA256")
            return receipt
        with atomic_binary_writer(path) as out:
            out.write(payload)
        return receipt

    def get(self, receipt: RenderReceipt) -> bytes:
        payload = self._path(receipt.sha256).read_bytes()
        if len(payload) != receipt.byte_count or hashlib.sha256(payload).hexdigest() != receipt.sha256:
            raise ValueError("Rendered media asset was altered after materialization")
        return payload
