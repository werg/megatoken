"""Detached canonical E(source) features for bilateral RAG utility losses.

The source is an immutable ORIGINAL byte interval returned by the eligible ANN
row. Neither the retrieved source features nor teacher answer labels can enter
the predictor's causal query controls. Actual RAG byte observations still recur
through the ordinary native E/C track.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional as F

from megatoken.memory.base import Record
from megatoken.memory.contracts import Collection
from megatoken.memory.retrieval import Interval
from megatoken.tensors import byte_tensor
from megatoken.io import sha256_file
from hashlib import sha256


class NativeSourceFeatures:
    def __init__(self, model, assets_root: str | Path, *,
                 max_chunk_bytes: int = 16384, strict_hash: bool = False):
        root = Path(assets_root).resolve()
        if not root.is_dir() or max_chunk_bytes < 1:
            raise ValueError("RAG value training requires a pinned source asset directory")
        self.root, self.model, self.max_chunk_bytes = root, model, max_chunk_bytes
        self.strict_hash = strict_hash
        self._verified_assets: dict[tuple[str, str], bool] = {}
        self._features: dict[tuple[int, int], Tensor] = {}

    def read_interval(self, row: Record, interval: Interval) -> bytes:
        """Fetch only reserved original source coordinates and verify identity."""
        span = row.evidence_span
        if row.collection is not Collection.SOURCE or span is None or (
            interval.content_id != span.source_id or interval.version != span.version
            or not span.start <= interval.start < interval.end <= span.end
        ):
            raise ValueError("Source obligation differs from original versioned byte interval")
        if interval.end - interval.start > self.max_chunk_bytes:
            raise ValueError("Source evidence exceeds bounded byte budget")
        path = (self.root / row.asset_ref).resolve()
        if not path.is_relative_to(self.root) or not path.is_file():
            raise ValueError("Source evidence path is absent or escapes immutable assets")
        with path.open("rb") as stream:
            stream.seek(interval.start)
            content = stream.read(interval.end - interval.start)
        if len(content) != interval.end - interval.start:
            raise ValueError("Original source evidence is truncated")
        if self.strict_hash:
            digest = span.content_hash
            if len(digest) != 64 or any(x not in "0123456789abcdef" for x in digest):
                raise ValueError("Training source ranges require pinned SHA256 content digests")
            identity = (str(path), digest)
            if identity not in self._verified_assets:
                # Catalog contracts may identify either the full immutable
                # source version or the exact chunk. Accept precisely those
                # two declared bytes, never a different stale source revision.
                whole = sha256_file(path)
                local = sha256(content).hexdigest()
                if digest not in {whole, local}:
                    raise ValueError("Original source bytes differ from pinned source hash")
                self._verified_assets[identity] = True
        return content

    @torch.no_grad()
    def get(self, row: Record) -> Tensor:
        if row.collection is not Collection.SOURCE or row.evidence_span is None:
            raise ValueError("Only exact original source bytes have E(source) features")
        identity = (row.id, self.model.core.model_version)
        if identity in self._features:
            return self._features[identity]
        interval = row.evidence_span
        content = self.read_interval(
            row, Interval(interval.source_id, interval.version, interval.start, interval.end),
        )
        device = self.model.core.initial_facets.device
        bytes_tensor, lengths = byte_tensor([content], device)
        raw = self.model.core.encode_chunk(bytes_tensor, lengths)
        # A stable semantic feature of the actual source. This is a DETACHED
        # objective target/value, never an online query embedding teacher.
        feature = F.normalize(raw.detach().float().mean(dim=1), dim=-1)[0]
        self._features[identity] = feature
        return feature

    def invalidate_after_step(self):
        # E itself is trainable in the main optimizer. Cached features are
        # version-local and cannot be used as authoritative values afterwards.
        self._features.clear()
        self._verified_assets.clear()
