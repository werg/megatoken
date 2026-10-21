"""Audited same-source text hidden-state targets for native E alignment.

JSONL rows are the descriptor JSON emitted by cache_embeddinggemma2_hidden.py.
The source SHA256 must match actual prepared record bytes. Teacher vectors are
never substituted for native byte E or exposed as next-token targets.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import torch

from megatoken.omni.embeddinggemma2_hidden import HIDDEN_TAP

_SHA = re.compile(r"[0-9a-f]{64}\Z")


class TextHiddenSidecar:
    def __init__(self, manifest: str, root: str, *, extractor: str):
        self.root = Path(root).resolve()
        self.extractor = extractor
        self.rows = {}
        for line in Path(manifest).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            asset = entry.get("asset", {})
            source_sha = entry.get("teacher_input_sha256") or entry.get("source_sha256", "")
            shape = asset.get("shape")
            if (
                entry.get("format") != "megatoken.embeddinggemma2.hidden.v1"
                or entry.get("modality") not in ("text", "document")
                or entry.get("hidden_tap") != HIDDEN_TAP
                or not _SHA.fullmatch(source_sha)
                or not _SHA.fullmatch(asset.get("sha256", ""))
                or asset.get("extractor") != extractor
                or not isinstance(shape, list)
                or len(shape) != 2
                or type(shape[0]) is not int
                or not 1 <= shape[0] <= 8192
                or shape[1] != 512
                or entry.get("valid_tokens") != shape[0]
            ):
                raise ValueError("Text sidecar needs an exact unpooled EmbeddingGemma2 cache")
            location = (self.root / asset.get("path", "")).resolve()
            if not location.is_relative_to(self.root) or not location.is_file():
                raise ValueError("Text teacher asset must be inside the prepared root")
            if source_sha in self.rows:
                raise ValueError("Duplicate text teacher source SHA256")
            self.rows[source_sha] = (location, asset["sha256"], shape)
        if not self.rows:
            raise ValueError("Empty pinned EmbeddingGemma2 text alignment sidecar")

    def __contains__(self, source: bytes) -> bool:
        return hashlib.sha256(source).hexdigest() in self.rows

    def get(self, source: bytes, device):
        record = self.rows.get(hashlib.sha256(source).hexdigest())
        if record is None:
            return None
        path, digest, shape = record
        raw = path.read_bytes()
        if len(raw) != shape[0] * shape[1] * 4 or hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError("Text hidden-state source was mutated or substituted")
        features = np.frombuffer(raw, dtype="<f4").copy().reshape(shape)
        if not np.isfinite(features).all():
            raise ValueError("Text hidden-state source is nonfinite")
        return torch.from_numpy(features)[None].to(device)
