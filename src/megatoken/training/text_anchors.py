"""Same-text alignment of byte-E facets with teacher-feature facets.

The media bridge (teacher hidden states -> projection -> shared E) must land in
the same facet space that byte-E gives text. For a text chunk with cached
EmbeddingGemma2 features, the anchor is the mean squared difference between
E(bytes) and E(teacher features). It is never a lexical target. Which side moves
is decided by what is trainable: in ``media_bridge`` training only the bridge
does; in joint training both do.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import torch
from torch import Tensor

from megatoken.data.media_assets import TensorAsset, TensorAssetStore
from megatoken.tensors import byte_tensor


class TextAnchors:
    """JSONL rows ``{"text_sha256": <sha of exact record bytes>, "asset": TensorAsset,
    "embedding": TensorAsset}``: the teacher features and pooled embedding of that text.

    The assets live in whichever of ``stores`` (one per prepared root) holds them."""

    def __init__(self, path: str | Path, stores: Sequence[TensorAssetStore], *, extractor: str):
        self.stores = tuple(stores)
        self.assets: dict[str, TensorAsset] = {}
        self.embeddings: dict[str, TensorAsset] = {}
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            asset = TensorAsset.from_dict(row["asset"])
            embedding = TensorAsset.from_dict(row["embedding"])
            if asset.kind != "features" or asset.modality != "text":
                raise ValueError("Text anchors are EmbeddingGemma2 text features")
            if (
                embedding.kind != "embedding"
                or embedding.source_family != asset.source_family
                or embedding.extractor != asset.extractor
            ):
                raise ValueError("A text anchor's embedding must come from the same teacher pass")
            if asset.extractor != extractor:
                raise ValueError("Text anchor features come from a different teacher")
            if row["text_sha256"] in self.assets or asset.source_family != row["text_sha256"]:
                raise ValueError("Each anchor covers exactly the text it was computed from")
            self.assets[row["text_sha256"]] = asset
            self.embeddings[row["text_sha256"]] = embedding
        if not self.assets:
            raise ValueError("Empty text anchor sidecar")

    @staticmethod
    def key(payload: bytes) -> str:
        return hashlib.sha256(payload).hexdigest()

    def features(self, key: str, device: torch.device) -> Tensor:
        asset = self.assets[key]
        for store in self.stores:
            if (store.root / asset.relative_path).exists():
                return torch.from_numpy(store.load(asset)).to(device)[None]
        raise ValueError("A text anchor's features are in none of the prepared roots")


def anchor_loss(model, payload: bytes, features: Tensor) -> Tensor:
    """MSE between E(exact bytes) and E(teacher features of those bytes)."""
    ids, lengths = byte_tensor([payload], features.device)
    student = model.core.encode_chunk(ids, lengths)
    teacher = model.core.encode_media(features, extractor=model.config.media_extractor)
    return (student.float() - teacher.float()).square().mean()
