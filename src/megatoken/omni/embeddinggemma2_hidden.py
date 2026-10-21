"""Offline EmbeddingGemma2 hidden-state extraction (NOT 768-d pooled embeddings).

The model's language_model.embedding_projection is its 512->768 head. A
forward-pre-hook captures ordered [B,T,512] states immediately before that
projection, after modality adapters and the bidirectional text backbone.
Only local, immutable, revision-pinned snapshots are permitted.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

SNAPSHOT_MODEL = "google/embeddinggemma-2"
HIDDEN_TAP = "language_model.embedding_projection:input"
_MODALITIES = frozenset({"text", "document", "audio", "image", "video"})


def hidden_identity(revision: str, *, preprocessing: str = "chat-template-v1") -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("EmbeddingGemma2 teacher identity requires an immutable 40-char revision")
    if not preprocessing or not re.fullmatch(r"[a-z0-9_.-]+", preprocessing):
        raise ValueError("Preprocessor needs a pinned nonempty revision")
    return f"embeddinggemma2:{revision}:{HIDDEN_TAP}:{preprocessing}"


@dataclass(frozen=True)
class HiddenObservation:
    features: Tensor
    mask: Tensor
    modality: str
    extractor: str

    def __post_init__(self) -> None:
        if self.modality not in _MODALITIES:
            raise ValueError("Unsupported EmbeddingGemma2 hidden-state modality")
        if (
            self.features.ndim != 3 or self.features.shape[0] != 1
            or self.features.shape[-1] != 512
            or self.mask.shape != self.features.shape[:2]
            or self.mask.dtype != torch.bool or not bool(self.mask.any())
            or not bool(torch.isfinite(self.features).all())
            or not self.extractor.startswith("embeddinggemma2:")
        ):
            raise ValueError("Expected one finite masked [1,T,512] hidden-state observation")


class EmbeddingGemma2HiddenExtractor:
    """Frozen inference-only teacher; its weights never join Megatoken checkpoints."""

    def __init__(self, model, processor, *, revision: str, preprocessing: str = "chat-template-v1"):
        self.model = model.eval()
        self.processor = processor
        self.extractor = hidden_identity(revision, preprocessing=preprocessing)
        if (
            not hasattr(model, "language_model")
            or not hasattr(model.language_model, "embedding_projection")
            or model.language_model.embedding_projection.in_features != 512
            or model.language_model.embedding_projection.out_features != 768
        ):
            raise ValueError("Snapshot is not the known EmbeddingGemma2 512->768 head")
        for param in model.parameters():
            param.requires_grad_(False)

    @classmethod
    def from_local_snapshot(cls, snapshot: str | Path, *, revision: str, device: str = "cpu"):
        """Never silently load main/latest or download weights during preprocessing."""
        from transformers import AutoModel, AutoProcessor

        path = Path(snapshot).resolve()
        if not path.is_dir() or not (path / "config.json").is_file():
            raise FileNotFoundError("A downloaded local EmbeddingGemma2 snapshot is required")
        metadata = path / "teacher-lock.json"
        if not metadata.is_file():
            raise ValueError("Teacher snapshot needs a checked teacher-lock.json")
        lock = json.loads(metadata.read_text(encoding="utf-8"))
        if lock != {"model": SNAPSHOT_MODEL, "revision": revision}:
            raise ValueError("Teacher snapshot/processor revision identity mismatch")
        model = AutoModel.from_pretrained(path, local_files_only=True, trust_remote_code=False).to(device)
        processor = AutoProcessor.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        return cls(model, processor, revision=revision)

    @torch.no_grad()
    def encode(self, modality: str, source, *, max_tokens: int = 8192) -> HiddenObservation:
        if modality not in _MODALITIES:
            raise ValueError("Expected text, document, audio, image or video")
        if not 1 <= max_tokens <= 8192:
            raise ValueError("EmbeddingGemma2 sequence budget must fit its 8192-token context")
        # Documents are text after auditable parsing; scanned pages must be
        # rendered separately as images. Do not pretend the model decodes PDFs.
        kind = "text" if modality == "document" else modality
        conversation = [{"role": "user", "content": [{"type": kind, kind: source}]}]
        inputs = self.processor.apply_chat_template(
            conversation, tokenize=True, return_dict=True, return_tensors="pt"
        )
        if "attention_mask" not in inputs:
            raise ValueError("Processor did not preserve a valid-token mask")
        mask = inputs["attention_mask"].bool()
        if mask.ndim != 2 or mask.shape[0] != 1 or int(mask.sum()) > max_tokens:
            raise ValueError("Teacher context exceeds the declared token budget; chunk or sample first")
        # Hook rather than last_hidden_state: the latter is *already* projected
        # to 768, as is any SentenceTransformer.encode() result.
        captured: list[Tensor] = []
        def capture(module, arguments):
            if len(arguments) != 1:
                raise RuntimeError("Unexpected EmbeddingGemma2 projection signature")
            captured.append(arguments[0].detach())
        handle = self.model.language_model.embedding_projection.register_forward_pre_hook(capture)
        try:
            device = next(self.model.parameters()).device
            batch = {k: v.to(device) if isinstance(v, Tensor) else v for k, v in inputs.items()}
            self.model(**batch, return_dict=True)
        finally:
            handle.remove()
        if len(captured) != 1 or captured[0].shape[:2] != mask.shape:
            raise RuntimeError("EmbeddingGemma2 internal-state hook was not invoked exactly once")
        return HiddenObservation(captured[0].float().cpu(), mask.cpu(), modality, self.extractor)
