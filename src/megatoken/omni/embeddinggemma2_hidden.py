"""Offline EmbeddingGemma2 hidden-state extraction (not its 768-d pooled embedding).

The teacher's ``language_model.embedding_projection`` maps final normalized
512-wide hidden states to 768-wide embedding coordinates, per token. A
forward-pre-hook captures its ordered ``[T, 512]`` input after the modality
towers and the bidirectional text backbone. Weights are loaded only from the
locked pack manifest; the extractor identity is derived from that lock.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from megatoken.media_teacher import HIDDEN_WIDTH, MAX_TOKENS, locked_teacher

MODALITIES = ("text", "audio", "image", "video")


@dataclass(frozen=True)
class HiddenObservation:
    """One event's ordered teacher states; every position is real evidence."""

    features: Tensor
    modality: str
    extractor: str

    def __post_init__(self) -> None:
        if (
            self.modality not in MODALITIES
            or self.features.ndim != 2
            or not 1 <= self.features.shape[0] <= MAX_TOKENS
            or self.features.shape[1] != HIDDEN_WIDTH
            or self.features.dtype != torch.float32
            or not bool(torch.isfinite(self.features).all())
        ):
            raise ValueError("Expected finite FP32 [T, 512] EmbeddingGemma2 hidden states")


class EmbeddingGemma2HiddenExtractor:
    """Frozen, resident, inference-only teacher; never part of a Megatoken checkpoint."""

    def __init__(self, model, processor, *, extractor: str):
        projection = getattr(getattr(model, "language_model", None), "embedding_projection", None)
        if projection is None or (projection.in_features, projection.out_features) != (
            HIDDEN_WIDTH,
            768,
        ):
            raise ValueError("Snapshot is not the EmbeddingGemma2 512->768 embedding head")
        self.model = model.eval().requires_grad_(False)
        self.processor = processor
        self.extractor = extractor
        self._projection = projection

    @classmethod
    def from_pack(cls, base: Path, packs: dict, *, device: str = "cuda"):
        """Load bf16 weights from a verified pack manifest (``read_locked_manifest``)."""
        from transformers import AutoModel, AutoProcessor

        snapshot, extractor = locked_teacher(base, packs)
        model = AutoModel.from_pretrained(
            snapshot, dtype=torch.bfloat16, local_files_only=True, trust_remote_code=False
        ).to(device)
        processor = AutoProcessor.from_pretrained(
            snapshot, local_files_only=True, trust_remote_code=False
        )
        return cls(model, processor, extractor=extractor)

    @torch.inference_mode()
    def encode(self, modality: str, source: str) -> HiddenObservation:
        """``source`` is the text itself, or a local audio/image/video file path."""
        if modality not in MODALITIES:
            raise ValueError("Teacher input must be text, audio, image or video")
        content = (
            {"type": "text", "text": source}
            if modality == "text"
            else {"type": modality, "path": str(source)}
        )
        inputs = self.processor.apply_chat_template(
            [{"role": "user", "content": [content]}],
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        mask = inputs.get("attention_mask")
        if mask is None or mask.shape[0] != 1 or not bool(mask.bool().all()):
            raise ValueError("A single unpadded teacher sequence is required")
        if mask.shape[1] > MAX_TOKENS:
            raise ValueError("Teacher context exceeds 8192 tokens; split the event first")
        captured: list[Tensor] = []

        def capture(module, args, kwargs):
            captured.append(args[0] if args else kwargs["input"])

        handle = self._projection.register_forward_pre_hook(capture, with_kwargs=True)
        try:
            device = self._projection.weight.device
            self.model(
                **{k: v.to(device) if isinstance(v, Tensor) else v for k, v in inputs.items()}
            )
        finally:
            handle.remove()
        if len(captured) != 1 or captured[0].shape[:2] != mask.shape:
            raise RuntimeError("The hidden-state tap must fire exactly once per sequence")
        return HiddenObservation(captured[0][0].float().cpu(), modality, self.extractor)
