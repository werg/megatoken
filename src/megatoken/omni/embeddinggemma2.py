"""The one EmbeddingGemma2 runtime: per-token hidden states and pooled embeddings.

One forward pass gives both things Megatoken takes from the teacher:

- ``features``: the ordered ``[T, 512]`` input of
  ``language_model.embedding_projection`` (final normalized hidden states,
  after the modality towers and the bidirectional backbone). Media, and text
  read in compressed form, enter E as these.
- ``embedding``: the teacher's own sentence embedding, the mean over every
  token (prompt included) of that projection's 768-wide output, L2-normalized,
  exactly as its sentence-transformers configuration pools. Retrieval keys and
  embedding targets are these.

Inputs are formatted by the processor's chat template, optionally prefixed by
a task prompt (``title: ... | text: ``, ``task: ... | query: ``). Weights load
only from the locked pack manifest, and every output carries the pinned
teacher identity. Nothing is truncated: sequences over 8192 tokens are refused.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from megatoken.teacher import EMBEDDING_WIDTH, HIDDEN_WIDTH, MAX_TOKENS, locked_teacher

MODALITIES = ("text", "audio", "image", "video")


@dataclass(frozen=True)
class TeacherOutput:
    """One input's ordered teacher states and pooled embedding."""

    features: Tensor
    embedding: Tensor
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
        if self.embedding.shape != (EMBEDDING_WIDTH,) or not bool(
            torch.isfinite(self.embedding).all()
        ):
            raise ValueError("Expected a finite 768-wide EmbeddingGemma2 embedding")


class EmbeddingGemma2:
    """Frozen, resident, inference-only teacher; never part of a Megatoken checkpoint."""

    def __init__(self, model, processor, *, extractor: str):
        projection = getattr(getattr(model, "language_model", None), "embedding_projection", None)
        if projection is None or (projection.in_features, projection.out_features) != (
            HIDDEN_WIDTH,
            EMBEDDING_WIDTH,
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

    @classmethod
    def from_manifest(cls, manifest: str | Path, *, device: str = "cuda"):
        from megatoken.omni.loader import read_locked_manifest

        base, packs = read_locked_manifest(str(manifest))
        return cls.from_pack(base, packs, device=device)

    def encode(self, modality: str, source: str, *, prompt: str = "") -> TeacherOutput:
        """``source`` is the text itself, or a local audio/image/video file path."""
        return self.encode_batch([(modality, source)], prompt=prompt)[0]

    def text_pieces(self, text: str, *, prompt: str = "", margin: int = 8) -> list[str]:
        """Split ``text`` into consecutive pieces that each fit one teacher input.

        Pieces are cut by the teacher's own token count (the prompt, special
        tokens and a small margin included), at a line break in the last tenth
        of a full piece when there is one; joined, they are ``text`` exactly.
        """
        tokenizer = self.processor.tokenizer
        budget = MAX_TOKENS - margin - len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        starts = [
            begin
            for begin, _ in tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)[
                "offset_mapping"
            ]
        ]
        pieces, start, index = [], 0, 0
        while index + budget < len(starts):
            end = starts[index + budget]
            cut = text.rfind("\n", starts[index + budget * 9 // 10], end)
            end = cut + 1 if cut >= 0 else end
            pieces.append(text[start:end])
            start = end
            index = bisect.bisect_left(starts, end, lo=index + 1)
        pieces.append(text[start:])
        return pieces

    def embed_texts(
        self, texts: list[str], *, prompt: str, batch_tokens: int = 32768
    ) -> np.ndarray:
        """Unit ``[N, 768]`` embeddings of ``prompt + text``, batched by length."""
        out = np.zeros((len(texts), EMBEDDING_WIDTH), dtype=np.float32)
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        start = 0
        while start < len(order):
            # ~3 characters per token sizes a batch of similar lengths.
            longest = max(1, len(texts[order[min(start + 255, len(order) - 1)]]) // 3)
            ids = order[start : start + max(1, min(256, batch_tokens // longest))]
            outputs = self.encode_batch([("text", texts[i]) for i in ids], prompt=prompt)
            for i, item in zip(ids, outputs, strict=True):
                out[i] = item.embedding.numpy()
            start += len(ids)
        return out

    @torch.inference_mode()
    def encode_batch(
        self, items: list[tuple[str, str]], *, prompt: str = ""
    ) -> list[TeacherOutput]:
        if not items or any(modality not in MODALITIES for modality, _ in items):
            raise ValueError("Teacher input must be text, audio, image or video")
        conversations = []
        for modality, source in items:
            content = (
                [{"type": "text", "text": prompt + source}]
                if modality == "text"
                else [
                    *([{"type": "text", "text": prompt}] if prompt else []),
                    {"type": modality, "path": str(source)},
                ]
            )
            conversations.append([{"role": "user", "content": content}])
        inputs = self.processor.apply_chat_template(
            conversations,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={"padding": True},
        )
        mask = inputs.get("attention_mask")
        if mask is None or mask.shape[0] != len(items):
            raise ValueError("The processor must return one attention mask row per input")
        mask = mask.bool()
        if int(mask.sum(1).max()) > MAX_TOKENS:
            raise ValueError("Teacher context exceeds 8192 tokens; split the input first")
        captured: list[tuple[Tensor, Tensor]] = []

        def capture(module, args, kwargs, output):
            captured.append((args[0] if args else kwargs["input"], output))

        handle = self._projection.register_forward_hook(capture, with_kwargs=True)
        try:
            device = self._projection.weight.device
            self.model(
                **{k: v.to(device) if isinstance(v, Tensor) else v for k, v in inputs.items()}
            )
        finally:
            handle.remove()
        if len(captured) != 1 or captured[0][0].shape[:2] != mask.shape:
            raise RuntimeError("The hidden-state tap must fire exactly once per batch")
        hidden, projected = (x.float().cpu() for x in captured[0])
        results = []
        for row, (modality, _) in enumerate(items):
            keep = mask[row]
            pooled = projected[row][keep].mean(0)
            results.append(
                TeacherOutput(
                    hidden[row][keep],
                    torch.nn.functional.normalize(pooled, dim=0),
                    modality,
                    self.extractor,
                )
            )
        return results
