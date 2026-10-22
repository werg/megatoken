"""Encode-and-reconstruct: train the feature encoder without the core.

Each optimizer step also draws a batch of prepared feature items (media inputs
and outputs alike, and compressed-text pieces) from the training split. Their
EmbeddingGemma2 states go through E only (projection, shared reductions,
readout) to 32 raw facets; nothing passes through C or P. Two objectives:

- **reconstruction** (media with codec latents): the item's realizer
  reproduces its own codec latents from a noisy sample of those facets, the
  same noisy-codec reading used by in-episode reconstruction;
- **embedding**: the embedding readout maps the facets into the teacher space,
  scored with the centered alignment distance against the teacher's own pooled
  embedding of the item (``generation.embedding.alignment_distance``).

This shapes the media projection, E, the realizer conditions and the embedding
readout at encoder cost, while the episodes' prediction-based reproduction
keeps shaping the same projections through the core. How many items each
step draws, and the objectives' weights, are set per stage in the stage plan.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from megatoken.data import FeatureItem
from megatoken.data.media_assets import MEDIA
from megatoken.generation.embedding import EmbeddingSpec, alignment_distance
from megatoken.teacher import EMBEDDING_WIDTH, HIDDEN_WIDTH

MODALITIES = ("text", *MEDIA)


@dataclass(frozen=True, slots=True)
class AutoencodePlan:
    """Per-step items and objective weights for one stage (``stages.json``)."""

    items: int = 0
    reconstruction_weight: float = 1.0
    embedding_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.items < 0 or min(self.reconstruction_weight, self.embedding_weight) < 0:
            raise ValueError("Encode-and-reconstruct items and weights are nonnegative")

    @classmethod
    def from_stage(cls, constraints: dict[str, Any]) -> AutoencodePlan:
        return cls(**constraints.get("autoencode", {}))


class FeatureSampler:
    """Seeded draws of training-split feature items, modalities in equal turns."""

    def __init__(self, dataset: Any, *, seed: int, worker: int) -> None:
        self.dataset = dataset
        self.counts = {m: dataset.feature_count("train", m) for m in MODALITIES}
        self.modalities = [m for m in MODALITIES if self.counts[m]]
        self.random = random.Random(f"autoencode:{seed}:{worker}")
        self.turn = self.random.randrange(max(1, len(self.modalities)))

    def __bool__(self) -> bool:
        return bool(self.modalities)

    def state_dict(self) -> dict[str, Any]:
        """Everything the next draw depends on, for exact resume."""
        return {"random": self.random.getstate(), "turn": self.turn}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        version, internal, gauss = state["random"]
        self.random.setstate((version, tuple(internal), gauss))
        self.turn = int(state["turn"])

    def draw(self, count: int) -> list[FeatureItem]:
        items = []
        for _ in range(count):
            modality = self.modalities[self.turn % len(self.modalities)]
            self.turn += 1
            position = self.random.randrange(self.counts[modality])
            items.append(self.dataset.feature_item("train", modality, position))
        return items


def embedding_center(sampler: FeatureSampler, limit: int = 256) -> Tensor:
    """The mean unit teacher embedding of up to ``limit`` training feature items.

    It is the reference the centered alignment distance subtracts, so the
    readout is scored on what distinguishes an item, not on the space's mean.
    """
    vectors = []
    for modality in sampler.modalities:
        for position in range(min(sampler.counts[modality], limit // len(sampler.modalities) + 1)):
            item, assets = sampler.dataset.feature_item("train", modality, position)
            vectors.append(
                torch.nn.functional.normalize(
                    torch.from_numpy(assets.load(item.embedding)).float(), dim=-1
                )
            )
    return torch.stack(vectors).mean(0)


def autoencode_losses(
    model: Any,
    items: list[FeatureItem],
    *,
    space_fingerprint: str,
    center: Tensor,
    generator: torch.Generator | None = None,
) -> dict[str, tuple[Tensor, int]]:
    """Summed reconstruction and embedding distances of ``items``, and their counts."""
    if center.shape != (EMBEDDING_WIDTH,) or model.embedding_realizer.dimension != EMBEDDING_WIDTH:
        raise ValueError("Embedding distillation needs the teacher's 768-wide space")
    device = model.core.initial_facets.device
    zero = model.core.initial_facets.float().sum() * 0
    reconstruction, reconstructed = zero, 0
    embedding, embedded = zero, 0
    by_modality: dict[str, list[FeatureItem]] = {}
    for item in items:
        by_modality.setdefault(item.record.modality, []).append(item)
    for modality, group in by_modality.items():
        sources = [torch.from_numpy(assets.load(record.source)) for record, assets in group]
        length = max(source.shape[0] for source in sources)
        features = torch.zeros(len(group), length, HIDDEN_WIDTH, device=device)
        mask = torch.zeros(len(group), length, dtype=torch.bool, device=device)
        for row, source in enumerate(sources):
            features[row, : source.shape[0]] = source.to(device)
            mask[row, : source.shape[0]] = True
        raw = model.core.encode_media(features, mask, extractor=model.config.media_extractor)
        readout = model.embedding_realizer(
            raw, EmbeddingSpec(space_fingerprint, "document", center.shape[0])
        )
        targets = torch.stack(
            [torch.from_numpy(assets.load(record.embedding)) for record, assets in group]
        ).to(device)
        embedding = embedding + alignment_distance(readout, targets, center).sum()
        embedded += len(group)
        if modality not in MEDIA:
            continue
        noise = math.log(model.config.codec_noise_std)
        sample = raw.float() + torch.randn(
            raw.shape, device=device, generator=generator
        ) * math.exp(noise)
        realizer = model.media_realizers.resident(modality)
        for row, (record, assets) in enumerate(group):
            if record.target is None:
                continue
            if realizer.identity != record.target.extractor:
                raise ValueError("Target latents were encoded by a different realizer pack")
            latents = torch.from_numpy(assets.load(record.target)).to(device)[None]
            valid = None
            if record.target.valid_count:
                frames = torch.arange(latents.shape[1], device=device)
                valid = frames[None] < record.target.valid_count
            reconstruction = reconstruction + realizer.loss(
                sample[row : row + 1].to(raw.dtype), latents, valid, generator
            )
            reconstructed += 1
    return {
        "reconstruction": (reconstruction, reconstructed),
        "embedding": (embedding, embedded),
    }
