"""Test doubles for the single media path: teacher features and fake realizers.

``FakeRealizer`` is a real ``Realizer`` (same identity, frozen-donor ownership
and facet condition) whose frozen "donor" is a tiny linear map, so the model's
media code runs end to end without downloading Pocket or Sana.
"""

from __future__ import annotations

import hashlib

import numpy as np
import torch
from torch import Tensor, nn

from megatoken.data.media_assets import MediaRecord, OutputGeometry, TensorAssetStore
from megatoken.omni.realizers import FacetCondition, Realizer, masked_square_error
from megatoken.teacher import HIDDEN_WIDTH, PINNED_TEACHER

NAMES = {"audio": "pocket", "image": "sana", "video": "sana_video"}
# Latent geometry per event: audio [frames, 4], image [4, 2, 2], video [frames, 4, 2, 2].
LATENT = {"audio": (6, 4), "image": (4, 2, 2), "video": (3, 4, 2, 2)}


def features(tokens: int = 12, *, seed: int = 0, device: str = "cpu") -> Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(1, tokens, HIDDEN_WIDTH, generator=generator).to(device)


class FakeRealizer(Realizer):
    _donor = "donor"

    def __init__(self, modality: str, facet_count: int, facet_width: int):
        super().__init__()
        self.name, self.modality = NAMES[modality], modality
        self._hold(donor=nn.Linear(8, int(np.prod(LATENT[modality]))))
        self.condition = FacetCondition(facet_count, facet_width, 8)
        self.checkpoint_identity = hashlib.sha256(modality.encode()).hexdigest()
        self.sample_rate = 24000

    def _latent(self, raw: Tensor) -> Tensor:
        value = self.donor(self.condition(raw).mean(dim=1).to(self.donor.weight.dtype))
        return value.view(raw.shape[0], *LATENT[self.modality])

    def encode(self, media: Tensor) -> Tensor:
        return torch.zeros(media.shape[0], *LATENT[self.modality])

    def loss(self, raw: Tensor, latents: Tensor, mask: Tensor | None, generator=None) -> Tensor:
        return masked_square_error(self._latent(raw), latents, mask)

    def realize(self, raw: Tensor, spec, generator=None) -> Tensor:
        level = self._latent(raw).float().mean(dim=tuple(range(1, 1 + len(LATENT[self.modality]))))
        if self.modality == "audio":
            return level[:, None].expand(-1, spec.sample_count).tanh()
        shape = (spec.frames, 3, spec.height, spec.width)
        return level.view(-1, 1, 1, 1, 1).expand(-1, *shape).sigmoid()


def install_fake_realizers(model, modalities=("audio", "image", "video")):
    for modality in modalities:
        model.media_realizers.register(
            FakeRealizer(modality, model.config.n_facets, model.config.facet_dim)
        )
    return model


def geometry(modality: str) -> OutputGeometry:
    if modality == "audio":
        return OutputGeometry("audio", sample_count=48, sample_rate=24000)
    if modality == "image":
        return OutputGeometry("image", height=16, width=16)
    return OutputGeometry("video", frames=4, height=16, width=16, fps_num=8)


def prepared_record(
    store: TensorAssetStore,
    modality: str,
    *,
    seed: int = 0,
    tokens: int = 12,
    target: bool = True,
    extractor: str = PINNED_TEACHER,
    valid_frames: int = 0,
) -> MediaRecord:
    """A complete prepared media event written into ``store``."""
    family = hashlib.sha256(f"{modality}:{seed}".encode()).hexdigest()
    source = store.put(
        features(tokens, seed=seed)[0].numpy(),
        modality=modality,
        kind="features",
        extractor=extractor,
        source_family=family,
    )
    embedding = teacher_embedding(store, modality, family, seed=seed, extractor=extractor)
    if not target:
        return MediaRecord(modality, source, embedding=embedding)
    rng = np.random.default_rng(seed)
    clock = {"audio": dict(sample_rate=24000), "video": dict(fps_num=8)}.get(modality, {})
    latents = store.put(
        rng.standard_normal(LATENT[modality]).astype(np.float32),
        modality=modality,
        kind="codec_latents",
        extractor=f"{NAMES[modality]}:{hashlib.sha256(modality.encode()).hexdigest()}",
        source_family=family,
        valid_count=valid_frames,
        **clock,
    )
    return MediaRecord(modality, source, latents, geometry(modality), embedding=embedding)


def teacher_embedding(store, modality, family, *, seed=0, extractor=PINNED_TEACHER):
    """A unit 768-d stand-in for the teacher's pooled embedding of the same input."""
    vector = np.random.default_rng(seed + 7).standard_normal(768).astype(np.float32)
    return store.put(
        vector / np.linalg.norm(vector),
        modality=modality,
        kind="embedding",
        extractor=extractor,
        source_family=family,
    )
