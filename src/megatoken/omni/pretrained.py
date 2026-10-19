"""Checked pretrained encoder/renderer boundary for independent donor packs.

External donor frameworks are injected explicitly; unknown checkpoint APIs are
never silently treated as compatible. All native generator inputs are raw Z.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import torch
from megatoken.omni.packs import PackRegistry

@dataclass(frozen=True)
class FrozenFeatureView:
    name: str
    model_id: str
    feature_width: int
    sample_rate: int = 0
    time_origin: float = 0.
    def validate(self, ordered, mask):
        if ordered.ndim != 3 or ordered.shape[-1] != self.feature_width:
            raise ValueError("Donor feature width or rank mismatch")
        if mask.shape != ordered.shape[:2] or mask.dtype != torch.bool:
            raise ValueError("Ordered frozen features require explicit valid mask")
        if not torch.isfinite(ordered).all():
            raise ValueError("Nonfinite external feature stream")
        return ordered * mask.unsqueeze(-1)

class FrozenEncoder:
    def __init__(self, name: str, view: FrozenFeatureView, encode: Callable):
        self.name, self.view, self.encode_fn = name, view, encode
    @torch.no_grad()
    def __call__(self, source, *, available_length=None):
        if available_length is not None:
            if available_length < 0 or available_length > source.shape[-1]:
                raise ValueError("Invalid source prefix boundary")
            source = source[..., :available_length]
        embeddings, mask = self.encode_fn(source)
        return self.view.validate(embeddings, mask)

class FacetsOnlyRenderer:
    """Donor adapter sees 32 raw facets, a nonsemantic structural spec, and RNG."""
    def __init__(self, packs: PackRegistry, generator_name: str, codec_name: str,
                 predict_latents: Callable, decode_latents: Callable, facet_width=512):
        packs.verify_pair(generator_name, codec_name)
        self.generator_name, self.codec_name = generator_name, codec_name
        self.predict_latents, self.decode_latents = predict_latents, decode_latents
        self.facet_width = facet_width
    def latent(self, raw_facets, output_spec, noise=None):
        if raw_facets.ndim != 3 or raw_facets.shape[1:] != (32, self.facet_width):
            raise ValueError("Realizer receives exactly one raw [32,D] event")
        return self.predict_latents(raw_facets, output_spec, noise)
    def realize(self, raw_facets, output_spec, noise=None):
        return self.decode_latents(self.latent(raw_facets, output_spec, noise))

SELECTED_PACK_ROLES = {
    "speech_semantic": "whistle",
    "speech_codec": "pocket_mimi_continuous",
    "speech_generator": "pocket_tts",
    "visual_encoder": "minicpm_v_4_7_tower",
    "image_generator": "sana_600m",
    "image_codec": "sana_dc_ae",
    "video_generator": "sana_video_2b",
    "video_codec": "ltx2_video_vae",
}
