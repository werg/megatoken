"""Media input encoding and realization around the canonical 32-facet megatoken.

Input: pinned EmbeddingGemma2 hidden states -> one projection -> the shared E
hierarchy (``MediaEncoder``). Output: one resident pretrained realizer per
modality, conditioned on the selected raw facets (``MediaRealizers``; see
``megatoken.omni.realizers``). Media samples and pixels never become core tokens.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from megatoken.teacher import HIDDEN_WIDTH

from .attention import AttentionBranch
from .config import ModelConfig
from .layers import RMSNorm, SwiGLU
from .state import RealizationContext

# The one realizer per modality.
REALIZERS = {"audio": "pocket", "image": "sana", "video": "sana_video"}
PROVENANCE = ("requested", "predicted", "teacher_only")


@dataclass(frozen=True)
class MediaOutputSpec:
    """Physical output bounds for one realized event, independent of its content.

    ``provenance`` records where the extent came from; a ``teacher_only``
    extent (read off a gold target) may supervise training but never serve.
    """

    modality: str
    sample_count: int = 0
    sample_rate: int = 0
    frames: int = 1
    height: int = 0
    width: int = 0
    fps_num: int = 0
    fps_den: int = 1
    steps: int = 30
    provenance: str = "requested"

    def __post_init__(self) -> None:
        if self.provenance not in PROVENANCE:
            raise ValueError("Undeclared output-extent provenance")
        if not 1 <= self.steps <= 200 or self.fps_den < 1:
            raise ValueError("Invalid sampler steps or frame clock")
        if self.modality == "audio":
            if self.sample_rate != 24000 or not 1 <= self.sample_count <= 24000 * 120:
                raise ValueError("Speech output is 24 kHz, at most 120 s")
            if self.height or self.width or self.fps_num or self.frames != 1:
                raise ValueError("Audio output cannot carry visual geometry")
        elif self.modality in ("image", "video"):
            if not (8 <= self.height <= 2048 and 8 <= self.width <= 2048):
                raise ValueError("Visual output needs 8..2048 pixel height and width")
            if self.sample_count or self.sample_rate:
                raise ValueError("Visual output cannot carry an audio clock")
            if self.modality == "image" and (self.frames != 1 or self.fps_num):
                raise ValueError("A still image has one frame and no frame clock")
            if self.modality == "video" and (not 1 <= self.frames <= 193 or self.fps_num <= 0):
                raise ValueError("Video output needs 1..193 frames and its frame clock")
        else:
            raise ValueError("Media output must be audio, image or video")

    @classmethod
    def from_geometry(cls, geometry, *, steps: int = 30) -> MediaOutputSpec:
        """The spec for a prepared record's ``OutputGeometry``."""
        return cls(
            geometry.modality,
            sample_count=geometry.sample_count,
            sample_rate=geometry.sample_rate,
            frames=geometry.frames,
            height=geometry.height,
            width=geometry.width,
            fps_num=geometry.fps_num,
            fps_den=geometry.fps_den,
            steps=steps,
            provenance=geometry.provenance,
        )


def validate_public_media_spec(spec: MediaOutputSpec) -> None:
    """A free P/R action cannot borrow a future target's extent."""
    if spec.provenance == "teacher_only":
        raise ValueError("Teacher-only output geometry is not available to free generation")


@dataclass(frozen=True)
class MediaTrainTarget:
    """A resolved media event: observed teacher features and realizer latent target."""

    modality: str
    spec: MediaOutputSpec
    source: Tensor
    extractor: str
    target: Tensor
    target_mask: Tensor | None
    realizer: str
    mode: str
    role_id: int = 2

    def __post_init__(self) -> None:
        if self.mode not in {"bootstrap", "recurrent"}:
            raise ValueError("Media training needs bootstrap or recurrent mode")
        if self.spec.modality != self.modality:
            raise ValueError("Media target modality disagrees with its output specification")
        if (
            self.source.ndim != 3
            or self.source.shape[0] != 1
            or self.source.shape[2] != HIDDEN_WIDTH
        ):
            raise ValueError("Media source must be one row of [1, tokens, 512] teacher features")
        if self.target.shape[0] != 1 or not self.target.is_floating_point():
            raise ValueError("Media target must be one row of realizer latents")
        if self.target_mask is not None and (
            self.target_mask.dtype != torch.bool or not bool(self.target_mask.any())
        ):
            raise ValueError("A latent validity mask must be boolean and nonempty")


class MediaEncoder(nn.Module):
    """The only media input: pinned teacher hidden states -> the shared E hierarchy.

    Audio, image and video arrive as ordered frozen EmbeddingGemma2 states
    ``[B, T, 512]`` (cached offline; see ``megatoken.teacher``). One learned
    projection maps them to ``source_dim``; 32 learned queries read every
    teacher token at once, then E's shared facet refinement produces the
    event's raw facets. Modality reaches C/P through the modality embedding,
    not through separate stems.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.extractor = config.media_extractor
        self.projection = nn.Linear(HIDDEN_WIDTH, config.source_dim, bias=False)
        self.norm = RMSNorm(config.source_dim, config.norm_eps)
        self.queries = nn.Parameter(torch.randn(config.n_facets, config.facet_dim) * 0.02)
        self.attention = AttentionBranch(
            config.facet_dim,
            config.source_dim,
            config.query_heads,
            config.kv_heads,
            config.attention_head_dim,
            config.norm_eps,
        )
        self.ffn_norm = RMSNorm(config.facet_dim, config.norm_eps)
        self.ffn = SwiGLU(config.facet_dim, 2 * config.facet_dim)

    def forward(
        self, core, features: Tensor, mask: Tensor | None = None, *, extractor: str
    ) -> Tensor:
        if extractor != self.extractor:
            raise ValueError("Media features come from a different teacher than this model's")
        if (
            features.ndim != 3
            or features.shape[1] < 1
            or features.shape[2] != HIDDEN_WIDTH
            or not features.is_floating_point()
            or features.device != core.initial_facets.device
        ):
            raise ValueError("Media features must be device-local [batch, tokens, 512]")
        if mask is None:
            mask = torch.ones(features.shape[:2], dtype=torch.bool, device=features.device)
        elif (
            mask.shape != features.shape[:2]
            or mask.dtype != torch.bool
            or not bool(mask.any(dim=1).all())
        ):
            raise ValueError("Media mask must mark at least one real token per row")
        projected = self.norm(self.projection(features.to(self.projection.weight.dtype)))
        projected = projected.masked_fill(~mask[..., None], 0)
        # One read of every teacher token by 32 learned queries, then E's
        # shared facet refinement.
        queries = self.queries.to(projected.dtype)[None].expand(projected.shape[0], -1, -1)
        facets = queries + self.attention(queries, projected, mask)
        facets = facets + self.ffn(self.ffn_norm(facets))
        return core.frontend._finish(facets)


class MediaRealizers(nn.Module):
    """Exactly one resident pretrained realizer per modality, conditioned on raw Z."""

    def __init__(self) -> None:
        super().__init__()
        self.packs = nn.ModuleDict()

    def register(self, realizer: nn.Module) -> None:
        if REALIZERS.get(realizer.modality) != realizer.name or realizer.name in self.packs:
            raise ValueError("One locked realizer is permitted per modality")
        self.packs[realizer.name] = realizer

    def resident(self, modality: str) -> nn.Module:
        name = REALIZERS[modality]
        if name not in self.packs:
            raise ValueError(f"The {name} realizer for {modality} is not installed")
        return self.packs[name]

    def loss(self, context: RealizationContext, target: MediaTrainTarget) -> Tensor:
        """The realizer's own objective, as one per-event mean over real latents."""
        realizer = self.resident(target.modality)
        if realizer.identity != target.realizer:
            raise ValueError("Target latents were encoded by a different realizer pack")
        return realizer.loss(context.facets, target.target, target.target_mask)

    def realize(
        self,
        context: RealizationContext,
        spec: MediaOutputSpec,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Audio [B, samples] at 24 kHz; image/video [B, frames, 3, H, W] in [0, 1]."""
        return self.resident(spec.modality).realize(context.facets, spec, generator=generator)
