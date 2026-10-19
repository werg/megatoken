"""Semantic media adapters and learnable realizers for canonical megatokens.

This is a native, dependency-light neural baseline, not a pretrained speech
vocoder or a production image/video diffusion decoder. All modalities produce
the same [batch, facets, width] input to the shared central processor.
Audio waveform samples and visual pixels never become additional core tokens.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import ModelConfig
from .layers import RMSNorm
from .state import RealizationContext

AUDIO_SAMPLE_RATE = 16000


@dataclass(frozen=True)
class MediaOutputSpec:
    """Bounded physical rendering parameters, independent of semantic chunks."""

    modality: str
    sample_count: int = 0
    frames: int = 1
    height: int = 64
    width: int = 64

    def __post_init__(self) -> None:
        if self.modality == "audio":
            if not 1 <= self.sample_count <= AUDIO_SAMPLE_RATE * 60:
                raise ValueError("Audio output must contain 1 to 60 seconds at 16 kHz")
        elif self.modality in {"image", "video"}:
            if (
                not 1 <= self.frames <= 32
                or self.modality == "image" and self.frames != 1
                or not 8 <= self.height <= 256
                or not 8 <= self.width <= 256
            ):
                raise ValueError("Visual output exceeds the native baseline's bounds")
        else:
            raise ValueError("Media output must be audio, image or video")


class AudioStem(nn.Module):
    """Causal-origin waveform -> ordered spectral features, local to a chunk."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.projection = nn.Conv1d(201, config.source_dim, kernel_size=5, stride=2, padding=2)
        self.norm = RMSNorm(config.source_dim, config.norm_eps)

    def forward(self, waveform: Tensor, sample_rate: int = AUDIO_SAMPLE_RATE) -> Tensor:
        if sample_rate != AUDIO_SAMPLE_RATE:
            raise ValueError("Resample externally to the pinned 16 kHz audio sample rate")
        if waveform.ndim != 2 or waveform.shape[1] < 1:
            raise ValueError("Audio observation requires [batch, nonempty mono samples]")
        if not waveform.is_floating_point() or not torch.isfinite(waveform).all():
            raise ValueError("Waveform must contain finite floating-point audio samples")
        samples = F.pad(waveform, (0, max(0, 400 - waveform.shape[1])))
        window = torch.hann_window(400, device=samples.device, dtype=samples.dtype)
        spectrum = torch.stft(
            samples.float(), n_fft=400, hop_length=160, win_length=400,
            window=window.float(), center=False, return_complex=True,
        )
        spectrum_features = torch.log1p(spectrum.abs()).to(self.projection.weight.dtype)
        features = self.projection(spectrum_features).transpose(1, 2)
        if features.shape[1] > 256:
            features = F.adaptive_avg_pool1d(features.transpose(1, 2), 256).transpose(1, 2)
        return self.norm(features)


class VisualStem(nn.Module):
    """Image and video share one stem; an image is a single-frame clip."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.projection = nn.Conv3d(
            3, config.source_dim, kernel_size=3, stride=(1, 4, 4), padding=1
        )
        self.norm = RMSNorm(config.source_dim, config.norm_eps)

    def forward(self, frames: Tensor) -> Tensor:
        if frames.ndim == 4:
            frames = frames[:, None]
        if frames.ndim != 5 or frames.shape[2] != 3:
            raise ValueError("Visual observation must be [batch, frames, RGB, H, W]")
        if not frames.is_floating_point() or not torch.isfinite(frames).all():
            raise ValueError("Visual observation requires finite floating-point pixels")
        if min(frames.shape[1], frames.shape[3], frames.shape[4]) < 1:
            raise ValueError("Visual observation cannot have empty axes")
        # Local temporal/spatial reduction is independent of core meridian rate.
        source = frames.transpose(1, 2).contiguous()
        source = F.adaptive_avg_pool3d(
            source,
            (min(16, frames.shape[1]), min(64, frames.shape[3]), min(64, frames.shape[4])),
        )
        features = self.projection(source.to(self.projection.weight.dtype))
        # Time-major spatial flatten preserves local causal event order.
        flat = features.permute(0, 2, 3, 4, 1).flatten(1, 3)
        return self.norm(flat)


class MediaStems(nn.Module):
    """Only the modality-specific feature stems; final E hierarchy is shared."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.audio = AudioStem(config)
        self.vision = VisualStem(config)

    def encode(self, core, modality: str, value: Tensor, *, sample_rate: int = AUDIO_SAMPLE_RATE):
        if modality == "audio":
            features = self.audio(value, sample_rate=sample_rate)
        elif modality in {"image", "video"}:
            features = self.vision(value)
        else:
            raise ValueError("Media input must be audio, image or video")
        return self.encode_ordered_features(core, features)

    @staticmethod
    def encode_ordered_features(
        core,
        features: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        """Bridge frozen/precomputed media features to the canonical E hierarchy.

        A pinned external feature extractor and its trainable projection may run
        outside this module, but must output ordered source_dim features.  This
        shared path owns no additional encoder/decoder or historical KV weights.
        Feature extractor identity and preprocessing must be pinned by the
        calling data/serving pipeline, never inferred from tensor contents.
        """
        if (
            features.ndim != 3
            or features.shape[1] < 1
            or features.shape[2] != core.config.source_dim
        ):
            raise ValueError("Media source features must be [batch, tokens, source_dim]")
        if features.device != core.initial_facets.device or not features.is_floating_point():
            raise ValueError("Media features must be floating point on the model device")
        if not bool(torch.isfinite(features).all()):
            raise ValueError("Media features must be finite")
        if mask is None:
            mask = torch.ones(features.shape[:2], dtype=torch.bool, device=features.device)
        elif (
            mask.shape != features.shape[:2]
            or mask.dtype != torch.bool
            or mask.device != features.device
            or not bool(mask.any(dim=1).all())
        ):
            raise ValueError("Media source mask must mark at least one token per batch row")
        features = features.masked_fill(~mask[..., None], 0)
        level = 0
        # Reuse the same E reduction and facet readout weights as text.
        while features.shape[1] > core.config.n_facets:
            level += 1
            features, mask = core.frontend.reduction(features, mask, level=level)
        return core.frontend._finish(core.frontend._readout(features, mask))


class MediaRealizers(nn.Module):
    """One conditioning interface, separate audio and shared image/video edges.

    Raw audio and pixels are produced with small trainable native decoders so the
    data/training interfaces can be exercised without pretrained codec assets.
    Quality is an empirical training question; higher-fidelity codecs may later
    replace the physical output heads without changing E/C/P geometry.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        d = config.facet_dim
        hidden = min(64, d)
        self.fuse = nn.Linear(3 * d, d)
        self.control = nn.Linear(config.control_dim, d, bias=False)
        self.audio_expand = nn.Linear(d, 64)
        self.audio_filter = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=9, padding=4),
            nn.SiLU(),
            nn.Conv1d(16, 1, kernel_size=9, padding=4),
        )
        self.visual_features = nn.Linear(d, hidden)
        self.visual_positions = nn.Parameter(torch.randn(64, hidden) * 0.01)
        self.visual_filter = nn.Sequential(
            nn.Conv3d(hidden, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv3d(hidden, 3, kernel_size=1),
        )

    def _features(self, context: RealizationContext, control: Tensor | None) -> Tensor:
        raw = context.facets
        early = raw if context.early is None else context.early
        final = raw if context.final is None else context.final
        features = self.fuse(torch.cat((raw, early, final), dim=-1))
        if control is not None:
            if control.shape != (raw.shape[0], self.control.in_features):
                raise ValueError("Realizer control must be one shared control vector per event")
            features = features + self.control(control.to(features.dtype))[:, None, :]
        return features

    def audio(
        self,
        context: RealizationContext,
        spec: MediaOutputSpec,
        control: Tensor | None = None,
    ) -> Tensor:
        if spec.modality != "audio":
            raise ValueError("Expected an audio rendering spec")
        source = self._features(context, control)
        signal = self.audio_expand(source).flatten(1)[:, None]
        signal = F.interpolate(signal, size=spec.sample_count, mode="linear", align_corners=False)
        return self.audio_filter(signal).tanh()[:, 0]

    def visual(
        self,
        context: RealizationContext,
        spec: MediaOutputSpec,
        control: Tensor | None = None,
    ) -> Tensor:
        if spec.modality not in {"image", "video"}:
            raise ValueError("Expected an image or video rendering spec")
        source = self._features(context, control)
        # Up to 32 distinct temporal slices are drawn from all 32 facets.
        timeline = F.interpolate(
            self.visual_features(source).transpose(1, 2), size=spec.frames, mode="linear",
            align_corners=False,
        )
        positions = self.visual_positions.transpose(0, 1).reshape(1, -1, 1, 8, 8)
        coarse = timeline[..., None, None] + positions
        rendered = self.visual_filter(coarse).sigmoid()
        rendered = F.interpolate(
            rendered, size=(spec.frames, spec.height, spec.width), mode="trilinear",
            align_corners=False,
        )
        # [batch, frames, RGB, height, width], same shape for image (one frame).
        return rendered.transpose(1, 2).contiguous()

    def realize(
        self,
        context: RealizationContext,
        spec: MediaOutputSpec,
        control: Tensor | None = None,
    ) -> Tensor:
        if spec.modality == "audio":
            return self.audio(context, spec, control)
        return self.visual(context, spec, control)
