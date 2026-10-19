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
    codec_id: str = ""
    pack_id: str = ""
    latent_shape: tuple[int, ...] = ()
    sample_rate: int = AUDIO_SAMPLE_RATE
    fps_num: int = 0
    fps_den: int = 1
    extent_source: str = "requested"  # requested/predicted/teacher_only

    def __post_init__(self) -> None:
        if self.extent_source not in {"requested", "predicted", "teacher_only"}:
            raise ValueError("Structural extent provenance is required")
        if bool(self.codec_id) != bool(self.pack_id):
            raise ValueError("Codec and generator pack must be selected together")
        if self.codec_id and (not self.latent_shape or any(n < 1 for n in self.latent_shape)):
            raise ValueError("Native rendering requires nonempty pinned latent geometry")
        if self.modality == "audio":
            if self.sample_rate not in {16000, 24000} or not 1 <= self.sample_count <= self.sample_rate * 120:
                raise ValueError("Audio output requires supported rate and bounded duration")
        elif self.modality in {"image", "video"}:
            if (
                not 1 <= self.frames <= (193 if self.codec_id else 32)
                or self.modality == "image" and self.frames != 1
                or not 8 <= self.height <= (1280 if self.codec_id else 256)
                or not 8 <= self.width <= (1280 if self.codec_id else 256)
            ):
                raise ValueError("Visual output exceeds the native baseline's bounds")
        else:
            raise ValueError("Media output must be audio, image or video")
        if self.modality == "video" and self.codec_id and (
            self.fps_num <= 0 or self.fps_den <= 0
        ):
            raise ValueError("Native video output requires its display frame clock")


@dataclass(frozen=True)
class MediaTrainTarget:
    """A resolved event, independent of the training dataset storage layout."""

    modality: str
    spec: MediaOutputSpec
    source: Tensor
    target: Tensor
    mode: str
    role_id: int = 2
    source_kind: str = "raw"
    extractor: str = ""
    target_kind: str = "raw"
    target_mask: Tensor | None = None
    views: tuple[Tensor, ...] = ()
    view_kinds: tuple[str, ...] = ()
    view_extractors: tuple[str, ...] = ()
    source_masks: tuple[Tensor, ...] = ()
    source_sample_rate: int = AUDIO_SAMPLE_RATE

    def __post_init__(self) -> None:
        if self.mode not in {"bootstrap", "recurrent"}:
            raise ValueError("Media training needs bootstrap or recurrent mode")
        if self.spec.modality != self.modality:
            raise ValueError("Media target modality disagrees with output specification")
        expected = (
            (1, *self.spec.latent_shape) if self.target_kind == "codec_latents" else
            ((1, self.spec.sample_count) if self.modality == "audio" else
             (1, self.spec.frames, 3, self.spec.height, self.spec.width))
        )
        if (self.target_kind == "codec_latents") != bool(self.spec.codec_id):
            raise ValueError("Codec target and media output spec do not agree")
        if self.target_mask is not None and (not torch.broadcast_shapes(
                self.target_mask.shape, self.target.shape) == self.target.shape
                or not self.target_mask.dtype == torch.bool):
            raise ValueError("Codec target valid mask must broadcast to its coordinates")
        if len(self.views) != len(self.view_kinds) or len(self.views) != len(self.view_extractors):
            raise ValueError("Every external view needs a kind and extractor identity")
        if self.source_masks and len(self.source_masks) != len(self.views) + 1:
            raise ValueError("Source view masks must cover source plus every extra view")
        if self.target.shape != expected or not self.target.is_floating_point():
            raise ValueError("Media target tensor differs from requested realization geometry")
        if self.source.shape[0] != 1 or not self.source.is_floating_point():
            raise ValueError("Media source tensor must have one batch row")
        if self.source_kind not in {"raw", "features", "codec_latents"} or (
            self.source_kind == "features" and not self.extractor
        ) or (self.source_kind == "raw" and self.extractor):
            raise ValueError("Media source kind and extractor identity disagree")
        if not bool(torch.isfinite(self.source).all()) or not bool(
            torch.isfinite(self.target).all()
        ) or any(not bool(torch.isfinite(view).all()) for view in self.views):
            raise ValueError("Media source and target tensors must be finite")


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
        self.external_audio = (
            nn.Linear(config.audio_feature_dim, config.source_dim, bias=False)
            if config.audio_feature_dim else None
        )
        self.external_vision = (
            nn.Linear(config.vision_feature_dim, config.source_dim, bias=False)
            if config.vision_feature_dim else None
        )

    def encode(self, core, modality: str, value: Tensor, *, sample_rate: int = AUDIO_SAMPLE_RATE):
        if modality == "audio":
            features = self.audio(value, sample_rate=sample_rate)
        elif modality in {"image", "video"}:
            features = self.vision(value)
        else:
            raise ValueError("Media input must be audio, image or video")
        return self.encode_ordered_features(core, features)

    def encode_external(
        self,
        core,
        modality: str,
        tokens: Tensor,
        *,
        extractor: str,
    ) -> Tensor:
        """Learn only a small projection of pinned frozen donor features.

        Donor inference and caching happen before training. No donor backbone
        or private donor KV is registered in the recurrent Megatoken model.
        """
        if modality == "audio":
            adapter, identity = self.external_audio, core.config.audio_feature_id
        elif modality in {"image", "video"}:
            adapter, identity = self.external_vision, core.config.vision_feature_id
        else:
            raise ValueError("External feature input requires audio, image or video")
        if adapter is None or identity != extractor:
            raise ValueError("External encoder fingerprint/feature width differs from model")
        if (
            tokens.ndim != 3 or tokens.shape[-1] != adapter.in_features
            or tokens.shape[1] < 1 or tokens.device != core.initial_facets.device
            or not tokens.is_floating_point()
        ):
            raise ValueError("External media tokens need device-local [batch, tokens, feature_dim]")
        if not bool(torch.isfinite(tokens).all()):
            raise ValueError("External feature values must be finite")
        projected = adapter(tokens.to(adapter.weight.dtype))
        return self.encode_ordered_features(core, projected)

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
        self.fuse = nn.Linear(d, d)
        self.control = nn.Linear(config.control_dim, d, bias=False)
        self.audio_expand = nn.Linear(d, 64)
        self.audio_filter = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=9, padding=4),
            nn.SiLU(),
            nn.Conv1d(16, 1, kernel_size=9, padding=4),
        )
        self.visual_features = nn.Linear(d, hidden)
        self.visual_positions = nn.Parameter(torch.randn(64, hidden) * 0.01)
        # Initialized explicitly from pinned pretrained pack manifests before the
        # optimizer owns adapters. No text, source latent or control bypass.
        self.native = nn.ModuleDict()
        self.visual_filter = nn.Sequential(
            nn.Conv3d(hidden, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv3d(hidden, 3, kernel_size=1),
        )

    def _features(self, context: RealizationContext, control: Tensor | None) -> Tensor:
        raw = context.facets
        features = self.fuse(raw.to(self.fuse.weight.dtype))
        if control is not None:
            raise ValueError("v8 realizer cannot read a control vector; select raw Z with P")
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

    def install_native(self, modality: str, realizer: nn.Module) -> None:
        if modality not in {"audio", "image", "video"} or modality in self.native:
            raise ValueError("One native pack owner is permitted per modality")
        self.native[modality] = realizer

    def loss_native(self, context: RealizationContext, target: MediaTrainTarget) -> Tensor:
        if target.modality not in self.native:
            raise ValueError("Native codec target requires the matching pretrained pack")
        realizer = self.native[target.modality]
        if realizer.codec_id != target.spec.codec_id or realizer.pack_id != target.spec.pack_id:
            raise ValueError("Mismatched native codec and generator weights")
        return realizer.flow_loss(context.facets, target.target, target.target_mask)

    def realize(
        self,
        context: RealizationContext,
        spec: MediaOutputSpec,
        control: Tensor | None = None,
        noise: Tensor | None = None,
    ) -> Tensor:
        if control is not None:
            raise ValueError("Native realization reads raw selected facets, not prompt controls")
        if spec.codec_id:
            if spec.modality not in self.native:
                raise ValueError("Missing pinned native generator/codec pack")
            pack = self.native[spec.modality]
            if pack.codec_id != spec.codec_id or pack.pack_id != spec.pack_id:
                raise ValueError("Requested codec/generator is incompatible with loaded pack")
            return pack.sample(context.facets, spec, noise=noise)
        if spec.modality == "audio":
            return self.audio(context, spec)
        return self.visual(context, spec)
