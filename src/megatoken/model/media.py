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
    # Native means legacy PCM/pixel debug path; production means matched codec.
    renderer: str = "native"
    sample_rate: int = AUDIO_SAMPLE_RATE
    fps_num: int = 0
    fps_den: int = 1
    latent_shape: tuple[int, ...] = ()
    shape_provenance: str = "requested"
    steps: int = 30
    # Alternative pinned generator/codec contract from concordant training.
    codec_id: str = ""
    pack_id: str = ""
    extent_source: str = "requested"

    def __post_init__(self) -> None:
        if self.renderer not in ("native", "pocket", "sana", "sana_video"):
            raise ValueError("Unknown media realizer identity")
        if self.shape_provenance not in ("requested", "predicted", "teacher_only"):
            raise ValueError("Undeclared target-geometry provenance")
        if self.extent_source not in ("requested", "predicted", "teacher_only"):
            raise ValueError("Structural extent provenance is required")
        if bool(self.codec_id) != bool(self.pack_id):
            raise ValueError("Codec and generator pack must be selected together")
        if self.codec_id and not self.latent_shape:
            raise ValueError("Native rendering requires pinned latent geometry")
        if self.sample_rate < 1 or self.steps < 1 or self.fps_den < 1:
            raise ValueError("Invalid sample/renderer clock")
        if (
            self.renderer == "pocket"
            and self.modality != "audio"
            or (self.renderer == "sana" and self.modality != "image")
            or (self.renderer == "sana_video" and self.modality != "video")
        ):
            raise ValueError("Renderer is incompatible with media modality")
        if self.codec_id and self.renderer != "native":
            raise ValueError("Select a pretrained renderer OR a pinned native codec pack")
        if self.latent_shape and any(type(n) is not int or n < 1 for n in self.latent_shape):
            raise ValueError("Invalid native latent target dimensions")
        if self.modality == "audio":
            limit = 120 if self.codec_id else 60
            if self.codec_id and self.sample_rate not in (16000, 24000):
                raise ValueError("Native speech needs a supported sample rate")
            if not 1 <= self.sample_count <= self.sample_rate * limit:
                raise ValueError("Audio output exceeds the declared renderer duration limit")
        elif self.modality in {"image", "video"}:
            if (
                not 1
                <= self.frames
                <= (32 if self.renderer == "native" and not self.codec_id else 193)
                or self.modality == "image"
                and self.frames != 1
                or not 8
                <= self.height
                <= (256 if self.renderer == "native" and not self.codec_id else 2048)
                or not 8
                <= self.width
                <= (256 if self.renderer == "native" and not self.codec_id else 2048)
            ):
                raise ValueError("Visual output exceeds the native baseline's bounds")
        else:
            raise ValueError("Media output must be audio, image or video")
        if self.modality == "video" and self.codec_id and self.fps_num <= 0:
            raise ValueError("Native video output requires its display frame clock")


def validate_public_media_spec(spec: MediaOutputSpec) -> None:
    """A free P/R action cannot borrow future target duration or latent shape."""
    if spec.shape_provenance == "teacher_only" or spec.extent_source == "teacher_only":
        raise ValueError("Teacher-only output geometry is not available to free generation")
    if spec.latent_shape and not spec.codec_id:
        raise ValueError("Public media generation cannot read target codec geometry")
    if spec.modality == "audio" and spec.renderer == "pocket" and spec.sample_rate != 24000:
        raise ValueError("Pocket must render its matching 24 kHz codec")


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
    source_views: tuple[tuple[str, Tensor, Tensor], ...] = ()
    target_valid_mask: Tensor | None = None
    # Prepared explicit masks and typed source views for native-pack training.
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
            (1, self.spec.sample_count)
            if self.modality == "audio"
            else (1, self.spec.frames, 3, self.spec.height, self.spec.width)
        )
        if self.target_kind not in ("raw", "codec_latents"):
            raise ValueError("Media targets must be raw or native codec latents")
        if self.target_kind == "codec_latents":
            if (
                self.spec.renderer == "native" and not self.spec.codec_id
            ) or not self.spec.latent_shape:
                raise ValueError("Codec targets require an explicit pretrained latent spec")
            expected = (1, *self.spec.latent_shape)
        if self.target.shape != expected or not self.target.is_floating_point():
            raise ValueError("Media target tensor differs from requested realization geometry")
        if self.target_valid_mask is not None:
            try:
                torch.broadcast_shapes(self.target_valid_mask.shape, self.target.shape)
            except RuntimeError as exc:
                raise ValueError("Codec validity mask must broadcast over target") from exc
            if self.target_valid_mask.dtype != torch.bool or not bool(self.target_valid_mask.any()):
                raise ValueError("Codec target mask must be nonempty boolean")
        if self.target_mask is not None:
            try:
                torch.broadcast_shapes(self.target_mask.shape, self.target.shape)
            except RuntimeError as exc:
                raise ValueError("Codec validity mask must broadcast over target") from exc
            if self.target_mask.dtype != torch.bool or not bool(self.target_mask.any()):
                raise ValueError("Codec target mask must be nonempty boolean")
        if len(self.views) != len(self.view_kinds) or len(self.views) != len(self.view_extractors):
            raise ValueError("Every external view needs a kind and extractor identity")
        if any(not torch.isfinite(view).all() for view in self.views):
            raise ValueError("Prepared source views must be finite")
        if self.source_masks and len(self.source_masks) != len(self.views) + 1:
            raise ValueError("Source view masks must cover source plus every extra view")
        if self.source.shape[0] != 1 or not self.source.is_floating_point():
            raise ValueError("Media source tensor must have one batch row")
        if (
            self.source_kind not in {"raw", "features", "codec_latents"}
            or (self.source_kind == "features" and not self.extractor)
            or (self.source_kind == "raw" and self.extractor)
        ):
            raise ValueError("Media source kind and extractor identity disagree")
        for identity, tokens, mask in self.source_views:
            if (
                not identity
                or tokens.ndim != 3
                or tokens.shape[0] != 1
                or mask.shape != tokens.shape[:2]
                or mask.dtype != torch.bool
                or not bool(mask.any())
                or not bool(torch.isfinite(tokens).all())
            ):
                raise ValueError("Auxiliary input view must be pinned, finite and masked")
        if not bool(torch.isfinite(self.source).all()) or not bool(
            torch.isfinite(self.target).all()
        ):
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
            samples.float(),
            n_fft=400,
            hop_length=160,
            win_length=400,
            window=window.float(),
            center=False,
            return_complex=True,
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
        self.view_dropout = config.media_view_dropout
        self.unified_projection = (
            nn.Linear(config.embeddinggemma2_hidden_dim, config.source_dim, bias=False)
            if config.embeddinggemma2_hidden_dim else None
        )
        self.external_audio = (
            nn.Linear(config.audio_feature_dim, config.source_dim, bias=False)
            if config.audio_feature_dim
            else None
        )
        self.external_vision = (
            nn.Linear(config.vision_feature_dim, config.source_dim, bias=False)
            if config.vision_feature_dim
            else None
        )
        self.acoustic = (
            nn.Linear(config.audio_acoustic_feature_dim, config.source_dim, bias=False)
            if config.audio_acoustic_feature_dim
            else None
        )
        self.detail = (
            nn.Linear(config.vision_detail_feature_dim, config.source_dim, bias=False)
            if config.vision_detail_feature_dim
            else None
        )
        self.visual_codec = (
            nn.Linear(config.vision_codec_feature_dim, config.source_dim, bias=False)
            if config.vision_codec_feature_dim
            else None
        )
        self.missing_view = nn.Parameter(torch.zeros(2, config.source_dim))
        self.view_adapters = nn.ModuleDict(
            {
                identity: nn.Linear(width, config.source_dim, bias=False)
                for identity, width in config.media_feature_views.items()
            }
        )
        self.view_embeddings = nn.ParameterDict(
            {
                identity: nn.Parameter(torch.zeros(config.source_dim))
                for identity in config.media_feature_views
            }
        )

    def encode(self, core, modality: str, value: Tensor, *, sample_rate: int = AUDIO_SAMPLE_RATE):
        if self.unified_projection is not None:
            raise ValueError("Unified EmbeddingGemma2 mode requires pinned hidden-state features, not raw media")
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
        if self.unified_projection is not None:
            if modality not in {"audio", "image", "video", "document", "text_summary"}:
                raise ValueError("Unsupported unified teacher modality")
            adapter, identity = self.unified_projection, core.config.embeddinggemma2_hidden_id
        elif modality == "audio":
            adapter, identity = self.external_audio, core.config.audio_feature_id
        elif modality in {"image", "video"}:
            adapter, identity = self.external_vision, core.config.vision_feature_id
        else:
            raise ValueError("External feature input requires audio, image or video")
        if adapter is None or identity != extractor:
            raise ValueError("External encoder fingerprint/feature width differs from model")
        if (
            tokens.ndim != 3
            or tokens.shape[-1] != adapter.in_features
            or tokens.shape[1] < 1
            or tokens.device != core.initial_facets.device
            or not tokens.is_floating_point()
        ):
            raise ValueError("External media tokens need device-local [batch, tokens, feature_dim]")
        if not bool(torch.isfinite(tokens).all()):
            raise ValueError("External feature values must be finite")
        projected = adapter(tokens.to(adapter.weight.dtype))
        return self.encode_ordered_features(core, projected)

    def encode_views(
        self,
        core,
        modality: str,
        views: tuple[tuple[str, Tensor, Tensor], ...],
    ) -> Tensor:
        """Reduce all present declared views through ONE central E readout."""
        if not views:
            raise ValueError("At least one visible media view is required")
        if self.unified_projection is not None:
            if len(views) != 1:
                raise ValueError("Unified EmbeddingGemma2 input is one ordered hidden-state view")
            identity, tokens, valid = views[0]
            if identity != core.config.embeddinggemma2_hidden_id:
                raise ValueError("Unified teacher hidden-state identity mismatch")
            if valid.shape != tokens.shape[:2] or valid.dtype != torch.bool or not bool(valid.any(dim=1).all()):
                raise ValueError("Unified hidden-state mask must be nonempty and boolean")
            if tokens.device != core.initial_facets.device:
                raise ValueError("Unified teacher hidden states must be on the model device")
            features = self.unified_projection(tokens.to(self.unified_projection.weight.dtype))
            return self.encode_ordered_features(core, features, valid)
        projected, masks = [], []
        for identity, tokens, valid in views:
            if identity in self.view_adapters:
                adapter = self.view_adapters[identity]
                vector = self.view_embeddings[identity]
            elif modality == "audio" and identity == core.config.audio_feature_id:
                adapter, vector = self.external_audio, None
            elif modality in ("image", "video") and identity == core.config.vision_feature_id:
                adapter, vector = self.external_vision, None
            else:
                raise ValueError("Unpinned feature view or incorrect modality")
            if adapter is None or tokens.ndim != 3 or tokens.shape[-1] != adapter.in_features:
                raise ValueError("Feature view projection width mismatch")
            if (
                valid.shape != tokens.shape[:2]
                or valid.dtype != torch.bool
                or not bool(valid.any(dim=1).all())
            ):
                raise ValueError("Feature view needs an explicit nonempty valid mask")
            content = adapter(tokens.to(adapter.weight.dtype))
            if vector is not None:
                content = content + vector
            projected.append(content)
            masks.append(valid)
        if self.training and self.view_dropout and len(masks) > 1:
            # Missing modality views remain explicit masked slots. Never erase
            # *every* available source view or condition dropout on gold peers.
            keep = torch.rand(len(masks), device=masks[0].device) >= self.view_dropout
            if not bool(keep.any()):
                keep[0] = True
            masks = [
                mask if bool(keep[index]) else torch.zeros_like(mask)
                for index, mask in enumerate(masks)
            ]
        return self.encode_ordered_features(
            core,
            torch.cat(projected, dim=1),
            torch.cat(masks, dim=1),
        )

    def encode_multiview(
        self,
        core,
        modality: str,
        tensors: tuple[Tensor, ...],
        kinds: tuple[str, ...],
        extractors: tuple[str, ...],
        masks: tuple[Tensor | None, ...] = (),
        sample_rate: int = AUDIO_SAMPLE_RATE,
        view_dropout: float = 0.0,
    ) -> Tensor:
        """Fuse declared frozen views in E, never via current R or central KV.

        All source tokens retain their ordering until shared E reduction.
        Temporal windows and crop maps are pinned in the source asset metadata.
        """
        if not tensors or len(tensors) != len(kinds) or len(tensors) != len(extractors):
            raise ValueError("Every E view needs tensor/kind/extractor provenance")
        if self.unified_projection is not None:
            if len(tensors) != 1 or kinds[0] != "features":
                raise ValueError("Unified mode accepts exactly one cached EmbeddingGemma2 hidden state")
            mask = (masks[0] if masks else None)
            if mask is None:
                mask = torch.ones(tensors[0].shape[:2], dtype=torch.bool, device=tensors[0].device)
            return self.encode_views(core, modality, ((extractors[0], tensors[0], mask),))
        if masks and len(masks) != len(tensors):
            raise ValueError("Each E view must have its own valid-coordinate mask")
        projected, valid = [], []
        identities = (
            [
                (core.config.audio_feature_id, self.external_audio),
                (core.config.audio_acoustic_feature_id, self.acoustic),
            ]
            if modality == "audio"
            else [
                (core.config.vision_feature_id, self.external_vision),
                (core.config.vision_detail_feature_id, self.detail),
                (core.config.vision_codec_feature_id, self.visual_codec),
            ]
        )
        for i, (value, kind, extractor) in enumerate(zip(tensors, kinds, extractors, strict=True)):
            if kind == "raw":
                if extractor:
                    raise ValueError("Raw view cannot claim a frozen encoder")
                if modality == "audio":
                    features = self.audio(value, sample_rate=sample_rate)
                else:
                    features = self.vision(value)
            elif kind == "features":
                adapter = next(
                    (
                        layer
                        for ident, layer in identities
                        if ident and ident == extractor and layer is not None
                    ),
                    None,
                )
                if adapter is None or value.ndim != 3 or value.shape[-1] != adapter.in_features:
                    raise ValueError("External E view fingerprint or width mismatch")
                features = adapter(value.to(adapter.weight.dtype))
            else:
                raise ValueError("Native codec coordinates cannot bypass the E feature projector")
            mask = (
                torch.ones(features.shape[:2], dtype=torch.bool, device=features.device)
                if not masks or masks[i] is None
                else masks[i].to(device=features.device, dtype=torch.bool)
            )
            if mask.shape != features.shape[:2]:
                raise ValueError("E view mask must match projected token coordinates")
            if i and self.training and view_dropout > 0:
                keep = torch.rand(features.shape[0], device=features.device) >= view_dropout
                mask = mask & keep[:, None]
            features = features.masked_fill(~mask[..., None], 0)
            projected.append(features)
            valid.append(mask)
        if not bool(valid[0].any(dim=1).all()):
            raise ValueError("Primary observed view must contain real evidence")
        # Missing supplementary streams produce a learned missing-view token;
        # they are not misrepresented as zero-valued observations.
        row = 0 if modality == "audio" else 1
        missing = self.missing_view[row].to(projected[0].dtype)
        projected.append(missing[None, None].expand(projected[0].shape[0], 1, -1))
        valid.append(torch.ones_like(valid[0][:, :1]))
        return self.encode_ordered_features(
            core, torch.cat(projected, dim=1), torch.cat(valid, dim=1)
        )

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
        self.pretrained = nn.ModuleDict()
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
            raise ValueError("v8 realizer cannot read prompt controls after raw Z selection")
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
            self.visual_features(source).transpose(1, 2),
            size=spec.frames,
            mode="linear",
            align_corners=False,
        )
        positions = self.visual_positions.transpose(0, 1).reshape(1, -1, 1, 8, 8)
        coarse = timeline[..., None, None] + positions
        rendered = self.visual_filter(coarse).sigmoid()
        rendered = F.interpolate(
            rendered,
            size=(spec.frames, spec.height, spec.width),
            mode="trilinear",
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

    def register_pretrained(self, name: str, student: nn.Module) -> None:
        if name not in ("pocket", "sana", "sana_video") or name in self.pretrained:
            raise ValueError("Invalid or duplicate pretrained realization owner")
        self.pretrained[name] = student

    def training_loss(
        self, raw: Tensor, spec: MediaOutputSpec, target: Tensor, valid_mask: Tensor | None = None
    ) -> tuple[Tensor, int]:
        if spec.renderer not in self.pretrained:
            raise ValueError("Missing trained donor/codec for selected media spec")
        student = self.pretrained[spec.renderer]
        numerator, coordinate_count = student.native_loss(raw, target, valid_mask=valid_mask)
        if coordinate_count.item() == 0:
            raise ValueError("Empty codec latent target")
        # Production ledger counts media events; normalise native coordinates
        # inside each event to avoid video latents overpowering audio loss.
        return numerator / coordinate_count.clamp_min(1), 1

    def realize(
        self,
        context: RealizationContext,
        spec: MediaOutputSpec,
        control: Tensor | None = None,
        generator: torch.Generator | None = None,
        noise: Tensor | None = None,
        seed: int | None = None,
    ) -> Tensor:
        if spec.codec_id:
            if control is not None:
                raise ValueError(
                    "Native realization reads raw selected facets, not prompt controls"
                )
            if spec.modality not in self.native:
                raise ValueError("Missing pinned native generator/codec pack")
            pack = self.native[spec.modality]
            if pack.codec_id != spec.codec_id or pack.pack_id != spec.pack_id:
                raise ValueError("Requested codec/generator is incompatible with loaded pack")
            return pack.sample(context.facets, spec, noise=noise, seed=seed)
        if seed is not None and generator is None:
            generator = torch.Generator(device=context.facets.device)
            generator.manual_seed(seed)
        if spec.renderer != "native":
            if control is not None:
                # Prompt controls affect P selection, not current raw-to-R.
                raise ValueError("Pretrained renderer must not receive P controls")
            if spec.renderer not in self.pretrained:
                raise ValueError("Pretrained generator is not resident")
            student = self.pretrained[spec.renderer]
            if spec.renderer == "pocket":
                if spec.sample_rate != student.sample_rate:
                    raise ValueError("Pocket output spec must use the exact Mimi native rate")
                from math import ceil

                frames = max(1, ceil(spec.sample_count / student.sample_rate * student.frame_rate))
                output = student.realize(
                    context.facets,
                    max_frames=frames,
                    steps=spec.steps,
                    generator=generator,
                )
                if output.ndim != 2:
                    raise ValueError("Pocket realization must be [B, mono samples]")
                # Renderer EOS may shorten speech; pad deterministically, never
                # interpolate to a target-derived duration or shift its pitch.
                if output.shape[1] < spec.sample_count:
                    output = F.pad(output, (0, spec.sample_count - output.shape[1]))
                return output[:, : spec.sample_count]
            return student.realize(
                context.facets,
                height=spec.height,
                width=spec.width,
                frames=spec.frames,
                steps=spec.steps,
                generator=generator,
            )
        if spec.modality == "audio":
            return self.audio(context, spec, control)
        return self.visual(context, spec, control)
