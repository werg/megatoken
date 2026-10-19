"""Versioned geometry for raw facet recurrence with bounded causal KV context."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from typing import Any

ARCHITECTURE_VERSION = 8


@dataclass(frozen=True)
class ModelConfig:
    """Explicit encoder, central processor, and predictor allocations."""

    name: str = "32x512"
    architecture_version: int = ARCHITECTURE_VERSION
    n_facets: int = 32
    facet_dim: int = 512
    encoder_layers: int = 2
    encoder_hidden: int = 1024
    central_layers: int = 4
    central_hidden: int = 1024
    prediction_layers: int = 12
    remix_hidden: int = 1024
    max_history_chunks: int = 1024
    max_tracks: int = 8
    source_dim: int = 256
    # Zero disables a donor feature projector. Pinned extractor revisions live
    # in model geometry so incompatible feature caches cannot silently resume.
    audio_feature_dim: int = 0
    vision_feature_dim: int = 0
    audio_feature_id: str = ""
    vision_feature_id: str = ""
    audio_acoustic_feature_dim: int = 0
    audio_acoustic_feature_id: str = ""
    vision_detail_feature_dim: int = 0
    vision_detail_feature_id: str = ""
    vision_codec_feature_dim: int = 0
    vision_codec_feature_id: str = ""
    byte_dim: int = 8
    source_stride: int = 8
    ngram_dim: int = 16
    ngram_rows: int = 2_500_000
    exact_bigrams: bool = True
    ngram_orders: tuple[int, ...] = (1, 2, 3, 4, 8)
    control_dim: int = 128
    codec_noise_std: float = 0.1
    prior_noise_std: float = 0.1
    p_experts: int = 64
    p_hidden: int = 1536
    p_always_hidden: int = 1024
    u_experts: int = 32
    u_hidden: int = 1024
    u_always_hidden: int = 512
    eligible_experts: int = 8
    router_dim: int = 64
    routing_version: int = 3
    query_heads: int = 4
    kv_heads: int = 2
    attention_head_dim: int = 128
    vocab_size: int = 157_184
    output_rank: int = 128
    max_capacity: int = 1024
    norm_eps: float = 1e-6
    expert_backend: str = "auto"
    checkpoint_stages: bool = False

    def __post_init__(self) -> None:
        if self.architecture_version != ARCHITECTURE_VERSION:
            raise ValueError(
                "Incompatible architecture version: direct realization requires version 8; "
                "earlier facet checkpoints cannot be loaded as this architecture"
            )
        positive = (
            "n_facets",
            "facet_dim",
            "encoder_layers",
            "encoder_hidden",
            "central_layers",
            "central_hidden",
            "prediction_layers",
            "source_dim",
            "byte_dim",
            "source_stride",
            "ngram_dim",
            "ngram_rows",
            "control_dim",
            "p_experts",
            "p_hidden",
            "p_always_hidden",
            "u_experts",
            "u_hidden",
            "u_always_hidden",
            "eligible_experts",
            "router_dim",
            "remix_hidden",
            "max_history_chunks",
            "max_tracks",
            "query_heads",
            "kv_heads",
            "attention_head_dim",
            "vocab_size",
            "output_rank",
            "max_capacity",
        )
        for name in positive:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.name in {"v12", "v13", "recurrent32", "compact32"}:
            raise ValueError("Old architecture names are not aliases for the version-4 presets")
        for name in ("codec_noise_std", "prior_noise_std"):
            value = getattr(self, name)
            if not math.isfinite(value) or not math.exp(-4.0) < value < math.exp(1.0):
                raise ValueError(f"{name} must lie strictly within exp(-4) and exp(1)")
        if self.n_facets % 4:
            raise ValueError("n_facets must be divisible by four for pair schedules")
        if self.query_heads % self.kv_heads:
            raise ValueError("kv_heads must divide query_heads")
        if self.attention_head_dim % 2:
            raise ValueError("attention_head_dim must be even for temporal RoPE")
        if not self.ngram_orders or any(order < 1 or order > 8 for order in self.ngram_orders):
            raise ValueError("ngram_orders must contain orders in [1, 8]")
        if self.ngram_rows <= self.exact_rows:
            raise ValueError("ngram_rows must leave space for the hashed fallback")
        for dim, identity in (
            (self.audio_feature_dim, self.audio_feature_id),
            (self.vision_feature_dim, self.vision_feature_id),
            (self.audio_acoustic_feature_dim, self.audio_acoustic_feature_id),
            (self.vision_detail_feature_dim, self.vision_detail_feature_id),
            (self.vision_codec_feature_dim, self.vision_codec_feature_id),
        ):
            if type(dim) is not int or dim < 0 or (
                bool(dim) != bool(identity)
            ) or (identity and (not isinstance(identity, str) or len(identity) < 12)):
                raise ValueError(
                    "External feature projectors require a positive width and pinned ID"
                )
        if self.expert_backend not in {"auto", "torch", "triton"}:
            raise ValueError("Backend must be auto, torch or triton")

    @property
    def exact_rows(self) -> int:
        return 256 + (65_536 if self.exact_bigrams else 0)

    @property
    def output_vocab(self) -> int:
        return self.vocab_size

    @property
    def pair_dim(self) -> int:
        return 2 * self.facet_dim

    @property
    def n_stages(self) -> int:
        """Central plus predictive depth; encoder/realizer work is additional."""
        return self.central_layers + self.prediction_layers

    @property
    def state_geometry(self) -> tuple[int, ...]:
        return (
            self.architecture_version,
            self.n_facets,
            self.facet_dim,
            self.central_layers,
            self.prediction_layers,
            self.max_history_chunks,
            self.max_tracks,
            self.kv_heads,
            self.attention_head_dim,
            self.audio_feature_dim,
            self.vision_feature_dim,
            self.audio_feature_id,
            self.vision_feature_id,
            self.audio_acoustic_feature_dim,
            self.audio_acoustic_feature_id,
            self.vision_detail_feature_dim,
            self.vision_detail_feature_id,
            self.vision_codec_feature_dim,
            self.vision_codec_feature_id,
        )

    @classmethod
    def tiny(cls, **overrides: Any) -> ModelConfig:
        values: dict[str, Any] = dict(
            name="tiny",
            n_facets=4,
            facet_dim=32,
            source_dim=32,
            encoder_hidden=48,
            central_hidden=48,
            control_dim=16,
            byte_dim=4,
            ngram_dim=8,
            ngram_rows=2048,
            exact_bigrams=False,
            p_experts=8,
            p_hidden=48,
            p_always_hidden=24,
            u_experts=8,
            u_hidden=48,
            u_always_hidden=24,
            router_dim=16,
            remix_hidden=48,
            max_history_chunks=8,
            attention_head_dim=8,
            vocab_size=256,
            output_rank=16,
            max_capacity=128,
        )
        values.update(overrides)
        return cls(**values)

    @classmethod
    def compact32(cls, **overrides: Any) -> ModelConfig:
        values: dict[str, Any] = dict(
            name="32x256",
            facet_dim=256,
            source_dim=128,
            encoder_hidden=512,
            central_hidden=512,
            attention_head_dim=64,
        )
        values.update(overrides)
        return cls(**values)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> ModelConfig:
        if values.get("architecture_version") != ARCHITECTURE_VERSION:
            raise ValueError(
                "Saved model configuration must explicitly declare architecture_version=8; "
                "earlier checkpoints are intentionally unsupported"
            )
        unknown = set(values) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown model configuration fields: {sorted(unknown)}")
        values = dict(values)
        if "ngram_orders" in values:
            values["ngram_orders"] = tuple(values["ngram_orders"])
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
