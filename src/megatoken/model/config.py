"""Versioned geometry for raw facet recurrence with bounded causal KV context."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, fields
from typing import Any

from megatoken.teacher import PINNED_TEACHER, validate_extractor

ARCHITECTURE_VERSION = 11
# Reserved system tracks. Track ``RESERVED_TRACKS[i]`` occupies index
# ``max_tracks + i`` and never counts against the user-visible track registry.
TRANSDUCER_QUERY_TRACK = "__transducer_query__"
KNOWLEDGE_TRACK = "__transducer_knowledge__"
METADATA_TRACK = "__metadata__"
RESERVED_TRACKS = (TRANSDUCER_QUERY_TRACK, KNOWLEDGE_TRACK, METADATA_TRACK)


@dataclass(frozen=True)
class ModelConfig:
    """Explicit encoder, central processor, and predictor allocations."""

    name: str = "32x512"
    architecture_version: int = ARCHITECTURE_VERSION
    n_facets: int = 32
    facet_dim: int = 512
    encoder_layers: int = 2
    central_layers: int = 4
    prediction_layers: int = 12
    max_history_chunks: int = 1024
    max_tracks: int = 8
    # The one media input: pinned EmbeddingGemma2 hidden states (see
    # megatoken/teacher.py). Part of the geometry, so feature caches from a
    # different teacher can never resume into this model.
    media_extractor: str = PINNED_TEACHER
    byte_dim: int = 8
    source_stride: int = 8
    ngram_dim: int = 16
    ngram_rows: int = 2_500_000
    exact_bigrams: bool = True
    ngram_orders: tuple[int, ...] = (1, 2, 3, 4, 8)
    codec_noise_std: float = 0.1
    prior_noise_std: float = 0.1
    p_experts: int = 64
    u_experts: int = 32
    projection_experts: int = 8
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
    # The continuous-state transducer memory. Values are ``transducer_value_rows``
    # native-width rows (64 x 512 = 32,768 at the selected geometry).
    transducer_location_dim: int = 256
    transducer_query_dim: int = 1024
    transducer_value_rows: int = 64
    transducer_neighbors: int = 1024
    transducer_selected: int = 32
    transducer_exploration: int = 2
    transducer_max_lookups: int = 256
    transducer_max_edges: int = 1_000_000
    transducer_byte_budget: int = 24_000_000_000
    # Every shared expert matrix has its own low-rank adapter factors. E/C/P/R
    # controllers select sparse mixtures from the latest causal lookup.
    adapter_experts: int = 8
    adapter_rank: int = 8
    adapter_topk: int = 2

    def __post_init__(self) -> None:
        if self.architecture_version != ARCHITECTURE_VERSION:
            raise ValueError(
                "Incompatible architecture version: this model requires version 11; "
                "earlier checkpoints cannot be loaded as this architecture"
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
            "projection_experts",
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
            "adapter_experts",
            "adapter_rank",
            "adapter_topk",
        )
        for name in positive:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("codec_noise_std", "prior_noise_std"):
            value = getattr(self, name)
            if not math.isfinite(value) or not math.exp(-4.0) < value < math.exp(1.0):
                raise ValueError(f"{name} must lie strictly within exp(-4) and exp(1)")
        if self.adapter_topk > self.adapter_experts:
            raise ValueError("adapter_topk cannot exceed adapter_experts")
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
        validate_extractor(self.media_extractor)
        if self.expert_backend not in {"auto", "torch", "triton"}:
            raise ValueError("Backend must be auto, torch or triton")
        _ = self.transducer_geometry  # validates the bank geometry

    @property
    def exact_rows(self) -> int:
        return 256 + (65_536 if self.exact_bigrams else 0)

    @property
    def pair_dim(self) -> int:
        return 2 * self.facet_dim

    # Shared shapes are architectural invariants, not independent hidden-width
    # knobs. Every single-facet FFN uses D -> 2D -> D; pair FFNs use 2D -> 4D -> 2D.
    @property
    def source_dim(self) -> int:
        return self.facet_dim

    @property
    def control_dim(self) -> int:
        return self.facet_dim

    @property
    def encoder_hidden(self) -> int:
        return 2 * self.facet_dim

    central_hidden = encoder_hidden
    remix_hidden = encoder_hidden
    u_hidden = encoder_hidden
    u_always_hidden = encoder_hidden

    @property
    def p_hidden(self) -> int:
        return 2 * self.pair_dim

    p_always_hidden = p_hidden

    @property
    def transducer_geometry(self):
        from megatoken.memory.transducer import TransducerGeometry

        return TransducerGeometry(
            location_dim=self.transducer_location_dim,
            query_dim=self.transducer_query_dim,
            value_dim=self.transducer_value_rows * self.facet_dim,
            neighbors=self.transducer_neighbors,
            selected=self.transducer_selected,
            exploration=self.transducer_exploration,
            max_lookups=self.transducer_max_lookups,
            max_edges=self.transducer_max_edges,
            byte_budget=self.transducer_byte_budget,
        )

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
            self.media_extractor,
        )

    @classmethod
    def tiny(cls, **overrides: Any) -> ModelConfig:
        values: dict[str, Any] = dict(
            name="tiny",
            n_facets=4,
            facet_dim=32,
            byte_dim=4,
            ngram_dim=8,
            ngram_rows=2048,
            exact_bigrams=False,
            p_experts=8,
            u_experts=8,
            projection_experts=4,
            router_dim=16,
            max_history_chunks=8,
            attention_head_dim=8,
            vocab_size=256,
            output_rank=16,
            max_capacity=128,
            transducer_location_dim=8,
            transducer_query_dim=16,
            transducer_value_rows=4,
            transducer_neighbors=4,
            transducer_selected=4,
            transducer_exploration=1,
            transducer_max_lookups=16,
            transducer_max_edges=256,
            transducer_byte_budget=1_000_000,
        )
        values.update(overrides)
        return cls(**values)

    @classmethod
    def compact32(cls, **overrides: Any) -> ModelConfig:
        values: dict[str, Any] = dict(
            name="32x256",
            facet_dim=256,
            attention_head_dim=64,
        )
        values.update(overrides)
        return cls(**values)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> ModelConfig:
        if values.get("architecture_version") != ARCHITECTURE_VERSION:
            raise ValueError(
                "Saved model configuration must explicitly declare architecture_version=11; "
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
        """JSON-canonical (sequences as lists), so a dict that went through a JSON
        manifest compares equal to a freshly built one."""
        return json.loads(json.dumps(asdict(self)))

