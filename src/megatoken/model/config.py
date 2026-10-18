"""One explicit architecture configuration for training and serving.

The large expert and native-table allocations follow the v13 proposal. The
remaining modules are a new implementation, not an assertion of its old ledger.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any


@dataclass(frozen=True)
class ModelConfig:
    name: str = "v13"
    n_facets: int = 128
    facet_dim: int = 256
    n_stages: int = 16
    factual_stages: int = 6
    source_dim: int = 128
    byte_dim: int = 8
    source_stride: int = 8
    ngram_dim: int = 16
    ngram_rows: int = 2_500_000
    exact_bigrams: bool = True
    ngram_orders: tuple[int, ...] = (1, 2, 3, 4, 8)
    control_dim: int = 128
    root_dim: int = 64
    n_regions: int = 8
    regional_dim: int = 8
    p_experts: int = 128
    p_hidden: int = 1536
    p_always_hidden: int = 1024
    u_experts: int = 64
    u_hidden: int = 1024
    u_always_hidden: int = 512
    eligible_experts: int = 8
    router_dim: int = 64
    router_exploration: float = 0.0
    routing_version: int = 1
    gdn_sites: tuple[int, ...] = (1, 2, 4, 6)
    gdn_streams: int = 12
    gdn_heads: int = 4
    gdn_head_dim: int = 64
    query_heads: int = 4
    historical_kv_heads: int = 2
    live_kv_heads: int = 1
    attention_head_dim: int = 64
    historical_slots: int = 32
    max_history_records: int = 256
    literal_window: int = 4096
    carry_slots: int = 4
    vocab_size: int = 157_184
    output_rank: int = 128
    max_capacity: int = 4096
    norm_eps: float = 1e-6
    expert_backend: str = "torch"
    recurrent_backend: str = "torch"
    checkpoint_stages: bool = False

    def __post_init__(self) -> None:
        positive = (
            "n_facets",
            "facet_dim",
            "source_dim",
            "byte_dim",
            "source_stride",
            "ngram_dim",
            "ngram_rows",
            "control_dim",
            "root_dim",
            "n_regions",
            "regional_dim",
            "p_experts",
            "p_hidden",
            "p_always_hidden",
            "u_experts",
            "u_hidden",
            "u_always_hidden",
            "eligible_experts",
            "router_dim",
            "gdn_streams",
            "gdn_heads",
            "gdn_head_dim",
            "query_heads",
            "historical_kv_heads",
            "live_kv_heads",
            "attention_head_dim",
            "historical_slots",
            "max_history_records",
            "literal_window",
            "carry_slots",
            "vocab_size",
            "output_rank",
            "max_capacity",
        )
        for name in positive:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.n_stages != 16 or self.factual_stages != 6:
            raise ValueError("The causal boundary contract requires 16 stages and cutoff 6")
        if self.n_facets % 4:
            raise ValueError("n_facets must be divisible by four for pair schedules")
        if self.query_heads % self.historical_kv_heads or self.query_heads % self.live_kv_heads:
            raise ValueError("KV head counts must divide query_heads")
        if len(self.gdn_sites) != 4 or len(set(self.gdn_sites)) != 4:
            raise ValueError("Exactly four independent GDN sites are required")
        if any(s < 1 or s > self.factual_stages for s in self.gdn_sites):
            raise ValueError("GDN writes must be factual stages")
        if not self.ngram_orders or any(n < 1 or n > 8 for n in self.ngram_orders):
            raise ValueError("ngram_orders must contain orders in [1, 8]")
        if self.ngram_rows <= self.exact_rows:
            raise ValueError("ngram_rows must leave space for the hashed fallback")
        if not 0 <= self.router_exploration <= 0.25:
            raise ValueError("router_exploration must be in [0, .25]")
        if self.expert_backend not in {"torch", "triton"} or self.recurrent_backend not in {
            "torch",
            "triton",
        }:
            raise ValueError("Backend must be torch or triton; CUDA is explicitly gated")

    @property
    def exact_rows(self) -> int:
        return 256 + (65_536 if self.exact_bigrams else 0)

    @property
    def output_vocab(self) -> int:
        return self.vocab_size

    @property
    def pair_dim(self) -> int:
        return 2 * self.facet_dim

    @classmethod
    def tiny(cls, **overrides: Any) -> ModelConfig:
        values: dict[str, Any] = dict(
            name="tiny",
            n_facets=16,
            facet_dim=32,
            source_dim=32,
            control_dim=16,
            root_dim=8,
            regional_dim=2,
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
            gdn_streams=3,
            gdn_heads=2,
            gdn_head_dim=8,
            attention_head_dim=8,
            historical_slots=4,
            max_history_records=8,
            literal_window=128,
            vocab_size=256,
            output_rank=16,
            max_capacity=256,
        )
        values.update(overrides)
        return cls(**values)

    @classmethod
    def v12(cls, **overrides: Any) -> ModelConfig:
        values: dict[str, Any] = dict(
            name="v12",
            p_experts=8,
            p_hidden=768,
            p_always_hidden=256,
            u_experts=8,
            u_hidden=768,
            u_always_hidden=256,
            source_dim=64,
            control_dim=64,
            historical_kv_heads=1,
        )
        values.update(overrides)
        return cls(**values)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> ModelConfig:
        unknown = set(values) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown model configuration fields: {sorted(unknown)}")
        values = dict(values)
        for name in ("ngram_orders", "gdn_sites"):
            if name in values:
                values[name] = tuple(values[name])
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
