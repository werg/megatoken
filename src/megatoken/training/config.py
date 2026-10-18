"""Strict training configuration; misspelled keys are errors."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from megatoken.contracts import DEFAULT_LOSS_WEIGHTS
from megatoken.io import content_hash, read_mapping
from megatoken.training.schedule import PHASES


@dataclass(frozen=True)
class TrainConfig:
    phase: str = "B0"
    model: str = "tiny"
    dataset: str = "artifacts/demo"
    output: str = "runs/bootstrap"
    codec: str = "byte"
    device: str = "auto"
    precision: str = "bf16"
    seed: int = 17
    microbatch_episodes: int = 8
    accumulate_scored: int = 65_536
    accumulate_ingested: int = 1_048_576
    max_accumulation_windows: int = 256
    max_steps: int | None = None
    budget: int | None = None
    max_history_tokens: int = 4096
    max_window_records: int = 8
    scored_decisions: int = 2
    max_record_bytes: int = 4096
    generated_capacity: int = 128
    effort: int = 6
    checkpoint_every: int = 100
    keep_checkpoints: int = 3
    eval_every: int = 100
    eval_episodes: int = 16
    log_every: int = 1
    gradient_checkpointing: bool = True
    compile: bool = False
    compile_mode: str = "default"
    expert_backend: str = "torch"
    representation_mode: str = "auto"
    deterministic: bool = False
    require_blackwell: bool = False
    grad_clip: float = 1.0
    weight_decay: float = 0.1
    loss_weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_LOSS_WEIGHTS))
    model_overrides: dict[str, Any] = field(default_factory=dict)
    allow_byte_codec: bool = False
    allow_unverified_data: bool = False
    use_stage_mixture: bool = True
    use_effort_mixture: bool = True
    engram_import: str | None = None

    def __post_init__(self) -> None:
        if self.phase not in PHASES:
            raise ValueError(f"phase must be one of {tuple(PHASES)}")
        if self.precision not in {"bf16", "fp32"}:
            raise ValueError("Only bf16 or fp32 are supported; FP8 requires a measured extension")
        if self.effort not in {1, 2, 3, 6}:
            raise ValueError("effort must be one of 1, 2, 3, 6")
        if self.representation_mode not in {"auto", "bootstrap", "recurrent"}:
            raise ValueError("representation_mode must be auto, bootstrap, or recurrent")
        positive = (
            "microbatch_episodes",
            "accumulate_scored",
            "accumulate_ingested",
            "max_accumulation_windows",
            "max_history_tokens",
            "max_window_records",
            "scored_decisions",
            "max_record_bytes",
            "generated_capacity",
            "keep_checkpoints",
            "log_every",
            "eval_episodes",
        )
        for name in positive:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("checkpoint_every", "eval_every"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} cannot be negative")
        if self.max_steps is not None and self.max_steps <= 0:
            raise ValueError("max_steps must be positive when supplied")
        if self.budget is not None and self.budget <= 0:
            raise ValueError("budget must be positive when supplied")
        if self.grad_clip <= 0 or self.weight_decay < 0:
            raise ValueError("Invalid gradient clipping or decay")
        if set(self.loss_weights) != set(DEFAULT_LOSS_WEIGHTS):
            raise ValueError(f"loss_weights must define exactly {tuple(DEFAULT_LOSS_WEIGHTS)}")
        if any(value < 0 for value in self.loss_weights.values()):
            raise ValueError("Loss weights cannot be negative")

    @property
    def representation_training(self) -> str:
        if self.representation_mode != "auto":
            return self.representation_mode
        return "bootstrap" if self.phase == "B0" else "recurrent"

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> TrainConfig:
        unknown = set(value) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown training settings: {sorted(unknown)}")
        merged = dict(value)
        if "loss_weights" in merged:
            merged["loss_weights"] = {**DEFAULT_LOSS_WEIGHTS, **merged["loss_weights"]}
        return cls(**merged)

    @classmethod
    def load(cls, path: str | Path, **overrides: Any) -> TrainConfig:
        return cls.from_dict(
            {**read_mapping(path), **{k: v for k, v in overrides.items() if v is not None}}
        )

    def fingerprint(self) -> str:
        return content_hash(asdict(self))
