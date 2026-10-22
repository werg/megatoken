"""Strict training configuration; misspelled keys are errors."""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from megatoken.contracts import DEFAULT_LOSS_WEIGHTS
from megatoken.data.recipes import stage_recipe
from megatoken.io import read_mapping
from megatoken.training.schedule import PHASES, Phase


@dataclass(frozen=True)
class DecoupledConfig:
    """Independent learner settings; worker_id may be supplied by sandbox environment."""

    coordinator_url: str
    run_id: str
    workers: int = 8
    worker_id: int | None = None
    local_steps: int = 256
    io_workers: int = 4
    timeout_seconds: float = 120.0
    max_offline_steps: int = 2048
    wire_dtype: str = "bf16"
    allow_private_http: bool = False
    checkpoint_s3: str | None = None

    def __post_init__(self) -> None:
        if not self.coordinator_url or not self.run_id:
            raise ValueError("Decoupled training needs coordinator_url and run_id")
        if not 1 <= self.workers <= 256:
            raise ValueError("decoupled workers must be in [1, 256]")
        if min(self.local_steps, self.io_workers) < 1:
            raise ValueError("decoupled steps and I/O worker count must be positive")
        if self.timeout_seconds <= 0 or self.max_offline_steps < 0:
            raise ValueError("Invalid decoupled timeout/offline tolerance")
        if self.wire_dtype not in {"bf16", "fp32"}:
            raise ValueError("decoupled wire_dtype must be bf16 or fp32")
        if self.checkpoint_s3 is not None and not self.checkpoint_s3.startswith("s3://"):
            raise ValueError("checkpoint_s3 must use an s3://bucket/prefix URI")
        if self.worker_id is not None and not 0 <= self.worker_id < self.workers:
            raise ValueError("Invalid decoupled worker_id")

    @property
    def resolved_worker_id(self) -> int:
        raw = self.worker_id
        if raw is None:
            raw = int(os.environ.get("MEGATOKEN_WORKER_ID", "-1"))
        if not 0 <= raw < self.workers:
            raise ValueError("Set MEGATOKEN_WORKER_ID in [0, workers) on each sandbox")
        return raw

    @classmethod
    def from_dict(cls, mapping: dict[str, Any]) -> DecoupledConfig:
        unknown = set(mapping) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown decoupled settings: {sorted(unknown)}")
        return cls(**mapping)


@dataclass(frozen=True)
class TrainConfig:
    phase: str = "B0"
    model: str = "32x512"
    # Prepared roots read as one corpus (data.storage.PreparedCorpus); each
    # stage stream is read from whichever roots contain it.
    datasets: list[str] = field(default_factory=lambda: ["artifacts/demo"])
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
    # A smaller budget than the stage's own limit, in the stage's own meter.
    budget: int | None = None
    max_history_tokens: int = 4096
    max_window_records: int = 8
    scored_decisions: int = 2
    max_record_bytes: int = 4096
    generated_capacity: int = 128
    effort: int = 6
    checkpoint_every: int = 100
    keep_checkpoints: int = 3
    eval_every: int = 0
    # Held-out windows scored per stage stream at each evaluation.
    eval_windows: int = 16
    log_every: int = 1
    gradient_checkpointing: bool = True
    compile: bool = False
    compile_mode: str = "default"
    expert_backend: str = "auto"
    representation_mode: str = "auto"
    # Optimizer-step accumulation target for a stage metered in G (events).
    accumulate_events: int = 32
    # Locked pack manifest (omni/loader.py): the Pocket/Sana/Sana-Video realizers
    # are installed before optimizer enumeration and checkpoint restore.
    packs: str | None = None
    # "media_bridge" trains only the media input projection, modality embedding
    # and realizer conditions (identically on every DiLoCo worker); "all" trains
    # everything. The stage plan decides: the MB stage requires media_bridge.
    trainable: str = "all"
    # Same-text alignment of byte-E facets with teacher-feature facets: JSONL
    # rows {"text_sha256", "asset"} of EmbeddingGemma2 text features.
    text_anchors: str | None = None
    text_anchor_weight: float = 0.1
    max_media_events_per_window: int = 5
    deterministic: bool = False
    require_blackwell: bool = False
    grad_clip: float = 1.0
    weight_decay: float = 0.1
    inner_optimizer: str = "adamw"
    router_balance_mode: str = "global"
    decoupled: DecoupledConfig | None = None
    loss_weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_LOSS_WEIGHTS))
    model_overrides: dict[str, Any] = field(default_factory=dict)
    allow_byte_codec: bool = False
    allow_unverified_data: bool = False
    use_stage_mixture: bool = False
    use_effort_mixture: bool = False
    engram_import: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.decoupled, dict):
            object.__setattr__(self, "decoupled", DecoupledConfig.from_dict(self.decoupled))
        if self.decoupled is not None and not isinstance(self.decoupled, DecoupledConfig):
            raise ValueError("decoupled must be a settings object or mapping")
        if self.router_balance_mode != "global":
            raise ValueError(
                "The local router fairness objective is removed; "
                "all training uses optimizer-step global router replay"
            )
        if self.inner_optimizer not in {"adamw", "muon"}:
            raise ValueError("inner_optimizer must be adamw or muon")
        if self.phase not in PHASES:
            raise ValueError(f"phase must be one of {tuple(PHASES)}")
        if self.precision not in {"bf16", "fp32"}:
            raise ValueError("Only bf16 or fp32 are supported; FP8 requires a measured extension")
        if self.effort not in {1, 2, 3, 6}:
            raise ValueError("effort must be one of 1, 2, 3, 6")
        if self.expert_backend not in {"auto", "torch", "triton"}:
            raise ValueError("expert_backend must be auto, torch, or triton")
        if self.representation_mode not in {"auto", "bootstrap", "recurrent"}:
            raise ValueError("representation_mode must be auto, bootstrap, or recurrent")
        if self.trainable not in {"all", "media_bridge"}:
            raise ValueError("trainable is all or media_bridge")
        if self.trainable != stage_recipe(self.phase).constraints.get("trainable", "all"):
            raise ValueError(f"Stage {self.phase} trains {self.trainable!r} parameters only")
        if self.trainable == "media_bridge" and self.packs is None:
            raise ValueError("Media bridge training needs the realizer packs")
        if not 0 < self.text_anchor_weight <= 1:
            raise ValueError("Text anchor weight must be in (0, 1]")
        if self.packs is not None and not self.packs:
            raise ValueError("Pack manifest path must be nonempty")
        if not isinstance(self.datasets, list) or not self.datasets:
            raise ValueError("datasets must be a nonempty list of prepared roots")
        if not all(isinstance(root, str) and root for root in self.datasets):
            raise ValueError("datasets must be nonempty path strings")
        positive = (
            "microbatch_episodes",
            "accumulate_events",
            "accumulate_scored",
            "accumulate_ingested",
            "max_accumulation_windows",
            "max_history_tokens",
            "max_window_records",
            "max_media_events_per_window",
            "scored_decisions",
            "max_record_bytes",
            "generated_capacity",
            "keep_checkpoints",
            "log_every",
            "eval_windows",
        )
        for name in positive:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("checkpoint_every", "eval_every"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} cannot be negative")
        if self.max_steps is not None and self.max_steps <= 0:
            raise ValueError("max_steps must be positive when supplied")
        if self.budget is not None and not 0 < self.budget <= PHASES[self.phase].budget:
            raise ValueError(
                f"budget is in stage {self.phase}'s meter {PHASES[self.phase].meter} and must "
                f"be in (0, {PHASES[self.phase].budget}], the stage's own limit"
            )
        if self.grad_clip <= 0 or self.weight_decay < 0:
            raise ValueError("Invalid gradient clipping or decay")
        if set(self.loss_weights) != set(DEFAULT_LOSS_WEIGHTS):
            raise ValueError(f"loss_weights must define exactly {tuple(DEFAULT_LOSS_WEIGHTS)}")
        if any(value < 0 for value in self.loss_weights.values()):
            raise ValueError("Loss weights cannot be negative")

    @property
    def schedule(self) -> Phase:
        """The stage's learning-rate schedule, budget and meter (``budget`` may shorten it)."""
        phase = PHASES[self.phase]
        return dataclasses.replace(phase, budget=self.budget or phase.budget)

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
