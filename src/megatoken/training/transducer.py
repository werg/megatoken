"""v10 retrieval boundary schedule, hindsight curriculum and exact step checkpoints.

This module is the native training orchestration layer; v9 source RAG stays an
independent exact-citation service and must not mutate the transducer agenda.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from megatoken.memory.transducer import Lookup, SearchAgenda, TransducerBank
from megatoken.memory.transducer_growth import TransducerGrowth
from megatoken.memory.transducer_optimizer import BlockScaleDescent
from megatoken.model.multitrack import MultiTrackState
from megatoken.model.transducer import HindsightValues, NativeTransducer

from .transducer_diloco import TransducerCoordinator


@dataclass
class BoundaryScheduler:
    """One session, one outstanding serialized lookup; internal events never recurse."""

    training: bool = False
    seed: int = 0
    max_lookups: int = 256
    lookups: int = 0
    ordinary_since_boundary: int = 0
    pending: bool = False
    artificial_interval: int = 8
    _rng: torch.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = torch.Generator().manual_seed(self.seed)
        self._next_interval()

    def _next_interval(self) -> None:
        self.artificial_interval = (
            int(torch.randint(4, 13, (1,), generator=self._rng))
            if self.training else 8
        )

    def first_metadata(self) -> bool:
        """Call immediately after metadata is committed, before predicting an answer."""
        return self.begin()

    def begin(self) -> bool:
        if self.pending or self.lookups >= self.max_lookups:
            return False
        self.pending = True
        return True

    def ordinary_commit(
        self, *, completed_turn: bool = False, source_arrived: bool = False
    ) -> bool:
        self.ordinary_since_boundary += 1
        if completed_turn or source_arrived or (
            self.ordinary_since_boundary >= self.artificial_interval
        ):
            return self.begin()
        return False

    def internal_commit(self) -> None:
        """Do not increment the external causal clock or schedule a nested lookup."""
        if not self.pending:
            raise RuntimeError("Internal retrieval event outside a boundary")

    def finish(self, *, committed: bool) -> None:
        if not self.pending:
            raise RuntimeError("No outstanding boundary to finish")
        self.pending = False
        if committed:
            self.lookups += 1
            self.ordinary_since_boundary = 0
            self._next_interval()

    def state_dict(self) -> dict:
        return {
            "training": self.training,
            "seed": self.seed,
            "max_lookups": self.max_lookups,
            "lookups": self.lookups,
            "ordinary_since_boundary": self.ordinary_since_boundary,
            "pending": self.pending,
            "artificial_interval": self.artificial_interval,
            "rng": self._rng.get_state(),
        }

    def load_state_dict(self, state: dict) -> None:
        if state["pending"]:
            raise ValueError("Resume the previous complete checkpoint, not a partial lookup")
        if self.training != state["training"] or self.max_lookups != state["max_lookups"]:
            raise ValueError("Incompatible boundary scheduler state")
        self.lookups = int(state["lookups"])
        self.ordinary_since_boundary = int(state["ordinary_since_boundary"])
        self.pending = False
        self.artificial_interval = int(state["artificial_interval"])
        self._rng.set_state(state["rng"])


@dataclass(frozen=True)
class HindsightSchedule:
    warmup_steps: int = 16384
    overlay_steps: int = 65536

    def stage(self, step: int) -> str:
        return "full" if step < 4096 else "gap1" if step < 8192 else "gap2"

    def overlay(self, step: int, sequence: int, *, learned: bool) -> float:
        if step < 0 or sequence < 0:
            raise ValueError("Training step and sequence counters must be nonnegative")
        if step < self.warmup_steps:
            return 1.0
        if step < self.warmup_steps + self.overlay_steps:
            return 1 - (step - self.warmup_steps) / self.overlay_steps
        return 1.0 if sequence % 8 == 0 else 0.0

    def compute_hindsight(self, step: int, sequence: int, *, learned: bool) -> bool:
        # After the learned-value switch the full hindsight pass is reserved
        # for the 1-in-8 hindsight-only sequences, including continuing write-out.
        return not learned or self.overlay(step, sequence, learned=learned) > 0


@dataclass
class PendingBoundary:
    lookup: Lookup
    hindsight: Tensor | None
    sink_only: bool
    alpha: float
    step: int


class TransducerTrainingOwner:
    """Connects the native differentiable lookup to sparse updates and structural stats."""

    def __init__(
        self,
        native: NativeTransducer,
        optimizer: BlockScaleDescent,
        growth: TransducerGrowth,
        coordinator: TransducerCoordinator | None = None,
    ) -> None:
        self.native = native
        self.bank = native.bank
        if optimizer.bank is not self.bank or growth.bank is not self.bank:
            raise ValueError("Transducer training components must share one bank")
        self.optimizer = optimizer
        self.growth = growth
        self.coordinator = coordinator
        self.schedule = HindsightSchedule()
        self.pending: list[PendingBoundary] = []

    def hindsight(
        self,
        raw: Tensor,
        roles: Tensor,
        groups: Tensor,
        meridians: Tensor,
        tracks: Tensor,
        *,
        step: int,
        sequence: int,
    ) -> HindsightValues | None:
        if not self.schedule.compute_hindsight(
            step, sequence, learned=self.bank.learned_values
        ):
            return None
        return self.native.hindsight_pass(
            raw, roles, groups, meridians, tracks, step=step
        )

    def boundary(
        self,
        state: MultiTrackState,
        agenda: SearchAgenda,
        *,
        step: int,
        sequence: int,
        hindsight: Tensor | None,
        sink_only: bool = False,
    ) -> tuple[MultiTrackState, Lookup | None]:
        alpha = self.schedule.overlay(
            step, sequence, learned=self.bank.learned_values
        )
        if alpha > 0 and hindsight is None:
            raise ValueError("Required training hindsight was not computed")
        committed, lookup = self.native.boundary(
            state, agenda,
            training=True,
            hindsight=hindsight,
            overlay=alpha,
            exploration_seed=step * 1000003 + sequence,
        )
        if lookup is not None:
            lookup.value.retain_grad()
            lookup.landing.retain_grad()
            self.pending.append(PendingBoundary(
                lookup, hindsight, sink_only, alpha, step
            ))
        return committed, lookup

    def after_backward(self) -> dict[str, int]:
        """Must run after the causal future loss's backward and before sparse step."""
        written_out = 0
        observed = 0
        for pending in self.pending:
            lookup = pending.lookup
            gradient_value = lookup.value.grad
            if gradient_value is None:
                raise RuntimeError("Future loss did not differentiate the knowledge readout")
            gradient_location = lookup.landing.grad
            if gradient_location is None:
                gradient_location = torch.zeros_like(lookup.landing)
            self.growth.observe(
                lookup,
                gradient_value=gradient_value,
                gradient_location=gradient_location,
                step=pending.step,
                hindsight=pending.hindsight,
                sink_only=pending.sink_only,
            )
            observed += 1
            if pending.hindsight is not None and not pending.sink_only:
                if not self.bank.learned_values:
                    for edge_id, weight in lookup.edge_weights.items():
                        self.optimizer.add_write(
                            edge_id,
                            pending.hindsight,
                            float(weight.detach()),
                        )
                if pending.alpha > 0:
                    written_out += self.growth.write_out(
                        lookup,
                        pending.hindsight,
                        gradient_value,
                        step=pending.step,
                        sink_only=False,
                    ) is not None
        self.pending.clear()
        return {"observed": observed, "writeouts": written_out}

    def optimizer_barrier(self, step: int) -> dict[str, int]:
        if self.pending:
            raise RuntimeError("Backward evidence must be collected before publication")
        return self.optimizer.step(step)

    def checkpoint(
        self,
        destination: str | Path,
        *,
        model_state: dict[str, Any],
        training_state: dict[str, Any],
        scheduler: BoundaryScheduler,
        agenda: SearchAgenda,
    ) -> None:
        """One atomic checkpoint containing exact active masters and random state."""
        if self.pending or scheduler.pending or (
            self.coordinator is not None and self.coordinator.pending
        ):
            raise RuntimeError("Only a complete forward/backward/merge window is checkpointable")
        snapshot = {
            "identity": self.bank.identity,
            "model": model_state,
            "bank": self.bank.state_snapshot(),
            "sparse_optimizer": self.optimizer.state_dict(),
            "growth": self.growth.state_dict(),
            "coordinator": None if self.coordinator is None else self.coordinator.state_dict(),
            "training": training_state,
            "scheduler": scheduler.state_dict(),
            "agenda": {
                "locations": [location.detach().cpu() for location in agenda.locations],
                "lookups": agenda.lookups,
            },
            "rng_cpu": torch.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
            temp_path = Path(temporary.name)
        try:
            torch.save(snapshot, temp_path)
            os.replace(temp_path, path)
        finally:
            temp_path.unlink(missing_ok=True)

    def restore(
        self,
        checkpoint: str | Path,
        *,
        scheduler: BoundaryScheduler,
    ) -> tuple[dict, dict, SearchAgenda]:
        raw = torch.load(checkpoint, weights_only=True, map_location="cpu")
        if raw["identity"] != self.bank.identity:
            raise ValueError("Refusing to load a v9 semantic KB checkpoint into v10")
        self.bank.load_snapshot(raw["bank"])
        self.optimizer.load_state_dict(raw["sparse_optimizer"])
        self.growth.load_state_dict(raw["growth"])
        if self.coordinator is not None:
            if raw["coordinator"] is None:
                raise ValueError("Distributed run lacks coordinator checkpoint")
            self.coordinator.load_state_dict(raw["coordinator"])
        scheduler.load_state_dict(raw["scheduler"])
        agenda = SearchAgenda([
            x.to(self.bank.device) for x in raw["agenda"]["locations"]
        ], lookups=raw["agenda"]["lookups"])
        self.bank.refresh(agenda)
        torch.set_rng_state(raw["rng_cpu"])
        if raw["rng_cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(raw["rng_cuda"])
        return raw["model"], raw["training"], agenda
