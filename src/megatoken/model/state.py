"""Explicit, immutable cache records with model-version and episode identity."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
from torch import Tensor

from .routing import RoutingProgram


@dataclass(frozen=True)
class KVMemory:
    keys: Tensor  # [batch, kv heads, retained slots, head dim]
    values: Tensor
    mask: Tensor  # [batch, retained slots]

    def detach(self) -> KVMemory:
        return KVMemory(self.keys.detach(), self.values.detach(), self.mask)

    def reset(self, rows: Tensor) -> KVMemory:
        keep = ~rows
        return KVMemory(
            self.keys * keep[:, None, None, None],
            self.values * keep[:, None, None, None],
            self.mask & keep[:, None],
        )


@dataclass(frozen=True)
class FactualState:
    facets: Tensor
    recurrent: tuple[Tensor, ...]
    history_a: KVMemory
    history_b: KVMemory
    literal: KVMemory
    source: Tensor
    source_mask: Tensor
    carry: Tensor
    records_seen: Tensor
    last_record_ids: Tensor
    episode_ids: Tensor
    model_version: int
    routing_version: int
    routing_aux_loss: Tensor
    last_program: RoutingProgram | None = None

    @property
    def batch_size(self) -> int:
        return self.facets.shape[0]

    def detach(self) -> FactualState:
        """An explicit TBPTT boundary; callers must meter its approximation."""
        return replace(
            self,
            facets=self.facets.detach(),
            recurrent=tuple(s.detach() for s in self.recurrent),
            history_a=self.history_a.detach(),
            history_b=self.history_b.detach(),
            literal=self.literal.detach(),
            source=self.source.detach(),
            carry=self.carry.detach(),
            routing_aux_loss=self.routing_aux_loss.detach(),
            last_program=None,
        )


@dataclass(frozen=True)
class PredictionState:
    factual: FactualState
    facets: Tensor
    root: Tensor
    control: Tensor
    effort: int
    stage: int
    program: RoutingProgram
    previous_programs: tuple[RoutingProgram, ...]
    route_context: Tensor
    deterministic: bool = True
    output_refreshed: bool = False

    @property
    def source(self) -> Tensor:
        return self.factual.source

    @property
    def source_mask(self) -> Tensor:
        return self.factual.source_mask

    @property
    def routing_aux_loss(self) -> Tensor:
        programs = (*self.previous_programs, self.program)
        return torch.stack([program.aux_loss for program in programs]).mean()


def assert_cache_version(state: FactualState, model_version: int, routing_version: int) -> None:
    if state.model_version != model_version:
        raise RuntimeError(
            f"Stale cache from model version {state.model_version}; current version is {model_version}. "
            "Reconstruct actual history after parameter updates."
        )
    if state.routing_version != routing_version:
        raise RuntimeError("Routing program version changed; reconstruct the factual state")
