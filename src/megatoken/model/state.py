"""Selected raw facets and cached causal context for future predictions."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
from torch import Tensor


@dataclass(frozen=True)
class KVHistory:
    """Projected central context from selected inputs, in causal order.

    Keys/values are [batch, kv_heads, slots, head_dim]. None mask means every
    slot is valid, preserving fused attention eligibility for unpadded histories.
    This is an explicit bounded cache, not an alternative recurrent state.
    """

    keys: Tensor
    values: Tensor
    mask: Tensor | None = None

    def detach(self) -> KVHistory:
        return replace(self, keys=self.keys.detach(), values=self.values.detach())

    @property
    def slots(self) -> int:
        return self.keys.shape[2]


@dataclass(frozen=True)
class CoreState:
    """Exact selected raw tensor, plus C's final state for future predictions.

    Realization reads ``facets`` directly. Only the startup sentinel lacks
    processed features. History contains every committed selected input.
    """

    facets: Tensor
    final: Tensor | None
    history: KVHistory
    records_seen: Tensor
    last_record_ids: Tensor
    role_ids: Tensor
    episode_ids: Tensor
    model_version: int
    routing_version: int
    geometry: tuple[int, ...]

    @property
    def batch_size(self) -> int:
        return self.facets.shape[0]

    @property
    def device(self) -> torch.device:
        return self.facets.device

    def detach(self) -> CoreState:
        """An explicit TBPTT boundary across both facet recurrence and history."""
        return replace(
            self,
            facets=self.facets.detach(),
            final=None if self.final is None else self.final.detach(),
            history=self.history.detach(),
        )


@dataclass(frozen=True)
class RealizationContext:
    """The canonical raw-facet block consumed directly by every realizer."""

    facets: Tensor

    @property
    def route_context(self) -> Tensor:
        return self.facets.mean(1)

    @property
    def batch_size(self) -> int:
        return self.facets.shape[0]


def assert_cache_version(state: CoreState, model_version: int, routing_version: int) -> None:
    if state.model_version != model_version:
        raise RuntimeError(
            f"Stale cache from model version {state.model_version}; current version is {model_version}. "
            "Reconstruct the ordered observation/action history after parameter updates."
        )
    if state.routing_version != routing_version:
        raise RuntimeError("Routing program version changed; reconstruct the core state")
