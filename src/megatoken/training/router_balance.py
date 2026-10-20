"""Optimizer-step-global router-only balancing without additional optimizer state.

Memory is O(distinct routing contexts), not O(stages x contexts): hard counts
are accumulated eagerly as sufficient statistics, and each detached context is
held once with a per-stage executed-prefix weight. There is no replay cap.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from megatoken.model.routing import EpochRouter, RowRoute


@dataclass
class _ContextEntry:
    context: Tensor  # detached view; storage is kept alive by this reference
    version: int
    stage_weights: dict[int, int] = field(default_factory=dict)


@dataclass
class _RouterState:
    router: EpochRouter
    counts: Tensor
    contexts: dict[tuple, _ContextEntry] = field(default_factory=dict)


@dataclass
class RouterBalanceWindow:
    """Captured inputs are detached; downstream fairness affects router heads only."""
    states: dict[int, _RouterState] = field(default_factory=dict)

    def capture(self, router: EpochRouter, route: RowRoute) -> None:
        if not torch.is_grad_enabled() or route.router_context is None:
            return
        used = route.expert_ids.detach()
        if not used.numel():
            return
        state = self.states.get(id(router))
        if state is None:
            state = self.states[id(router)] = _RouterState(
                router, torch.zeros(router.experts, dtype=torch.float64, device=used.device)
            )
        assignment = used.flatten()
        state.counts.scatter_add_(
            0, assignment, torch.ones_like(assignment, dtype=state.counts.dtype)
        )
        context = route.router_context.detach()
        key = (context.data_ptr(), tuple(context.shape), context.stride(), context.dtype,
               context.device, context._version)
        entry = state.contexts.get(key)
        if entry is None:
            entry = state.contexts[key] = _ContextEntry(context, context._version)
        entry.stage_weights[route.router_stage] = (
            entry.stage_weights.get(route.router_stage, 0) + used.shape[1]
        )

    @property
    def assignments(self) -> int:
        return int(sum(float(s.counts.sum()) for s in self.states.values()))

    def global_counts(self, distributed, *, routers: dict[str, EpochRouter]):
        # Every rank contributes a fixed order and shape, even when it has no routes.
        values = {}
        for name, router in sorted(routers.items()):
            state = self.states.get(id(router))
            values[name] = (
                state.counts.to(distributed.device) if state is not None else
                torch.zeros(router.experts, dtype=torch.float64, device=distributed.device)
            )
        return distributed.sum_tensors(values)

    def backward(self, distributed, *, routers: dict[str, EpochRouter],
                 coefficient: float) -> dict[str, float]:
        """Exact pooled hard-load/soft-palette proxy, stopping at captured inputs."""
        if coefficient < 0:
            raise ValueError("Balance coefficient cannot be negative")
        counts = self.global_counts(distributed, routers=routers)
        diagnostics = {}
        total_loss = None
        pool_scale = 1 / len(routers) if routers else 0.0
        for name, router in sorted(routers.items()):
            hard = counts[name]
            total_assignments = hard.sum()
            a = float(total_assignments)
            diagnostics[f"balance_{name}_assignments"] = a
            if a == 0:
                continue
            fraction = (hard / total_assignments).to(torch.float32).detach()
            soft_sum = None
            state = self.states.get(id(router))
            for entry in () if state is None else state.contexts.values():
                if entry.context._version != entry.version:
                    raise RuntimeError("Captured router context was modified in place")
                for stage, weight in entry.stage_weights.items():
                    probabilities = router.balance_probabilities(entry.context, stage)
                    contribution = (probabilities * weight).sum(0)
                    soft_sum = contribution if soft_sum is None else soft_sum + contribution
            if soft_sum is None:
                continue
            loss = (coefficient * pool_scale * router.experts *
                    (fraction * soft_sum / a).sum())
            total_loss = loss if total_loss is None else total_loss + loss
            diagnostics[f"balance_{name}_loss"] = float(loss.detach())
        if total_loss is not None and coefficient:
            total_loss.backward()
        self.states.clear()
        return diagnostics

    def clear(self):
        self.states.clear()
