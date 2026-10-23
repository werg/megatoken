"""Optimizer-step-global balancing for routers with replayable soft preferences.

Memory is O(distinct routing contexts), not O(stages x contexts): hard counts
are accumulated eagerly as sufficient statistics, and each detached context is
held once with a per-stage executed-prefix weight. There is no replay cap.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from megatoken.model.adapters import AdapterRoute, OwnerAdapters
from megatoken.model.core import FacetedCore
from megatoken.model.routing import EpochRouter, RowRoute

# Every supported router has ``experts``, ``balance_probabilities`` and a
# ``balance_recorder``. The latter is deliberately attached to a site/controller,
# not to a shared expert bank: balancing should train the selector that made a
# choice, never the parameters shared by its experts.
BalancedRouter = EpochRouter | OwnerAdapters | torch.nn.Module


def balance_routers(model_or_core: FacetedCore | torch.nn.Module) -> dict[str, BalancedRouter]:
    """Return the individual expert selectors that can record balance routes.

    Routed sites use shared selectors: one bootstrap selector and one selector
    per shape pool. Include those once, alongside the explicit P/U routers and
    role-local retrieval adapter controllers. Site aliases are intentionally
    omitted because they would count the same shared selector repeatedly.
    """
    core = getattr(model_or_core, "core", model_or_core)
    routers: dict[str, BalancedRouter] = {
        "p": core.p_router,
        "u": core.u_router,
        **{f"adapter_{name}": owner for name, owner in core.adapters.owners.items()},
    }
    pools = getattr(core, "expert_pools", None)
    if pools is not None:
        routers.update(pools.routing_selectors())
    return routers


@dataclass
class _ContextEntry:
    context: Tensor  # detached view; storage is kept alive by this reference
    version: int
    stage_weights: dict[int, int] = field(default_factory=dict)


@dataclass
class _RouterState:
    router: BalancedRouter
    counts: Tensor
    contexts: dict[tuple, _ContextEntry] = field(default_factory=dict)


@dataclass
class RouterBalanceWindow:
    """Captured inputs are detached; downstream fairness affects router heads only."""

    states: dict[int, _RouterState] = field(default_factory=dict)

    def capture(self, router: BalancedRouter, route: RowRoute | AdapterRoute) -> None:
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
        key = (
            context.data_ptr(),
            tuple(context.shape),
            context.stride(),
            context.dtype,
            context.device,
            context._version,
        )
        entry = state.contexts.get(key)
        if entry is None:
            entry = state.contexts[key] = _ContextEntry(context, context._version)
        entry.stage_weights[route.router_stage] = (
            entry.stage_weights.get(route.router_stage, 0) + used.shape[1]
        )

    @property
    def assignments(self) -> int:
        return int(sum(float(s.counts.sum()) for s in self.states.values()))

    def global_counts(self, distributed, *, routers: dict[str, BalancedRouter]):
        # Every rank contributes a fixed order and shape, even when it has no routes.
        values = {}
        for name, router in sorted(routers.items()):
            state = self.states.get(id(router))
            values[name] = (
                state.counts.to(distributed.device)
                if state is not None
                else torch.zeros(router.experts, dtype=torch.float64, device=distributed.device)
            )
        return distributed.sum_tensors(values)

    def backward(
        self, distributed, *, routers: dict[str, BalancedRouter], coefficient: float
    ) -> dict[str, float]:
        """Exact pooled hard-load/soft-palette proxy, stopping at captured inputs."""
        if coefficient < 0:
            raise ValueError("Balance coefficient cannot be negative")
        counts = self.global_counts(distributed, routers=routers)
        diagnostics = {}
        total_loss = None
        # Each router's term reaches only its own selector. Keep the historical
        # coefficient scale while each site receives its own named diagnostic.
        pool_scale = 0.5
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
            loss = coefficient * pool_scale * router.experts * (fraction * soft_sum / a).sum()
            total_loss = loss if total_loss is None else total_loss + loss
            diagnostics[f"balance_{name}_loss"] = float(loss.detach())
        # Frozen routers (the media-bridge stage trains no router) take no balance update.
        if total_loss is not None and coefficient and total_loss.requires_grad:
            total_loss.backward()
        self.states.clear()
        return diagnostics

    def clear(self):
        self.states.clear()
