"""Optimizer-step-global router-only balancing without additional optimizer state."""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from megatoken.model.routing import EpochRouter, RowRoute


@dataclass(frozen=True)
class RouterReceipt:
    router: EpochRouter
    stage: int
    input: Tensor
    selected: Tensor  # [rows, actual used prefix], before capacity drops

    @property
    def assignments(self) -> int:
        return self.selected.numel()


@dataclass
class RouterBalanceWindow:
    """Captured inputs are detached; downstream fairness affects router heads only."""
    receipts: list[RouterReceipt] = field(default_factory=list)
    max_assignments: int = 262144

    def capture(self, router: EpochRouter, route: RowRoute) -> None:
        if not torch.is_grad_enabled() or route.router_context is None:
            return
        used = route.expert_ids.detach()
        if not used.numel():
            return
        if sum(r.assignments for r in self.receipts) + used.numel() > self.max_assignments:
            raise RuntimeError("Router replay capture exceeded the declared memory budget")
        self.receipts.append(
            RouterReceipt(router, route.router_stage,
                          route.router_context.detach().contiguous().clone(),
                          used.contiguous().clone())
        )

    def global_counts(self, distributed, *, routers: dict[str, EpochRouter]):
        # Every rank contributes a fixed order and shape, even when it has no routes.
        values = {}
        for name, router in sorted(routers.items()):
            counts = torch.zeros(router.experts, dtype=torch.float64, device=distributed.device)
            for receipt in self.receipts:
                if receipt.router is router:
                    assignment = receipt.selected.flatten()
                    counts.scatter_add_(
                        0, assignment, torch.ones_like(assignment, dtype=counts.dtype)
                    )
            values[name] = counts
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
            for receipt in self.receipts:
                if receipt.router is not router:
                    continue
                probabilities = router.balance_probabilities(
                    receipt.input, receipt.stage
                )
                contribution = (
                    probabilities * receipt.selected.shape[1]
                ).sum(0)
                soft_sum = contribution if soft_sum is None else soft_sum + contribution
            if soft_sum is None:
                continue
            loss = (coefficient * pool_scale * router.experts *
                    (fraction * soft_sum / a).sum())
            total_loss = loss if total_loss is None else total_loss + loss
            diagnostics[f"balance_{name}_loss"] = float(loss.detach())
        if total_loss is not None and coefficient:
            total_loss.backward()
        self.receipts.clear()
        return diagnostics

    def clear(self):
        self.receipts.clear()
