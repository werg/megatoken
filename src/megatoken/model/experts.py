"""Globally shared packed expert banks and transparent dispatch accounting."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from megatoken.kernels.grouped import grouped_linear


def _adapter_delta(
    rows: Tensor,
    expert_ids: Tensor,
    adapter_weights: Tensor | None,
    down: Tensor,
    up: Tensor,
    *,
    experts: int,
    adapter_experts: int,
    adapter_topk: int,
    backend: str,
) -> Tensor | None:
    """Evaluate selected per-base-expert low-rank updates without expanding weights.

    ``down`` and ``up`` are ``[base_experts, adapters, rank, in]`` and
    ``[base_experts, adapters, out, rank]``. Only nonzero mixture entries are
    dispatched, grouped by the flattened ``(base expert, adapter)`` identity.
    """
    if adapter_weights is None:
        return None
    if adapter_weights.shape != (*rows.shape[:-1], adapter_experts):
        raise ValueError("Adapter weights must match row dimensions and adapter count")
    if adapter_weights.device != rows.device:
        raise ValueError("Adapter weights and rows must be on the same device")
    flat_rows = rows.reshape(-1, rows.shape[-1])
    flat_ids = expert_ids.reshape(-1)
    weights = adapter_weights.reshape(-1, adapter_experts)
    if weights.shape[0] != flat_rows.shape[0]:
        raise ValueError("Each row must have one adapter mixture")
    count = min(adapter_topk, adapter_experts)
    selected_weight, selected_adapter = weights.topk(count, dim=-1)
    active = selected_weight != 0
    row_index, slot_index = active.nonzero(as_tuple=True)
    if row_index.numel() == 0:
        return None
    selected_adapter = selected_adapter[row_index, slot_index]
    selected_weight = selected_weight[row_index, slot_index]
    factor_ids = flat_ids[row_index] * adapter_experts + selected_adapter
    factor_down = down.reshape(experts * adapter_experts, down.shape[-2], down.shape[-1])
    factor_up = up.reshape(experts * adapter_experts, up.shape[-2], up.shape[-1])
    selected_rows = flat_rows[row_index]
    hidden = grouped_linear(selected_rows, factor_down, factor_ids, backend)
    output = grouped_linear(hidden, factor_up, factor_ids, backend)
    # Place the compact dispatches back in fixed row/slot order, then reduce.
    # This avoids a repeated-destination index_add in deterministic training.
    slots = flat_rows.shape[0] * count
    flat_slot = row_index * count + slot_index
    weighted_output = output * selected_weight[:, None].to(output.dtype)
    scattered = output.new_zeros((slots, output.shape[-1])).index_copy(0, flat_slot, weighted_output)
    mixed = scattered.view(flat_rows.shape[0], count, -1).sum(1)
    return mixed.reshape(*rows.shape[:-1], output.shape[-1])


class _RouteAccounting(nn.Module):
    """Shared transparent dispatch counters for linear expert banks."""

    def _init_accounting(self, experts: int) -> None:
        self.register_buffer("routed_rows", torch.zeros(experts, dtype=torch.long), persistent=False)
        self.register_buffer("learning_rows", torch.zeros(experts, dtype=torch.long), persistent=False)
        self.register_buffer("eligible_counts", torch.zeros(experts, dtype=torch.long), persistent=False)
        self.register_buffer("padded_rows", torch.zeros((), dtype=torch.long), persistent=False)
        self.register_buffer("dispatches", torch.zeros((), dtype=torch.long), persistent=False)

    @torch.no_grad()
    def record_routes(
        self,
        expert_ids: Tensor,
        eligible: Tensor | None = None,
        block_rows: int = 32,
        *,
        gradient_enabled: bool = False,
    ) -> None:
        """Called outside checkpointed transforms, so replay is not learning exposure."""
        ids = expert_ids.detach().flatten()
        counts = torch.zeros_like(self.routed_rows).scatter_add_(0, ids, torch.ones_like(ids))
        self.routed_rows.add_(counts)
        if gradient_enabled:
            self.learning_rows.add_(counts)
        if eligible is not None:
            palette = eligible.detach().flatten()
            self.eligible_counts.scatter_add_(0, palette, torch.ones_like(palette))
        self.padded_rows.add_(((counts + block_rows - 1) // block_rows * block_rows).sum())
        self.dispatches.add_(1)

    @torch.no_grad()
    def metrics(self, *, reset: bool = False) -> dict[str, Tensor]:
        result = {
            name: getattr(self, name).detach().clone()
            for name in ("routed_rows", "learning_rows", "eligible_counts", "padded_rows", "dispatches")
        }
        if reset:
            for name in result:
                getattr(self, name).zero_()
        return result


class ExpertBank(_RouteAccounting):
    def __init__(
        self,
        experts: int,
        dim: int,
        hidden: int,
        backend: str = "torch",
        family: str = "expert",
        adapter_experts: int = 8,
        adapter_rank: int = 8,
        adapter_topk: int = 2,
    ) -> None:
        super().__init__()
        if min(experts, dim, hidden, adapter_experts, adapter_rank, adapter_topk) < 1:
            raise ValueError("Expert and adapter dimensions must be positive")
        self.experts, self.dim, self.hidden, self.backend = experts, dim, hidden, backend
        self.family = family
        self.adapter_experts, self.adapter_rank, self.adapter_topk = (
            adapter_experts,
            adapter_rank,
            min(adapter_topk, adapter_experts),
        )
        self.gate_up = nn.Parameter(torch.empty(experts, 2 * hidden, dim))
        self.down = nn.Parameter(torch.empty(experts, dim, hidden))
        nn.init.normal_(self.gate_up, std=1 / math.sqrt(dim))
        nn.init.normal_(self.down, std=1 / math.sqrt(hidden))
        self.gate_up_adapter_down = nn.Parameter(
            torch.randn(experts, adapter_experts, adapter_rank, dim) / math.sqrt(dim)
        )
        self.gate_up_adapter_up = nn.Parameter(torch.zeros(experts, adapter_experts, 2 * hidden, adapter_rank))
        self.down_adapter_down = nn.Parameter(
            torch.randn(experts, adapter_experts, adapter_rank, hidden) / math.sqrt(hidden)
        )
        self.down_adapter_up = nn.Parameter(torch.zeros(experts, adapter_experts, dim, adapter_rank))
        self._init_accounting(experts)

    def forward(
        self,
        rows: Tensor,
        expert_ids: Tensor,
        confidence: Tensor,
        adapter_weights: Tensor | None = None,
    ) -> Tensor:
        if rows.shape[:-1] != expert_ids.shape or confidence.shape != expert_ids.shape:
            raise ValueError("Every expert row requires exactly one assignment and confidence")
        if rows.shape[-1] != self.dim:
            raise ValueError("Expert input dimension mismatch")
        if adapter_weights is not None and adapter_weights.shape != (
            *rows.shape[:-1], self.adapter_experts
        ):
            raise ValueError("Adapter weights must match row dimensions and adapter count")
        original_shape = rows.shape
        x = rows.reshape(-1, self.dim)
        ids = expert_ids.reshape(-1)
        weights = None if adapter_weights is None else adapter_weights.reshape(-1, self.adapter_experts)
        with torch.profiler.record_function(f"fmt::experts/{self.family}"):
            gate_up = grouped_linear(x, self.gate_up, ids, self.backend)
            gate_up_delta = _adapter_delta(
                x, ids, weights, self.gate_up_adapter_down, self.gate_up_adapter_up,
                experts=self.experts, adapter_experts=self.adapter_experts,
                adapter_topk=self.adapter_topk, backend=self.backend,
            )
            if gate_up_delta is not None:
                gate_up = gate_up + gate_up_delta.reshape_as(gate_up)
            gate, up = gate_up.chunk(2, -1)
            hidden = F.silu(gate) * up
            output = grouped_linear(hidden, self.down, ids, self.backend)
            down_delta = _adapter_delta(
                hidden, ids, weights, self.down_adapter_down, self.down_adapter_up,
                experts=self.experts, adapter_experts=self.adapter_experts,
                adapter_topk=self.adapter_topk, backend=self.backend,
            )
            if down_delta is not None:
                output = output + down_delta.reshape_as(output)
        return (output * confidence.reshape(-1, 1).to(output.dtype)).reshape(original_shape)


class LinearExpertBank(_RouteAccounting):
    """A routed linear bank with expert-specific low-rank matrix adapters."""

    def __init__(
        self,
        experts: int,
        dim: int,
        output_dim: int,
        backend: str = "torch",
        family: str = "linear",
        adapter_experts: int = 8,
        adapter_rank: int = 8,
        adapter_topk: int = 2,
    ) -> None:
        super().__init__()
        if min(experts, dim, output_dim, adapter_experts, adapter_rank, adapter_topk) < 1:
            raise ValueError("Expert and adapter dimensions must be positive")
        self.experts, self.dim, self.output_dim, self.backend = experts, dim, output_dim, backend
        self.family = family
        self.adapter_experts, self.adapter_rank, self.adapter_topk = (
            adapter_experts,
            adapter_rank,
            min(adapter_topk, adapter_experts),
        )
        self.weight = nn.Parameter(torch.empty(experts, output_dim, dim))
        nn.init.normal_(self.weight, std=1 / math.sqrt(dim))
        self.adapter_down = nn.Parameter(
            torch.randn(experts, adapter_experts, adapter_rank, dim) / math.sqrt(dim)
        )
        self.adapter_up = nn.Parameter(torch.zeros(experts, adapter_experts, output_dim, adapter_rank))
        self._init_accounting(experts)

    def forward(
        self,
        rows: Tensor,
        expert_ids: Tensor,
        confidence: Tensor,
        adapter_weights: Tensor | None = None,
    ) -> Tensor:
        if rows.shape[:-1] != expert_ids.shape or confidence.shape != expert_ids.shape:
            raise ValueError("Every expert row requires exactly one assignment and confidence")
        if rows.shape[-1] != self.dim:
            raise ValueError("Expert input dimension mismatch")
        if adapter_weights is not None and adapter_weights.shape != (
            *rows.shape[:-1], self.adapter_experts
        ):
            raise ValueError("Adapter weights must match row dimensions and adapter count")
        original_shape = rows.shape
        x = rows.reshape(-1, self.dim)
        ids = expert_ids.reshape(-1)
        weights = None if adapter_weights is None else adapter_weights.reshape(-1, self.adapter_experts)
        with torch.profiler.record_function(f"fmt::experts/{self.family}"):
            output = grouped_linear(x, self.weight, ids, self.backend)
            delta = _adapter_delta(
                x, ids, weights, self.adapter_down, self.adapter_up,
                experts=self.experts, adapter_experts=self.adapter_experts,
                adapter_topk=self.adapter_topk, backend=self.backend,
            )
            if delta is not None:
                output = output + delta.reshape_as(output)
        return (output * confidence.reshape(-1, 1).to(output.dtype)).reshape(
            *original_shape[:-1], self.output_dim
        )
