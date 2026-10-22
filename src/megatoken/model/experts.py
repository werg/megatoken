"""Globally shared packed expert banks and transparent dispatch accounting."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from megatoken.kernels.grouped import grouped_linear


class ExpertBank(nn.Module):
    def __init__(
        self, experts: int, dim: int, hidden: int, backend: str = "torch", family: str = "expert"
    ) -> None:
        super().__init__()
        self.experts, self.dim, self.hidden, self.backend = experts, dim, hidden, backend
        self.family = family
        self.gate_up = nn.Parameter(torch.empty(experts, 2 * hidden, dim))
        self.down = nn.Parameter(torch.empty(experts, dim, hidden))
        nn.init.normal_(self.gate_up, std=1 / math.sqrt(dim))
        nn.init.normal_(self.down, std=1 / math.sqrt(hidden))
        self.register_buffer(
            "routed_rows", torch.zeros(experts, dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "learning_rows", torch.zeros(experts, dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "eligible_counts", torch.zeros(experts, dtype=torch.long), persistent=False
        )
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
            for name in (
                "routed_rows",
                "learning_rows",
                "eligible_counts",
                "padded_rows",
                "dispatches",
            )
        }
        if reset:
            for name in result:
                getattr(self, name).zero_()
        return result

    def forward(self, rows: Tensor, expert_ids: Tensor, confidence: Tensor) -> Tensor:
        if rows.shape[:-1] != expert_ids.shape or confidence.shape != expert_ids.shape:
            raise ValueError("Every expert row requires exactly one assignment and confidence")
        original_shape = rows.shape
        x = rows.reshape(-1, self.dim)
        ids = expert_ids.reshape(-1)
        with torch.profiler.record_function(f"fmt::experts/{self.family}"):
            gate, up = grouped_linear(x, self.gate_up, ids, self.backend).chunk(2, -1)
            y = grouped_linear(F.silu(gate) * up, self.down, ids, self.backend)
        return (y * confidence.reshape(-1, 1).to(y.dtype)).reshape(original_shape)
