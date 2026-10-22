"""Epoch-compiled, batch-independent routing with real confidence gradients."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .layers import RMSNorm, position_features


@dataclass(frozen=True)
class RowRoute:
    expert_ids: Tensor
    confidence: Tensor
    eligible: Tensor
    palette_probabilities: Tensor
    balance_loss: Tensor
    stabilization_loss: Tensor
    router_context: Tensor | None = None
    router_stage: int = 0


@dataclass(frozen=True)
class RoutingProgram:
    epoch: str
    version: int
    context: Tensor
    stages: tuple[int, ...]
    routes: tuple[RowRoute, ...]

    def at(self, stage: int) -> RowRoute:
        try:
            return self.routes[self.stages.index(stage)]
        except ValueError as exc:
            raise ValueError(f"Stage {stage} is outside the {self.epoch!r} routing epoch") from exc

    @property
    def balance_loss(self) -> Tensor:
        return torch.stack([route.balance_loss for route in self.routes]).mean()

    @property
    def stabilization_loss(self) -> Tensor:
        return torch.stack([route.stabilization_loss for route in self.routes]).mean()


class EpochRouter(nn.Module):
    """Select a small palette, then one expert per fixed row descriptor.

    Only ``context`` at an allowed epoch and fixed stage/row descriptors affect
    routes. Evolving pair activations and batch membership never enter selection.
    Softmax is evaluated BEFORE hard selection, for both palette and row gates.
    Excluded experts get balance gradients, but no fabricated task gradients.
    """

    def __init__(
        self,
        input_dim: int,
        router_dim: int,
        experts: int,
        max_eligible: int,
        *,
        version: int = 1,
    ) -> None:
        super().__init__()
        self.experts = experts
        self.router_dim = router_dim
        self.max_eligible = min(max_eligible, experts)
        self.version = version
        self.norm = RMSNorm(input_dim)
        self.context_projection = nn.Linear(input_dim, router_dim, bias=False)
        self.palette_codes = nn.Parameter(torch.empty(experts, router_dim))
        self.row_codes = nn.Parameter(torch.empty(experts, router_dim))
        nn.init.normal_(self.palette_codes, std=0.02)
        nn.init.normal_(self.row_codes, std=0.02)
        self.balance_recorder = None

    def balance_probabilities(self, context: Tensor, stage: int) -> Tensor:
        """Replay only the full-bank palette soft probabilities; no dispatch."""
        feature = self.context_projection(self.norm(context.detach())).float()
        descriptor = position_features(1, self.router_dim, feature.device, offset=stage)[0]
        logits = F.linear(feature + descriptor, self.palette_codes.float())
        return (logits / math.sqrt(self.router_dim)).softmax(-1)

    def record_executed(self, route: RowRoute) -> None:
        if self.balance_recorder is not None:
            self.balance_recorder.capture(self, route)

    def route(
        self,
        context: Tensor,
        stage: int,
        rows: int,
        *,
        eligible_limit: int | None = None,
    ) -> RowRoute:
        if context.ndim != 2 or rows < 1:
            raise ValueError("Routing requires context[batch,dim] and a positive row count")
        eligible_count = min(self.max_eligible, eligible_limit or self.max_eligible, rows)
        feature = self.context_projection(self.norm(context)).float()
        descriptor = position_features(1, self.router_dim, feature.device, offset=stage)[0]
        feature = feature + descriptor
        palette_logits = F.linear(feature, self.palette_codes.float()) / math.sqrt(self.router_dim)
        eligible = torch.argsort(palette_logits, dim=-1, descending=True, stable=True)[
            :, :eligible_count
        ]
        palette_probability = palette_logits.softmax(-1)
        selected_palette = palette_probability.gather(-1, eligible)
        row_positions = position_features(rows, self.router_dim, feature.device, offset=stage * 257)
        queries = feature[:, None, :] + row_positions[None]
        selected_codes = self.row_codes.float()[eligible]
        row_logits = torch.einsum("bnr,bkr->bnk", queries, selected_codes) / math.sqrt(
            self.router_dim
        )
        row_probability = row_logits.softmax(-1)
        local_choice = row_logits.argmax(-1)
        ids = eligible.gather(-1, local_choice)
        row_confidence = row_probability.gather(-1, local_choice[..., None]).squeeze(-1)
        palette_confidence = selected_palette.gather(-1, local_choice)
        # Normalize around unit scale under uniform routing, but strictly bound
        # confident palettes. The previous unbounded product could amplify one
        # routed expert by experts*eligible (512x in production). Keep a smooth,
        # nonzero gradient through both selected palette and row probabilities.
        unbounded = palette_confidence * self.experts * row_confidence * eligible_count
        confidence = 2.0 * unbounded / (1.0 + unbounded)
        assignments = ids.detach().flatten()
        fractions = torch.zeros(self.experts, device=ids.device, dtype=torch.float32)
        fractions.scatter_add_(0, assignments, torch.ones_like(assignments, dtype=torch.float32))
        fractions = fractions / assignments.numel()
        balance = self.experts * (palette_probability.mean(0) * fractions).sum()
        stabilization = (
            palette_logits.logsumexp(-1).square().mean() + row_logits.logsumexp(-1).square().mean()
        )
        return RowRoute(
            ids,
            confidence,
            eligible,
            palette_probability,
            balance,
            stabilization,
            context.detach() if self.balance_recorder is not None else None,
            stage,
        )

    def compile(
        self,
        context: Tensor,
        stages: tuple[int, ...],
        rows: int,
        *,
        epoch: str,
    ) -> RoutingProgram:
        if not stages or len(set(stages)) != len(stages):
            raise ValueError("A routing epoch needs distinct stages")
        routes = tuple(self.route(context, stage, rows) for stage in stages)
        return RoutingProgram(epoch, self.version, context, stages, routes)
