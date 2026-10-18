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

    def prefix(self, rows: int) -> RowRoute:
        if rows > self.expert_ids.shape[1]:
            raise ValueError("Route program has too few compiled rows")
        return RowRoute(
            self.expert_ids[:, :rows],
            self.confidence[:, :rows],
            self.eligible,
            self.palette_probabilities,
            self.balance_loss,
            self.stabilization_loss,
        )


@dataclass(frozen=True)
class RoutingProgram:
    epoch: str
    version: int
    context: Tensor
    stages: tuple[int, ...]
    routes: tuple[RowRoute, ...]

    def at(self, stage: int, rows: int | None = None) -> RowRoute:
        try:
            route = self.routes[self.stages.index(stage)]
        except ValueError as exc:
            raise ValueError(f"Stage {stage} is outside the {self.epoch!r} routing epoch") from exc
        return route if rows is None else route.prefix(rows)

    @property
    def balance_loss(self) -> Tensor:
        return torch.stack([route.balance_loss for route in self.routes]).mean()

    @property
    def stabilization_loss(self) -> Tensor:
        return torch.stack([route.stabilization_loss for route in self.routes]).mean()

    @property
    def aux_loss(self) -> Tensor:
        return self.balance_loss * 0.005 + self.stabilization_loss * 1e-4


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
        exploration: float = 0.0,
        version: int = 1,
    ) -> None:
        super().__init__()
        self.experts = experts
        self.router_dim = router_dim
        self.max_eligible = min(max_eligible, experts)
        self.exploration = exploration
        self.version = version
        self.norm = RMSNorm(input_dim)
        self.context_projection = nn.Linear(input_dim, router_dim, bias=False)
        self.palette_codes = nn.Parameter(torch.empty(experts, router_dim))
        self.row_codes = nn.Parameter(torch.empty(experts, router_dim))
        nn.init.normal_(self.palette_codes, std=0.02)
        nn.init.normal_(self.row_codes, std=0.02)

    def route(
        self,
        context: Tensor,
        stage: int,
        rows: int,
        *,
        eligible_limit: int | None = None,
        record_ids: Tensor | None = None,
        exploration_seed: int = 0,
        allow_exploration: bool = True,
    ) -> RowRoute:
        if context.ndim != 2 or rows < 1:
            raise ValueError("Routing requires context[batch,dim] and a positive row count")
        eligible_count = min(self.max_eligible, eligible_limit or self.max_eligible, rows)
        feature = self.context_projection(self.norm(context)).float()
        descriptor = position_features(1, self.router_dim, feature.device, offset=stage)[0]
        feature = feature + descriptor
        palette_logits = F.linear(feature, self.palette_codes.float()) / math.sqrt(self.router_dim)
        selection_logits = palette_logits
        if self.training and self.exploration and allow_exploration:
            if record_ids is None or record_ids.shape != context.shape[:1]:
                raise ValueError("Training exploration requires stable per-episode record IDs")
            # Bounded stateless exploration, identical for an episode in any batch.
            identity = record_ids[:, None].double()
            expert = torch.arange(self.experts, device=feature.device)[None].double()
            perturbation = torch.sin(
                identity * 12.9898 + expert * 78.233 + stage * 3.19 + exploration_seed
            )
            selection_logits = palette_logits + perturbation.float() * self.exploration
        eligible = torch.argsort(selection_logits, dim=-1, descending=True, stable=True)[
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
        return RowRoute(ids, confidence, eligible, palette_probability, balance, stabilization)

    def compile(
        self,
        context: Tensor,
        stages: tuple[int, ...],
        rows: int,
        *,
        epoch: str,
        record_ids: Tensor | None = None,
        exploration_seed: int = 0,
        allow_exploration: bool = True,
    ) -> RoutingProgram:
        if not stages or len(set(stages)) != len(stages):
            raise ValueError("A routing epoch needs distinct stages")
        routes = tuple(
            self.route(
                context,
                stage,
                rows,
                record_ids=record_ids,
                exploration_seed=exploration_seed,
                allow_exploration=allow_exploration,
            )
            for stage in stages
        )
        return RoutingProgram(epoch, self.version, context, stages, routes)


def expert_gradient_statistics(bank: nn.Module) -> dict[str, Tensor]:
    """Report task-gradient slices, not just parameter movement under AdamW."""
    result: dict[str, Tensor] = {}
    for name in ("gate_up", "down"):
        parameter = getattr(bank, name)
        grad = parameter.grad
        result[f"{name}_gradient_norm"] = (
            torch.zeros(parameter.shape[0], device=parameter.device)
            if grad is None
            else grad.detach().float().flatten(1).norm(dim=-1)
        )
    result["nonzero_gradient"] = (
        result["gate_up_gradient_norm"] + result["down_gradient_norm"]
    ) > 0
    return result
