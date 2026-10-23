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
        # Bound by FacetedCore to its shared router expert registry. Keep a
        # private projection only for standalone EpochRouter use in small unit
        # tests and compatibility callers.
        self._feature_provider = None
        self._routing_role = "other"
        self._context_projection = nn.Linear(input_dim, router_dim, bias=False)
        self.palette_codes = nn.Parameter(torch.empty(experts, router_dim))
        self.row_codes = nn.Parameter(torch.empty(experts, router_dim))
        nn.init.normal_(self.palette_codes, std=0.02)
        nn.init.normal_(self.row_codes, std=0.02)
        self.balance_recorder = None

    def balance_probabilities(self, context: Tensor, stage: int) -> Tensor:
        """Replay only the full-bank palette soft probabilities; no dispatch."""
        # Captures store the compact feature emitted by the shared router
        # experts, so fairness replay never repeats or differentiates through
        # that shared bootstrap computation.
        feature = context.detach().float()
        descriptor = position_features(1, self.router_dim, feature.device, offset=stage)[0]
        logits = F.linear(feature + descriptor, self.palette_codes.float())
        return (logits / math.sqrt(self.router_dim)).softmax(-1)

    @property
    def context_projection(self):
        """Compatibility view of the shared router bank's parameter dtype."""
        provider = getattr(self, "_feature_provider", None)
        if provider is not None:
            return _ProjectionCompatibilityView(provider.router_bank.gate_up)
        return self._context_projection

    def bind_feature_provider(self, provider, *, role: str) -> None:
        if role not in {"P", "U", "other"}:
            raise ValueError("EpochRouter role must be P, U, or other")
        if int(provider.router_dim) != self.router_dim:
            raise ValueError("EpochRouter and shared router width disagree")
        object.__setattr__(self, "_feature_provider", provider)
        self._routing_role = role
        self._context_projection = None
        self.norm = None

    def _features(self, context: Tensor) -> Tensor:
        provider = self._feature_provider
        if provider is not None:
            role = "R" if self._routing_role == "U" else self._routing_role
            return provider.controller_features(
                context, role=role, stage=0, reuse_active=True
            ).float()
        if self._context_projection is None or self.norm is None:
            raise RuntimeError("Standalone EpochRouter has no context projection")
        return self._context_projection(self.norm(context)).float()

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
        feature = self._features(context)
        return self._route_feature(feature, stage, rows, eligible_limit=eligible_limit)

    def _route_feature(
        self,
        feature: Tensor,
        stage: int,
        rows: int,
        *,
        eligible_limit: int | None = None,
    ) -> RowRoute:
        if feature.ndim != 2 or rows < 1:
            raise ValueError("Routing features must be [batch,router_dim] and rows positive")
        eligible_count = min(self.max_eligible, eligible_limit or self.max_eligible, rows)
        base_feature = feature
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
            base_feature.detach() if self.balance_recorder is not None else None,
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
        feature = self._features(context)
        routes = tuple(self._route_feature(feature, stage, rows) for stage in stages)
        return RoutingProgram(epoch, self.version, context, stages, routes)


class _ProjectionCompatibilityView:
    """Read-only `.weight` compatibility for callers that inspect dtype."""

    def __init__(self, weight: Tensor):
        self.weight = weight
