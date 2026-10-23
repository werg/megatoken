"""Causal retrieval controllers for adapters on every shared expert matrix.

The E/C/P/R owners keep small controllers, not residual adapter networks. Their
lookup-, track- and stage-conditioned sparse weights select the expert-specific
low-rank factors owned by the global shape pools. A session stores only these
preferences and never a modified copy of the model. Before the first lookup,
or for an all-sink lookup, the base experts run without an adapter correction.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import RESERVED_TRACKS, ModelConfig

OWNERS = ("e", "c", "p", "r")
# The chance a lookup's whole mixture is dropped in training, so the model stays
# usable without adapters (as it is before the first lookup).
ADAPTER_DROPOUT = 0.1


@dataclass(frozen=True)
class AdapterMixture:
    """What the latest lookup asked of the adapters; persists until the next lookup."""

    logits: Tensor  # [batch, owners, experts]: the lookup's preference, before track and stage
    strength: Tensor  # [batch]: one minus the share of the lookup the sink took
    # The detached lookup value, kept only while router balance is recorded.
    context: Tensor | None = None

    @property
    def batch_size(self) -> int:
        return self.logits.shape[0]

    def detach(self) -> AdapterMixture:
        return replace(self, logits=self.logits.detach(), strength=self.strength.detach())


@dataclass(frozen=True)
class AdapterRoute:
    """One executed routing decision, in the form the global router balance records."""

    expert_ids: Tensor  # [batch, topk]
    router_context: Tensor  # [batch, value_dim]
    router_stage: int  # the site's stage and the event's track, together


@dataclass(frozen=True)
class AdapterPlan:
    """One owner's expert weights at each of its sites, for the events of one call."""

    owner: OwnerAdapters
    weights: Tensor  # [sites, batch, events, experts]; zero off the chosen experts


class OwnerAdapters(nn.Module):
    """One execution role's controller for shared experts' adapter factors."""

    def __init__(
        self,
        *,
        width: int,
        value_dim: int,
        experts: int,
        rank: int,
        topk: int,
        max_tracks: int,
        sites: int,
        input_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.width, self.experts, self.rank, self.topk = width, experts, rank, topk
        # A compact shared-router feature fans out to role-local adapter heads.
        self.router = nn.Linear(input_dim or value_dim, experts, bias=False)
        # Rows 0..max_tracks-1 are user tracks, then the reserved system tracks.
        self.track_preference = nn.Embedding(max_tracks + len(RESERVED_TRACKS), experts)
        self.stage_preference = nn.Embedding(sites, experts)
        self.bias = nn.Parameter(torch.zeros(experts))
        nn.init.normal_(self.track_preference.weight, std=0.01)
        nn.init.normal_(self.stage_preference.weight, std=0.01)
        self.balance_recorder = None

    def semantic(self, value: Tensor) -> Tensor:
        """[..., value_dim] -> [..., experts] FP32 preference for one lookup value."""
        normalized = F.layer_norm(value.float(), value.shape[-1:], eps=1e-6)
        return self.router(normalized.to(self.router.weight.dtype)).float()

    def preference(self, semantic: Tensor, track: Tensor, site: Tensor | int) -> Tensor:
        """Expert logits at a (track, site): the lookup's preference plus the site's own."""
        stage = self.stage_preference.weight[site]
        return semantic + self.track_preference(track).float() + stage.float() + self.bias.float()

    def balance_probabilities(self, context: Tensor, key: int) -> Tensor:
        """The global router balance's replay: soft preferences for one (site, track)."""
        site, track = divmod(key, self.track_preference.num_embeddings)
        track = torch.full(context.shape[:1], track, device=context.device)
        return self.preference(self.semantic(context.detach()), track, site).softmax(-1)

    def record_executed(self, route: AdapterRoute) -> None:
        if self.balance_recorder is not None:
            self.balance_recorder.capture(self, route)


class AdapterBank(nn.Module):
    """The four causal routing controllers; factors live only in shared pools."""

    def __init__(self, config: ModelConfig, feature_provider=None) -> None:
        super().__init__()
        c = config
        object.__setattr__(self, "_feature_provider", feature_provider)
        self.owners = nn.ModuleDict(
            {
                name: OwnerAdapters(
                    width=c.facet_dim,
                    value_dim=c.transducer_value_rows * c.facet_dim,
                    experts=c.adapter_experts,
                    rank=c.adapter_rank,
                    topk=c.adapter_topk,
                    max_tracks=c.max_tracks,
                    sites=sites,
                    input_dim=c.router_dim,
                )
                for name, sites in zip(
                    OWNERS, (1, c.central_layers, c.prediction_layers, 1), strict=True
                )
            }
        )

    @property
    def recording(self) -> bool:
        return any(owner.balance_recorder is not None for owner in self.owners.values())

    def select(self, value: Tensor, strength: Tensor) -> AdapterMixture:
        """The mixture of one lookup: ``value`` [value_dim], ``strength`` scalar in [0, 1]."""
        provider = self._feature_provider
        if provider is None:
            # Standalone adapter banks remain constructible without a core.
            # Keep their controller input finite and compact even then.
            width = int(self.owners[OWNERS[0]].router.in_features)
            if value.shape[-1] < width:
                features = F.pad(value, (0, width - value.shape[-1]))
            else:
                pieces = (value.shape[-1] + width - 1) // width
                features = F.pad(value, (0, pieces * width - value.shape[-1]))
                features = features.reshape(-1, pieces, width).mean(-2)
            features = F.layer_norm(features.float(), (width,), eps=1e-6)
            features = features.reshape(-1, width)
        else:
            # Compute this pooled lookup feature once, then fan it out to the
            # four role-local heads. No adapter mixture is active at lookup
            # time, avoiding recursion through the adapter factors.
            features = provider.controller_features(
                value.reshape(1, -1), role="other", stage=0, adapter_weights=None
            )
        logits = torch.stack([self.owners[name].semantic(features).squeeze(0) for name in OWNERS])
        context = features.detach() if self.recording and torch.is_grad_enabled() else None
        return AdapterMixture(logits[None], strength.float().reshape(1), context)

    def plan(
        self, name: str, mixture: AdapterMixture | None, track_ids: Tensor
    ) -> AdapterPlan | None:
        """Route one owner's sites for the events of ``track_ids`` [batch, events].

        Routing is decided here, outside any checkpointed computation, so replay
        never counts as exposure for the global router balance.
        """
        if mixture is None:
            return None
        owner = self.owners[name]
        if track_ids.shape[0] != mixture.batch_size:
            raise ValueError("Every adapter event needs a mixture of its own session")
        semantic = mixture.logits[:, OWNERS.index(name)]  # [batch, experts]
        stages = torch.arange(owner.stage_preference.num_embeddings, device=semantic.device)
        logits = owner.preference(
            semantic[None, :, None], track_ids[None], stages[:, None, None]
        )  # [sites, batch, events, experts]
        top = logits.topk(owner.topk, dim=-1)
        chosen = top.values.softmax(-1)
        weights = torch.zeros_like(logits).scatter(-1, top.indices, chosen)
        weights = weights * mixture.strength[None, :, None, None]
        if mixture.context is not None and owner.balance_recorder is not None:
            self._record(owner, mixture, track_ids, top.indices)
        return AdapterPlan(owner, weights)

    @staticmethod
    def _record(
        owner: OwnerAdapters, mixture: AdapterMixture, track_ids: Tensor, chosen: Tensor
    ) -> None:
        tracks = owner.track_preference.num_embeddings
        for event, track in enumerate(track_ids[0].tolist()):
            for site in range(chosen.shape[0]):
                owner.record_executed(
                    AdapterRoute(chosen[site, :, event], mixture.context, site * tracks + track)
                )
