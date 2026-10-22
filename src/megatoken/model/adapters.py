"""The knowledge-routed adapter mixture: small low-rank experts steered by each lookup.

C, P and R each own a bank of low-rank experts and a controller. After every
lookup the controllers read the lookup's blended knowledge value (the wide value
the knowledge event is read from) and keep, per owner, one preference over the
experts. That preference, scaled by how much the lookup actually found (one
minus the sink's share), is the *mixture*. It lives on the multitrack state, so
it applies to every meridian on every track until the next lookup replaces it;
before the first lookup there is none and nothing changes.

At each site (every C layer, every P stage, and the selected raw block before
realization) the controller adds a track and a stage preference to the
lookup's, picks the top experts with softmax weights, and adds their low-rank
update of the site's input to the site's output.

The router reads the full 32,768-wide value through one 8-way linear map
(262 thousand weights per owner) rather than through a 16-million-weight
projection to facet width first: the map is itself the learned projection, to
the only space the router needs, and the session carries just the 3 x 8
resulting preferences.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import RESERVED_TRACKS, ModelConfig

OWNERS = ("c", "p", "r")
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

    def delta(self, source: Tensor, site: int) -> Tensor:
        """The mixture's low-rank update of ``source`` at ``site``.

        ``source`` holds whole events: its leading dimensions flatten to
        [batch, events, rows] and its last is the facet width.
        """
        weights = self.weights[site]
        batch, events, _ = weights.shape
        rows = source.reshape(batch, events, -1, source.shape[-1])
        return self.owner.low_rank(rows, weights).reshape(source.shape)


class OwnerAdapters(nn.Module):
    """One owner's experts and the controller that picks among them."""

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
    ) -> None:
        super().__init__()
        self.width, self.experts, self.rank, self.topk = width, experts, rank, topk
        self.down = nn.Parameter(torch.randn(experts, rank, width) / math.sqrt(width))
        self.up = nn.Parameter(torch.randn(experts, rank, width) * 1e-3)
        self.router = nn.Linear(value_dim, experts, bias=False)
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

    def low_rank(self, x: Tensor, weights: Tensor) -> Tensor:
        """Mix every expert's rank-``rank`` update with ``weights`` (zero off the chosen ones).

        ``x`` is [batch, events, rows, width], ``weights`` [batch, events, experts].
        Every expert is computed densely, in one pair of matmuls: the experts x
        rank hidden features cost less than gathering each row's chosen factors.
        """
        down = self.down.reshape(-1, self.width).to(x.dtype)
        hidden = F.linear(x, down).unflatten(-1, (self.experts, self.rank))
        hidden = hidden * weights.to(x.dtype)[:, :, None, :, None]
        up = self.up.reshape(-1, self.width).to(x.dtype)
        return hidden.flatten(-2) @ up / self.rank


class AdapterBank(nn.Module):
    """The three owners' adapters, held by the core that owns the layers they steer."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        c = config
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
                )
                for name, sites in zip(
                    OWNERS, (c.central_layers, c.prediction_layers, 1), strict=True
                )
            }
        )

    @property
    def recording(self) -> bool:
        return any(owner.balance_recorder is not None for owner in self.owners.values())

    def select(self, value: Tensor, strength: Tensor) -> AdapterMixture:
        """The mixture of one lookup: ``value`` [value_dim], ``strength`` scalar in [0, 1]."""
        logits = torch.stack([self.owners[name].semantic(value) for name in OWNERS])
        context = value.detach()[None] if self.recording and torch.is_grad_enabled() else None
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

    def realization(
        self, mixture: AdapterMixture | None, track_ids: Tensor | None, raw: Tensor
    ) -> Tensor:
        """R's input: selected raw blocks [batch, events, facets, width] plus the update.

        Every realizer (text, audio, image, video) reads exactly this block, so
        one update here conditions them all.
        """
        if mixture is None:
            return raw
        if track_ids is None:
            raise ValueError("Realization adapters need the track of every event")
        return raw + self.plan("r", mixture, track_ids).delta(raw, 0)

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
