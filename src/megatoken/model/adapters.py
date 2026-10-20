"""Independent, context-wide sparse LoRA factor banks shared across native tracks."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from megatoken.memory.semantic import MemoryMaterialization

MODULES = ("c", "p", "r")


class LowRankFamily(nn.Module):
    def __init__(self, width: int, experts: int, rank: int):
        super().__init__()
        self.a = nn.Parameter(torch.randn(experts, rank, width) * (1 / math.sqrt(width)))
        self.b = nn.Parameter(torch.randn(experts, width, rank) * 1e-3)
        self.rank = rank

    def forward(self, x: Tensor, ids: Tensor, weights: Tensor) -> Tensor:
        # All factors have their own owner; expert ID is not a base-MoE pair index.
        selected_a = self.a[ids].to(x.dtype)
        selected_b = self.b[ids].to(x.dtype)
        hidden = torch.einsum("bnd,bnkrd->bnkr", x, selected_a)
        result = torch.einsum("bnkr,bnkdr->bnkd", hidden, selected_b)
        return (result * weights[..., None]).sum(-2) / self.rank


class ExpertController(nn.Module):
    """One module's expert selection: semantic router plus track/stage/module bias."""

    def __init__(self, *, value_dim: int, experts: int, max_tracks: int, max_stage: int):
        super().__init__()
        self.semantic_norm = nn.LayerNorm(value_dim)
        self.router = nn.Linear(value_dim, experts, bias=False)
        # Rows 0..max_tracks-1 are user tracks; row max_tracks is the reserved
        # retrieval-query track; the final row is overflow slack.
        self.track_preference = nn.Embedding(max_tracks + 2, experts)
        self.stage_preference = nn.Embedding(max_stage + 1, experts)
        self.bias = nn.Parameter(torch.zeros(experts))
        nn.init.normal_(self.track_preference.weight, std=0.01)
        nn.init.normal_(self.stage_preference.weight, std=0.01)

    def forward(self, semantics: Tensor, tracks: Tensor, stage: int) -> Tensor:
        preference = self.track_preference(tracks) + self.stage_preference.weight[stage] + self.bias
        semantics = semantics.to(self.router.weight.dtype)
        return self.router(self.semantic_norm(semantics))[:, None, :] + preference


class MemoryAdapters(nn.Module):
    """Snapshot controls are functions of the same semantic groups as complete KV.

    C, P and R each own their LoRA factors AND their expert controller. They read
    the same semantic snapshot, but no selection parameter is shared, so pinning
    C/R for latent RL never freezes how P uses memory.
    """

    def __init__(
        self,
        *,
        width: int,
        value_dim: int,
        experts: int = 4,
        rank: int = 8,
        topk: int = 2,
        max_tracks: int = 8,
        max_stage: int = 32,
    ):
        super().__init__()
        if min(width, value_dim, experts, rank, topk) <= 0 or topk > experts:
            raise ValueError("Invalid LoRA expert/rank/selection budget")
        self.width, self.experts, self.topk = width, experts, topk
        self.families = nn.ModuleDict(
            {name: LowRankFamily(width, experts, rank) for name in MODULES}
        )
        self.controllers = nn.ModuleDict(
            {
                name: ExpertController(
                    value_dim=value_dim,
                    experts=experts,
                    max_tracks=max_tracks,
                    max_stage=max_stage,
                )
                for name in MODULES
            }
        )

    def module_parameters(self, module: str):
        """Factors and expert controller owned by exactly one of C, P or R."""
        yield from self.families[module].parameters()
        yield from self.controllers[module].parameters()

    def apply(
        self,
        x: Tensor,
        *,
        module: str,
        stage: int,
        track_ids: Tensor | None,
        snapshot: MemoryMaterialization | None,
    ) -> Tensor:
        if snapshot is None or snapshot.strengths.shape[1] == 0:
            return x
        if module not in self.families or x.ndim != 3 or x.shape[-1] != self.width:
            raise ValueError("Adapter requires one declared model family and native rows")
        controller = self.controllers[module]
        if not 0 <= stage < controller.stage_preference.num_embeddings:
            raise ValueError("Adapter stage lies outside registered control geometry")
        batch, length, _ = x.shape
        if track_ids is None:
            tracks = torch.zeros(batch, length, device=x.device, dtype=torch.long)
        else:
            if track_ids.shape != (batch, length):
                raise ValueError("Adapter control must identify every flattened track row")
            tracks = track_ids.clamp_min(0).clamp_max(
                controller.track_preference.num_embeddings - 1
            )
        delta = torch.zeros_like(x)
        for group in range(snapshot.strengths.shape[1]):
            strength = snapshot.strengths[:, group]
            logits = controller(snapshot.semantics[:, group], tracks, stage)
            chosen = torch.topk(logits, self.topk, dim=-1).indices
            probabilities = torch.softmax(logits, -1).gather(-1, chosen)
            probabilities = probabilities / probabilities.sum(-1, keepdim=True).clamp_min(1e-8)
            delta = delta + self.families[module](x, chosen, probabilities.to(x.dtype)) * strength[
                :, None, None
            ].to(x.dtype)
        # Exactly zero strength never activates the adapters, even with biases.
        return x + delta
