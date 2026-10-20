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


class MemoryAdapters(nn.Module):
    """Snapshot controls are functions of the same semantic groups as complete KV."""

    def __init__(self, *, width: int, value_dim: int, experts: int = 4,
                 rank: int = 8, topk: int = 2, max_tracks: int = 8,
                 max_stage: int = 32):
        super().__init__()
        if min(width, value_dim, experts, rank, topk) <= 0 or topk > experts:
            raise ValueError("Invalid LoRA expert/rank/selection budget")
        self.width, self.experts, self.topk = width, experts, topk
        self.families = nn.ModuleDict(
            {name: LowRankFamily(width, experts, rank) for name in MODULES}
        )
        self.semantic_norm = nn.LayerNorm(value_dim)
        self.controller = nn.Linear(value_dim, experts, bias=False)
        # Rows 0..max_tracks-1 are user tracks; row max_tracks is the reserved
        # retrieval-query track; the final row is overflow slack.
        self.track_preference = nn.Embedding(max_tracks + 2, experts)
        self.stage_preference = nn.Embedding(max_stage + 1, experts)
        # One bias per owning module so each family's selection bias has one owner.
        self.module_preference = nn.ParameterDict(
            {name: nn.Parameter(torch.zeros(experts)) for name in MODULES}
        )
        nn.init.normal_(self.track_preference.weight, std=0.01)
        nn.init.normal_(self.stage_preference.weight, std=0.01)

    def module_parameters(self, module: str):
        """Factors and selection bias owned by exactly one of C, P or R."""
        yield from self.families[module].parameters()
        yield self.module_preference[module]

    def shared_control_parameters(self):
        """The one controller path every module reads; owned with C's pinned map."""
        for module in (self.semantic_norm, self.controller,
                       self.track_preference, self.stage_preference):
            yield from module.parameters()

    def apply(self, x: Tensor, *, module: str, stage: int,
              track_ids: Tensor | None, snapshot: MemoryMaterialization | None) -> Tensor:
        if snapshot is None or snapshot.strengths.shape[1] == 0:
            return x
        if module not in self.families or x.ndim != 3 or x.shape[-1] != self.width:
            raise ValueError("Adapter requires one declared model family and native rows")
        if not 0 <= stage < self.stage_preference.num_embeddings:
            raise ValueError("Adapter stage lies outside registered control geometry")
        batch, length, _ = x.shape
        if track_ids is None:
            tracks = torch.zeros(batch, length, device=x.device, dtype=torch.long)
        else:
            if track_ids.shape != (batch, length):
                raise ValueError("Adapter control must identify every flattened track row")
            tracks = track_ids.clamp_min(0).clamp_max(self.track_preference.num_embeddings - 1)
        bias = (self.track_preference(tracks) +
                self.stage_preference.weight[stage] +
                self.module_preference[module])
        delta = torch.zeros_like(x)
        for group in range(snapshot.strengths.shape[1]):
            strength = snapshot.strengths[:, group]
            aggregate = snapshot.semantics[:, group]
            logits = self.controller(self.semantic_norm(aggregate))[:, None, :] + bias
            chosen = torch.topk(logits, self.topk, dim=-1).indices
            probabilities = torch.softmax(logits, -1).gather(-1, chosen)
            probabilities = probabilities / probabilities.sum(-1, keepdim=True).clamp_min(1e-8)
            delta = delta + self.families[module](
                x, chosen, probabilities.to(x.dtype)
            ) * strength[:, None, None].to(x.dtype)
        # Exactly zero strength never activates the adapters, even with biases.
        return x + delta
