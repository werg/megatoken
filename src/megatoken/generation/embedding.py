"""Deterministic typed embedding R over *selected raw facets*, not C activations."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class EmbeddingSpec:
    space_fingerprint: str
    role: str
    dimension: int
    normalized: bool = True
    outputs: int = 1

    def __post_init__(self) -> None:
        if not self.space_fingerprint or self.role not in {"query", "document"}:
            raise ValueError("Embedding requires a pinned semantic space and asymmetrical role")
        if self.dimension < 1 or not self.normalized or self.outputs != 1:
            raise ValueError("The canonical embedding R returns one unit vector per event")


class EmbeddingRealizer(nn.Module):
    """One learned output slot sees every facet with its non-temporal identity."""

    def __init__(self, facets: int, width: int, dimension: int):
        super().__init__()
        if min(facets, width, dimension) < 1:
            raise ValueError("Invalid embedding readout shape")
        self.facets, self.width, self.dimension = facets, width, dimension
        self.norm = nn.LayerNorm(width)
        self.facet_codes = nn.Parameter(torch.randn(facets, width) * 0.02)
        self.role_slot = nn.Parameter(torch.randn(2, width) * 0.02)
        self.address = nn.Linear(width, width, bias=False)
        self.payload = nn.Linear(width, width, bias=False)
        self.query = nn.Linear(width, dimension, bias=False)
        self.document = nn.Linear(width, dimension, bias=False)

    def forward(self, raw: Tensor, spec: EmbeddingSpec) -> Tensor:
        if raw.ndim != 3 or raw.shape[1:] != (self.facets, self.width):
            raise ValueError("Embedding R must consume one complete native raw-facet block")
        pools = getattr(self.address, "_registry", None)
        if pools is None:
            return self._forward_impl(raw, spec)
        with pools.routing_epoch_if_absent(raw.float().mean(1), role="r"):
            return self._forward_impl(raw, spec)

    def _forward_impl(self, raw: Tensor, spec: EmbeddingSpec) -> Tensor:
        if raw.ndim != 3 or raw.shape[1:] != (self.facets, self.width):
            raise ValueError("Embedding R must consume one complete native raw-facet block")
        if spec.dimension != self.dimension:
            raise ValueError("Embedding dimension and pinned semantic space disagree")
        # No C, state.final, teacher or source-token alignment is accessed.
        role = 0 if spec.role == "query" else 1
        # Compute in the module's own dtype (bf16 models keep bf16 parameters).
        dtype = self.address.weight.dtype
        facets = self.norm(raw.to(dtype) + self.facet_codes[None].to(dtype))
        slot = self.role_slot[role].to(dtype)
        score = (self.address(facets) * slot[None, None]).sum(-1) / math.sqrt(self.width)
        pooled = torch.sum(score.softmax(-1)[..., None] * self.payload(facets), dim=1)
        projection = self.query if role == 0 else self.document
        return F.normalize(projection(pooled).float(), dim=-1, eps=1e-8)


def alignment_distance(output: Tensor, target: Tensor, center: Tensor) -> Tensor:
    """Per-row ``(1 - cos) + (1 - centered cos)`` to unit teacher targets.

    Raw cosine alone is nearly satisfied by the teacher's role mean in an
    anisotropic space; the centered term scores only what distinguishes this
    target from that mean, and the raw term pins how much of the mean to keep.
    """
    target = F.normalize(target.detach().float(), dim=-1)
    raw = (output * target).sum(-1)
    centered = F.cosine_similarity(output - center, target - center, dim=-1, eps=1e-8)
    return (1 - raw) + (1 - centered)
