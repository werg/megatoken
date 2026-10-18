"""Shared residual attention/FFN blocks with explicit parameter ownership."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .attention import AttentionFeatures, AttentionRead, KVProjection, join_attention
from .config import ModelConfig
from .layers import RMSNorm, SwiGLU


class DenseFacetBlock(nn.Module):
    """The same dense block serves local encoding and contextual central work.

    A local encoder block owns its KV projection. A central block receives the
    central processor's shared projection explicitly, so registering the block
    never duplicates or aliases that parameter owner.
    """

    def __init__(self, config: ModelConfig, hidden: int, *, shared_kv: bool = False) -> None:
        super().__init__()
        c = config
        self.attention = AttentionRead(
            c.facet_dim, c.query_heads, c.kv_heads, c.attention_head_dim, c.norm_eps
        )
        self.kv = (
            None
            if shared_kv
            else KVProjection(c.facet_dim, c.kv_heads, c.attention_head_dim, c.norm_eps)
        )
        self.ffn_norm = RMSNorm(c.facet_dim, c.norm_eps)
        self.ffn = SwiGLU(c.facet_dim, hidden)
        self.residual_gates = nn.Parameter(torch.full((2,), -2.0))

    def forward(
        self,
        facets: Tensor,
        history: AttentionFeatures | None = None,
        *,
        projector: KVProjection | None = None,
    ) -> Tensor:
        projection = self.kv if projector is None else projector
        if projection is None:
            raise ValueError("A block with shared KV requires its explicit projection")
        if self.kv is not None and projector is not None:
            raise ValueError("A local block cannot bypass its owned KV projection")
        attention = join_attention(projection(facets), history)
        facets = facets + self.residual_gates[0].sigmoid() * self.attention.read(facets, attention)
        return facets + self.residual_gates[1].sigmoid() * self.ffn(self.ffn_norm(facets))
