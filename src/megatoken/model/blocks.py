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

    def __init__(
        self,
        config: ModelConfig,
        hidden: int,
        *,
        shared_kv: bool = False,
        temporary_read: bool = False,
    ) -> None:
        super().__init__()
        if temporary_read and not shared_kv:
            raise ValueError("Temporary KV reads belong only to the central processor")
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
        self.temporary_gate = nn.Parameter(torch.tensor(-2.0)) if temporary_read else None

    def forward(
        self,
        facets: Tensor,
        history: AttentionFeatures | None = None,
        *,
        projector: KVProjection | None = None,
        temporary: AttentionFeatures | None = None,
        temporal_positions: Tensor | None = None,
        slot_codes: Tensor | None = None,
    ) -> Tensor:
        projection = self.kv if projector is None else projector
        if projection is None:
            raise ValueError("A block with shared KV requires its explicit projection")
        if self.kv is not None and projector is not None:
            raise ValueError("A local block cannot bypass its owned KV projection")
        current = projection(
            facets, temporal_positions=temporal_positions, slot_codes=slot_codes
        )
        attention = join_attention(current, history)
        updated = self.attention.read(
            facets, attention, temporal_positions=temporal_positions, slot_codes=slot_codes
        )
        facets = facets + self.residual_gates[0].sigmoid() * updated
        if temporary is not None:
            if self.temporary_gate is None:
                raise ValueError("This layer has no intermediate KV reader")
            # A separate softmax/gate stops short-lived 32-slot keys from
            # competing for mass with thousands of historical KV slots.
            response = self.attention.read(
                facets, temporary, temporal_positions=temporal_positions,
                slot_codes=slot_codes,
            )
            facets = facets + self.temporary_gate.sigmoid() * response
        return facets + self.residual_gates[1].sigmoid() * self.ffn(self.ffn_norm(facets))
