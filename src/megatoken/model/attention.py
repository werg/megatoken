"""Shared grouped-query attention over current facets and bounded causal context."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .layers import RMSNorm


@dataclass(frozen=True)
class AttentionFeatures:
    """Projected keys and values accepted by the shared attention read operator."""

    keys: Tensor
    values: Tensor
    mask: Tensor | None


class AttentionBranch(nn.Module):
    def __init__(
        self,
        query_dim: int,
        source_dim: int,
        query_heads: int,
        kv_heads: int,
        head_dim: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if query_heads % kv_heads:
            raise ValueError("KV heads must divide query heads")
        self.query_heads, self.kv_heads, self.head_dim = query_heads, kv_heads, head_dim
        self.query_norm = RMSNorm(query_dim, eps)
        self.source_norm = RMSNorm(source_dim, eps)
        self.query = nn.Linear(query_dim, query_heads * head_dim, bias=False)
        self.key_value = nn.Linear(source_dim, 2 * kv_heads * head_dim, bias=False)
        self.output = nn.Linear(query_heads * head_dim, query_dim, bias=False)

    def encode(self, source: Tensor, mask: Tensor | None = None) -> AttentionFeatures:
        batch, count, _ = source.shape
        kv = self.key_value(self.source_norm(source)).reshape(
            batch, count, 2, self.kv_heads, self.head_dim
        )
        key, value = kv.unbind(2)
        return AttentionFeatures(key.transpose(1, 2), value.transpose(1, 2), mask)

    def read(self, query: Tensor, features: AttentionFeatures) -> Tensor:
        if not features.keys.shape[2]:
            return torch.zeros_like(query)
        batch, count, _ = query.shape
        q = (
            self.query(self.query_norm(query))
            .reshape(batch, count, self.query_heads, self.head_dim)
            .transpose(1, 2)
        )
        result = F.scaled_dot_product_attention(
            q,
            features.keys.to(q.dtype),
            features.values.to(q.dtype),
            attn_mask=None if features.mask is None else features.mask[:, None, None, :],
            dropout_p=0.0,
            enable_gqa=self.query_heads != self.kv_heads,
        )
        return self.output(result.transpose(1, 2).reshape(batch, count, -1))

    def forward(self, query: Tensor, source: Tensor, mask: Tensor | None = None) -> Tensor:
        return self.read(query, self.encode(source, mask))
