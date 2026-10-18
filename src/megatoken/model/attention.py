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
    mask: Tensor | None = None


def join_attention(live: AttentionFeatures, history: AttentionFeatures | None) -> AttentionFeatures:
    """Append current slots to one projected history without changing its owners."""
    if history is None or not history.keys.shape[2]:
        return live
    mask = None
    if history.mask is not None or live.mask is not None:
        masks = []
        for features in (history, live):
            masks.append(
                features.mask
                if features.mask is not None
                else torch.ones(
                    features.keys.shape[0],
                    features.keys.shape[2],
                    device=features.keys.device,
                    dtype=torch.bool,
                )
            )
        mask = torch.cat(masks, dim=1)
    return AttentionFeatures(
        torch.cat((history.keys.to(live.keys.dtype), live.keys), dim=2),
        torch.cat((history.values.to(live.values.dtype), live.values), dim=2),
        mask,
    )


class AttentionRead(nn.Module):
    """Independently owned query/output transforms over shared projected KV."""

    def __init__(
        self,
        query_dim: int,
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
        self.query = nn.Linear(query_dim, query_heads * head_dim, bias=False)
        self.output = nn.Linear(query_heads * head_dim, query_dim, bias=False)

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


def _encode(
    source: Tensor,
    norm: nn.Module,
    projection: nn.Linear,
    kv_heads: int,
    head_dim: int,
    mask: Tensor | None,
) -> AttentionFeatures:
    batch, count, _ = source.shape
    kv = projection(norm(source)).reshape(batch, count, 2, kv_heads, head_dim)
    key, value = kv.unbind(2)
    return AttentionFeatures(key.transpose(1, 2), value.transpose(1, 2), mask)


class KVProjection(nn.Module):
    """A single normalized KV owner shared by central and predictive readers."""

    def __init__(self, source_dim: int, kv_heads: int, head_dim: int, eps: float = 1e-6):
        super().__init__()
        self.kv_heads, self.head_dim = kv_heads, head_dim
        self.source_norm = RMSNorm(source_dim, eps)
        self.key_value = nn.Linear(source_dim, 2 * kv_heads * head_dim, bias=False)

    def forward(self, source: Tensor, mask: Tensor | None = None) -> AttentionFeatures:
        return _encode(source, self.source_norm, self.key_value, self.kv_heads, self.head_dim, mask)


class AttentionBranch(AttentionRead):
    """Convenience branch with its own source projection for local/cross attention."""

    def __init__(
        self,
        query_dim: int,
        source_dim: int,
        query_heads: int,
        kv_heads: int,
        head_dim: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__(query_dim, query_heads, kv_heads, head_dim, eps)
        self.source_norm = RMSNorm(source_dim, eps)
        self.key_value = nn.Linear(source_dim, 2 * kv_heads * head_dim, bias=False)

    def encode(self, source: Tensor, mask: Tensor | None = None) -> AttentionFeatures:
        return _encode(source, self.source_norm, self.key_value, self.kv_heads, self.head_dim, mask)

    def forward(self, query: Tensor, source: Tensor, mask: Tensor | None = None) -> Tensor:
        return self.read(query, self.encode(source, mask))
