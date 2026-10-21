"""Shared grouped-query attention over current facets and bounded causal context."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .layers import RMSNorm


def temporal_rope(x: Tensor, positions: Tensor, *, base: float = 1_000_000.0) -> Tensor:
    """Rotate query/key heads by committed megatoken position, never facet index.

    Input x is [batch, heads, slots, head_dim], positions is [batch].
    FP64 phase construction preserves adjacent late-record differences; rotation
    arithmetic uses FP32 and returns the original dtype for fused attention.
    """
    if x.ndim != 4 or x.shape[-1] % 2:
        raise ValueError("Temporal RoPE requires four-dimensional even-width heads")
    if positions.shape != (x.shape[0],):
        raise ValueError("RoPE requires one committed time per batch row")
    dim = x.shape[-1]
    frequency = torch.exp(
        -math.log(base) * torch.arange(0, dim, 2, device=x.device, dtype=torch.float64) / dim
    )
    angles = positions.to(device=x.device, dtype=torch.float64)[:, None, None, None]
    angles = angles * frequency[None, None, None, :]
    cosine = angles.cos().float()
    sine = angles.sin().float()
    paired = x.float().reshape(*x.shape[:-1], dim // 2, 2)
    left, right = paired.unbind(-1)
    rotated = torch.stack((left * cosine - right * sine, left * sine + right * cosine), dim=-1)
    return rotated.reshape_as(x).to(dtype=x.dtype)


@dataclass(frozen=True)
class AttentionFeatures:
    """Projected keys and values accepted by the shared attention read operator."""

    keys: Tensor
    values: Tensor
    mask: Tensor | None = None
    track_ids: Tensor | None = None  # Per-slot categorical identity, -1 if unknown.
    log_strength: Tensor | None = None  # [batch, slots] additive log gate.


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
    logs = None
    if history.log_strength is not None or live.log_strength is not None:
        logs = torch.cat(
            tuple(
                item.log_strength.to(live.keys.dtype)
                if item.log_strength is not None
                else item.keys.new_zeros(
                    item.keys.shape[0], item.keys.shape[2], dtype=live.keys.dtype
                )
                for item in (history, live)
            ),
            dim=1,
        )
    return AttentionFeatures(
        torch.cat((history.keys.to(live.keys.dtype), live.keys), dim=2),
        torch.cat((history.values.to(live.values.dtype), live.values), dim=2),
        mask,
        None
        if live.track_ids is None or history.track_ids is None
        else torch.cat((history.track_ids, live.track_ids), dim=1),
        logs,
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
        *,
        track_relations: bool = False,
    ) -> None:
        super().__init__()
        if query_heads % kv_heads:
            raise ValueError("KV heads must divide query heads")
        self.query_heads, self.kv_heads, self.head_dim = query_heads, kv_heads, head_dim
        self.query_norm = RMSNorm(query_dim, eps)
        self.query = nn.Linear(query_dim, query_heads * head_dim, bias=False)
        self.output = nn.Linear(query_heads * head_dim, query_dim, bias=False)
        # Runtime IDs have equality, never ordinal or geometric distance.
        self.same_track_bias = nn.Parameter(torch.zeros(query_heads)) if track_relations else None

    def read(
        self,
        query: Tensor,
        features: AttentionFeatures,
        *,
        temporal_positions: Tensor | None = None,
        slot_codes: Tensor | None = None,
        query_track_ids: Tensor | None = None,
    ) -> Tensor:
        if not features.keys.shape[2]:
            return torch.zeros_like(query)
        batch, count, _ = query.shape
        addressed = query if slot_codes is None else query + slot_codes.to(query.dtype)
        q = (
            self.query(self.query_norm(addressed))
            .reshape(batch, count, self.query_heads, self.head_dim)
            .transpose(1, 2)
        )
        if temporal_positions is not None:
            q = temporal_rope(q, temporal_positions)
        attn_mask = None if features.mask is None else features.mask[:, None, None, :]
        if features.log_strength is not None:
            if features.log_strength.shape != (batch, features.keys.shape[2]):
                raise ValueError("Per-slot KB log-strength has wrong attention shape")
            attn_mask = features.log_strength.to(q.dtype)[:, None, None, :]
            if features.mask is not None:
                attn_mask = attn_mask.masked_fill(~features.mask[:, None, None, :], float("-inf"))
        if self.same_track_bias is not None and (
            (query_track_ids is None) != (features.track_ids is None)
        ):
            raise ValueError("Track-aware attention requires both query and key identities")
        if (
            self.same_track_bias is not None
            and query_track_ids is not None
            and features.track_ids is not None
        ):
            if query_track_ids.shape != (batch, count) or features.track_ids.shape != (
                batch,
                features.keys.shape[2],
            ):
                raise ValueError("Attention track IDs must identify every query/key slot")
            same = query_track_ids[:, :, None] == features.track_ids[:, None, :]
            same = same & (query_track_ids[:, :, None] >= 0) & (features.track_ids[:, None, :] >= 0)
            bias = same[:, None].to(q.dtype) * self.same_track_bias.to(q.dtype)[None, :, None, None]
            if features.mask is not None:
                bias = bias.masked_fill(~features.mask[:, None, None, :], float("-inf"))
            attn_mask = (
                bias if attn_mask is None or attn_mask.dtype == torch.bool else attn_mask + bias
            )
        result = F.scaled_dot_product_attention(
            q,
            features.keys.to(q.dtype),
            features.values.to(q.dtype),
            attn_mask=attn_mask,
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
    temporal_positions: Tensor | None = None,
    slot_codes: Tensor | None = None,
    track_ids: Tensor | None = None,
) -> AttentionFeatures:
    batch, count, _ = source.shape
    addressed = source if slot_codes is None else source + slot_codes.to(source.dtype)
    kv = projection(norm(addressed)).reshape(batch, count, 2, kv_heads, head_dim)
    key, value = kv.unbind(2)
    keys = key.transpose(1, 2)
    if temporal_positions is not None:
        keys = temporal_rope(keys, temporal_positions)
    return AttentionFeatures(keys, value.transpose(1, 2), mask, track_ids)


class KVProjection(nn.Module):
    """Normalized projection; C has three private temporary and one final KV owner."""

    def __init__(self, source_dim: int, kv_heads: int, head_dim: int, eps: float = 1e-6):
        super().__init__()
        self.kv_heads, self.head_dim = kv_heads, head_dim
        self.source_norm = RMSNorm(source_dim, eps)
        self.key_value = nn.Linear(source_dim, 2 * kv_heads * head_dim, bias=False)

    def forward(
        self,
        source: Tensor,
        mask: Tensor | None = None,
        *,
        temporal_positions: Tensor | None = None,
        slot_codes: Tensor | None = None,
        track_ids: Tensor | None = None,
    ) -> AttentionFeatures:
        return _encode(
            source,
            self.source_norm,
            self.key_value,
            self.kv_heads,
            self.head_dim,
            mask,
            temporal_positions,
            slot_codes,
            track_ids,
        )


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
