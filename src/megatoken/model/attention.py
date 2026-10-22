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

    Input x is [batch, heads, slots, head_dim]; positions is [batch] (one
    meridian) or [batch, slots] (a whole recorded sequence at once). FP64 phase
    construction preserves adjacent late-record differences; rotation
    arithmetic uses FP32 and returns the original dtype for fused attention.
    """
    if x.ndim != 4 or x.shape[-1] % 2:
        raise ValueError("Temporal RoPE requires four-dimensional even-width heads")
    if positions.shape == (x.shape[0],):
        angles = positions.to(device=x.device, dtype=torch.float64)[:, None, None, None]
    elif positions.shape == (x.shape[0], x.shape[2]):
        angles = positions.to(device=x.device, dtype=torch.float64)[:, None, :, None]
    else:
        raise ValueError("RoPE requires one committed time per batch row or per slot")
    dim = x.shape[-1]
    frequency = torch.exp(
        -math.log(base) * torch.arange(0, dim, 2, device=x.device, dtype=torch.float64) / dim
    )
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
        None
        if live.track_ids is None or history.track_ids is None
        else torch.cat((history.track_ids, live.track_ids), dim=1),
    )


class AttentionRead(nn.Module):
    """Independently owned query/output transforms over shared projected KV.

    With ``sink=True`` every head also attends to one learned, position-free
    key/value slot that is never stored in KVHistory and never masked, so no
    query row can be fully masked. No dense mask is built unless something
    (a validity mask, an explicit query/key mask or the same-track
    bias) needs one, which keeps fused attention kernels eligible.
    """

    def __init__(
        self,
        query_dim: int,
        query_heads: int,
        kv_heads: int,
        head_dim: int,
        eps: float = 1e-6,
        *,
        track_relations: bool = False,
        sink: bool = False,
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
        self.sink_key = nn.Parameter(torch.zeros(kv_heads, 1, head_dim)) if sink else None
        self.sink_value = nn.Parameter(torch.zeros(kv_heads, 1, head_dim)) if sink else None

    def read(
        self,
        query: Tensor,
        features: AttentionFeatures,
        *,
        temporal_positions: Tensor | None = None,
        slot_codes: Tensor | None = None,
        query_track_ids: Tensor | None = None,
        query_key_mask: Tensor | None = None,
    ) -> Tensor:
        batch, count, _ = query.shape
        slots = features.keys.shape[2]
        sink = self.sink_key is not None
        if not slots and not sink:
            return torch.zeros_like(query)
        addressed = query if slot_codes is None else query + slot_codes.to(query.dtype)
        q = (
            self.query(self.query_norm(addressed))
            .reshape(batch, count, self.query_heads, self.head_dim)
            .transpose(1, 2)
        )
        # With one time per query row the sink key rotates with it, so its
        # logit is position-free and fused kernels stay eligible. With a time
        # per query slot the sink logit is added exactly as a mask column.
        sink_logit = None
        if sink and temporal_positions is not None and temporal_positions.ndim == 2:
            sink_key = self.sink_key.to(q.dtype).repeat_interleave(
                self.query_heads // self.kv_heads, dim=0
            )
            sink_logit = (q * sink_key[None, :, 0][:, :, None]).sum(-1) / math.sqrt(self.head_dim)
        if temporal_positions is not None:
            q = temporal_rope(q, temporal_positions)
        keys, values = features.keys.to(q.dtype), features.values.to(q.dtype)
        if sink:
            sink_key = self.sink_key.to(q.dtype)[None].expand(batch, -1, -1, -1)
            if sink_logit is not None:
                sink_key = torch.zeros_like(sink_key)
            elif temporal_positions is not None:
                # Rotate with the query's own time: the sink's logit never depends on it.
                sink_key = temporal_rope(sink_key, temporal_positions)
            keys = torch.cat((keys, sink_key), dim=2)
            values = torch.cat(
                (values, self.sink_value.to(q.dtype)[None].expand(batch, -1, -1, -1)), dim=2
            )
        allowed = None  # [batch, 1 | count, slots] boolean
        if features.mask is not None:
            if features.mask.shape == (batch, slots):
                allowed = features.mask[:, None]
            elif features.mask.shape == (batch, count, slots):
                allowed = features.mask
            else:
                raise ValueError("Attention feature mask has invalid query/key shape")
        if query_key_mask is not None:
            if query_key_mask.shape != (batch, count, slots):
                raise ValueError("Explicit attention mask must identify every query/key pair")
            allowed = query_key_mask if allowed is None else allowed & query_key_mask
        additive = None  # [batch, 1 | heads, 1 | count, slots]
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
                slots,
            ):
                raise ValueError("Attention track IDs must identify every query/key slot")
            same = query_track_ids[:, :, None] == features.track_ids[:, None, :]
            same = same & (query_track_ids[:, :, None] >= 0) & (features.track_ids[:, None, :] >= 0)
            bonus = (
                same[:, None].to(q.dtype) * self.same_track_bias.to(q.dtype)[None, :, None, None]
            )
            additive = bonus if additive is None else additive + bonus
        if sink_logit is not None and additive is None:
            additive = q.new_zeros(batch, 1, 1, slots)
        attn_mask = None
        if additive is not None:
            if allowed is not None:
                additive = additive.masked_fill(~allowed[:, None], float("-inf"))
            attn_mask = additive
        elif allowed is not None:
            attn_mask = allowed[:, None]
        if attn_mask is not None and sink:
            # The sink column is always visible (broadcast dims stay broadcast).
            if attn_mask.dtype == torch.bool:
                visible = attn_mask.new_ones(*attn_mask.shape[:-1], 1)
            elif sink_logit is not None:
                visible = sink_logit[..., None]
                attn_mask = attn_mask.expand(batch, self.query_heads, count, slots)
            else:
                visible = attn_mask.new_zeros(*attn_mask.shape[:-1], 1)
            attn_mask = torch.cat((attn_mask, visible), dim=-1)
        result = F.scaled_dot_product_attention(
            q,
            keys,
            values,
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
