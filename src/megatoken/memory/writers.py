"""Post-pretraining source and pair writers in the final FROZEN semantic space."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class WriterOutput:
    key: Tensor
    value: Tensor
    strength: Tensor
    lifetime_logit: Tensor
    valid: Tensor

    def __post_init__(self):
        shape = self.key.shape[:2]
        if self.value.shape[:2] != shape or self.strength.shape != shape or (
            self.lifetime_logit.shape != shape or self.valid.shape != shape
        ):
            raise ValueError("Writer outputs must have identical source-slot correspondence")


@dataclass(frozen=True)
class PairWriterOutput:
    value: Tensor
    dependence_logits: Tensor
    strength: Tensor
    lifetime_logit: Tensor


class SourceWriter(nn.Module):
    """Offline bounded contextual writer over real native E(source) facet blocks.

    Physical suffix and full passes have disjoint attention masks; no hidden
    cache or teacher answer can carry forbidden future or target information.
    """
    def __init__(self, *, facets: int, facet_dim: int, semantic_dim: int,
                 value_dim: int, hidden: int = 256, heads: int = 4,
                 max_blocks: int = 256):
        super().__init__()
        if min(facets, facet_dim, semantic_dim, value_dim, hidden, heads, max_blocks) <= 0:
            raise ValueError("Writer dimensions must be positive")
        if hidden % heads:
            raise ValueError("Writer head count must divide hidden width")
        self.facets = facets
        self.max_blocks = max_blocks
        self.heads = heads
        self.norm = nn.LayerNorm(facet_dim)
        self.facet_pool = nn.Linear(facet_dim, hidden)
        self.facet_gate = nn.Linear(facet_dim, 1, bias=False)
        self.block_address = nn.Parameter(torch.randn(max_blocks, hidden) * 0.01)
        self.self_attention = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.feed = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 4 * hidden),
                                  nn.GELU(), nn.Linear(4 * hidden, hidden))
        self.norm_output = nn.LayerNorm(hidden)
        self.key = nn.Linear(hidden, semantic_dim, bias=False)
        self.value = nn.Linear(hidden, value_dim, bias=False)
        self.strength = nn.Linear(hidden, 1)
        self.lifetime = nn.Linear(hidden, 1)
        nn.init.constant_(self.strength.bias, -2.0)
        nn.init.constant_(self.lifetime.bias, -9.0)

    def forward(self, encoded: Tensor, *, mode: str, valid_blocks: Tensor,
                local_gap: int = 2) -> WriterOutput:
        if mode not in {"suffix", "full"}:
            raise ValueError("Post-pretraining writer must declare permitted source view")
        if encoded.ndim != 4 or not 0 < encoded.shape[1] <= self.max_blocks:
            raise ValueError("Writer requires a bounded [batch,blocks,facets,width] tensor")
        batch, count, f, d = encoded.shape
        if f != self.facets or d != self.norm.normalized_shape[0]:
            raise ValueError("Writer facets must match the native E output layout")
        if valid_blocks.shape != (batch, count):
            raise ValueError("Every source block requires an availability mask")
        slots = self.norm(encoded.float())
        facets = self.facet_gate(slots).squeeze(-1).softmax(-1)
        pooled = (self.facet_pool(slots) * facets[..., None]).sum(-2)
        # Suffix query positions must not contain the current (or earlier)
        # source content: only the attention VALUES are allowed to supply it.
        # A residual from pooled[i] here would leak source block i into the
        # writer target even when the attention mask starts at i + local_gap.
        x = pooled + self.block_address[:count]
        queries = (self.block_address[:count].unsqueeze(0).expand(batch, -1, -1)
                   if mode == "suffix" else x)
        q = torch.arange(count, device=x.device)[:, None]
        k = torch.arange(count, device=x.device)[None, :]
        if mode == "suffix":
            if local_gap < 1:
                raise ValueError("Suffix source writer requires a strictly positive gap")
            allowed = k >= q + local_gap
        else:
            allowed = torch.ones(count, count, device=x.device, dtype=torch.bool)
        effective = valid_blocks.detach().bool()
        if not bool(effective.any(-1).all()):
            raise ValueError("Writer requires at least one available source block")
        # Build a per-example physical mask. When a query has no permitted
        # suffix, route it to a valid fallback solely to prevent all-masked
        # NaNs. Such output slots are never scored or written.
        physical = allowed[None] & effective[:, None, :]
        original_valid = physical.any(-1)
        fallback = F.one_hot(
            effective.long().argmax(-1), num_classes=count
        ).bool()[:, None, :]
        physical = physical | ((~original_valid)[:, :, None] & fallback)
        mask = (~physical).repeat_interleave(self.heads, dim=0)
        attention = self.self_attention(
            queries, x, x, attn_mask=mask, key_padding_mask=~effective,
            need_weights=False,
        )[0]
        # In suffix mode, the residual also contains only the position query.
        # Invalid fallback positions are masked from all writer losses/outputs.
        representation = queries + attention
        result = self.norm_output(representation + self.feed(representation))
        valid = effective & original_valid
        return WriterOutput(
            F.normalize(self.key(result), dim=-1),
            self.value(result),
            self.strength(result).squeeze(-1).sigmoid(),
            self.lifetime(result).squeeze(-1),
            valid,
        )


class RelationWriter(nn.Module):
    """Pair keys/values and original source-window stats -> learned pair field targets."""

    def __init__(self, *, key_dim: int, value_dim: int, statistics_dim: int,
                 hidden: int = 256):
        super().__init__()
        if min(key_dim, value_dim, statistics_dim, hidden) < 1:
            raise ValueError("Relation writer requires declared member/statistics widths")
        self.member = nn.Sequential(
            nn.LayerNorm(2 * (key_dim + value_dim) + statistics_dim),
            nn.Linear(2 * (key_dim + value_dim) + statistics_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.value = nn.Linear(hidden, value_dim)
        self.dependence = nn.Linear(hidden, 2)
        self.strength = nn.Linear(hidden, 1)
        self.lifetime = nn.Linear(hidden, 1)
        nn.init.constant_(self.dependence.bias, -2.0)

    def forward(self, left_key: Tensor, right_key: Tensor,
                left_value: Tensor, right_value: Tensor,
                statistics: Tensor) -> PairWriterOutput:
        # Canonical left/right endpoint ordering is enforced by caller.
        features = torch.cat(
            (left_key, right_key, left_value, right_value, statistics), -1
        )
        hidden = self.member(features)
        return PairWriterOutput(
            self.value(hidden), self.dependence(hidden),
            self.strength(hidden).squeeze(-1).sigmoid(),
            self.lifetime(hidden).squeeze(-1),
        )


@dataclass(frozen=True)
class AssignmentTargets:
    """Explicit source-output-slot correspondence; no lossy averaged target."""
    slots: Tensor
    key: Tensor
    value: Tensor
    strength: Tensor
    lifetime_logit: Tensor
    distance: Tensor
    valid: Tensor

    def __post_init__(self):
        if self.slots.ndim != 2 or self.slots.dtype != torch.long:
            raise ValueError("Assignment correspondence must be [batch,assignments] int64")
        if self.valid.shape != self.slots.shape or self.distance.shape != self.valid.shape:
            raise ValueError("Assignments require matched masks and original distances")


def writer_distillation(
    predicted: WriterOutput, targets: AssignmentTargets, *,
    distance_temperature: float = 0.2,
    value_weight: float = 1.0, gate_weight: float = 0.25,
    lifetime_weight: float = 0.1,
) -> tuple[Tensor, int]:
    if distance_temperature <= 0 or min(value_weight, gate_weight, lifetime_weight) < 0:
        raise ValueError("Writer target weights and temperature must be nonnegative")
    n, a = targets.slots.shape
    valid = targets.valid.detach().bool()
    # Masked padding may use sentinel IDs, but scored assignments must always
    # identify an actual writer output slot. Clamping a live out-of-range ID
    # silently trains the wrong source and violates assignment provenance.
    if bool(((targets.slots < 0) | (targets.slots >= predicted.key.shape[1]))[valid].any()):
        raise ValueError("Scored writer assignment references a missing output slot")
    indices = targets.slots.clamp(0, predicted.key.shape[1] - 1)
    valid = valid & predicted.valid.gather(1, indices)
    batch = torch.arange(n, device=indices.device)[:, None]
    key = predicted.key[batch, indices]
    val = predicted.value[batch, indices]
    strength = predicted.strength[batch, indices]
    lifetime = predicted.lifetime_logit[batch, indices]
    if targets.key.shape != key.shape or targets.value.shape != val.shape:
        raise ValueError("Optimized final assignment targets do not match writer geometry")
    distance_weight = torch.exp(
        -targets.distance.detach().float().clamp_min(0) / distance_temperature
    )
    target_key = F.normalize(targets.key.detach().float(), dim=-1)
    key_error = 1 - (key.float() * target_key).sum(-1)
    value_error = (val.float() - targets.value.detach().float()).square().mean(-1)
    strength_error = (strength.float() - targets.strength.detach().float()).square()
    lifetime_error = (lifetime.float() - targets.lifetime_logit.detach().float()).square()
    terms = (key_error + value_weight * value_error +
             gate_weight * strength_error + lifetime_weight * lifetime_error)
    return (terms * distance_weight * valid).sum(), int(valid.sum())


def relation_distillation(
    predicted: PairWriterOutput, *, value: Tensor, strength: Tensor,
    lifetime_logit: Tensor, dependence_logits: Tensor,
    valid: Tensor, distance: Tensor, temperature: float = 0.2,
) -> tuple[Tensor, int]:
    """Pair targets include BOTH ordered directional dependence fields."""
    if predicted.dependence_logits.shape != dependence_logits.shape or (
        predicted.value.shape != value.shape
    ):
        raise ValueError("Pair writer output/target layout differs")
    weight = torch.exp(-distance.detach().clamp_min(0) / temperature)
    mismatch = (
        (predicted.value - value.detach()).square().mean(-1) +
        (predicted.dependence_logits - dependence_logits.detach()).square().mean(-1) +
        (predicted.strength - strength.detach()).square() +
        0.1 * (predicted.lifetime_logit - lifetime_logit.detach()).square()
    )
    return (mismatch * weight * valid.detach()).sum(), int(valid.sum())
