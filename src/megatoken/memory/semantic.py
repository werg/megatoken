"""Trainable semantic mixtures project to complete, anchored native facet K/V."""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from megatoken.model.attention import AttentionFeatures, temporal_rope
from .contracts import NativeKVLayout


class MemoryKVProjector(nn.Module):
    """Registered native geometry: ONE K/V head set for each of F facet identities."""

    def __init__(self, semantic_width: int, facet_width: int, kv_heads: int,
                 head_dim: int, facets: int):
        super().__init__()
        if min(semantic_width, facet_width, kv_heads, head_dim, facets) < 1:
            raise ValueError("Invalid full-facet KB projector geometry")
        self.semantic_width = semantic_width
        self.facets, self.kv_heads, self.head_dim = facets, kv_heads, head_dim
        self.value_norm = nn.LayerNorm(semantic_width)
        self.to_facets = nn.Linear(semantic_width, facet_width)
        self.kb_descriptor = nn.Parameter(torch.randn(facet_width) * 0.01)
        self.facet_norm = nn.LayerNorm(facet_width)
        self.to_kv = nn.Linear(facet_width, 2 * kv_heads * head_dim, bias=False)

    def forward(self, semantic: Tensor, slot_codes: Tensor) -> tuple[Tensor, Tensor]:
        if semantic.ndim != 2 or semantic.shape[-1] != self.semantic_width or (
            slot_codes.shape != (self.facets, self.to_facets.out_features)
        ):
            raise ValueError("Memory projector requires full native facet geometry")
        features = self.to_facets(self.value_norm(semantic.float()))
        features = features[:, None] + slot_codes.float()[None] + self.kb_descriptor
        kv = self.to_kv(F.silu(self.facet_norm(features)))
        kv = kv.reshape(semantic.shape[0], self.facets, 2, self.kv_heads, self.head_dim)
        key, value = kv.unbind(2)
        return key.transpose(1, 2), value.transpose(1, 2)


@dataclass(frozen=True)
class Constituent:
    """A pinned selected point or verified pair, with independently decaying weight."""
    identity: str
    semantic: Tensor       # [batch, semantic_dim], live FP32 parameter leaf
    coefficient: Tensor    # [batch], live lookup weight from query/key/correlation
    lifetime_logit: Tensor # [batch] or scalar, live learned decay rate
    admitted_at: int
    version: str
    valid: Tensor | None = None

    def __post_init__(self) -> None:
        if not self.identity or not self.version or self.admitted_at < 0:
            raise ValueError("KB constituent requires immutable admission identity")
        if self.semantic.ndim != 2 or self.coefficient.shape != self.semantic.shape[:1]:
            raise ValueError("Memory semantic and coefficient shapes differ")
        if self.valid is not None and self.valid.shape != self.coefficient.shape:
            raise ValueError("Invalid per-row source/expiry mask")


@dataclass(frozen=True)
class AdmittedGroup:
    group_id: str
    constituents: tuple[Constituent, ...]
    anchor: int
    revision: str
    compatible_group: str

    def __post_init__(self) -> None:
        if not self.group_id or not self.revision or not self.compatible_group:
            raise ValueError("KB group needs stable identity and compatibility version")
        if self.anchor < 0 or not self.constituents:
            raise ValueError("KB group must contain contributors and a nonnegative anchor")
        if any(c.admitted_at > self.anchor for c in self.constituents):
            raise ValueError("Constituents cannot arrive after their KB group anchor")


@dataclass(frozen=True)
class MemoryMaterialization:
    features: AttentionFeatures
    semantics: Tensor     # [batch, groups, semantic_dim]
    strengths: Tensor     # [batch, groups], exact zero for no evidence
    identity: str

    @property
    def summary(self) -> Tensor:
        # Normalized by bounded group mass; no unsupported "average" bias.
        total = self.strengths.sum(1, keepdim=True)
        return (self.semantics * self.strengths[..., None]).sum(1) / total.clamp_min(1e-8)


@dataclass(frozen=True)
class KBContext:
    """External complete-facet context. Never appended to native raw/final/history."""
    layout: NativeKVLayout
    groups: tuple[AdmittedGroup, ...] = ()
    generation: int = 0
    max_positions: int = 4
    null_mass: float = 1.0

    def __post_init__(self) -> None:
        if self.max_positions < 1 or self.null_mass <= 0:
            raise ValueError("KB positions and null mass must be positive")
        if len(self.groups) > self.max_positions:
            raise ValueError("KB position budget is exceeded")
        ids = [g.group_id for g in self.groups]
        if len(set(ids)) != len(ids):
            raise ValueError("KB group identities must be unique")

    def admit(self, group: AdmittedGroup) -> KBContext:
        previous = next((g for g in self.groups if g.group_id == group.group_id), None)
        if previous is group:
            return self
        if previous is not None and group.anchor < previous.anchor:
            raise ValueError("A KB refresh cannot move its original anchor backwards")
        retained = tuple(g for g in self.groups if g.group_id != group.group_id)
        return replace(self, groups=(*retained, group)[-self.max_positions:],
                       generation=self.generation + 1)

    def evict(self, group_id: str) -> KBContext:
        kept = tuple(g for g in self.groups if g.group_id != group_id)
        return self if len(kept) == len(self.groups) else replace(
            self, groups=kept, generation=self.generation + 1
        )

    def detach(self) -> KBContext:
        groups = tuple(replace(g, constituents=tuple(
            replace(c, semantic=c.semantic.detach(), coefficient=c.coefficient.detach(),
                    lifetime_logit=c.lifetime_logit.detach(),
                    valid=None if c.valid is None else c.valid.detach())
            for c in g.constituents
        )) for g in self.groups)
        return replace(self, groups=groups)

    def materialize(self, *, age: Tensor, projector: MemoryKVProjector,
                    slot_codes: Tensor, dtype: torch.dtype = torch.float32,
                    batch_size: int | None = None) -> MemoryMaterialization:
        if age.ndim != 1 or (batch_size is not None and age.shape[0] != batch_size):
            raise ValueError("Memory materialization needs one committed meridian clock per row")
        batch = age.shape[0]
        if (self.layout.facets, self.layout.kv_heads, self.layout.head_dim) != (
            projector.facets, projector.kv_heads, projector.head_dim
        ):
            raise ValueError("Materialized KB geometry differs from native layout")
        all_keys, all_values, masks, log_strengths, semantics, strengths = [], [], [], [], [], []
        for group in self.groups:
            if group.anchor > int(age.min()):
                raise ValueError("Cannot read memory before its causal admission anchor")
            weights = []
            values = []
            for contributor in group.constituents:
                value = contributor.semantic
                if value.shape != (batch, projector.semantic_width):
                    raise ValueError("KB constituent width or batch differs from state")
                coefficient = contributor.coefficient.float().clamp_min(0)
                lifetime = F.softplus(contributor.lifetime_logit.float())
                duration = (age.float() - contributor.admitted_at).clamp_min(0)
                weight = coefficient * torch.exp(-lifetime * duration)
                if contributor.valid is not None:
                    weight = weight * contributor.valid.detach().to(weight.dtype)
                weights.append(weight)
                values.append(value.float())
            coeff = torch.stack(weights, dim=1)
            mass = coeff.sum(1)
            aggregate = torch.stack(values, dim=1)
            aggregate = (aggregate * coeff[..., None]).sum(1) / mass[:, None].clamp_min(1e-8)
            aggregate = torch.where((mass > 0)[:, None], aggregate, torch.zeros_like(aggregate))
            gate = mass / (self.null_mass + mass)
            k, v = projector(aggregate, slot_codes)
            anchor = torch.full((batch,), group.anchor, device=age.device, dtype=age.dtype)
            k = temporal_rope(k, anchor)
            all_keys.append(k.to(dtype))
            all_values.append(v.to(dtype))
            mask = (gate > 0)[:, None].expand(batch, self.layout.facets)
            masks.append(mask)
            log_strengths.append(gate.clamp_min(1e-30).log()[:, None].expand_as(mask))
            semantics.append(aggregate)
            strengths.append(gate)
        if all_keys:
            keys = torch.cat(all_keys, dim=2)
            vals = torch.cat(all_values, dim=2)
            mask = torch.cat(masks, dim=1)
            log_strength = torch.cat(log_strengths, dim=1)
            latent = torch.stack(semantics, dim=1)
            gates = torch.stack(strengths, dim=1)
        else:
            shape = (batch, self.layout.kv_heads, 0, self.layout.head_dim)
            keys = projector.kb_descriptor.new_zeros(shape, dtype=dtype)
            vals = keys
            mask = torch.empty(batch, 0, dtype=torch.bool, device=age.device)
            log_strength = projector.kb_descriptor.new_zeros(batch, 0)
            latent = projector.kb_descriptor.new_zeros(batch, 0, projector.semantic_width)
            gates = projector.kb_descriptor.new_zeros(batch, 0)
        # External KB identity is distinct from every categorical native track.
        track_ids = torch.full((batch, keys.shape[2]), -2, device=age.device, dtype=torch.long)
        identity_payload = [
            (g.group_id, g.revision, g.anchor,
             [(x.identity, x.version, x.admitted_at) for x in g.constituents])
            for g in self.groups
        ]
        fingerprint = hashlib.sha256(json.dumps(
            [self.generation, self.layout.projector_revision, identity_payload,
             age.detach().cpu().tolist()], sort_keys=True
        ).encode()).hexdigest()
        return MemoryMaterialization(
            AttentionFeatures(keys, vals, mask, track_ids, log_strength),
            latent, gates, fingerprint,
        )
