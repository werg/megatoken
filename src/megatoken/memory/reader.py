"""Differentiable selected-row readout; ANN discovery stays discrete and detached."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from megatoken.memory.semantic import AdmittedGroup, Constituent


@dataclass(frozen=True)
class EvidenceScores:
    similarity: Tensor  # [batch, candidates]
    coefficient: Tensor  # original unboosted, positive weights
    weights: Tensor  # sum < 1 because null mass is reserved
    null_weight: Tensor  # [batch]
    valid: Tensor  # [batch, candidates]


def selected_evidence(
    queries: Tensor,
    keys: Tensor,
    *,
    masks: Tensor | None = None,
    strengths: Tensor | None = None,
    duplicate_density: Tensor | None = None,
    threshold: float = 0.25,
    null_mass: float = 1.0,
    smoothing: float = 0.05,
    hard: bool = False,
) -> EvidenceScores:
    """Gradients reach live query AND selected stored-key leaves.

    No gradients pass through ANN candidate IDs, masks, or detached density.
    """
    if (
        queries.ndim != 2
        or keys.ndim != 3
        or (keys.shape[0] != queries.shape[0] or keys.shape[2] != queries.shape[1])
    ):
        raise ValueError("Expected causal [batch,dim] queries and selected [batch,k,dim] keys")
    if null_mass <= 0 or smoothing <= 0:
        raise ValueError("Null strength and smoothing must be positive")
    b, k, _ = keys.shape
    valid = (
        torch.ones(b, k, device=keys.device, dtype=torch.bool)
        if masks is None
        else masks.detach().bool()
    )
    if valid.shape != (b, k):
        raise ValueError("Candidate visibility mask has wrong shape")
    normed_q = F.normalize(queries.float(), dim=-1)
    normed_k = F.normalize(keys.float(), dim=-1)
    similarity = torch.einsum("bd,bkd->bk", normed_q, normed_k)
    activation = similarity - threshold
    positive = (
        activation.clamp_min(0)
        if hard
        else 0.5 * (activation + (activation.square() + smoothing**2).sqrt())
    )
    gains = positive
    if strengths is not None:
        if strengths.shape != (b, k):
            raise ValueError("Candidate strength shape differs")
        gains = gains * strengths.float().clamp_min(0)
    if duplicate_density is not None:
        if duplicate_density.shape != (b, k):
            raise ValueError("Candidate duplicate-density shape differs")
        gains = gains / duplicate_density.detach().float().clamp_min(1)
    gains = gains * valid.to(gains.dtype)
    mass = gains.sum(-1)
    weights = gains / (null_mass + mass[:, None])
    null = null_mass / (null_mass + mass)
    return EvidenceScores(similarity, gains, weights, null, valid)


def local_source_value_loss(
    scores: EvidenceScores,
    source_features: Tensor,
    *,
    positive: Tensor,
    complete_annotations: Tensor,
    positive_weights: Tensor | None = None,
    supported_mass_scale: float = 0.1,
    null_scale: float = 0.1,
) -> tuple[Tensor, int]:
    """Source features AND useful targets are detached. Unknown is never negative.

    Positive assistance/ANN misses must be handled outside this local unit;
    if annotated positives remain but no positive was found, the example is
    omitted, rather than receiving a misleading null target.
    """
    b, k = scores.weights.shape
    if source_features.ndim != 3 or source_features.shape[:2] != (b, k):
        raise ValueError("Detached E source features must match scored candidates")
    if positive.shape != (b, k) or complete_annotations.shape != (b,):
        raise ValueError("Each example needs explicit support labels and completeness")
    positives = positive.detach().bool() & scores.valid
    masses = (
        positives.float()
        if positive_weights is None
        else (positive_weights.detach().float().clamp_min(0) * positives.float())
    )
    masses = masses / masses.sum(-1, keepdim=True).clamp_min(1e-8)
    values = F.normalize(source_features.detach().float(), dim=-1)
    targets = (masses[..., None] * values).sum(1).detach()
    estimate = (scores.weights[..., None] * values).sum(1)
    residual = 0.5 * (estimate - targets).square().sum(-1)
    supported = positives.any(-1)
    # A labeled positive that ANN failed to select, or that visibility masked,
    # is NOT a no-answer example. Only an explicitly complete, truly empty
    # support set is eligible for null supervision.
    explicit_null = complete_annotations.detach().bool() & ~positive.detach().bool().any(-1)
    # Do not fabricate negatives for completely unannotated retrieval events.
    supported_loss = residual - supported_mass_scale * (
        scores.weights.masked_fill(~positives, 0).sum(-1).clamp_min(1e-8).log()
    )
    null_loss = -null_scale * scores.null_weight.clamp_min(1e-8).log()
    selected = torch.where(
        supported,
        supported_loss,
        torch.where(explicit_null, null_loss, torch.zeros_like(null_loss)),
    )
    valid_examples = supported | explicit_null
    return selected[valid_examples].sum(), int(valid_examples.sum())


def build_memory_groups(
    *,
    queries: Tensor,
    keys: Tensor,
    semantics: Tensor,
    strengths: Tensor,
    lifetime_logits: Tensor,
    record_ids: Sequence[str],
    scopes: Sequence[str],
    groups: Sequence[str],
    versions: Sequence[str],
    eligible: Tensor,
    anchor: int,
    threshold: float = 0.25,
    null_mass: float = 1.0,
    max_groups: int = 4,
) -> tuple[tuple[AdmittedGroup, ...], EvidenceScores]:
    """Only compatible groups are mixed; conflicting versions remain distinct."""
    if not (len(record_ids) == len(scopes) == len(groups) == len(versions) == keys.shape[1]):
        raise ValueError("Candidate identity lists must align with selected keys")
    if (
        semantics.shape[:2] != keys.shape[:2]
        or strengths.shape != keys.shape[:2]
        or (lifetime_logits.shape != keys.shape[:2])
    ):
        raise ValueError("All selected semantic controls must align")
    scores = selected_evidence(
        queries,
        keys,
        masks=eligible,
        strengths=strengths,
        threshold=threshold,
        null_mass=null_mass,
    )
    grouped: dict[tuple[str, str, str], list[int]] = {}
    for i, (scope, group, version) in enumerate(zip(scopes, groups, versions, strict=True)):
        grouped.setdefault((scope, group, version), []).append(i)
    selected_groups = []
    for (scope, group, version), indices in sorted(grouped.items()):
        constituents = tuple(
            Constituent(
                identity=record_ids[i],
                semantic=semantics[:, i],
                coefficient=scores.coefficient[:, i],
                lifetime_logit=lifetime_logits[:, i],
                admitted_at=anchor,
                version=versions[i],
                valid=eligible[:, i].detach(),
            )
            for i in indices
        )
        selected_groups.append(
            AdmittedGroup(
                group_id=f"{scope}/{group}/{version}",
                constituents=constituents,
                anchor=anchor,
                revision=version,
                compatible_group=f"{scope}/{group}/{version}",
            )
        )
    if len(selected_groups) > max_groups:
        raise ValueError("Too many incompatible semantic groups for the active KV budget")
    return tuple(selected_groups), scores
