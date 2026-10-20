"""Source-supported directional pair evidence, with checked-only singleton attenuation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor


@dataclass(frozen=True)
class SupportCheck:
    """A positive may be observed before the entire declared window was checked."""
    strength: float
    complete: bool
    distinct_units: bool

    def __post_init__(self) -> None:
        if not 0 <= self.strength <= 1:
            raise ValueError("Context support must lie in [0,1]")


@dataclass(frozen=True)
class PairEvidence:
    point: Tensor
    pair: Tensor
    activation: Tensor


def pair_evidence(
    point_weights: Tensor,
    *,
    left: Tensor,
    right: Tensor,
    dependence_logits: Tensor,
    prior: Tensor,
    pair_strength: Tensor,
    left_partner: Tensor,
    right_partner: Tensor,
    left_checked: Tensor,
    right_checked: Tensor,
    distinct_support: Tensor,
    beta: float = 0.5,
    pair_gain: float = 2.0,
) -> PairEvidence:
    """Live point/query/key/logit gradients; support flags and history are detached.

    left_partner[j] is observed support for RIGHT, independent of LEFT support.
    right_partner[j] is observed support for LEFT, independent of RIGHT support.
    """
    if not 0 < beta < 1 or pair_gain < 0:
        raise ValueError("Missing-partner cap and pair gain must be bounded")
    n = point_weights.numel()
    p = left.numel()
    if dependence_logits.shape != (p, 2):
        raise ValueError("Each pair requires two ordered dependence logits")
    if any(x.shape != (p,) for x in (
        left, right, prior, pair_strength, left_partner, right_partner,
        left_checked, right_checked, distinct_support
    )):
        raise ValueError("Pair fields must align")
    if p == 0:
        return PairEvidence(point_weights, point_weights.new_zeros(0),
                            point_weights.new_zeros(0))
    if (left < 0).any() or (right >= n).any() or (left >= right).any():
        raise ValueError("Pair endpoints must be sorted indices into the selected points")
    g = torch.sigmoid(dependence_logits)
    # Unknown partner: no negative evidence. Absence is penalized only after
    # a full independently eligible context check (not ANN omission).
    left_miss = g[:, 0] * left_checked.detach().to(g.dtype) * (
        1 - left_partner.detach()
    )
    right_miss = g[:, 1] * right_checked.detach().to(g.dtype) * (
        1 - right_partner.detach()
    )
    max_miss = torch.zeros_like(point_weights)
    max_miss = max_miss.scatter_reduce(0, left, left_miss, reduce="amax", include_self=True)
    max_miss = max_miss.scatter_reduce(0, right, right_miss, reduce="amax", include_self=True)
    adjusted = point_weights * (1 - beta * max_miss)
    valid = distinct_support.detach().to(point_weights.dtype)
    # Only original unboosted point scores and independent causal support
    # activate the pair; graph bonuses never manufacture their own evidence.
    contributions = torch.maximum(
        g[:, 0] * point_weights[left] * left_partner.detach(),
        g[:, 1] * point_weights[right] * right_partner.detach(),
    )
    pair = pair_gain * pair_strength * prior * contributions * valid
    return PairEvidence(adjusted, pair, valid)


def independent_support(
    primary_unit: str, partner_units: Mapping[str, float], *,
    primary_ancestors: frozenset[str] = frozenset(),
    checked_complete: bool,
) -> SupportCheck:
    """Repeated queries/duplicate source ancestry cannot count as a second leg."""
    positives = [value for name, value in partner_units.items()
                 if name != primary_unit and name not in primary_ancestors]
    return SupportCheck(max(positives, default=0.0), checked_complete, bool(positives))
