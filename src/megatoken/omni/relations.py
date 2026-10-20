"""Counted concordance objectives with no target-peer access to the policy."""
from __future__ import annotations

import torch
from torch.nn import functional as F


def pairwise_semantic_relation(left, right, *, valid=None):
    """Aligned semantic feature comparisons; raw Z need not be identical."""
    if left.shape != right.shape:
        raise ValueError("Relation features require aligned coordinates")
    if left.ndim < 2:
        raise ValueError("Relation features require a batch dimension")
    diff = (F.normalize(left.float(), dim=-1) - F.normalize(right.float(), dim=-1)).square().sum(-1)
    if valid is None:
        return diff.sum(), torch.tensor(diff.numel(), device=diff.device)
    mask = valid.to(torch.bool)
    if mask.shape != diff.shape:
        raise ValueError("Pair validity must match relation sites")
    return diff.masked_select(mask).sum(), mask.sum()

def endpoint_targets(video, first_index=0, last_index=None):
    """Take displayed endpoints after video sampling and crop, not codec padding."""
    if video.ndim != 5 or video.shape[2] != 3:
        raise ValueError("Video must be [batch, frames, rgb, height, width]")
    last = video.shape[1] - 1 if last_index is None else last_index
    if not 0 <= first_index <= last < video.shape[1]:
        raise ValueError("Endpoint indices outside displayed interval")
    return video[:, first_index], video[:, last]

def coordinate_kl(posterior, prior, active_coordinates: int):
    """Return globally aggregatable KL numerator and coordinate denominator."""
    if active_coordinates <= 0:
        raise ValueError("No active KL coordinates")
    numerator = posterior.detach().kl(prior).sum()
    return numerator, torch.tensor(active_coordinates, device=numerator.device)
