"""Measured sampled-output checks for multimodal packs and edit trajectories.

This evaluates decoded outputs, not gold-conditioned P inputs. All metrics have
denominators and evaluator/provenance metadata; higher-level adequacy and style
judgments remain externally measured, not invented proxy numbers.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np


def _edit_distance(a, b) -> int:
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Word-level Levenshtein distance over reference length (case-insensitive)."""
    ref, hyp = reference.lower().split(), hypothesis.lower().split()
    if not ref:
        return float(len(hyp) > 0)
    return _edit_distance(ref, hyp) / len(ref)


def character_error_rate(reference: str, hypothesis: str) -> float:
    ref, hyp = reference.lower(), hypothesis.lower()
    if not ref:
        return float(len(hyp) > 0)
    return _edit_distance(ref, hyp) / len(ref)


def endpoint_rmse(actual_video: np.ndarray, first: np.ndarray, last: np.ndarray):
    if actual_video.ndim != 4 or first.shape != actual_video.shape[1:] or last.shape != first.shape:
        raise ValueError("Video and independently sampled endpoint pixels need identical geometry")
    # Float first: unsigned 8-bit pixels would wrap around on subtraction.
    video = actual_video.astype(np.float64)
    first, last = first.astype(np.float64), last.astype(np.float64)
    return float(np.sqrt(np.mean((video[0] - first) ** 2))), float(
        np.sqrt(np.mean((video[-1] - last) ** 2))
    )


def edit_preservation_rmse(before: np.ndarray, after: np.ndarray, change_mask: np.ndarray) -> float:
    """Score preservation only OUTSIDE the requested changed region."""
    if before.shape != after.shape:
        raise ValueError("Edit preservation needs aligned media geometry")
    mask = np.asarray(change_mask, dtype=bool)
    if mask.shape != before.shape and mask.shape != before.shape[-2:]:
        raise ValueError("Changed-region mask needs aligned spatial dimensions")
    outside = ~np.broadcast_to(mask, before.shape)
    if not outside.any():
        raise ValueError("No unchanged coordinates available for edit-preservation score")
    difference = before[outside].astype(np.float64) - after[outside].astype(np.float64)
    return float(np.sqrt(np.mean(difference**2)))


def timed_event_matches(reference, predicted, *, tolerance_seconds: float):
    """Greedy nearest matched named events; no caption-based timing credit."""
    if tolerance_seconds < 0:
        raise ValueError("Timing tolerance must be nonnegative")
    reference = tuple(reference)
    predicted = list(predicted)
    correct = 0
    for name, time in reference:
        eligible = [
            (abs(float(value) - float(time)), i)
            for i, (label, value) in enumerate(predicted)
            if label == name
        ]
        if eligible:
            delta, index = min(eligible)
            if delta <= tolerance_seconds:
                correct += 1
                predicted.pop(index)
    return correct, len(reference)


@dataclass
class MetricAccumulator:
    """Separate numerators and valid site counts, including grouped rank merge."""

    totals: dict[str, float]
    counts: dict[str, int]

    def __init__(self):
        self.totals = defaultdict(float)
        self.counts = defaultdict(int)

    def record(self, name, numerator, count=1):
        if count < 0 or not np.isfinite(numerator):
            raise ValueError("Invalid metric numerator/count")
        self.totals[name] += float(numerator)
        self.counts[name] += int(count)

    def merge(self, other):
        for k in other.totals:
            self.record(k, other.totals[k], other.counts[k])

    def report(self):
        return {
            key: {
                "score": self.totals[key] / self.counts[key],
                "numerator": self.totals[key],
                "count": self.counts[key],
            }
            for key in sorted(self.counts)
            if self.counts[key]
        }
