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
            current.append(min(previous[j] + 1, current[j - 1] + 1,
                               previous[j - 1] + (x != y)))
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
    return float(np.sqrt(np.mean((actual_video[0] - first)**2))), float(
        np.sqrt(np.mean((actual_video[-1] - last)**2))
    )


def temporal_motion(video: np.ndarray) -> float:
    if video.ndim != 4 or len(video) < 2:
        raise ValueError("Temporal-motion evaluation needs a multi-frame video")
    return float(np.mean(np.abs(np.diff(video.astype(np.float32), axis=0))))


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
    return float(np.sqrt(np.mean((before[outside] - after[outside])**2)))


def timed_event_matches(reference, predicted, *, tolerance_seconds: float):
    """Greedy nearest matched named events; no caption-based timing credit."""
    if tolerance_seconds < 0:
        raise ValueError("Timing tolerance must be nonnegative")
    reference = tuple(reference)
    predicted = list(predicted)
    correct = 0
    for name, time in reference:
        eligible = [(abs(float(value) - float(time)), i) for i, (label,value)
                    in enumerate(predicted) if label == name]
        if eligible:
            delta, index = min(eligible)
            if delta <= tolerance_seconds:
                correct += 1
                predicted.pop(index)
    return correct, len(reference)


def sequence_order_accuracy(reference, predicted):
    """Relative event order among named events; zero credit for hallucinated sequence."""
    if len(reference) < 2:
        return 0, 0
    gold_pairs = [(reference[i],reference[j]) for i in range(len(reference))
                  for j in range(i+1,len(reference)) if reference[i] != reference[j]]
    indices = {name: i for i,name in enumerate(predicted)}
    return sum(1 for a,b in gold_pairs if a in indices and b in indices
               and indices[a] < indices[b]), len(gold_pairs)


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
            self.record(k,other.totals[k],other.counts[k])
    def report(self):
        return {key: {"score":self.totals[key]/self.counts[key],
                      "numerator":self.totals[key],
                      "count":self.counts[key]}
                for key in sorted(self.counts) if self.counts[key]}


def four_way_deltas(scores: dict):
    """Separate codec loss, bottleneck loss, and policy prediction loss."""
    needed=("native","codec_roundtrip","bottleneck","causal")
    if set(scores) != set(needed) or any(not np.isfinite(v) for v in scores.values()):
        raise ValueError("Require all four declared, measured baselines")
    return {
        "codec_minus_native":scores["codec_roundtrip"]-scores["native"],
        "bottleneck_minus_codec":scores["bottleneck"]-scores["codec_roundtrip"],
        "causal_minus_bottleneck":scores["causal"]-scores["bottleneck"],
    }
