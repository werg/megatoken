"""Deterministic family selection with coverage statistics."""
from __future__ import annotations
from dataclasses import dataclass
import random

DEFAULT_WEIGHTS = {
    "replay": 0.20,
    "transcription_and_verbalization": 0.20,
    "speech_translation": 0.15,
    "image_description_and_creation": 0.15,
    "audiovisual_and_endpoints": 0.15,
    "sequential_editing": 0.15,
}

@dataclass(frozen=True)
class TaskCoverage:
    total: int = 0
    joint: int = 0
    multisource: int = 0
    counterfactual: int = 0
    def fractions(self):
        n = max(1, self.total)
        return {"joint": self.joint/n, "multisource": self.multisource/n,
                "counterfactual": self.counterfactual/n}

class FamilySampler:
    def __init__(self, weights=None, seed=0):
        self.weights = dict(DEFAULT_WEIGHTS if weights is None else weights)
        if not self.weights or any(v < 0 for v in self.weights.values()) or abs(sum(self.weights.values()) - 1) > 1e-6:
            raise ValueError("Task shares must sum to one")
        self.rng = random.Random(seed)
        self.coverage = TaskCoverage()
    def choose(self, available):
        eligible = sorted(set(available) & self.weights.keys())
        if not eligible:
            raise ValueError("No supported family in selected manifest")
        weights = [self.weights[k] for k in eligible]
        if sum(weights) == 0:
            raise ValueError("Available task families have zero configured share")
        return self.rng.choices(eligible, weights=weights, k=1)[0]
    def record(self, *, outputs, inputs, counterfactual=False):
        c = self.coverage
        self.coverage = TaskCoverage(c.total + 1, c.joint + (outputs >= 2),
                c.multisource + (inputs >= 2), c.counterfactual + bool(counterfactual))
