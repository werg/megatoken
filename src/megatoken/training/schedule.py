"""Schedules advance with a phase's declared token meter, never optimizer steps."""

from __future__ import annotations

import math
from dataclasses import dataclass

from megatoken.data.recipes import load_recipes, stage_recipe


@dataclass(frozen=True)
class Phase:
    name: str
    meter: str
    budget: int
    peak_lr: float
    warmup_fraction: float = 0.02
    decay_fraction: float = 0.10
    start_ratio: float = 0.1
    final_ratio: float = 0.2

    def __post_init__(self) -> None:
        if self.name not in load_recipes()["phases"]:
            raise ValueError(f"Unknown training phase {self.name!r}")
        if self.meter not in {"D", "I", "P", "G", "Q"} or self.budget <= 0 or self.peak_lr <= 0:
            raise ValueError("A phase needs a positive budget/LR and a D, I, P, G, or Q meter")
        if not 0 <= self.warmup_fraction < 1 or not 0 <= self.decay_fraction <= 1:
            raise ValueError("Invalid schedule fractions")
        if self.warmup_fraction + self.decay_fraction > 1:
            raise ValueError("Warmup and final decay must not overlap")
        if not 0 <= self.start_ratio <= 1 or not 0 <= self.final_ratio <= 1:
            raise ValueError("Learning-rate ratios must lie in [0, 1]")

    def learning_rate(self, consumed: int) -> float:
        progress = min(1.0, max(0.0, consumed / self.budget))
        if self.warmup_fraction and progress < self.warmup_fraction:
            ratio = self.start_ratio + (1 - self.start_ratio) * progress / self.warmup_fraction
        elif self.decay_fraction and progress > 1 - self.decay_fraction:
            position = (progress - 1 + self.decay_fraction) / self.decay_fraction
            ratio = (
                self.final_ratio + (1 - self.final_ratio) * (1 + math.cos(math.pi * position)) / 2
            )
        else:
            ratio = 1.0
        return self.peak_lr * ratio


def _phase(name: str) -> Phase:
    recipe = stage_recipe(name)
    return Phase(name, recipe.meter, recipe.limit, **recipe.learning_rate)


PHASES = {name: _phase(name) for name in load_recipes()["phases"]}
