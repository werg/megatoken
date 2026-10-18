"""Load the single canonical set of v13 data budgets and stage mixtures."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from importlib.resources import files
from typing import Any

from .accounting import Budget


@dataclass(frozen=True)
class StageRecipe:
    phase: str
    meter: str
    limit: int
    mixture: dict[str, float]
    effort_mix: dict[int, float]
    history_schedule: tuple[dict[str, Any], ...]
    constraints: dict[str, Any]

    @property
    def budget(self) -> Budget:
        return Budget(self.meter, self.limit)

    @property
    def exposures(self) -> dict[str, int]:
        return {stream: round(share * self.limit) for stream, share in self.mixture.items()}


def load_recipes() -> dict[str, Any]:
    result = json.loads(files("megatoken.data").joinpath("stages.json").read_text())
    if result["schema_version"] != 1:
        raise ValueError("Unsupported stage recipe version")
    return result


def stage_recipe(phase: str) -> StageRecipe:
    phases = load_recipes()["phases"]
    if phase not in phases:
        raise ValueError(f"Unknown phase {phase!r}; choose from {', '.join(phases)}")
    raw = phases[phase]
    for name in ("mixture", "effort_mix"):
        values = raw[name]
        if any(value < 0 for value in values.values()) or not math.isclose(
            sum(values.values()), 1.0, abs_tol=1e-10
        ):
            raise ValueError(f"Invalid {phase} {name}")
    return StageRecipe(
        phase=phase,
        meter=raw["meter"],
        limit=raw["limit"],
        mixture=raw["mixture"],
        effort_mix={int(key): value for key, value in raw["effort_mix"].items()},
        history_schedule=tuple(raw["history_schedule"]),
        constraints=raw["constraints"],
    )
