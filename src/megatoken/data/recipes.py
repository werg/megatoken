"""Load the single canonical set of v13 data budgets and stage mixtures."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from importlib import import_module
from importlib.resources import files
from typing import Any

# Explicit code producers for procedural and locally collected streams. The
# values name real callables so a catalog entry cannot claim nonexistent data.
STREAM_PRODUCERS = {
    "context_management": "megatoken.data.context_curriculum:curriculum_episode",
    "dynamic_execution": "megatoken.data.dynamic_execution:execution_episode",
    "duplex_execution": "megatoken.data.execution_traces:prepare_traces",
    "timed_conversation": "megatoken.data.timed_media:prepare_rows",
    "streaming_translation": "megatoken.data.timed_media:prepare_rows",
    "av_understanding": "megatoken.data.concordance:compile_task",
    "endpoints": "megatoken.data.concordance:compile_task",
    "endpoint_conditioned_video": "megatoken.data.concordance:compile_task",
    "speech_counterfactual": "megatoken.data.concordance:compile_task",
    "cross_track_evidence": "megatoken.data.concordance:compile_task",
}


def resolve_stream_producer(stream: str):
    """Resolve a registered stream producer to its importable callable."""
    try:
        path = STREAM_PRODUCERS[stream]
        module_name, attribute = path.split(":", 1)
        producer = getattr(import_module(module_name), attribute)
    except (KeyError, ImportError, AttributeError, ValueError) as error:
        raise ValueError(f"Unknown or unavailable stream producer for {stream!r}") from error
    if not callable(producer):
        raise ValueError(f"Stream producer for {stream!r} is not callable")
    return producer


@dataclass(frozen=True)
class StageRecipe:
    phase: str
    meter: str
    limit: int
    mixture: dict[str, float]
    effort_mix: dict[int, float]
    constraints: dict[str, Any]


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
        constraints=raw["constraints"],
    )
