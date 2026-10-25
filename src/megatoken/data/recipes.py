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
    "semantic_execution": "megatoken.data.semantic_corpus:prepare_semantic_corpus",
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
    training: dict[str, Any]
    learning_rate: dict[str, float]
    required_donor_profiles: tuple[str, ...]
    previous_phase: str | None


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
        training={**load_recipes()["training_defaults"], **raw["training"]},
        learning_rate=raw["learning_rate"],
        required_donor_profiles=tuple(raw.get("required_donor_profiles", ())),
        previous_phase=raw["previous_phase"],
    )


def default_training_features() -> dict[str, Any]:
    """Authoritative runtime defaults, resolved by every TrainConfig constructor."""
    return dict(load_recipes()["training_defaults"])


def validate_stage_supply(config, corpus, *, split="train"):
    """Check declared default supply before expensive teacher preparation or training."""
    if not config.use_stage_mixture:
        return
    from .coverage import require_coverage

    recipe = stage_recipe(config.phase)
    required = (
        recipe.mixture
        if split == "train"
        else {name: weight for name, weight in recipe.mixture.items() if name in STREAM_PRODUCERS}
    )
    missing = [
        name
        for name, weight in required.items()
        if weight > 0
        and not any(
            part.manifest["counts"].get(f"split:{split}|stream:{name}", 0) for part in corpus.parts
        )
    ]
    if missing:
        raise ValueError(f"Required {split} stage streams are absent: {missing}")
    require_coverage(
        corpus, recipe.constraints.get("required_behaviors", {}), recipe.mixture, split=split
    )


def validate_training_donors(config) -> None:
    """Verify selected donor identities before production model allocation."""
    if config.purpose != "production":
        return
    from pathlib import Path

    from megatoken.donors import DONOR_PROFILES
    from megatoken.io import sha256_file

    required = stage_recipe(config.phase).required_donor_profiles
    if not required:
        return
    for name, root in zip(required, config.engram_imports or (), strict=True):
        metadata = json.loads((Path(root) / "engram.json").read_text())
        profile = DONOR_PROFILES[name]
        expected = {
            key: getattr(profile, key)
            for key in (
                "repository",
                "revision",
                "tokenizer_sha256",
                "config_sha256",
                "index_sha256",
            )
        }
        if (
            metadata.get("donor_profile") != name
            or metadata.get("algorithm") != "native-context-to-byte-hash-v1"
            or any(
                metadata.get("donor_metadata", {}).get(key) != value
                for key, value in expected.items()
            )
        ):
            raise ValueError(
                f"Production donor must be the pinned {name} asset in curriculum order"
            )
        if sha256_file(Path(root) / "descriptors.npz") != metadata["descriptor_sha256"]:
            raise ValueError(f"Production donor checksum differs: {name}")


def validate_stage_start(config, previous_phase: str | None) -> None:
    """Production advances through the one curriculum or resumes its current stage."""
    if config.purpose != "production":
        return
    expected = stage_recipe(config.phase).previous_phase
    if previous_phase == config.phase:
        return
    if previous_phase != expected:
        raise ValueError(
            f"Production stage {config.phase} requires a checkpoint from {expected}; "
            f"received {previous_phase}"
        )


def validate_teacher_encoding_source(config, encoding_from) -> None:
    """Keep every production stage in the existing bank's value space."""
    if (
        config.purpose == "production"
        and stage_recipe(config.phase).previous_phase is not None
        and not encoding_from
    ):
        raise ValueError(
            "Production stages after bootstrap require the preceding teacher "
            "encoding via --encoding-from"
        )
