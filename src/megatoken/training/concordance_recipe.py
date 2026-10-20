"""Versioned concordant task-family recipe and measured coverage, existing G stream.

Selection uses the production MixtureWindowStream. No independent learner,
network, optimizer or source preparation happens in this module.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from megatoken.data.concordance import TASK_FAMILIES


@dataclass(frozen=True, slots=True)
class ConcordanceRecipe:
    phase: str
    mixture: dict[str, float]
    auxiliary_weight: float
    max_media_events_per_window: int
    quotas: dict[str, float]
    fingerprint: str

    def __post_init__(self) -> None:
        if not self.mixture or any(
            key not in TASK_FAMILIES and key != "general_replay" or
            not math.isfinite(share) or share <= 0
            for key, share in self.mixture.items()
        ):
            raise ValueError("Invalid or unsupported concordance stream mixture")
        if not math.isclose(sum(self.mixture.values()), 1.0, abs_tol=1e-9):
            raise ValueError("Concordance task stream shares must sum to 100%")
        if not 0 < self.auxiliary_weight <= 1:
            raise ValueError("Auxiliary weight must be in (0,1]")
        if self.max_media_events_per_window < 1:
            raise ValueError("Media microbatch event cap must be positive")


def load_concordance_recipe(path: str | Path, *, phase: str | None = None) -> ConcordanceRecipe:
    import hashlib
    raw = Path(path).read_bytes()
    obj = json.loads(raw)
    if obj.get("schema_version") != 1 or obj.get("plan") != "pretrained-omnimodal-bottleneck-v2":
        raise ValueError("Unrecognized concordance recipe identity")
    active = phase or obj["active_phase"]
    if active not in {"A", "B", "C", "D"}:
        raise ValueError("Concordance phase must be A/B/C/D")
    part = obj["phases"][active]
    return ConcordanceRecipe(
        active, part["mixture"], part["auxiliary_weight"],
        obj["max_media_events_per_window"], obj["quotas"],
        hashlib.sha256(raw).hexdigest(),
    )


@dataclass
class TaskCoverage:
    """Report what actually ran, rather than claiming recipe target quotas."""
    events: int = 0
    tasks: int = 0
    joint_tasks: int = 0
    multiple_input_tasks: int = 0
    variant_tasks: int = 0
    media_events: int = 0
    family_counts: Counter[str] | None = None

    def __post_init__(self):
        if self.family_counts is None:
            self.family_counts = Counter()

    def add(self, episode: object) -> None:
        annotations = json.loads(episode.supervision.annotations_json)
        family = annotations.get("task_family", episode.stream)
        self.family_counts[family] += 1
        self.tasks += 1
        self.events += len(episode.records)
        if episode.meridians:
            self.joint_tasks += int(any(
                len(group.events) >= 2 and all(
                    episode.records[next(i for i, record in enumerate(episode.records)
                                         if record.record_id == event.record_id)].policy_mask
                    for event in group.events
                )
                for group in episode.meridians
            ))
        self.multiple_input_tasks += int(sum(
            not record.policy_mask for record in episode.records
        ) >= 2)
        self.variant_tasks += int(
            family.endswith("_edit") or
            bool(annotations.get("counterfactual_base")) or
            any(dep.kind in {"edit", "preserve"} for dep in episode.dependencies)
        )
        self.media_events += sum(
            e.modality in {"audio", "image", "video"}
            for group in episode.meridians for e in group.events
        )

    def report(self, recipe: ConcordanceRecipe) -> dict[str, object]:
        count = max(1, self.tasks)
        measured = {
            "joint_output_fraction": self.joint_tasks / count,
            "multiple_input_fraction": self.multiple_input_tasks / count,
            "variant_fraction": self.variant_tasks / count,
        }
        return {
            "tasks": self.tasks, "events": self.events, "media_events": self.media_events,
            "phase": recipe.phase, "recipe_fingerprint": recipe.fingerprint,
            "families": dict(self.family_counts),
            "actual_fractions": measured,
            "quota_met": {
                "joint_output": measured["joint_output_fraction"]
                >= recipe.quotas["joint_output_min_fraction"],
                "multiple_input": measured["multiple_input_fraction"]
                >= recipe.quotas["multiple_input_target_fraction"],
                "counterfactual_or_edit": measured["variant_fraction"]
                >= recipe.quotas["counterfactual_or_edit_target_fraction"],
            },
        }
