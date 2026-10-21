"""The one task-mixture recipe for media phases: prepared streams and their shares.

A recipe names prepared dataset streams (``text``, ``<modality>_identity``,
concordant task families such as ``transcription``) and the share of training
events drawn from each. Shares are events, not accelerator seconds: a video
event costs far more than a text window. ``auxiliary_weight`` scales the
reconstruction/alignment auxiliaries relative to the primary objectives.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

from megatoken.io import read_mapping

RECIPE_VERSION = "megatoken-task-recipe-v1"
AUXILIARY_LOSSES = (
    "reconstruction",
    "audio_reconstruction",
    "visual_reconstruction",
    "encoder_alignment",
)


@dataclass(frozen=True, slots=True)
class TaskRecipe:
    mixture: dict[str, float]
    auxiliary_weight: float
    fingerprint: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.mixture, dict)
            or not self.mixture
            or any(
                not isinstance(name, str)
                or not name
                or type(share) not in (int, float)
                or not math.isfinite(share)
                or share <= 0
                for name, share in self.mixture.items()
            )
            or not math.isclose(math.fsum(self.mixture.values()), 1.0, abs_tol=1e-9)
        ):
            raise ValueError("Task recipe shares must be positive and sum to one")
        if type(self.auxiliary_weight) not in (int, float) or not 0 < self.auxiliary_weight <= 1:
            raise ValueError("Auxiliary weight must be in (0, 1]")


def load_task_recipe(path: str | Path) -> TaskRecipe:
    spec = read_mapping(path)
    if not isinstance(spec, dict) or set(spec) != {"recipe", "mixture", "auxiliary_weight"}:
        raise ValueError("A task recipe has exactly recipe, mixture and auxiliary_weight")
    if spec["recipe"] != RECIPE_VERSION:
        raise ValueError(f"Unsupported task recipe {spec['recipe']!r}")
    return TaskRecipe(
        dict(spec["mixture"]),
        spec["auxiliary_weight"],
        hashlib.sha256(Path(path).read_bytes()).hexdigest(),
    )
