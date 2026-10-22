"""The relation kinds an aligned bundle can certify between two of its artifacts.

Bundles themselves are compiled by ``megatoken.data.concordance``; this module only
holds the relation vocabulary shared with the relation-projection objective.
"""

from __future__ import annotations

from dataclasses import dataclass

RELATIONS = frozenset(
    (
        "same_spoken_content",
        "translation_of",
        "describes",
        "synchronized_with",
        "first_frame_of",
        "last_frame_of",
        "edit_of",
        "preserves",
    )
)


@dataclass(frozen=True)
class Relation:
    kind: str
    left: str
    right: str
    weight: float = 1.0

    def __post_init__(self):
        if (
            self.kind not in RELATIONS
            or not self.left
            or not self.right
            or not 0 < self.weight <= 10
        ):
            raise ValueError("Unsupported or invalid relation")
