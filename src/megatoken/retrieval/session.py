"""What one session has mounted and seen: the choice of items at a lookup.

A session mounts several collections. At each lookup its 768-wide query searches
all of them; the best items above their score threshold that the session has not
observed yet (at most ``max_sources``) are the ones observed next. Choosing is
pure: nothing is recorded until the lookup commits, so a cancelled lookup leaves
the session as it was. Training and serving use this one class.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .collection import Hit, Item, SourceCollection

DEFAULT_MAX_SOURCES = 2
# The ranked list one search returns: what training ranks and what recall is read from.
DEPTH = 16


@dataclass(frozen=True)
class Candidate:
    collection: str
    hit: Hit


@dataclass(frozen=True)
class Retrieved:
    """An item chosen at one lookup; ``item.origin`` is the original file reference."""

    collection: str
    item: Item
    score: float
    lookup: int


@dataclass
class Search:
    """The latest lookup's query and ranked candidates (training reads it for its losses)."""

    query: Any
    candidates: list[Candidate]


@dataclass
class SourceSession:
    collections: Mapping[str, SourceCollection]
    max_sources: int = DEFAULT_MAX_SOURCES
    # None: each collection's own threshold.
    threshold: float | None = None
    observed: dict[str, set[int]] = field(default_factory=dict)
    log: list[Retrieved] = field(default_factory=list)
    last: Search | None = None

    def __post_init__(self) -> None:
        if self.max_sources < 1 or not self.collections:
            raise ValueError("A source session mounts a collection and reads at least one item")
        self.observed = {name: set() for name in self.collections}

    def candidates(self, query: np.ndarray, depth: int = DEPTH) -> list[Candidate]:
        """The best ``depth`` unobserved items over every collection, best first.

        Ties go to the earlier-mounted collection, then the lower row.
        """
        merged = []
        for order, (name, collection) in enumerate(self.collections.items()):
            for hit in collection.search(query, depth, exclude=self.observed[name]):
                merged.append((-hit.score, order, hit.row, Candidate(name, hit)))
        merged.sort(key=lambda entry: entry[:3])
        return [entry[3] for entry in merged[:depth]]

    def floor(self, name: str) -> float:
        """The score an item of collection ``name`` must reach to be observed."""
        return self.collections[name].threshold if self.threshold is None else self.threshold

    def choose(self, candidates: Sequence[Candidate], lookup: int) -> list[Retrieved]:
        """The first ``max_sources`` candidates whose score reaches their threshold."""
        chosen = []
        for candidate in candidates:
            if candidate.hit.score >= self.floor(candidate.collection):
                collection = self.collections[candidate.collection]
                item = collection.item(candidate.hit.row)
                chosen.append(Retrieved(candidate.collection, item, candidate.hit.score, lookup))
            if len(chosen) == self.max_sources:
                break
        return chosen

    def commit(self, retrieved: Sequence[Retrieved]) -> None:
        for entry in retrieved:
            self.observed[entry.collection].add(entry.item.row)
        self.log.extend(retrieved)
