"""Typed retrieval records. Address, semantic value, and native attention K are distinct."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np

from .contracts import Collection, Footprint, Revisions, SemanticSpace, SourceRange, Visibility


@dataclass(frozen=True)
class Record:
    id: int
    collection: Collection
    scope: str
    key: np.ndarray
    footprint: Footprint
    key_revision: int
    payload_revision: int
    content_id: str = ""
    asset_ref: str = ""
    value: np.ndarray | None = None
    strength: float = 1.0
    lifetime_logit: float = -10.0
    birth_step: int = 0
    group: str = ""
    source_assignment: str = ""
    active: bool = True
    evidence_span: SourceRange | None = None

    def __post_init__(self) -> None:
        if self.id < 1 or self.collection is Collection.SOURCE and (
            not self.content_id or not self.asset_ref or self.value is not None
        ):
            raise ValueError("Source records need immutable content and no learned payload")
        if self.collection is Collection.MEMORY and self.value is None:
            raise ValueError("Memory points require a freely learned semantic value")


@dataclass(frozen=True)
class Relation:
    id: int
    left: int
    right: int
    scope: str
    prior: float
    value: np.ndarray
    dependence_logits: tuple[float, float]
    footprint: Footprint
    strength: float = 1.0
    lifetime_logit: float = -10.0
    payload_revision: int = 0

    def __post_init__(self) -> None:
        if not (0 <= self.prior <= 1) or self.left >= self.right or self.left < 1:
            raise ValueError("Canonical pairs require sorted distinct members and bounded prior")


@dataclass(frozen=True)
class Hit:
    record: Record
    similarity: float


@dataclass(frozen=True)
class SearchBatch:
    hits: tuple[Hit, ...]
    shortfall: int
    candidates_evaluated: int
    revisions: Revisions

    @property
    def ids(self) -> tuple[int, ...]:
        return tuple(hit.record.id for hit in self.hits)


class SearchBackend(Protocol):
    def search(
        self, query: np.ndarray, *, collection: Collection, scope: str,
        visibility: Visibility, limit: int, excluded: set[int] | None = None,
    ) -> SearchBatch: ...


class RecordStore(Protocol):
    space: SemanticSpace
    def scan(self, collection: Collection, scope: str) -> Sequence[Record]: ...
    def revisions(self) -> Revisions: ...
