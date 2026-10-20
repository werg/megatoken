"""Versioned semantic space, complete provenance, and causal visibility contracts."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from hashlib import sha256


class Collection(StrEnum):
    SOURCE = "source"
    MEMORY = "memory"


@dataclass(frozen=True)
class SemanticSpace:
    name: str
    revision: str
    teacher: str
    teacher_revision: str
    dimension: int
    document_recipe: str
    query_recipe: str
    context_recipe: str
    normalized: bool = True

    def __post_init__(self) -> None:
        if not all((self.name, self.revision, self.teacher, self.teacher_revision,
                    self.document_recipe, self.query_recipe, self.context_recipe)):
            raise ValueError("Semantic space and teacher/preprocessing revisions must be pinned")
        if not 1 <= self.dimension <= 16384 or not self.normalized:
            raise ValueError("The initial semantic space requires unit keys and a valid dimension")

    @property
    def fingerprint(self) -> str:
        return sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True, order=True)
class SourceRange:
    """Half-open BYTE range of a versioned original source; not tokenizer offsets."""
    source_id: str
    version: str
    start: int
    end: int
    content_hash: str
    scope: str
    available_at: int = 0
    valid_until: int | None = None
    family: str = ""
    task_family: str = ""
    ancestors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.source_id or not self.version or not self.scope or not self.content_hash:
            raise ValueError("Source identity, version, scope, and content digest are required")
        if self.start < 0 or self.end <= self.start or self.available_at < 0:
            raise ValueError("Expected positive half-open byte range and availability")
        if self.valid_until is not None and self.valid_until <= self.available_at:
            raise ValueError("Invalid source validity window")


@dataclass(frozen=True)
class Footprint:
    """ALL ranges seen by a query teacher, contextual key, support, or writer."""
    ranges: tuple[SourceRange, ...]

    def __post_init__(self) -> None:
        if not self.ranges:
            raise ValueError("An ungrounded memory is not source supported")

    def to_json(self) -> str:
        return json.dumps([asdict(r) for r in self.ranges], sort_keys=True)

    @classmethod
    def from_json(cls, value: str) -> Footprint:
        return cls(tuple(SourceRange(**{**r, "ancestors": tuple(r.get("ancestors", ()))})
                         for r in json.loads(value)))

    @property
    def fingerprint(self) -> str:
        return sha256(self.to_json().encode()).hexdigest()


@dataclass(frozen=True)
class Visibility:
    scope: str
    committed_cursor: int
    source_versions: dict[str, str] = field(default_factory=dict)
    prohibited_ancestors: frozenset[str] = frozenset()
    excluded_families: frozenset[str] = frozenset()
    excluded_tasks: frozenset[str] = frozenset()
    at_time: int | None = None

    def permits(self, footprint: Footprint) -> bool:
        for r in footprint.ranges:
            if r.scope != self.scope or r.available_at > self.committed_cursor:
                return False
            if r.valid_until is not None and (
                self.at_time if self.at_time is not None else self.committed_cursor
            ) >= r.valid_until:
                return False
            if self.source_versions and self.source_versions.get(r.source_id) != r.version:
                return False
            if r.family in self.excluded_families or r.task_family in self.excluded_tasks:
                return False
            if self.prohibited_ancestors.intersection((r.source_id, *r.ancestors)):
                return False
        return True


@dataclass(frozen=True)
class QueryBoundary:
    event_id: str
    meridian: int
    purpose: str
    originating_track: str
    request_hash: str
    space_fingerprint: str
    snapshot_id: str
    deadline: int

    def __post_init__(self) -> None:
        if not all((self.event_id, self.purpose, self.originating_track, self.request_hash,
                    self.space_fingerprint, self.snapshot_id)):
            raise ValueError("Query must identify its causal request and semantic space")
        if self.deadline < self.meridian or self.meridian < 0:
            raise ValueError("Query scheduling cannot travel backwards")


@dataclass(frozen=True)
class Revisions:
    key: int = 0
    index: int = 0
    payload: int = 0
    policy: int = 0
    graph: int = 0
    growth: int = 0

    def scoring_identity(self, query_model: str, space: SemanticSpace) -> str:
        return f"{space.fingerprint}:{query_model}:{self.key}:{self.index}"

    def materialization_identity(self, model: str) -> tuple:
        return (model, self.key, self.payload, self.policy, self.graph, self.growth)


@dataclass(frozen=True)
class NativeKVLayout:
    facets: int
    kv_heads: int
    head_dim: int
    facet_revision: str
    projector_revision: str
    kb_track: str = "__semantic_kb__"

    def __post_init__(self) -> None:
        if min(self.facets, self.kv_heads, self.head_dim) <= 0 or self.head_dim % 2:
            raise ValueError("Memory K/V must match positive, RoPE-compatible native geometry")
        if not self.facet_revision or not self.projector_revision:
            raise ValueError("Materialized cache layout requires pinned model identifiers")
        if not self.kb_track:
            raise ValueError("Dedicated KB track name required")


def unit_finite(values, *, dimension: int):
    """Portable little-endian FP32 normalized address; reject nonfinite payloads."""
    import numpy as np
    v = np.asarray(values, dtype="<f4")
    if v.shape != (dimension,) or not np.isfinite(v).all():
        raise ValueError("Incorrect, missing, or nonfinite semantic vector")
    norm = float(np.linalg.norm(v.astype("float64")))
    if not math.isfinite(norm) or norm < 1e-12:
        raise ValueError("Zero semantic address")
    return np.ascontiguousarray(v / norm, dtype="<f4")
