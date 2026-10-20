"""Frozen-coordinate shadow rebuild without hidden optimized free-memory leaves."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import os
import uuid

import numpy as np

from .contracts import Footprint, SemanticSpace, SourceRange
from .sqlite import SQLiteKB


@dataclass(frozen=True)
class ReproducedSource:
    """Literal bytes retained; only their document ADDRESS is regenerated."""
    content_id: str
    scope: str
    footprint: Footprint
    evidence_span: SourceRange
    asset_ref: str
    key: np.ndarray


@dataclass(frozen=True)
class ReproducedPoint:
    source_assignment: str
    scope: str
    footprint: Footprint
    key: np.ndarray
    value: np.ndarray
    strength: float
    lifetime_logit: float
    original_source: str
    assignment_distance: float
    output_slot: int


@dataclass(frozen=True)
class ReproducedPair:
    left_assignment: str
    right_assignment: str
    scope: str
    footprint: Footprint
    prior: float
    value: np.ndarray
    strength: float
    lifetime_logit: float
    dependence_logits: tuple[float, float]


@dataclass(frozen=True)
class RebuildResult:
    snapshot: Path
    mapping: dict[str, int]
    pair_ids: tuple[int, ...]
    point_coverage: float
    pair_coverage: float
    source_coverage: float = 1.0


def regenerate_bank(
    path: str | Path, space: SemanticSpace, *, value_dim: int,
    points: list[ReproducedPoint], pairs: list[ReproducedPair],
    expected_assignments: set[str], expected_pairs: set[tuple[str, str]],
    sources: list[ReproducedSource] | None = None,
    expected_sources: set[str] | None = None,
) -> RebuildResult:
    """Publish to a fresh path only after point AND pair target coverage pass.

    Does not load or preserve optimized memory values from the free bank. The
    caller supplies only frozen writer outputs and immutable source assignments.
    """
    path = Path(path)
    if path.exists():
        raise FileExistsError("A rebuilt KB must be a new shadow snapshot")
    assignments = {p.source_assignment for p in points}
    if len(assignments) != len(points) or not assignments.issubset(expected_assignments):
        raise ValueError("Writer output assignments are duplicated or unexpected")
    pair_names = {
        tuple(sorted((p.left_assignment, p.right_assignment))) for p in pairs
    }
    if len(pair_names) != len(pairs) or not pair_names.issubset(expected_pairs):
        raise ValueError("Pair rebuild subset or endpoint ordering is inconsistent")
    if assignments != expected_assignments or pair_names != expected_pairs:
        raise RuntimeError("Rebuild cannot pass with missing point/pair assignment targets")
    actual_sources = sources or []
    expected_source_ids = set() if expected_sources is None else expected_sources
    if len({s.content_id for s in actual_sources}) != len(actual_sources) or (
        {s.content_id for s in actual_sources} != expected_source_ids
    ):
        raise RuntimeError("Shadow corpus source-key coverage must match original assets")
    path.parent.mkdir(parents=True, exist_ok=True)
    # Build off-path: a failed source/pair insertion cannot leave a published
    # sqlite file that blocks a deterministic retry or appears complete.
    staging = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with SQLiteKB(staging, space, value_dim=value_dim) as kb:
            mapping = {}
            relation_ids = []
            with kb.transaction():
                for source in sorted(actual_sources, key=lambda s: s.content_id):
                    kb.insert_source(
                        scope=source.scope, footprint=source.footprint,
                        key=source.key, content_id=source.content_id,
                        asset_ref=source.asset_ref,
                        evidence_span=source.evidence_span,
                    )
                for point in sorted(points, key=lambda p: p.source_assignment):
                    record_id = kb.insert_memory(
                        scope=point.scope, footprint=point.footprint, key=point.key,
                        value=point.value, assignment=point.source_assignment,
                        birth_step=0, strength=point.strength,
                        lifetime_logit=point.lifetime_logit,
                    )
                    mapping[point.source_assignment] = record_id
                    # Preserve original assignment lineage and target weight for
                    # auditing without any optimized target value in serving rows.
                    kb.db.execute(
                        """INSERT INTO contributions
                        (assignment_id,source_content_id,record_id,footprint,original_key,
                        initial_value,assignment_distance,optimization_lineage)
                        VALUES (?,?,?,?,?,?,?,?)""",
                        (point.source_assignment, point.original_source, record_id,
                         point.footprint.to_json(),
                         np.asarray(point.key, dtype="<f4").tobytes(),
                         np.asarray(point.value, dtype="<f4").tobytes(),
                         point.assignment_distance,
                         json.dumps({"writer_output_slot": point.output_slot},
                                    sort_keys=True)),
                    )
                for pair in sorted(pairs, key=lambda p: (p.left_assignment, p.right_assignment)):
                    relation_ids.append(kb.insert_relation(
                        scope=pair.scope,
                        left=mapping[pair.left_assignment],
                        right=mapping[pair.right_assignment],
                        prior=pair.prior, value=pair.value,
                        dependence_logits=pair.dependence_logits,
                        footprint=pair.footprint,
                        strength=pair.strength, lifetime_logit=pair.lifetime_logit,
                    ))
                kb.put_growth_state("rebuild_manifest", {
                    "expected_points": len(expected_assignments),
                    "expected_pairs": len(expected_pairs),
                    "mode": "writer-only",
                })
        os.replace(staging, path)
        return RebuildResult(path, mapping, tuple(relation_ids), 1.0, 1.0, 1.0)
    finally:
        staging.unlink(missing_ok=True)
