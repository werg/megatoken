"""Exact local navigation links; independent of parser/chunk identity."""

from __future__ import annotations

import json
import math
from dataclasses import replace

from .schema import Episode, canonical_json

EXTRACTOR = "local-navigation-v1"
KINDS = frozenset({"next_part", "tool_result", "navigation"})


def local_relations(episode: Episode) -> list[dict]:
    """Validate explicit links and add exact physical-part/call-result links."""
    annotations = json.loads(episode.supervision.annotations_json)
    declared = annotations.get("knowledge_items", [])
    known = {r.record_id for r in episode.records} | {r["source"] for r in declared}
    rows = []
    for row in annotations.get("relations", []):
        if row.get("kind") not in KINDS:
            continue
        if row["left"] not in known or row["right"] not in known:
            raise ValueError("Navigation relation has a dangling endpoint")
        weight = float(row.get("weight", 1.0))
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError("Navigation weight must be finite and positive")
        rows.append({**row, "weight": weight})
    parts = {}
    calls = {}
    for record in episode.records:
        logical = record.logical_id or record.record_id
        previous = parts.get(logical)
        if previous is not None and previous.part_index + 1 == record.part_index:
            rows.append(
                dict(kind="next_part", left=previous.record_id, right=record.record_id, weight=1.0)
            )
        parts[logical] = record
        if record.action_id:
            calls[record.action_id] = record
        if record.result_of and record.part_index == 0:
            call = calls.get(record.result_of)
            if call is None:
                raise ValueError("Tool result has no preceding call")
            rows.append(
                dict(kind="tool_result", left=call.record_id, right=record.record_id, weight=1.0)
            )
    # Preserve explicit weights, with stable ordering and no double-counting.
    unique = {}
    for row in rows:
        unique.setdefault((row["kind"], row["left"], row["right"]), row)
    return list(unique.values())


def with_local_relations(episode: Episode) -> Episode:
    annotations = json.loads(episode.supervision.annotations_json)
    retained = [r for r in annotations.get("relations", []) if r.get("kind") not in KINDS]
    annotations["relations"] = retained + local_relations(episode)
    annotations["navigation_extractor"] = EXTRACTOR
    return replace(
        episode,
        supervision=replace(episode.supervision, annotations_json=canonical_json(annotations)),
    )
