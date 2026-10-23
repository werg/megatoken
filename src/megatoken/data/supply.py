"""Measured structured supply, with original families separate from task views."""

from __future__ import annotations

import json
from collections import Counter


def dimensions(episode):
    """Count actual groups/events; never infer independence from a source name."""
    counts = Counter()
    annotation = json.loads(episode.supervision.annotations_json)
    language = annotation.get("language", {})
    if not language:
        language = annotation.get("verification", {}).get("language", {})
    if isinstance(language, dict) and language.get("source") and language.get("target"):
        counts[f"language_pair:{language['source']}->{language['target']}"] += 1
    for group in episode.meridians:
        counts[f"group_width:{len(group.events)}"] += 1
        counts["group_modalities:" + "+".join(sorted({e.modality for e in group.events}))] += 1
        counts[
            "group_clock:declared" if group.selected_at_ns is not None else "group_clock:unknown"
        ] += 1
    for record in episode.records:
        counts[f"record_provenance:{record.provenance}"] += 1
    for group in episode.meridians:
        for event in group.events:
            counts[f"event_modality:{event.modality}"] += 1
            if event.speaker:
                counts["speaker_identity:declared"] += 1
            elif event.modality == "audio":
                counts["speaker_identity:unknown"] += 1
            if event.available_at_ns is not None:
                counts["availability:declared"] += 1
    for action in episode.actions:
        if action.execution_id and action.tool_name.startswith("track_"):
            counts[f"executed_lifecycle:{action.tool_name.removeprefix('track_')}"] += 1
    return counts


class SupplyCensus:
    """SQLite family dedup keeps full-corpus memory bounded."""

    def __init__(self, database):
        self.database = database
        self.counts = Counter()
        self.database.execute(
            "CREATE TABLE supply_families(scope TEXT, family TEXT, PRIMARY KEY(scope,family)) WITHOUT ROWID"
        )

    def add(self, episode, split, stream):
        features = dimensions(episode)
        features["episode_views"] = 1
        for dimension, count in features.items():
            scope = f"{split}|{stream}|{dimension}"
            self.counts[scope] += count
            self.database.execute(
                "INSERT OR IGNORE INTO supply_families VALUES (?,?)",
                (scope, episode.lineage.family_id),
            )

    def report(self):
        families = dict(
            self.database.execute("SELECT scope, COUNT(*) FROM supply_families GROUP BY scope")
        )
        return {
            "version": 1,
            "counts": dict(sorted(self.counts.items())),
            "original_families": dict(sorted(families.items())),
            "denominators": "group_* count committed groups; event_* and availability count events; record_* count physical records; language_pair and episode_views count episodes; original_families deduplicate each dimension",
        }
