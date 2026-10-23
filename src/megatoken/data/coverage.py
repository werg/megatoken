"""Evidence-derived curriculum supply; stream names alone prove no behavior."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from fractions import Fraction


def behaviors(episode) -> set[str]:
    result = set()
    actions = {a.action_id: a for a in episode.actions if a.execution_id}
    records = {r.record_id: r for r in episode.records}
    for action in actions.values():
        if action.tool_name.startswith("track_"):
            result.add("lifecycle_" + action.tool_name.removeprefix("track_"))
    events = [e for group in episode.meridians for e in group.events]
    for group in episode.meridians:
        calls = {records[e.record_id].action_id for e in group.events}
        # This behavior measures verified calls chosen together. It does not
        # claim that their wall-clock execution intervals overlapped.
        if episode.supervision.verified_success and sum(call in actions for call in calls) > 1:
            result.add("executed_parallel_calls")
        if (
            len({e.track_id for e in group.events}) < len(group.events)
            and len({e.modality for e in group.events}) > 1
        ):
            result.add("same_track_media_peers")
    audio = [
        e
        for e in events
        if e.modality == "audio"
        and e.media_span is not None
        and e.media_span.start_tick is not None
        and e.speaker
    ]
    if audio:
        result.add("speaker_preserving_audio")
    for i, left in enumerate(audio):
        for right in audio[i + 1 :]:
            a, b = left.media_span, right.media_span
            if left.speaker == right.speaker or a.clock_id != b.clock_id:
                continue
            if Fraction(a.start_tick, a.ticks_per_second) < Fraction(
                b.end_tick, b.ticks_per_second
            ) and Fraction(b.start_tick, b.ticks_per_second) < Fraction(
                a.end_tick, a.ticks_per_second
            ):
                if all(records[e.record_id].provenance == "natural" for e in (left, right)):
                    result.add("natural_speaker_overlap")
    timed = {
        e.modality
        for e in events
        if e.media_span is not None and e.media_span.start_tick is not None
    }
    if {"audio", "video"} <= timed:
        audio_clocks = {
            e.media_span.clock_id for e in events if e.modality == "audio" and e.media_span
        }
        video_clocks = {
            e.media_span.clock_id for e in events if e.modality == "video" and e.media_span
        }
        if audio_clocks & video_clocks:
            result.add("timed_audiovisual")
    if any(e.source_revision and e.available_at_ns is not None for e in events):
        result.add("versioned_source_arrival")
    annotations = json.loads(episode.supervision.annotations_json)
    # Co-selection proves that several executed calls belonged to one decision,
    # but not that their work overlapped in wall-clock time. Count actual
    # execution overlap only when the source provides start/end receipts.
    intervals = annotations.get("execution_intervals", ())
    if isinstance(intervals, list):
        intervals_by_action = {}
        for row in intervals:
            if not isinstance(row, dict):
                continue
            action_id = row.get("action_id")
            start, end = row.get("started_at_ns"), row.get("completed_at_ns")
            action = actions.get(action_id)
            if (
                action is not None
                and row.get("execution_id") == action.execution_id
                and type(start) is int
                and type(end) is int
                and 0 <= start < end
            ):
                intervals_by_action[action_id] = (start, end)
        measured = tuple(intervals_by_action.values())
        for index, (a_start, a_end) in enumerate(measured):
            if any(a_start < b_end and b_start < a_end for b_start, b_end in measured[index + 1 :]):
                result.add("executed_overlapping_calls")
                break
    for entry in annotations.get("native_action_tapes", ()):
        tape = entry["tape"]
        if any(sum(mask) > 1 for mask in tape["modality_masks"]):
            result.add("chosen_modality")
        if entry.get("outcome_credit") is not None:
            result.add("verified_policy_outcome")
        if tape["render_status"] != "rendered" or not all(tape["valid"]):
            result.add("recorded_selection_failure")
    return result


class Coverage:
    def __init__(self, database):
        self.counts = Counter()
        self.database = database
        database.execute(
            "CREATE TABLE behavior_families(scope TEXT, family TEXT, PRIMARY KEY(scope,family)) WITHOUT ROWID"
        )

    def add(self, episode, split, stream):
        for behavior in behaviors(episode):
            key = f"{split}|{stream}|{behavior}"
            self.counts[key] += 1
            self.database.execute(
                "INSERT OR IGNORE INTO behavior_families VALUES (?,?)",
                (key, episode.lineage.family_id),
            )

    def report(self):
        rows = self.database.execute(
            "SELECT scope, family FROM behavior_families ORDER BY scope, family"
        ).fetchall()
        family_ids = defaultdict(list)
        for scope, family in rows:
            family_ids[scope].append(family)
        families = {scope: len(ids) for scope, ids in family_ids.items()}
        return {
            "version": 2,
            "episodes": dict(sorted(self.counts.items())),
            "original_families": dict(sorted(families.items())),
            "original_family_ids": dict(sorted(family_ids.items())),
        }


def require_coverage(corpus, requirements, streams, *, split="train"):
    """Require unique source families for each behavior in one corpus split.

    ``train`` preserves the training caller's prior behavior. Evaluation can
    request ``validation`` independently, so training supply cannot mask a
    missing held-out behavior.
    """
    if not isinstance(split, str) or not split:
        raise ValueError("Coverage split must be a nonempty name")
    if any(type(value) is not int or value < 0 for value in requirements.values()):
        raise ValueError("Behavior family requirements must be nonnegative integers")
    available_ids = defaultdict(set)
    legacy_counts = Counter()
    legacy_parts = Counter()
    requested_streams = set(streams)
    for part in corpus.parts:
        report = part.manifest.get("behavior_coverage", {})
        family_ids = report.get("original_family_ids", {})
        for key, count in report.get("original_families", {}).items():
            if type(count) is not int or count < 0:
                raise ValueError(f"Malformed original family count for {key}")
            part_split, stream, behavior = key.split("|", 2)
            if part_split != split or stream not in requested_streams:
                continue
            ids = family_ids.get(key)
            if ids is None:
                legacy_counts[behavior] += count
                legacy_parts[behavior] += 1
            else:
                if (
                    not isinstance(ids, list)
                    or len(ids) != count
                    or any(not isinstance(identity, str) or not identity for identity in ids)
                    or len(set(ids)) != len(ids)
                ):
                    raise ValueError(f"Malformed original family identities for {key}")
                available_ids[behavior].update(ids)
    available = Counter({name: len(ids) for name, ids in available_ids.items()})
    for behavior, count in legacy_counts.items():
        # Old manifests carry only counts. They are safe as one source, but
        # cannot be deduplicated against another root's families.
        if legacy_parts[behavior] > 1 or available[behavior]:
            raise ValueError(
                f"Cannot verify unique original families for {behavior!r} across corpus roots; "
                "reprepare manifests with family identities"
            )
        available[behavior] += count
    missing = {name: count for name, count in requirements.items() if available[name] < count}
    if missing:
        raise ValueError(
            f"Required {split} behaviors lack prepared original families: {missing}; available: {dict(available)}"
        )
    return dict(available)
