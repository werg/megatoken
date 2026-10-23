"""Evidence-derived curriculum supply; stream names alone prove no behavior."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from fractions import Fraction


def behaviors(episode) -> set[str]:
    result = set()
    actions = {a.action_id: a for a in episode.actions if a.execution_id}
    records = {r.record_id: r for r in episode.records}
    for action in actions.values():
        if action.tool_name.startswith("track_"):
            result.add("lifecycle_" + action.tool_name.removeprefix("track_"))
    from .context import context_announcement

    if any(
        context_announcement(record.payload, record.role) is not None for record in episode.records
    ):
        from .context import validate_context_announcements

        validate_context_announcements(episode)
        result.add("public_context_window")
    annotations = json.loads(episode.supervision.annotations_json)
    for receipt in annotations.get("context_read_receipts", ()):
        action = actions.get(receipt.get("action_id"))
        record = records.get(receipt.get("result_record_id"))
        source = records.get(receipt.get("source_event_id"))
        if (
            action is not None
            and action.tool_name == "track_fetch"
            and record is not None
            and record.result_of == action.action_id
            and source is not None
            and json.loads(action.arguments_json).get("source_event") == source.record_id
        ):
            from .schema import digest_bytes

            if (
                record.execution_id == action.execution_id
                and digest_bytes(record.payload) == receipt.get("sha256")
                and type(receipt.get("available")) is bool
                and (receipt["available"] is False or record.payload == source.payload)
            ):
                result.add(
                    "explicit_context_read"
                    if receipt.get("available") is True
                    else "context_unavailable"
                )
    events = [e for group in episode.meridians for e in group.events]
    # Recognize an actual wait -> requested audio arrival -> continuation ->
    # translated output sequence. A stream label or prefix-only example is not
    # evidence that the policy learned to wait across arrivals.
    positions = {e.record_id: i for i, g in enumerate(episode.meridians) for e in g.events}
    event_by_id = {e.record_id: e for e in events}
    continued_outputs = {
        d.writer_id: d.reader_id for d in episode.dependencies if d.kind == "translation_continued"
    }
    for wait in actions.values():
        if wait.tool_name != "track_wait":
            continue
        args = json.loads(wait.arguments_json)
        wait_position = max(positions[r] for r in wait.record_ids)
        for prerequisite in args.get("prerequisites", ()):
            source = event_by_id.get(prerequisite)
            if source is None or source.modality != "audio" or source.role != "user":
                continue
            if positions[prerequisite] <= wait_position:
                continue
            for continuation in actions.values():
                if continuation.tool_name != "track_continue":
                    continue
                if json.loads(continuation.arguments_json).get("address") != args.get("address"):
                    continue
                call_position = min(positions[r] for r in continuation.record_ids)
                if call_position <= positions[prerequisite]:
                    continue
                receipts = [r for r in records.values() if r.result_of == continuation.action_id]
                if any(
                    receipt.record_id in continued_outputs
                    and positions[continued_outputs[receipt.record_id]] > call_position
                    for receipt in receipts
                ):
                    result.add("streaming_wait_emit")
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
    spans = [
        (
            Fraction(e.media_span.start_tick, e.media_span.ticks_per_second),
            Fraction(e.media_span.end_tick, e.media_span.ticks_per_second),
        )
        for e in audio
    ]
    for i, left in enumerate(audio):
        if "natural_speaker_overlap" in result:
            break
        for j in range(i + 1, len(audio)):
            right = audio[j]
            if (
                left.speaker == right.speaker
                or left.media_span.clock_id != right.media_span.clock_id
            ):
                continue
            if spans[i][0] < spans[j][1] and spans[j][0] < spans[i][1]:
                if all(records[e.record_id].provenance == "natural" for e in (left, right)):
                    result.add("natural_speaker_overlap")
                    break
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
    delay = annotations.get("controlled_read_delay_seconds")
    overhead = annotations.get("controlled_worker_creation_delay_seconds")
    if all(type(x) in (int, float) and math.isfinite(x) and x >= 0 for x in (delay, overhead)):
        created = "lifecycle_create" in result
        if not created and delay < 2 * overhead and not annotations.get("dependent_reads"):
            result.add("cheap_work_kept_serial")
        if created and delay > 2 * overhead and "executed_overlapping_calls" in result:
            result.add("slow_independent_work_delegated")
        if annotations.get("dependent_reads") and not created:
            reads = [a for a in actions.values() if a.tool_name == "read"]
            if len(reads) >= 2:
                earlier, later = sorted(reads, key=lambda a: positions[a.record_ids[0]])[:2]
                producers = {
                    r.record_id for r in records.values() if r.result_of == earlier.action_id
                }
                if any(
                    d.writer_id in producers and d.reader_id in later.record_ids
                    for d in episode.dependencies
                ):
                    result.add("dependent_work_kept_serial")
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
            if (
                not isinstance(ids, list)
                or len(ids) != count
                or any(not isinstance(identity, str) or not identity for identity in ids)
                or len(set(ids)) != len(ids)
            ):
                raise ValueError(f"Malformed original family identities for {key}")
            available_ids[behavior].update(ids)
    available = Counter({name: len(ids) for name, ids in available_ids.items()})
    missing = {name: count for name, count in requirements.items() if available[name] < count}
    if missing:
        raise ValueError(
            f"Required {split} behaviors lack prepared original families: {missing}; available: {dict(available)}"
        )
    return dict(available)
