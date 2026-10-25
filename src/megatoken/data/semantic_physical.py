"""Physical framing for the shared semantic witness without changing its authority.

One logical request may occupy several native records. Only its closing record
can precede a result, and the single LogicalAction retains the exact original
bytes. Token counts use the same codec and chunk planner as model serving.
"""

import json
from dataclasses import replace

from .schema import Dependency, Episode, Meridian, TrackEvent, make_records


def physical_semantic_episode(episode: Episode, *, codec, max_positions: int, max_history: int):
    if type(max_positions) is not int or max_positions < 1:
        raise ValueError("Semantic physical token budget must be positive")
    if type(max_history) is not int or max_history < 1:
        raise ValueError("Semantic history budget must be positive")
    if codec is None:
        return episode
    records, meridians, spans, programs_by_track = [], [], {}, {}
    events = {event.record_id: event for group in episode.meridians for event in group.events}

    def append_chunk(chunk, event):
        meridians.append(
            Meridian(
                len(meridians),
                (
                    TrackEvent(
                        chunk.record_id,
                        event.track_id,
                        "text",
                        len(meridians),
                        chunk.role,
                        request_revision=event.request_revision,
                    ),
                ),
            )
        )
        records.append(chunk)

    for record in episode.records:
        event = events[record.record_id]
        chunks = make_records(
            record.payload,
            logical_id=record.record_id,
            role=record.role,
            turn_index=record.turn_index,
            policy_mask=record.policy_mask,
            provenance=record.provenance,
            action_id=record.action_id,
            result_of=record.result_of,
            execution_id=record.execution_id,
            codec=codec,
            max_tokens=max_positions,
            max_bytes=16384,
            content_kind="text",
        )
        spans[record.record_id] = tuple(chunk.record_id for chunk in chunks)
        for chunk in chunks:
            append_chunk(chunk, event)
        if not record.policy_mask:
            observation = json.loads(record.payload)
            kind = observation.get("kind")
            if kind == "invocation":
                programs_by_track[event.track_id] = observation["programId"]
            if (
                kind in {"invocation", "source-read", "source-evict"}
                or kind == "model-context"
                and "selectedState" in observation
            ):
                context = {
                    "semantic_context": {
                        "schema": "semantic-native-context/4",
                        "reason": kind,
                        "programId": observation.get(
                            "programId", programs_by_track.get(event.track_id)
                        ),
                    },
                    "context_window": {
                        "local_events": 0 if kind == "source-evict" else max_history,
                        "cross_events": 0 if kind == "source-evict" else max_history,
                        "pinned_events": [],
                    },
                }
                context_chunks = make_records(
                    json.dumps(context, separators=(",", ":")).encode(),
                    logical_id="semantic:context:" + record.record_id,
                    role="system",
                    turn_index=record.turn_index,
                    policy_mask=False,
                    provenance="executed",
                    codec=codec,
                    max_tokens=max_positions,
                    max_bytes=16384,
                    content_kind="text",
                )
                if len(context_chunks) != 1:
                    raise ValueError(
                        "Native context announcement exceeds the physical token budget"
                    )
                append_chunk(context_chunks[0], event)
    actions = tuple(
        replace(action, record_ids=tuple(part for old in action.record_ids for part in spans[old]))
        for action in episode.actions
    )
    dependencies = tuple(
        Dependency(
            spans[edge.writer_id][-1],
            spans[edge.reader_id][0],
            edge.kind,
        )
        for edge in episode.dependencies
    )
    annotations = json.loads(episode.supervision.annotations_json)
    annotations["semantic_geometry"] = {
        "schema": "semantic-native-geometry/4",
        "maxPositions": max_positions,
        "maxHistory": max_history,
    }
    return replace(
        episode,
        records=tuple(records),
        actions=actions,
        dependencies=dependencies,
        meridians=tuple(meridians),
        supervision=replace(
            episode.supervision, annotations_json=json.dumps(annotations, separators=(",", ":"))
        ),
    )
