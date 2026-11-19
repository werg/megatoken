"""Physical framing for the shared semantic witness without changing its authority.

One logical request may occupy several native records. Only its closing record
can precede a result, and the single LogicalAction retains the exact original
bytes. Token counts use the same codec and chunk planner as model serving.
"""

import json
from dataclasses import replace

from .schema import Dependency, Episode, Meridian, make_records


def validate_semantic_creation_observations(episode: Episode) -> None:
    """A creation receipt derives from code and the adjacent available request.

    No generated decision may intervene between creation and its invocation.
    Exact reconstruction prevents a private target or later result being
    smuggled into an otherwise unscored system observation.
    """
    from megatoken.lifecycle import (
        TrackDescription,
        semantic_execution_creation,
        track_description_observation,
    )

    groups = tuple(episode.iter_logical_records())
    events = {event.record_id: event for group in episode.meridians for event in group.events}
    revisions = {revision.record_id: revision for revision in episode.track_revisions}
    programs = {}
    for index, parts in enumerate(groups):
        creation = parts[0].logical_id.startswith("semantic:lifecycle:")
        if parts[0].policy_mask:
            if creation:
                raise ValueError("Semantic creation must remain an unscored observation")
            continue
        payload = b"".join(part.payload for part in parts)
        item = json.loads(payload)
        if item.get("kind") == "program":
            programs[item["programId"]] = item
        if not creation:
            continue
        if index + 1 >= len(groups) or any(
            part.policy_mask or part.role != "system" for part in parts
        ):
            raise ValueError("Semantic creation needs an adjacent observed invocation")
        following = groups[index + 1]
        if any(part.policy_mask or part.role != "user" for part in following):
            raise ValueError("Semantic creation cannot expose a future generated target")
        invocation = json.loads(b"".join(part.payload for part in following))
        program = programs.get(invocation.get("programId"))
        budget = item.get("budget")
        address = events[parts[0].record_id].track_id
        if (
            invocation.get("kind") != "invocation"
            or program is None
            or type(budget) is not int
            or not 1 <= budget <= 100000
            or events[following[0].record_id].track_id != address
        ):
            raise ValueError("Semantic creation differs from its available invocation")
        description = TrackDescription(
            address,
            status="waiting",
            **semantic_execution_creation(program, invocation, budget=budget),
        )
        revision = revisions.get(parts[-1].record_id)
        if payload != track_description_observation(description) or (
            revision is None
            or revision.track_id != address
            or revision.purpose != description.purpose
            or revision.revision != description.revision
        ):
            raise ValueError("Semantic creation must match its exact public ingress receipt")


def physical_semantic_episode(
    episode: Episode,
    *,
    codec,
    max_positions: int,
    max_history: int,
    selection_instruction=None,
    emit_context_windows=True,
):
    if type(max_positions) is not int or max_positions < 1:
        raise ValueError("Semantic physical token budget must be positive")
    if type(max_history) is not int or max_history < 1:
        raise ValueError("Semantic history budget must be positive")
    if codec is None:
        return episode
    records, meridians, spans = [], [], {}
    events = {event.record_id: event for group in episode.meridians for event in group.events}

    def append_chunk(chunk, event):
        meridians.append(
            Meridian(
                len(meridians),
                (
                    replace(
                        event,
                        record_id=chunk.record_id,
                        modality="text",
                        meridian_id=len(meridians),
                        role=chunk.role,
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
        for chunk_index, chunk in enumerate(chunks):
            chunk_event = event
            if record.policy_mask and chunk_index > 0:
                if selection_instruction is None:
                    from megatoken.controls import semantic_selection_instruction

                    instruction = semantic_selection_instruction(continuation=True)
                else:
                    instruction = selection_instruction(continuation=True)
                chunk_event = replace(event, instruction=instruction)
            append_chunk(chunk, chunk_event)
        if emit_context_windows and not record.policy_mask and record.payload.startswith(b"{"):
            observation = json.loads(record.payload)
            kind = observation.get("kind")
            if (
                kind in {"invocation", "source-read", "source-evict"}
                or kind == "model-context"
                and "selectedState" in observation
            ):
                from megatoken.controls import semantic_context_window_observation

                window = 0 if kind == "source-evict" else max_history
                context_chunks = make_records(
                    semantic_context_window_observation(window, window),
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
        "schema": "semantic-native-geometry/7"
        if annotations.get("semantic_execution", {}).get("schema") == "semantic-execution/7"
        else "semantic-native-geometry/4",
        "maxPositions": max_positions,
        "maxHistory": max_history,
    }
    return replace(
        episode,
        records=tuple(records),
        actions=actions,
        dependencies=dependencies,
        meridians=tuple(meridians),
        track_revisions=tuple(
            replace(revision, record_id=spans[revision.record_id][-1])
            for revision in episode.track_revisions
        ),
        supervision=replace(
            episode.supervision, annotations_json=json.dumps(annotations, separators=(",", ":"))
        ),
    )
