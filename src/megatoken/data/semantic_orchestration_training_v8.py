"""Validate source-qualified V8 lifecycle teacher evidence."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict

from megatoken.controls import (
    canonical_semantic_observation,
    semantic_orchestration_ack_observation_v8,
    semantic_orchestration_input_v8,
    semantic_orchestration_instruction_v8,
    semantic_selection_instruction_v8,
)
from megatoken.execution_context import execution_binding
from megatoken.lifecycle import ExecutionBinding, TrackRegistry

from .semantic_json import stable_algorithm_hash, stable_algorithm_json

TRANSCRIPT_SCHEMA = "semantic-orchestration-transcript/1"
TEACHER_COMMIT_SCHEMA = "semantic-orchestration-teacher-commit/1"
ORCHESTRATION_RECORD_SCHEMA = "semantic-orchestration/1"
ORCHESTRATION_POLICY = {
    "schema": "semantic-orchestration-policy/1",
    "name": "create-join-publish-complete",
    "workAccounting": "one-per-complete-teacher-target/1",
    "maxActive": 1,
    "maxWaiting": 17,
    "coordinator": {
        "track": "track:0",
        "purpose": "Execute the declared V8 child tasks and publish their exact results",
        "completion": "All declared children completed and their exact results were published",
        "budget": 100000,
    },
    "child": {
        "track": "track:1",
        "purpose": "Run the exact declared V8 source",
        "completion": "Deliver its exact result",
        "budget": 64,
    },
}


class SemanticOrchestrationTranscriptError(ValueError):
    """Persisted V8 teacher evidence differs from independently replayed state."""


def _fail(message):
    raise SemanticOrchestrationTranscriptError(message)


def _same(left, right):
    return stable_algorithm_json(left) == stable_algorithm_json(right)


def _exact(value, keys, label):
    if type(value) is not dict or set(value) != set(keys):
        _fail(f"Malformed V8 orchestration {label}")


def _jsonable(value):
    return json.loads(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False))


def _track_value(track):
    return _jsonable(asdict(track))


def _teacher_commit(track, completion_dependency=None):
    receipt = {
        "schema": TEACHER_COMMIT_SCHEMA,
        "committed": True,
        "track": _track_value(track),
    }
    if completion_dependency is not None:
        receipt["completion_dependency"] = completion_dependency
    return receipt


def _public_cut(public, through):
    return {**public, "timeline": public["timeline"][:through]}


def _context(*, session, track, children, tracks, receipts, published, public_digest):
    return {
        "schema": "semantic-orchestration-context/8",
        "session": session,
        "track": track,
        "children": _jsonable(children),
        "tracks": [_track_value(item) for item in tracks],
        "receipts": _jsonable(receipts),
        "published": list(published),
        "publicDigest": public_digest,
        "instruction": semantic_orchestration_instruction_v8(),
    }


def _decision(event, ordinal, name, arguments, context, anchor):
    _exact(event, {"ordinal", "afterPublicSequence", "kind", "context", "record"}, "decision event")
    record = stable_algorithm_json({"name": name, "arguments": arguments})
    if (
        event["ordinal"] != ordinal
        or event["afterPublicSequence"] != anchor
        or event["kind"] != "decision"
        or not _same(event["context"], context)
        or event["record"] != record
    ):
        _fail("Teacher lifecycle target or its causal context differs from replay")
    return record, arguments


def _ack(event, ordinal, decision_ordinal, receipt, anchor):
    _exact(event, {"ordinal", "afterPublicSequence", "kind", "for", "receipt"}, "ack event")
    if (
        event["ordinal"] != ordinal
        or event["afterPublicSequence"] != anchor
        or event["kind"] != "ack"
        or event["for"] != decision_ordinal
        or not _same(event["receipt"], receipt)
    ):
        _fail("Teacher lifecycle acknowledgement differs from its native registry transition")


def _validate_selection_origin(sidecar):
    allowed = {"owned-interpreter-teacher", "certified_teacher"}
    if "selectionOrigin" in sidecar and sidecar["selectionOrigin"] not in allowed:
        _fail("Orchestration teacher source has an unqualified selection origin")
    source_artifact = sidecar.get("sourceArtifact")
    nested = source_artifact.get("sidecar") if type(source_artifact) is dict else None
    if type(nested) is dict:
        if "selectionOrigin" in nested and nested["selectionOrigin"] not in allowed:
            _fail("Nested orchestration source has an unqualified selection origin")
        if (
            sidecar.get("schema") == "semantic-region-evidence/1"
            and nested.get("selectionOrigin") not in allowed
        ):
            _fail("Owned semantic source is missing its qualified teacher origin")
    elif sidecar.get("schema") == "semantic-region-evidence/1":
        _fail("Owned semantic source artifact is missing its teacher evidence")


def validate_orchestration_transcript_v8(document):
    """Replay a single-child teacher transcript against source and native state.

    Producer source correspondence and frozen family membership are separately
    rechecked at corpus admission. This validator checks the exact V8 source
    row, native child result, lifecycle dependency and loss-bearing target plan.
    """
    if type(document) is not dict or type(document.get("public")) is not dict:
        _fail("A V8 public artifact is required")
    public, contexts = document["public"], document.get("contexts")
    sidecar = document.get("sidecar")
    transcript = sidecar.get("orchestrationTranscript") if type(sidecar) is dict else None
    if type(contexts) is not list or type(transcript) is not dict:
        _fail("A source-qualified orchestration transcript is required")
    _validate_selection_origin(sidecar)
    _exact(
        transcript,
        {
            "schema",
            "origin",
            "policy",
            "session",
            "coordinatorTrack",
            "ingress",
            "child",
            "events",
            "digest",
        },
        "transcript envelope",
    )
    if (
        transcript["schema"] != TRANSCRIPT_SCHEMA
        or transcript["origin"] != "source-qualified-teacher"
    ):
        _fail("Only the explicitly source-qualified teacher origin is admitted")
    expected_policy = {
        **ORCHESTRATION_POLICY,
        "digest": stable_algorithm_hash(ORCHESTRATION_POLICY),
    }
    if not _same(transcript["policy"], expected_policy):
        _fail("Teacher lifecycle policy differs from the fixed V8 profile")
    body = {key: value for key, value in transcript.items() if key != "digest"}
    transcript_digest = stable_algorithm_hash(transcript)
    if transcript["digest"] != stable_algorithm_hash(body):
        _fail("Orchestration transcript digest is invalid")

    timeline = public.get("timeline")
    if not isinstance(timeline, list) or len(timeline) < 3:
        _fail("Single-child V8 transcript requires source, root input and terminal output")
    program_rows = [
        row for row in timeline if isinstance(row, dict) and row.get("kind") == "program"
    ]
    invocation_rows = [
        row for row in timeline if isinstance(row, dict) and row.get("kind") == "invocation"
    ]
    if len(program_rows) != 1 or len(invocation_rows) != 1:
        _fail("The bounded teacher profile requires exactly one published program and root call")
    program, invocation = program_rows[0], invocation_rows[0]
    source, incoming = program.get("fields", {}), invocation.get("fields", {})
    if (
        timeline[:2] != [program, invocation]
        or program.get("sequence") != 1
        or program.get("mode") != "observed"
        or invocation.get("sequence") != 2
        or invocation.get("mode") != "observed"
        or program.get("kind") != "program"
        or invocation.get("kind") != "invocation"
        or incoming.get("activation") != "call"
        or incoming.get("parentInvocationId") is not None
        or incoming.get("closureId") is not None
        or incoming.get("taskRevision") != 1
        or incoming.get("realmPolicy") not in {"fresh", "retained"}
        or incoming.get("session") != source.get("session")
        or not isinstance(incoming.get("track"), str)
        or not incoming["track"]
        or any(row.get("kind") in {"program", "invocation"} for row in timeline[2:])
        or any(
            row.get("mode") == "generated"
            and row.get("fields", {}).get("invocationId") != incoming.get("invocationId")
            for row in timeline
        )
        or timeline[-1].get("mode") != "generated"
        or timeline[-1].get("fields", {}).get("invocationId") != incoming.get("invocationId")
    ):
        _fail("Transcript ingress or terminal public cut is not one exact root child")
    session = incoming.get("session")
    if not isinstance(session, str) or not session or transcript["session"] != session:
        _fail("Transcript session differs from the observed root input")
    if any(row.get("fields", {}).get("session") != session for row in timeline):
        _fail("Transcript public rows cross execution sessions")
    if (
        source.get("digest") != stable_algorithm_hash(source.get("source"))
        or incoming.get("programId") != source.get("programId")
        or incoming.get("sourceDigest") != source.get("digest")
    ):
        _fail("Root invocation is not bound to its published source bytes")
    try:
        binding = asdict(execution_binding(program, invocation, completion_target="track:0"))
    except (KeyError, TypeError, ValueError) as error:
        raise SemanticOrchestrationTranscriptError("Invalid source-bound root execution") from error
    binding = _jsonable(binding)
    ingress = transcript["ingress"]
    _exact(
        ingress,
        {
            "publicDigest",
            "throughSequence",
            "programSequence",
            "invocationSequence",
            "sourceDigest",
            "invocationId",
            "execution",
        },
        "ingress",
    )
    initial_public = _public_cut(public, 2)
    initial_digest = stable_algorithm_hash(initial_public)
    if (
        transcript["coordinatorTrack"] != ORCHESTRATION_POLICY["coordinator"]["track"]
        or ingress["throughSequence"] != 2
        or ingress["programSequence"] != 1
        or ingress["invocationSequence"] != 2
        or ingress["publicDigest"] != initial_digest
        or ingress["sourceDigest"] != source["digest"]
        or ingress["invocationId"] != incoming.get("invocationId")
        or not _same(ingress["execution"], binding)
    ):
        _fail("Transcript ingress does not bind the exact public source and root call")

    final_public_digest = stable_algorithm_hash(public)
    final_context_digest = stable_algorithm_hash(contexts)
    terminal_sequence = len(timeline)
    if timeline[-1].get("sequence") != terminal_sequence:
        _fail("Transcript requires the terminal child boundary as its final public row")
    try:
        from megatoken.semantic_orchestration_v8 import derive_completion

        exact_completion, exact_heap = derive_completion(public, contexts, incoming["invocationId"])
    except (KeyError, TypeError, ValueError) as error:
        raise SemanticOrchestrationTranscriptError("V8 child terminal does not replay") from error

    registry = TrackRegistry(
        max_active=ORCHESTRATION_POLICY["maxActive"],
        max_waiting=ORCHESTRATION_POLICY["maxWaiting"],
    )
    coordinator_policy = ORCHESTRATION_POLICY["coordinator"]
    source_capabilities = source.get("capabilities", [])
    coordinator_capabilities = ["text", *sorted(set(source_capabilities))]
    coordinator = registry.create(
        coordinator_policy["purpose"],
        completion=coordinator_policy["completion"],
        capabilities=tuple(dict.fromkeys(coordinator_capabilities)),
        budget=coordinator_policy["budget"],
    )
    if coordinator.address != coordinator_policy["track"]:
        _fail("Fresh lifecycle registry did not allocate the fixed coordinator address")
    child_policy = ORCHESTRATION_POLICY["child"]
    child_capabilities = list(dict.fromkeys(("text", *source_capabilities)))
    child_state = {
        "program": program,
        "invocation": invocation,
        "execution": binding,
        "track": None,
        "completion": None,
    }
    state_children = [_jsonable(child_state)]
    initial_children = _jsonable(state_children)
    tracks = list(registry.tracks.values())
    receipts, published = [], []
    public_digest = initial_digest
    events = transcript["events"]
    if type(events) is not list or len(events) != 10:
        _fail("The bounded V8 teacher transcript requires exactly ten lifecycle events")
    anchors = [
        2,
        2,
        2,
        2,
        2,
        terminal_sequence,
        terminal_sequence,
        terminal_sequence,
        terminal_sequence,
        terminal_sequence,
    ]
    for ordinal, (event, anchor) in enumerate(zip(events, anchors, strict=True), 1):
        if (
            type(event) is not dict
            or event.get("ordinal") != ordinal
            or event.get("afterPublicSequence") != anchor
        ):
            _fail("Transcript event ordering or public anchor changed")

    create_record, create_arguments = _decision(
        events[0],
        1,
        "track_create",
        {
            "purpose": child_policy["purpose"],
            "completion": child_policy["completion"],
            "budget": child_policy["budget"],
            "capabilities": child_capabilities,
            "execution": binding,
        },
        _context(
            session=session,
            track=coordinator.address,
            children=state_children,
            tracks=tracks,
            receipts=receipts,
            published=published,
            public_digest=public_digest,
        ),
        2,
    )
    registry.charge((coordinator.address,))
    child_track = registry.admit_execution(
        "teacher:semantic-orchestration/1",
        parent=coordinator.address,
        purpose=create_arguments["purpose"],
        completion=create_arguments["completion"],
        budget=create_arguments["budget"],
        capabilities=tuple(create_arguments["capabilities"]),
        execution=ExecutionBinding(**create_arguments["execution"]),
    )
    dependency = registry.completion_dependency(child_track.address)
    if child_track.address != child_policy["track"]:
        _fail("Fresh lifecycle registry did not allocate the fixed child address")
    state_children[0]["track"] = child_track.address
    tracks = list(registry.tracks.values())
    _ack(events[1], 2, 1, _teacher_commit(child_track, dependency), 2)

    _exact(events[2], {"ordinal", "afterPublicSequence", "kind", "receipt"}, "bootstrap event")
    bootstrapped = registry.activate_execution(
        child_track.address,
        source_digest=binding["source_digest"],
        input_ref=binding["input_ref"],
    )
    bootstrap_receipt = {
        "schema": "semantic-orchestration-bootstrap/8",
        "track": _track_value(bootstrapped),
        "bootstrapped": True,
    }
    if (
        events[2]["ordinal"] != 3
        or events[2]["afterPublicSequence"] != 2
        or events[2]["kind"] != "bootstrap"
        or not _same(events[2]["receipt"], bootstrap_receipt)
    ):
        _fail("Child bootstrap receipt differs from the source-bound reservation")
    tracks = list(registry.tracks.values())

    join_record, join_arguments = _decision(
        events[3],
        4,
        "track_join",
        {"address": coordinator.address, "completion_dependency": dependency},
        _context(
            session=session,
            track=coordinator.address,
            children=state_children,
            tracks=tracks,
            receipts=receipts,
            published=published,
            public_digest=public_digest,
        ),
        2,
    )
    registry.charge((coordinator.address,))
    waiting_coordinator = registry.wait_for_execution(
        coordinator.address, join_arguments["completion_dependency"], delivered=set()
    )
    _ack(events[4], 5, 4, _teacher_commit(waiting_coordinator), 2)
    tracks = list(registry.tracks.values())
    if registry.tracks[child_track.address].status != "active":
        _fail("The exact joined child did not acquire the released active slot")

    exact_child_completion = {
        "schema": "semantic-child-completion/8",
        "session": session,
        "invocationId": incoming["invocationId"],
        "track": child_track.address,
        "completion": exact_completion,
        "heap": exact_heap,
        "publicDigest": final_public_digest,
        "contextsDigest": final_context_digest,
    }
    child_evidence = transcript["child"]
    _exact(
        child_evidence,
        {
            "track",
            "dependency",
            "terminalPublicSequence",
            "publicDigest",
            "contextsDigest",
            "completion",
            "delivered",
        },
        "child evidence",
    )
    if (
        child_evidence["track"] != child_track.address
        or not _same(child_evidence["dependency"], dependency)
        or child_evidence["terminalPublicSequence"] != terminal_sequence
        or child_evidence["publicDigest"] != final_public_digest
        or child_evidence["contextsDigest"] != final_context_digest
        or not _same(child_evidence["completion"], exact_child_completion)
    ):
        _fail("Child completion does not bind its exact V8 terminal public/context cut")
    child_target_count = sum(
        row.get("mode") == "generated"
        and row.get("fields", {}).get("invocationId") == incoming["invocationId"]
        for row in timeline
    )
    if child_target_count > child_policy["budget"]:
        _fail("Source has more logical child targets than the teacher work budget allows")
    registry.charge((child_track.address,) * child_target_count)
    delivered = registry.complete_execution(
        child_track.address,
        kind=exact_completion["kind"],
        value=exact_completion["value"],
    )
    registry.wake_ready_executions(delivered={dependency["id"]})
    if not _same(child_evidence["delivered"], delivered):
        _fail("Delivered typed completion differs from the native dependency receipt")
    state_children[0]["completion"] = exact_child_completion
    receipts = [delivered]
    public_digest = final_public_digest
    tracks = list(registry.tracks.values())
    _exact(
        events[5],
        {"ordinal", "afterPublicSequence", "kind", "receipt", "delivered"},
        "completion event",
    )
    if (
        events[5]["ordinal"] != 6
        or events[5]["afterPublicSequence"] != terminal_sequence
        or events[5]["kind"] != "completion"
        or not _same(events[5]["receipt"], exact_child_completion)
        or not _same(events[5]["delivered"], delivered)
    ):
        _fail("Completion event differs from the source-qualified native child delivery")

    progress = stable_algorithm_json(exact_completion)
    progress_record, progress_arguments = _decision(
        events[6],
        7,
        "track_progress",
        {"address": coordinator.address, "progress": progress},
        _context(
            session=session,
            track=coordinator.address,
            children=state_children,
            tracks=tracks,
            receipts=receipts,
            published=published,
            public_digest=public_digest,
        ),
        terminal_sequence,
    )
    registry.charge((coordinator.address,))
    progress_coordinator = registry.change(
        coordinator.address, "progress", progress=progress_arguments["progress"]
    )
    published = [incoming["invocationId"]]
    tracks = list(registry.tracks.values())
    _ack(events[7], 8, 7, _teacher_commit(progress_coordinator), terminal_sequence)

    complete_record, complete_arguments = _decision(
        events[8],
        9,
        "track_complete",
        {"address": coordinator.address},
        _context(
            session=session,
            track=coordinator.address,
            children=state_children,
            tracks=tracks,
            receipts=receipts,
            published=published,
            public_digest=public_digest,
        ),
        terminal_sequence,
    )
    registry.charge((coordinator.address,))
    complete_coordinator = registry.change(coordinator.address, "complete")
    _ack(events[9], 10, 9, _teacher_commit(complete_coordinator), terminal_sequence)
    if (
        registry.tracks[child_evidence["track"]].status != "completed"
        or registry.tracks[coordinator.address].status != "completed"
        or set(published) != {incoming["invocationId"]}
    ):
        _fail("Teacher transcript did not complete and publish its actual child")

    decisions = {
        1: ("track_create", create_arguments, create_record),
        4: ("track_join", join_arguments, join_record),
        7: ("track_progress", progress_arguments, progress_record),
        9: ("track_complete", complete_arguments, complete_record),
    }
    return {
        "transcript": transcript,
        "transcriptDigest": transcript_digest,
        "public": public,
        "contexts": contexts,
        "initialPublic": initial_public,
        "initialChildren": initial_children,
        "session": session,
        "coordinatorTrack": coordinator.address,
        "childTrack": child_track.address,
        "invocationId": incoming["invocationId"],
        "dependency": dependency,
        "decisions": decisions,
        "recordCount": len(decisions),
        "childTargetCount": child_target_count,
        "terminalSequence": terminal_sequence,
    }


def orchestration_selection_instruction_v8(*, continuation=False):
    return semantic_orchestration_instruction_v8(continuation=continuation)


def orchestration_transcript_rows_v8(validation):
    """Return exact loss-bearing targets and observations, keyed by public row."""
    transcript = validation["transcript"]
    buckets, actions, dependencies = {}, [], []

    def add(
        anchor,
        identity,
        role,
        payload,
        *,
        masked=False,
        action_id=None,
        result_of=None,
        track=None,
    ):
        spec = {
            "recordId": identity,
            "role": role,
            "payload": payload,
            "policyMask": masked,
            "provenance": "teacher" if masked else "executed",
            "actionId": action_id,
            "resultOf": result_of,
            "executionId": action_id or result_of,
            "instruction": orchestration_selection_instruction_v8() if masked else "",
            # The V8 serving coordinator owns all lifecycle inputs, actions,
            # receipts and child completion delivery.
            "track": track or validation["coordinatorTrack"],
        }
        buckets.setdefault(anchor, []).append(spec)
        return identity

    add(
        2,
        "semantic:v8:orchestration:ingress",
        "user",
        canonical_semantic_observation(
            semantic_orchestration_input_v8(
                public=validation["initialPublic"], children=validation["initialChildren"]
            )
        ),
    )

    for event in transcript["events"]:
        ordinal, anchor, kind = event["ordinal"], event["afterPublicSequence"], event["kind"]
        if kind == "decision":
            name, arguments, record = validation["decisions"][ordinal]
            action_id = f"semantic:v8:orchestration:action:{ordinal}"
            context_id = add(
                anchor,
                f"semantic:v8:orchestration:context:{ordinal}",
                "system",
                canonical_semantic_observation(event["context"]),
            )
            target_id = add(
                anchor,
                f"semantic:v8:orchestration:target:{ordinal}",
                "assistant",
                record.encode("utf-8"),
                masked=True,
                action_id=action_id,
            )
            actions.append(
                {
                    "actionId": action_id,
                    "callId": action_id,
                    "toolName": name,
                    "arguments": arguments,
                    "targetId": target_id,
                    "serializedSha256": hashlib.sha256(record.encode("utf-8")).hexdigest(),
                }
            )
            dependencies.append((context_id, target_id, "v8-orchestration-context"))
        elif kind == "ack":
            receipt = event["receipt"]
            acknowledgement = semantic_orchestration_ack_observation_v8(
                track=receipt["track"],
                completion_dependency=receipt.get("completion_dependency"),
            )
            add(
                anchor,
                f"semantic:v8:orchestration:ack:{ordinal}",
                "environment",
                canonical_semantic_observation(acknowledgement),
                result_of=f"semantic:v8:orchestration:action:{event['for']}",
            )
        elif kind == "bootstrap":
            child = validation["initialChildren"][0]
            for name in ("program", "invocation"):
                add(
                    anchor,
                    f"semantic:v8:orchestration:bootstrap-input:{name}",
                    "user",
                    json.dumps(
                        child[name], sort_keys=True, separators=(",", ":"), allow_nan=False
                    ).encode("utf-8"),
                    track=validation["childTrack"],
                )
            add(
                anchor,
                f"semantic:v8:orchestration:bootstrap:{ordinal}",
                "environment",
                canonical_semantic_observation(event["receipt"]),
            )
        elif kind == "completion":
            add(
                anchor,
                f"semantic:v8:orchestration:delivery:{ordinal}",
                "environment",
                json.dumps(
                    event["delivered"], sort_keys=True, separators=(",", ":"), allow_nan=False
                ).encode("utf-8"),
            )
            add(
                anchor,
                f"semantic:v8:orchestration:completion:{ordinal}",
                "environment",
                canonical_semantic_observation(event["receipt"]),
            )
        else:
            _fail("Unknown teacher transcript event during materialization")
    return {"buckets": buckets, "actions": actions, "dependencies": dependencies}


def validate_orchestration_records_v8(episode, validation):
    """Ensure an ordinary prepared episode contains every exact transcript row."""
    materialized = orchestration_transcript_rows_v8(validation)
    by_logical = {}
    for record in getattr(episode, "records", ()):
        by_logical.setdefault(record.logical_id or record.record_id, []).append(record)
    logical_order = []
    for logical_id, parts in by_logical.items():
        indices = [part.part_index for part in parts]
        if indices != list(range(len(parts))) or any(
            part.part_count != len(parts) for part in parts
        ):
            return False
        logical_order.append(logical_id)
    events = {
        event.record_id: event
        for group in getattr(episode, "meridians", ())
        for event in group.events
    }
    records = tuple(getattr(episode, "records", ()))
    meridians = tuple(getattr(episode, "meridians", ()))
    if len(records) != len(meridians) or any(
        group.meridian_id != index
        or len(group.events) != 1
        or group.events[0].record_id != record.record_id
        or group.events[0].meridian_id != index
        for index, (record, group) in enumerate(zip(records, meridians, strict=True))
    ):
        return False
    tracks = {track.track_id: track for track in getattr(episode, "tracks", ())}
    expected_track_specs = {
        validation["coordinatorTrack"]: (
            "assistant",
            ORCHESTRATION_POLICY["coordinator"]["purpose"],
        ),
        validation["childTrack"]: ("assistant", ORCHESTRATION_POLICY["child"]["purpose"]),
    }
    if set(tracks) != set(expected_track_specs) or any(
        tracks[track_id].role != role
        or tracks[track_id].default_modality != "text"
        or tracks[track_id].description != purpose
        or tracks[track_id].settings_json != "{}"
        for track_id, (role, purpose) in expected_track_specs.items()
    ):
        return False
    expected_transcript_ids = {
        spec["recordId"] for specs in materialized["buckets"].values() for spec in specs
    }
    expected_public_ids = set()
    expected_logical_order = []
    public_dependencies = []
    generated_contexts = {
        row["sequence"]: context
        for context, row in zip(
            validation["contexts"],
            (row for row in validation["public"]["timeline"] if row["mode"] == "generated"),
            strict=True,
        )
    }

    def valid_physical_record(
        identity,
        *,
        payload,
        role,
        policy_mask,
        track,
        instruction,
        provenance,
    ):
        parts = sorted(by_logical.get(identity, ()), key=lambda item: item.part_index)
        if not parts or b"".join(part.payload for part in parts) != payload:
            return False
        for part in parts:
            event = events.get(part.record_id)
            expected_instruction = instruction(part) if callable(instruction) else instruction
            if (
                part.role != role
                or part.policy_mask != policy_mask
                or part.provenance not in provenance
                or event is None
                or event.track_id != track
                or event.role != role
                or event.modality != "text"
                or event.instruction != expected_instruction
            ):
                return False
        return True

    for sequence, row in enumerate(validation["public"]["timeline"], 1):
        if sequence > 2:
            if row["mode"] == "generated":
                context_id = f"semantic:v8:context:{sequence}"
                target_id = f"semantic:v8:target:{sequence}"
                expected_public_ids.update((context_id, target_id))
                expected_logical_order.extend((context_id, target_id))
                if not valid_physical_record(
                    context_id,
                    payload=canonical_semantic_observation(generated_contexts[sequence]),
                    role="system",
                    policy_mask=False,
                    track=validation["childTrack"],
                    instruction="",
                    provenance={"executed"},
                ) or not valid_physical_record(
                    target_id,
                    payload=row["fields"]["line"].encode("utf-8"),
                    role="assistant",
                    policy_mask=True,
                    track=validation["childTrack"],
                    instruction=lambda part: semantic_selection_instruction_v8(
                        continuation=part.part_index > 0
                    ),
                    provenance={"unknown", "teacher"},
                ):
                    return False
                context_parts = sorted(by_logical[context_id], key=lambda item: item.part_index)
                target_parts = sorted(by_logical[target_id], key=lambda item: item.part_index)
                public_dependencies.append(
                    (context_parts[-1].record_id, target_parts[0].record_id, "v8-public-context")
                )
            else:
                identity = f"semantic:v8:observation:{sequence}"
                expected_public_ids.add(identity)
                expected_logical_order.append(identity)
                role = "user" if row["kind"] in {"program", "invocation"} else "environment"
                if not valid_physical_record(
                    identity,
                    payload=canonical_semantic_observation(row),
                    role=role,
                    policy_mask=False,
                    track=validation["childTrack"],
                    instruction="",
                    provenance={"executed"},
                ):
                    return False
        for spec in materialized["buckets"].get(sequence, ()):
            expected_logical_order.append(spec["recordId"])

    expected_v8_ids = expected_transcript_ids | expected_public_ids
    if {
        identity for identity in by_logical if identity.startswith("semantic:v8:")
    } != expected_v8_ids:
        return False
    if logical_order != expected_logical_order or tuple(
        part.record_id for logical_id in expected_logical_order for part in by_logical[logical_id]
    ) != tuple(record.record_id for record in records):
        return False

    def invalid_part(part, spec):
        event = events.get(part.record_id)
        instruction = (
            orchestration_selection_instruction_v8(continuation=part.part_index > 0)
            if spec["policyMask"]
            else ""
        )
        return (
            part.role != spec["role"]
            or part.policy_mask != spec["policyMask"]
            or part.provenance != spec["provenance"]
            or part.action_id != spec["actionId"]
            or part.result_of != spec["resultOf"]
            or part.execution_id != spec["executionId"]
            or event is None
            or event.track_id != spec["track"]
            or event.role != spec["role"]
            or event.modality != "text"
            or event.instruction != instruction
        )

    for specs in materialized["buckets"].values():
        for spec in specs:
            parts = sorted(by_logical.get(spec["recordId"], ()), key=lambda item: item.part_index)
            if (
                not parts
                or b"".join(part.payload for part in parts) != spec["payload"]
                or any(invalid_part(part, spec) for part in parts)
            ):
                return False
    expected_dependencies = set(public_dependencies)
    for writer_id, reader_id, kind in materialized["dependencies"]:
        writer_parts = sorted(by_logical.get(writer_id, ()), key=lambda item: item.part_index)
        reader_parts = sorted(by_logical.get(reader_id, ()), key=lambda item: item.part_index)
        if not writer_parts or not reader_parts:
            return False
        expected_dependencies.add((writer_parts[-1].record_id, reader_parts[0].record_id, kind))
    logical_by_record = {
        record.record_id: record.logical_id or record.record_id
        for record in getattr(episode, "records", ())
    }
    actual_orchestration_dependencies = {
        (edge.writer_id, edge.reader_id, edge.kind)
        for edge in getattr(episode, "dependencies", ())
        if logical_by_record.get(edge.writer_id, "").startswith("semantic:v8:")
        or logical_by_record.get(edge.reader_id, "").startswith("semantic:v8:")
    }
    if expected_dependencies != actual_orchestration_dependencies:
        return False
    actions = {item.action_id: item for item in getattr(episode, "actions", ())}
    if len(actions) != len(getattr(episode, "actions", ())) or len(actions) != len(
        materialized["actions"]
    ):
        return False
    for expected in materialized["actions"]:
        action = actions.get(expected["actionId"])
        if (
            action is None
            or action.call_id != expected["callId"]
            or action.tool_name != expected["toolName"]
            or not _same(json.loads(action.arguments_json), expected["arguments"])
            or action.record_ids
            != tuple(
                part.record_id
                for part in sorted(
                    by_logical.get(expected["targetId"], ()), key=lambda item: item.part_index
                )
            )
            or action.serialized_sha256 != expected["serializedSha256"]
            or action.execution_id != expected["actionId"]
            or action.schema_id != ORCHESTRATION_RECORD_SCHEMA
        ):
            return False
    track_actions = [item for item in actions.values() if item.tool_name.startswith("track_")]
    return len(track_actions) == len(materialized["actions"])
