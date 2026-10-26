"""Actual free-running coordinator selections over the shared native child runtime.

Only source/input proposals cross ingress. Guest witness generation and host
execution remain the selected driver/broker's responsibilities in the same model
session. Controlled fixture decisions have an explicit separate authority.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict

from megatoken.contracts import ROLE_TO_ID
from megatoken.generation.structure import ActionSpec
from megatoken.lifecycle import ExecutionBinding, lifecycle_tools
from megatoken.semantic_model_adapter import (
    SemanticSelectionError,
    _canonical,
    _selection_fragments,
    _validate_prefix,
)


def validate_restored_orchestration(adapter, value):
    if (
        not isinstance(value, dict)
        or set(value)
        != {
            "session",
            "track",
            "representation",
            "origin",
            "children",
            "decisions",
            "fixtureDigest",
        }
        or value["representation"] != "compact"
        or value["origin"] not in {"model", "controlled"}
        or (value["origin"] == "controlled" and not adapter.allow_controlled_lifecycle)
        or not isinstance(value["children"], list)
        or not value["children"]
        or not isinstance(value["decisions"], list)
    ):
        raise SemanticSelectionError("Invalid or unauthorized restored orchestration authority")
    for child in value["children"]:
        binding = ExecutionBinding(**child["execution"])
        if (
            binding.invocation_id != child["invocation"]["invocationId"]
            or binding.source_digest != child["program"]["digest"]
        ):
            raise SemanticSelectionError(
                "Restored coordinator changed its immutable child proposal"
            )


def open_orchestration(
    adapter, *, session, proposals, maxActive=1, representation="compact", origin="model"
):
    if (
        not isinstance(session, str)
        or not session
        or not isinstance(proposals, list)
        or not 1 <= len(proposals) <= 16
        or type(maxActive) is not int
        or not 1 <= maxActive <= 64
        or representation != "compact"
        or origin not in {"model", "controlled"}
        or (origin == "controlled" and not adapter.allow_controlled_lifecycle)
    ):
        raise SemanticSelectionError(
            "Orchestration requires declared source proposals and authorized authority"
        )
    parsed = []
    for prefix in proposals:
        program, invocation, events = _validate_prefix(prefix)
        if (
            len(events) != 2
            or program["session"] != session
            or invocation["parentInvocationId"] is not None
        ):
            raise SemanticSelectionError(
                "Coordinator ingress accepts only its observed source and root inputs"
            )
        parsed.append((program, invocation))
    if len({inv["invocationId"] for _, inv in parsed}) != len(parsed):
        raise SemanticSelectionError("Coordinator child invocation identities must be unique")
    if session in adapter.sessions:
        raise SemanticSelectionError("Orchestration opening requires a fresh native session")
    state = adapter._opened(session)
    native = state.runtime_session
    native.tracks.max_active = maxActive
    grants = sorted({grant for program, _ in parsed for grant in program["capabilities"]})
    coordinator = adapter.runtime.track_action(
        native,
        "create",
        purpose="Execute the observed child tasks in order",
        completion="Every child result is delivered and published",
        capabilities=("text", *grants),
        budget=min(100000, (adapter.work_budget + 4) * len(parsed) + 1),
    )
    children = []
    for program, invocation in parsed:
        reference = adapter.runtime.register_execution_context(
            native, program=program, invocation=invocation
        )
        binding = ExecutionBinding(
            program_id=program["programId"],
            source_digest=program["digest"],
            entry=invocation["entry"],
            invocation_id=invocation["invocationId"],
            realm_id=invocation["realmId"],
            trigger_id=invocation["triggerId"],
            input_ref=reference,
            realm_policy=invocation["realmPolicy"],
            completion_mode=invocation["completionPolicy"],
            capability_grants=tuple(program["capabilities"]),
        )
        children.append(
            {"execution": asdict(binding), "program": program, "invocation": invocation}
        )
    state.orchestration = {
        "session": session,
        "track": coordinator.address,
        "representation": representation,
        "origin": origin,
        "children": children,
        "decisions": [],
        "fixtureDigest": hashlib.sha256(_canonical(proposals)).hexdigest(),
    }
    maximum = adapter.runtime.generator.generator.config.max_history_chunks
    adapter.runtime.configure_context(
        native, track=coordinator.address, local_events=maximum, cross_events=0, pinned_events=()
    )
    before = set(native.context_archive.events)
    adapter.runtime.observe_text(
        native,
        coordinator.address,
        _canonical(
            {
                "schema": "semantic-orchestration-input/4",
                "task": "Create each observed child in order, join its runtime completion, publish its exact returned kind/value as progress, then complete this coordinator.",
                "children": children,
            }
        ),
        role="user",
        effort=adapter.effort,
    )
    supplied = set(native.context_archive.events) - before
    visible = adapter.runtime.context_status(native, track=coordinator.address)["visible_events"]
    if not supplied or not supplied <= set(visible):
        adapter.runtime.track_action(native, "cancel", address=coordinator.address)
        raise SemanticSelectionError("Coordinator source proposals exceed visible native context")
    return status_orchestration(adapter, session=session, track=coordinator.address)


def status_orchestration(adapter, *, session, track):
    state = adapter.sessions.get(session)
    if state is None or state.orchestration is None or state.orchestration["track"] != track:
        raise SemanticSelectionError("Unknown coordinator session or track")
    native = state.runtime_session
    roles = {number: name for name, number in ROLE_TO_ID.items()}
    return {
        "schema": "semantic-orchestration-state/4",
        **state.orchestration,
        "status": native.tracks.tracks[track].status,
        "tracks": [asdict(description) for description in native.tracks.tracks.values()],
        "receipts": native.execution_receipts,
        "observations": [
            {
                "eventId": identity,
                "track": event.track,
                "role": roles[event.role_id],
                "payloadHex": event.payload.hex(),
                "terminal": event.terminal,
            }
            for identity, (event, _) in native.context_archive.events.items()
        ],
    }


def cancel_orchestration(adapter, *, session, track, reason):
    status_orchestration(adapter, session=session, track=track)
    if not isinstance(reason, str) or not reason:
        raise SemanticSelectionError("Coordinator cancellation requires an observed reason")
    state = adapter.sessions[session]
    native = state.runtime_session
    if native.tracks.tracks[track].status not in {"completed", "cancelled"}:
        adapter.runtime.track_action(native, "cancel", address=track)
    for description in tuple(native.tracks.tracks.values()):
        if description.status not in {"completed", "cancelled"}:
            adapter.runtime.track_action(native, "cancel", address=description.address)
    for work in state.work.values():
        if native.tracks.tracks[work.actual_track].status == "cancelled":
            work.completed = True
    native.execution_receipts.append(
        {"schema": "semantic-orchestration-cancel/4", "track": track, "reason": reason}
    )
    return status_orchestration(adapter, session=session, track=track)


def step_orchestration(adapter, *, session, track, controlledRecord=None):
    snapshot = status_orchestration(adapter, session=session, track=track)
    state = adapter.sessions[session]
    native = state.runtime_session
    if not native.tracks.selectable(track, native.completed_events):
        raise SemanticSelectionError(
            "Coordinator is blocked or closed until actual child completion"
        )
    if snapshot["origin"] == "controlled":
        if not adapter.allow_controlled_lifecycle or not isinstance(controlledRecord, str):
            raise SemanticSelectionError(
                "Controlled lifecycle requires explicit fixture bytes and server authority"
            )
        group = adapter.runtime.commit_fixture_action(
            native,
            track=track,
            payload=controlledRecord.encode(),
            fixture_digest=snapshot["fixtureDigest"],
            decision=len(snapshot["decisions"]),
            family="lifecycle",
        )
    else:
        if controlledRecord is not None:
            raise SemanticSelectionError("Model orchestration cannot receive controlled decisions")
        control = _canonical(
            {
                "instruction": "Emit exactly one complete lifecycle tool action JSON object {name,arguments}. Use only the observed child proposals and delivered runtime receipts. Never invent completion dependencies or child results. Use this coordinator's actual address for join, progress and completion.",
                "coordinatorTrack": track,
                "tools": lifecycle_tools(),
            }
        )
        group = adapter.runtime.generate(
            native,
            tracks=(track,),
            specs=(
                ActionSpec(
                    text_kind="utf8",
                    min_tokens=1,
                    max_tokens=min(
                        adapter.max_tokens, adapter.runtime.generator.generator.config.max_capacity
                    ),
                    require_terminal=True,
                ),
            ),
            controls=(control,),
            effort=adapter.effort,
        )
    tape = group.tape
    evidence = {
        "origin": snapshot["origin"],
        "representation": snapshot["representation"],
        "payloadHex": [
            payload.hex() if isinstance(payload, bytes) else None for payload in tape.payloads
        ],
        "selectedEventIds": list(tape.event_ids),
        "meridian": tape.meridian_id,
        "fragments": _selection_fragments(tape),
    }

    def refuse(reason):
        state.orchestration["decisions"].append({"failure": reason, "evidence": evidence})
        adapter.runtime.track_action(native, "cancel", address=track)
        raise SemanticSelectionError(reason, evidence=evidence)

    if (
        len(tape.payloads) != 1
        or not tape.valid[0]
        or tape.modalities[0] != "text"
        or tape.tracks[0] != track
        or not tape.terminal[0]
        or not isinstance(tape.payloads[0], bytes)
    ):
        refuse("Coordinator did not select one complete valid lifecycle action")
    try:
        record = tape.payloads[0].decode("utf-8")
    except UnicodeDecodeError:
        refuse("Coordinator selected invalid UTF-8")
    outcomes = adapter.runtime.execute_lifecycle(native, group)
    selection = {
        "modelSelected": snapshot["origin"] == "model",
        "origin": snapshot["origin"],
        "selectedEventId": tape.event_ids[0],
        "meridian": tape.meridian_id,
        "modelTrack": track,
        "codecFingerprint": adapter.runtime.generator.generator.codec.fingerprint,
        "fragments": _selection_fragments(tape),
    }
    if len(outcomes) != 1:
        refuse("Coordinator selected an invalid or unrecognized lifecycle action")
    decision = {"record": record, "selection": selection, "outcomes": outcomes}
    state.orchestration["decisions"].append(decision)
    return {**status_orchestration(adapter, session=session, track=track), "decision": decision}
