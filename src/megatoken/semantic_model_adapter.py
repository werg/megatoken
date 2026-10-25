"""Stateful Megatoken selected-semantic serving bridge.

This is the model-owned inference path. It does not use the tracer teacher,
native JS execution, expected results, or a rule-based repair of selections.
A complete model output is committed by MultiTrackRuntime and then validated by
the independent JS selected controller on the client side.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

from megatoken.generation.structure import ActionSpec
from megatoken.lifecycle import ExecutionBinding


class SemanticSelectionError(ValueError):
    pass


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise SemanticSelectionError(f"{name} must be a nonempty public identity")
    return value


def _validate_prefix(prefix: Any) -> tuple[dict, dict, list[dict]]:
    if not isinstance(prefix, dict) or prefix.get("schema") != "selected-inference-prefix/1":
        raise SemanticSelectionError("Unknown selected-model inference prefix version")
    events = prefix.get("committed")
    if not isinstance(events, list) or len(events) < 2 or any(not isinstance(e, dict) for e in events):
        raise SemanticSelectionError("Model execution needs published code before invocation")
    programs = [
        event
        for event in events
        if event.get("kind") == "program"
        and event.get("programId") == prefix.get("programId")
    ]
    calls = [
        event
        for event in events
        if event.get("kind") == "invocation"
        and event.get("invocationId") == prefix.get("invocationId")
    ]
    if len(programs) != 1 or len(calls) != 1 or events.index(programs[0]) >= events.index(calls[0]):
        raise SemanticSelectionError("Exactly one code publication must precede invocation")
    source, digest = programs[0].get("source"), programs[0].get("digest")
    if (
        not isinstance(source, str)
        or hashlib.sha256(source.encode("utf-8")).hexdigest() != digest
        or digest != prefix.get("sourceDigest")
        or prefix.get("source") != source
        or prefix.get("entry") != calls[0].get("entry")
        or programs[0].get("programId") != calls[0].get("programId")
        or calls[0].get("track") != prefix.get("trackId")
        or calls[0].get("realmId") != prefix.get("realmId")
        or calls[0].get("completionPolicy") != prefix.get("completionPolicy")
    ):
        raise SemanticSelectionError("Code, entry, digest, realm or completion changed")
    session = programs[0].get("session")
    if not isinstance(session, str) or not session or any(
        event.get("session") != session for event in events
    ):
        raise SemanticSelectionError("Model prefix crosses session authority")
    positions = [item.get("sequence") for item in events]
    if not all(type(x) is int and x > 0 for x in positions) or positions != sorted(set(positions)):
        raise SemanticSelectionError("Selected prefix has reordered or duplicated causal receipts")
    for item in events:
        mode = item.get("mode")
        if item.get("kind") in {"program", "invocation", "dispatch", "result", "event",
                                "handler", "handler-close"}:
            if mode != "observed":
                raise SemanticSelectionError("Only the harness can publish observed results")
        elif item.get("kind") == "semantic":
            if mode != "generated" or not isinstance(item.get("line"), str):
                raise SemanticSelectionError("Selected records require real committed model bytes")
        else:
            raise SemanticSelectionError("Unregistered model-visible public event family")
    return programs[0], calls[0], events


@dataclass
class _Work:
    session_key: str
    program_id: str
    invocation_id: str
    requested_track: str
    actual_track: str
    source_event: str | None
    seen: list[bytes] = field(default_factory=list)
    selected: list[bytes] = field(default_factory=list)
    selections: int = 0
    completed: bool = False


@dataclass
class _Session:
    runtime_session: Any
    work: dict[str, _Work] = field(default_factory=dict)
    programs: dict[str, str] = field(default_factory=dict)


class SelectedSemanticInference:
    """Use actual MultiTrackRuntime generation, committed state, and context.

    The caller retains the first-party v4 public episode; this adapter retains
    the matching Megatoken model continuation and pins its codec/track history.
    A bad model selection stays bad. The broker never gets a fabricated call.
    """

    def __init__(
        self,
        runtime: Any,
        *,
        effort: int = 1,
        max_tokens: int = 512,
        work_budget: int = 4096,
        max_sessions: int = 32,
    ) -> None:
        if not 1 <= effort <= 6 or not 1 <= max_tokens <= 8192:
            raise ValueError("Semantic inference effort/token budget is invalid")
        if not 1 <= work_budget <= 100000 or not 1 <= max_sessions <= 256:
            raise ValueError("Semantic inference session capacity is invalid")
        self.runtime = runtime
        self.effort = effort
        self.max_tokens = max_tokens
        self.work_budget = work_budget
        self.max_sessions = max_sessions
        self.sessions: dict[str, _Session] = {}
        self.lock = RLock()

    def _opened(self, session_key: str) -> _Session:
        existing = self.sessions.get(session_key)
        if existing is not None:
            return existing
        if len(self.sessions) >= self.max_sessions:
            raise SemanticSelectionError("Selected session capacity exhausted")
        session = _Session(self.runtime.open(
            metadata={"task": "semantic-execution/4", "modality": "text"},
            effort=self.effort, memory_writes=False,
        ))
        self.sessions[session_key] = session
        return session

    def _observe(self, session: Any, track: str, event: dict) -> None:
        # Runtime chunking is a transport concern; the source and input remain
        # ordered observed records, not trainable model outputs.
        kind = event["kind"]
        role = "user" if kind in ("program", "invocation") else "environment"
        self.runtime.observe_text(
            session, track, _canonical(event), role=role, effort=self.effort,
        )

    def _bootstrap(self, state: _Session, program: dict, invocation: dict) -> _Work:
        from megatoken.lifecycle import TrackDescription

        track = _text(invocation["track"], "requested track")
        program_id = _text(program["programId"], "program ID")
        invocation_id = _text(invocation["invocationId"], "invocation ID")
        digest = _text(program["digest"], "source digest")
        data_ref = hashlib.sha256(_canonical(invocation["inputGraph"])).hexdigest()
        grants = tuple(program.get("capabilities", []))
        binding = ExecutionBinding(
            program_id=program_id,
            source_digest=digest,
            entry=_text(invocation["entry"], "invocation entry"),
            invocation_id=invocation_id,
            realm_id=_text(invocation["realmId"], "realm"),
            trigger_id=_text(invocation["triggerId"], "trigger"),
            input_ref=data_ref,
            realm_policy=invocation.get("realmPolicy", "fresh"),
            completion_mode=(
                "owned-quiescence"
                if invocation["completionPolicy"] == "owned-quiescence"
                else "call-return"
            ),
            capability_grants=grants,
        )
        description = self.runtime.track_action(
            state.runtime_session,
            "create",
            purpose=f"Execute {program_id}.{invocation['entry']}",
            completion=f"Invocation {invocation_id} completion",
            capabilities=("text", *grants),
            budget=self.work_budget,
            execution=binding,
        )
        if not isinstance(description, TrackDescription) or description.status != "waiting":
            raise SemanticSelectionError("Dynamic execution track must await observed bootstrap")
        # Immutable program is observed on this newly allocated execution track
        # before input and invocation; its digest must match the binding.
        self._observe(state.runtime_session, description.address, program)
        self._observe(state.runtime_session, description.address, invocation)
        state.runtime_session.tracks.activate_execution(
            description.address, source_digest=digest, input_ref=data_ref,
        )
        # Existing task/model context may evict source; code remains
        # source-event retrievable from the context archive when needed.
        work = _Work(
            session_key=program["session"], program_id=program_id,
            invocation_id=invocation_id, requested_track=track,
            actual_track=description.address, source_event=None,
        )
        state.work[invocation_id] = work
        state.programs[program_id] = digest
        return work

    def _advance(self, state: _Session, work: _Work, events: list[dict]) -> None:
        for event in events:
            encoded = _canonical(event)
            if event["kind"] == "semantic":
                expected = work.selected.pop(0) if work.selected else None
                if expected is None or event["line"].encode("utf-8") != expected:
                    raise SemanticSelectionError(
                        "Model transcript does not match its previously committed selection"
                    )
                # Selected meridians are already in Megatoken's model state;
                # never ingest them as fresh observed evidence a second time.
            elif event["kind"] not in ("program", "invocation"):
                self._observe(state.runtime_session, work.actual_track, event)
            work.seen.append(encoded)

    def select(self, prefix: dict) -> dict:
        program, invocation, events = _validate_prefix(prefix)
        with self.lock:
            session_key = _text(program["session"], "session")
            state = self._opened(session_key)
            id_ = _text(invocation["invocationId"], "invocation")
            work = state.work.get(id_)
            if work is None:
                work = self._bootstrap(state, program, invocation)
                work.seen.extend(_canonical(event) for event in (program, invocation))
            if (
                work.program_id != program["programId"]
                or work.requested_track != invocation["track"]
                or work.completed
            ):
                raise SemanticSelectionError("Track reuse, provenance mismatch or completed invocation")
            # Check exact old bytes to prevent retroactive teacher/result edits.
            if len(events) < len(work.seen) or any(
                before != _canonical(events[index])
                for index, before in enumerate(work.seen)
            ):
                raise SemanticSelectionError("Previously delivered model-visible prefix was rewritten")
            self._advance(state, work, events[len(work.seen):])
            # A previously selected record must be committed and reported back
            # before another generated semantic action can be sampled.
            if work.selected:
                raise SemanticSelectionError("Previous model selection was not committed")
            if not state.runtime_session.tracks.selectable(
                work.actual_track, state.runtime_session.completed_events
            ):
                raise SemanticSelectionError("Selected execution track has not become runnable")
            config = self.runtime.generator.generator.config
            maximum = min(self.max_tokens, config.max_capacity)
            instruction = (
                b"Emit exactly one complete canonical semantic-trace/3 witness. "
                b"Use only the observed source, this invocation and prior selected "
                b"semantics. An external call is a validated request with already "
                b"evaluated arguments and source span. Never predict observations, "
                b"events, native results or future tool receipts."
            )
            # The *real* checkpoint/recurrence and native tokenizer are used.
            generated = self.runtime.generate(
                state.runtime_session,
                tracks=(work.actual_track,),
                specs=(ActionSpec(text_kind="utf8", min_tokens=1,
                                  max_tokens=maximum, require_terminal=True),),
                controls=(instruction,),
                effort=self.effort,
            )
            tape = generated.tape
            if (
                len(tape.payloads) != 1
                or not tape.valid[0]
                or tape.modalities[0] != "text"
                or tape.tracks[0] != work.actual_track
            ):
                raise SemanticSelectionError(
                    "Model did not render one complete valid text action"
                )
            original = tape.payloads[0]
            if not isinstance(original, bytes):
                raise SemanticSelectionError("Selected model action must preserve exact UTF-8 bytes")
            try:
                output = original.decode("utf-8", errors="strict")
            except UnicodeDecodeError as error:
                raise SemanticSelectionError("Model emitted invalid UTF-8") from error
            work.selected.append(original)
            work.selections += 1
            return {
                "schema": "selected-model-output/1",
                "record": output,
                "selection": {
                    "modelTrack": work.actual_track,
                    "ownerTrack": work.requested_track,
                    "selectedEventId": tape.event_ids[0],
                    "meridian": tape.meridian_id,
                    "codecFingerprint": self.runtime.generator.generator.codec.fingerprint,
                    "modelSelected": True,
                },
            }

    def close_invocation(self, session: str, invocation: str, prefix: dict) -> None:
        """A complete public return/end must acknowledge the last actual selected tape."""
        program, inv, events = _validate_prefix(prefix)
        if program["session"] != session or inv["invocationId"] != invocation:
            raise SemanticSelectionError("Completion belongs to another selected invocation")
        if not events or events[-1].get("kind") != "semantic" or not events[-1].get("line", "").startswith("end "):
            raise SemanticSelectionError("Model completion needs the selected terminal end")
        with self.lock:
            state = self.sessions.get(session)
            if state is None or invocation not in state.work:
                raise SemanticSelectionError("Unknown selected invocation")
            work = state.work[invocation]
            if len(events) < len(work.seen) or any(
                before != _canonical(events[index])
                for index, before in enumerate(work.seen)
            ):
                raise SemanticSelectionError("Completion rewrites earlier model history")
            self._advance(state, work, events[len(work.seen):])
            if work.selected:
                raise SemanticSelectionError("Cannot complete before last selection commits")
            state.runtime_session.tracks.change(work.actual_track, "complete")
            work.completed = True

    def cancel_invocation(self, session: str, invocation: str) -> None:
        with self.lock:
            state = self.sessions.get(session)
            if state is None or invocation not in state.work:
                raise SemanticSelectionError("Cannot cancel unknown execution")
            work = state.work[invocation]
            state.runtime_session.tracks.change(work.actual_track, "cancel")
            work.completed = True
