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
    def __init__(self, message, *, evidence=None):
        super().__init__(message)
        self.evidence = evidence


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise SemanticSelectionError(f"{name} must be a nonempty public identity")
    return value


def _selection_fragments(tape):
    result = []
    emitted = getattr(tape, "emitted_ids", ())
    for index, identity in enumerate(tape.event_ids):
        fragment = {
            "selectedEventId": identity,
            "meridian": tape.meridian_id,
            "modelTrack": tape.tracks[index],
            "payloadHex": tape.payloads[index].hex()
            if isinstance(tape.payloads[index], bytes)
            else None,
        }
        if len(emitted) == len(tape.event_ids):
            fragment["emittedTokenIds"] = list(emitted[index])
        result.append(fragment)
    return result


def _validate_prefix(prefix: Any) -> tuple[dict, dict, list[dict]]:
    from megatoken.data.semantic_contract import validate_envelope_record

    if not isinstance(prefix, dict) or prefix.get("schema") != "selected-inference-prefix/2":
        raise SemanticSelectionError("Unknown selected-model inference prefix version")
    events = prefix.get("committed")
    if (
        not isinstance(events, list)
        or len(events) < 2
        or any(not isinstance(e, dict) for e in events)
    ):
        raise SemanticSelectionError("Model execution needs published code before invocation")
    programs = [
        event
        for event in events
        if event.get("kind") == "program" and event.get("programId") == prefix.get("programId")
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
    if (
        not isinstance(session, str)
        or not session
        or any(event.get("session") != session for event in events)
    ):
        raise SemanticSelectionError("Model prefix crosses session authority")
    positions = [item.get("sequence") for item in events]
    if not all(type(x) is int and x > 0 for x in positions) or positions != sorted(set(positions)):
        raise SemanticSelectionError("Selected prefix has reordered or duplicated causal receipts")
    for item in events:
        try:
            validate_envelope_record(item)
        except ValueError as error:
            raise SemanticSelectionError(
                "Model prefix violates the shared envelope contract: " + str(error)
            ) from error
        if item.get("schema") != "semantic-execution/4":
            raise SemanticSelectionError("Model prefix mixes incompatible public contracts")
        mode = item.get("mode")
        if item.get("kind") in {
            "program",
            "invocation",
            "dispatch",
            "result",
            "event",
            "handler",
            "handler-close",
            "source-evict",
            "source-read",
            "cancel",
            "reconcile",
            "callback-reattach",
            "model-context",
        }:
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
    source_events: tuple[str, ...] = ()
    source_reads: list[dict] = field(default_factory=list)
    promise_effects: dict[str, str] = field(default_factory=dict)
    settled_promises: set[str] = field(default_factory=set)
    waiting_promise: str | None = None
    request_receipts: dict[str, dict] = field(default_factory=dict)
    rejections: list[dict] = field(default_factory=list)
    representation: str | None = None
    selection_records: dict[str, str] = field(default_factory=dict)
    state_bootstrap: dict | None = None
    realm_cutoff: int = 0
    selection_history: list[str] = field(default_factory=list)
    escaped_closures: dict[str, dict] = field(default_factory=dict)


@dataclass
class _Session:
    runtime_session: Any
    work: dict[str, _Work] = field(default_factory=dict)
    programs: dict[str, str] = field(default_factory=dict)
    orchestration: dict | None = None
    subscriptions: dict[str, dict] = field(default_factory=dict)


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
        model_identity: str | None = None,
        verify_context: bool = True,
        allow_controlled_lifecycle: bool = False,
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
        self.model_identity = model_identity
        self.verify_context = verify_context
        self.allow_controlled_lifecycle = allow_controlled_lifecycle
        if verify_context and not callable(getattr(runtime, "context_status", None)):
            raise ValueError("Production semantic inference requires checked source visibility")
        self.lock = RLock()
        self.initialization = None

    def initialize(self, *, seed: int, checkpointIdentity: str) -> dict:
        """Attest the actual loaded weights and seed every inference RNG before work."""
        import random

        import numpy as np
        import torch

        with self.lock:
            if self.sessions:
                raise SemanticSelectionError(
                    "Inference identity must be initialized before any active session"
                )
            if type(seed) is not int or seed not in {17, 29, 43}:
                raise SemanticSelectionError(
                    "Semantic evaluation requires a declared seed: 17, 29, or 43"
                )
            actual = self._identity()
            if checkpointIdentity != actual:
                raise SemanticSelectionError(
                    "Requested checkpoint does not match the actually loaded model"
                )
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            self.initialization = {
                "schema": "semantic-inference-identity/4",
                "seed": seed,
                "checkpointIdentity": actual,
                "tokenizerIdentity": self.runtime.generator.generator.codec.fingerprint,
            }
            return dict(self.initialization)

    def _opened(self, session_key: str) -> _Session:
        existing = self.sessions.get(session_key)
        if existing is not None:
            return existing
        if len(self.sessions) >= self.max_sessions:
            raise SemanticSelectionError("Selected session capacity exhausted")
        session = _Session(
            self.runtime.open(
                metadata={"task": "semantic-execution/4", "modality": "text"},
                effort=self.effort,
                memory_writes=False,
            )
        )
        self.sessions[session_key] = session
        return session

    def _observe(self, session: Any, track: str, event: dict) -> tuple[str, ...]:
        # Runtime chunking is a transport concern; the source and input remain
        # ordered observed records, not trainable model outputs.
        kind = event["kind"]
        role = "user" if kind in ("program", "invocation") else "environment"
        before = set(getattr(session, "occupied_events", set()))
        self.runtime.observe_text(
            session,
            track,
            _canonical(event),
            role=role,
            effort=self.effort,
        )
        if not self.verify_context:
            return ()
        return tuple(
            event_id for event_id in session.context_archive.events if event_id not in before
        )

    def _bootstrap(
        self,
        state: _Session,
        program: dict,
        invocation: dict,
        *,
        bootstrap_events=None,
        native_parent=None,
        external_ingress=None,
    ) -> _Work:
        from megatoken.lifecycle import TrackDescription

        track = _text(invocation["track"], "requested track")
        program_id = _text(program["programId"], "program ID")
        invocation_id = _text(invocation["invocationId"], "invocation ID")
        digest = _text(program["digest"], "source digest")
        bootstrap_events = [program, invocation] if bootstrap_events is None else bootstrap_events
        if any(event.get("mode") != "observed" for event in bootstrap_events):
            raise SemanticSelectionError(
                "Invocation bootstrap cannot contain generated future semantics"
            )
        if program_id in state.programs and state.programs[program_id] != digest:
            raise SemanticSelectionError(
                "A program revision cannot rewrite an existing source identity"
            )
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
            completion_mode=invocation["completionPolicy"],
            capability_grants=grants,
        )
        parent = invocation.get("parentInvocationId")
        parent_track = state.work[parent].actual_track if parent in state.work else None
        if parent is not None and parent_track is None:
            raise SemanticSelectionError("Child invocation lacks its runtime parent")
        if native_parent is not None:
            if native_parent not in state.runtime_session.tracks.tracks:
                raise SemanticSelectionError("Retained subscription lacks its actual native owner")
            parent_track = native_parent
        if external_ingress is not None:
            parent_track = None
        existing = [
            description
            for description in state.runtime_session.tracks.tracks.values()
            if description.execution and description.execution.invocation_id == invocation_id
        ]
        if existing:
            if len(existing) != 1:
                raise SemanticSelectionError("Invocation has ambiguous runtime ownership")
            description = existing[0]
            bound = description.execution
            for name in (
                "program_id",
                "source_digest",
                "entry",
                "realm_id",
                "trigger_id",
                "input_ref",
                "realm_policy",
                "completion_mode",
            ):
                if getattr(bound, name) != getattr(binding, name):
                    raise SemanticSelectionError(
                        "Selected child prefix differs from its allocated execution binding"
                    )
            coordinator_parent = (
                parent is None
                and description.parent is not None
                and state.runtime_session.tracks.tracks[description.parent].execution is None
            )
            if (
                (description.parent != parent_track and not coordinator_parent)
                or not set(bound.capability_grants) <= set(grants)
                or description.status in {"completed", "cancelled"}
            ):
                raise SemanticSelectionError(
                    "Selected child cannot change owner, grants, or closed disposition"
                )
            if not description.bootstrap_ready:
                if not callable(getattr(self.runtime, "bootstrap_execution", None)):
                    raise SemanticSelectionError(
                        "Child execution has not published its required context"
                    )
                self.runtime.bootstrap_execution(state.runtime_session, description.address)
            source_events = tuple(
                getattr(state.runtime_session, "execution_source_events", {}).get(
                    description.address, ()
                )
            )
            registered_program = state.runtime_session.execution_programs[program_id]
            registered_input = state.runtime_session.execution_inputs[
                invocation_id + ":" + data_ref
            ]
            transport = {"sequence", "ownerSequence"}
            if any(
                {key: value for key, value in observed.items() if key not in transport}
                != {key: value for key, value in actual.items() if key not in transport}
                for observed, actual in (
                    (registered_program, program),
                    (registered_input, invocation),
                )
            ):
                raise SemanticSelectionError(
                    "Selected child changed its immutable observed proposal"
                )
            if (
                registered_program != program
                or registered_input != invocation
                or len(bootstrap_events) != 2
            ):
                # Proposal admission and actual harness publication have distinct
                # transport positions. Publish the actual receipt explicitly;
                # never rewrite already observed proposal bytes in model history.
                for event in bootstrap_events:
                    published = self._observe(state.runtime_session, description.address, event)
                    if event is program:
                        source_events = published
                state.runtime_session.execution_source_events[description.address] = source_events
        else:
            description = self.runtime.track_action(
                state.runtime_session,
                "create",
                purpose=f"Execute {program_id}.{invocation['entry']}",
                completion=f"Invocation {invocation_id} completion",
                capabilities=("text", *grants),
                budget=self.work_budget,
                execution=binding,
                **({"parent": parent_track} if parent_track else {}),
            )
            if not isinstance(description, TrackDescription) or description.status != "waiting":
                raise SemanticSelectionError(
                    "Dynamic execution track must await observed bootstrap"
                )
            # Immutable program is observed on this newly allocated execution track
            # before input and invocation; its digest must match the binding.
            if external_ingress is not None:
                self._observe(state.runtime_session, description.address, external_ingress)
            for event in bootstrap_events:
                published = self._observe(state.runtime_session, description.address, event)
                if event is program:
                    source_events = published
            state.runtime_session.tracks.activate_execution(
                description.address,
                source_digest=digest,
                input_ref=data_ref,
            )
        # Existing task/model context may evict source; code remains
        # source-event retrievable from the context archive when needed.
        work = _Work(
            session_key=program["session"],
            program_id=program_id,
            invocation_id=invocation_id,
            requested_track=track,
            actual_track=description.address,
            source_event=None,
            source_events=source_events,
        )
        state.work[invocation_id] = work
        state.programs[program_id] = digest
        if self.verify_context:
            maximum = self.runtime.generator.generator.config.max_history_chunks
            self.runtime.configure_context(
                state.runtime_session,
                track=work.actual_track,
                local_events=maximum,
                cross_events=maximum,
                pinned_events=(),
            )
        return work

    def _ensure_source(self, state: _Session, work: _Work, program: dict) -> list[dict]:
        if not self.verify_context:
            return []
        session = state.runtime_session
        status = self.runtime.context_status(session, track=work.actual_track)
        if work.source_events and set(work.source_events) <= set(status["visible_events"]):
            return []
        original = work.source_events
        # A source reread is observed now. It never inserts an event into the
        # earlier selection's context and never changes original source bytes.
        work.source_events = self._observe(session, work.actual_track, program)
        config = self.runtime.generator.generator.config
        if not work.source_events or len(work.source_events) >= config.max_history_chunks:
            raise SemanticSelectionError("Published source cannot fit the model's declared history")
        self.runtime.configure_context(
            session,
            track=work.actual_track,
            local_events=config.max_history_chunks,
            cross_events=config.max_history_chunks,
            pinned_events=work.source_events if len(work.source_events) <= 8 else (),
        )
        visible = self.runtime.context_status(session, track=work.actual_track)["visible_events"]
        if not set(work.source_events) <= set(visible):
            raise SemanticSelectionError(
                "Source reread was evicted before the next model selection"
            )
        receipt = {
            "sourceDigest": program["digest"],
            "originalEventIds": list(original),
            "observedEventIds": list(work.source_events),
            "mode": "checked-reread",
        }
        work.source_reads.append(receipt)
        return [receipt]

    def _advance(self, state: _Session, work: _Work, events: list[dict]) -> None:
        for event in events:
            encoded = _canonical(event)
            if event["kind"] == "semantic":
                expected = work.selected[0] if work.selected else None
                if expected is None or event["line"].encode("utf-8") != expected:
                    raise SemanticSelectionError(
                        "Model transcript does not match its previously committed selection"
                    )
                work.selected.pop(0)
                from megatoken.data.witness_v4 import parse_record

                kind, fields = parse_record(event["line"])
                if kind == "request" and "promise" in fields:
                    work.promise_effects[fields["effect"]] = fields["promise"]
                elif kind == "await" and fields["invocation"] == work.invocation_id:
                    work.waiting_promise = fields["promise"]
                    if work.waiting_promise not in work.settled_promises:
                        state.runtime_session.tracks.change(
                            work.actual_track,
                            "wait",
                            prerequisites=(
                                f"semantic-settlement:{work.invocation_id}:{work.waiting_promise}",
                            ),
                        )
                elif kind == "resume":
                    work.waiting_promise = None
                # Selected meridians are already in Megatoken's model state;
                # never ingest them as fresh observed evidence a second time.
            elif event["kind"] not in ("program", "invocation"):
                observed = self._observe(state.runtime_session, work.actual_track, event)
                if event["kind"] == "source-read":
                    if (
                        event["programId"] != work.program_id
                        or event["digest"] != state.programs[work.program_id]
                        or hashlib.sha256(event["source"].encode()).hexdigest() != event["digest"]
                    ):
                        raise SemanticSelectionError(
                            "Observed source reread changed its immutable source"
                        )
                    work.source_events = observed
                if event["kind"] == "model-context" and "selectedState" in event:
                    work.state_bootstrap = {
                        "schema": "semantic-selected-bootstrap-receipt/4",
                        "digest": hashlib.sha256(_canonical(event)).hexdigest(),
                        "prefixDigest": event["realmPrefixDigest"],
                        "visibleThrough": event["visibleThrough"],
                        "observedEventIds": list(observed),
                        "reason": "public-context",
                        "publicSequence": event["sequence"],
                    }
                    work.selection_history.clear()
                if event["kind"] == "result" and event.get("effectId") in work.promise_effects:
                    promise = work.promise_effects[event["effectId"]]
                    work.settled_promises.add(promise)
                    dependency = f"semantic-settlement:{work.invocation_id}:{promise}"
                    state.runtime_session.completed_events.add(dependency)
                    if work.waiting_promise == promise:
                        state.runtime_session.tracks.wake_ready_executions(
                            delivered=state.runtime_session.completed_events
                        )
            work.seen.append(encoded)

    def _realm_prefix(self, state: _Session, prefix: dict, invocation: dict) -> list[dict]:
        """Acknowledge other owners' actual selections before transferring a realm turn.

        A callback can become runnable before the suspended parent asks for its
        next selection. The controller's public realm prefix is the causal
        acknowledgement of that parent's already selected await, not permission
        to invent another selection or to replay teacher execution.
        """
        from megatoken.data.semantic_contract import validate_envelope_record

        realm = prefix.get("realmCommitted")
        if realm is None:
            if invocation["activation"] == "callback" or invocation["realmPolicy"] == "retained":
                raise SemanticSelectionError("Retained work requires its public realm prefix")
            return prefix["committed"]
        if not isinstance(realm, list) or not realm:
            raise SemanticSelectionError("Committed realm context must be a nonempty public prefix")
        positions = [event.get("sequence") for event in realm if isinstance(event, dict)]
        if (
            len(positions) != len(realm)
            or any(type(x) is not int or x < 1 for x in positions)
            or positions != sorted(set(positions))
        ):
            raise SemanticSelectionError("Committed realm context is reordered or duplicated")
        calls = {
            event.get("invocationId"): event for event in realm if event.get("kind") == "invocation"
        }
        programs = {
            event.get("programId"): event for event in realm if event.get("kind") == "program"
        }
        if len(calls) != sum(event.get("kind") == "invocation" for event in realm) or len(
            programs
        ) != sum(event.get("kind") == "program" for event in realm):
            raise SemanticSelectionError(
                "Committed realm context reused an invocation or program identity"
            )
        if any(
            call.get("realmId") != invocation["realmId"]
            or call.get("programId") not in programs
            or programs[call["programId"]].get("digest") != call.get("sourceDigest")
            for call in calls.values()
        ):
            raise SemanticSelectionError(
                "Committed realm context crosses realm or source authority"
            )
        if set(programs) != {call["programId"] for call in calls.values()}:
            raise SemanticSelectionError(
                "Committed realm context contains unpublished foreign code"
            )
        by_position = {event["sequence"]: event for event in realm}
        if any(by_position.get(event["sequence"]) != event for event in prefix["committed"]):
            raise SemanticSelectionError(
                "Invocation prefix differs from its committed realm context"
            )
        from megatoken.data.semantic_context import validate_selected_context

        for event_index, event in enumerate(realm):
            try:
                validate_envelope_record(event)
            except ValueError as error:
                raise SemanticSelectionError(
                    "Invalid committed realm envelope: " + str(error)
                ) from error
            if event.get("session") != invocation["session"] or (
                event.get("invocationId") is not None and event["invocationId"] not in calls
            ):
                raise SemanticSelectionError("Committed realm context contains an unowned event")
            if event["kind"] == "model-context":
                try:
                    validate_selected_context(event, realm[:event_index])
                except ValueError as error:
                    raise SemanticSelectionError(str(error)) from error
            if (
                event["kind"] == "semantic"
                and event["invocationId"] not in state.work
                and event["invocationId"] != invocation["invocationId"]
            ):
                raise SemanticSelectionError(
                    "Committed realm context invents an unrecorded selected owner"
                )
            if event["kind"] == "handler-close" and event["subscriptionId"] in state.subscriptions:
                subscription = state.subscriptions[event["subscriptionId"]]
                if subscription.get("closed") is not None and subscription["closed"] != event:
                    raise SemanticSelectionError("Retained subscription closure was rewritten")
                subscription["closed"] = event
        for other_id, other in state.work.items():
            if other_id == invocation["invocationId"] or other_id not in calls:
                continue
            owned = [
                event
                for event in realm
                if event.get("kind") == "program"
                and event.get("programId") == other.program_id
                or event.get("invocationId") == other_id
                or event.get("kind") in {"source-read", "source-evict"}
                and event.get("track") == other.requested_track
                and event.get("programId") == other.program_id
            ]
            if len(owned) < len(other.seen) or any(
                before != _canonical(owned[index]) for index, before in enumerate(other.seen)
            ):
                raise SemanticSelectionError(
                    "Committed realm context rewrites another selected owner"
                )
            self._advance(state, other, owned[len(other.seen) :])
        self._retire_subscriptions(state)
        return realm

    def _retire_subscriptions(self, state: _Session) -> None:
        native = state.runtime_session
        for subscription in state.subscriptions.values():
            owner = subscription["track"]
            if not subscription.get("closed") or native.tracks.tracks[owner].status in {
                "completed",
                "cancelled",
            }:
                continue
            if any(
                child.parent == owner and child.status in {"active", "waiting"}
                for child in native.tracks.tracks.values()
            ):
                continue
            self._observe(native, owner, subscription["closed"])
            self.runtime.track_action(native, "complete", address=owner)

    def _subscription_owner(self, state: _Session, invocation: dict, realm: list[dict]) -> str:
        """A persistent registration keeps its own host owner after its call returns.

        The new callback retains its original JavaScript parent in the observed
        invocation. Native lifetime ownership moves only to a checked live
        subscription; a completed task is never reopened or assigned new work.
        """
        event = next(
            (
                row
                for row in realm
                if row["kind"] == "event" and row.get("invocationId") == invocation["invocationId"]
            ),
            None,
        )
        handler = next(
            (
                row
                for row in realm
                if row["kind"] == "handler"
                and event is not None
                and row["subscriptionId"] == event["subscriptionId"]
            ),
            None,
        )
        if (
            handler is None
            or handler["lifetime"] != "subscription"
            or handler["ownerInvocationId"] != invocation["parentInvocationId"]
            or handler["programId"] != invocation["programId"]
            or handler["realmId"] != invocation["realmId"]
            or handler["closureId"] != invocation["closureId"]
            or handler["entry"] != invocation["entry"]
            or handler["trackId"] != invocation["track"]
            or any(
                row["kind"] == "handler-close"
                and row["subscriptionId"] == handler["subscriptionId"]
                and row["sequence"] < event["sequence"]
                for row in realm
            )
        ):
            raise SemanticSelectionError(
                "Closed invocation can only yield to its live persistent subscription"
            )
        existing = state.subscriptions.get(handler["subscriptionId"])
        if existing is not None:
            if existing["handler"] != handler or existing.get("closed") is not None:
                raise SemanticSelectionError(
                    "Persistent callback changed or outlived its subscription owner"
                )
            return existing["track"]
        parent = state.work[invocation["parentInvocationId"]]
        description = state.runtime_session.tracks.tracks[parent.actual_track]
        if description.status != "completed":
            raise SemanticSelectionError(
                "Cancelled registration cannot create persistent callback work"
            )
        owner = self.runtime.track_action(
            state.runtime_session,
            "create",
            purpose="Own live subscription " + handler["subscriptionId"],
            completion="Observed subscription closure and all delivered callbacks complete",
            status="waiting",
            capabilities=description.capabilities,
            budget=self.work_budget,
        )
        self._observe(
            state.runtime_session,
            owner.address,
            {
                "schema": "semantic-subscription-owner/4",
                "kind": "subscription-owner",
                "subscriptionId": handler["subscriptionId"],
                "registrationInvocationId": parent.invocation_id,
                "registrationTrack": parent.actual_track,
                "realmId": handler["realmId"],
                "sourceDigest": state.programs[parent.program_id],
                "handler": handler,
            },
        )
        state.subscriptions[handler["subscriptionId"]] = {
            "track": owner.address,
            "handler": handler,
            "closed": None,
        }
        return owner.address

    def _external_closure_ingress(self, state: _Session, invocation: dict, prefix: dict) -> dict:
        parent = state.work[invocation["parentInvocationId"]]
        proof = parent.escaped_closures.get(invocation["closureId"])
        owner = state.runtime_session.tracks.tracks[parent.actual_track]
        snapshot = prefix.get("selectedState")
        heap = dict(snapshot.get("heap", [])) if isinstance(snapshot, dict) else {}
        closure = heap.get(invocation["closureId"], {})
        if (
            proof is None
            or owner.status != "completed"
            or owner.execution.realm_id != invocation["realmId"]
            or owner.execution.program_id != invocation["programId"]
            or closure.get("kind") != "closure"
            or closure.get("code") != invocation["entry"]
            or any(
                event.get("kind") == "cancel" and event.get("invocationId") == parent.invocation_id
                for event in prefix.get("realmCommitted", ())
            )
        ):
            raise SemanticSelectionError(
                "External closure callback lacks an actual committed escape receipt"
            )
        return {
            "schema": "semantic-external-closure-ingress/4",
            "kind": "external-closure-ingress",
            "invocationId": invocation["invocationId"],
            "closureId": invocation["closureId"],
            "registrationInvocationId": parent.invocation_id,
            "registrationTrack": parent.actual_track,
            "realmId": invocation["realmId"],
            "sourceDigest": invocation["sourceDigest"],
            "triggerId": invocation["triggerId"],
            "escapeEvidence": proof,
        }

    def _completion_escapes(self, work: _Work, snapshot: dict | None, receipt: dict) -> None:
        if snapshot is None:
            return
        heap, visited = dict(snapshot.get("heap", [])), set()

        def visit(value):
            if not isinstance(value, list) or not value or value[0] != "ref" or value[1] in visited:
                return
            visited.add(value[1])
            node = heap.get(value[1])
            if node is None:
                return
            if node["kind"] == "closure":
                work.escaped_closures[value[1]] = {
                    "kind": "completion",
                    "dependency": receipt["dependency"],
                    "value": receipt["value"],
                }
                return
            for _, child in node.get("properties", []):
                visit(child)

        if receipt["kind"] == "return":
            visit(receipt["value"])

    def _publish_selected_state(
        self,
        state: _Session,
        work: _Work,
        prefix: dict,
        invocation: dict,
        realm: list[dict],
        *,
        reason: str,
    ) -> None:
        snapshot = prefix.get("selectedState")
        if (
            not isinstance(snapshot, dict)
            or set(snapshot)
            != {
                "schema",
                "heap",
                "locals",
                "captureAliases",
                "stack",
                "continuations",
                "ready",
                "callReturn",
                "pending",
                "completion",
            }
            or snapshot["schema"] != "semantic-continuation/4"
        ):
            raise SemanticSelectionError(
                "Retained execution requires its exact committed selected-state snapshot"
            )
        bootstrap = {
            "schema": "semantic-selected-bootstrap/4",
            "kind": "selected-bootstrap",
            "reason": reason,
            "invocationId": work.invocation_id,
            "realmId": invocation["realmId"],
            "sourceDigest": prefix["sourceDigest"],
            "visibleThrough": realm[-1]["sequence"],
            "prefixDigest": hashlib.sha256(_canonical(realm)).hexdigest(),
            "state": snapshot,
        }
        ids = self._observe(state.runtime_session, work.actual_track, bootstrap)
        if self.verify_context:
            visible = self.runtime.context_status(state.runtime_session, track=work.actual_track)[
                "visible_events"
            ]
            if not ids or not set(ids) <= set(visible):
                raise SemanticSelectionError(
                    "Committed selected state exceeds the visible model context"
                )
        work.state_bootstrap = {
            "schema": "semantic-selected-bootstrap-receipt/4",
            "digest": hashlib.sha256(_canonical(bootstrap)).hexdigest(),
            "prefixDigest": bootstrap["prefixDigest"],
            "visibleThrough": bootstrap["visibleThrough"],
            "observedEventIds": list(ids),
            "reason": reason,
        }
        work.selection_history.clear()

    def _release_selected_payloads(self, state, work):
        """Discard archived physical generations without removing causal history.

        Requests may reach native commitment after their prefix acknowledgement,
        so physical generations remain until terminal completion or cancellation.
        Idempotent native commitment receipts and causal history remain available.
        """
        session = state.runtime_session
        for revision, group in tuple(session.selected_groups.items()):
            if group.tape.tracks == (work.actual_track,):
                self.runtime.release_generation(session, revision)

    def select(self, prefix: dict, *, representation="explicit", controlledRecord=None) -> dict:
        if representation not in {"explicit", "compact", "outcome-only"}:
            raise SemanticSelectionError("Unknown selected inference representation")
        program, invocation, events = _validate_prefix(prefix)
        if representation == "outcome-only" and (program.get("capabilities") or len(events) != 2):
            raise SemanticSelectionError(
                "Noninteractive outcomes require only pure source and invocation input"
            )
        with self.lock:
            session_key = _text(program["session"], "session")
            state = self._opened(session_key)
            if (
                state.orchestration is not None
                and state.orchestration["representation"] != representation
            ):
                raise SemanticSelectionError(
                    "Child representation differs from its coordinator experiment"
                )
            controlled = (
                state.orchestration is not None and state.orchestration["origin"] == "controlled"
            )
            if controlledRecord is not None and (
                not controlled
                or not self.allow_controlled_lifecycle
                or not isinstance(controlledRecord, str)
            ):
                raise SemanticSelectionError(
                    "Controlled guest input requires explicit engineering authority"
                )
            if controlled and controlledRecord is None:
                raise SemanticSelectionError(
                    "Controlled coordinator children require explicit fixture bytes"
                )
            id_ = _text(invocation["invocationId"], "invocation")
            if state.orchestration is not None:
                proposed = {
                    child["invocation"]["invocationId"] for child in state.orchestration["children"]
                }
                allocated = [
                    track
                    for track in state.runtime_session.tracks.tracks.values()
                    if track.execution is not None and track.execution.invocation_id == id_
                ]
                if id_ in proposed and (
                    len(allocated) != 1 or allocated[0].parent != state.orchestration["track"]
                ):
                    raise SemanticSelectionError(
                        "Child selection requires its actual coordinator creation"
                    )
                if id_ not in proposed and invocation.get("parentInvocationId") not in state.work:
                    raise SemanticSelectionError(
                        "Unproposed execution has no coordinator-owned parent"
                    )
            realm = self._realm_prefix(state, prefix, invocation)
            work = state.work.get(id_)
            new_work = work is None
            if work is None:
                bootstrap_events = events[: events.index(invocation) + 1]
                native_parent = None
                external_ingress = None
                parent = state.work.get(invocation.get("parentInvocationId"))
                if (
                    invocation["activation"] == "callback"
                    and parent is not None
                    and parent.completed
                ):
                    if any(
                        event["kind"] == "event" and event.get("invocationId") == id_
                        for event in realm
                    ):
                        native_parent = self._subscription_owner(state, invocation, realm)
                    else:
                        external_ingress = self._external_closure_ingress(state, invocation, prefix)
                work = self._bootstrap(
                    state,
                    program,
                    invocation,
                    bootstrap_events=bootstrap_events,
                    native_parent=native_parent,
                    external_ingress=external_ingress,
                )
                work.seen.extend(_canonical(event) for event in bootstrap_events)
                if (
                    invocation["activation"] == "callback"
                    and invocation.get("parentInvocationId") in state.work
                    and native_parent is None
                    and external_ingress is None
                ):
                    parent = state.work[invocation["parentInvocationId"]]
                    registry = state.runtime_session.tracks
                    dependency = registry.completion_dependency(work.actual_track)
                    registry.wait_for_execution(
                        parent.actual_track,
                        dependency,
                        delivered=state.runtime_session.completed_events,
                    )
                    self._observe(
                        state.runtime_session,
                        parent.actual_track,
                        {
                            "schema": "semantic-realm-turn/4",
                            "kind": "realm-turn",
                            "invocationId": parent.invocation_id,
                            "callbackInvocationId": id_,
                            "completionDependency": dependency,
                            "visibleThrough": realm[-1]["sequence"],
                            "prefixDigest": hashlib.sha256(_canonical(realm)).hexdigest(),
                        },
                    )
            if work.representation is not None and work.representation != representation:
                raise SemanticSelectionError(
                    "An invocation cannot change its selected representation"
                )
            work.representation = representation
            if (
                work.program_id != program["programId"]
                or work.requested_track != invocation["track"]
                or work.completed
            ):
                raise SemanticSelectionError(
                    "Track reuse, provenance mismatch or completed invocation"
                )
            # Check exact old bytes to prevent retroactive teacher/result edits.
            if len(events) < len(work.seen) or any(
                before != _canonical(events[index]) for index, before in enumerate(work.seen)
            ):
                raise SemanticSelectionError(
                    "Previously delivered model-visible prefix was rewritten"
                )
            self._advance(state, work, events[len(work.seen) :])
            if work.waiting_promise is not None and prefix.get("selectedState") is not None:
                snapshot = prefix["selectedState"]
                if snapshot.get("schema") != "semantic-continuation/4":
                    raise SemanticSelectionError(
                        "Cannot wake from an incompatible selected-state snapshot"
                    )
                promises = dict(snapshot["heap"])
                promise = promises.get(work.waiting_promise)
                if (
                    promise
                    and promise.get("kind") == "promise"
                    and promise.get("state") in {"fulfilled", "rejected"}
                ):
                    # Adoption/finally settlement is a deterministic consequence
                    # of the committed public prefix, supplied by the controller.
                    work.settled_promises.add(work.waiting_promise)
                    dependency = f"semantic-settlement:{work.invocation_id}:{work.waiting_promise}"
                    state.runtime_session.completed_events.add(dependency)
                    state.runtime_session.tracks.wake_ready_executions(
                        delivered=state.runtime_session.completed_events
                    )
            # A previously selected record must be committed and reported back
            # before another generated semantic action can be sampled.
            if work.selected:
                raise SemanticSelectionError("Previous model selection was not committed")
            if not state.runtime_session.tracks.selectable(
                work.actual_track, state.runtime_session.completed_events
            ):
                raise SemanticSelectionError("Selected execution track has not become runnable")
            source_reads = self._ensure_source(state, work, program)
            retained = new_work and (
                invocation["realmPolicy"] == "retained" or invocation["activation"] == "callback"
            )
            foreign_progress = any(
                event["sequence"] > work.realm_cutoff
                and event.get("invocationId") not in {None, id_}
                and event.get("kind") in {"semantic", "result"}
                for event in realm
            )
            missing_history = False
            if self.verify_context:
                visible = set(
                    self.runtime.context_status(state.runtime_session, track=work.actual_track)[
                        "visible_events"
                    ]
                )
                needed = set(work.selection_history)
                if work.state_bootstrap is not None:
                    needed.update(work.state_bootstrap["observedEventIds"])
                missing_history = bool(needed - visible)
            public_state = next(
                (
                    event
                    for event in reversed(events)
                    if event["kind"] == "model-context" and "selectedState" in event
                ),
                None,
            )
            fresh_public_state = (
                public_state is not None
                and public_state["selectedState"] == prefix.get("selectedState")
                and public_state["sequence"] == realm[-1]["sequence"]
            )
            if (retained or foreign_progress or missing_history) and not fresh_public_state:
                self._publish_selected_state(
                    state,
                    work,
                    prefix,
                    invocation,
                    realm,
                    reason="retained-activation"
                    if retained
                    else "realm-resume"
                    if foreign_progress
                    else "history-reread",
                )
                source_reads.extend(self._ensure_source(state, work, program))
            if fresh_public_state and (retained or foreign_progress):
                work.state_bootstrap["reason"] = (
                    "retained-activation" if retained else "realm-resume"
                )
            if work.state_bootstrap is not None:
                if self.verify_context:
                    visible = set(
                        self.runtime.context_status(state.runtime_session, track=work.actual_track)[
                            "visible_events"
                        ]
                    )
                    if not set(work.state_bootstrap["observedEventIds"]) <= visible:
                        raise SemanticSelectionError(
                            "Source and current selected state cannot fit together in model context"
                        )
                    if work.state_bootstrap["reason"] == "realm-resume":
                        work.state_bootstrap["visibleEventIds"] = sorted(visible)
                        work.state_bootstrap["maxHistoryChunks"] = (
                            self.runtime.generator.generator.config.max_history_chunks
                        )
            work.realm_cutoff = realm[-1]["sequence"]
            config = self.runtime.generator.generator.config
            maximum = min(self.max_tokens, config.max_capacity)
            instruction = (
                b"Emit exactly one complete canonical semantic-trace/4 witness. "
                b"Use only the observed source, this invocation and prior selected "
                b"semantics. An external call is a validated request with already "
                b"evaluated arguments and source span. Never predict observations, "
                b"events, native results or future tool receipts."
            )
            if representation == "outcome-only":
                instruction = (
                    b"Predict the pure program outcome directly from the observed source and invocation. "
                    b"Emit exactly one JSON object with schema semantic-outcome/4, completion {kind: return or throw, value: canonical tagged value}, "
                    b"and state {roots: canonical tagged original argument roots followed by result, objects: complete reachable owned data graph}. "
                    b"Do not emit semantic witness lines or call tools. Never predict observations."
                )
            elif representation == "compact":
                instruction += (
                    b" Use bounded compact expressions only for adjacent pure computation: "
                    b"at most 8 operators and 32 expression nodes. Preserve source dependency order. "
                    b"Never fuse across effects, mutation, binding, branches, calls, awaits, callbacks, "
                    b"returns, throws, or cleanup boundaries."
                )
            else:
                instruction += b" Emit each explicit source computation as its own witness record."
            # A logical semantic record may span several real model meridians.
            # Native commitment waits for the final terminal; no peer result or
            # second generated tool envelope is inserted between fragments.
            fragments, chunks = [], []

            def refuse(message, evidence):
                rejection = {"reason": message, **evidence}
                work.rejections.append(rejection)
                state.runtime_session.tracks.change(work.actual_track, "cancel")
                work.completed = True
                self._release_selected_payloads(state, work)
                raise SemanticSelectionError(message, evidence=rejection)

            for fragment in range(64):
                before_generation = (
                    set(state.runtime_session.context_archive.events)
                    if self.verify_context
                    else set()
                )
                if controlled:
                    generated = self.runtime.commit_fixture_action(
                        state.runtime_session,
                        track=work.actual_track,
                        payload=controlledRecord.encode(),
                        fixture_digest=state.orchestration["fixtureDigest"],
                        decision=work.selections,
                        family="semantic",
                    )
                else:
                    generated = self.runtime.generate(
                        state.runtime_session,
                        tracks=(work.actual_track,),
                        specs=(
                            ActionSpec(
                                text_kind="utf8",
                                min_tokens=1,
                                max_tokens=maximum,
                                require_terminal=None,
                            ),
                        ),
                        controls=(
                            instruction
                            if fragment == 0
                            else instruction
                            + b" Continue the exact current record bytes; end only when the witness is complete.",
                        ),
                        effort=self.effort,
                    )
                tape = generated.tape
                if self.verify_context:
                    work.selection_history.extend(
                        event_id
                        for event_id in state.runtime_session.context_archive.events
                        if event_id not in before_generation
                    )
                if (
                    len(tape.payloads) != 1
                    or not tape.valid[0]
                    or tape.modalities[0] != "text"
                    or tape.tracks[0] != work.actual_track
                    or not isinstance(tape.payloads[0], bytes)
                ):
                    rejection = {
                        "meridian": tape.meridian_id,
                        "eventIds": list(tape.event_ids),
                        "valid": list(tape.valid),
                        "payloadHex": [
                            payload.hex() if isinstance(payload, bytes) else None
                            for payload in (*chunks, *tape.payloads)
                        ],
                        "fragments": [*fragments, *_selection_fragments(tape)],
                    }
                    refuse("Model did not render one complete valid text action", rejection)
                chunks.append(tape.payloads[0])
                fragments.extend(_selection_fragments(tape))
                if sum(map(len, chunks)) > 65536:
                    refuse(
                        "Selected logical witness exceeds its byte budget",
                        {"fragments": fragments, "payloadHex": b"".join(chunks).hex()},
                    )
                if tape.terminal[0]:
                    break
            else:
                refuse(
                    "Selected witness did not terminate within its fragment budget",
                    {"fragments": fragments, "payloadHex": b"".join(chunks).hex()},
                )
            original = b"".join(chunks)
            try:
                output = original.decode("utf-8", errors="strict")
            except UnicodeDecodeError:
                refuse(
                    "Model emitted invalid UTF-8",
                    {"fragments": fragments, "payloadHex": original.hex()},
                )
            work.selected.append(original)
            work.selections += 1
            work.selection_records[f"{tape.meridian_id}:{tape.event_ids[0]}"] = hashlib.sha256(
                original
            ).hexdigest()
            return {
                "schema": "selected-outcome-output/4"
                if representation == "outcome-only"
                else "selected-model-output/2",
                "representation": representation,
                "record": output,
                "selection": {
                    "modelTrack": work.actual_track,
                    "ownerTrack": work.requested_track,
                    "selectedEventId": tape.event_ids[0],
                    "meridian": tape.meridian_id,
                    "fragments": fragments,
                    "codecFingerprint": self.runtime.generator.generator.codec.fingerprint,
                    "modelSelected": not controlled,
                    "origin": "controlled-test-injected" if controlled else "actual-model-selected",
                    "sourceReads": source_reads,
                    "stateBootstrap": work.state_bootstrap,
                },
            }

    def select_outcome(self, prefix: dict) -> dict:
        """One actual noninteractive model answer for the pure common-support control."""
        from megatoken.data.witness_v4 import graph, value
        from megatoken.io import strict_json_loads

        result = self.select(prefix, representation="outcome-only")
        program, invocation, _ = _validate_prefix(prefix)
        state = self.sessions[program["session"]]
        work = state.work[invocation["invocationId"]]
        try:
            predicted = strict_json_loads(result["record"])
            if (
                not isinstance(predicted, dict)
                or set(predicted) != {"schema", "completion", "state"}
                or predicted["schema"] != "semantic-outcome/4"
                or not isinstance(predicted["completion"], dict)
                or set(predicted["completion"]) != {"kind", "value"}
                or predicted["completion"]["kind"] not in {"return", "throw"}
            ):
                raise ValueError("Noninteractive model answer has an invalid completion schema")
            value(predicted["completion"]["value"])
            graph(predicted["state"])
            if (
                len(predicted["state"]["roots"]) != len(invocation["inputGraph"]["roots"]) + 1
                or predicted["state"]["roots"][-1] != predicted["completion"]["value"]
            ):
                raise ValueError("Noninteractive result must be the last observable graph root")
        except (ValueError, KeyError, TypeError) as error:
            evidence = {
                "reason": str(error),
                "selection": result["selection"],
                "payloadHex": result["record"].encode().hex(),
                "representation": "outcome-only",
            }
            work.rejections.append(evidence)
            state.runtime_session.tracks.change(work.actual_track, "cancel")
            work.completed = True
            self._release_selected_payloads(state, work)
            raise SemanticSelectionError(
                "Model emitted an invalid noninteractive outcome", evidence=evidence
            ) from error
        work.selected.clear()
        work.completed = True
        state.runtime_session.tracks.change(work.actual_track, "complete")
        self._release_selected_payloads(state, work)
        return {**result, "outcome": predicted}

    def open_orchestration(self, **options):
        from .semantic_orchestration import open_orchestration

        with self.lock:
            return open_orchestration(self, **options)

    def step_orchestration(self, **options):
        from .semantic_orchestration import step_orchestration

        with self.lock:
            return step_orchestration(self, **options)

    def status_orchestration(self, **options):
        from .semantic_orchestration import status_orchestration

        with self.lock:
            return status_orchestration(self, **options)

    def cancel_orchestration(self, **options):
        from .semantic_orchestration import cancel_orchestration

        with self.lock:
            return cancel_orchestration(self, **options)

    def close_invocation(self, session: str, invocation: str, prefix: dict) -> None:
        """A complete public return/end must acknowledge the last actual selected tape."""
        program, inv, events = _validate_prefix(prefix)
        if program["session"] != session or inv["invocationId"] != invocation:
            raise SemanticSelectionError("Completion belongs to another selected invocation")
        if (
            not events
            or events[-1].get("kind") != "semantic"
            or not events[-1].get("line", "").startswith("end ")
        ):
            raise SemanticSelectionError("Model completion needs the selected terminal end")
        with self.lock:
            state = self.sessions.get(session)
            if state is None or invocation not in state.work:
                raise SemanticSelectionError("Unknown selected invocation")
            work = state.work[invocation]
            self._realm_prefix(state, prefix, inv)
            if len(events) < len(work.seen) or any(
                before != _canonical(events[index]) for index, before in enumerate(work.seen)
            ):
                raise SemanticSelectionError("Completion rewrites earlier model history")
            self._advance(state, work, events[len(work.seen) :])
            if work.selected:
                raise SemanticSelectionError("Cannot complete before last selection commits")
            from megatoken.data.witness_v4 import parse_record

            terminal = []
            for event in events:
                if event.get("kind") != "semantic":
                    continue
                kind, fields = parse_record(event["line"])
                if (
                    kind in {"return", "throw"}
                    and fields.get("invocation", invocation) == invocation
                ):
                    terminal.append((kind, fields))
            if not terminal:
                raise SemanticSelectionError("Completion needs an owned root return or throw")
            completion_kind, completion_fields = terminal[-1]
            snapshot = prefix.get("selectedState")
            if snapshot is not None and snapshot.get("completion") is not None:
                from megatoken.data.witness_v4 import value as validate_value

                completion = snapshot["completion"]
                if (
                    snapshot.get("schema") != "semantic-continuation/4"
                    or set(completion) != {"kind", "value"}
                    or completion["kind"] not in {"return", "throw"}
                ):
                    raise SemanticSelectionError("Invalid committed controller completion")
                validate_value(completion["value"])
                completion_kind, completion_fields = completion["kind"], completion
            elif self.verify_context:
                raise SemanticSelectionError(
                    "Production completion requires the committed controller result"
                )
            if callable(getattr(self.runtime, "complete_execution", None)):
                receipt = self.runtime.complete_execution(
                    state.runtime_session,
                    work.actual_track,
                    kind=completion_kind,
                    value=completion_fields["value"],
                )
            else:
                receipt = state.runtime_session.tracks.complete_execution(
                    work.actual_track, kind=completion_kind, value=completion_fields["value"]
                )
            self._completion_escapes(work, snapshot, receipt)
            work.completed = True
            self._retire_subscriptions(state)
            self._release_selected_payloads(state, work)

    def cancel_invocation(self, session: str, invocation: str) -> None:
        with self.lock:
            state = self.sessions.get(session)
            if state is None or invocation not in state.work:
                raise SemanticSelectionError("Cannot cancel unknown execution")
            work = state.work[invocation]
            if state.runtime_session.tracks.tracks[work.actual_track].status not in {
                "completed",
                "cancelled",
            }:
                state.runtime_session.tracks.change(work.actual_track, "cancel")
            work.completed = True
            self._release_selected_payloads(state, work)

    def commit_request(
        self, session: str, invocation: str, selection: dict, record: str, validation: dict
    ) -> dict:
        """Derive one native action from the original selected runtime group.

        This trusted host RPC is called only after JS selected-state validation.
        It returns a typed dispatch descriptor; the registered JS adapter then
        performs that action. Nothing asks the model to describe it again.
        """
        from megatoken.data.semantic_execution import _host_request
        from megatoken.data.witness_v4 import parse_record

        with self.lock:
            state = self.sessions.get(session)
            if state is None or invocation not in state.work:
                raise SemanticSelectionError(
                    "Native commitment requires an existing model invocation"
                )
            work = state.work[invocation]
            if (
                not isinstance(record, str)
                or not isinstance(validation, dict)
                or not isinstance(selection, dict)
            ):
                raise SemanticSelectionError("Malformed host source-validation receipt")
            raw = record.encode("utf-8")
            digest = hashlib.sha256(raw).hexdigest()
            kind, fields = parse_record(record)
            if (
                kind != "request"
                or validation.get("schema") != "selected-source-validation/4"
                or validation.get("sourceDigest") != state.programs[work.program_id]
                or validation.get("invocationId") != invocation
                or validation.get("effectId") != fields["effect"]
                or validation.get("payloadDigest") != digest
                or selection.get("modelTrack") != work.actual_track
                or selection.get("ownerTrack") != work.requested_track
            ):
                raise SemanticSelectionError(
                    "Native commitment differs from its original source, owner or bytes"
                )
            action_key = f"{selection.get('meridian')}:{selection.get('selectedEventId')}"
            previous = work.request_receipts.get(action_key)
            if previous is not None:
                if previous["recordDigest"] != digest:
                    raise SemanticSelectionError("Native action retry changed its selected bytes")
                return dict(previous)
            fragments = selection.get("fragments")
            if not isinstance(fragments, list) or not 1 <= len(fragments) <= 64:
                raise SemanticSelectionError(
                    "Native action requires its exact selected fragment receipts"
                )
            groups = []
            controlled = (
                state.orchestration is not None and state.orchestration["origin"] == "controlled"
            )
            if controlled and not self.allow_controlled_lifecycle:
                raise SemanticSelectionError("Controlled request authority is disabled")
            committed = list(state.runtime_session.selected_groups.values())
            if controlled:
                committed.extend(state.runtime_session.controlled_fixture_actions.values())
            for fragment in fragments:
                group = next(
                    (
                        candidate
                        for candidate in committed
                        if candidate.tape.meridian_id == fragment.get("meridian")
                        and candidate.tape.event_ids == (fragment.get("selectedEventId"),)
                        and candidate.tape.tracks == (work.actual_track,)
                    ),
                    None,
                )
                if group is None or fragment.get("modelTrack") != work.actual_track:
                    raise SemanticSelectionError(
                        "Native action fragment was not selected by this model runtime"
                    )
                groups.append(group)
            group = groups[-1]
            if (
                group.tape.meridian_id != selection.get("meridian")
                or group.tape.event_ids != (selection.get("selectedEventId"),)
                or b"".join(part.tape.payloads[0] for part in groups) != raw
                or work.selection_records.get(action_key) != digest
            ):
                raise SemanticSelectionError(
                    "Native action was not selected by the committed model runtime"
                )
            _, signature = _host_request(fields, heap=validation.get("heap"), require_promise=True)
            descriptor = {
                "schema": "semantic-native-action/4",
                "actionKey": action_key,
                "invocationId": invocation,
                "effectId": fields["effect"],
                "operation": fields["operation"],
                "signature": signature,
                "payload": fields["payload"],
                "recordDigest": digest,
                "modelTrack": work.actual_track,
                "ownerTrack": work.requested_track,
                "authority": "controlled-test-injected" if controlled else "actual-model-selected",
            }

            class DescriptorAdapter:
                def execute_selected(self, *, action_id, payload):
                    if action_id != action_key or payload != raw:
                        raise SemanticSelectionError(
                            "Derived native action lost original selected bytes"
                        )
                    return _canonical(descriptor)

            class CaptureRunner:
                result = None

                def submit(self, action_id, call):
                    self.result = call()

            runner = CaptureRunner()
            submitted = self.runtime.dispatch_semantic_requests(
                state.runtime_session,
                group,
                runner,
                {fields["operation"]: DescriptorAdapter()},
                fragments=tuple(groups),
                allow_controlled_fixture=controlled,
                verify_source=lambda track, request: {
                    "verified": track == work.actual_track and request == fields,
                    "heap": validation.get("heap"),
                },
            )
            if submitted != (action_key,) or runner.result != _canonical(descriptor):
                raise SemanticSelectionError(
                    "Native request commitment did not produce exactly one action"
                )
            work.request_receipts[action_key] = descriptor
            from megatoken.data.host_operations import HOST_OPERATIONS

            if HOST_OPERATIONS[fields["operation"]].get("callbackGraph"):

                def escaped(value):
                    if isinstance(value, list):
                        if len(value) == 2 and value[0] == "callback":
                            work.escaped_closures[value[1]] = {
                                "kind": "committed-request",
                                "actionKey": action_key,
                                "recordDigest": digest,
                                "operation": fields["operation"],
                            }
                        else:
                            for child in value:
                                escaped(child)
                    elif isinstance(value, dict):
                        for child in value.values():
                            escaped(child)

                escaped(json.loads(fields["payload"]))
            return dict(descriptor)

    def _identity(self) -> str:
        if self.model_identity is None:
            model = self.runtime.generator.generator
            if not callable(getattr(model, "state_dict", None)):
                raise SemanticSelectionError("Checkpoint requires an immutable model identity")
            digest = hashlib.sha256()
            for name, tensor in sorted(model.state_dict().items()):
                digest.update(name.encode())
                digest.update(str(tensor.dtype).encode())
                digest.update(str(tuple(tensor.shape)).encode())
                digest.update(
                    tensor.detach()
                    .cpu()
                    .contiguous()
                    .view(-1)
                    .view(__import__("torch").uint8)
                    .numpy()
                    .tobytes()
                )
            self.model_identity = digest.hexdigest()
        return self.model_identity

    def checkpoint(self, session: str, prefixes: list[dict]) -> dict:
        """Save at acknowledged semantic boundaries, without running or repairing work.

        Restoration intentionally establishes a new observed model checkpoint
        bootstrap. It does not claim bit-identical KV-cache or sampled replay.
        Semantic state and native action recovery remain the harness's authority.
        """
        with self.lock:
            state = self.sessions.get(session)
            if state is None:
                raise SemanticSelectionError("Unknown selected session checkpoint")
            by_id = {prefix.get("invocationId"): prefix for prefix in prefixes}
            if len(by_id) != len(prefixes) or set(by_id) != set(state.work):
                raise SemanticSelectionError(
                    "Checkpoint must account for every selected invocation"
                )
            work_items = []
            for identity, work in state.work.items():
                _, invocation, events = _validate_prefix(by_id[identity])
                if (
                    invocation["invocationId"] != identity
                    or len(events) < len(work.seen)
                    or any(
                        _canonical(events[index]) != before
                        for index, before in enumerate(work.seen)
                    )
                ):
                    raise SemanticSelectionError("Checkpoint rewrites committed model history")
                self._advance(state, work, events[len(work.seen) :])
                if work.selected:
                    raise SemanticSelectionError(
                        "A partial unacknowledged selection cannot checkpoint"
                    )
                work_items.append(
                    {
                        "prefix": by_id[identity],
                        "actualTrack": work.actual_track,
                        "completed": work.completed,
                        "selections": work.selections,
                        "sourceReads": work.source_reads,
                        "promiseEffects": work.promise_effects,
                        "settledPromises": sorted(work.settled_promises),
                        "waitingPromise": work.waiting_promise,
                        "representation": work.representation,
                        "selectionRecords": work.selection_records,
                        "stateBootstrap": work.state_bootstrap,
                        "realmCutoff": work.realm_cutoff,
                        "selectionHistory": work.selection_history,
                        "escapedClosures": work.escaped_closures,
                    }
                )
                work_items[-1]["requestReceipts"] = work.request_receipts
                work_items[-1]["rejections"] = work.rejections
            packet = {
                "schema": "semantic-model-checkpoint/4",
                "session": session,
                "equivalence": "observed-checkpoint-bootstrap",
                "modelIdentity": self._identity(),
                "codecFingerprint": self.runtime.generator.generator.codec.fingerprint,
                "work": work_items,
                "orchestration": state.orchestration,
                "subscriptions": state.subscriptions,
                "lifecycle": state.runtime_session.tracks.checkpoint(),
                "executionContexts": {
                    "programs": getattr(state.runtime_session, "execution_programs", {}),
                    "inputs": getattr(state.runtime_session, "execution_inputs", {}),
                    "publication": getattr(state.runtime_session, "execution_publication", {}),
                    "receipts": getattr(state.runtime_session, "execution_receipts", []),
                    "delivered": sorted(
                        getattr(state.runtime_session, "execution_delivered", set())
                    ),
                    "completedEvents": sorted(state.runtime_session.completed_events),
                },
            }
            packet["digest"] = hashlib.sha256(_canonical(packet)).hexdigest()
            return json.loads(_canonical(packet))

    def restore(self, packet: dict) -> dict:
        with self.lock:
            if (
                not isinstance(packet, dict)
                or packet.get("schema") != "semantic-model-checkpoint/4"
            ):
                raise SemanticSelectionError("Incompatible selected-model checkpoint")
            body = {key: value for key, value in packet.items() if key != "digest"}
            if (
                packet.get("digest") != hashlib.sha256(_canonical(body)).hexdigest()
                or packet.get("modelIdentity") != self._identity()
                or packet.get("codecFingerprint")
                != self.runtime.generator.generator.codec.fingerprint
                or packet.get("equivalence") != "observed-checkpoint-bootstrap"
            ):
                raise SemanticSelectionError("Model, tokenizer or checkpoint bytes changed")
            session_key = _text(packet.get("session"), "session")
            if session_key in self.sessions:
                raise SemanticSelectionError("Restoration requires a fresh model session")
            from megatoken.lifecycle import TrackRegistry

            saved_registry = TrackRegistry.restore(packet["lifecycle"])
            contexts = packet["executionContexts"]
            if set(contexts) != {
                "programs",
                "inputs",
                "publication",
                "receipts",
                "delivered",
                "completedEvents",
            }:
                raise SemanticSelectionError(
                    "Checkpoint has incomplete execution context accounting"
                )
            prepared = []
            for item in packet["work"]:
                program, invocation, events = _validate_prefix(item["prefix"])
                if program["session"] != session_key:
                    raise SemanticSelectionError("Checkpoint work belongs to another session")
                prepared.append((item, program, invocation, events))
            orchestration = packet["orchestration"]
            if orchestration is not None:
                from .semantic_orchestration import validate_restored_orchestration

                validate_restored_orchestration(self, orchestration)
                if orchestration["session"] != session_key:
                    raise SemanticSelectionError("Restored coordinator belongs to another session")
            subscriptions = packet["subscriptions"]
            if not isinstance(subscriptions, dict):
                raise SemanticSelectionError(
                    "Checkpoint lacks its explicit retained subscription ownership"
                )
            known_invocations = {inv["invocationId"] for _, _, inv, _ in prepared}
            for identity, subscription in subscriptions.items():
                if not isinstance(subscription, dict) or set(subscription) != {
                    "track",
                    "handler",
                    "closed",
                }:
                    raise SemanticSelectionError("Malformed retained subscription checkpoint")
                handler, closed = subscription["handler"], subscription["closed"]
                owner = saved_registry.tracks.get(subscription["track"])
                if (
                    not isinstance(handler, dict)
                    or handler.get("kind") != "handler"
                    or handler.get("subscriptionId") != identity
                    or handler.get("lifetime") != "subscription"
                    or handler.get("ownerInvocationId") not in known_invocations
                    or owner is None
                    or owner.execution is not None
                    or owner.status == "active"
                    or owner.status == "completed"
                    and closed is None
                    or closed is not None
                    and (
                        not isinstance(closed, dict)
                        or closed.get("kind") != "handler-close"
                        or closed.get("subscriptionId") != identity
                        or closed.get("sequence", 0) <= handler["sequence"]
                    )
                ):
                    raise SemanticSelectionError(
                        "Restored subscription changed its handler, owner or lifetime"
                    )
            state = self._opened(session_key)
            state.orchestration = orchestration
            state.subscriptions = subscriptions
            try:
                session = state.runtime_session
                session.tracks = saved_registry
                session.execution_programs = contexts["programs"]
                session.execution_inputs = contexts["inputs"]
                session.execution_publication = contexts["publication"]
                session.execution_receipts = contexts["receipts"]
                session.execution_delivered = set(contexts["delivered"])
                session.completed_events.update(contexts["completedEvents"])
                session.execution_source_events = {}
                if self.verify_context:
                    for address, description in saved_registry.tracks.items():
                        session.memory.state = self.runtime.generator.tracks.describe_track(
                            session.state, address, description.purpose, description.revision
                        )
                known_tracks = {item["actualTrack"] for item, _, _, _ in prepared}
                if len(known_tracks) != len(prepared):
                    raise SemanticSelectionError("Two restored model invocations share a track")
                # Allocated children which have not sampled their first action
                # are still real owned work. Restore their observed bootstrap
                # and reservation instead of dropping or recreating them.
                for address, description in saved_registry.tracks.items():
                    if address in known_tracks:
                        continue
                    binding = description.execution
                    if binding is None:
                        self._observe(
                            session,
                            address,
                            {
                                "kind": "checkpoint-track",
                                "track": address,
                                "purpose": description.purpose,
                                "revision": description.revision,
                            },
                        )
                        subscription = next(
                            (
                                value
                                for value in subscriptions.values()
                                if value["track"] == address
                            ),
                            None,
                        )
                        if subscription is not None:
                            self._observe(
                                session,
                                address,
                                {
                                    "schema": "semantic-subscription-bootstrap/4",
                                    "kind": "subscription-bootstrap",
                                    "checkpoint": packet["digest"],
                                    "handler": subscription["handler"],
                                    "closed": subscription["closed"],
                                },
                            )
                        if (
                            state.orchestration is not None
                            and address == state.orchestration["track"]
                        ):
                            self.runtime.observe_text(
                                session,
                                address,
                                _canonical(
                                    {
                                        "schema": "semantic-orchestration-bootstrap/4",
                                        **state.orchestration,
                                        "receipts": contexts["receipts"],
                                    }
                                ),
                                role="environment",
                                effort=self.effort,
                            )
                        continue
                    publication = contexts["publication"].get(address, 0)
                    if publication not in {0, 1, 2} or description.bootstrap_ready != (
                        publication == 2
                    ):
                        raise SemanticSelectionError(
                            "Pending child bootstrap receipt is inconsistent"
                        )
                    if publication:
                        program = contexts["programs"][binding.program_id]
                        invocation = contexts["inputs"][
                            binding.invocation_id + ":" + binding.input_ref
                        ]
                        if (
                            program["digest"] != binding.source_digest
                            or invocation["programId"] != binding.program_id
                        ):
                            raise SemanticSelectionError(
                                "Pending child changed its registered source"
                            )
                        session.execution_source_events[address] = self._observe(
                            session, address, program
                        )
                        if publication == 2:
                            self._observe(session, address, invocation)
                for item, program, invocation, events in prepared:
                    if item["representation"] not in {
                        "explicit",
                        "compact",
                        "outcome-only",
                    } and not (item["representation"] is None and item["selections"] == 0):
                        raise SemanticSelectionError(
                            "Restored invocation lacks its exact representation"
                        )
                    address = item["actualTrack"]
                    description = saved_registry.tracks.get(address)
                    if description is None or description.execution is None:
                        raise SemanticSelectionError("Restored model work lacks its original owner")
                    binding = description.execution
                    if (
                        binding.invocation_id != invocation["invocationId"]
                        or binding.program_id != program["programId"]
                        or binding.source_digest != program["digest"]
                        or binding.entry != invocation["entry"]
                        or binding.realm_id != invocation["realmId"]
                        or binding.realm_policy != invocation["realmPolicy"]
                        or binding.trigger_id != invocation["triggerId"]
                        or binding.completion_mode != invocation["completionPolicy"]
                        or not set(binding.capability_grants) <= set(program["capabilities"])
                        or binding.input_ref
                        != hashlib.sha256(_canonical(invocation["inputGraph"])).hexdigest()
                    ):
                        raise SemanticSelectionError(
                            "Restored invocation changed its execution binding"
                        )
                    if type(item["completed"]) is not bool or item["completed"] != (
                        description.status in {"completed", "cancelled"}
                    ):
                        raise SemanticSelectionError(
                            "Restored continuation disagrees with its lifecycle disposition"
                        )
                    source_events = self._observe(session, address, program)
                    self._observe(session, address, invocation)
                    work = _Work(
                        session_key,
                        program["programId"],
                        invocation["invocationId"],
                        invocation["track"],
                        address,
                        None,
                        source_events=source_events,
                    )
                    state.work[work.invocation_id] = work
                    state.programs[work.program_id] = program["digest"]
                    bootstrap = {
                        "schema": "semantic-model-bootstrap/4",
                        "kind": "checkpoint-bootstrap",
                        "equivalence": packet["equivalence"],
                        "checkpoint": packet["digest"],
                        "priorContext": events,
                        "selectedState": item["prefix"].get("selectedState"),
                    }
                    self._observe(state.runtime_session, work.actual_track, bootstrap)
                    work.seen = [_canonical(event) for event in events]
                    work.selections = item["selections"]
                    work.source_reads = item["sourceReads"]
                    work.completed = item["completed"]
                    work.promise_effects = item["promiseEffects"]
                    work.settled_promises = set(item["settledPromises"])
                    work.waiting_promise = item["waitingPromise"]
                    work.request_receipts = item["requestReceipts"]
                    work.rejections = item["rejections"]
                    work.representation = item["representation"]
                    work.selection_records = item["selectionRecords"]
                    work.state_bootstrap = item["stateBootstrap"]
                    work.realm_cutoff = item["realmCutoff"]
                    work.selection_history = item["selectionHistory"]
                    work.escaped_closures = item["escapedClosures"]
                return {
                    "schema": "semantic-model-restore/4",
                    "session": session_key,
                    "checkpoint": packet["digest"],
                    "equivalence": packet["equivalence"],
                    "nativeEffects": 0,
                    "restoredInvocations": len(prepared),
                    "pendingExecutions": sum(
                        description.execution is not None and address not in known_tracks
                        for address, description in saved_registry.tracks.items()
                    ),
                }
            except Exception:
                self.sessions.pop(session_key, None)
                raise
