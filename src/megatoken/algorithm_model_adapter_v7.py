"""Native model-facing serving for the strict algorithm/7 profile.

The driver below owns only the native model turn boundary: public bootstrap
observations, context visibility, UTF-8 record continuation and canonical role
decoding.  Profile-specific context validation and causal replay stay in the
algorithm adapter.  No teacher values or future observations enter this path.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

from megatoken.contracts import MAIN_TRACK
from megatoken.controls import build_request_control, canonical_semantic_observation
from megatoken.generation.structure import ActionSpec

from .data.algorithm_context_v7 import (
    AlgorithmContextError,
    algorithm_execution_metadata,
    algorithm_selection_instruction,
    restore_algorithm_continuation_v7,
    stable_algorithm_hash,
    stable_algorithm_json,
    validate_algorithm_context,
)
from .data.algorithm_harness_v7 import AlgorithmHarnessError, AlgorithmReplayV7, validate_algorithm_program_v7


class AlgorithmServingError(ValueError):
    """A stale context or invalid native algorithm selection was rejected."""


@dataclass
class _DriverSession:
    runtime_session: Any
    persistent_visible_ids: set[str] = field(default_factory=set)
    latest_context_ids: set[str] = field(default_factory=set)
    committed_record_ids: set[str] = field(default_factory=set)
    current_fragment_ids: set[str] = field(default_factory=set)
    generated_lines: list[str] = field(default_factory=list)


class V7NativeModelDriver:
    """Shared native turn boundary for strict v7 profile adapters.

    A caller supplies one fixed v7 task metadata object, context checker,
    instruction provider and record decoder.  This driver never decides what a
    record means and never performs an external operation.
    """

    def __init__(
        self,
        runtime: Any,
        *,
        metadata: Mapping[str, Any],
        instruction_provider: Callable[..., str],
        record_decoder: Callable[..., Any],
        effort: int = 1,
        max_tokens: int = 512,
        max_record_bytes: int = 65536,
        verify_context: bool = True,
        track: str = MAIN_TRACK,
    ):
        if not 1 <= effort <= 6 or not 1 <= max_tokens <= 8192:
            raise ValueError("Native v7 inference effort or token budget is invalid")
        if not 1 <= max_record_bytes <= 65536:
            raise ValueError("Native v7 record byte budget is invalid")
        if verify_context is not True:
            raise ValueError("Native v7 serving must verify exact public context availability")
        if not isinstance(metadata, Mapping) or not isinstance(instruction_provider, Callable) or not isinstance(record_decoder, Callable):
            raise TypeError("Native v7 serving requires fixed metadata, controls and a record decoder")
        self.runtime = runtime
        self.metadata = copy.deepcopy(dict(metadata))
        self.instruction_provider = instruction_provider
        self.record_decoder = record_decoder
        self.effort = effort
        self.max_tokens = max_tokens
        self.max_record_bytes = max_record_bytes
        self.verify_context = True
        self.track = track
        self.sessions: dict[str, _DriverSession] = {}
        self.lock = RLock()

    def open(self, session_id: str) -> _DriverSession:
        if not isinstance(session_id, str) or not session_id:
            raise AlgorithmServingError("Native v7 session identity is required")
        with self.lock:
            if session_id in self.sessions:
                raise AlgorithmServingError("Native v7 session identity cannot be reused")
            runtime_session = self.runtime.open(
                metadata=copy.deepcopy(self.metadata),
                effort=self.effort,
                memory_writes=False,
            )
            state = _DriverSession(runtime_session)
            self.sessions[session_id] = state
            return state

    def _state(self, session_id: str) -> _DriverSession:
        try:
            return self.sessions[session_id]
        except KeyError as error:
            raise AlgorithmServingError("Native v7 session was not opened") from error

    @staticmethod
    def _payload(value: bytes | Mapping[str, Any]) -> bytes:
        if isinstance(value, bytes):
            return value
        if isinstance(value, Mapping):
            return canonical_semantic_observation(value)
        raise TypeError("Native v7 observations must be UTF-8 bytes or public JSON objects")

    def observe(
        self,
        session_id: str,
        value: bytes | Mapping[str, Any],
        *,
        role: str,
        visibility: str = "transient",
    ) -> tuple[str, ...]:
        state = self._state(session_id)
        if visibility not in {"persistent", "context", "generated", "transient"}:
            raise ValueError("Unknown native v7 visibility class")
        payload = self._payload(value)
        archive = getattr(state.runtime_session, "context_archive", None)
        before = set(getattr(archive, "events", {})) if archive is not None else set()
        self.runtime.observe_text(
            state.runtime_session,
            self.track,
            payload,
            role=role,
            effort=self.effort,
            content_kind="text",
        )
        archive = getattr(state.runtime_session, "context_archive", None)
        events = set(getattr(archive, "events", {})) if archive is not None else set()
        added = tuple(events - before)
        if not added:
            raise AlgorithmServingError("Native runtime did not retain the exact public observation")
        if visibility == "persistent":
            state.persistent_visible_ids.update(added)
        elif visibility == "context":
            state.latest_context_ids = set(added)
            state.current_fragment_ids.clear()
        elif visibility == "generated":
            state.committed_record_ids.update(added)
        return added

    def _require_visible_context(self, state: _DriverSession) -> None:
        required = (
            state.persistent_visible_ids
            | state.latest_context_ids
            | state.committed_record_ids
            | state.current_fragment_ids
        )
        if not required:
            return
        if not hasattr(self.runtime, "context_status"):
            raise AlgorithmServingError("Native runtime cannot verify current v7 context availability")
        visible = set(
            self.runtime.context_status(state.runtime_session, track=self.track)["visible_events"]
        )
        missing = required - visible
        if missing:
            raise AlgorithmServingError("Public algorithm definitions or current context are not visible")

    def select_record(
        self,
        session_id: str,
        context: Mapping[str, Any],
        *,
        validate_context: Callable[[Mapping[str, Any]], Any],
        commit_record: Callable[[str, Any], Any],
        context_role: str = "system",
    ) -> dict[str, Any]:
        """Expose one checked context, sample one record, and commit its raw bytes."""
        state = self._state(session_id)
        try:
            validate_context(context)
        except (TypeError, ValueError, KeyError) as error:
            raise AlgorithmServingError(f"Native v7 context was rejected: {error}") from error
        self.observe(session_id, context, role=context_role, visibility="context")
        self._require_visible_context(state)
        config = self.runtime.generator.generator.config
        fragment_limit = min(self.max_tokens, config.max_capacity)
        fragments: list[bytes] = []
        for fragment in range(64):
            archive_before = getattr(state.runtime_session, "context_archive", None)
            before = set(getattr(archive_before, "events", {})) if archive_before is not None else set()
            generated = self.runtime.generate(
                state.runtime_session,
                tracks=(self.track,),
                specs=(
                    ActionSpec(
                        text_kind="utf8",
                        min_tokens=1,
                        max_tokens=fragment_limit,
                        require_terminal=None,
                    ),
                ),
                controls=(
                    build_request_control(
                        instruction=self.instruction_provider(continuation=fragment > 0)
                    ),
                ),
                effort=self.effort,
                continuation_mode="encoded_text",
            )
            tape = generated.tape
            if (
                len(tape.payloads) != 1
                or not tape.valid[0]
                or tape.modalities[0] != "text"
                or tape.tracks[0] != self.track
                or not isinstance(tape.payloads[0], bytes)
            ):
                raise AlgorithmServingError("Native model did not emit one valid UTF-8 record fragment")
            fragments.append(tape.payloads[0])
            if self.verify_context:
                archive = getattr(state.runtime_session, "context_archive", None)
                current_events = set(getattr(archive, "events", {})) if archive is not None else set()
                committed_fragments = current_events - before
                if not committed_fragments:
                    committed_fragments = set(getattr(tape, "event_ids", ()))
                if not committed_fragments:
                    raise AlgorithmServingError("Native runtime did not retain the current model fragment")
                state.current_fragment_ids.update(committed_fragments)
            if sum(map(len, fragments)) > self.max_record_bytes:
                raise AlgorithmServingError("Native model record exceeds its public byte budget")
            if tape.terminal[0]:
                break
            if not tape.terminal[0]:
                self._require_visible_context(state)
        else:
            raise AlgorithmServingError("Native model record did not terminate within its fragment budget")

        payload = b"".join(fragments)
        try:
            line = payload.decode("utf-8", errors="strict")
            record = self.record_decoder(line, role="generated")
            receipt = commit_record(line, record)
        except (UnicodeDecodeError, TypeError, ValueError, KeyError) as error:
            raise AlgorithmServingError(f"Native model emitted an invalid algorithm/7 record: {error}") from error
        state.generated_lines.append(line)
        state.committed_record_ids.update(state.current_fragment_ids)
        state.current_fragment_ids.clear()
        self._require_visible_context(state)
        return {"line": line, "record": record, "receipt": receipt, "payload": payload}


@dataclass
class _AlgorithmSession:
    replay: AlgorithmReplayV7
    driver: _DriverSession
    producer_definitions: dict[str, Any] = field(default_factory=dict)
    consumer_definitions: dict[str, Any] = field(default_factory=dict)
    contexts: list[dict[str, Any]] = field(default_factory=list)
    timeline: list[dict[str, Any]] = field(default_factory=list)
    dispatched: set[str] = field(default_factory=set)


class AlgorithmModelAdapterV7:
    """Selected algorithm execution using the actual Megatoken runtime."""

    def __init__(
        self,
        runtime: Any,
        *,
        effort: int = 1,
        max_tokens: int = 512,
        max_record_bytes: int = 65536,
        verify_context: bool = True,
    ):
        from .data import algorithm_v7

        self.runtime = runtime
        self.driver = V7NativeModelDriver(
            runtime,
            metadata=algorithm_execution_metadata(),
            instruction_provider=algorithm_selection_instruction,
            record_decoder=algorithm_v7.parse_algorithm_record,
            effort=effort,
            max_tokens=max_tokens,
            max_record_bytes=max_record_bytes,
            verify_context=verify_context,
        )
        self.sessions: dict[str, _AlgorithmSession] = {}
        self.lock = RLock()

    def open(
        self,
        session_id: str,
        program: Mapping[str, Any],
        input_graph: Mapping[str, Any],
        *,
        invocation_id: str = "invocation1",
    ) -> dict[str, Any]:
        with self.lock:
            if session_id in self.sessions:
                raise AlgorithmServingError("Algorithm serving session identity cannot be reused")
            try:
                validated = validate_algorithm_program_v7(program)
                capabilities = list(dict.fromkeys(row["operation"] for row in validated["boundaries"]))
                replay = AlgorithmReplayV7(
                    validated,
                    input_graph,
                    capabilities=capabilities,
                    invocation_id=invocation_id,
                    session_id=session_id,
                )
            except (AlgorithmHarnessError, TypeError, ValueError, KeyError) as error:
                raise AlgorithmServingError(f"Invalid native algorithm definition or input: {error}") from error
            driver = self.driver.open(session_id)
            timeline = [
                {"mode": "observed", "kind": "program", "program": copy.deepcopy(validated)},
                {"mode": "observed", "kind": "input", "graph": copy.deepcopy(dict(input_graph))},
            ]
            self.sessions[session_id] = _AlgorithmSession(replay, driver, timeline=timeline)
            self.driver.observe(session_id, timeline[0], role="user", visibility="persistent")
            self.driver.observe(session_id, timeline[1], role="environment", visibility="persistent")
            return {"sessionId": session_id, "invocationId": invocation_id, "track": MAIN_TRACK}

    def _session(self, session_id: str) -> _AlgorithmSession:
        try:
            return self.sessions[session_id]
        except KeyError as error:
            raise AlgorithmServingError("Algorithm serving session was not opened") from error

    def _context(self, state: _AlgorithmSession) -> dict[str, Any]:
        try:
            context = state.replay.model_context(state.producer_definitions)
            validate_algorithm_context(
                context,
                state.replay.history,
                program=state.replay.program,
                input_graph=state.replay.input_graph,
                invocation_id=state.replay.invocation_id,
                live_state=state.replay.live_state(),
                definitions=state.consumer_definitions,
            )
            return context
        except (AlgorithmContextError, AlgorithmHarnessError, TypeError, ValueError, KeyError) as error:
            raise AlgorithmServingError(f"Current algorithm context failed verification: {error}") from error

    def select(self, session_id: str, *, context: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Sample and validate one record from the exact live algorithm context."""
        state = self._session(session_id)
        expected_context = self._context(state)
        if context is not None:
            try:
                if stable_algorithm_json(context) != stable_algorithm_json(expected_context):
                    raise AlgorithmServingError("Supplied algorithm context is stale or has altered definitions")
            except (TypeError, ValueError) as error:
                if isinstance(error, AlgorithmServingError):
                    raise
                raise AlgorithmServingError(f"Supplied algorithm context is malformed: {error}") from error

        def validate(current):
            if stable_algorithm_json(current) != stable_algorithm_json(expected_context):
                raise AlgorithmServingError("Algorithm context changed before model selection")

        def commit(line, _record):
            return state.replay.commit(line)

        try:
            selected = self.driver.select_record(
                session_id,
                expected_context,
                validate_context=validate,
                commit_record=commit,
            )
        except (AlgorithmServingError, AlgorithmHarnessError, TypeError, ValueError, KeyError) as error:
            if isinstance(error, AlgorithmServingError):
                raise
            raise AlgorithmServingError(f"Native algorithm selection was rejected: {error}") from error
        state.contexts.append(copy.deepcopy(expected_context))
        state.timeline.append({"mode": "generated", "kind": "semantic", "line": selected["line"]})
        selected["context"] = copy.deepcopy(expected_context)
        selected["dispatch"] = copy.deepcopy(selected["receipt"]["dispatch"])
        selected["completion"] = copy.deepcopy(state.replay.completion)
        return selected

    def record_dispatch(
        self,
        session_id: str,
        request_id: str,
        *,
        key: str,
        attempt: int = 1,
    ) -> dict[str, Any]:
        """Record the host's already completed dispatch acknowledgement."""
        state = self._session(session_id)
        pending = state.replay.pending
        if (
            pending is None
            or request_id != pending["id"]
            or key != pending["key"]
            or type(attempt) is not int
            or attempt != 1
            or request_id in state.dispatched
        ):
            raise AlgorithmServingError("Dispatch acknowledgement differs from the selected request")
        row = {"mode": "observed", "kind": "dispatch", "request": request_id, "key": key, "attempt": attempt}
        self.driver.observe(session_id, row, role="environment")
        state.timeline.append(row)
        state.dispatched.add(request_id)
        return copy.deepcopy(row)

    def observe(self, session_id: str, line: str) -> dict[str, Any]:
        """Commit one host-originated observation after its acknowledged dispatch."""
        from .data.algorithm_v7 import parse_algorithm_record

        state = self._session(session_id)
        try:
            record = parse_algorithm_record(line, role="observed")
        except (TypeError, ValueError) as error:
            raise AlgorithmServingError(f"Malformed external algorithm observation: {error}") from error
        if record.get("request") not in state.dispatched:
            raise AlgorithmServingError("External observation needs its one prior dispatch acknowledgement")
        try:
            result = state.replay.observe(line)
        except (AlgorithmHarnessError, TypeError, ValueError, KeyError) as error:
            raise AlgorithmServingError(f"External observation differs from current algorithm prefix: {error}") from error
        row = {"mode": "observed", "kind": "observation", "line": line}
        self.driver.observe(session_id, line.encode("utf-8"), role="environment")
        state.timeline.append(row)
        return {"row": row, "result": result, "completion": copy.deepcopy(state.replay.completion)}

    def checkpoint(self, session_id: str) -> dict[str, Any]:
        return self._session(session_id).replay.checkpoint()

    def restore(
        self,
        session_id: str,
        packet: Mapping[str, Any],
        *,
        contexts: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Rebuild a native context from public contexts and continuation history.

        Restoring history observes the already committed public dispatch receipt;
        it never invokes a host service. A selected request at the checkpoint tail
        stays pending and is returned to the caller for explicit reconciliation.
        """
        if session_id in self.sessions:
            raise AlgorithmServingError("Cannot restore over an open algorithm session")
        try:
            verified = restore_algorithm_continuation_v7(packet)
            if not isinstance(contexts, Sequence) or isinstance(contexts, (str, bytes)):
                raise AlgorithmServingError("Continuation needs its public context history")
            replay = AlgorithmReplayV7(
                packet["program"],
                packet["inputGraph"],
                capabilities=packet["capabilities"],
                invocation_id=packet["invocationId"],
                session_id=packet["sessionId"],
            )
            driver = self.driver.open(session_id)
            state = _AlgorithmSession(replay, driver)
            self.sessions[session_id] = state
            program_row = {"mode": "observed", "kind": "program", "program": copy.deepcopy(replay.program)}
            input_row = {"mode": "observed", "kind": "input", "graph": copy.deepcopy(replay.input_graph)}
            state.timeline.extend((program_row, input_row))
            self.driver.observe(session_id, program_row, role="user", visibility="persistent")
            self.driver.observe(session_id, input_row, role="environment", visibility="persistent")
            context_index = 0
            for history_row in packet["history"]:
                if history_row["mode"] == "generated":
                    if context_index >= len(contexts):
                        raise AlgorithmServingError("Continuation omitted a public model context")
                    context = copy.deepcopy(dict(contexts[context_index]))
                    expected = replay.model_context(state.producer_definitions)
                    if stable_algorithm_json(context) != stable_algorithm_json(expected):
                        raise AlgorithmServingError("Continuation context is stale or names another definition")
                    validate_algorithm_context(
                        context,
                        replay.history,
                        program=replay.program,
                        input_graph=replay.input_graph,
                        invocation_id=replay.invocation_id,
                        live_state=replay.live_state(),
                        definitions=state.consumer_definitions,
                    )
                    self.driver.observe(session_id, context, role="system", visibility="context")
                    state.contexts.append(context)
                    state.timeline.append({"mode": "generated", "kind": "semantic", "line": history_row["line"]})
                    replay.commit(history_row["line"])
                    self.driver.observe(
                        session_id,
                        history_row["line"].encode("utf-8"),
                        role="assistant",
                        visibility="generated",
                    )
                    context_index += 1
                else:
                    if replay.pending is not None:
                        request_id = replay.pending["id"]
                        dispatch_row = {
                            "mode": "observed",
                            "kind": "dispatch",
                            "request": request_id,
                            "key": replay.pending["key"],
                            "attempt": 1,
                        }
                        self.driver.observe(session_id, dispatch_row, role="environment")
                        state.timeline.append(dispatch_row)
                        state.dispatched.add(request_id)
                    replay.observe(history_row["line"])
                    observation = {"mode": "observed", "kind": "observation", "line": history_row["line"]}
                    self.driver.observe(session_id, history_row["line"].encode("utf-8"), role="environment")
                    state.timeline.append(observation)
            if context_index != len(contexts):
                raise AlgorithmServingError("Continuation contains an unused future model context")
            if stable_algorithm_json(replay.live_state()) != stable_algorithm_json(verified.live_state()):
                raise AlgorithmServingError("Restored native state differs from the verified continuation")
            self.driver._require_visible_context(driver)
            return {
                "sessionId": session_id,
                "invocationId": replay.invocation_id,
                "track": MAIN_TRACK,
                "pending": copy.deepcopy(replay.pending),
                "completion": copy.deepcopy(replay.completion),
            }
        except (AlgorithmServingError, AlgorithmContextError, AlgorithmHarnessError, TypeError, ValueError, KeyError) as error:
            self.sessions.pop(session_id, None)
            self.driver.sessions.pop(session_id, None)
            if isinstance(error, AlgorithmServingError):
                raise
            raise AlgorithmServingError(f"Algorithm continuation restore failed: {error}") from error


def create_algorithm_model_adapter_v7(runtime: Any, **options) -> AlgorithmModelAdapterV7:
    """Construct the algorithm-only v7 adapter around a native runtime."""
    return AlgorithmModelAdapterV7(runtime, **options)

