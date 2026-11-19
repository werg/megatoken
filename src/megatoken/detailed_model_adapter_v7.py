"""Native model serving for the strict detailed/7 representation."""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from megatoken.contracts import MAIN_TRACK

from .algorithm_model_adapter_v7 import V7NativeModelDriver
from .data.detailed_context_v7 import (
    DetailedContextV7Error,
    detailed_execution_metadata,
    detailed_selection_instruction,
    validate_detailed_context,
    validate_detailed_continuation_v7,
)
from .data.detailed_v7 import DetailedV7Error, parse_record
from .data.semantic_contract_v7 import PUBLIC_SCHEMA, TRACE_SCHEMA, validate_envelope_record


class DetailedServingError(ValueError):
    """A stale detailed context or invalid native selection was rejected."""


@dataclass
class _DetailedServingSession:
    driver: V7NativeModelDriver
    runtime_state: Any
    timeline: list[dict[str, Any]] = field(default_factory=list)
    owner_sequences: dict[str, int] = field(default_factory=dict)
    contexts: list[dict[str, Any]] = field(default_factory=list)
    poisoned: bool = False


class DetailedModelAdapterV7:
    """Serve selected detailed records through the actual native model runtime.

    The caller supplies two independent checks: one for each observed public row,
    and one that reconstructs the exact continuation state before a target. The
    adapter owns strict public ordering, context publication, record decoding,
    native role framing, and continuation restore. It never dispatches a host
    operation or reads a private evidence sidecar.
    """

    def __init__(
        self,
        runtime: Any,
        *,
        effort: int = 1,
        max_tokens: int = 512,
        max_record_bytes: int = 65536,
    ):
        self.runtime = runtime
        self.effort = effort
        self.max_tokens = max_tokens
        self.max_record_bytes = max_record_bytes
        self.sessions: dict[str, _DetailedServingSession] = {}

    def open(self, session_id: str) -> dict[str, Any]:
        """Open an isolated detailed session with the fixed v7 task metadata."""
        if not isinstance(session_id, str) or not session_id or session_id in self.sessions:
            raise DetailedServingError("Detailed session identity is missing or already used")
        driver = V7NativeModelDriver(
            self.runtime,
            metadata=detailed_execution_metadata(),
            instruction_provider=detailed_selection_instruction,
            record_decoder=parse_record,
            effort=self.effort,
            max_tokens=self.max_tokens,
            max_record_bytes=self.max_record_bytes,
            verify_context=True,
            track=MAIN_TRACK,
            # Every next detailed context carries the full current public prefix.
            retain_committed_records=False,
        )
        runtime_state = driver.open(session_id)
        self.sessions[session_id] = _DetailedServingSession(
            driver,
            runtime_state,
        )
        return {
            "sessionId": session_id,
            "task": detailed_execution_metadata()["task"],
            "track": MAIN_TRACK,
        }

    def _session(self, session_id: str) -> _DetailedServingSession:
        try:
            state = self.sessions[session_id]
        except KeyError as error:
            raise DetailedServingError("Detailed native session was not opened") from error
        if state.poisoned or state.driver.sessions[session_id].poisoned:
            state.poisoned = True
            raise DetailedServingError(
                "Detailed session is poisoned; restore a fresh public prefix"
            )
        return state

    @staticmethod
    def _validate_row(
        session_id: str,
        timeline: Sequence[Mapping[str, Any]],
        owner_sequences: Mapping[str, int],
        row: Mapping[str, Any],
        *,
        mode: str,
    ) -> dict[str, Any]:
        try:
            validate_envelope_record(row)
        except (TypeError, ValueError, KeyError) as error:
            raise DetailedServingError(f"Invalid detailed/7 public row: {error}") from error
        if row.get("session") != session_id or row.get("mode") != mode:
            raise DetailedServingError(
                "Detailed row session or authority differs from the live prefix"
            )
        if row.get("sequence") != len(timeline) + 1:
            raise DetailedServingError(
                "Detailed public sequence is stale, skipped, or from the future"
            )
        expected_owner_sequence = owner_sequences.get(row["track"], 0) + 1
        if row.get("ownerSequence") != expected_owner_sequence:
            raise DetailedServingError("Detailed track owner sequence is stale or out of order")
        return copy.deepcopy(dict(row))

    @staticmethod
    def _observation_role(row: Mapping[str, Any]) -> str:
        return "user" if row["kind"] in {"program", "invocation"} else "environment"

    @staticmethod
    def _validate_target(
        target: Mapping[str, Any], timeline: Sequence[Mapping[str, Any]]
    ) -> dict[str, Any]:
        if not isinstance(target, Mapping) or set(target) != {"sequence", "track", "invocationId"}:
            raise DetailedServingError(
                "Detailed target must bind only its next sequence, public track, and invocation"
            )
        if type(target.get("sequence")) is not int or target["sequence"] != len(timeline) + 1:
            raise DetailedServingError("Detailed target is not the next uncommitted public row")
        if not all(
            isinstance(target.get(key), str) and target[key] for key in ("track", "invocationId")
        ):
            raise DetailedServingError("Detailed target lacks its public track or invocation")
        invocation = next(
            (
                row
                for row in reversed(timeline)
                if row.get("kind") == "invocation"
                and row.get("invocationId") == target["invocationId"]
            ),
            None,
        )
        if invocation is None or invocation.get("track") != target["track"]:
            raise DetailedServingError(
                "Detailed target track differs from its observed invocation owner"
            )
        return copy.deepcopy(dict(target))

    def observe(
        self,
        session_id: str,
        row: Mapping[str, Any],
        *,
        validate_observation: Callable[[Sequence[Mapping[str, Any]], Mapping[str, Any]], Any],
    ) -> dict[str, Any]:
        """Validate and observe one already available public envelope row.

        ``validate_observation(prior, row)`` must independently check its causal
        meaning before the row is appended to native model context.
        """
        state = self._session(session_id)
        if not callable(validate_observation):
            raise TypeError("Detailed serving needs an independent observation validator")
        checked = self._validate_row(
            session_id,
            state.timeline,
            state.owner_sequences,
            row,
            mode="observed",
        )
        prior = copy.deepcopy(state.timeline)
        try:
            validate_observation(prior, checked)
        except (DetailedServingError, TypeError, ValueError, KeyError) as error:
            if isinstance(error, DetailedServingError):
                raise
            raise DetailedServingError(f"Detailed observation was rejected: {error}") from error
        role = self._observation_role(checked)
        before_context = set(state.driver.sessions[session_id].latest_context_ids)
        try:
            state.driver.observe(
                session_id,
                checked,
                role=role,
                visibility="transient",
                track=MAIN_TRACK,
            )
        except (TypeError, ValueError, KeyError) as error:
            if state.driver.sessions[session_id].latest_context_ids != before_context:
                state.poisoned = True
            raise DetailedServingError(f"Detailed native observation failed: {error}") from error
        state.timeline.append(checked)
        state.owner_sequences[checked["track"]] = checked["ownerSequence"]
        return copy.deepcopy(checked)

    def select_record(
        self,
        session_id: str,
        context: Mapping[str, Any],
        target: Mapping[str, Any],
        *,
        expected_state_provider: Callable[
            [Sequence[Mapping[str, Any]], Mapping[str, Any]], Mapping[str, Any]
        ],
        commit_record: Callable[[str, tuple[str, dict]], Any],
    ) -> dict[str, Any]:
        """Select, independently commit, and append one detailed generated row."""
        state = self._session(session_id)
        if not callable(expected_state_provider) or not callable(commit_record):
            raise DetailedServingError("Detailed selection needs live-state and commit callbacks")
        prior = copy.deepcopy(state.timeline)
        target_row = self._validate_target(target, prior)
        try:
            expected_state = expected_state_provider(prior, target_row)
        except (DetailedServingError, TypeError, ValueError, KeyError) as error:
            if isinstance(error, DetailedServingError):
                raise
            raise DetailedServingError(
                f"Detailed selected state could not be replayed: {error}"
            ) from error
        if not isinstance(expected_state, Mapping):
            raise DetailedServingError("Detailed selected-state replay did not return an object")
        before_context = set(state.driver.sessions[session_id].latest_context_ids)

        def check_context(candidate: Mapping[str, Any]) -> bool:
            try:
                validate_detailed_context(
                    candidate,
                    prior,
                    {
                        "sequence": target_row["sequence"],
                        "invocationId": target_row["invocationId"],
                    },
                    expected_state=expected_state,
                )
                validate_detailed_continuation_v7(
                    candidate,
                    prior_timeline=prior,
                    target={
                        "sequence": target_row["sequence"],
                        "invocationId": target_row["invocationId"],
                    },
                    expected_state=expected_state,
                )
            except (DetailedContextV7Error, TypeError, ValueError, KeyError) as error:
                raise DetailedServingError(
                    f"Detailed current prefix was rejected: {error}"
                ) from error
            return True

        def commit(line: str, parsed: tuple[str, dict]) -> dict[str, Any]:
            row = {
                "schema": PUBLIC_SCHEMA,
                "session": session_id,
                "sequence": target_row["sequence"],
                "ownerSequence": state.owner_sequences.get(target_row["track"], 0) + 1,
                "track": target_row["track"],
                "kind": "semantic",
                "mode": "generated",
                "invocationId": target_row["invocationId"],
                "traceFormat": TRACE_SCHEMA,
                "line": line,
            }
            checked_row = self._validate_row(
                session_id,
                state.timeline,
                state.owner_sequences,
                row,
                mode="generated",
            )
            try:
                receipt = commit_record(line, parsed)
            except (
                DetailedServingError,
                DetailedV7Error,
                TypeError,
                ValueError,
                KeyError,
            ) as error:
                if isinstance(error, DetailedServingError):
                    raise
                raise DetailedServingError(
                    f"Selected detailed record failed prefix replay: {error}"
                ) from error
            state.timeline.append(checked_row)
            state.owner_sequences[checked_row["track"]] = checked_row["ownerSequence"]
            return {"commit": receipt, "timelineRow": copy.deepcopy(checked_row)}

        try:
            selected = state.driver.select_record(
                session_id,
                context,
                validate_context=check_context,
                commit_record=commit,
                context_role="system",
                track=MAIN_TRACK,
            )
        except (DetailedServingError, DetailedV7Error, TypeError, ValueError, KeyError) as error:
            if state.driver.sessions[session_id].latest_context_ids != before_context:
                state.poisoned = True
            if isinstance(error, DetailedServingError):
                raise
            raise DetailedServingError(f"Detailed native selection failed: {error}") from error
        state.contexts.append(copy.deepcopy(dict(context)))
        selected["context"] = copy.deepcopy(dict(context))
        selected["timelineRow"] = copy.deepcopy(state.timeline[-1])
        return selected

    def restore(
        self,
        session_id: str,
        *,
        prior_timeline: Sequence[Mapping[str, Any]],
        contexts: Sequence[Mapping[str, Any]],
        target: Mapping[str, Any],
        expected_state_provider: Callable[
            [Sequence[Mapping[str, Any]], Mapping[str, Any]], Mapping[str, Any]
        ],
        validate_observation: Callable[[Sequence[Mapping[str, Any]], Mapping[str, Any]], Any],
    ) -> dict[str, Any]:
        """Rebuild the exact public prefix in model context without dispatching.

        Every historical generated row is preceded by its saved full-prefix
        context. Observed dispatches and results are replayed as public bytes;
        no service API is called during restore.
        """
        if session_id in self.sessions:
            raise DetailedServingError("Cannot restore over an already open detailed session")
        if not callable(expected_state_provider) or not callable(validate_observation):
            raise DetailedServingError("Detailed restore requires public replay callbacks")
        if not isinstance(prior_timeline, Sequence) or isinstance(prior_timeline, (str, bytes)):
            raise DetailedServingError("Detailed restore needs its exact public timeline")
        if not isinstance(contexts, Sequence) or isinstance(contexts, (str, bytes)):
            raise DetailedServingError("Detailed restore needs its generated-record contexts")
        state = None
        try:
            self.open(session_id)
            state = self._session(session_id)
            context_index = 0
            for raw_row in prior_timeline:
                row = dict(raw_row)
                prior = copy.deepcopy(state.timeline)
                if row.get("kind") == "semantic":
                    if context_index >= len(contexts):
                        raise DetailedServingError(
                            "Detailed restore omitted a generated-record context"
                        )
                    _checked = self._validate_row(
                        session_id,
                        state.timeline,
                        state.owner_sequences,
                        row,
                        mode="generated",
                    )
                    context = copy.deepcopy(dict(contexts[context_index]))
                    target_prefix = {
                        "sequence": row["sequence"],
                        "invocationId": row["invocationId"],
                    }
                    expected_state = expected_state_provider(prior, row)
                    validate_detailed_continuation_v7(
                        context,
                        prior_timeline=prior,
                        target=target_prefix,
                        expected_state=expected_state,
                    )
                    state.driver.observe(
                        session_id,
                        context,
                        role="system",
                        visibility="context",
                        track=MAIN_TRACK,
                    )
                    state.driver._require_visible_on_track(state.runtime_state, MAIN_TRACK)
                    parse_record(row.get("line"), role="generated")
                    state.driver.observe(
                        session_id,
                        row["line"].encode("utf-8"),
                        role="assistant",
                        visibility="transient",
                        track=MAIN_TRACK,
                    )
                    state.timeline.append(_checked)
                    state.owner_sequences[_checked["track"]] = _checked["ownerSequence"]
                    state.contexts.append(context)
                    context_index += 1
                else:
                    self.observe(
                        session_id,
                        row,
                        validate_observation=validate_observation,
                    )
            if context_index != len(contexts):
                raise DetailedServingError("Detailed restore includes a future model context")
            target_row = self._validate_target(target, state.timeline)
            expected_state = expected_state_provider(copy.deepcopy(state.timeline), target_row)
            return {
                "sessionId": session_id,
                "timeline": copy.deepcopy(state.timeline),
                "target": target_row,
                "expectedState": copy.deepcopy(dict(expected_state)),
            }
        except (
            DetailedServingError,
            DetailedContextV7Error,
            DetailedV7Error,
            TypeError,
            ValueError,
            KeyError,
        ) as error:
            if state is not None:
                state.driver.sessions.pop(session_id, None)
            self.sessions.pop(session_id, None)
            if isinstance(error, DetailedServingError):
                raise
            raise DetailedServingError(f"Detailed public restore failed: {error}") from error


def create_detailed_model_adapter_v7(runtime: Any, **options) -> DetailedModelAdapterV7:
    """Construct the strict detailed/7 adapter around a native model runtime."""
    return DetailedModelAdapterV7(runtime, **options)
