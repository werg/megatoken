"""Exact public-prefix and continuation checks for selected-detailed/1."""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping, Sequence
from typing import Any

from .algorithm_v7 import stable_algorithm_hash, stable_algorithm_json
from .detailed_v7 import parse_record
from .semantic_contract_v7 import (
    CONTEXT_SCHEMA,
    CONTINUATION_SCHEMA,
    DETAILED,
    PUBLIC_SCHEMA,
    REPRESENTATION,
    TRACE_SCHEMA,
    SemanticContractV7Error,
    validate_envelope_record,
)

PREFIX_SCHEMA = DETAILED["contextEnvelope"]["prefixSchema"]
CONTEXT_FIELDS = set(DETAILED["contextEnvelope"]["fields"])
PREFIX_FIELDS = set(DETAILED["contextEnvelope"]["prefixFields"])
STATE_FIELDS = set(DETAILED["modelContext"]["stateFields"])
HASH = re.compile(r"[0-9a-f]{64}\Z")


class DetailedContextV7Error(ValueError):
    """A public detailed context is stale, malformed, or causally incomplete."""


def _fail(message: str) -> None:
    raise DetailedContextV7Error(message)


def _same(left: Any, right: Any) -> bool:
    try:
        return stable_algorithm_json(left) == stable_algorithm_json(right)
    except (TypeError, ValueError):
        return False


def _exact_fields(item: Any, expected: set[str], label: str) -> None:
    if not isinstance(item, Mapping) or set(item) != expected:
        _fail(f"{label} has missing or undeclared fields")


def detailed_execution_metadata() -> dict[str, str]:
    return {"task": PUBLIC_SCHEMA, "modality": "text"}


def detailed_selection_instruction(*, continuation: bool = False) -> str:
    instruction = (
        "Emit exactly one complete canonical detailed semantic-trace/7 record. Use only the "
        "observed immutable program and input, the full current detailed context, committed "
        "selected records, and already observed environment feedback. Check ownership, source "
        "spans, heap state, and pending work before each selection. Use only source-authorized "
        "effect boundaries. Never predict an observation or use private evidence or future data."
    )
    if continuation:
        instruction += " Continue the exact current record bytes and stop only at its end."
    return instruction


def _source_scope(
    timeline: Sequence[Mapping[str, Any]], invocation_id: str
) -> tuple[dict, set[str], set[str], set[str], set[str]]:
    invocation = next(
        (
            row
            for row in timeline
            if row.get("kind") == "invocation" and row.get("invocationId") == invocation_id
        ),
        None,
    )
    if invocation is None:
        _fail("Detailed context has no prior observed invocation")
    realm = invocation["realmId"]
    realm_calls = [
        row for row in timeline if row.get("kind") == "invocation" and row.get("realmId") == realm
    ]
    invocations = {row["invocationId"] for row in realm_calls}
    programs = {row["programId"] for row in realm_calls}
    tracks = {row["track"] for row in realm_calls}
    subscriptions = {
        row["subscriptionId"]
        for row in timeline
        if row.get("kind") == "handler" and row.get("realmId") == realm
    }
    return invocation, invocations, programs, tracks, subscriptions


def _realm_committed(
    timeline: Sequence[Mapping[str, Any]], invocation_id: str
) -> list[Mapping[str, Any]]:
    invocation, identities, programs, tracks, subscriptions = _source_scope(timeline, invocation_id)
    return [
        row
        for row in timeline
        if (
            row.get("kind") == "program"
            and row.get("programId") in programs
            or row.get("invocationId") in identities
            or row.get("subscriptionId") in subscriptions
            or row.get("kind") in {"source-read", "source-evict"}
            and row.get("track") in tracks
            and row.get("programId") in programs
        )
    ]


def _committed(
    timeline: Sequence[Mapping[str, Any]], invocation: Mapping[str, Any]
) -> list[Mapping[str, Any]]:
    program_id = invocation["programId"]
    invocation_id = invocation["invocationId"]
    track = invocation["track"]
    return [
        row
        for row in timeline
        if (
            row.get("kind") == "program"
            and row.get("programId") == program_id
            or row.get("invocationId") == invocation_id
            or row.get("kind") in {"source-read", "source-evict"}
            and row.get("track") == track
            and row.get("programId") == program_id
        )
    ]


def _validate_prefix_rows(rows: Any, *, visible_through: int, label: str) -> None:
    if not isinstance(rows, list):
        _fail(f"{label} must be an ordered public timeline list")
    prior_sequence = 0
    for row in rows:
        try:
            validate_envelope_record(row)
        except (SemanticContractV7Error, TypeError, ValueError, KeyError) as error:
            raise DetailedContextV7Error(f"Invalid {label} row: {error}") from error
        sequence = row["sequence"]
        if sequence <= prior_sequence or sequence > visible_through:
            _fail(f"{label} contains duplicate, unordered, or future public evidence")
        prior_sequence = sequence


def validate_detailed_context(
    context: Mapping[str, Any],
    prior_timeline: Sequence[Mapping[str, Any]],
    target: Mapping[str, Any] | int,
    *,
    expected_state: Mapping[str, Any] | None = None,
) -> bool:
    """Check the exact current prefix and a generated target's causal boundary.

    ``prior_timeline`` is the only available history. The context may neither
    omit a relevant committed row nor include a future one. When the caller
    has independently restored the selected continuation, ``expected_state``
    binds every state field to that replay result.
    """
    _exact_fields(context, CONTEXT_FIELDS, "Detailed context envelope")
    if context.get("schema") != CONTEXT_SCHEMA or context.get("representation") != REPRESENTATION:
        _fail("Detailed context requires semantic-context/7 and representation:detailed")
    prefix = context["prefix"]
    _exact_fields(prefix, PREFIX_FIELDS, "Detailed inference prefix")
    if (
        prefix.get("schema") != PREFIX_SCHEMA
        or prefix.get("representation") != REPRESENTATION
        or prefix.get("traceFormat") != TRACE_SCHEMA
    ):
        _fail("Detailed prefix requires selected-inference-prefix/3")
    try:
        target_sequence = target if type(target) is int else target["sequence"]
        target_invocation = None if type(target) is int else target.get("invocationId")
    except (KeyError, TypeError):
        _fail("Detailed target must have a sequence and invocation")
    if type(target_sequence) is not int or target_sequence < 1:
        _fail("Detailed target sequence must be positive")
    if type(target) is not int and "kind" in target:
        try:
            validate_envelope_record(target)
            parse_record(target.get("line"), role="generated")
        except (SemanticContractV7Error, TypeError, ValueError, KeyError) as error:
            raise DetailedContextV7Error(
                f"Detailed target is not a generated v7 record: {error}"
            ) from error
        if target.get("kind") != "semantic" or target.get("traceFormat") != TRACE_SCHEMA:
            _fail("Detailed context can only precede a generated semantic target")
    elif type(target) is not int:
        if target_invocation is None:
            _fail("Uncommitted detailed target requires its invocation identity")
    prior = list(prior_timeline)
    if not prior or prior[-1].get("sequence") >= target_sequence:
        _fail("Detailed context must precede its generated target")
    invocation, _, _, _, _ = _source_scope(prior, prefix.get("invocationId"))
    if target_invocation is not None and target_invocation != invocation["invocationId"]:
        _fail("Detailed context is bound to another target invocation")
    program = next(
        (
            row
            for row in prior
            if row.get("kind") == "program" and row.get("programId") == invocation["programId"]
        ),
        None,
    )
    if program is None:
        _fail("Detailed context program is missing from the committed prefix")

    expected = {
        "programId": invocation["programId"],
        "entry": invocation["entry"],
        "source": program["source"],
        "sourceDigest": program["digest"],
        "invocationId": invocation["invocationId"],
        "trackId": invocation["track"],
        "realmId": invocation["realmId"],
        "completionPolicy": invocation["completionPolicy"],
        "taskRevision": invocation["taskRevision"],
        "parentInvocationId": invocation["parentInvocationId"],
        "receiver": invocation["receiver"],
        "closureId": invocation["closureId"],
        "activation": invocation["activation"],
    }
    for key, value_ in expected.items():
        if not _same(prefix.get(key), value_):
            _fail(f"Detailed prefix changed observed invocation field {key}")
    if not isinstance(prefix["maxOutstanding"], int) or not 1 <= prefix["maxOutstanding"] <= 4096:
        _fail("Detailed prefix outstanding-work limit is invalid")
    if prefix.get("sourceDigest") != stable_algorithm_hash(prefix["source"]):
        _fail("Detailed prefix source bytes differ from their digest")

    expected_committed = _committed(prior, invocation)
    expected_realm = _realm_committed(prior, invocation["invocationId"])
    if not _same(prefix["committed"], expected_committed):
        _fail("Detailed committed prefix omits, changes, or adds public rows")
    if not _same(prefix["realmCommitted"], expected_realm):
        _fail("Detailed realm prefix omits, changes, or adds public rows")
    visible_through = max((row["sequence"] for row in expected_realm), default=0)
    if context.get("visibleThrough") != visible_through or visible_through >= target_sequence:
        _fail("Detailed context visibleThrough does not match the exact prior realm prefix")
    if not isinstance(context.get("prefixDigest"), str) or not HASH.fullmatch(
        context["prefixDigest"]
    ):
        _fail("Detailed context requires a lowercase SHA-256 prefix digest")
    if stable_algorithm_hash(prefix) != context["prefixDigest"]:
        _fail("Detailed context prefix digest changed")

    _validate_prefix_rows(
        prefix["committed"], visible_through=visible_through, label="committed prefix"
    )
    _validate_prefix_rows(
        prefix["realmCommitted"], visible_through=visible_through, label="realm prefix"
    )
    state = prefix["selectedState"]
    if (
        not isinstance(state, dict)
        or set(state) != STATE_FIELDS
        or state.get("schema") != CONTINUATION_SCHEMA
    ):
        _fail("Detailed selected state requires the exact semantic-continuation/7 shape")
    for key in ("heap", "locals", "captureAliases", "stack", "continuations", "ready"):
        if not isinstance(state[key], list):
            _fail("Detailed continuation tables must be ordered arrays")
    if expected_state is not None and not _same(state, expected_state):
        _fail("Detailed selected continuation differs from the independent public-prefix replay")
    return True


def validate_detailed_model_context_event(
    item: Mapping[str, Any],
    prior_timeline: Sequence[Mapping[str, Any]],
) -> bool:
    """Validate an observed selected-state event embedded in the timeline."""
    if not isinstance(item, Mapping) or item.get("kind") != "model-context":
        _fail("Expected an observed detailed model-context event")
    fields = {"selectedState", "realmPrefixDigest", "visibleThrough"}
    if not fields <= set(item):
        _fail("Selected model-context event lacks its exact continuation receipt")
    if item.get("visibleThrough") != item.get("sequence", 0) - 1:
        _fail("Selected model-context event changed its prior public cutoff")
    prior = list(prior_timeline)
    if not prior or prior[-1].get("sequence") != item["visibleThrough"]:
        _fail("Selected model-context event requires its immediately preceding checked source read")
    invocation, _, _, _, _ = _source_scope(prior, item.get("invocationId"))
    last = prior[-1]
    if (
        last.get("kind") != "source-read"
        or last.get("track") != invocation["track"]
        or last.get("programId") != invocation["programId"]
        or last.get("digest") != invocation["sourceDigest"]
    ):
        _fail("Selected model-context event must immediately follow its checked source reread")
    relevant = _realm_committed(prior, invocation["invocationId"])
    if stable_algorithm_hash(relevant) != item["realmPrefixDigest"]:
        _fail("Selected model-context event changed its committed realm prefix")
    state = item["selectedState"]
    if (
        not isinstance(state, dict)
        or set(state) != STATE_FIELDS
        or state.get("schema") != CONTINUATION_SCHEMA
    ):
        _fail("Selected model-context event has a malformed v7 continuation")
    random_state = None
    seeds = {row.get("randomSeed") for row in relevant if row.get("kind") == "program"}
    if len(seeds) == 1:
        random_state = next(iter(seeds))
    for row in relevant:
        if row.get("kind") != "semantic":
            continue
        _, fields_ = parse_record(row.get("line"), role="generated")
        if fields_.get("expr") != {"op": "random"}:
            continue
        if type(random_state) is not int or not 0 <= random_state <= 0xFFFFFFFF:
            _fail("Selected model-context random state has no committed source seed")
        random_state = (random_state * 1664525 + 1013904223) & 0xFFFFFFFF
        if fields_.get("result") != ["num", random_state / 4294967296]:
            _fail("Selected model-context random draw differs from deterministic replay")
    if state.get("randomState") != random_state:
        _fail("Selected model-context random state differs from its committed draws")
    return True


def validate_detailed_continuation_v7(
    context_or_prefix: Mapping[str, Any],
    *,
    prior_timeline: Sequence[Mapping[str, Any]] | None = None,
    target: Mapping[str, Any] | int | None = None,
    expected_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate and return a public-only continuation without dispatching.

    A full context plus its exact prior timeline receives the stronger prefix
    and replay checks. A bare translated prefix is accepted only for structural
    restore validation; callers admitting selections must provide history.
    """
    if "prefix" in context_or_prefix:
        if prior_timeline is None or target is None:
            _fail("Continuation restore requires its exact prior timeline and target")
        validate_detailed_context(
            context_or_prefix, prior_timeline, target, expected_state=expected_state
        )
        prefix = context_or_prefix["prefix"]
    else:
        prefix = context_or_prefix
        _exact_fields(prefix, PREFIX_FIELDS, "Detailed continuation prefix")
        if (
            prefix.get("schema") != PREFIX_SCHEMA
            or prefix.get("representation") != REPRESENTATION
            or prefix.get("traceFormat") != TRACE_SCHEMA
            or prefix.get("selectedState", {}).get("schema") != CONTINUATION_SCHEMA
        ):
            _fail("Cannot restore a non-v7 detailed public continuation")
        if expected_state is not None and not _same(prefix["selectedState"], expected_state):
            _fail("Detailed public continuation was mutated after replay")
    restored = copy.deepcopy(prefix["selectedState"])
    return restored
