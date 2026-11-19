"""Public context and restore checks for semantic algorithm execution/7.

The context is part of the model-visible prefix.  This module treats definition
publication as an explicit cache protocol and checks every snapshot against the
prefix already replayed by the native consumer.
"""

from __future__ import annotations

import copy
import hashlib
import math
from collections.abc import Mapping, MutableMapping, Sequence
from typing import Any


ALGORITHM_CONTEXT_SCHEMA = "semantic-context/7"
ALGORITHM_CONTINUATION_SCHEMA = "semantic-continuation/7"


class AlgorithmContextError(ValueError):
    """A public algorithm context or continuation is stale or malformed."""


def algorithm_execution_metadata() -> dict[str, str]:
    """Session metadata shared by prepared examples and model serving."""
    return {"task": "semantic-execution/7", "modality": "text"}


def algorithm_selection_instruction(*, continuation: bool = False) -> str:
    """The one public selection instruction used in training and serving."""
    instruction = (
        "Emit exactly one complete canonical semantic-trace/7 record. Use only "
        "the observed algorithm definition, input graph, and current committed "
        "algorithm context. Check each selected expression against its certified "
        "operation and use only source-authorized effect boundaries. Never predict "
        "observations, private evidence, or future responses."
    )
    if continuation:
        instruction += " Continue the exact current record bytes; end only when the record is complete."
    return instruction


def _check_json_data(value: Any, *, ancestors: set[int] | None = None) -> None:
    """Reject Python-only values before hashing a JSON protocol object."""
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float:
        if not math.isfinite(value) or (value == 0.0 and math.copysign(1.0, value) < 0):
            raise AlgorithmContextError("Context contains a noncanonical JSON number")
        return
    ancestors = set() if ancestors is None else ancestors
    identity = id(value)
    if identity in ancestors:
        raise AlgorithmContextError("Context contains cyclic JSON data")
    if isinstance(value, list):
        ancestors.add(identity)
        try:
            for item in value:
                _check_json_data(item, ancestors=ancestors)
        finally:
            ancestors.remove(identity)
        return
    if isinstance(value, dict) and all(type(key) is str for key in value):
        ancestors.add(identity)
        try:
            for item in value.values():
                _check_json_data(item, ancestors=ancestors)
        finally:
            ancestors.remove(identity)
        return
    raise AlgorithmContextError("Context must contain ordinary JSON data only")


def stable_algorithm_json(value: Any) -> str:
    """Use the native core's JS-compatible canonical JSON implementation."""
    _check_json_data(value)
    from . import algorithm_v7

    serializer = getattr(algorithm_v7, "stable_algorithm_json", None)
    if serializer is None:
        raise AlgorithmContextError("Native algorithm core has no stable JSON serializer")
    try:
        return serializer(value)
    except (TypeError, ValueError) as error:
        raise AlgorithmContextError(str(error)) from error


def stable_algorithm_hash(value: Any) -> str:
    """Hash one JSON value with the same canonical spelling as the JS producer."""
    _check_json_data(value)
    from . import algorithm_v7

    hasher = getattr(algorithm_v7, "stable_algorithm_hash", None)
    if hasher is not None:
        try:
            return hasher(value)
        except (TypeError, ValueError) as error:
            raise AlgorithmContextError(str(error)) from error
    return hashlib.sha256(stable_algorithm_json(value).encode("utf-8")).hexdigest()


def _exact_fields(item: Any, required: set[str], optional: set[str], label: str) -> None:
    if not isinstance(item, Mapping):
        raise AlgorithmContextError(f"{label} must be an object")
    keys = set(item)
    if not required <= keys or keys - required - optional:
        raise AlgorithmContextError(f"{label} has missing or unknown fields")


def _same(left: Any, right: Any) -> bool:
    try:
        return stable_algorithm_json(left) == stable_algorithm_json(right)
    except AlgorithmContextError:
        return False


def validate_algorithm_context(
    item: Mapping[str, Any],
    prior: Sequence[Mapping[str, Any]],
    *,
    program: Mapping[str, Any],
    input_graph: Mapping[str, Any],
    invocation_id: str,
    live_state: Mapping[str, Any],
    definitions: MutableMapping[str, Any],
) -> bool:
    """Validate an exact current-prefix context and update its consumer cache.

    ``prior`` is the controller history: role-tagged generated records and
    observed external records only. Dispatch receipts are intentionally absent.
    Cache entries are keyed by the immutable digest supplied by the producer;
    the first context must publish each definition and later contexts must use
    references only.
    """
    _exact_fields(
        item,
        {
            "schema",
            "representation",
            "definition",
            "input",
            "invocationId",
            "state",
            "visibleThrough",
            "prefixDigest",
        },
        {"program", "inputGraph"},
        "Algorithm context",
    )
    if item.get("schema") != ALGORITHM_CONTEXT_SCHEMA or item.get("representation") != "algorithm":
        raise AlgorithmContextError("Wrong explicit semantic-context/7 representation")
    if type(invocation_id) is not str or not invocation_id or item.get("invocationId") != invocation_id:
        raise AlgorithmContextError("Algorithm context invocation identity changed")
    if type(item.get("visibleThrough")) is not int or item["visibleThrough"] != len(prior):
        raise AlgorithmContextError("Algorithm context cutoff is stale or points into the future")
    if not _same(item.get("state"), live_state):
        raise AlgorithmContextError("Algorithm context state differs from the checked prefix")
    if item.get("prefixDigest") != stable_algorithm_hash(list(prior)):
        raise AlgorithmContextError("Algorithm context prefix digest changed")

    definition_digest = program.get("definitionDigest")
    input_digest = stable_algorithm_hash(input_graph)
    if item.get("definition") != definition_digest or item.get("input") != input_digest:
        raise AlgorithmContextError("Algorithm context refers to another definition or input graph")

    for digest, key, expected, label in (
        (definition_digest, "program", program, "definition"),
        (input_digest, "inputGraph", input_graph, "input graph"),
    ):
        if not isinstance(digest, str) or len(digest) != 64:
            raise AlgorithmContextError(f"Invalid {label} digest")
        cached = definitions.get(digest)
        published = key in item
        if cached is None:
            if not published or not _same(item[key], expected):
                raise AlgorithmContextError(f"First {label} use must publish the exact immutable value")
            definitions[digest] = copy.deepcopy(expected)
        elif not _same(cached, expected):
            raise AlgorithmContextError(f"Consumer {label} cache is stale or tampered")
        elif published:
            raise AlgorithmContextError(f"Cached {label} must be referenced without republishing")
    return True


def restore_algorithm_continuation_v7(packet: Mapping[str, Any]):
    """Restore a checked controller continuation by replaying its prefix only.

    The controller packet contains semantic record history, never a host adapter
    or a dispatch callback.  Replay therefore cannot invoke an external service.
    """
    _exact_fields(
        packet,
        {
            "schema",
            "representation",
            "program",
            "inputGraph",
            "capabilities",
            "invocationId",
            "sessionId",
            "history",
            "state",
            "integrity",
        },
        set(),
        "Algorithm continuation",
    )
    if (
        packet.get("schema") != ALGORITHM_CONTINUATION_SCHEMA
        or packet.get("representation") != "algorithm"
    ):
        raise AlgorithmContextError("Wrong semantic-continuation/7 version")
    body = {key: copy.deepcopy(value) for key, value in packet.items() if key != "integrity"}
    if packet.get("integrity") != stable_algorithm_hash(body):
        raise AlgorithmContextError("Algorithm continuation integrity mismatch")
    from .algorithm_harness_v7 import AlgorithmReplayV7, validate_algorithm_program_v7

    try:
        validate_algorithm_program_v7(packet["program"])
        replay = AlgorithmReplayV7(
            packet["program"],
            packet["inputGraph"],
            capabilities=packet["capabilities"],
            invocation_id=packet["invocationId"],
            session_id=packet["sessionId"],
        )
        if not isinstance(packet["history"], list):
            raise AlgorithmContextError("Continuation history must be a list")
        for row in packet["history"]:
            _exact_fields(row, {"mode", "line"}, set(), "Continuation history row")
            if row["mode"] == "generated":
                replay.commit(row["line"])
            elif row["mode"] == "observed":
                replay.observe(row["line"])
            else:
                raise AlgorithmContextError("Continuation history has an unknown authority")
        if not _same(replay.live_state(), packet["state"]):
            raise AlgorithmContextError("Continuation state differs from its checked prefix")
        return replay
    except AlgorithmContextError:
        raise
    except (TypeError, ValueError, KeyError) as error:
        raise AlgorithmContextError(f"Algorithm continuation replay failed: {error}") from error

