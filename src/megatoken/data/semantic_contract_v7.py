"""Independent v7 execution envelope checks from the shared contract.

The detailed profile uses the same public execution envelope as algorithm/7,
with a role-checked ``detail `` record at generated and observed semantic
boundaries.  This module loads the frozen JSON declaration and verifies its
cross-language digests with the ECMAScript-compatible serializer.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from importlib.resources import files
from typing import Any

from .algorithm_v7 import stable_algorithm_hash
from .detailed_v7 import DetailedV7Error, graph, parse_record, value

CONTRACT = json.loads(
    files("megatoken.data").joinpath("contracts/semantic-execution-v7.json").read_bytes()
)
DEFINITION = CONTRACT["definition"]
SCHEMA_DIGEST = CONTRACT["digest"]
FIXTURE_DIGEST = CONTRACT["fixtureDigest"]


class SemanticContractV7Error(ValueError):
    """A v7 envelope violates the shared semantic execution contract."""


def _fail(message: str) -> None:
    raise SemanticContractV7Error(message)


if (
    stable_algorithm_hash(DEFINITION) != SCHEMA_DIGEST
    or stable_algorithm_hash(CONTRACT["fixtures"]) != FIXTURE_DIGEST
):
    raise SemanticContractV7Error(
        "Shared semantic execution v7 contract or fixtures need regeneration"
    )


ROUTING = re.compile(DEFINITION["profiles"]["detailed"]["identity"]["routing"])
HEX_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
PUBLIC_SCHEMA = DEFINITION["execution"]
TRACE_SCHEMA = DEFINITION["trace"]
CONTEXT_SCHEMA = DEFINITION["context"]
DETAILED_PROFILE = DEFINITION["profiles"]["detailed"]["profile"]
REPRESENTATION = DEFINITION["profiles"]["detailed"]["representation"]
DETAILED = DEFINITION["profiles"]["detailed"]
ENVELOPE = DETAILED["envelope"]
COMMON_FIELDS = set(DETAILED["commonFields"])
CONTINUATION_SCHEMA = DETAILED["continuation"]
STATE_FIELDS = set(DETAILED["modelContext"]["stateFields"])
PUBLIC_FIELDS = set(DETAILED["publicFields"])


def _exact_fields(item: Any, required: set[str], optional: set[str], label: str) -> None:
    if not isinstance(item, Mapping):
        _fail(f"{label} must be a JSON object")
    keys = set(item)
    if not required <= keys or keys - required - optional:
        _fail(f"{label} has missing or undeclared fields")


def _identity(value_: Any, label: str) -> None:
    if not isinstance(value_, str) or not ROUTING.fullmatch(value_):
        _fail(f"Malformed v7 {label}")


def _plain(value_: Any) -> bool:
    return isinstance(value_, dict)


def _string_list(value_: Any, label: str, *, unique: bool = False) -> None:
    if not isinstance(value_, list) or any(not isinstance(item, str) for item in value_):
        _fail(f"{label} must be a string list")
    if unique and len(value_) != len(set(value_)):
        _fail(f"{label} contains duplicates")


def _state(state: Any, sequence: int) -> None:
    if (
        not isinstance(state, dict)
        or set(state) != STATE_FIELDS
        or state.get("schema") != CONTINUATION_SCHEMA
    ):
        _fail("Selected model context requires the exact semantic-continuation/7 state")
    for key in ("heap", "locals", "captureAliases", "stack", "continuations", "ready"):
        if not isinstance(state[key], list):
            _fail("Selected continuation table must remain an ordered array")
    for key in ("heap", "locals", "captureAliases", "continuations"):
        rows = state[key]
        seen: set[str] = set()
        for row in rows:
            if not isinstance(row, list) or len(row) != 2 or not isinstance(row[0], str):
                _fail("Selected continuation table has malformed identity/value rows")
            if row[0] in seen:
                _fail("Selected continuation table has duplicate identities")
            seen.add(row[0])
    for _, tagged in state["locals"]:
        try:
            value(tagged)
        except DetailedV7Error as error:
            raise SemanticContractV7Error(str(error)) from error
    if any(not isinstance(identity, str) for identity in (*state["stack"], *state["ready"])):
        _fail("Selected scheduling identities must be strings")
    random_state = state.get("randomState")
    if random_state is not None and (
        type(random_state) is not int or not 0 <= random_state <= 0xFFFFFFFF
    ):
        _fail("Selected deterministic RNG state must be null or uint32")
    if type(state.get("pending")) not in (dict, type(None)) or type(
        state.get("completion")
    ) not in (dict, type(None)):
        _fail("Selected pending/completion state has the wrong shape")
    if type(state.get("callReturn")) not in (dict, type(None)):
        _fail("Selected call-return state has the wrong shape")


def validate_envelope_record(item: Mapping[str, Any]) -> bool:
    """Validate one public timeline row against the frozen v7 declaration."""
    if not isinstance(item, Mapping) or item.get("kind") not in ENVELOPE:
        _fail("Unknown shared semantic-execution/7 envelope kind")
    kind = item["kind"]
    declaration = ENVELOPE[kind]
    required = COMMON_FIELDS | set(declaration["required"])
    optional = set(declaration["optional"])
    _exact_fields(item, required, optional, f"{kind} envelope")
    if item["schema"] != PUBLIC_SCHEMA or item["mode"] != declaration["mode"]:
        _fail("Envelope version or generated/observed authority differs from contract")
    for key in (
        "session",
        "track",
        "programId",
        "invocationId",
        "realmId",
        "triggerId",
        "subscriptionId",
        "eventId",
        "trackId",
        "ingressKey",
    ):
        if key in item:
            _identity(item[key], key)
    for key in ("sequence", "ownerSequence"):
        if type(item[key]) is not int or item[key] < 1:
            _fail("Envelope ordering requires positive integer counters")

    if kind == "program":
        source = item["source"]
        if (
            not isinstance(source, str)
            or not source
            or stable_algorithm_hash(source) != item["digest"]
            or not isinstance(item["entry"], str)
            or not isinstance(item["sites"], list)
            or not isinstance(item["functions"], list)
            or not isinstance(item["capabilities"], list)
            or len(item["capabilities"]) != len(set(item["capabilities"]))
            or any(
                not isinstance(capability, str) or not capability
                for capability in item["capabilities"]
            )
            or not isinstance(item["dependencies"], list)
            or item["languageProfile"] != "owned-javascript/1"
            or type(item["module"]) is not bool
            or (item["sourceMap"] is not None and not isinstance(item["sourceMap"], dict))
            or (item["sourceMetadata"] is not None and not isinstance(item["sourceMetadata"], dict))
        ):
            _fail("Malformed immutable program publication")
        if "randomSeed" in item and (
            type(item["randomSeed"]) is not int or not 0 <= item["randomSeed"] <= 0xFFFFFFFF
        ):
            _fail("Program deterministic seed must be an unsigned 32-bit integer")
        for dependency in item["dependencies"]:
            if (
                not isinstance(dependency, dict)
                or set(dependency) != {"programId", "digest"}
                or not isinstance(dependency["programId"], str)
                or not ROUTING.fullmatch(dependency["programId"])
                or not isinstance(dependency["digest"], str)
                or not HEX_SHA256.fullmatch(dependency["digest"])
            ):
                _fail("Program dependency must pin an immutable source digest")
    elif kind == "invocation":
        deadline = item["deadline"]
        if deadline is not None and (
            type(deadline) not in (int, float) or not math.isfinite(deadline) or deadline < 0
        ):
            _fail("Invocation deadline must be finite milliseconds or null")
        if item["completionPolicy"] not in DETAILED["completionPolicies"]:
            _fail("Unknown invocation completion policy")
        try:
            value(item["receiver"])
            graph(item["inputGraph"])
        except DetailedV7Error as error:
            raise SemanticContractV7Error(str(error)) from error
        if item["activation"] not in {"call", "module-init", "callback"}:
            _fail("Unknown invocation activation")
        if type(item["taskRevision"]) is not int or item["taskRevision"] < 1:
            _fail("Invocation task revision must be positive")
        if item["parentInvocationId"] is not None:
            _identity(item["parentInvocationId"], "parent invocation")
        if item["closureId"] is not None:
            _identity(item["closureId"], "closure")
        if not isinstance(item["bindings"], dict):
            _fail("Invocation bindings must be an object")
    elif kind == "semantic":
        if item["traceFormat"] != TRACE_SCHEMA:
            _fail("Generated detailed target must use semantic-trace/7")
        try:
            parse_record(item["line"], role="generated")
        except DetailedV7Error as error:
            raise SemanticContractV7Error(str(error)) from error
    elif kind == "result":
        if item["errorOrigin"] not in {None, "adapter-throw", "adapter-rejection"} or item[
            "status"
        ] not in {"fulfilled", "rejected"}:
            _fail("Observed result must preserve declared completion/error origin")
        if item["status"] == "fulfilled" and item["errorOrigin"] is not None:
            _fail("Fulfilled result cannot carry native error provenance")
        try:
            value(item["result"])
            _, fields = parse_record(item["line"], role="observed")
        except DetailedV7Error as error:
            raise SemanticContractV7Error(str(error)) from error
        if fields["result"] != item["result"] or fields["status"] != item["status"]:
            _fail("Observed result row and detail record disagree")
    elif kind == "handler":
        if item["lifetime"] not in {"subscription", "call"}:
            _fail("Handler requires an explicit callback lifetime")
        if type(item["queueLimit"]) is not int or not 1 <= item["queueLimit"] <= 4096:
            _fail("Handler queue limit is outside its declared range")
    elif kind == "dispatch" and "nativeAction" in item:
        native = item["nativeAction"]
        if native is not None and (
            not isinstance(native, dict)
            or native.get("schema") != "semantic-native-action/4"
            or native.get("invocationId") != item["invocationId"]
            or native.get("effectId") != item["effectId"]
        ):
            _fail("Native dispatch descriptor changed action ownership")
    elif kind == "model-context":
        selected_fields = {"selectedState", "realmPrefixDigest", "visibleThrough"}
        present = selected_fields & set(item)
        if present:
            if not selected_fields <= set(item):
                _fail("Selected context requires its state, prefix digest and cutoff together")
            _state(item["selectedState"], item["sequence"])
            if (
                not isinstance(item["realmPrefixDigest"], str)
                or not HEX_SHA256.fullmatch(item["realmPrefixDigest"])
                or item["visibleThrough"] != item["sequence"] - 1
            ):
                _fail("Selected-state context changed its prior public cutoff")
        if not isinstance(item["sourceReads"], list):
            _fail("Model context sourceReads must be a public list")
    elif kind == "handler-close":
        _string_list(item["discarded"], "Discarded callback events", unique=True)
    elif kind == "cancel":
        if (
            not isinstance(item["reason"], str)
            or not item["reason"]
            or type(item["uncertain"]) is not list
        ):
            _fail("Cancellation receipt is malformed")
        _string_list(item["uncertain"], "Uncertain cancellation effects", unique=True)
    elif kind == "reconcile":
        if item["disposition"] not in {"not-dispatched", "lost", "fulfilled", "rejected"}:
            _fail("Unknown reconciliation disposition")
    elif kind == "callback-reattach":
        pass
    elif kind == "source-read":
        if (
            not isinstance(item["source"], str)
            or stable_algorithm_hash(item["source"]) != item["digest"]
        ):
            _fail("Source read does not match its immutable digest")
    return True
