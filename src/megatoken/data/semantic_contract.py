"""Independent Python envelope validation from the frozen shared field declaration."""

from __future__ import annotations

import hashlib
import json
import math
import re
from importlib.resources import files

from .host_operations import canonical_json
from .witness_v5 import parse_record, value

CONTRACT = json.loads(
    files("megatoken.data").joinpath("contracts/semantic-harness-v5.json").read_bytes()
)
DEFINITION = CONTRACT["definition"]
SCHEMA_DIGEST = CONTRACT["digest"]
FIXTURE_DIGEST = CONTRACT["fixtureDigest"]
if (
    hashlib.sha256(canonical_json(DEFINITION).encode()).hexdigest() != SCHEMA_DIGEST
    or hashlib.sha256(canonical_json(CONTRACT["fixtures"]).encode()).hexdigest() != FIXTURE_DIGEST
):
    raise ValueError(
        "Shared semantic contract or parity fixtures were changed without regeneration"
    )
ROUTING = re.compile(DEFINITION["identity"]["routing"])


def validate_envelope_record(item):
    if not isinstance(item, dict) or item.get("kind") not in DEFINITION["envelope"]:
        raise ValueError("Unknown shared semantic envelope record kind")
    declaration = DEFINITION["envelope"][item["kind"]]
    required = set(DEFINITION["commonFields"]) | set(declaration["required"])
    if not required <= set(item) <= required | set(declaration["optional"]):
        raise ValueError("Missing or unexpected fields in shared semantic envelope")
    if item["schema"] != DEFINITION["execution"] or item["mode"] != declaration["mode"]:
        raise ValueError("Envelope version or generated/observed authority differs from contract")
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
        if key in item and (not isinstance(item[key], str) or not ROUTING.fullmatch(item[key])):
            raise ValueError("Malformed shared routing identity: " + key)
    for key in ("sequence", "ownerSequence"):
        if type(item[key]) is not int or item[key] < 1:
            raise ValueError("Envelope ordering requires positive integers")
    kind = item["kind"]
    if kind == "program":
        if "randomSeed" in item and (
            type(item["randomSeed"]) is not int or not 0 <= item["randomSeed"] <= 0xFFFFFFFF
        ):
            raise ValueError("Program deterministic seed must be an unsigned 32-bit integer")
        if (
            item["languageProfile"] != "owned-javascript/1"
            or type(item["module"]) is not bool
            or not isinstance(item["dependencies"], list)
        ):
            raise ValueError("Program requires declared language, module and dependency context")
        for dependency in item["dependencies"]:
            if (
                not isinstance(dependency, dict)
                or set(dependency) != {"programId", "digest"}
                or not isinstance(dependency["programId"], str)
                or not ROUTING.fullmatch(dependency["programId"])
                or not isinstance(dependency["digest"], str)
                or not re.fullmatch("[a-f0-9]{64}", dependency["digest"])
            ):
                raise ValueError("Program dependency must pin its immutable source digest")
        for key in ("sourceMap", "sourceMetadata"):
            if item[key] is not None and not isinstance(item[key], dict):
                raise ValueError("Source mapping metadata must be an explicit object or null")
    elif kind == "invocation":
        deadline = item["deadline"]
        if deadline is not None and (
            type(deadline) not in (int, float) or not math.isfinite(deadline) or deadline < 0
        ):
            raise ValueError("Invocation deadline must be finite epoch milliseconds or null")
        if item["completionPolicy"] not in DEFINITION["completionPolicies"]:
            raise ValueError("Unknown invocation completion policy")
        value(item["receiver"])
    elif kind == "semantic":
        if item["traceFormat"] != DEFINITION["trace"]:
            raise ValueError("Incompatible generated witness format")
        parse_record(item["line"])
    elif kind == "result":
        if item["errorOrigin"] not in {None, "adapter-throw", "adapter-rejection"} or item[
            "status"
        ] not in {"fulfilled", "rejected"}:
            raise ValueError("Observed result must preserve declared completion/error origin")
        if item["status"] == "fulfilled" and item["errorOrigin"] is not None:
            raise ValueError("Fulfilled result cannot carry native error provenance")
        value(item["result"])
    elif kind == "handler":
        if item["lifetime"] not in {"subscription", "call"}:
            raise ValueError("Handler requires an explicit callback lifetime")
    elif kind == "model-context":
        fields = {"selectedState", "realmPrefixDigest", "visibleThrough"}
        if fields & set(item):
            state = item.get("selectedState")
            if (
                not fields <= set(item)
                or not isinstance(state, dict)
                or set(state) != set(DEFINITION["modelContext"]["stateFields"])
                or state.get("schema") != DEFINITION["continuation"]
                or any(
                    not isinstance(state.get(key), list)
                    for key in (
                        "heap",
                        "locals",
                        "captureAliases",
                        "stack",
                        "continuations",
                        "ready",
                    )
                )
                or not isinstance(item["realmPrefixDigest"], str)
                or not re.fullmatch("[0-9a-f]{64}", item["realmPrefixDigest"])
                or item["visibleThrough"] != item["sequence"] - 1
            ):
                raise ValueError("Selected context requires exact state and prior public cutoff")
            random_state = state.get("randomState")
            if random_state is not None and (
                type(random_state) is not int or not 0 <= random_state <= 0xFFFFFFFF
            ):
                raise ValueError("Selected-state deterministic RNG value must be null or uint32")
    elif kind == "dispatch" and "nativeAction" in item:
        native = item["nativeAction"]
        if native is not None and (
            not isinstance(native, dict)
            or native.get("schema") != "semantic-native-action/4"
            or native.get("invocationId") != item["invocationId"]
            or native.get("effectId") != item["effectId"]
        ):
            raise ValueError("Native dispatch descriptor must preserve selected action ownership")
    return True
