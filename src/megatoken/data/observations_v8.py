"""Validate V8 host observations directly, without manufacturing a legacy envelope."""

from __future__ import annotations

import math
import re

from .detailed_v8 import parse_record
from .semantic_records import value

ROUTING = re.compile(r"[a-z][a-z0-9:_-]*\Z")
DIGEST = re.compile(r"[a-f0-9]{64}\Z")
OBSERVATIONS = {
    "program": (
        {
            "programId",
            "digest",
            "entry",
            "source",
            "sites",
            "functions",
            "capabilities",
            "dependencies",
            "sourceMap",
            "languageProfile",
            "module",
            "sourceMetadata",
            "regionProgram",
        },
        {"randomSeed"},
    ),
    "invocation": (
        {
            "programId",
            "invocationId",
            "realmId",
            "triggerId",
            "completionPolicy",
            "realmPolicy",
            "inputGraph",
            "entry",
            "sourceDigest",
            "receiver",
            "closureId",
            "activation",
            "parentInvocationId",
            "taskRevision",
            "deadline",
            "bindings",
            "ingressKey",
        },
        set(),
    ),
    "dispatch": ({"invocationId", "effectId", "key", "attempt"}, {"nativeAction"}),
    "result": ({"invocationId", "effectId", "status", "result", "line", "errorOrigin"}, set()),
    "handler": (
        {
            "subscriptionId",
            "programId",
            "realmId",
            "trackId",
            "realmPolicy",
            "entry",
            "queueLimit",
            "closureId",
            "lifetime",
            "ownerInvocationId",
        },
        set(),
    ),
    "handler-close": ({"subscriptionId", "discarded"}, set()),
    "cancel": ({"invocationId", "reason", "uncertain"}, set()),
    "realm-close": ({"realmId", "reason"}, set()),
    "reconcile": ({"invocationId", "effectId", "disposition", "evidenceId"}, set()),
    "callback-reattach": ({"subscriptionId", "realmId", "evidenceId"}, set()),
    "event": (
        {"subscriptionId", "eventId", "invocationId", "triggerId", "programId", "realmId"},
        set(),
    ),
    "source-read": ({"programId", "digest", "source"}, set()),
    "source-evict": ({"programId", "digest"}, set()),
}


def validate_observation_v8(row):
    if type(row) is not dict or set(row) != {"sequence", "mode", "kind", "fields"}:
        raise ValueError("V8 observation requires its exact public envelope")
    kind = row["kind"]
    if row["mode"] != "observed" or type(kind) is not str or kind not in OBSERVATIONS:
        raise ValueError("Unsupported V8 observation authority or kind")
    if type(row["sequence"]) is not int or row["sequence"] < 1:
        raise ValueError("V8 observation requires a positive public sequence")
    fields = row["fields"]
    required, optional = OBSERVATIONS[kind]
    required = required | {"session", "track"}
    if type(fields) is not dict or not required <= set(fields) <= required | optional:
        raise ValueError("Missing or unexpected fields in V8 observation")
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
        if key in fields and (type(fields[key]) is not str or not ROUTING.fullmatch(fields[key])):
            raise ValueError("Malformed V8 observation routing identity: " + key)
    if kind == "program":
        if "randomSeed" in fields and (
            type(fields["randomSeed"]) is not int or not 0 <= fields["randomSeed"] <= 0xFFFFFFFF
        ):
            raise ValueError("Program deterministic seed must be an unsigned 32-bit integer")
        if (
            fields["languageProfile"] != "owned-javascript/1"
            or type(fields["module"]) is not bool
            or type(fields["dependencies"]) is not list
        ):
            raise ValueError("Program requires declared language, module and dependency context")
        for dependency in fields["dependencies"]:
            if (
                type(dependency) is not dict
                or set(dependency) != {"programId", "digest"}
                or type(dependency["programId"]) is not str
                or not ROUTING.fullmatch(dependency["programId"])
                or type(dependency["digest"]) is not str
                or not DIGEST.fullmatch(dependency["digest"])
            ):
                raise ValueError("Program dependency must pin its immutable source digest")
        for key in ("sourceMap", "sourceMetadata"):
            if fields[key] is not None and type(fields[key]) is not dict:
                raise ValueError("Source mapping metadata must be an explicit object or null")
    elif kind == "invocation":
        deadline = fields["deadline"]
        if deadline is not None and (
            type(deadline) not in (int, float) or not math.isfinite(deadline) or deadline < 0
        ):
            raise ValueError("Invocation deadline must be finite epoch milliseconds or null")
        if fields["completionPolicy"] not in {
            "call-return",
            "await-returned-promise",
            "owned-quiescence",
        }:
            raise ValueError("Unknown invocation completion policy")
        if (
            fields["realmPolicy"] not in {"fresh", "retained"}
            or fields["activation"] not in {"call", "callback", "module-init"}
            or type(fields["taskRevision"]) is not int
            or fields["taskRevision"] < 1
        ):
            raise ValueError("Invalid invocation lifecycle contract")
        value(fields["receiver"])
    elif kind == "result":
        if fields["errorOrigin"] not in {
            None,
            "adapter-throw",
            "adapter-rejection",
            "guest-callback-throw",
        } or fields["status"] not in {"fulfilled", "rejected"}:
            raise ValueError("Observed result must preserve declared completion/error origin")
        if fields["status"] == "fulfilled" and fields["errorOrigin"] is not None:
            raise ValueError("Fulfilled result cannot carry native error provenance")
        value(fields["result"])
        operation, observed = parse_record(fields["line"], role="observed")
        if (
            operation != "observe"
            or observed["effect"] != fields["effectId"]
            or observed["status"] != fields["status"]
            or observed["result"] != fields["result"]
        ):
            raise ValueError("V8 result envelope differs from its observation")
    elif kind == "handler":
        if (
            fields["lifetime"] not in {"subscription", "call"}
            or fields["realmPolicy"] not in {"retained", "fresh"}
            or type(fields["queueLimit"]) is not int
            or not 1 <= fields["queueLimit"] <= 4096
            or fields["track"] != "host"
        ):
            raise ValueError("Handler requires host authority and its bounded lifecycle policy")
        if fields["closureId"] is not None:
            value(["ref", fields["closureId"]])
        if fields["ownerInvocationId"] is not None and (
            type(fields["ownerInvocationId"]) is not str
            or not ROUTING.fullmatch(fields["ownerInvocationId"])
        ):
            raise ValueError(
                "Handler requires a valid owning invocation or explicit host ownership"
            )
    elif kind == "handler-close":
        if (
            fields["track"] != "host"
            or type(fields["discarded"]) is not list
            or any(
                type(identity) is not str or not ROUTING.fullmatch(identity)
                for identity in fields["discarded"]
            )
            or len(set(fields["discarded"])) != len(fields["discarded"])
        ):
            raise ValueError("Handler close requires host authority and distinct discarded events")
    elif kind in {"reconcile", "callback-reattach"}:
        if type(fields["evidenceId"]) is not str or not fields["evidenceId"]:
            raise ValueError("Recovery receipt requires explicit independent evidence")
        if kind == "reconcile" and fields["disposition"] not in {
            "not-dispatched",
            "lost",
            "fulfilled",
            "rejected",
        }:
            raise ValueError("Unknown external effect reconciliation disposition")
    elif kind in {"cancel", "realm-close"}:
        if type(fields["reason"]) is not str:
            raise ValueError("Lifecycle closure requires an explicit reason")
        if kind == "realm-close" and (fields["track"] != "host" or not fields["reason"]):
            raise ValueError("Realm closure requires host observation authority and a reason")
        if kind == "cancel" and (
            type(fields["uncertain"]) is not list
            or any(
                type(identity) is not str or not ROUTING.fullmatch(identity)
                for identity in fields["uncertain"]
            )
            or len(set(fields["uncertain"])) != len(fields["uncertain"])
        ):
            raise ValueError("Cancellation requires distinct uncertain effect identities")
    elif kind == "dispatch" and fields.get("nativeAction") is not None:
        native = fields["nativeAction"]
        if (
            type(native) is not dict
            or native.get("schema") != "semantic-native-action/4"
            or native.get("invocationId") != fields["invocationId"]
            or native.get("effectId") != fields["effectId"]
        ):
            raise ValueError("Native dispatch descriptor must preserve selected action ownership")
    return True
