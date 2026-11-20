"""Independent v8 prediction-boundary checks without a future target."""

from __future__ import annotations

import copy

from megatoken.controls import canonical_semantic_observation

from .native_v8 import NativeV8Error, _replay_shared_state, validate_v8_artifact
from .observations_v8 import OBSERVATIONS, validate_observation_v8
from .semantic_contract_v8 import DEFINITION, validate_public
from .semantic_json import stable_algorithm_hash

MAX_PREFIX_BYTES = 32 << 20
MAX_PREFIX_RECORDS = 16384


def validate_v8_committed_prefix(public, contexts):
    """Check exact public observation shapes and replay every committed target."""
    if len(canonical_semantic_observation([public, contexts])) > MAX_PREFIX_BYTES:
        raise NativeV8Error("V8 prediction prefix exceeds its byte budget")
    timeline = validate_public(public)
    if len(timeline) > MAX_PREFIX_RECORDS:
        raise NativeV8Error("V8 prediction prefix exceeds its record budget")
    sessions = set()
    programs = {}
    invocations = set()
    for row in timeline:
        fields = row["fields"]
        sessions.add(fields.get("session"))
        if row["mode"] == "observed":
            if row["kind"] not in OBSERVATIONS:
                raise NativeV8Error("V8 prediction contains an unsupported observation")
            validate_observation_v8(row)
        elif set(fields) != {"session", "track", "invocationId", "line"} | (
            {"entry"} if row["kind"] in {"region", "algorithm"} else set()
        ):
            raise NativeV8Error("V8 prediction has unexpected generated routing fields")
        if row["kind"] == "program":
            identity = fields["programId"]
            if identity in programs:
                raise NativeV8Error("V8 prediction republishes an existing program")
            programs[identity] = fields
        elif row["kind"] == "invocation":
            identity = fields["invocationId"]
            if identity in invocations:
                raise NativeV8Error("V8 prediction repeats an invocation")
            invocations.add(identity)
        elif row["kind"] == "source-read":
            previous = programs.get(fields["programId"])
            if previous is None or any(
                previous[key] != fields[key] for key in ("source", "digest")
            ):
                raise NativeV8Error("V8 source read differs from its published source")
    if len(sessions) != 1 or None in sessions:
        raise NativeV8Error("V8 prediction crosses session or invocation authority")
    # This also checks every preceding context and restored continuation.
    validate_v8_artifact({"public": public, "contexts": contexts})
    return {
        "session": next(iter(sessions)),
        "public": copy.deepcopy(public),
        "contexts": copy.deepcopy(contexts),
        "publicDigest": stable_algorithm_hash(public),
    }


def validate_v8_prediction_prefix(public, contexts, context):
    """Reconstruct the complete next context from already committed public rows.

    No candidate, expected record, source oracle, or private admission evidence is
    accepted. Prior generated rows have their own strictly earlier contexts.
    The return value contains copied public inputs and reproducible identities.
    """
    if len(canonical_semantic_observation([public, contexts, context])) > MAX_PREFIX_BYTES:
        raise NativeV8Error("V8 prediction prefix exceeds its byte budget")
    checked = validate_v8_committed_prefix(public, contexts)
    timeline = public["timeline"]
    if type(context) is not dict or set(context) != set(DEFINITION["contextFields"]):
        raise NativeV8Error("V8 prediction requires the exact public context fields")
    if (
        context["schema"] != DEFINITION["context"]
        or context["representation"] != "mixed"
        or context["profile"] != DEFINITION["profile"]
        or type(context["visibleThrough"]) is not int
        or context["visibleThrough"] != len(timeline)
        or context["prefixDigest"] != stable_algorithm_hash(timeline)
        or context["invocationId"]
        not in {row["fields"]["invocationId"] for row in timeline if row["kind"] == "invocation"}
    ):
        raise NativeV8Error("V8 prediction context differs from the exact committed cutoff")
    boundary = _replay_shared_state(
        timeline, contexts, {}, roundtrip=False, boundary_invocation=context["invocationId"]
    )
    expected = {
        "schema": DEFINITION["context"],
        "representation": "mixed",
        "profile": DEFINITION["profile"],
        "invocationId": context["invocationId"],
        **boundary,
        "visibleThrough": len(timeline),
        "prefixDigest": stable_algorithm_hash(timeline),
    }
    if context != expected:
        differences = sorted(key for key in expected if context[key] != expected[key])
        raise NativeV8Error(
            "V8 prediction context differs from independent replay: " + repr(differences)
        )
    return {
        **checked,
        "context": copy.deepcopy(context),
        "contextDigest": stable_algorithm_hash(context),
    }
