"""Immutable typed child inputs, independent of a selected wire implementation.

V8 observations stay in their original public envelopes. Registering a source
and input grants ingress authority only; it does not issue a source-admission
receipt or execute the supplied program.
"""

from __future__ import annotations

import hashlib
import json

from megatoken.lifecycle import ExecutionBinding


def execution_fields(record):
    return record["fields"] if "fields" in record else record


def validate_execution_context(program, invocation):
    if not isinstance(program, dict) or not isinstance(invocation, dict):
        raise ValueError("Execution context requires observed program and invocation objects")
    v8 = "fields" in program or "fields" in invocation
    if v8:
        from megatoken.data.observations_v8 import validate_observation_v8
        from megatoken.data.semantic_json import stable_algorithm_hash

        validate_observation_v8(program)
        validate_observation_v8(invocation)
        source, incoming = program["fields"], invocation["fields"]
        if (
            program["kind"] != "program"
            or invocation["kind"] != "invocation"
            or program["sequence"] >= invocation["sequence"]
            or source["session"] != incoming["session"]
            or type(source["source"]) is not str
            or source["digest"] != stable_algorithm_hash(source["source"])
            or incoming["sourceDigest"] != source["digest"]
            or incoming["programId"] != source["programId"]
        ):
            raise ValueError("V8 execution context differs from its observed source identity")
        input_ref = stable_algorithm_hash(incoming["inputGraph"])
    else:
        source, incoming = program, invocation
        if (
            source.get("schema") != "semantic-execution/6"
            or source.get("kind") != "program"
            or source.get("mode") != "observed"
            or incoming.get("schema") != "semantic-execution/6"
            or incoming.get("kind") != "invocation"
            or incoming.get("mode") != "observed"
            or not isinstance(source.get("source"), str)
            or hashlib.sha256(source["source"].encode()).hexdigest() != source.get("digest")
            or incoming.get("sourceDigest") != source["digest"]
            or incoming.get("programId") != source.get("programId")
        ):
            raise ValueError("Execution context requires exact observed program and invocation")
        input_ref = hashlib.sha256(
            json.dumps(
                incoming["inputGraph"], sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
        ).hexdigest()
    # Preserve the existing native registry's JSON normalization, including
    # tuple-to-array conversion, before publishing either immutable entry.
    program, invocation = json.loads(
        json.dumps([program, invocation], sort_keys=True, separators=(",", ":"), allow_nan=False)
    )
    return program, invocation, input_ref


def execution_binding(program, invocation, *, completion_target=""):
    program, invocation, input_ref = validate_execution_context(program, invocation)
    source, incoming = execution_fields(program), execution_fields(invocation)
    return ExecutionBinding(
        program_id=source["programId"],
        source_digest=source["digest"],
        entry=incoming["entry"],
        invocation_id=incoming["invocationId"],
        realm_id=incoming["realmId"],
        trigger_id=incoming["triggerId"],
        input_ref=input_ref,
        realm_policy=incoming["realmPolicy"],
        completion_mode=incoming["completionPolicy"],
        capability_grants=tuple(source["capabilities"]),
        completion_target=completion_target,
    )
