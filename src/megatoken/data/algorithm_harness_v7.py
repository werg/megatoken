"""Independent native replay and Episode adaptation for algorithm/7 traces.

This consumer accepts only the public algorithm discriminator. It checks every
generated computation, effect boundary, request and external observation in
causal order, then frames only the public prefix into the native Episode path.
The evidence sidecar is checked for integrity and admission, but never becomes a
model-visible record.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from typing import Any

from megatoken.controls import canonical_semantic_observation
from megatoken.data.host_operations import HOST_OPERATIONS, canonical_json
from megatoken.data.schema import (
    Dependency,
    Episode,
    Lineage,
    LogicalAction,
    Meridian,
    PhysicalRecord,
    Supervision,
    TrackEvent,
    TrackSpec,
    digest_bytes,
)
from megatoken.data.semantic_physical import physical_semantic_episode

from .algorithm_context_v7 import (
    AlgorithmContextError,
    algorithm_execution_metadata,
    algorithm_selection_instruction,
    stable_algorithm_hash,
    stable_algorithm_json,
    validate_algorithm_context,
)

IDENTITY = re.compile(r"^[a-z][a-z0-9:_-]*$")
HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
PUBLIC_SCHEMA = "semantic-execution/7"
TRACE_SCHEMA = "semantic-trace/7"
PROGRAM_SCHEMA = "semantic-algorithm-program/1"
PROFILE_SCHEMA = "algorithm-execution/1"


class AlgorithmHarnessError(ValueError):
    """An algorithm/7 artifact cannot be admitted as a native Episode."""


@dataclass(frozen=True)
class AlgorithmHarnessReceipt:
    records: int
    generated: int
    actions: int
    observations: int
    admission: str
    codec_fingerprint: str | None
    completion: dict[str, Any]
    public_digest: str
    context_digest: str


def _fail(message: str) -> None:
    raise AlgorithmHarnessError(message)


def _exact_fields(item: Any, required: set[str], optional: set[str] | None = None, label="record"):
    optional = optional or set()
    if not isinstance(item, Mapping):
        _fail(f"{label} must be an object")
    keys = set(item)
    if not required <= keys or keys - required - optional:
        _fail(f"{label} has missing or unknown fields")


def _identity(value: Any, label="identity") -> str:
    if not isinstance(value, str) or not IDENTITY.fullmatch(value):
        _fail(f"Invalid algorithm {label}")
    return value


def _sha_text(value: str) -> str:
    return stable_algorithm_hash(value)


def _same(left: Any, right: Any) -> bool:
    try:
        return stable_algorithm_json(left) == stable_algorithm_json(right)
    except (AlgorithmContextError, TypeError, ValueError):
        return False


def _core_api():
    from . import algorithm_v7

    return algorithm_v7


def validate_algorithm_program_v7(program: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the immutable source/core definition and every effect grant."""
    _exact_fields(
        program,
        {
            "schema",
            "traceFormat",
            "source",
            "entry",
            "sourceDigest",
            "core",
            "coreDigest",
            "boundaries",
            "profile",
            "definitionDigest",
        },
        label="Algorithm program",
    )
    if program.get("schema") != PROGRAM_SCHEMA or program.get("traceFormat") != TRACE_SCHEMA:
        _fail("Wrong immutable algorithm program version")
    source = program.get("source")
    if not isinstance(source, str) or not source:
        _fail("Algorithm source must be nonempty exact UTF-8 text")
    if not isinstance(program.get("entry"), str) or not program["entry"]:
        _fail("Algorithm entry name is missing")
    if program.get("sourceDigest") != _sha_text(source):
        _fail("Algorithm source digest differs from its exact source bytes")
    core = program.get("core")
    api = _core_api()
    try:
        api.validate_algorithm(core)
    except (TypeError, ValueError, KeyError) as error:
        raise AlgorithmHarnessError(f"Invalid certified algorithm core: {error}") from error
    if program.get("coreDigest") != stable_algorithm_hash(core):
        _fail("Algorithm core digest changed")
    boundaries = program.get("boundaries")
    if not isinstance(boundaries, list):
        _fail("Algorithm effect boundaries must be a list")
    request_steps = [step for step in core["steps"] if step["kind"] == "request"]
    if len(boundaries) != len(request_steps):
        _fail("Source authorization must cover every and only certified request")
    seen_boundaries: set[str] = set()
    for boundary, step in zip(boundaries, request_steps, strict=True):
        _exact_fields(
            boundary,
            {
                "id",
                "request",
                "operation",
                "source",
                "awaited",
                "sourceDigest",
                "capability",
                "observationDomain",
            },
            label="Source effect boundary",
        )
        identity = _identity(boundary.get("id"), "effect boundary")
        if identity in seen_boundaries:
            _fail("Repeated source effect boundary identity")
        seen_boundaries.add(identity)
        source_span = boundary.get("source")
        _exact_fields(source_span, {"start", "end"}, label="Effect source span")
        if (
            type(source_span.get("start")) is not int
            or type(source_span.get("end")) is not int
            or source_span["start"] < 0
            or source_span["end"] <= source_span["start"]
            or boundary.get("request") != step["id"]
            or boundary.get("operation") != step["operation"]
            or boundary.get("sourceDigest") != program["sourceDigest"]
            or boundary.get("capability") != step["operation"]
            or type(boundary.get("awaited")) is not bool
            or boundary.get("observationDomain") != "non-bigint-scalar/1"
        ):
            _fail("Effect authorization differs from its immutable source/core boundary")
        if step["boundary"] != identity or step["operation"] not in HOST_OPERATIONS:
            _fail("Certified request lacks a registered source capability")
    profile = program.get("profile")
    if profile not in {"bounded-pure-algorithm/1", "serial-scalar-effects/1"}:
        _fail("Unsupported algorithm execution profile")
    if (profile == "bounded-pure-algorithm/1") != (not request_steps):
        _fail("Algorithm profile does not match its certified effect boundaries")
    body = {
        key: copy.deepcopy(value) for key, value in program.items() if key != "definitionDigest"
    }
    if program.get("definitionDigest") != stable_algorithm_hash(body):
        _fail("Immutable algorithm definition digest changed")
    return copy.deepcopy(dict(program))


def _validate_input_graph(program: Mapping[str, Any], input_graph: Mapping[str, Any]):
    _exact_fields(input_graph, {"roots", "objects"}, label="Algorithm input graph")
    roots, objects = input_graph.get("roots"), input_graph.get("objects")
    if not isinstance(roots, list) or not isinstance(objects, list) or len(objects) > 256:
        _fail("Algorithm input graph has an invalid shape or exceeds its node budget")
    if len(roots) != len(program["core"]["inputs"]):
        _fail("Algorithm input graph must contain every declared input exactly once")
    seen: set[str] = set()
    for node in objects:
        if not isinstance(node, Mapping):
            _fail("Algorithm graph node must be an object")
        _exact_fields(node, {"id", "kind", "properties"}, {"prototype"}, "Algorithm graph node")
        node_id = _identity(node.get("id"), "heap identity")
        if node_id in seen or node.get("kind") not in {"array", "object"}:
            _fail("Algorithm input graph has a duplicate or unsupported object identity")
        seen.add(node_id)
        if "prototype" in node and node["prototype"] is not None:
            _fail("Algorithm input graph uses an unsupported custom prototype")
        properties = node.get("properties")
        if not isinstance(properties, list):
            _fail("Algorithm graph properties must be an ordered list")
        keys: set[str] = set()
        for prop in properties:
            _exact_fields(prop, {"key", "value"}, {"version"}, "Algorithm graph property")
            if (
                not isinstance(prop.get("key"), str)
                or prop["key"] == "__proto__"
                or prop["key"] in keys
            ):
                _fail("Algorithm graph has a malformed or duplicate data property")
            keys.add(prop["key"])
    try:
        heap_builder = _core_api().algorithm_heap_from_graph
        heap = heap_builder(input_graph)
        # This checks input types, referenced heap identities, dense arrays, and
        # all tagged leaves before replay can trust them.
        _core_api().evaluate_algorithm(program["core"], roots, heap=heap, observations=[])
    except (TypeError, ValueError, KeyError, AttributeError) as error:
        raise AlgorithmHarnessError(f"Invalid algorithm input graph: {error}") from error
    return copy.deepcopy(dict(input_graph)), heap


def _term_references(term: Any, result: set[str] | None = None) -> set[str]:
    result = set() if result is None else result
    if isinstance(term, Mapping):
        if term.get("op") == "ref" and isinstance(term.get("id"), str):
            result.add(term["id"])
        for value in term.values():
            if isinstance(value, (Mapping, list)):
                _term_references(value, result)
    elif isinstance(term, list):
        for value in term:
            if isinstance(value, (Mapping, list)):
                _term_references(value, result)
    return result


def _subterms(term: Mapping[str, Any], result: list[Mapping[str, Any]] | None = None):
    result = [] if result is None else result
    result.append(term)
    for value in term.values():
        if isinstance(value, Mapping) and "op" in value:
            _subterms(value, result)
        elif isinstance(value, list):
            for child in value:
                if isinstance(child, Mapping) and "op" in child:
                    _subterms(child, result)
    return result


def _scalar_to_json(value: list[Any]):
    tag = value[0]
    if tag == "null":
        return None
    if tag == "undefined":
        raise ValueError("undefined is not serializable in canonical JSON arguments")
    if tag == "bool" or tag == "str":
        return value[1]
    if (
        tag == "num"
        and type(value[1]) in (int, float)
        and value[1] not in ("-0", "NaN", "+Infinity", "-Infinity")
    ):
        return value[1]
    raise ValueError("Effect argument requires a codec outside canonical scalar JSON")


def _transport(operation: str, arguments: list[list[Any]]) -> tuple[str, str, dict[str, Any]]:
    spec = HOST_OPERATIONS.get(operation)
    if spec is None:
        _fail("Effect request names an unregistered host operation")
    if len(arguments) != spec["arity"]:
        _fail("Effect request has the wrong argument count")
    if spec["callbackIndices"] or any(value[0] == "ref" for value in arguments):
        _fail("Serial scalar algorithm effects cannot carry callbacks or references")
    if spec["key"] is not None:
        key_index = spec["key"]
        if arguments[key_index][0] != "str":
            _fail("Effect request does not provide its declared string key")
    codec = spec["codec"]
    if codec == "canonical-json/1":
        try:
            payload = canonical_json([_scalar_to_json(value) for value in arguments])
            native_arguments = [_scalar_to_json(value) for value in arguments]
        except (TypeError, ValueError) as error:
            raise AlgorithmHarnessError(str(error)) from error
        action_arguments = {"args": native_arguments}
    elif codec == "owned-graph/1":
        graph = {"schema": "owned-graph/1", "roots": copy.deepcopy(arguments), "objects": []}
        payload = canonical_json(graph)
        action_arguments = {"graph": graph}
    else:
        _fail("Effect request names an unsupported host transport codec")
    if len(payload.encode("utf-8")) > spec["limits"]["payloadBytes"]:
        _fail("Effect request exceeds its registered payload budget")
    return payload, codec, action_arguments


class AlgorithmReplayV7:
    """Checked prefix replay equivalent to the JS AlgorithmController."""

    def __init__(
        self,
        program: Mapping[str, Any],
        input_graph: Mapping[str, Any],
        *,
        capabilities: Sequence[str] | None = None,
        invocation_id: str = "invocation1",
        session_id: str = "algorithm1",
    ):
        self.program = validate_algorithm_program_v7(program)
        self.input_graph, self.heap = _validate_input_graph(self.program, input_graph)
        self.invocation_id = _identity(invocation_id, "invocation identity")
        self.session_id = _identity(session_id, "session identity")
        expected_capabilities = list(
            dict.fromkeys(item["operation"] for item in self.program["boundaries"])
        )
        if capabilities is None:
            capabilities = expected_capabilities
        if (
            not isinstance(capabilities, (list, tuple))
            or any(not isinstance(value, str) for value in capabilities)
            or len(set(capabilities)) != len(capabilities)
            or list(capabilities) != expected_capabilities
        ):
            _fail("Algorithm continuation capabilities differ from its source boundaries")
        self.capabilities = tuple(capabilities)
        self.capability_set = set(capabilities)
        self.inputs = {
            input_row["id"]: copy.deepcopy(value)
            for input_row, value in zip(
                self.program["core"]["inputs"], self.input_graph["roots"], strict=True
            )
        }
        self.values: dict[str, list[Any]] = {}
        self.symbols: dict[str, Mapping[str, Any]] = {}
        self.cursor = 0
        self.pending: dict[str, Any] | None = None
        self.abrupt: list[Any] | None = None
        self.completion: dict[str, Any] | None = None
        self.history: list[dict[str, str]] = []
        self.claims: list[dict[str, Any]] = []
        self.first_violation: dict[str, Any] | None = None
        self.boundaries = {row["id"]: row for row in self.program["boundaries"]}
        self.declared_values = {
            step["id"] for step in self.program["core"]["steps"] if step.get("id")
        }

    def environment(self) -> dict[str, Any]:
        return {"inputs": self.inputs, "values": self.values, "heap": self.heap}

    def next_step(self):
        if self.completion is not None:
            return None
        if self.abrupt is not None:
            return {
                "kind": "throw",
                "value": {"op": "literal", "value": copy.deepcopy(self.abrupt)},
            }
        return (
            self.program["core"]["steps"][self.cursor]
            if self.cursor < len(self.program["core"]["steps"])
            else None
        )

    def expand(self, term: Mapping[str, Any], seen: set[str] | None = None):
        seen = set() if seen is None else seen
        if term.get("op") == "ref" and term.get("id") in self.symbols:
            identity = term["id"]
            if identity in seen:
                _fail("Algorithm selection contains a cyclic checked dependency")
            return self.expand(self.symbols[identity], seen | {identity})
        result = copy.deepcopy(dict(term))
        for key, value in result.items():
            if key == "args" and isinstance(value, list):
                result[key] = [self.expand(child, seen) for child in value]
            elif isinstance(value, Mapping) and "op" in value:
                result[key] = self.expand(value, seen)
        return result

    def _claim(self, record: Mapping[str, Any], expected: Any, consistent: bool):
        claim = {
            "index": len(self.history),
            "kind": record["kind"],
            "consistent": bool(consistent),
            "expected": copy.deepcopy(expected),
        }
        self.claims.append(claim)
        if not consistent and self.first_violation is None:
            self.first_violation = copy.deepcopy(claim)
        return claim

    def _append_history(self, mode: str, line: str):
        self.history.append({"mode": mode, "line": line})

    def commit(self, line: str):
        api = _core_api()
        try:
            record = api.parse_algorithm_record(line, role="generated")
        except (TypeError, ValueError) as error:
            raise AlgorithmHarnessError(
                f"Invalid generated semantic-trace/7 record: {error}"
            ) from error
        if record["kind"] == "observe":
            _fail("External observations cannot be generated by the model")
        step = self.next_step()
        if self.completion is not None or step is None:
            _fail("Algorithm is already complete")
        if self.pending is not None or step["kind"] == "observe":
            _fail("An external observation is required before the next prediction")

        if record["kind"] == "compute":
            identity = record["id"]
            if identity in self.values or identity in self.inputs:
                _fail("Algorithm value identity was reused")
            expected_term = (
                step["term"]
                if step["kind"] == "compute"
                else step["value"]
                if step["kind"] in {"return", "throw"}
                else None
            )
            if expected_term is None:
                _fail("Computation cannot cross a certified effect boundary")
            expanded = self.expand(record["term"])
            expected_expanded = self.expand(expected_term)
            canonical = step["kind"] == "compute" and identity == step["id"]
            if canonical:
                if not _same(expanded, expected_expanded):
                    _fail("Selected expression differs from the certified computation")
            elif identity in self.declared_values or not any(
                _same(expanded, self.expand(term)) for term in _subterms(expected_expanded)
            ):
                _fail("Selected expression is not the certified computation or one of its subterms")
            try:
                expected = api.evaluate_term(record["term"], self.environment())
            except (TypeError, ValueError, KeyError) as error:
                raise AlgorithmHarnessError(
                    f"Cannot evaluate selected algorithm claim: {error}"
                ) from error
            claim = self._claim(record, expected, _same(record["result"], expected))
            self.values[identity] = copy.deepcopy(record["result"])
            self.symbols[identity] = copy.deepcopy(record["term"])
            if canonical:
                self.cursor += 1
        elif record["kind"] == "request":
            if (
                step["kind"] != "request"
                or record["id"] != step["id"]
                or record["boundary"] != step["boundary"]
                or record["operation"] != step["operation"]
            ):
                _fail("Request is not the next certified source effect boundary")
            boundary = self.boundaries.get(record["boundary"])
            if (
                not boundary
                or boundary["request"] != record["id"]
                or boundary["operation"] != record["operation"]
                or boundary["sourceDigest"] != self.program["sourceDigest"]
                or record["operation"] not in self.capability_set
            ):
                _fail("Effect request lacks its exact source and capability authorization")
            if self.first_violation is not None:
                _fail("An earlier false claim cannot authorize an external request")
            try:
                expected_args = [
                    api.evaluate_term(term, self.environment()) for term in step["arguments"]
                ]
            except (TypeError, ValueError, KeyError) as error:
                raise AlgorithmHarnessError(
                    f"Cannot evaluate certified effect arguments: {error}"
                ) from error
            if not _same(record["arguments"], expected_args):
                _fail("Effect request arguments differ from the committed dependencies")
            payload, codec, action_arguments = _transport(record["operation"], record["arguments"])
            dispatch = {
                "id": record["id"],
                "boundary": record["boundary"],
                "operation": record["operation"],
                "arguments": copy.deepcopy(record["arguments"]),
                "payload": payload,
                "codec": codec,
                "payloadDigest": _sha_text(payload),
                "key": self.invocation_id + "/" + record["id"],
                "status": "requested",
            }
            claim = self._claim(record, expected_args, True)
            self.pending = dispatch
            self.cursor += 1
            self.last_dispatch_action_arguments = action_arguments
        elif record["kind"] in {"return", "throw"}:
            if step["kind"] != record["kind"]:
                _fail("Incorrect algorithm completion kind")
            try:
                expected = api.evaluate_term(step["value"], self.environment())
            except (TypeError, ValueError, KeyError) as error:
                raise AlgorithmHarnessError(
                    f"Cannot evaluate algorithm completion: {error}"
                ) from error
            self._claim(record, expected, _same(record["value"], expected))
            self.completion = {"kind": record["kind"], "value": copy.deepcopy(record["value"])}
            self.abrupt = None
            self.cursor = len(self.program["core"]["steps"])
            dispatch = None
        else:
            _fail("Generated record is outside the certified algorithm profile")

        self._append_history("generated", line)
        return {
            "record": record,
            "claim": copy.deepcopy(self.claims[-1]),
            "dispatch": copy.deepcopy(locals().get("dispatch")),
        }

    def observe(self, line: str):
        api = _core_api()
        try:
            record = api.parse_algorithm_record(line, role="observed")
        except (TypeError, ValueError) as error:
            raise AlgorithmHarnessError(
                f"Invalid observed semantic-trace/7 record: {error}"
            ) from error
        step = self.next_step()
        if (
            record["kind"] != "observe"
            or self.pending is None
            or step is None
            or step["kind"] != "observe"
            or record["request"] != self.pending["id"]
            or record["request"] != step["request"]
            or record["id"] != step["id"]
        ):
            _fail("Observation does not match the exact pending request and next input")
        status = record["status"]
        value = ["undefined"] if status == "completed" else copy.deepcopy(record["value"])
        if value[0] == "bigint" and self.program["profile"] == "serial-scalar-effects/1":
            self.completion = {
                "kind": "unsupported",
                "category": "unsupported.algorithm-observation",
                "reason": "BigInt observation requires detailed exceptional execution",
            }
            self.pending = None
            self.cursor += 1
            self._append_history("observed", line)
            return {"status": "unsupported", "value": value}
        if status == "rejected":
            self.abrupt = copy.deepcopy(value)
        else:
            self.values[step["id"]] = copy.deepcopy(value)
        self.pending = None
        self.cursor += 1
        self._append_history("observed", line)
        return {"status": status, "value": value}

    def live_state(self) -> dict[str, Any]:
        needed: set[str] = set()
        for step in self.program["core"]["steps"][self.cursor :]:
            needed |= _term_references(step)
        if self.abrupt is not None:
            needed.clear()
        step = self.next_step()
        if step is not None and step.get("kind") == "compute":
            expected = self.expand(step["term"])
            subterms = _subterms(expected)
            for identity, term in self.symbols.items():
                if any(_same(candidate, self.expand(term)) for candidate in subterms):
                    needed.add(identity)
        changed = True
        while changed:
            changed = False
            for identity in tuple(needed):
                if identity not in self.declared_values and identity in self.symbols:
                    for dependency in _term_references(self.symbols[identity]):
                        if dependency not in needed:
                            needed.add(dependency)
                            changed = True
        return {
            "cursor": self.cursor,
            "values": [
                [identity, copy.deepcopy(value)]
                for identity, value in self.values.items()
                if identity in needed
            ],
            "symbols": [
                [identity, copy.deepcopy(term)]
                for identity, term in self.symbols.items()
                if identity in needed and identity not in self.declared_values
            ],
            "pending": copy.deepcopy(self.pending),
            "abrupt": copy.deepcopy(self.abrupt),
            "completion": copy.deepcopy(self.completion),
            "firstViolation": (
                {"index": self.first_violation["index"], "kind": self.first_violation["kind"]}
                if self.first_violation is not None
                else None
            ),
        }

    def model_context(self, definitions: MutableMapping[str, Any]) -> dict[str, Any]:
        definition = self.program["definitionDigest"]
        input_digest = stable_algorithm_hash(self.input_graph)
        context = {
            "schema": "semantic-context/7",
            "representation": "algorithm",
            "definition": definition,
            "input": input_digest,
            "invocationId": self.invocation_id,
            "state": self.live_state(),
            "visibleThrough": len(self.history),
            "prefixDigest": stable_algorithm_hash(self.history),
        }
        self._add_definition_publication(context, definitions, definition, self.program, "program")
        self._add_definition_publication(
            context, definitions, input_digest, self.input_graph, "inputGraph"
        )
        return context

    def checkpoint(self) -> dict[str, Any]:
        body = {
            "schema": "semantic-continuation/7",
            "representation": "algorithm",
            "program": copy.deepcopy(self.program),
            "inputGraph": copy.deepcopy(self.input_graph),
            "capabilities": list(self.capabilities),
            "invocationId": self.invocation_id,
            "sessionId": self.session_id,
            "history": copy.deepcopy(self.history),
            "state": self.live_state(),
        }
        return {**body, "integrity": stable_algorithm_hash(body)}

    @staticmethod
    def _add_definition_publication(context, definitions, digest, value, field):
        if digest not in definitions:
            definitions[digest] = copy.deepcopy(value)
            context[field] = copy.deepcopy(value)
        elif not _same(definitions[digest], value):
            _fail("Consumer immutable definition cache was altered")


def _validate_sidecar(
    export: Mapping[str, Any],
    public: Mapping[str, Any],
    contexts: list,
    program: Mapping[str, Any],
    replay: AlgorithmReplayV7,
    generated_certificates: list[dict[str, Any]],
    *,
    allow_local: bool,
    admission_verifier: Callable[[Mapping[str, Any]], bool] | None,
):
    sidecar = export.get("sidecar")
    if not isinstance(sidecar, Mapping) or sidecar.get("schema") != "semantic-algorithm-evidence/2":
        _fail("Algorithm evidence sidecar must use semantic-algorithm-evidence/2")
    _exact_fields(
        sidecar,
        {
            "schema",
            "admission",
            "referenceStatus",
            "selectionOrigin",
            "profile",
            "representation",
            "sourceDigest",
            "archiveDigest",
            "definitionDigest",
            "coreDigest",
            "provenance",
            "receipt",
            "certificates",
            "publicDigest",
            "contextDigest",
            "nativeAdmission",
            "optimizerSteps",
            "learningClaim",
            "sourceArtifact",
            "sourceArtifactDigest",
            "sourceEvidence",
            "family",
            "sourceFamily",
            "fixtureDigest",
        },
        {
            "sourceAdmission",
            "nativeAdmissionStatus",
            "splitIdentity",
            "split",
            "sourceComponent",
            "familyRegistryDigest",
            "datasetBinding",
        },
        "Algorithm evidence sidecar",
    )
    admission = sidecar.get("admission")
    if admission not in {"locally_verified", "admitted"}:
        _fail("Algorithm evidence must carry a current source-checked admission status")
    if sidecar.get("profile") != PROFILE_SCHEMA or sidecar.get("representation") != TRACE_SCHEMA:
        _fail("Algorithm evidence profile and representation differ from the public envelope")
    if sidecar.get("referenceStatus") != "agreed":
        _fail("Algorithm evidence needs independently reconciled source checks")
    if admission == "admitted" and not {"sourceAdmission", "nativeAdmissionStatus"} <= set(sidecar):
        _fail("Admitted algorithm evidence needs its source and native admission receipts")
    binding_fields = {
        "splitIdentity",
        "split",
        "sourceComponent",
        "familyRegistryDigest",
        "datasetBinding",
    }
    if set(sidecar) & binding_fields and not binding_fields <= set(sidecar):
        _fail("Bound algorithm data must carry the complete frozen split and family binding")
    if admission != "admitted" and not allow_local:
        _fail("Locally checked algorithm data requires allow_local=True")
    if admission == "admitted" and (
        sidecar.get("referenceStatus") != "agreed"
        or admission_verifier is None
        or not admission_verifier(export)
    ):
        _fail("Admitted algorithm data needs independently verified source/effect evidence")
    if (
        sidecar.get("sourceDigest") != program["sourceDigest"]
        or sidecar.get("coreDigest") != program["coreDigest"]
        or sidecar.get("definitionDigest") != program["definitionDigest"]
    ):
        _fail("Private source evidence refers to a different immutable algorithm definition")
    for key, value in (("publicDigest", public), ("contextDigest", contexts)):
        if sidecar.get(key) != stable_algorithm_hash(value):
            _fail(f"Algorithm evidence {key} differs from public bytes")
    for key in (
        "archiveDigest",
        "sourceDigest",
        "coreDigest",
        "definitionDigest",
        "publicDigest",
        "contextDigest",
    ):
        if not isinstance(sidecar.get(key), str) or not HEX_SHA256.fullmatch(sidecar[key]):
            _fail(f"Algorithm evidence is missing a valid {key}")
    certificates = sidecar.get("certificates")
    if not isinstance(certificates, list) or len(certificates) != len(generated_certificates):
        _fail("Source-certified record receipts differ from the exact checked public prefix")
    for actual, expected in zip(certificates, generated_certificates, strict=True):
        _exact_fields(
            actual,
            {"record", "rule", "digest", "sourceDigest", "definitionDigest"},
            {"journalEvents"},
            "Algorithm record certificate",
        )
        if any(actual[key] != expected[key] for key in expected):
            _fail("Source-certified record receipt changed its exact checked claim")
        if "journalEvents" in actual and (
            expected["rule"] != "certified-effect-boundary/1"
            or not isinstance(actual["journalEvents"], list)
            or len(actual["journalEvents"]) != 2
            or any(not isinstance(event, str) or not event for event in actual["journalEvents"])
        ):
            _fail("Private effect receipt must bind only a request and its observation")
    source_artifact = sidecar.get("sourceArtifact")
    source_artifact_digest = sidecar.get("sourceArtifactDigest")
    source_evidence = sidecar.get("sourceEvidence")
    if (
        not isinstance(source_artifact, Mapping)
        or not isinstance(source_artifact.get("public"), Mapping)
        or not isinstance(source_artifact.get("sidecar"), Mapping)
        or source_artifact["sidecar"].get("schema") != "semantic-supervision/2"
        or not isinstance(source_artifact_digest, str)
        or not HEX_SHA256.fullmatch(source_artifact_digest)
        or source_artifact_digest != stable_algorithm_hash(source_artifact)
    ):
        _fail("Algorithm evidence must retain its digest-checked source artifact")
    _exact_fields(
        source_evidence,
        {"identityDigest", "sourceChecks", "referenceOutcome"},
        label="Algorithm source evidence",
    )
    source_private = source_artifact["sidecar"]
    reference = source_private.get("reference")
    if (
        not isinstance(reference, Mapping)
        or source_evidence.get("identityDigest") != source_private.get("identityDigest")
        or not _same(source_evidence.get("sourceChecks"), reference.get("sourceChecks"))
        or not _same(source_evidence.get("referenceOutcome"), reference.get("outcome"))
        or source_private.get("sourceFamily") != sidecar.get("sourceFamily")
        or sidecar.get("family") != sidecar.get("sourceFamily")
    ):
        _fail("Algorithm source evidence or source family differs from its digest-checked artifact")
    if not isinstance(sidecar.get("fixtureDigest"), str) or not HEX_SHA256.fullmatch(
        sidecar["fixtureDigest"]
    ):
        _fail("Algorithm evidence must bind the exact source fixture")
    return admission, sidecar


def adapt_algorithm_harness_v7(
    export: Mapping[str, Any],
    *,
    codec: Any | None = None,
    max_positions: int = 8192,
    max_history: int = 1024,
    allow_local: bool = False,
    admission_verifier: Callable[[Mapping[str, Any]], bool] | None = None,
) -> tuple[Episode, AlgorithmHarnessReceipt]:
    """Check and frame one complete public algorithm/7 selection history."""
    if not isinstance(export, Mapping):
        _fail("Algorithm import requires a structured export")
    public, contexts = export.get("public"), export.get("contexts")
    if not isinstance(public, Mapping) or not isinstance(contexts, list):
        _fail("Algorithm export needs a public envelope and its public contexts")
    if public.get("representation") != "algorithm":
        _fail("Public v7 envelope requires the explicit representation:'algorithm' discriminator")
    _exact_fields(
        public,
        {
            "schema",
            "traceFormat",
            "contextFormat",
            "representation",
            "profile",
            "timeline",
            "completion",
            "claims",
            "firstViolation",
        },
        label="Public algorithm envelope",
    )
    if (
        public.get("schema") != PUBLIC_SCHEMA
        or public.get("traceFormat") != TRACE_SCHEMA
        or public.get("contextFormat") != "semantic-context/7"
        or public.get("representation") != "algorithm"
        or public.get("profile") != PROFILE_SCHEMA
    ):
        _fail("Wrong explicit semantic-execution/7 algorithm envelope")
    timeline = public.get("timeline")
    if not isinstance(timeline, list) or not 2 <= len(timeline) <= 65536:
        _fail("Missing or over-budget algorithm execution timeline")
    _exact_fields(timeline[0], {"mode", "kind", "program"}, label="Program publication")
    _exact_fields(timeline[1], {"mode", "kind", "graph"}, label="Input graph publication")
    if (
        timeline[0].get("mode") != "observed"
        or timeline[0].get("kind") != "program"
        or timeline[1].get("mode") != "observed"
        or timeline[1].get("kind") != "input"
    ):
        _fail("An immutable program and exact input graph must be observed before selection")
    program = validate_algorithm_program_v7(timeline[0]["program"])
    input_graph = timeline[1]["graph"]
    capabilities = list(dict.fromkeys(item["operation"] for item in program["boundaries"]))
    invocation_ids = {row.get("invocationId") for row in contexts if isinstance(row, Mapping)}
    if len(invocation_ids) != 1:
        _fail("Algorithm contexts must retain exactly one immutable invocation identity")
    invocation_id = _identity(next(iter(invocation_ids)), "invocation identity")
    replay = AlgorithmReplayV7(
        program, input_graph, capabilities=capabilities, invocation_id=invocation_id
    )
    definitions: dict[str, Any] = {}
    context_index = 0
    generated_certificates: list[dict[str, Any]] = []
    records: list[PhysicalRecord] = []
    meridians: list[Meridian] = []
    dependencies: list[Dependency] = []
    actions: list[LogicalAction] = []
    action_by_request: dict[str, LogicalAction] = {}
    last_generated: str | None = None
    observed = 0
    generated = 0
    dispatch_seen: dict[str, dict[str, Any]] = {}
    native_track = "assistant"
    provenance = (
        "teacher"
        if export.get("sidecar", {}).get("admission") == "admitted"
        and export.get("sidecar", {}).get("selectionOrigin") == "owned-interpreter-teacher"
        else "student"
        if export.get("sidecar", {}).get("selectionOrigin") == "model"
        else "unknown"
    )

    def add_record(
        payload: bytes,
        role: str,
        *,
        policy=False,
        record_id: str,
        turn: int,
        event_kind: str,
        event_request: str = "",
        action_id: str | None = None,
        result_of: str | None = None,
        execution_id: str | None = None,
        instruction="",
    ):
        records.append(
            PhysicalRecord(
                record_id,
                role,
                payload,
                turn_index=turn,
                logical_id=record_id,
                policy_mask=policy,
                provenance=provenance if policy else "executed",
                action_id=action_id,
                result_of=result_of,
                execution_id=execution_id,
            )
        )
        meridian = len(meridians)
        meridians.append(
            Meridian(
                meridian,
                (
                    TrackEvent(
                        record_id,
                        native_track,
                        "text",
                        meridian,
                        role,
                        request=event_request,
                        instruction=instruction,
                    ),
                ),
            )
        )

    # Program and graph are native observations, followed by exact causal
    # timeline replay. Private sidecars never enter this record list.
    program_record = "algorithm:v7:program"
    add_record(
        canonical_semantic_observation(timeline[0]),
        "user",
        record_id=program_record,
        turn=0,
        event_kind="program",
    )
    input_record = "algorithm:v7:input"
    add_record(
        canonical_semantic_observation(timeline[1]),
        "environment",
        record_id=input_record,
        turn=1,
        event_kind="input",
    )
    timeline_index = 2
    while timeline_index < len(timeline):
        item = timeline[timeline_index]
        if not isinstance(item, Mapping):
            _fail("Algorithm timeline row must be an object")
        kind, mode = item.get("kind"), item.get("mode")
        if kind == "semantic":
            _exact_fields(item, {"mode", "kind", "line"}, label="Generated algorithm record")
            if mode != "generated" or not isinstance(item.get("line"), str):
                _fail("Only generated canonical algorithm records carry model loss")
            if context_index >= len(contexts):
                _fail("Every model-selected record needs its exact public current-prefix context")
            context = contexts[context_index]
            try:
                validate_algorithm_context(
                    context,
                    replay.history,
                    program=program,
                    input_graph=input_graph,
                    invocation_id=invocation_id,
                    live_state=replay.live_state(),
                    definitions=definitions,
                )
            except (AlgorithmContextError, TypeError, ValueError, KeyError) as error:
                raise AlgorithmHarnessError(
                    f"Invalid algorithm context at record {context_index}: {error}"
                ) from error
            context_id = f"algorithm:v7:context:{context_index}"
            add_record(
                canonical_semantic_observation(context),
                "system",
                record_id=context_id,
                turn=len(records),
                event_kind="model-context",
                instruction="",
            )
            generated_id = f"algorithm:v7:selection:{generated}"
            try:
                result = replay.commit(item["line"])
            except (
                AlgorithmContextError,
                AlgorithmHarnessError,
                TypeError,
                ValueError,
                KeyError,
            ) as error:
                raise AlgorithmHarnessError(
                    f"Algorithm selection failed causal replay: {error}"
                ) from error
            request_id = result["record"].get("id") if result["record"]["kind"] == "request" else ""
            action_id = None
            execution_id = None
            if result["dispatch"] is not None:
                dispatch = result["dispatch"]
                action_id = f"algorithm:{invocation_id}:{dispatch['id']}"
                execution_id = action_id + ":execution"
                spec = HOST_OPERATIONS[dispatch["operation"]]
                action = LogicalAction(
                    action_id=action_id,
                    call_id=action_id + ":call",
                    tool_name=dispatch["operation"],
                    arguments_json=canonical_json(replay.last_dispatch_action_arguments),
                    record_ids=(generated_id,),
                    serialized_sha256=digest_bytes(item["line"].encode("utf-8")),
                    execution_id=execution_id,
                    schema_id=spec["signature"],
                )
                actions.append(action)
                action_by_request[dispatch["id"]] = action
                dispatch_seen[dispatch["id"]] = dispatch
            add_record(
                item["line"].encode("utf-8"),
                "assistant",
                policy=True,
                record_id=generated_id,
                turn=len(records),
                event_kind="semantic",
                event_request=request_id,
                action_id=action_id,
                instruction=algorithm_selection_instruction(),
            )
            dependencies.append(Dependency(context_id, generated_id, "algorithm-context"))
            if last_generated:
                dependencies.append(Dependency(last_generated, generated_id, "algorithm-prefix"))
            last_generated = generated_id
            generated_certificates.append(
                {
                    "record": len(replay.history) - 1,
                    "rule": "certified-effect-boundary/1"
                    if result["record"]["kind"] == "request"
                    else "checked-algorithm-operation/1",
                    "digest": stable_algorithm_hash(item["line"]),
                    "sourceDigest": program["sourceDigest"],
                    "definitionDigest": program["definitionDigest"],
                }
            )
            context_index += 1
            generated += 1
            timeline_index += 1
            continue

        if kind == "dispatch":
            _exact_fields(
                item, {"mode", "kind", "request", "key", "attempt"}, label="Observed dispatch"
            )
            request = item.get("request")
            dispatch = replay.pending
            if (
                mode != "observed"
                or dispatch is None
                or request != dispatch["id"]
                or item.get("key") != dispatch["key"]
                or type(item.get("attempt")) is not int
                or item["attempt"] != 1
                or request in dispatch_seen
                and dispatch_seen[request].get("timelineDispatch")
            ):
                _fail("Dispatch must follow its selected request exactly once with its checked key")
            dispatch_seen[request] = {
                **dispatch_seen.get(request, dispatch),
                "timelineDispatch": True,
            }
            dispatch_id = f"algorithm:v7:dispatch:{request}"
            add_record(
                canonical_semantic_observation(item),
                "environment",
                record_id=dispatch_id,
                turn=len(records),
                event_kind="dispatch",
                event_request=request,
            )
            observed += 1
            timeline_index += 1
            continue

        if kind == "observation":
            _exact_fields(
                item, {"mode", "kind", "line"}, {"reconciliation"}, "Observed algorithm result"
            )
            if mode != "observed" or not isinstance(item.get("line"), str):
                _fail("External algorithm observation must remain an observed public record")
            if "reconciliation" in item and (
                not isinstance(item["reconciliation"], str) or not item["reconciliation"]
            ):
                _fail("Observation reconciliation receipt must be nonempty text")
            try:
                observation = _core_api().parse_algorithm_record(item["line"], role="observed")
                request = observation["request"]
                if request not in dispatch_seen or not dispatch_seen[request].get(
                    "timelineDispatch"
                ):
                    _fail("External observation lacks its prior selected dispatch")
                replay.observe(item["line"])
            except (AlgorithmHarnessError, TypeError, ValueError, KeyError) as error:
                raise AlgorithmHarnessError(
                    f"Invalid external algorithm observation: {error}"
                ) from error
            action = action_by_request.get(request)
            observation_id = f"algorithm:v7:observation:{request}"
            add_record(
                item["line"].encode("utf-8"),
                "environment",
                record_id=observation_id,
                turn=len(records),
                event_kind="observation",
                event_request=request,
                result_of=action.action_id if action else None,
                execution_id=action.execution_id if action else None,
            )
            observed += 1
            timeline_index += 1
            continue

        _fail("Algorithm timeline contains an unknown event family")

    if context_index != len(contexts):
        _fail("Algorithm export contains a future or unused context")
    if replay.pending is not None:
        _fail("Selected algorithm request is missing its dispatched observed result")
    if replay.completion is None or replay.completion.get("kind") not in {"return", "throw"}:
        _fail("Complete supported algorithm execution is required")
    if replay.first_violation is not None:
        _fail("A false algorithm claim cannot be admitted as checked training evidence")
    if not _same(public.get("completion"), replay.completion):
        _fail("Public completion differs from causal replay")
    if (
        not _same(public.get("claims"), replay.claims)
        or public.get("firstViolation") != replay.first_violation
    ):
        _fail("Public claim summary differs from current-prefix checks")
    admission, sidecar = _validate_sidecar(
        export,
        public,
        contexts,
        program,
        replay,
        generated_certificates,
        allow_local=allow_local,
        admission_verifier=admission_verifier,
    )
    if len(action_by_request) != len(dispatch_seen):
        _fail("Every selected external request must have one checked dispatch and result")

    public_digest = stable_algorithm_hash(public)
    context_digest = stable_algorithm_hash(contexts)
    revision = hashlib.sha256(b"\n".join(record.payload for record in records)).hexdigest()
    lineage_family = (
        sidecar.get("family") or sidecar.get("sourceLineage") or program["sourceDigest"]
    )
    if not isinstance(lineage_family, str) or not lineage_family:
        lineage_family = program["sourceDigest"]
    split = sidecar.get("split")
    if split not in {None, "train", "validation", "test", "quarantine"}:
        _fail("Algorithm sidecar has an invalid frozen split")
    ancestors = sidecar.get("ancestorIds", [])
    if not isinstance(ancestors, list) or any(
        not isinstance(value, str) or not value for value in ancestors
    ):
        _fail("Algorithm private lineage has malformed ancestor IDs")
    lineage = Lineage(
        source_id="semantic-algorithm:" + public_digest[:20],
        revision=revision,
        family_id=lineage_family,
        ancestor_ids=tuple(ancestors),
        task_id=sidecar.get("taskId"),
        parent_episode_id=sidecar.get("parentEpisodeId"),
        source_uri="https://github.com/werg/tracing-js-self",
        frozen_split=split,
    )
    supervision = Supervision(
        outcome="success" if admission == "admitted" else "unknown",
        verification="external_tests" if admission == "admitted" else "none",
        annotations_json=json.dumps(
            {
                "session_metadata": algorithm_execution_metadata(),
                "semantic_execution": {
                    "schema": PUBLIC_SCHEMA,
                    "representation": "algorithm",
                    "profile": PROFILE_SCHEMA,
                    "admission": admission,
                    "publicDigest": public_digest,
                    "contextDigest": context_digest,
                },
                "semantic_algorithm": {
                    "sourceDigest": program["sourceDigest"],
                    "coreDigest": program["coreDigest"],
                    "definitionDigest": program["definitionDigest"],
                    "generatedRecords": generated,
                    "observations": observed,
                    "selectionOrigin": sidecar.get("selectionOrigin", "unknown"),
                },
            },
            separators=(",", ":"),
        ),
    )
    episode = Episode(
        "semantic:algorithm:" + public_digest,
        tuple(records),
        lineage,
        stream="semantic_execution",
        actions=tuple(actions),
        supervision=supervision,
        dependencies=tuple(dependencies),
        tracks=(TrackSpec(native_track, "assistant"),),
        meridians=tuple(meridians),
    )
    try:
        episode = physical_semantic_episode(
            episode,
            codec=codec,
            max_positions=max_positions,
            max_history=max_history,
            selection_instruction=algorithm_selection_instruction,
            emit_context_windows=False,
        )
    except ValueError as error:
        raise AlgorithmHarnessError(str(error)) from error
    return episode, AlgorithmHarnessReceipt(
        records=len(episode.records),
        generated=generated,
        actions=len(actions),
        observations=observed,
        admission=admission,
        codec_fingerprint=getattr(codec, "fingerprint", None),
        completion=copy.deepcopy(replay.completion),
        public_digest=public_digest,
        context_digest=context_digest,
    )
