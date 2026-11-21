"""Public-prefix checks for semantic execution/8.

This module intentionally distinguishes public replay engineering checks from
source-bound production admission. It never promotes a document whose selected
shared continuation and private source correspondence were not independently
replayed by the full native controller.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping

from megatoken.controls import (
    canonical_semantic_observation,
    semantic_execution_metadata_v8,
    semantic_selection_instruction_v8,
)

from . import semantic_primitives as primitives
from .algorithm_v8 import replay_algorithm_line
from .detailed_v8 import encode_record, parse_record
from .detailed_v8 import value as validate_value
from .protocol_v8 import (
    ProtocolV8Error,
    _array_index,
    _assert_fresh_heap_identity,
    _external_machine,
    _internal_machine,
    apply_protocol_to_shared_state,
    assert_detailed_record_compatible,
    decode_protocol_record,
    validate_completion_detail,
    validate_protocol_state,
)
from .schema import (
    Dependency,
    Episode,
    Lineage,
    Meridian,
    PhysicalRecord,
    Supervision,
    TrackEvent,
    TrackSpec,
    encode_episode,
)
from .semantic_contract_v8 import (
    CONTRACT_DIGEST,
    SemanticContractV8Error,
    validate_public,
)
from .semantic_json import stable_algorithm_hash
from .semantic_physical import physical_semantic_episode
from .semantic_primitives import _truthy as _detail_truthy

_UNARY_OPERATORS = {
    operator: name for name, (arity, operator) in primitives.OPERATIONS.items() if arity == 1
}
_BINARY_OPERATORS = {
    operator: name
    for name, (arity, operator) in primitives.OPERATIONS.items()
    if arity == 2 and name not in {"and", "or", "coalesce"}
}


class NativeV8Error(ValueError):
    pass


def _fail(msg):
    raise NativeV8Error(msg)


def _exact(obj, keys, label):
    if not isinstance(obj, Mapping) or set(obj) != set(keys):
        _fail(f"Malformed v8 {label}")


def validate_v8_artifact(document):
    """Replay public v8 records, causal contexts, and shared continuations.

    The engineering receipt reports replay, context, and continuation checks.
    Masks, tokenizer, and live source correspondence remain untested here;
    this function always returns admitted=False.
    """
    if not isinstance(document, dict) or not isinstance(document.get("public"), dict):
        _fail("Expected semantic-execution/8 public artifact")
    public = document["public"]
    try:
        timeline = validate_public(public)
    except SemanticContractV8Error as e:
        raise NativeV8Error(str(e)) from e
    contexts = document.get("contexts")
    if type(contexts) is not list:
        _fail("V8 contexts must be an ordered public list")
    generated = [(i, row) for i, row in enumerate(timeline) if row["mode"] == "generated"]
    if len(contexts) != len(generated):
        _fail("Every generated v8 target requires one earlier context")
    programs = {}
    generated_count = 0
    algorithm_count = 0
    detail_count = 0
    region_count = 0
    protocol_count = 0
    for context, (_index, row) in zip(contexts, generated, strict=True):
        _exact(
            context,
            {
                "schema",
                "representation",
                "profile",
                "invocationId",
                "programs",
                "input",
                "state",
                "regionCursor",
                "obligations",
                "visibleThrough",
                "prefixDigest",
            },
            "context",
        )
        if (
            context["schema"] != "semantic-context/8"
            or context["representation"] != "mixed"
            or context["profile"] != "compositional-regions/1"
        ):
            _fail("V8 context version")
        cutoff = context["visibleThrough"]
        if (
            type(cutoff) is not int
            or cutoff < 1
            or cutoff > row["sequence"] - 1
            or context["prefixDigest"] != stable_algorithm_hash(timeline[:cutoff])
        ):
            _fail("Context includes future information or has a false prefix digest")
        if context["invocationId"] != row["fields"].get("invocationId"):
            _fail("Context invocation mismatch")
        state = context["state"]
        if type(state) is not dict or set(state) != set(
            __import__("megatoken.data.semantic_contract_v8", fromlist=["DEFINITION"]).DEFINITION[
                "sharedStateFields"
            ]
        ):
            _fail("V8 shared continuation fields")
        for name in (
            "heap",
            "locals",
            "stack",
            "effects",
            "jobs",
            "ready",
            "captureAliases",
            "continuations",
            "frameFunctions",
            "jobErrors",
        ):
            if not isinstance(state[name], list):
                _fail("V8 continuation ordered table")
        if any(
            not isinstance(row, list) or len(row) != 2 or not isinstance(row[0], str)
            for key in ("locals", "captureAliases", "continuations", "frameFunctions")
            for row in state[key]
        ):
            _fail("V8 continuation identity table")
        try:
            value_validator = (
                __import__(
                    "megatoken.data.protocol_v8", fromlist=["validate_protocol_value"]
                ).validate_protocol_value
                if state["protocol"] is not None
                else validate_value
            )
            for _, tagged in state["locals"]:
                value_validator(tagged)
            value_validator(state["receiver"])
        except (TypeError, ValueError) as e:
            raise NativeV8Error(f"Malformed tagged v8 continuation value: {e}") from e
        if state["protocol"] is not None:
            try:
                validate_protocol_state(state["protocol"], state["heap"], state)
            except ProtocolV8Error as e:
                raise NativeV8Error(f"Malformed shared protocol state: {e}") from e
        if (
            not isinstance(context["regionCursor"], dict)
            or set(context["regionCursor"]) != {"active", "consumed"}
            or not isinstance(context["regionCursor"]["consumed"], list)
        ):
            _fail("V8 region cursor")
        if (
            not isinstance(context["obligations"], dict)
            or set(context["obligations"]) != {"intents", "handlers", "callbacks", "closureQueue"}
            or any(not isinstance(v, list) for v in context["obligations"].values())
        ):
            _fail("V8 obligations")
        if (
            not isinstance(context["programs"], list)
            or not isinstance(context["input"], dict)
            or not isinstance(context["obligations"], dict)
        ):
            _fail("V8 context publication/state shape")
        for published in context["programs"]:
            if type(published) is not dict or set(published) != {"id", "program"}:
                _fail("Program publication")
            program = published["program"]
            if type(program) is not dict or program.get("schema") != "semantic-region-program/1":
                _fail("Region program version")
            if (
                program.get("sourceDigest") != stable_algorithm_hash(program.get("source"))
                or program.get("traceFormat") != "semantic-trace/8"
                or program.get("profile") != "compositional-regions/1"
            ):
                _fail("Program source or profile digest")
            if program.get("definitionDigest") != stable_algorithm_hash(
                {k: v for k, v in program.items() if k != "definitionDigest"}
            ):
                _fail("Region definition digest")
            source_rows = [
                r["fields"]
                for r in timeline[:cutoff]
                if r["kind"] == "program" and r["fields"].get("programId") == published["id"]
            ]
            if not source_rows or source_rows[-1].get("regionProgram") != program:
                _fail("Context program was not previously published byte-for-byte")
            programs[published["id"]] = program
        prior_invocations = [
            r["fields"]
            for r in timeline[:cutoff]
            if r["kind"] == "invocation"
            and r["fields"].get("invocationId") == context["invocationId"]
        ]
        if not prior_invocations or context["input"].get("graph") != prior_invocations[-1].get(
            "inputGraph"
        ):
            _fail("Context input graph was not previously observed")
        fields = row["fields"]
        if row["kind"] in ("algorithm", "region", "semantic", "protocol"):
            # A target must follow a program and invocation publication inside
            # the exact model-visible prefix. No importer cache can satisfy it.
            if not context["programs"]:
                _fail("Generated target has no publicly visible region definition")
            program = context["programs"][-1]["program"]
            if row["kind"] == "algorithm":
                core = program.get("algorithm")
                if core is None:
                    _fail("Algorithm target lacks its published core")
                entry = fields.get("entry")
                if (
                    not isinstance(entry, dict)
                    or type(entry.get("visibleThrough")) is not int
                    or entry != region_entry_v8(program, context, "whole")
                ):
                    _fail("Stale, malformed, or false algorithm entry")
                if (
                    any(
                        previous["mode"] == "generated"
                        and previous["fields"].get("invocationId") == context["invocationId"]
                        for previous in timeline[:cutoff]
                    )
                    or state["stack"] != [context["invocationId"]]
                    or any(
                        state[name]
                        for name in ("completion", "pending", "effects", "jobs", "continuations")
                    )
                    or context["input"].get("retained")
                    or (state.get("protocol") or {}).get("sequence")
                ):
                    _fail("Whole algorithm requires the untouched invocation entry")
                graph = context["input"].get("graph")
                if not isinstance(graph, dict):
                    _fail("Algorithm target lacks a public input graph")
                replay_algorithm_line(core, graph, fields.get("line"))
                algorithm_count += 1
            elif row["kind"] == "region":
                _replay_region(program, context, fields.get("line"), fields.get("entry"))
                region_count += 1
            else:
                line = fields.get("line")
                if row["kind"] == "protocol":
                    try:
                        decode_protocol_record(line)
                    except ProtocolV8Error as e:
                        raise NativeV8Error(f"Invalid v8 protocol record: {e}") from e
                    protocol_count += 1
                else:
                    if not isinstance(line, str) or not line.startswith("detail "):
                        _fail("V8 semantic record must carry a checked detailed record")
                    try:
                        kind, detail_fields = parse_record(line)
                    except ValueError as e:
                        raise NativeV8Error(f"Invalid v8 detail record: {e}") from e
                    detail_count += 1
            generated_count += 1
    final_restored = _replay_shared_state(timeline, contexts, programs, roundtrip=True)
    final_uninterrupted = _replay_shared_state(timeline, contexts, programs, roundtrip=False)
    if final_restored != final_uninterrupted:
        _fail("Checkpoint restoration changed the final shared continuation")
    return {
        "schema": "semantic-native-admission/8",
        "contractDigest": CONTRACT_DIGEST,
        "checks": {
            "replay": "passed"
            if generated_count
            and algorithm_count + region_count + detail_count + protocol_count == generated_count
            else "unsupported",
            "contexts": "passed",
            "continuation": "passed",
            "masks": "untested",
            "sourceBinding": "untested",
            "tokenizer": "untested",
        },
        "counts": {
            "contexts": len(contexts),
            "generatedRecords": generated_count,
            "algorithmRecords": algorithm_count,
            "regionRecords": region_count,
            "detailedRecords": detail_count,
            "protocolRecords": protocol_count,
        },
        "admitted": False,
    }


def prepare_native_v8_training(document, *, codec, max_positions=8192, max_history=1024):
    """Build the exact native physical rows, masks, token counts and digest."""
    validate_v8_artifact(document)
    public = document["public"]
    timeline = public["timeline"]
    generated = [(i, row) for i, row in enumerate(timeline) if row["mode"] == "generated"]
    contexts = document["contexts"]
    rows = []
    context_for = {
        row["sequence"]: context for context, (_, row) in zip(contexts, generated, strict=True)
    }
    for sequence, row in enumerate(timeline, 1):
        if row["mode"] == "generated":
            context = context_for[sequence]
            rows.append(
                (
                    f"semantic:v8:context:{sequence}",
                    "system",
                    canonical_semantic_observation(context),
                    False,
                )
            )
            payload = row["fields"]["line"].encode("utf-8")
            rows.append((f"semantic:v8:target:{sequence}", "assistant", payload, True))
        else:
            role = "user" if row["kind"] in {"program", "invocation"} else "environment"
            rows.append(
                (
                    f"semantic:v8:observation:{sequence}",
                    role,
                    canonical_semantic_observation(row),
                    False,
                )
            )
    logical_records = []
    meridians = []
    ids = {}
    for index, (identity, role, payload, masked) in enumerate(rows):
        rec = PhysicalRecord(
            identity,
            role,
            payload,
            turn_index=index,
            logical_id=identity,
            policy_mask=masked,
            provenance="unknown" if masked else "executed",
        )
        logical_records.append(rec)
        ids[identity] = identity
        meridians.append(
            Meridian(
                index,
                (
                    TrackEvent(
                        identity,
                        "assistant",
                        "text",
                        index,
                        role,
                        instruction=semantic_selection_instruction_v8() if masked else "",
                    ),
                ),
            )
        )
    dependencies = []
    for sequence, _ in generated:
        target = f"semantic:v8:target:{timeline[sequence]['sequence']}"
        context = f"semantic:v8:context:{timeline[sequence]['sequence']}"
        dependencies.append(Dependency(context, target, "v8-public-context"))
    digest = stable_algorithm_hash(public)
    episode = Episode(
        "semantic:v8:" + digest[:24],
        tuple(logical_records),
        Lineage("semantic-execution-v8", digest, "semantic-execution-v8"),
        stream="semantic_execution",
        supervision=Supervision(
            annotations_json=json.dumps(
                {
                    "session_metadata": semantic_execution_metadata_v8(),
                    "semantic_execution": {
                        "schema": "semantic-execution/8",
                        "representation": "mixed",
                        "profile": "compositional-regions/1",
                    },
                },
                separators=(",", ":"),
            )
        ),
        dependencies=tuple(dependencies),
        tracks=(TrackSpec("assistant", "assistant"),),
        meridians=tuple(meridians),
    )
    episode = physical_semantic_episode(
        episode,
        codec=codec,
        max_positions=max_positions,
        max_history=max_history,
        selection_instruction=lambda continuation=False: semantic_selection_instruction_v8(
            continuation=continuation
        ),
        emit_context_windows=False,
    )
    encoded = encode_episode(episode, codec)
    token_rows = []
    generated_tokens = observed_tokens = 0
    masked_logicals = set()
    unmasked_targets = set()
    for item in encoded:
        record = item.record
        labels = [
            token if eligible else -100
            for token, eligible in zip(item.token_ids, item.free_mask, strict=True)
        ]
        if record.policy_mask:
            generated_tokens += item.P
            masked_logicals.add(record.logical_id)
        elif record.logical_id.startswith("semantic:v8:target:"):
            unmasked_targets.add(record.logical_id)
        else:
            observed_tokens += item.tokens
        token_rows.append(
            {
                "recordId": record.record_id,
                "logicalId": record.logical_id,
                "partIndex": record.part_index,
                "partCount": record.part_count,
                "role": record.role,
                "policyMask": record.policy_mask,
                "payloadSha256": __import__("hashlib").sha256(record.payload).hexdigest(),
                "tokenIds": list(item.token_ids),
                "labels": labels,
                "codecFingerprint": item.codec_fingerprint,
            }
        )
    native_digest = stable_algorithm_hash(
        {
            "schema": "semantic-native-training/8",
            "codecFingerprint": codec.fingerprint,
            "rows": token_rows,
        }
    )
    expected_targets = {
        f"semantic:v8:target:{row['sequence']}" for row in timeline if row["mode"] == "generated"
    }
    if masked_logicals != expected_targets or unmasked_targets:
        raise NativeV8Error("Native v8 physical policy masks do not match generated targets")
    return episode, {
        "nativeTrainingDigest": native_digest,
        "generatedTokens": generated_tokens,
        "observedTokens": observed_tokens,
        "physicalRecords": len(episode.records),
        "logicalRecords": len(timeline) + len(contexts),
        "contexts": len(contexts),
        "generatedRecords": len(generated),
        "eosTokenId": None,
        "terminalTokens": 0,
        "maskValidation": True,
        "framing": "native-physical-record-boundaries/1",
    }


def region_entry_v8(program, context, region):
    """Construct host-owned entry metadata from current public authority only."""
    if region == "whole":
        if not isinstance(program.get("algorithm"), dict):
            _fail("Algorithm entry requires a published pure core")
        digest = stable_algorithm_hash(program["algorithm"])
    else:
        definition = next((item for item in program["regions"] if item.get("id") == region), None)
        if definition is None:
            _fail("Unknown region definition")
        digest = definition["digest"]
    return {
        "definition": program["definitionDigest"],
        "region": region,
        "regionDigest": digest,
        "visibleThrough": context["visibleThrough"],
        "stateDigest": stable_algorithm_hash(context["state"]),
    }


def _replay_region(program, context, line, entry):
    if not isinstance(line, str) or not line.startswith("region "):
        _fail("V8 region prefix")
    try:
        target = json.loads(line[len("region ") :])
    except json.JSONDecodeError as e:
        raise NativeV8Error("Malformed region JSON") from e
    _exact(target, {"region", "bindings", "decisions", "completion"}, "region target")
    if not isinstance(entry, dict) or set(entry) != {
        "definition",
        "region",
        "regionDigest",
        "visibleThrough",
        "stateDigest",
    }:
        _fail("Trusted region entry fields")
    if (
        entry["definition"] != program["definitionDigest"]
        or entry["stateDigest"] != stable_algorithm_hash(context["state"])
        or entry["visibleThrough"] != context["visibleThrough"]
    ):
        _fail("Stale or false region entry")
    definition = next((r for r in program["regions"] if r.get("id") == target["region"]), None)
    if (
        definition is None
        or entry["region"] != definition["id"]
        or entry["regionDigest"] != definition["digest"]
    ):
        _fail("Unknown region definition")
    local_values = {name: value for name, value in context["state"].get("locals", [])}
    locals_ = {
        name: local_values.get(cell)
        for name, cell in context["state"].get("captureAliases", [])
        if cell in local_values
    }
    locals_.update({name: value for name, value in local_values.items() if name not in locals_})
    values = {}
    bindings = []
    decisions = []
    completion = {"kind": definition["completion"]}

    def term(node):
        op = node.get("op")
        if op == "literal":
            return node["value"]
        if op == "local":
            name = node["name"]
            if name not in values and name not in locals_:
                _fail("Unavailable local at region entry")
            return values.get(name, locals_.get(name))
        if op == "select":
            c = term(node["test"])
            truth = c[0] == "symbol" or _detail_truthy(c)
            selected = "consequent" if truth else "alternate"
            decisions.append({"kind": "select", "selected": selected})
            return term(node[selected])
        if op == "logical":
            left = term(node["left"])
            operator = node["operator"]
            truth = left[0] == "symbol" or _detail_truthy(left)
            take_right = (
                (operator == "&&" and truth)
                or (operator == "||" and not truth)
                or (operator == "??" and left[0] in ("null", "undefined"))
            )
            decisions.append({"kind": operator, "selected": "right" if take_right else "left"})
            return term(node["right"]) if take_right else left
        if op in ("binary", "unary"):
            operator = node["operator"]
            mappings = _UNARY_OPERATORS if op == "unary" else _BINARY_OPERATORS
            operation = mappings.get(operator)
            if operation is None:
                _fail("Unsupported local primitive operator")
            from . import semantic_primitives as primitives

            args = (
                [term(node["arg"])] if op == "unary" else [term(node["left"]), term(node["right"])]
            )
            return primitives._operation(operation, args)
        _fail("Unsupported local term")

    for op in definition["operations"]:
        if op["kind"] == "bind":
            value = term(op["term"])
            values[op["binding"]] = value
            bindings.append([op["binding"], value])
        elif op["kind"] == "return":
            completion = {"kind": "return", "value": term(op["term"])}
        else:
            _fail("Unsupported compact region operation")
    expected = {
        "region": definition["id"],
        "bindings": bindings,
        "decisions": decisions,
        "completion": completion,
    }
    if target != expected:
        _fail("Region target disagrees with independent local replay")
    return target


def _invocation_bindings(function, invocation):
    roots = invocation.get("inputGraph", {}).get("roots", [])
    params = function.get("params", [])
    if not isinstance(params, list) or any(
        name is not None and (not isinstance(name, str) or not name) for name in params
    ):
        _fail("Malformed published parameter bindings")
    # Default/rest/pattern parameters are published as null. Their cells are
    # initialized by later checked records, not by the positional input graph.
    bindings = {
        name: copy.deepcopy(roots[index] if index < len(roots) else ["undefined"])
        for index, name in enumerate(params)
        if name is not None
    }
    declared = invocation.get("bindings", {})
    if not isinstance(declared, dict):
        _fail("Malformed invocation bindings")
    for name, index in declared.items():
        if (
            not isinstance(name, str)
            or not name
            or type(index) is not int
            or not 0 <= index < len(roots)
        ):
            _fail("Invocation binding has no observed argument")
        bindings[name] = copy.deepcopy(roots[index])
    return bindings


def _closure_entry_function(functions, entry, closure):
    """Bind both public spellings to the same immutable source function site."""
    if not isinstance(closure, dict) or closure.get("kind") != "closure":
        _fail("Closure invocation has no published guest closure")

    def resolve(reference):
        matches = [
            function
            for function in functions
            if isinstance(function, dict)
            and isinstance(reference, str)
            and (
                function.get("name") == reference
                or "function:" + str(function.get("start")) == reference
            )
        ]
        if len(matches) != 1 or type(matches[0].get("start")) is not int or matches[0]["start"] < 0:
            _fail("Closure entry must resolve to one immutable function site")
        return matches[0]

    selected = resolve(entry)
    declared = resolve(closure.get("code"))
    if selected["start"] != declared["start"]:
        _fail("Closure entry differs from its immutable function site")
    return selected


def _initial_state(
    program,
    invocation,
    functions=(),
    random_seed=None,
    *,
    retained_state=None,
    invocation_ordinal=0,
):
    graph = invocation.get("inputGraph")
    if not isinstance(graph, dict) or set(graph) != {"roots", "objects"}:
        _fail("Invocation input graph")
    from .semantic_records import graph as validate_graph

    heap = copy.deepcopy(retained_state["heap"]) if retained_state is not None else []
    retained = {identity for identity, _ in heap}
    try:
        validate_graph(graph, retained=retained)
    except ValueError as error:
        raise NativeV8Error("Invalid invocation input graph: " + str(error)) from error
    for node in graph["objects"]:
        if node["id"] in retained:
            _fail("Invocation graph overwrites a retained heap identity")
        row = {
            "kind": node["kind"],
            "prototype": node.get("prototype", "ordinary"),
            "properties": [[p["key"], p["value"]] for p in node["properties"]],
        }
        heap.append([node["id"], row])
    receiver = invocation.get("receiver", ["undefined"])
    if receiver[0] == "ref" and receiver[1] not in dict(heap):
        _fail("Invocation receiver references an unknown identity")
    closure_id = invocation.get("closureId")
    closure = dict(heap).get(closure_id) if closure_id is not None else None
    fn = (
        _closure_entry_function(functions, invocation.get("entry"), closure)
        if closure_id is not None
        else next(
            (
                f
                for f in functions
                if f.get("name") == invocation.get("entry")
                or "function:" + str(f.get("start")) == invocation.get("entry")
            ),
            {},
        )
    )
    locals_ = copy.deepcopy(retained_state["locals"]) if retained_state is not None else []
    aliases = [
        [capture["binding"], capture["cell"]] for capture in (closure or {}).get("captures", [])
    ]
    ordinal = 0
    for name, tagged in _invocation_bindings(fn, invocation).items():
        cell = f"a{invocation_ordinal}b{ordinal}"
        while cell in dict(locals_):
            ordinal += 1
            cell = f"a{invocation_ordinal}b{ordinal}"
        ordinal += 1
        _set_row(locals_, cell, tagged)
        _set_row(aliases, name, cell)
    if any(cell not in dict(locals_) for _, cell in aliases):
        _fail("Closure captures an unavailable retained cell")
    if random_seed is not None and (
        type(random_seed) is not int or not 0 <= random_seed <= 0xFFFFFFFF
    ):
        _fail("Malformed deterministic random seed")
    return {
        "heap": heap,
        "locals": locals_,
        "stack": [invocation["invocationId"]],
        "effects": [],
        "jobs": [],
        "ready": [],
        "randomState": retained_state["randomState"]
        if retained_state is not None and random_seed is not None
        else random_seed,
        "pending": None,
        "lastReturn": None,
        "completion": None,
        "receiver": copy.deepcopy(receiver),
        "captureAliases": aliases,
        "continuations": [],
        "reactionSequence": 0,
        "blockedSync": None,
        "callReturn": None,
        "activeJob": None,
        "scheduler": copy.deepcopy(retained_state["scheduler"])
        if retained_state is not None
        else {"registration": 0, "ready": 0},
        "frameFunctions": [[invocation["invocationId"], invocation["entry"]]],
        "jobErrors": [],
        "protocol": None,
    }


def _replay_shared_state(
    timeline, contexts, programs_from_context, *, roundtrip, boundary_invocation=None
):
    snapshots = {
        row["sequence"]: ctx
        for ctx, (_, row) in zip(
            contexts,
            [(i, r) for i, r in enumerate(timeline) if r["mode"] == "generated"],
            strict=True,
        )
    }
    published = {}
    state = None
    program = None
    dispatched = set()
    settled = set()
    consumed = []
    obligations = {"intents": [], "handlers": [], "callbacks": [], "closureQueue": []}
    task_revisions = {}
    owner_sequences = {}
    track_sequences = {}
    controllers = {}
    active_invocation = None
    completion_policy = "call-return"
    completion_policies = {}
    functions = []
    invocation_inputs = {}
    invocation_metadata = {}
    ingress_keys = set()
    callback_allocator = [0]
    event_allocator = 0
    pending_event = None
    pending_reconciliation = None
    cancelled_invocations = set()
    closed_realms = set()

    def share_realm(owner, source):
        realm = invocation_metadata[owner]["realmId"]
        for identity, saved in controllers.items():
            if invocation_metadata[identity]["realmId"] == realm:
                _merge_callback_state(saved, source)

    def same_realm(first, second):
        return (
            first is not None
            and second is not None
            and invocation_metadata[first]["realmId"] == invocation_metadata[second]["realmId"]
        )

    for prior in timeline:
        track = prior.get("fields", {}).get("track")
        track_sequences[track] = track_sequences.get(track, 0) + 1
        owner_sequences[prior["sequence"]] = track_sequences[track]
    for row in timeline:
        kind = row["kind"]
        fields = row["fields"]
        if row["mode"] == "observed":
            from .observations_v8 import validate_observation_v8

            try:
                validate_observation_v8(row)
            except ValueError as error:
                raise NativeV8Error(str(error)) from error
        if pending_event is not None and (
            kind != "invocation" or fields.get("invocationId") != pending_event["invocationId"]
        ):
            _fail("Event delivery must be followed by its exact callback invocation")
        if pending_reconciliation is not None:
            if kind != "result" or any(
                fields.get(key) != value for key, value in pending_reconciliation.items()
            ):
                _fail("Reconciliation must be followed by its exact observed result")
            pending_reconciliation = None
        previous_readiness = {
            invocation_metadata[identity]["realmId"]: saved["scheduler"]["ready"]
            for identity, saved in controllers.items()
        }
        if kind == "program":
            program = fields.get("regionProgram")
            if not isinstance(program, dict):
                _fail("Program record omits its immutable region definition")
            if fields.get("digest") != program.get("sourceDigest") or program.get(
                "source"
            ) != fields.get("source"):
                _fail("Public region definition differs from immutable source publication")
            random_seed = fields.get("randomSeed")
            if "randomSeed" in fields and (
                type(random_seed) is not int or not 0 <= random_seed <= 0xFFFFFFFF
            ):
                _fail("Malformed deterministic random seed")
            published[fields["programId"]] = (program, fields.get("functions", []), random_seed)
            continue
        if kind == "invocation":
            if fields["realmId"] in closed_realms:
                _fail("Closed realm cannot be resurrected")
            if fields["invocationId"] in controllers:
                _fail("Duplicate invocation controller")
            if fields["ingressKey"] in ingress_keys:
                _fail("Ingress key already owns another invocation")
            published_row = published.get(fields["programId"])
            if published_row is None:
                _fail("Invocation has no preceding program")
            program, functions, random_seed = published_row
            if fields.get("sourceDigest") != program.get("sourceDigest"):
                _fail("Invocation source/definition binding")
            retained_owner = next(
                (
                    identity
                    for identity in reversed(controllers)
                    if invocation_metadata[identity]["realmId"] == fields["realmId"]
                ),
                None,
            )
            base = controllers.get(retained_owner)
            if base is not None:
                if fields.get("realmPolicy") != "retained":
                    _fail("Fresh realm identity cannot be reused")
                previous = invocation_metadata[retained_owner]
                if (
                    previous["sourceDigest"] != program.get("sourceDigest")
                    or published[previous["programId"]][2] != random_seed
                ):
                    _fail("Retained invocation differs from its realm source or seed")
                if fields["taskRevision"] < previous["taskRevision"]:
                    _fail("Retained invocation has a stale task revision")
                if any(
                    saved["stack"]
                    for identity, saved in controllers.items()
                    if invocation_metadata[identity]["realmId"] == fields["realmId"]
                    and identity not in cancelled_invocations
                ):
                    _fail("Invocation cannot interrupt a running JavaScript stack")
            owner = fields.get("parentInvocationId")
            if owner is not None and (owner not in controllers or owner in cancelled_invocations):
                _fail("Child has no live owning invocation")
            if fields.get("closureId") is not None:
                # Resolve before importing inputs or allocating captured cells.
                # The observed entry cannot redirect a retained closure's code.
                _closure_entry_function(
                    functions,
                    fields.get("entry"),
                    _table(base, "heap").get(fields["closureId"]) if base is not None else None,
                )
            event = pending_event
            if event is not None:
                handler = dict(obligations["handlers"])[event["subscriptionId"]]
                realm_owner = next(
                    (
                        identity
                        for identity in reversed(controllers)
                        if invocation_metadata[identity]["realmId"] == handler["realmId"]
                    ),
                    None,
                )
                expected = {
                    "activation": "callback",
                    "programId": event["programId"],
                    "realmId": event["realmId"],
                    "track": event["track"],
                    "triggerId": event["triggerId"],
                    "realmPolicy": handler["realmPolicy"],
                    "entry": handler["entry"],
                    "closureId": handler["closureId"],
                    "parentInvocationId": handler["ownerInvocationId"],
                    "taskRevision": invocation_metadata[realm_owner]["taskRevision"]
                    if realm_owner is not None
                    else 1,
                    "receiver": ["undefined"],
                    "bindings": {},
                    "deadline": None,
                    "ingressKey": event["triggerId"],
                }
                if any(fields.get(key) != value for key, value in expected.items()):
                    _fail("Callback invocation differs from its observed event delivery")
                pending_event = None
            if fields.get("activation") == "callback":
                if event is None and (owner is None or base is None):
                    _fail("Callback has no independently replayed owner")
                if (
                    event is None
                    and owner is not None
                    and any(
                        fields[key] != invocation_metadata[owner][key]
                        for key in ("realmId", "programId", "track")
                    )
                ):
                    _fail("Callback differs from its retained owner's realm or program")
            elif fields.get("activation") != "call":
                _fail("Unsupported invocation activation in v8 shared replay")
            state = _initial_state(
                program,
                fields,
                functions,
                random_seed,
                retained_state=base,
                invocation_ordinal=len(controllers),
            )
            invocation_inputs[fields["invocationId"]] = {
                "graph": copy.deepcopy(fields["inputGraph"])
            }
            if base is not None and base["heap"]:
                invocation_inputs[fields["invocationId"]]["retained"] = [
                    identity for identity, _ in base["heap"]
                ]
            active_invocation = fields["invocationId"]
            invocation_metadata[active_invocation] = copy.deepcopy(fields)
            ingress_keys.add(fields["ingressKey"])
            if active_invocation.startswith("callbackinvoke"):
                ordinal = active_invocation.removeprefix("callbackinvoke")
                if not ordinal.isdecimal() or int(ordinal) <= callback_allocator[0]:
                    _fail("Invalid callback invocation allocation")
                callback_allocator[0] = int(ordinal)
            controllers[active_invocation] = state
            completion_policy = fields.get("completionPolicy", "call-return")
            completion_policies[active_invocation] = completion_policy
            share_realm(active_invocation, state)
            task_revisions[fields["invocationId"]] = fields.get("taskRevision")
            continue
        if row["mode"] == "generated":
            invocation = fields.get("invocationId")
            if invocation not in controllers:
                _fail("Generated record has no independently replayed invocation")
            if invocation in cancelled_invocations:
                _fail("Generated record follows invocation cancellation")
            if (
                invocation != active_invocation
                and same_realm(invocation, active_invocation)
                and active_invocation not in cancelled_invocations
                and state is not None
                and state["stack"]
            ):
                _fail("Generated record interrupts another invocation's JavaScript stack")
            active_invocation = invocation
            state = controllers[invocation]
            if state["completion"] is not None:
                _fail("Generated record follows invocation completion")
            completion_policy = completion_policies[invocation]
        if kind in {"cancel", "realm-close"}:
            if kind == "cancel":
                owner = fields["invocationId"]
                if owner not in controllers or owner in cancelled_invocations:
                    _fail("Cancellation requires one live invocation")
                if fields["track"] != invocation_metadata[owner]["track"]:
                    _fail("Cancellation belongs to another invocation track")
                intents = [
                    intent
                    for _, intent in obligations["intents"]
                    if intent["invocationId"] == owner
                ]
                uncertain = [
                    intent["effectId"]
                    for intent in intents
                    if intent["status"] in {"dispatched", "uncertain"}
                ]
                if fields["uncertain"] != uncertain:
                    _fail("Cancellation uncertain effects differ from committed dispatches")
                for intent in intents:
                    if intent["status"] in {"dispatched", "uncertain"}:
                        intent["status"] = "uncertain"
                    elif intent["status"] == "committed":
                        intent["status"] = "cancelled"
                cancelled_invocations.add(owner)
            else:
                realm = fields["realmId"]
                members = {
                    owner
                    for owner, metadata in invocation_metadata.items()
                    if metadata["realmId"] == realm
                }
                if not members or realm in closed_realms:
                    _fail("Realm closure requires one known open realm")
                if not members <= cancelled_invocations:
                    _fail("Realm closure precedes invocation cancellation")
                if any(
                    handler["state"] != "closed"
                    for _, handler in obligations["handlers"]
                    if handler["realmId"] == realm
                ):
                    _fail("Realm closure precedes handler closure")
                closed_realms.add(realm)
            continue
        if row["mode"] == "generated":
            context = snapshots.get(row["sequence"])
            if context is None or context["visibleThrough"] != row["sequence"] - 1:
                _fail("Generated prefix cutoff is not immediately causal")
            if state != context["state"]:
                differing = [
                    name
                    for name in context["state"]
                    if state.get(name) != context["state"].get(name)
                ]
                _fail(
                    "Public continuation differs from independently replayed shared state at "
                    + str(row["sequence"])
                    + " fields "
                    + repr(differing)
                )
            expected_cursor = {"active": None, "consumed": consumed}
            if context["regionCursor"] != expected_cursor:
                _fail("Region cursor differs from independently replayed region history")
            if context["obligations"] != obligations:
                _fail("Obligations differ from independently replayed request history")
            if roundtrip:
                state = _restore_native_checkpoint(
                    _native_checkpoint(timeline[: context["visibleThrough"]], context),
                    expected_prefix=timeline[: context["visibleThrough"]],
                    expected_state=context["state"],
                    expected_cursor=expected_cursor,
                    expected_obligations=obligations,
                )
            program, functions, _ = published[invocation_metadata[active_invocation]["programId"]]
            if kind == "region":
                frame = state["stack"][-1] if state["stack"] else fields["invocationId"]
                target = _replay_region(program, context, fields["line"], fields["entry"])
                _apply_region(state, target)
                consumed.append(
                    {
                        "invocationId": fields["invocationId"],
                        "frame": frame,
                        "region": target["region"],
                    }
                )
            elif kind == "algorithm":
                frame = state["stack"][-1] if state["stack"] else fields["invocationId"]
                _apply_algorithm(
                    state, program["algorithm"], context["input"]["graph"], fields["line"]
                )
                consumed.append(
                    {"invocationId": fields["invocationId"], "frame": frame, "region": "whole"}
                )
            elif kind == "protocol":
                protocol_record = decode_protocol_record(fields["line"])
                if protocol_record["kind"] == "promise.join.step":
                    _assert_realm_scheduler_turn(
                        {
                            identity: saved
                            for identity, saved in controllers.items()
                            if same_realm(identity, active_invocation)
                            and identity not in cancelled_invocations
                        },
                        active_invocation,
                        protocol_record["job"],
                    )
                if state.get("protocol") is None and (
                    len(state["stack"]) != 1
                    or (
                        (state["jobs"] or state["ready"])
                        and protocol_record["kind"]
                        not in ("promise.join", "promise.executor.enter")
                        and not any(
                            node.get("kind") == "promise" and node.get("external", False) is None
                            for _, node in state["heap"]
                        )
                    )
                ):
                    _fail("The first protocol profile requires a checked invocation turn")
                try:
                    state = apply_protocol_to_shared_state(state, fields["line"])
                except ProtocolV8Error as e:
                    raise NativeV8Error(
                        f"Invalid independently replayed protocol transition: {e}"
                    ) from e
            elif kind == "semantic":
                k, payload = parse_record(fields["line"])
                if k == "resume" or k == "job" and payload["action"] == "dispatch":
                    _assert_realm_scheduler_turn(
                        {
                            identity: saved
                            for identity, saved in controllers.items()
                            if same_realm(identity, active_invocation)
                            and identity not in cancelled_invocations
                        },
                        active_invocation,
                        payload["continuation"] if k == "resume" else payload["job"],
                    )
                if state.get("protocol") is not None:
                    try:
                        assert_detailed_record_compatible(state, k, payload)
                    except ProtocolV8Error as e:
                        raise NativeV8Error(f"Detailed record bypasses protocol state: {e}") from e
                _apply_detail(
                    state,
                    k,
                    payload,
                    row,
                    obligations,
                    program,
                    task_revisions,
                    functions=functions,
                    root=active_invocation,
                    completion_policy=completion_policy,
                    invocation_metadata=invocation_metadata[active_invocation],
                    callback_allocator=callback_allocator,
                )
        elif kind in ("dispatch", "result", "reconcile"):
            target = controllers.get(fields.get("invocationId"))
            if target is None:
                _fail("Observation has no independently replayed invocation state")
            if fields["track"] != invocation_metadata[fields["invocationId"]]["track"]:
                _fail("Effect observation belongs to another invocation track")
            if kind == "dispatch" and (
                fields["invocationId"] in cancelled_invocations or target["completion"] is not None
            ):
                _fail("Cannot dispatch a closed invocation")
            _apply_observation(target, row, dispatched, settled, obligations, owner_sequences)
            share_realm(fields["invocationId"], target)
            if kind == "reconcile" and fields["disposition"] in ("fulfilled", "rejected"):
                pending_reconciliation = {
                    "invocationId": fields["invocationId"],
                    "effectId": fields["effectId"],
                    "track": fields["track"],
                    "status": fields["disposition"],
                }
        elif kind in ("handler", "handler-close", "event", "callback-reattach"):
            if fields.get("realmId") in closed_realms:
                _fail("Callback observation targets a closed realm")
            target = controllers.get(fields.get("ownerInvocationId"), state)
            if kind == "handler":
                handler_program = published.get(fields.get("programId"))
                if handler_program is None or not (
                    fields["entry"] == handler_program[0].get("entry")
                    or any(
                        function.get("name") == fields["entry"]
                        or "function:" + str(function.get("start")) == fields["entry"]
                        for function in handler_program[1]
                    )
                ):
                    _fail(
                        "Handler differs from its retained realm source or immutable function site"
                    )
                owner = fields.get("ownerInvocationId")
                if owner is not None and (
                    owner not in controllers
                    or owner in cancelled_invocations
                    or any(
                        invocation_metadata[owner][key] != fields[target_key]
                        for key, target_key in (
                            ("programId", "programId"),
                            ("realmId", "realmId"),
                            ("track", "trackId"),
                        )
                    )
                ):
                    _fail("Handler differs from its retained realm source or owner")
                if fields.get("closureId") is not None and fields.get("realmPolicy") != "retained":
                    _fail("Closure handler requires its retained realm")
            if kind == "handler" and fields.get("closureId") is not None:
                owner = fields.get("ownerInvocationId")
                if owner is None:
                    owner = next(
                        (
                            identity
                            for identity in reversed(controllers)
                            if invocation_metadata[identity]["realmId"] == fields["realmId"]
                        ),
                        None,
                    )
                    target = controllers.get(owner)
                handler_program = published.get(fields.get("programId"))
                if (
                    owner not in controllers
                    or handler_program is None
                    or invocation_metadata[owner]["realmId"] != fields.get("realmId")
                    or invocation_metadata[owner]["sourceDigest"]
                    != handler_program[0].get("sourceDigest")
                ):
                    _fail("Handler closure differs from its retained realm source")
                _closure_entry_function(
                    handler_program[1],
                    fields.get("entry"),
                    _table(target, "heap").get(fields["closureId"]),
                )
            if kind == "event":
                handler = dict(obligations["handlers"]).get(fields["subscriptionId"])
                if handler is None:
                    _fail("Event has no active source handler")
                if handler["ownerInvocationId"] in cancelled_invocations:
                    _fail("Event belongs to a cancelled owning invocation")
                expected_realm = (
                    handler["realmId"]
                    if handler["realmPolicy"] == "retained"
                    else handler["realmId"] + fields["invocationId"]
                )
                if (
                    fields["programId"] != handler["programId"]
                    or fields["track"] != handler["trackId"]
                    or fields["realmId"] != expected_realm
                    or fields["triggerId"]
                    != "event:" + fields["subscriptionId"] + ":" + fields["eventId"]
                ):
                    _fail("Observed event differs from its registered handler routing")
                if any(
                    saved["stack"]
                    for identity, saved in controllers.items()
                    if invocation_metadata[identity]["realmId"] == expected_realm
                    and identity not in cancelled_invocations
                ):
                    _fail("Event cannot interrupt a running JavaScript stack")
                next_event = event_allocator + 1
                while "event" + str(next_event) in controllers:
                    next_event += 1
                if next_event > 9007199254740991 or fields["invocationId"] != "event" + str(
                    next_event
                ):
                    _fail("Invalid event invocation allocation")
            _apply_observation(target, row, dispatched, settled, obligations, owner_sequences)
            if kind == "event":
                event_allocator = next_event
                pending_event = copy.deepcopy(fields)
        if active_invocation is not None:
            controllers[active_invocation] = state
            share_realm(active_invocation, state)
        for realm, previous_ready in previous_readiness.items():
            _refresh_realm_readiness(
                {
                    identity: saved
                    for identity, saved in controllers.items()
                    if invocation_metadata[identity]["realmId"] == realm
                },
                previous_ready,
            )
    if boundary_invocation is None:
        return state
    if pending_event is not None:
        _fail("Prediction boundary precedes the delivered event's callback invocation")
    if pending_reconciliation is not None:
        _fail("Prediction boundary precedes the reconciled effect's observed result")
    if boundary_invocation in cancelled_invocations:
        _fail("Prediction boundary belongs to a cancelled invocation")
    boundary_state = controllers.get(boundary_invocation)
    if (
        boundary_state is not None
        and boundary_invocation != active_invocation
        and same_realm(boundary_invocation, active_invocation)
        and active_invocation not in cancelled_invocations
        and state is not None
        and state["stack"]
    ):
        _fail("V8 prediction boundary interrupts another invocation's JavaScript stack")
    if boundary_state is None or boundary_invocation not in invocation_inputs:
        _fail("V8 prediction boundary has no independently replayed invocation")
    return {
        "programs": [{"id": identity, "program": item[0]} for identity, item in published.items()],
        "input": invocation_inputs[boundary_invocation],
        "state": boundary_state,
        "regionCursor": {"active": None, "consumed": consumed},
        "obligations": obligations,
    }


def _merge_callback_state(target, source):
    """Carry heap/cells and scheduler state between callback and owner frames."""
    if not isinstance(target, dict) or not isinstance(source, dict):
        return
    for field in ("heap", "locals", "scheduler", "randomState"):
        if field in source:
            target[field] = copy.deepcopy(source[field])


def _controller_owner(state):
    """The first retained frame identity names this controller's invocation.

    Request source frames can be nested helpers or reaction jobs. Their host
    intents and observations still belong to the enclosing controller.
    """
    frames = state.get("frameFunctions", [])
    if not frames:
        _fail("Controller has no owning invocation")
    return frames[0][0]


def _refresh_realm_readiness(controllers, previous_ready):
    """Publish a transition's newly eligible waiters in realm registration order.

    The local transition helpers may already have published their own waiters.
    Stage that publication again alongside every retained controller. Previously
    queued work and explicit microtask enqueues keep their original order.
    """
    if not controllers:
        return
    schedulers = [state["scheduler"] for state in controllers.values()]
    if any(scheduler != schedulers[0] for scheduler in schedulers):
        _fail("Realm scheduler counters differ across invocation controllers")
    registration = schedulers[0]["registration"]
    counter = previous_ready
    candidates, reset, sequences = [], [], set()
    for owner, state in controllers.items():
        heap = dict(state["heap"])
        for table, waiting, ready, field in (
            (state["jobs"], "waiting", "queued", "status"),
            (state["continuations"], "waiting", "ready", "state"),
        ):
            for identity, item in table:
                sequence = item.get("sequence")
                if sequence is not None:
                    if (
                        type(sequence) is not int
                        or not 0 < sequence <= registration
                        or sequence in sequences
                    ):
                        _fail("Realm waiter registration is not unique")
                    sequences.add(sequence)
                if field == "status" and item.get("owner") != owner:
                    _fail("Realm job belongs to another invocation")
                newly_published = (
                    item[field] == ready
                    and sequence is not None
                    and item.get("readySequence", 0) > previous_ready
                )
                if newly_published:
                    reset.append((state, identity, item, field, waiting))
                elif item[field] == ready:
                    counter = max(counter, item.get("readySequence", 0))
                if item[field] == waiting or newly_published:
                    promise_id = item.get("promise")
                    if promise_id is None:
                        continue
                    promise = heap.get(promise_id)
                    if promise is None or promise.get("kind") != "promise":
                        _fail("Realm waiter references an unknown Promise")
                    if promise["state"] != "pending":
                        if sequence is None:
                            _fail("Realm Promise waiter lacks its registration order")
                        candidates.append((sequence, state, identity, item, field, ready))
    # Resolve all validation above before changing any queue or publication.
    for state, identity, item, field, waiting in reset:
        state["ready"][:] = [queued for queued in state["ready"] if queued != identity]
        item[field] = waiting
        item.pop("readySequence", None)
    for _, state, identity, item, field, ready in sorted(candidates, key=lambda row: row[0]):
        counter += 1
        item[field] = ready
        item["readySequence"] = counter
        state["ready"].append(identity)
    for state in controllers.values():
        state["scheduler"] = {"registration": registration, "ready": counter}


def _assert_realm_scheduler_turn(controllers, owner, identity):
    heads = []
    for current, state in controllers.items():
        if current != owner and state.get("activeJob") is not None:
            _fail("Scheduler turn interrupts an unfinished callback job")
        if not state["ready"]:
            continue
        head = state["ready"][0]
        item = dict(state["continuations"]).get(head) or dict(state["jobs"]).get(head)
        if item is None or type(item.get("readySequence")) is not int:
            _fail("Realm ready queue has no published readiness")
        heads.append((item["readySequence"], current, head))
    if not heads or min(heads)[1:] != (owner, identity):
        _fail("Scheduler turn violates realm-wide FIFO readiness")


def _native_checkpoint(prefix, context):
    checkpoint = {
        "schema": "semantic-native-checkpoint/8",
        "profile": "compositional-regions/1",
        "visibleThrough": context["visibleThrough"],
        "prefix": copy.deepcopy(prefix),
        "state": copy.deepcopy(context["state"]),
        "regionCursor": copy.deepcopy(context["regionCursor"]),
        "obligations": copy.deepcopy(context["obligations"]),
    }
    checkpoint["integrity"] = stable_algorithm_hash(checkpoint)
    return json.loads(json.dumps(checkpoint, separators=(",", ":"), ensure_ascii=False))


def _restore_native_checkpoint(
    checkpoint, *, expected_prefix, expected_state, expected_cursor=None, expected_obligations=None
):
    required = {
        "schema",
        "profile",
        "visibleThrough",
        "prefix",
        "state",
        "regionCursor",
        "obligations",
        "integrity",
    }
    if not isinstance(checkpoint, dict) or set(checkpoint) != required:
        _fail("Native v8 checkpoint fields")
    content = {k: v for k, v in checkpoint.items() if k != "integrity"}
    if checkpoint["integrity"] != stable_algorithm_hash(content):
        _fail("Native v8 checkpoint integrity")
    if (
        checkpoint["schema"] != "semantic-native-checkpoint/8"
        or checkpoint["profile"] != "compositional-regions/1"
    ):
        _fail("Native v8 checkpoint version")
    if type(checkpoint["visibleThrough"]) is not int or checkpoint["visibleThrough"] != len(
        checkpoint["prefix"]
    ):
        _fail("Native v8 checkpoint cutoff")
    if checkpoint["prefix"] != expected_prefix or checkpoint["state"] != expected_state:
        _fail("Native v8 checkpoint differs from verified replay boundary")
    if expected_cursor is not None and checkpoint["regionCursor"] != expected_cursor:
        _fail("Native v8 checkpoint region cursor differs from verified replay")
    if expected_obligations is not None and checkpoint["obligations"] != expected_obligations:
        _fail("Native v8 checkpoint obligations differ from verified replay")
    expected = set(
        __import__("megatoken.data.semantic_contract_v8", fromlist=["DEFINITION"]).DEFINITION[
            "sharedStateFields"
        ]
    )
    if not isinstance(checkpoint["state"], dict) or set(checkpoint["state"]) != expected:
        _fail("Native v8 checkpoint dropped shared state")
    return copy.deepcopy(checkpoint["state"])


def _set_row(rows, key, value):
    for row in rows:
        if row[0] == key:
            row[1] = value
            return
    rows.append([key, value])


def _table(state, field):
    return {key: value for key, value in state[field]}


def _next_random_state(state):
    seed = state.get("randomState")
    if type(seed) is not int or not 0 <= seed <= 0xFFFFFFFF:
        _fail("Random transition needs a committed deterministic seed")
    return (seed * 1664525 + 1013904223) & 0xFFFFFFFF


def _eval_detail_expression(expr, state):
    from . import semantic_primitives as primitives

    op = expr.get("op")
    if op == "literal":
        return copy.deepcopy(expr["value"])
    if op == "local":
        locals_ = _table(state, "locals")
        alias = dict(state["captureAliases"]).get(expr["name"], expr["name"])
        if alias not in locals_:
            _fail("Expression reads unavailable local")
        return copy.deepcopy(locals_[alias])
    if op == "read":
        heap = _table(state, "heap").get(expr["object"])
        if heap is None:
            _fail("Expression reads absent object")
        prop = dict(heap["properties"])
        return copy.deepcopy(prop.get(expr["key"], ["undefined"]))
    if op == "random":
        return ["num", _next_random_state(state) / 4294967296]
    if op == "unary":
        arg = _eval_detail_expression(expr["arg"], state)
        mapping = _UNARY_OPERATORS
        if expr["operator"] not in mapping:
            _fail("Unsupported unary operator")
        return primitives._operation(mapping[expr["operator"]], [arg])
    if op == "binary":
        left = _eval_detail_expression(expr["left"], state)
        right = _eval_detail_expression(expr["right"], state)
        mapping = _BINARY_OPERATORS
        if expr["operator"] not in mapping:
            _fail("Unsupported binary operator")
        return primitives._operation(mapping[expr["operator"]], [left, right])
    if op in ("select", "and", "or", "coalesce"):
        args = expr.get("args", [])
        if not args:
            _fail("Malformed lazy expression")
        left = _eval_detail_expression(args[0], state)
        if op == "select":
            return _eval_detail_expression(args[1 if _detail_truthy(left) else 2], state)
        take = (
            _detail_truthy(left)
            if op == "and"
            else not _detail_truthy(left)
            if op == "or"
            else left[0] in ("null", "undefined")
        )
        return _eval_detail_expression(args[1], state) if take else left
    _fail("Unsupported detailed expression")


def _apply_region(state, target):
    completion = target["completion"]
    if completion["kind"] == "return" and not state.get("stack"):
        _fail("Compact return requires an active invocation")
    if state.get("protocol") is not None:
        try:
            for name, value in target["bindings"]:
                assert_detailed_record_compatible(state, "bind", {"name": name, "value": value})
            if completion["kind"] == "return":
                assert_detailed_record_compatible(
                    state,
                    "return",
                    {"invocation": state["stack"][-1], "value": completion["value"]},
                )
        except ProtocolV8Error as error:
            raise NativeV8Error(str(error)) from error
    aliases = dict(state["captureAliases"])
    for name, value in target["bindings"]:
        _set_row(state["locals"], aliases.get(name, name), value)
    if completion["kind"] == "return":
        _apply_detail(
            state,
            "return",
            {"invocation": state["stack"][-1], "value": completion["value"]},
            {},
        )


def _assert_fresh_allocation(state, identity):
    try:
        _assert_fresh_heap_identity(state, identity)
    except ProtocolV8Error as error:
        raise NativeV8Error(str(error)) from error


def _apply_algorithm(state, core, graph, line):
    expected = json.loads(line[10:])
    identities = set()
    for allocation in expected["allocations"]:
        _assert_fresh_allocation(state, allocation["id"])
        if allocation["id"] in identities:
            _fail("Duplicate algorithm allocation identity")
        identities.add(allocation["id"])
    for allocation in expected["allocations"]:
        _set_row(
            state["heap"],
            allocation["id"],
            {
                "kind": allocation["kind"],
                "code": None,
                "captures": [],
                "prototype": "ordinary",
                "properties": [[p["key"], p["value"]] for p in allocation["properties"]],
            },
        )
    state["pending"] = {
        "kind": expected["completion"]["kind"],
        "value": expected["completion"]["value"],
    }
    state["stack"] = []


def _host_graph_heap(state, arguments):
    """Expose only independently selected plain data to the host codec checker."""
    owned = _table(state, "heap")
    heap = {}

    def visit(value, depth=0):
        if value[0] != "ref" or value[1] in heap:
            return
        if depth > 64:
            _fail("Owned host graph exceeds its depth limit")
        identity = value[1]
        node = owned.get(identity)
        if node is None:
            _fail("Owned host graph references an unknown selected identity")
        if node["kind"] == "closure":
            heap[identity] = {"id": identity, "kind": "closure", "properties": []}
            return
        if node["kind"] not in ("object", "array"):
            _fail("Owned host graph requires plain data or declared callbacks")
        allowed_prototypes = (
            ("ordinary", "Array.prototype")
            if node["kind"] == "array"
            else (None, "ordinary", "Object.prototype")
        )
        if node.get("prototype", "ordinary") not in allowed_prototypes or any(
            descriptor.get("kind") != "data" for _, descriptor in node.get("descriptors", [])
        ):
            _fail("Owned host graph cannot materialize custom prototypes or accessors")
        properties = node["properties"]
        if any(not isinstance(key, str) for key, _ in properties):
            _fail("Owned host graph cannot materialize symbol properties")
        snapshot = {
            "id": identity,
            "kind": node["kind"],
            "properties": [{"key": key, "value": copy.deepcopy(tag)} for key, tag in properties],
        }
        if node.get("prototype", "ordinary") is None:
            snapshot["prototype"] = None
        heap[identity] = snapshot
        for _, tag in properties:
            visit(tag, depth + 1)

    for argument in arguments:
        visit(argument)
    return heap


def _apply_detail(
    state,
    kind,
    fields,
    row,
    obligations=None,
    program=None,
    task_revisions=None,
    *,
    functions=(),
    root=None,
    completion_policy="call-return",
    invocation_metadata=None,
    callback_allocator=None,
):
    if kind in ("promise", "job", "await", "resume"):
        from .promise_scheduler_v8 import apply_detail

        machine = _internal_machine(state)
        if root is None:
            root = next(iter(machine["frameFunctions"]), None)
        try:
            apply_detail(machine, kind, fields, root=root, functions=functions)
        except ProtocolV8Error as error:
            raise NativeV8Error(str(error)) from error
        state.clear()
        state.update(_external_machine(machine))
        return
    if kind == "bind":
        alias = dict(state["captureAliases"]).get(fields["name"])
        _set_row(state["locals"], alias or fields["name"], fields["value"])
    elif kind == "allocate":
        _assert_fresh_allocation(state, fields["name"])
        if (
            state.get("protocol") is not None
            and len(fields["properties"]) + ("length" in fields) > 4096
        ):
            _fail("Detailed allocation exceeds own property bound")
        node = {
            "kind": fields["kind"],
            "code": None,
            "captures": [],
            "prototype": "ordinary",
            "properties": [[p["key"], p["value"]] for p in fields["properties"]],
        }
        if "code" in fields:
            node["code"] = fields["code"]
        if "captures" in fields:
            aliases = dict(state["captureAliases"])
            node["captures"] = [
                {**capture, "cell": aliases.get(capture["cell"], capture["cell"])}
                for capture in fields["captures"]
            ]
        if node["kind"] == "closure" and any(
            capture["cell"] not in dict(state["locals"]) for capture in node["captures"]
        ):
            _fail("Closure references a cell not yet committed")
        if "length" in fields:
            node["properties"].append(["length", ["num", fields["length"]]])
        _set_row(state["heap"], fields["name"], node)
    elif kind == "read":
        node = _table(state, "heap").get(fields["object"])
        if node is None:
            _fail("Read from absent object")
        props = dict(node["properties"])
        if (fields["key"] in props) != fields["present"] or props.get(
            fields["key"], ["undefined"]
        ) != fields["value"]:
            _fail("Read contradicts committed public heap")
    elif kind == "write":
        heap = _table(state, "heap")
        node = heap.get(fields["object"])
        if node is None:
            _fail("Write to absent object")
        props = dict(node["properties"])
        if node["kind"] == "array" and fields["key"] == "length":
            value = fields["value"]
            if (
                not fields["present"]
                or value[0] != "num"
                or type(value[1]) is not int
                or not 0 <= value[1] <= 4096
            ):
                _fail("Invalid array length mutation")
            props = {
                key: item
                for key, item in props.items()
                if not ((index := _array_index(key)) is not None and index >= value[1])
            }
        if fields["present"]:
            if (
                state.get("protocol") is not None
                and fields["key"] not in props
                and len(props) >= 4096
            ):
                _fail("Detailed write exceeds own property bound")
            props[fields["key"]] = fields["value"]
        else:
            props.pop(fields["key"], None)
        if (
            node["kind"] == "array"
            and fields["present"]
            and (index := _array_index(fields["key"])) is not None
            and index >= props["length"][1]
        ):
            props["length"] = ["num", index + 1]
        node["properties"] = [[k, v] for k, v in props.items()]
    elif kind == "call":
        state["stack"].append(fields["child"])
        state["frameFunctions"].append([fields["child"], fields["callee"]])
    elif kind in ("compute", "choose"):
        actual = _eval_detail_expression(fields["expr"], state)
        claimed = fields["result"] if kind == "compute" else fields["decision"]
        if actual != claimed:
            _fail("Detailed expression claim differs from independent replay")
        if kind == "compute" and fields.get("bind") is not None:
            alias = dict(state["captureAliases"]).get(fields["bind"])
            _set_row(state["locals"], alias or fields["bind"], claimed)
        elif kind == "choose":
            truth = _detail_truthy(claimed)
            path = fields["path"]
            if path in ("then", "else") and (path == "then") != truth:
                _fail("Branch decision path is inconsistent")
            if path in ("use-left", "evaluate-right"):
                op = fields["expr"].get("op")
                if op in ("and", "or", "coalesce"):
                    args = fields["expr"].get("args", [])
                    if len(args) != 2:
                        _fail("Malformed lazy expression")
                    left = _eval_detail_expression(args[0], state)
                    use_right = (
                        _detail_truthy(left)
                        if op == "and"
                        else not _detail_truthy(left)
                        if op == "or"
                        else left[0] in ("null", "undefined")
                    )
                    if (path == "evaluate-right") != use_right:
                        _fail("Logical decision path is inconsistent")
                # A detailed witness can publish only the already-evaluated
                # left decision (literal/local/binary), without its logical
                # operator. The checked value remains public; the separate
                # live source-admission gate checks its source path.
        if fields["expr"].get("op") == "random":
            state["randomState"] = _next_random_state(state)
    elif kind in ("return", "throw"):
        value = fields["value"]
        invocation = fields.get(
            "invocation", row.get("invocationId", state["stack"][-1] if state["stack"] else None)
        )
        if not state["stack"] or state["stack"][-1] != invocation:
            _fail("Completion invocation mismatch")
        if len(state["stack"]) == 1 and state.get("activeJob"):
            job = _table(state, "jobs").get(state["activeJob"])
            if not job or job["status"] != "running":
                _fail("Callback completion requires a running job")
            job["returned"] = {"kind": kind, "value": copy.deepcopy(value)}
            state["stack"].pop()
        elif len(state["stack"]) > 1:
            state["lastReturn"] = {"kind": kind, "value": copy.deepcopy(value)}
            state["stack"].pop()
        else:
            state["pending"] = {"kind": kind, "value": copy.deepcopy(value)}
            state["stack"] = []
    elif kind == "transfer":
        try:
            validate_completion_detail(state, kind, fields)
        except ProtocolV8Error as error:
            raise NativeV8Error(str(error)) from error
        completion_kind = fields["kind"]
        if completion_kind not in ("return", "throw", "cleanup"):
            _fail("Unknown completion transfer")
        state["pending"] = {
            "kind": completion_kind,
            "value": fields["value"],
            "handler": fields["handler"],
        }
    elif kind == "end":
        terminal = state["pending"]
        if completion_policy == "owned-quiescence" and state.get("jobErrors"):
            terminal = {"kind": "throw", "value": state["jobErrors"][0]["value"]}
        if not terminal or terminal["kind"] != fields["kind"]:
            _fail("End does not match selected completion")
        if (
            state["stack"]
            or state["ready"]
            or any(job["status"] != "completed" for _, job in state["jobs"])
            or any(cont["state"] != "resumed" for _, cont in state["continuations"])
            or any(effect["status"] == "requested" for _, effect in state["effects"])
        ):
            _fail("End has outstanding owned work")
        completion = copy.deepcopy(terminal)
        if completion_policy == "await-returned-promise" and terminal["value"][0] == "ref":
            node = _table(state, "heap").get(terminal["value"][1])
            if node and node.get("kind") == "promise":
                if node["state"] == "pending":
                    _fail("Returned Promise remains pending")
                completion = {
                    "kind": "return" if node["state"] == "fulfilled" else "throw",
                    "value": copy.deepcopy(node["result"]),
                }
        state["completion"] = completion
        state["pending"] = None
    elif kind == "request":
        owner = _controller_owner(state)
        if root is not None and root != owner:
            _fail("Request belongs to another invocation controller")
        if not state["stack"] or fields.get("invocation") != state["stack"][-1]:
            _fail("Request belongs to another invocation frame")
        effect = fields["effect"]
        promise = fields.get("promise")
        args = fields["arguments"]
        from . import semantic_primitives as primitives
        from .host_operations import HOST_OPERATIONS, materialize_request

        callback_slots = {"demo.withCallback": {1}, "demo.listen": {1}}
        descriptor = HOST_OPERATIONS.get(fields["operation"], {})
        argument_codec = descriptor.get("codec", "canonical-json/1")
        if argument_codec == "owned-graph/1":
            try:
                decoded, _ = materialize_request(
                    fields, heap=_host_graph_heap(state, args), require_promise=True
                )
            except ValueError as error:
                raise NativeV8Error("Invalid owned host graph request: " + str(error)) from error
        else:
            decoded = []
            for index, value in enumerate(args):
                if index in callback_slots.get(fields["operation"], set()):
                    if value[0] != "ref":
                        _fail("Host callback argument must be an owned closure")
                    callback = _table(state, "heap").get(value[1])
                    if callback is None or callback.get("kind") != "closure":
                        _fail("Host callback identity is not a guest function")
                    decoded.append({"$ownedCallback": value[1]})
                else:
                    item = primitives._decode_scalar(value)
                    if isinstance(item, tuple) and item and item[0] == "__algorithm_ref__":
                        _fail("Object request argument requires owned graph codec")
                    if item is primitives.UNDEFINED:
                        _fail("Undefined host JSON value is not admitted")
                    if type(item) is float and item.is_integer():
                        item = int(item)
                    if type(item) is int and value[0] == "bigint":
                        _fail("BigInt host JSON value is not admitted")
                    decoded.append(item)
            try:
                delivered = json.loads(fields["payload"])
            except (TypeError, json.JSONDecodeError) as e:
                raise NativeV8Error("Malformed host request payload") from e
            if delivered != decoded:
                _fail("Host request payload differs from its tagged arguments")
        if promise:
            _assert_fresh_allocation(state, promise)
        row_effect = {
            "operation": fields["operation"],
            "invocation": fields["invocation"],
            "site": fields["site"],
            "payload": fields["payload"],
            "arguments": decoded,
            "argumentCodec": argument_codec,
            "status": "requested",
            "fixture": None,
            "promise": promise,
        }
        state["effects"].append([effect, row_effect])
        if promise:
            _set_row(
                state["heap"],
                promise,
                {
                    "kind": "promise",
                    "properties": [],
                    "state": "pending",
                    "result": None,
                    "adopted": None,
                    "external": effect,
                    "reactions": [],
                },
            )
        if obligations is not None:
            identity = owner + "/" + effect
            request_fields = {
                "effect": effect,
                "operation": fields["operation"],
                "arguments": args,
                "invocation": fields["invocation"],
                "site": fields["site"],
                "payload": fields["payload"],
            }
            if promise is not None:
                request_fields["promise"] = promise
            witness = encode_record("request", request_fields, role="generated")
            if row.get("fields", {}).get("line", witness) != witness:
                _fail("Request witness differs from its canonical V8 detailed record")
            intents = {key: value for key, value in obligations["intents"]}
            if identity in intents:
                _fail("Duplicate effect intent")
            intents[identity] = {
                "status": "committed",
                "operation": fields["operation"],
                "payload": fields["payload"],
                "invocationId": owner,
                "effectId": effect,
                "attempts": 0,
                "taskRevision": (task_revisions or {}).get(owner),
                "payloadDigest": hashlib.sha256(fields["payload"].encode("utf-8")).hexdigest(),
                "witness": witness,
                "result": None,
            }
            obligations["intents"] = [[key, value] for key, value in intents.items()]
            if invocation_metadata is not None:
                callbacks = dict(obligations["callbacks"])
                for slot in callback_slots.get(fields["operation"], ()):
                    callback_allocator[0] += 1
                    closure_id = args[slot][1]
                    closure = _table(state, "heap")[closure_id]
                    callbacks[identity + "/" + str(slot)] = {
                        "state": "pending",
                        "slot": slot,
                        "closureId": closure_id,
                        "entry": closure["code"],
                        "realmId": invocation_metadata["realmId"],
                        "programId": invocation_metadata["programId"],
                        "trackId": invocation_metadata["track"],
                        "subscriptionId": "callback" + str(callback_allocator[0]),
                        "lifetime": "subscription"
                        if fields["operation"] == "demo.listen"
                        else "call",
                        "sequence": 0,
                        "queued": [],
                        "ownerInvocationId": owner,
                    }
                obligations["callbacks"] = list(map(list, callbacks.items()))
    elif kind in ("observe",):
        _fail("Observed delivery cannot appear as generated detail")
    else:
        _fail("Unsupported detailed transition kind: " + kind)


def _apply_observation(state, row, dispatched, settled, obligations=None, track_counts=None):
    # A rejected delivery must leave the effect, receipt, and Promise together
    # at their previous prefix, including failures after graph import.
    staged_state, staged_dispatched, staged_settled, staged_obligations = copy.deepcopy(
        (state, dispatched, settled, obligations)
    )
    _apply_observation_staged(
        staged_state, row, staged_dispatched, staged_settled, staged_obligations, track_counts
    )
    for target, staged in (
        (state, staged_state),
        (dispatched, staged_dispatched),
        (settled, staged_settled),
        (obligations, staged_obligations),
    ):
        if target is not None:
            target.clear()
            target.update(staged)


def _source_handler(fields):
    return {
        **{
            key: copy.deepcopy(fields[key])
            for key in (
                "programId",
                "realmId",
                "trackId",
                "realmPolicy",
                "entry",
                "queueLimit",
                "closureId",
                "lifetime",
                "ownerInvocationId",
            )
        },
        "seen": [],
        "queue": [],
        "state": "active",
    }


def _apply_observation_staged(state, row, dispatched, settled, obligations, track_counts):
    fields = row["fields"]
    if row["kind"] == "reconcile":
        identity = fields["invocationId"] + "/" + fields["effectId"]
        intent = dict((obligations or {}).get("intents", [])).get(identity)
        if (
            intent is None
            or intent["status"] not in ("dispatched", "uncertain")
            or identity not in dispatched
            or identity in settled
            or _controller_owner(state) != fields["invocationId"]
            or fields["effectId"] not in dict(state["effects"])
        ):
            _fail("Reconciliation lacks one outstanding dispatched effect")
        disposition = fields.get("disposition")
        if (
            disposition not in ("not-dispatched", "lost", "fulfilled", "rejected")
            or type(fields.get("evidenceId")) is not str
            or not fields["evidenceId"]
        ):
            _fail("Reconciliation requires explicit independent evidence")
        intent["status"] = (
            "committed"
            if disposition == "not-dispatched"
            else "lost"
            if disposition == "lost"
            else "uncertain"
        )
        if disposition == "not-dispatched":
            dispatched.remove(identity)
        return
    if row["kind"] == "callback-reattach":
        handler = dict((obligations or {}).get("handlers", [])).get(fields["subscriptionId"])
        callbacks = [
            callback
            for _, callback in (obligations or {}).get("callbacks", [])
            if callback.get("subscriptionId") == fields["subscriptionId"]
        ]
        if (
            handler is None
            or handler["state"] not in ("active", "lost")
            or handler["realmId"] != fields["realmId"]
            or handler["trackId"] != fields["track"]
            or type(fields.get("evidenceId")) is not str
            or not fields["evidenceId"]
            or any(callback["state"] == "revoked" for callback in callbacks)
        ):
            _fail("Reattachment requires the exact live handler and independent evidence")
        handler["state"] = "active"
        for callback in callbacks:
            callback["state"] = "active"
        return
    if row["kind"] == "event":
        if obligations is not None:
            handlers = {key: value for key, value in obligations["handlers"]}
            handler = handlers.get(fields["subscriptionId"])
            callback = next(
                (
                    v
                    for _, v in obligations["callbacks"]
                    if v.get("subscriptionId") == fields.get("subscriptionId")
                ),
                None,
            )
            if handler is None or handler["state"] not in ("active", "closing"):
                _fail("Event has no active source handler")
            if callback is not None:
                if (
                    callback.get("state") not in ("active", "revoked")
                    or callback.get("closureId")
                    != fields.get("closureId", callback.get("closureId"))
                    or callback.get("state") == "revoked"
                    and handler.get("state") != "closing"
                ):
                    _fail("Observed event has no active guest callback registration")
            if fields.get("eventId") in handler["seen"]:
                _fail("Duplicate source event identity")
            handler["seen"].append(fields["eventId"])
            suffix = fields["eventId"].removeprefix("callbackevent")
            if (
                fields["eventId"].startswith("callbackevent")
                and 0 < len(suffix) <= 16
                and suffix.isascii()
                and suffix.isdecimal()
                and suffix[0] != "0"
                and int(suffix) <= 9007199254740991
            ):
                for _, registration in obligations["callbacks"]:
                    if registration.get("subscriptionId") == fields["subscriptionId"]:
                        registration["sequence"] = max(registration.get("sequence", 0), int(suffix))
            obligations["handlers"] = [[key, value] for key, value in handlers.items()]
        return
    if row["kind"] == "handler":
        if obligations is not None:
            handlers = dict(obligations["handlers"])
            if fields["subscriptionId"] in handlers:
                _fail("Duplicate source handler")
            registrations = [
                callback
                for _, callback in obligations["callbacks"]
                if callback.get("subscriptionId") == fields["subscriptionId"]
                or callback.get("state") == "pending"
                and callback.get("ownerInvocationId") == fields["ownerInvocationId"]
                and callback.get("closureId") == fields["closureId"]
            ]
            if not registrations:
                # Host registration is already observed authority. It creates
                # no guest request or callback capability, even when a live
                # invocation owns the externally registered handler's lifetime.
                handlers[fields["subscriptionId"]] = _source_handler(fields)
                obligations["handlers"] = [[key, value] for key, value in handlers.items()]
                return
            if len(registrations) != 1:
                _fail("Handler must identify one pending callback capability")
            callback_args = []
            for identity, intent in obligations["intents"]:
                if intent.get("status") != "dispatched" or intent.get("operation") not in (
                    "demo.withCallback",
                    "demo.listen",
                ):
                    continue
                effect = next(
                    (item for item in state["effects"] if item[0] == intent["effectId"]), None
                )
                if effect is None:
                    continue
                for slot, value in enumerate(effect[1]["arguments"]):
                    if isinstance(value, dict) and set(value) == {"$ownedCallback"}:
                        if value["$ownedCallback"] == fields.get("closureId"):
                            callback_args.append((identity, slot, value["$ownedCallback"], intent))
            matches = [
                entry
                for entry in callback_args
                if entry[2] == fields.get("closureId")
                and entry[3]["invocationId"] == fields.get("ownerInvocationId")
            ]
            if len(matches) != 1:
                _fail("Handler does not identify exactly one dispatched callback capability")
            identity, slot, closure_id, intent = matches[0]
            callbacks = {key: value for key, value in obligations["callbacks"]}
            closure = _table(state, "heap").get(closure_id)
            callback_key = identity + "/" + str(slot)
            callback = {
                "state": "active",
                "slot": slot,
                "closureId": closure_id,
                "entry": closure.get("code"),
                "realmId": fields["realmId"],
                "programId": fields["programId"],
                "trackId": fields["trackId"],
                "subscriptionId": fields["subscriptionId"],
                "lifetime": fields["lifetime"],
                "sequence": 0,
                "queued": [],
                "ownerInvocationId": fields["ownerInvocationId"],
            }
            pending = callbacks.get(callback_key)
            if pending is not None and pending != {**callback, "state": "pending"}:
                _fail("Handler differs from its pending callback capability")
            if fields.get("subscriptionId") in {
                value.get("subscriptionId")
                for key, value in callbacks.items()
                if key != callback_key
            }:
                _fail("Duplicate callback registration")
            callbacks[callback_key] = callback
            obligations["callbacks"] = [[key, value] for key, value in callbacks.items()]
            handlers[fields["subscriptionId"]] = _source_handler(fields)
            obligations["handlers"] = [[key, value] for key, value in handlers.items()]
        return
    if row["kind"] == "handler-close":
        if obligations is not None:
            handlers = dict(obligations["handlers"])
            handler = handlers.get(fields["subscriptionId"])
            discarded = fields.get("discarded")
            if (
                handler is None
                or handler.get("state") not in ("active", "closing")
                or type(discarded) is not list
                or any(type(identity) is not str for identity in discarded)
                or len(set(discarded)) != len(discarded)
                or handler.get("queue")
                and discarded != handler["queue"]
                or any(identity in handler.get("seen", []) for identity in discarded)
            ):
                _fail("Handler close does not match its queued events")
            callbacks = {key: value for key, value in obligations["callbacks"]}
            matches = [
                value
                for value in callbacks.values()
                if value.get("subscriptionId") == fields.get("subscriptionId")
            ]
            if len(matches) > 1 or any(
                callback.get("state") not in ("active", "revoked") for callback in matches
            ):
                _fail("Handler close must revoke one active callback")
            for callback in matches:
                callback["state"] = "revoked"
            obligations["callbacks"] = [[key, value] for key, value in callbacks.items()]
            # Queued arrivals stay private until delivery or retirement. The
            # observed discard receipt first publishes their identities.
            handler.setdefault("seen", []).extend(discarded)
            handler["queue"] = []
            handler["state"] = "closed"
            obligations["handlers"] = [[key, value] for key, value in handlers.items()]
        return
    if row["kind"] == "dispatch":
        if (
            type(fields.get("attempt")) is not int
            or not 1 <= fields["attempt"] <= 9007199254740991
            or fields.get("key") != f"{fields.get('invocationId')}/{fields.get('effectId')}"
        ):
            _fail("Effect dispatch identity")
        effect = next((e for e in state["effects"] if e[0] == fields["effectId"]), None)
        if effect is None:
            _fail("Dispatch of unknown effect")
        if _controller_owner(state) != fields["invocationId"]:
            _fail("Dispatch belongs to another invocation")
        identity = fields["key"]
        if identity in dispatched or identity in settled:
            _fail("Duplicate effect dispatch")
        dispatched.add(identity)
        if obligations is not None:
            intents = {key: value for key, value in obligations["intents"]}
            intent = intents.get(fields["key"])
            if (
                intent is None
                or intent["status"] != "committed"
                or fields["attempt"] != intent["attempts"] + 1
            ):
                _fail("Dispatch without one requested intent")
            intent["status"] = "dispatched"
            intent["attempts"] = fields["attempt"]
            obligations["intents"] = [[key, value] for key, value in intents.items()]
        elif fields["attempt"] != 1:
            _fail("A repeated dispatch requires its reconciled obligation history")
    else:
        kind, obs = parse_record(fields["line"])
        if kind != "observe":
            _fail("Result line must be a detailed observation")
        if (
            fields.get("effectId") != obs.get("effect")
            or fields.get("status") != obs.get("status")
            or fields.get("result") != obs.get("result")
        ):
            _fail("Observed result envelope differs from its detailed observation")
        effect = next((e for e in state["effects"] if e[0] == obs["effect"]), None)
        if effect is None:
            _fail("Result of unknown effect")
        if _controller_owner(state) != fields["invocationId"]:
            _fail("Result belongs to another invocation")
        identity = fields["invocationId"] + "/" + obs["effect"]
        if identity not in dispatched or identity in settled:
            _fail("Result without one outstanding dispatch")
        if "graph" in obs or obs["result"][0] == "ref":
            from .host_operations import HOST_OPERATIONS

            descriptor = HOST_OPERATIONS.get(effect[1]["operation"], {})
            if descriptor.get("resultCodec") != "owned-graph/1":
                _fail("Observed graph requires the operation's declared owned result codec")
        for node in obs.get("graph", {}).get("objects", []):
            _assert_fresh_allocation(state, node["id"])
            state["heap"].append(
                [
                    node["id"],
                    {
                        "kind": node["kind"],
                        "prototype": node.get("prototype", "ordinary"),
                        "properties": [[p["key"], p["value"]] for p in node["properties"]],
                    },
                ]
            )
        if obs["result"][0] == "ref" and obs["result"][1] not in _table(state, "heap"):
            _fail("Observed result references an unknown identity")
        settled.add(identity)
        effect[1]["status"] = obs["status"]
        effect[1]["result"] = obs["result"]
        if obligations is not None:
            identity = fields["invocationId"] + "/" + obs["effect"]
            intents = {key: value for key, value in obligations["intents"]}
            intent = intents.get(identity)
            if intent is None or intent["status"] not in ("dispatched", "uncertain"):
                _fail("Result without one dispatched obligation")
            intent["status"] = "settled"
            seq = row["sequence"]
            observed_line = encode_record("observe", obs, role="observed")
            if fields.get("line") != observed_line:
                _fail("Result witness differs from its canonical V8 detailed record")
            intent["result"] = {
                "schema": "semantic-lifecycle/8",
                "session": fields["session"],
                "sequence": seq,
                "ownerSequence": (track_counts or {}).get(seq, seq - 1),
                "track": fields["track"],
                "mode": "observed",
                "kind": "result",
                "invocationId": fields["invocationId"],
                "effectId": obs["effect"],
                "status": obs["status"],
                "result": obs["result"],
                "line": observed_line,
                "errorOrigin": fields.get("errorOrigin"),
            }
            obligations["intents"] = [[key, value] for key, value in intents.items()]
            handlers = dict(obligations["handlers"])
            for key, callback in obligations["callbacks"]:
                if not key.startswith(identity + "/"):
                    continue
                if obs["status"] == "fulfilled" and callback["lifetime"] == "subscription":
                    callback["state"] = "active"
                else:
                    callback["state"] = "revoked"
                    handler = handlers.get(callback["subscriptionId"])
                    if handler is not None:
                        handler["state"] = "closing"
        promise = effect[1]["promise"]
        if promise:
            from .promise_scheduler_v8 import settle_promise

            machine = _internal_machine(state)
            try:
                settle_promise(machine, promise, obs["status"], obs["result"], external=True)
            except ProtocolV8Error as error:
                raise NativeV8Error(str(error)) from error
            state.clear()
            state.update(_external_machine(machine))
