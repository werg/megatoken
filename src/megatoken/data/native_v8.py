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

from megatoken.controls import canonical_semantic_observation, semantic_selection_instruction

from . import algorithm_v7 as v7
from .algorithm_v7 import _truthy as _detail_truthy
from .algorithm_v7 import stable_algorithm_hash
from .algorithm_v8 import replay_algorithm_line
from .detailed_v7 import parse_record
from .detailed_v7 import value as validate_value
from .protocol_v8 import (
    ProtocolV8Error,
    _assert_fresh_heap_identity,
    apply_protocol_to_shared_state,
    assert_detailed_record_compatible,
    decode_protocol_record,
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
from .semantic_physical import physical_semantic_episode

_UNARY_OPERATORS = {
    operator: name for name, (arity, operator) in v7.OPERATIONS.items() if arity == 1
}
_BINARY_OPERATORS = {
    operator: name
    for name, (arity, operator) in v7.OPERATIONS.items()
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
    """Check public v8 versions, causal contexts, and each available pure target.

    Returns a cautious receipt: detailed controller continuation and source
    correspondence are explicitly untested, so this is never serving admission.
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
                        instruction=semantic_selection_instruction() if masked else "",
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
                    "session_metadata": {"schema": "semantic-execution-metadata/8"},
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
        selection_instruction=lambda continuation=False: semantic_selection_instruction(
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
            from . import algorithm_v7 as v7

            args = (
                [term(node["arg"])] if op == "unary" else [term(node["left"]), term(node["right"])]
            )
            return v7._operation(operation, args)
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


def _initial_state(program, invocation, functions=()):
    graph = invocation.get("inputGraph")
    if not isinstance(graph, dict) or set(graph) != {"roots", "objects"}:
        _fail("Invocation input graph")
    heap = []
    for node in graph["objects"]:
        row = {
            "kind": node["kind"],
            "prototype": "ordinary",
            "properties": [[p["key"], p["value"]] for p in node["properties"]],
        }
        heap.append([node["id"], row])
    fn = next((f for f in functions if f.get("name") == invocation.get("entry")), {})
    params = fn.get("params", [])
    locals_ = []
    aliases = []
    for index, value in enumerate(graph["roots"]):
        cell = f"a0b{index}"
        locals_.append([cell, value])
        if index < len(params):
            aliases.append([params[index], cell])
    for name, index in invocation.get("bindings", {}).items():
        if type(index) is int and 0 <= index < len(graph["roots"]) and name not in dict(aliases):
            aliases.append([name, f"a0b{index}"])
    return {
        "heap": heap,
        "locals": locals_,
        "stack": [invocation["invocationId"]],
        "effects": [],
        "jobs": [],
        "ready": [],
        "randomState": None,
        "pending": None,
        "lastReturn": None,
        "completion": None,
        "receiver": invocation.get("receiver", ["undefined"]),
        "captureAliases": aliases,
        "continuations": [],
        "reactionSequence": 0,
        "blockedSync": None,
        "callReturn": None,
        "activeJob": None,
        "scheduler": {"registration": 0, "ready": 0},
        "frameFunctions": [[invocation["invocationId"], invocation["entry"]]],
        "jobErrors": [],
        "protocol": None,
    }


def _replay_shared_state(timeline, contexts, programs_from_context, *, roundtrip):
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
    owner_sequence = 0
    suspended = {}
    common_state = None
    active_invocation = None
    callback_invocations = set()
    for prior in timeline:
        if prior.get("fields", {}).get("track") != "host":
            owner_sequence += 1
        owner_sequences[prior["sequence"]] = owner_sequence
    for row in timeline:
        kind = row["kind"]
        fields = row["fields"]
        detail_kind = None
        if kind == "program":
            program = fields.get("regionProgram")
            if not isinstance(program, dict):
                _fail("Program record omits its immutable region definition")
            if fields.get("digest") != program.get("sourceDigest") or program.get(
                "source"
            ) != fields.get("source"):
                _fail("Public region definition differs from immutable source publication")
            published[fields["programId"]] = (program, fields.get("functions", []))
            continue
        if kind == "invocation":
            published_row = published.get(fields["programId"])
            if published_row is None:
                _fail("Invocation has no preceding program")
            program, functions = published_row
            if fields.get("sourceDigest") != program.get("sourceDigest"):
                _fail("Invocation source/definition binding")
            if fields.get("activation") == "callback" and (
                state is not None or common_state is not None
            ):
                owner = fields.get("parentInvocationId")
                if owner and owner not in suspended:
                    suspended[owner] = copy.deepcopy(state)
                base = state if state is not None else common_state
                callback_invocation = copy.deepcopy(fields)
                callback_invocation["inputGraph"]["roots"] = []
                callback_state = _initial_state(program, callback_invocation, functions)
                callback_state["heap"] = copy.deepcopy(base["heap"])
                callback_state["locals"] = copy.deepcopy(base["locals"])
                callback_state["scheduler"] = copy.deepcopy(base["scheduler"])
                callback_state["reactionSequence"] = base["reactionSequence"]
                callback_state["randomState"] = base["randomState"]
                callback_state["effects"] = []
                callback_state["jobs"] = []
                callback_state["ready"] = []
                callback_state["continuations"] = []
                callback_state["stack"] = [fields["invocationId"]]
                callback_state["pending"] = None
                callback_state["lastReturn"] = None
                callback_state["completion"] = None
                callback_state["callReturn"] = None
                callback_state["blockedSync"] = None
                callback_state["receiver"] = fields.get("receiver", ["undefined"])
                callback_state["captureAliases"] = []
                callback_state["frameFunctions"] = [[fields["invocationId"], fields["entry"]]]
                closure = next(
                    (
                        value
                        for identity, value in callback_state["heap"]
                        if identity == fields.get("closureId")
                    ),
                    None,
                )
                if closure is None or closure.get("kind") != "closure":
                    _fail("Callback invocation has no published guest closure")
                callback_state["captureAliases"] = [
                    [capture["binding"], capture["cell"]] for capture in closure.get("captures", [])
                ]
                function = next(
                    (
                        fn
                        for fn in functions
                        if fn.get("name") == fields.get("entry")
                        or "function:" + str(fn.get("start")) == fields.get("entry")
                    ),
                    None,
                )
                params = function.get("params", []) if function else []
                roots = fields.get("inputGraph", {}).get("roots", [])
                for index, value in enumerate(roots):
                    # Callback invocation identifiers end in their deterministic event ordinal.
                    suffix = "".join(c for c in fields["invocationId"] if c.isdigit()) or "0"
                    cell = f"a{suffix}b{index}"
                    _set_row(callback_state["locals"], cell, value)
                    _set_row(callback_state["captureAliases"], params[index], cell) if index < len(
                        params
                    ) else None
                state = callback_state
            elif fields.get("activation") == "call":
                state = _initial_state(program, fields, functions)
            else:
                _fail("Unsupported invocation activation in v8 shared replay")
            active_invocation = fields["invocationId"]
            if fields.get("activation") == "callback":
                callback_invocations.add(active_invocation)
                common_state = copy.deepcopy(state)
            task_revisions[fields["invocationId"]] = fields.get("taskRevision")
            continue
        if state is None and row["mode"] == "generated" and kind == "semantic":
            parsed_kind, parsed_fields = parse_record(fields["line"])
            if parsed_kind == "resume" and parsed_fields["invocation"] in suspended:
                state = copy.deepcopy(suspended[parsed_fields["invocation"]])
                active_invocation = parsed_fields["invocation"]
                if common_state is not None:
                    _merge_callback_state(state, common_state)
        if state is None and kind in ("handler", "handler-close", "event"):
            # Host handler rows affect obligations, not execution locals.
            _apply_observation({}, row, dispatched, settled, obligations, owner_sequences)
            continue
        if state is None:
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
            program = context["programs"][-1]["program"]
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
                if state.get("protocol") is None and (
                    state["effects"] or state["jobs"] or state["ready"] or len(state["stack"]) != 1
                ):
                    _fail("The first protocol profile must begin as a fresh synchronous invocation")
                try:
                    state = apply_protocol_to_shared_state(state, fields["line"])
                except ProtocolV8Error as e:
                    raise NativeV8Error(
                        f"Invalid independently replayed protocol transition: {e}"
                    ) from e
            elif kind == "semantic":
                k, payload = parse_record(fields["line"])
                detail_kind = k
                if state.get("protocol") is not None:
                    try:
                        assert_detailed_record_compatible(state, k, payload)
                    except ProtocolV8Error as e:
                        raise NativeV8Error(f"Detailed record bypasses protocol state: {e}") from e
                _apply_detail(state, k, payload, row, obligations, program, task_revisions)
        elif kind in ("dispatch", "result"):
            target = (
                state
                if fields.get("invocationId") == active_invocation
                else suspended.get(fields.get("invocationId"))
            )
            if target is None:
                _fail("Observation has no independently replayed invocation state")
            _apply_observation(target, row, dispatched, settled, obligations, owner_sequences)
            if target is not state:
                _merge_callback_state(state, target)
            for saved in suspended.values():
                _merge_callback_state(saved, target)
        elif kind in ("handler", "handler-close", "event"):
            _apply_observation(state, row, dispatched, settled, obligations, owner_sequences)
        if active_invocation is not None:
            for saved in suspended.values():
                _merge_callback_state(saved, state)
            common_state = (
                copy.deepcopy(state) if active_invocation not in suspended else common_state
            )
        if (
            detail_kind == "end"
            and active_invocation == fields.get("invocationId")
            and active_invocation in callback_invocations
        ):
            state = None
            active_invocation = None
    return state


def _merge_callback_state(target, source):
    """Carry heap/cells and scheduler state between callback and owner frames."""
    if not isinstance(target, dict) or not isinstance(source, dict):
        return
    for field in ("heap", "locals", "scheduler", "reactionSequence", "randomState"):
        if field in source:
            target[field] = copy.deepcopy(source[field])


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


def _eval_detail_expression(expr, state):
    from . import algorithm_v7 as v7

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
        _fail("Random transition needs explicit checked state")
    if op == "unary":
        arg = _eval_detail_expression(expr["arg"], state)
        mapping = _UNARY_OPERATORS
        if expr["operator"] not in mapping:
            _fail("Unsupported unary operator")
        return v7._operation(mapping[expr["operator"]], [arg])
    if op == "binary":
        left = _eval_detail_expression(expr["left"], state)
        right = _eval_detail_expression(expr["right"], state)
        mapping = _BINARY_OPERATORS
        if expr["operator"] not in mapping:
            _fail("Unsupported binary operator")
        return v7._operation(mapping[expr["operator"]], [left, right])
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
    aliases = dict(state["captureAliases"])
    for name, value in target["bindings"]:
        _set_row(state["locals"], aliases.get(name, name), value)
    if target["completion"]["kind"] == "return":
        state["pending"] = {"kind": "return", "value": target["completion"]["value"]}
        state["stack"] = []


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


def _apply_detail(state, kind, fields, row, obligations=None, program=None, task_revisions=None):
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
            node["captures"] = fields["captures"]
        if "length" in fields:
            node["properties"].append(["length", ["num", fields["length"]]])
        _set_row(state["heap"], fields["name"], node)
    elif kind == "write":
        heap = _table(state, "heap")
        node = heap.get(fields["object"])
        if node is None:
            _fail("Write to absent object")
        props = dict(node["properties"])
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
                args = fields["expr"].get("args", [])
                if not args:
                    _fail("Malformed lazy expression")
                left = _eval_detail_expression(args[0], state)
                use_right = (
                    _detail_truthy(left)
                    if op == "and"
                    else not _detail_truthy(left)
                    if op == "or"
                    else left[0] in ("null", "undefined")
                    if op == "coalesce"
                    else None
                )
                if use_right is None or (path == "evaluate-right") != use_right:
                    _fail("Logical decision path is inconsistent")
    elif kind == "return":
        value = fields["value"]
        invocation = fields.get(
            "invocation", row.get("invocationId", state["stack"][-1] if state["stack"] else None)
        )
        if len(state["stack"]) > 1 and state["stack"][-1] == invocation:
            state["lastReturn"] = {"kind": "return", "value": value}
            state["stack"].pop()
        else:
            state["pending"] = {"kind": "return", "value": value}
            state["stack"] = []
    elif kind == "throw":
        value = fields["value"]
        invocation = fields.get("invocation")
        if invocation is not None and state["stack"] and state["stack"][-1] != invocation:
            _fail("Throw invocation mismatch")
        if len(state["stack"]) > 1:
            state["lastReturn"] = {"kind": "throw", "value": value}
            state["stack"].pop()
        else:
            state["pending"] = {"kind": "throw", "value": value}
            state["stack"] = []
    elif kind == "transfer":
        completion_kind = fields["kind"]
        if completion_kind not in ("return", "throw", "cleanup"):
            _fail("Unknown completion transfer")
        state["pending"] = {
            "kind": completion_kind,
            "value": fields["value"],
            "handler": fields["handler"],
        }
    elif kind == "end":
        if state["stack"]:
            state["stack"].pop()
        state["completion"] = state["pending"] or {"kind": fields["kind"]}
        state["pending"] = None
    elif kind == "request":
        effect = fields["effect"]
        promise = fields.get("promise")
        args = fields["arguments"]
        from . import algorithm_v7 as v7

        callback_slots = {"demo.withCallback": {1}, "demo.listen": {1}}
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
                item = v7._decode_scalar(value)
                if isinstance(item, tuple) and item and item[0] == "__algorithm_ref__":
                    _fail("Object request argument requires owned graph codec")
                if item is v7.UNDEFINED:
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
            "argumentCodec": "canonical-json/1",
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
            from . import witness_v6

            identity = fields["invocation"] + "/" + effect
            rendered = [witness_v6._trace_value(v) for v in args]
            call = fields["operation"] + "(" + ", ".join(rendered) + ")"
            span = fields["site"]
            suffix = (" promise @" + promise) if promise else ""
            witness = (
                "request @"
                + effect
                + " "
                + call
                + " payload "
                + fields["payload"]
                + suffix
                + " in @"
                + fields["invocation"]
                + " at "
                + str(span["start"])
                + ".."
                + str(span["end"])
            )
            intents = {key: value for key, value in obligations["intents"]}
            if identity in intents:
                _fail("Duplicate effect intent")
            intents[identity] = {
                "status": "requested",
                "operation": fields["operation"],
                "payload": fields["payload"],
                "invocationId": fields["invocation"],
                "effectId": effect,
                "attempts": 0,
                "taskRevision": (task_revisions or {}).get(fields["invocation"]),
                "payloadDigest": hashlib.sha256(fields["payload"].encode("utf-8")).hexdigest(),
                "witness": witness,
                "result": None,
            }
            obligations["intents"] = [[key, value] for key, value in intents.items()]
    elif kind == "await":
        seq = state["scheduler"]["registration"] + 1
        state["scheduler"]["registration"] = seq
        _set_row(
            state["continuations"],
            fields["continuation"],
            {
                "state": "waiting",
                "promise": fields["promise"],
                "invocation": fields["invocation"],
                "sequence": seq,
                "activeJob": None,
                "captureAliases": copy.deepcopy(state["captureAliases"]),
                "readySequence": None,
            },
        )
        if fields.get("returnPromise"):
            state["callReturn"] = {"kind": "return", "value": ["ref", fields["returnPromise"]]}
        state["stack"] = []
    elif kind == "resume":
        continuation = _table(state, "continuations").get(fields["continuation"])
        if continuation is None:
            _fail("Resume without continuation")
        continuation["state"] = "resumed"
        state["ready"] = [v for v in state["ready"] if v != fields["continuation"]]
        if fields["invocation"] not in state["stack"]:
            state["stack"].append(fields["invocation"])
    elif kind == "promise":
        identity = fields["promise"]
        heap = _table(state, "heap")
        if fields["action"] == "create":
            _assert_fresh_allocation(state, identity)
            _set_row(
                state["heap"],
                identity,
                {
                    "kind": "promise",
                    "properties": [],
                    "state": "pending",
                    "result": None,
                    "adopted": None,
                    "external": None,
                    "reactions": [],
                },
            )
        elif fields["action"] == "resolve":
            node = heap.get(identity)
            if node is None or node["state"] != "pending":
                _fail("Resolve unknown or settled promise")
            node["state"] = "fulfilled"
            node["result"] = fields["value"]
    elif kind in ("observe",):
        _fail("Observed delivery cannot appear as generated detail")
    else:
        _fail("Unsupported detailed transition kind: " + kind)


def _apply_observation(state, row, dispatched, settled, obligations=None, track_counts=None):
    fields = row["fields"]
    if row["kind"] == "event":
        if obligations is not None:
            callback = next(
                (
                    v
                    for _, v in obligations["callbacks"]
                    if v.get("subscriptionId") == fields.get("subscriptionId")
                ),
                None,
            )
            if (
                callback is None
                or callback.get("state") != "active"
                or callback.get("closureId") != fields.get("closureId", callback.get("closureId"))
            ):
                _fail("Observed event has no active guest callback registration")
            handlers = {key: value for key, value in obligations["handlers"]}
            handler = handlers.get(fields["subscriptionId"])
            if handler is None or handler["state"] != "active":
                _fail("Event has no active source handler")
            if fields.get("eventId") in handler["seen"]:
                _fail("Duplicate source event identity")
            handler["seen"].append(fields["eventId"])
            obligations["handlers"] = [[key, value] for key, value in handlers.items()]
        return
    if row["kind"] == "handler":
        if obligations is not None:
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
            if fields.get("subscriptionId") in {
                value.get("subscriptionId") for _, value in obligations["callbacks"]
            }:
                _fail("Duplicate callback registration")
            callbacks = {key: value for key, value in obligations["callbacks"]}
            closure = _table(state, "heap").get(closure_id)
            callbacks[identity + "/" + str(slot)] = {
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
            obligations["callbacks"] = [[key, value] for key, value in callbacks.items()]
            handlers = {key: value for key, value in obligations["handlers"]}
            if fields["subscriptionId"] in handlers:
                _fail("Duplicate source handler")
            handlers[fields["subscriptionId"]] = {
                "programId": fields["programId"],
                "realmId": fields["realmId"],
                "trackId": fields["trackId"],
                "realmPolicy": fields["realmPolicy"],
                "entry": fields["entry"],
                "queueLimit": fields["queueLimit"],
                "seen": [],
                "queue": [],
                "state": "active",
                "closureId": fields["closureId"],
                "lifetime": fields["lifetime"],
                "ownerInvocationId": fields["ownerInvocationId"],
            }
            obligations["handlers"] = [[key, value] for key, value in handlers.items()]
        return
    if row["kind"] == "handler-close":
        if obligations is not None:
            callbacks = {key: value for key, value in obligations["callbacks"]}
            matches = [
                value
                for value in callbacks.values()
                if value.get("subscriptionId") == fields.get("subscriptionId")
            ]
            if len(matches) != 1 or matches[0].get("state") != "active":
                _fail("Handler close must revoke one active callback")
            matches[0]["state"] = "revoked"
            obligations["callbacks"] = [[key, value] for key, value in callbacks.items()]
            handlers = {key: value for key, value in obligations["handlers"]}
            handler = handlers.get(fields["subscriptionId"])
            if (
                handler is None
                or handler.get("state") != "active"
                or fields.get("discarded") != handler.get("queue")
            ):
                _fail("Handler close does not match its queued events")
            handler["state"] = "closed"
            obligations["handlers"] = [[key, value] for key, value in handlers.items()]
        return
    if row["kind"] == "dispatch":
        if (
            fields.get("attempt") != 1
            or fields.get("key") != f"{fields.get('invocationId')}/{fields.get('effectId')}"
        ):
            _fail("Effect dispatch identity")
        effect = next((e for e in state["effects"] if e[0] == fields["effectId"]), None)
        if effect is None:
            _fail("Dispatch of unknown effect")
        if fields["effectId"] in dispatched or fields["effectId"] in settled:
            _fail("Duplicate effect dispatch")
        dispatched.add(fields["effectId"])
        if obligations is not None:
            intents = {key: value for key, value in obligations["intents"]}
            intent = intents.get(fields["key"])
            if intent is None or intent["status"] != "requested" or intent["attempts"] != 0:
                _fail("Dispatch without one requested intent")
            intent["status"] = "dispatched"
            intent["attempts"] = fields["attempt"]
            obligations["intents"] = [[key, value] for key, value in intents.items()]
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
        if obs["effect"] not in dispatched or obs["effect"] in settled:
            _fail("Result without one outstanding dispatch")
        settled.add(obs["effect"])
        effect[1]["status"] = obs["status"]
        effect[1]["result"] = obs["result"]
        if obligations is not None:
            identity = effect[1]["invocation"] + "/" + obs["effect"]
            intents = {key: value for key, value in obligations["intents"]}
            intent = intents.get(identity)
            if intent is None or intent["status"] != "dispatched":
                _fail("Result without one dispatched obligation")
            intent["status"] = "settled"
            from . import witness_v6

            seq = row["sequence"]
            intent["result"] = {
                "schema": "semantic-execution/6",
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
                "line": "observe @"
                + obs["effect"]
                + " "
                + obs["status"]
                + " "
                + witness_v6._trace_value(obs["result"]),
                "errorOrigin": fields.get("errorOrigin"),
            }
            obligations["intents"] = [[key, value] for key, value in intents.items()]
        promise = effect[1]["promise"]
        if promise:
            node = _table(state, "heap").get(promise)
            node["state"] = obs["status"]
            node["result"] = obs["result"]
        for cid, cont in state["continuations"]:
            if cont["promise"] == promise and cont["state"] == "waiting":
                seq = state["scheduler"]["ready"] + 1
                state["scheduler"]["ready"] = seq
                cont["state"] = "ready"
                cont["readySequence"] = seq
                state["ready"].append(cid)
