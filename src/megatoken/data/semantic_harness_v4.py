"""Independent native consumer for the explicit semantic-execution/4 harness envelope.

It imports validated semantic-trace/4 requests and observed results, never teacher sidecars,
and uses discovered dynamic track addresses rather than eight fixed lanes.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from .host_operations import HOST_OPERATIONS
from .schema import (
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
from .semantic_context import validate_selected_context
from .semantic_contract import validate_envelope_record
from .semantic_request import _host_request
from .semantic_physical import physical_semantic_episode
from .witness_v4 import WitnessV4Error, parse_record
from .witness_v4 import graph as validate_graph
from .witness_v4 import value as validate_value


class SelectedHarnessError(ValueError):
    """An envelope cannot be safely admitted as a native causal episode."""


@dataclass(frozen=True)
class SelectedHarnessReceipt:
    records: int
    invocations: int
    tracks: int
    actions: int
    observations: int
    admission: str
    codec_fingerprint: str | None
    completions: tuple[dict, ...]


def _fail(message: str) -> None:
    raise SelectedHarnessError(message)


def _ident(value: Any) -> bool:
    return (
        isinstance(value, str)
        and bool(value)
        and value[0].islower()
        and all(c.isascii() and (c.islower() or c.isdigit() or c in ":_-") for c in value)
    )


def _witness(line: Any) -> tuple[str, dict[str, Any]]:
    try:
        return parse_record(line)
    except WitnessV4Error as error:
        raise SelectedHarnessError(str(error)) from error


def _entries(program: dict) -> set[str]:
    return {
        program["entry"],
        *(
            function.get("name")
            for function in program.get("functions", [])
            if function.get("name")
        ),
        *(
            f"function:{function['start']}"
            for function in program.get("functions", [])
            if type(function.get("start")) is int
        ),
    }


def _settle(activation, identity, status, result, *, external=False):
    promise = activation["promises"].get(identity)
    if promise is None or promise["state"] != "pending":
        _fail("Promise must exist and may settle only once")
    if promise.get("external") and not external:
        _fail("Only an observed broker result can settle an external Promise")
    if status == "fulfilled" and result[0] == "ref" and result[1] in activation["promises"]:
        target, seen = result[1], {identity}
        while target:
            if target in seen:
                _fail("Promise adoption cycle")
            seen.add(target)
            target = activation["promises"][target].get("target")
        promise["target"] = result[1]
    else:
        promise.update(state=status, result=result)


def _refresh_ready(activation):
    changed = True
    while changed:
        changed = False
        for promise in activation["promises"].values():
            if promise["state"] == "pending" and promise.get("target"):
                target = activation["promises"][promise["target"]]
                if target["state"] != "pending":
                    promise.update(state=target["state"], result=target["result"])
                    changed = True
        for job in activation["jobs"].values():
            if job["state"] == "waiting-cleanup":
                cleanup = activation["promises"][job["cleanup"]]
                if cleanup["state"] != "pending":
                    status, result = (
                        (cleanup["state"], cleanup["result"])
                        if cleanup["state"] == "rejected"
                        else (job["originalStatus"], job["originalResult"])
                    )
                    _settle(activation, job["target"], status, result)
                    job["state"] = "completed"
                    changed = True
    eligible = []
    for identity, work in [*activation["jobs"].items(), *activation["continuations"].items()]:
        if (
            work["state"] == "waiting"
            and activation["promises"][work["promise"]]["state"] != "pending"
        ):
            eligible.append((work["sequence"], identity, work))
    for _, identity, work in sorted(eligible):
        work["state"] = "ready"
        activation["ready"].append(identity)


def adapt_selected_harness_v4(
    export: Mapping[str, Any],
    *,
    codec: Any | None = None,
    max_positions: int = 8192,
    max_history: int = 1024,
    allow_local: bool = False,
    admission_verifier: Callable[[Mapping[str, Any]], bool] | None = None,
) -> tuple[Episode, SelectedHarnessReceipt]:
    """Import complete, controlled selected trajectories (not learned-success evidence)."""
    public = export.get("public") if isinstance(export, Mapping) else None
    sidecar = export.get("sidecar") if isinstance(export, Mapping) else None
    if not isinstance(public, Mapping) or not isinstance(sidecar, Mapping):
        _fail("A public envelope and separate supervision sidecar are required")
    if (
        public.get("schema") != "semantic-execution/4"
        or public.get("traceFormat") != "semantic-trace/4"
        or public.get("contextFormat") != "semantic-context/4"
        or public.get("profile") != "selected-harness/2"
        or sidecar.get("schema") != "semantic-supervision/2"
        or sidecar.get("representation") != "compact"
    ):
        _fail("Wrong explicit selected-harness contract/version")
    admission = sidecar.get("admission", "locally_verified")
    if admission != "admitted" and not allow_local:
        _fail("Unadmitted selected export requires allow_local=True")
    if admission == "admitted" and (
        sidecar.get("referenceStatus") != "agreed"
        or admission_verifier is None
        or not admission_verifier(export)
    ):
        _fail("Admission needs independently checked source/effect evidence")
    timeline = public.get("timeline")
    if not isinstance(timeline, list) or not 2 <= len(timeline) <= 65536:
        _fail("Missing or over-budget complete execution timeline")
    session = timeline[0].get("session") if isinstance(timeline[0], dict) else None
    if not _ident(session):
        _fail("Missing session identity")

    programs: dict[str, dict] = {}
    activations: dict[str, dict] = {}
    realms: dict[str, dict] = {}
    handlers: dict[str, dict] = {}
    pending_event: dict | None = None
    requests: dict[tuple[str, str], dict] = {}
    dispatched: set[tuple[str, str]] = set()
    settled: set[tuple[str, str]] = set()
    tracks: dict[str, int] = {}
    records: list[PhysicalRecord] = []
    meridians: list[Meridian] = []
    actions: list[LogicalAction] = []
    dependencies: list[Dependency] = []
    last_by_track: dict[str, str] = {}
    owner_sequences: dict[str, int] = {}
    visible_sources: dict[str, set[str]] = {}
    uncertain: set[tuple[str, str]] = set()
    reconciliations: dict[tuple[str, str], str] = {}
    observed = 0
    # A controlled, locally verified action is not an admitted teacher
    # projection. Its origin must not be relabeled to manufacture authority.
    selected_provenance = (
        "teacher"
        if admission == "admitted" and sidecar.get("selectionOrigin") == "certified_teacher"
        else "student"
        if sidecar.get("selectionOrigin") == "model"
        else "unknown"
    )

    for index, item in enumerate(timeline):
        if not isinstance(item, dict):
            _fail("Malformed timeline record")
        try:
            validate_envelope_record(item)
        except ValueError as error:
            raise SelectedHarnessError(str(error)) from error
        track = item.get("track")
        if (
            item.get("schema") != "semantic-execution/4"
            or item.get("session") != session
            or type(item.get("sequence")) is not int
            or item["sequence"] != index + 1
            or not _ident(track)
            or type(item.get("ownerSequence")) is not int
            or item["ownerSequence"] != owner_sequences.get(track, 0) + 1
        ):
            _fail("Session/track causal ordering or ownership mismatch")
        owner_sequences[track] = item["ownerSequence"]
        if track not in tracks:
            tracks[track] = len(tracks)
        kind, mode = item.get("kind"), item.get("mode")
        if kind in (
            "program",
            "invocation",
            "dispatch",
            "result",
            "handler",
            "event",
            "handler-close",
            "source-evict",
            "source-read",
            "cancel",
            "reconcile",
            "callback-reattach",
            "model-context",
        ):
            if mode != "observed":
                _fail("A host observation cannot be generated by the student")
            observed += 1
        elif kind == "semantic":
            if mode != "generated" or item.get("traceFormat") != "semantic-trace/4":
                _fail("Only validated witness selections carry model loss")
        else:
            _fail("Unregistered harness envelope kind")
        invocation_id = item.get("invocationId")
        if pending_event is not None and kind != "invocation":
            _fail("Delivered outside event must be followed by its observed invocation")
        action_id = None
        result_of = None
        execution_id = None
        if kind in {"source-evict", "source-read"}:
            program_id = item.get("programId")
            program = programs.get(program_id)
            if not program or item.get("digest") != program["digest"] or track == "host":
                _fail("Source availability receipt needs an existing program and work track")
            if kind == "source-read":
                if item.get("source") != program["source"]:
                    _fail("Source reread must preserve immutable code bytes")
                visible_sources.setdefault(track, set()).add(program_id)
            else:
                visible_sources.setdefault(track, set()).discard(program_id)
        elif kind == "model-context":
            activation = activations.get(invocation_id)
            reads = item.get("sourceReads")
            if not activation or activation["track"] != track or not isinstance(reads, list):
                _fail("Model context receipt requires its original activation")
            if any(
                not isinstance(read, dict)
                or read.get("sourceDigest") != programs[activation["program"]]["digest"]
                for read in reads
            ):
                _fail("Model source reread cannot change the original program")
            if validate_selected_context(item, timeline[:index]):
                snapshot = item["selectedState"]
                realm = realms[activation["realm"]]
                heap = dict(snapshot["heap"])
                if set(heap) != realm["refs"] | set(activation["promises"]):
                    _fail("Selected-state observation changed its committed heap identities")
                for identity, node in realm["heap"].items():
                    selected = heap[identity]
                    if selected.get("kind") != node["kind"] or dict(
                        selected.get("properties", [])
                    ) != {prop["key"]: prop["value"] for prop in node["properties"]}:
                        _fail("Selected-state observation changed its committed data graph")
                if (
                    snapshot["stack"] != activation["stack"]
                    or snapshot["ready"] != activation["ready"]
                ):
                    _fail("Selected-state observation changed owned scheduling state")
                if activation["generated"] == 0:
                    invocation = activation["invocation"]
                    program = programs[activation["program"]]
                    function = next(
                        (
                            row
                            for row in program["functions"]
                            if row.get("name") == invocation["entry"]
                            or "function:" + str(row["start"]) == invocation["entry"]
                        ),
                        {},
                    )
                    roots = invocation["inputGraph"]["roots"]
                    initial = {
                        name: roots[offset] if offset < len(roots) else ["undefined"]
                        for offset, name in enumerate(function.get("params", []))
                        if name
                    }
                    initial.update(
                        {name: roots[offset] for name, offset in invocation["bindings"].items()}
                    )
                    aliases, local = dict(snapshot["captureAliases"]), dict(snapshot["locals"])
                    if any(
                        name not in aliases or local.get(aliases[name]) != selected
                        for name, selected in initial.items()
                    ):
                        _fail("Selected-state observation changed an observed initial argument")
        elif kind == "cancel":
            activation = activations.get(invocation_id)
            effects = item.get("uncertain")
            if (
                not activation
                or activation["track"] != track
                or not isinstance(effects, list)
                or len(effects) != len(set(effects))
            ):
                _fail("Cancellation needs the original invocation and unresolved actions")
            expected = {effect for owner, effect in dispatched - settled if owner == invocation_id}
            if set(effects) != expected:
                _fail("Cancellation cannot erase a possibly dispatched external action")
            uncertain.update((invocation_id, effect) for effect in effects)
            activation["terminal"] = True
            activation["cancelled"] = True
            if realms[activation["realm"]]["active"] == invocation_id:
                realms[activation["realm"]]["active"] = None
        elif kind == "reconcile":
            key = (invocation_id, item.get("effectId"))
            if (
                key not in dispatched - settled
                or key in reconciliations
                or not _ident(item.get("evidenceId"))
                or item.get("disposition")
                not in {"not-dispatched", "lost", "fulfilled", "rejected"}
            ):
                _fail("Reconciliation requires an uncertain action and independent evidence")
            uncertain.discard(key)
            reconciliations[key] = item["disposition"]
            if item["disposition"] == "not-dispatched":
                dispatched.discard(key)
        elif kind == "callback-reattach":
            handler = handlers.get(item.get("subscriptionId"))
            if (
                not handler
                or handler["state"] != "active"
                or handler["realm"] != item.get("realmId")
                or handler["track"] != track
            ):
                _fail("Callback reattachment needs its existing live subscription")
        elif kind == "program":
            name, source, digest = (item.get("programId"), item.get("source"), item.get("digest"))
            if (
                track != "host"
                or not _ident(name)
                or name in programs
                or not isinstance(source, str)
                or not source
                or not isinstance(digest, str)
                or hashlib.sha256(source.encode("utf-8")).hexdigest() != digest
                or not isinstance(item.get("entry"), str)
                or not isinstance(item.get("sites"), list)
                or not isinstance(item.get("capabilities"), list)
                or (
                    "functions" in item
                    and (
                        not isinstance(item["functions"], list)
                        or any(
                            not isinstance(f, dict)
                            or not isinstance(f.get("name"), (str, type(None)))
                            for f in item["functions"]
                        )
                    )
                )
            ):
                _fail("Malformed immutable program publication")
            programs[name] = item
        elif kind == "handler":
            subscription = item.get("subscriptionId")
            program_id = item.get("programId")
            if (
                track != "host"
                or not _ident(subscription)
                or subscription in handlers
                or program_id not in programs
                or not _ident(item.get("realmId"))
                or not _ident(item.get("trackId"))
                or item.get("entry") not in _entries(programs[program_id])
                or item.get("realmPolicy") not in ("fresh", "retained")
                or type(item.get("queueLimit")) is not int
                or not 1 <= item["queueLimit"] <= 4096
            ):
                _fail("Handler registration needs an observed program and scoped queue")
            handlers[subscription] = {
                "state": "active",
                "program": program_id,
                "realm": item["realmId"],
                "track": item["trackId"],
                "entry": item["entry"],
                "policy": item["realmPolicy"],
                "seen": set(),
                "closure": item.get("closureId"),
                "owner": item.get("ownerInvocationId"),
                "lifetime": item.get("lifetime"),
            }
        elif kind == "event":
            subscription = item.get("subscriptionId")
            event = item.get("eventId")
            handler = handlers.get(subscription)
            if (
                not handler
                or handler["state"] != "active"
                or not _ident(event)
                or event in handler["seen"]
                or item.get("programId") != handler["program"]
                or item.get("realmId") != handler["realm"]
                or track != handler["track"]
                or not _ident(invocation_id)
                or item.get("triggerId") != f"event:{subscription}:{event}"
            ):
                _fail("Event requires an active unique handler and trusted observed trigger")
            handler["seen"].add(event)
            pending_event = item
        elif kind == "handler-close":
            handler = handlers.get(item.get("subscriptionId"))
            discard = item.get("discarded")
            if (
                track != "host"
                or not handler
                or handler["state"] != "active"
                or not isinstance(discard, list)
                or len(discard) > 4096
                or len(discard) != len(set(discard))
                or any(not _ident(event) for event in discard)
            ):
                _fail("Handler close requires active registration and bounded disposition")
            handler["state"] = "closed"
        elif kind == "invocation":
            program_id = item.get("programId")
            if (
                not _ident(invocation_id)
                or invocation_id in activations
                or program_id not in programs
                or not all(_ident(item.get(k)) for k in ("realmId", "triggerId"))
                or track == "host"
                or item.get("entry") not in _entries(programs[program_id])
                or item.get("sourceDigest") != programs[program_id]["digest"]
                or item.get("completionPolicy")
                not in ("call-return", "await-returned-promise", "owned-quiescence")
                or type(item.get("taskRevision")) is not int
                or item["taskRevision"] < 1
                or item.get("activation") not in ("call", "module-init", "callback")
            ):
                _fail("Invocation needs prepublished code, realm, trigger and fresh identity")
            try:
                retained = realms.get(item["realmId"], {}).get("refs", set())
                validate_graph(item.get("inputGraph"), retained=retained)
                validate_value(item.get("receiver"))
            except WitnessV4Error as error:
                raise SelectedHarnessError(str(error)) from error
            policy = item.get("realmPolicy")
            if policy not in ("fresh", "retained"):
                _fail("Unrecognized realm ownership policy")
            realm_id = item["realmId"]
            existing = realms.get(realm_id)
            if existing and (
                existing["active"]
                or existing["digest"] != item["sourceDigest"]
                or policy != "retained"
            ):
                _fail("Fresh, busy or mismatched retained JS realm")
            refs = set(existing["refs"]) if existing else set()
            if refs & {node["id"] for node in item["inputGraph"]["objects"]}:
                _fail("Invocation graph cannot overwrite retained object identities")
            refs.update(node["id"] for node in item["inputGraph"]["objects"])
            if item["receiver"][0] == "ref" and item["receiver"][1] not in refs:
                _fail("Invocation receiver is not an owned graph reference")
            if (
                item.get("closureId") is not None
                and item["closureId"] not in refs
                or item.get("parentInvocationId") is not None
                and item["parentInvocationId"] not in activations
                or not isinstance(item.get("bindings"), dict)
                or any(
                    not isinstance(key, str)
                    or type(value) is not int
                    or not 0 <= value < len(item["inputGraph"]["roots"])
                    for key, value in item["bindings"].items()
                )
            ):
                _fail("Invocation closure, parent or source bindings are invalid")
            if pending_event is not None:
                if (
                    pending_event["invocationId"] != invocation_id
                    or pending_event["track"] != track
                    or pending_event["triggerId"] != item["triggerId"]
                    or pending_event["programId"] != program_id
                    or pending_event["realmId"] != realm_id
                    or item["entry"] != handlers[pending_event["subscriptionId"]]["entry"]
                ):
                    _fail("Outside event was not delivered to its bound handler invocation")
                pending_event = None
            realms[realm_id] = {
                "digest": item["sourceDigest"],
                "active": invocation_id,
                "refs": refs,
                "heap": {
                    **(existing["heap"] if existing else {}),
                    **{node["id"]: node for node in item["inputGraph"]["objects"]},
                },
                "promises": existing["promises"] if existing else {},
            }
            visible_sources.setdefault(track, set()).add(program_id)
            activations[invocation_id] = {
                "program": program_id,
                "track": track,
                "realm": realm_id,
                "terminal": False,
                "returned": False,
                "revision": item["taskRevision"],
                "calls": {invocation_id},
                "promises": realms[realm_id]["promises"],
                "continuations": {},
                "stack": [invocation_id],
                "jobs": {},
                "ready": [],
                "sequence": 0,
                "callReturn": None,
                "completionPolicy": item["completionPolicy"],
                "jobErrors": [],
                "generated": 0,
                "invocation": item,
            }
        elif kind == "semantic":
            activation = activations.get(invocation_id)
            if not activation or activation["terminal"] or activation["track"] != track:
                _fail("Selected record lacks a runnable invocation on its own track")
            if activation["program"] not in visible_sources.get(track, set()):
                _fail("Evicted source requires a checked reread before execution")
            if not isinstance(item.get("line"), str) or len(item["line"].encode("utf-8")) > 65536:
                _fail("Generated semantic record exceeds the declared byte budget")
            semantic_kind, fields = _witness(item.get("line"))
            activation["generated"] += 1
            if semantic_kind in (
                "code",
                "input",
                "observe",
                "value-open",
                "value-part",
                "value-close",
            ):
                _fail("Student cannot generate source, environment or partial request")
            if semantic_kind == "request":
                if not activation["stack"] or fields.get("invocation") != activation["stack"][-1]:
                    _fail("Request belongs to an unknown dynamic invocation")
                program = programs[activation["program"]]
                if fields["operation"] not in program["capabilities"]:
                    _fail("Selected action exceeds observed grant")
                site = fields["site"]
                if not any(
                    isinstance(s, dict)
                    and s.get("start") == site["start"]
                    and s.get("end") == site["end"]
                    and s.get("operation") == fields["operation"]
                    and s.get("role") == "CallExpression"
                    for s in program["sites"]
                ):
                    _fail("External operation lacks a source-grounded callsite")
                materialized, signature = _host_request(
                    fields, heap=realms[activation["realm"]]["heap"], require_promise=True
                )
                key = (invocation_id, fields["effect"])
                if key in requests:
                    _fail("Effect identity reused in one invocation")
                call_id = f"{session}:{invocation_id}:{fields['effect']}"
                action_id = "semantic:" + call_id
                requests[key] = {
                    "id": action_id,
                    "execution": action_id + ":execution",
                    "line": item["line"],
                    "index": index,
                    "promise": fields.get("promise"),
                    "operation": fields["operation"],
                }
                if "promise" in fields:
                    if fields["promise"] in activation["promises"]:
                        _fail("Request cannot reuse an owned Promise identity")
                    activation["promises"][fields["promise"]] = {
                        "state": "pending",
                        "external": True,
                    }
                execution_id = requests[key]["execution"]
                actions.append(
                    LogicalAction(
                        action_id=action_id,
                        call_id=call_id,
                        tool_name=fields["operation"],
                        arguments_json=json.dumps(
                            {"graph": materialized}
                            if isinstance(materialized, dict)
                            else {"args": materialized},
                            separators=(",", ":"),
                            ensure_ascii=False,
                        ),
                        record_ids=("semantic:r" + str(index + 1),),
                        serialized_sha256=digest_bytes(item["line"].encode("utf-8")),
                        execution_id=execution_id,
                        schema_id=signature,
                    )
                )
            if semantic_kind == "call":
                if (
                    fields.get("invocation", invocation_id) not in activation["calls"]
                    or fields["child"] in activation["calls"]
                ):
                    _fail("Guest call must introduce a fresh owned dynamic activation")
                activation["calls"].add(fields["child"])
                activation["stack"].append(fields["child"])
            if semantic_kind == "allocate":
                refs = realms[activation["realm"]]["refs"]
                if fields["name"] in refs:
                    _fail("Allocation cannot reuse a retained object identity")
                refs.add(fields["name"])
                node = {
                    "id": fields["name"],
                    "kind": fields["kind"],
                    "properties": list(fields["properties"]),
                }
                if fields["kind"] == "array":
                    node["properties"] = [p for p in node["properties"] if p["key"] != "length"] + [
                        {"key": "length", "value": ["num", fields["length"]]}
                    ]
                realms[activation["realm"]]["heap"][fields["name"]] = node
            if semantic_kind in {"read", "write"}:
                node = realms[activation["realm"]]["heap"].get(fields["object"])
                if node is None:
                    _fail("Property access needs an owned committed identity")
                prior = next(
                    (prop for prop in node["properties"] if prop["key"] == fields["key"]), None
                )
                if semantic_kind == "read":
                    if (prior is not None) != fields["present"] or (
                        prior["value"] if prior else ["undefined"]
                    ) != fields["value"]:
                        _fail("Selected read contradicts committed property state")
                else:
                    if not fields["present"]:
                        node["properties"] = [
                            prop for prop in node["properties"] if prop["key"] != fields["key"]
                        ]
                    elif prior is not None:
                        prior["value"] = fields["value"]
                    else:
                        node["properties"].append({"key": fields["key"], "value": fields["value"]})
            if semantic_kind == "promise":
                promises = activation["promises"]
                promise_id, transition = fields["promise"], fields["action"]
                if transition == "create":
                    if promise_id in promises or promise_id in realms[activation["realm"]]["refs"]:
                        _fail("Promise identity already exists")
                    promises[promise_id] = {"state": "pending"}
                else:
                    promise = promises.get(promise_id)
                    if promise is None:
                        _fail("Promise transition lacks an owned Promise")
                    if transition in {"resolve", "reject", "adopt"}:
                        if transition == "adopt" and fields["target"] not in promises:
                            _fail("Invalid owned Promise adoption")
                        _settle(
                            activation,
                            promise_id,
                            "rejected" if transition == "reject" else "fulfilled",
                            ["ref", fields["target"]] if transition == "adopt" else fields["value"],
                        )
                    elif transition == "react":
                        identity = fields["job"]
                        if (
                            identity in activation["jobs"]
                            or identity in activation["continuations"]
                        ):
                            _fail("Owned reaction job identity already exists")
                        for key in ("closure", "rejectionClosure"):
                            if (
                                key in fields
                                and realms[activation["realm"]]["heap"]
                                .get(fields[key], {})
                                .get("kind")
                                != "closure"
                            ):
                                _fail("Promise reaction requires a committed owned closure")
                        if "target" in fields:
                            if (
                                fields["target"] in promises
                                or fields["target"] in realms[activation["realm"]]["refs"]
                            ):
                                _fail("Reaction result Promise must be fresh")
                            promises[fields["target"]] = {"state": "pending"}
                        activation["sequence"] += 1
                        activation["jobs"][identity] = {
                            **fields,
                            "state": "waiting",
                            "owner": invocation_id,
                            "sequence": activation["sequence"],
                        }
            if semantic_kind == "await":
                if (
                    fields["promise"] not in activation["promises"]
                    or fields["continuation"] in activation["continuations"]
                    or fields["continuation"] in activation["jobs"]
                    or not activation["stack"]
                    or fields["invocation"] != activation["stack"][-1]
                ):
                    _fail("Await needs the active owned invocation, Promise and new continuation")
                if "returnPromise" in fields:
                    if (
                        fields["returnPromise"] not in activation["promises"]
                        or len(activation["stack"]) != 1
                        or activation["callReturn"] is not None
                    ):
                        _fail("Immediate async call return needs the first root suspension")
                    activation["callReturn"] = ["ref", fields["returnPromise"]]
                activation["sequence"] += 1
                activation["continuations"][fields["continuation"]] = {
                    "state": "waiting",
                    "promise": fields["promise"],
                    "invocation": activation["stack"].pop(),
                    "sequence": activation["sequence"],
                }
                if not activation["stack"]:
                    realms[activation["realm"]]["active"] = None
            _refresh_ready(activation)
            if semantic_kind == "resume":
                continuation = activation["continuations"].get(fields["continuation"])
                if (
                    not continuation
                    or continuation["invocation"] != fields["invocation"]
                    or activation["stack"]
                    or not activation["ready"]
                    or activation["ready"][0] != fields["continuation"]
                ):
                    _fail("Resume violates owned FIFO order or invocation ownership")
                promise = activation["promises"][continuation["promise"]]
                if (
                    promise.get("state") != fields["status"]
                    or promise.get("result") != fields["result"]
                ):
                    _fail("Resume must use an already delivered settlement")
                del activation["continuations"][fields["continuation"]]
                activation["ready"].pop(0)
                activation["stack"].append(fields["invocation"])
                realm = realms[activation["realm"]]
                if realm["active"] not in (None, invocation_id):
                    _fail("A shared realm cannot run two guest turns at once")
                realm["active"] = invocation_id
            if semantic_kind == "job":
                identity, transition = fields["job"], fields["action"]
                if fields["owner"] != invocation_id:
                    _fail("Owned job belongs to another invocation")
                if transition == "enqueue":
                    if identity in activation["jobs"] or identity in activation["continuations"]:
                        _fail("Owned job identity already exists")
                    if (
                        "closure" in fields
                        and realms[activation["realm"]]["heap"]
                        .get(fields["closure"], {})
                        .get("kind")
                        != "closure"
                    ):
                        _fail("Enqueued job requires an owned closure")
                    activation["jobs"][identity] = {**fields, "state": "ready"}
                    activation["ready"].append(identity)
                else:
                    job = activation["jobs"].get(identity)
                    if job is None or activation["stack"]:
                        _fail("Owned job transition requires an unwound guest stack")
                    if transition == "dispatch":
                        if (
                            job["state"] != "ready"
                            or not activation["ready"]
                            or activation["ready"][0] != identity
                        ):
                            _fail("Owned job dispatch violates FIFO readiness")
                        activation["ready"].pop(0)
                        job["state"] = "running"
                        closure = job.get("closure")
                        if "promise" in job:
                            promise = activation["promises"][job["promise"]]
                            job.update(
                                originalStatus=promise["state"], originalResult=promise["result"]
                            )
                            if job["reaction"] == "then" and promise["state"] == "rejected":
                                closure = job.get("rejectionClosure")
                            elif job["reaction"] == "catch" and promise["state"] == "fulfilled":
                                closure = None
                        if closure:
                            activation["stack"].append("job:" + identity)
                            activation["calls"].add("job:" + identity)
                    else:
                        if job["state"] != "running":
                            _fail("Only a running owned job may complete")
                        job["state"] = "completed"
                        returned = job.get("returned")
                        if job.get("target"):
                            if returned and returned["kind"] == "throw":
                                _settle(activation, job["target"], "rejected", returned["value"])
                            elif (
                                job["reaction"] == "finally"
                                and returned
                                and returned["value"][0] == "ref"
                                and returned["value"][1] in activation["promises"]
                            ):
                                job.update(state="waiting-cleanup", cleanup=returned["value"][1])
                            elif job["reaction"] == "finally" or not returned:
                                _settle(
                                    activation,
                                    job["target"],
                                    job["originalStatus"],
                                    job["originalResult"],
                                )
                            else:
                                _settle(activation, job["target"], "fulfilled", returned["value"])
                        elif returned and returned["kind"] == "throw":
                            activation["jobErrors"].append(returned["value"])
            if semantic_kind in ("return", "throw"):
                owner = fields.get("invocation", invocation_id)
                if not activation["stack"] or activation["stack"][-1] != owner:
                    _fail("Return or throw does not own the active guest frame")
                activation["stack"].pop()
                if owner == invocation_id:
                    activation["returned"] = semantic_kind
                    activation["returnValue"] = fields["value"]
                elif owner.startswith("job:"):
                    activation["jobs"][owner[4:]]["returned"] = {
                        "kind": semantic_kind,
                        "value": fields["value"],
                    }
                if not activation["stack"]:
                    realms[activation["realm"]]["active"] = None
            _refresh_ready(activation)
            if semantic_kind == "end":
                expected = (
                    "throw"
                    if activation["completionPolicy"] == "owned-quiescence"
                    and activation["jobErrors"]
                    else activation["returned"]
                )
                if not expected or fields["kind"] != expected:
                    _fail("Terminal completion differs from selected return/throw")
                if (
                    activation["stack"]
                    or activation["continuations"]
                    or activation["ready"]
                    or any(job["state"] != "completed" for job in activation["jobs"].values())
                ):
                    _fail("Required owned jobs or continuations remain pending")
                if any(key[0] == invocation_id and key not in settled for key in requests):
                    _fail("Unresolved external effects preclude terminal admission")
                completion = {"kind": activation["returned"], "value": activation["returnValue"]}
                if (
                    completion["kind"] == "return"
                    and activation["completionPolicy"] != "call-return"
                    and completion["value"][0] == "ref"
                    and completion["value"][1] in activation["promises"]
                ):
                    returned_promise = activation["promises"][completion["value"][1]]
                    if returned_promise["state"] == "pending":
                        _fail("Awaited returned Promise remains pending at terminal admission")
                    completion = {
                        "kind": "return" if returned_promise["state"] == "fulfilled" else "throw",
                        "value": returned_promise["result"],
                    }
                if activation["completionPolicy"] == "owned-quiescence" and activation["jobErrors"]:
                    completion = {"kind": "throw", "value": activation["jobErrors"][0]}
                activation["completion"] = completion
                activation["terminal"] = True
                realms[activation["realm"]]["active"] = None
        elif kind == "dispatch":
            key = (invocation_id, item.get("effectId"))
            activation = activations.get(invocation_id)
            if (
                key not in requests
                or key in dispatched
                or not activation
                or activation["track"] != track
                or item.get("key") != invocation_id + "/" + item.get("effectId", "")
                or item.get("attempt") != 1
            ):
                _fail("Uncommitted, repeated or misrouted action dispatch")
            dispatched.add(key)
        elif kind == "result":
            key = (invocation_id, item.get("effectId"))
            activation = activations.get(invocation_id)
            if (
                key not in dispatched
                or key in settled
                or not activation
                or activation["track"] != track
                or (activation["terminal"] and key not in reconciliations)
            ):
                _fail("Duplicate or wrong-owner tool result")
            observed_kind, fields = _witness(item.get("line"))
            if (
                observed_kind != "observe"
                or fields["effect"] != item.get("effectId")
                or fields["status"] != item.get("status")
                or fields["result"] != item.get("result")
            ):
                _fail("Observed witness and broker receipt disagree")
            if (
                key in reconciliations
                and reconciliations[key] in {"fulfilled", "rejected"}
                and fields["status"] != reconciliations[key]
            ):
                _fail("Reconciled result contradicts independent disposition evidence")
            descriptor = HOST_OPERATIONS[requests[key]["operation"]]
            if ("graph" in fields or fields["result"][0] == "ref") and descriptor[
                "resultCodec"
            ] != "owned-graph/1":
                _fail("Observed graph requires the operation's declared owned result codec")
            settled.add(key)
            if "graph" in fields:
                realm = realms[activation["realm"]]
                for node in fields["graph"]["objects"]:
                    if node["id"] in realm["refs"]:
                        _fail("Observed resource graph cannot replace an existing owned identity")
                    realm["refs"].add(node["id"])
                    realm["heap"][node["id"]] = node
            promise_id = requests[key].get("promise")
            if promise_id is not None:
                promise = activation["promises"][promise_id]
                if promise["state"] != "pending":
                    _fail("Native result cannot settle an already completed Promise")
                _settle(activation, promise_id, fields["status"], fields["result"], external=True)
                _refresh_ready(activation)
            result_of = requests[key]["id"]
            execution_id = requests[key]["execution"]

        record_id = "semantic:r" + str(index + 1)
        role = (
            "assistant"
            if kind == "semantic"
            else ("user" if kind in ("program", "invocation") else "environment")
        )
        payload = (
            item["line"].encode("utf-8")
            if kind == "semantic"
            else json.dumps(
                item, ensure_ascii=False, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
        )
        records.append(
            PhysicalRecord(
                record_id,
                role,
                payload,
                turn_index=index,
                policy_mask=kind == "semantic",
                provenance=selected_provenance if kind == "semantic" else "executed",
                action_id=action_id,
                result_of=result_of,
                execution_id=execution_id if result_of else None,
            )
        )
        meridians.append(Meridian(index, (TrackEvent(record_id, track, "text", index, role),)))
        previous = last_by_track.get(track)
        if previous is not None and kind == "semantic":
            dependencies.append(Dependency(previous, record_id, "semantic"))
        # Result->request correlation lives in PhysicalRecord.result_of and
        # the logical action table. Dependency edges are prediction readers,
        # never a future *observed* result pretending to be a prediction.
        last_by_track[track] = record_id

    if pending_event is not None:
        _fail("Outside event is missing its handler activation")
    if not programs or not activations or not all(v["terminal"] for v in activations.values()):
        _fail("Complete selected invocations are required for corpus import")
    if dispatched - settled != uncertain:
        _fail("No accepted dispatch may disappear from public evidence")
    if admission == "admitted" and uncertain:
        _fail("Unreconciled external effects cannot be admitted as verified outcomes")
    family = sidecar.get("sourceLineage")
    if not isinstance(family, str) or not family:
        family = hashlib.sha256(
            "".join(p["digest"] for p in programs.values()).encode()
        ).hexdigest()[:20]
    digest = hashlib.sha256(b"\n".join(record.payload for record in records)).hexdigest()
    split = sidecar.get("split")
    if split not in (None, "train", "validation", "test", "quarantine"):
        _fail("Unknown frozen corpus split")
    if not isinstance(sidecar.get("ancestorIds", []), list) or any(
        not isinstance(identity, str) or not identity for identity in sidecar.get("ancestorIds", [])
    ):
        _fail("Private ancestry must retain explicit nonempty family identities")
    lineage = Lineage(
        source_id="semantic-harness:" + digest[:16],
        revision=digest,
        family_id=family,
        ancestor_ids=tuple(sidecar.get("ancestorIds", ())),
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
                "semantic_execution": {
                    "schema": "semantic-execution/4",
                    "representation": sidecar.get("representation"),
                    "continuation": "teacher_conditioned"
                    if selected_provenance == "teacher"
                    else "recorded",
                    "admission": admission,
                },
                "semantic_harness": {
                    "controlled_selections": selected_provenance == "unknown",
                    "admission": admission,
                    "traceFormat": "semantic-trace/4",
                },
            },
            separators=(",", ":"),
        ),
    )
    episode = Episode(
        "semantic:harness:" + digest,
        tuple(records),
        lineage,
        stream="semantic_execution",
        actions=tuple(actions),
        supervision=supervision,
        dependencies=tuple(dependencies),
        tracks=tuple(TrackSpec(name, "assistant") for name in tracks),
        meridians=tuple(meridians),
    )
    try:
        episode = physical_semantic_episode(
            episode, codec=codec, max_positions=max_positions, max_history=max_history
        )
    except ValueError as error:
        raise SelectedHarnessError(str(error)) from error
    return episode, SelectedHarnessReceipt(
        records=len(episode.records),
        invocations=len(activations),
        tracks=len(tracks),
        actions=len(actions),
        observations=observed,
        admission=admission,
        codec_fingerprint=getattr(codec, "fingerprint", None),
        completions=tuple(
            {"invocationId": identity, **activation["completion"]}
            for identity, activation in activations.items()
            if activation.get("completion")
        ),
    )
