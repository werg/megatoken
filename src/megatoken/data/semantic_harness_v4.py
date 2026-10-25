"""Independent native consumer for the explicit semantic-execution/4 harness envelope.

This is a *transitional* selected-state integration: semantic records remain
semantic-trace/3 until both validators and the executor migrate to trace/4.
It imports real validated requests and observed results, never teacher sidecars,
and uses discovered dynamic track addresses rather than eight fixed lanes.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

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
from .semantic_execution import _host_request
from .witness_v3 import WitnessV3Error, parse_record
from .witness_v3 import graph as validate_graph


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
    except WitnessV3Error as error:
        raise SelectedHarnessError(str(error)) from error


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
        or public.get("traceFormat") != "semantic-trace/3"
        or public.get("profile") != "selected-harness/1"
        or sidecar.get("schema") != "semantic-supervision/2"
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
    if not isinstance(timeline, list) or not 2 <= len(timeline) <= max_history:
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
        ):
            if mode != "observed":
                _fail("A host observation cannot be generated by the student")
            observed += 1
        elif kind == "semantic":
            if mode != "generated" or item.get("traceFormat") != "semantic-trace/3":
                _fail("Only validated witness selections carry model loss")
        else:
            _fail("Unregistered harness envelope kind")
        invocation_id = item.get("invocationId")
        if pending_event is not None and kind != "invocation":
            _fail("Delivered outside event must be followed by its observed invocation")
        action_id = None
        result_of = None
        execution_id = None
        if kind == "program":
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
                or item.get("entry") != programs[program_id]["entry"]
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
                "policy": item["realmPolicy"],
                "seen": set(),
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
                or item.get("entry") != programs[program_id]["entry"]
                or item.get("sourceDigest") != programs[program_id]["digest"]
                or item.get("completionPolicy") not in ("immediate-root", "owned-quiescence")
            ):
                _fail("Invocation needs prepublished code, realm, trigger and fresh identity")
            try:
                validate_graph(item.get("inputGraph"))
            except WitnessV3Error as error:
                raise SelectedHarnessError(str(error)) from error
            policy = item.get("realmPolicy", "fresh")
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
            if pending_event is not None:
                if (
                    pending_event["invocationId"] != invocation_id
                    or pending_event["track"] != track
                    or pending_event["triggerId"] != item["triggerId"]
                    or pending_event["programId"] != program_id
                    or pending_event["realmId"] != realm_id
                ):
                    _fail("Outside event was not delivered to its bound handler invocation")
                pending_event = None
            realms[realm_id] = {
                "digest": item["sourceDigest"],
                "active": invocation_id,
            }
            activations[invocation_id] = {
                "program": program_id,
                "track": track,
                "realm": realm_id,
                "terminal": False,
                "returned": False,
            }
        elif kind == "semantic":
            activation = activations.get(invocation_id)
            if not activation or activation["terminal"] or activation["track"] != track:
                _fail("Selected record lacks a runnable invocation on its own track")
            semantic_kind, fields = _witness(item.get("line"))
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
                if fields.get("invocation") != "i1":
                    _fail("Legacy selected-controller invocation alias must be explicit")
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
                _host_request(fields)
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
                }
                execution_id = requests[key]["execution"]
                actions.append(
                    LogicalAction(
                        action_id=action_id,
                        call_id=call_id,
                        tool_name=fields["operation"],
                        arguments_json=json.dumps(
                            {"args": json.loads(fields["payload"])},
                            separators=(",", ":"),
                            ensure_ascii=False,
                        ),
                        record_ids=("semantic:r" + str(index + 1),),
                        serialized_sha256=digest_bytes(item["line"].encode("utf-8")),
                        execution_id=execution_id,
                        schema_id=_host_request(fields)[1],
                    )
                )
            if semantic_kind in ("return", "throw"):
                activation["returned"] = semantic_kind
            if semantic_kind == "end":
                if not activation["returned"] or fields["kind"] != activation["returned"]:
                    _fail("Terminal completion differs from selected return/throw")
                if any(key[0] == invocation_id and key not in settled for key in requests):
                    _fail("Unresolved external effects preclude terminal admission")
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
                or activation["terminal"]
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
            settled.add(key)
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
        if codec is not None and len(codec.encode(payload)) > max_positions:
            _fail("Native tokenizer budget exceeded by physical observation/selection")
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
    if dispatched != settled:
        _fail("No accepted dispatch may disappear from public evidence")
    family = sidecar.get("sourceLineage")
    if not isinstance(family, str) or not family:
        family = hashlib.sha256(
            "".join(p["digest"] for p in programs.values()).encode()
        ).hexdigest()[:20]
    digest = hashlib.sha256(b"\n".join(record.payload for record in records)).hexdigest()
    split = sidecar.get("split")
    if split not in (None, "train", "validation", "test", "quarantine"):
        _fail("Unknown frozen corpus split")
    lineage = Lineage(
        source_id="semantic-harness:" + digest[:16],
        revision=digest,
        family_id=family,
        source_uri="https://github.com/werg/tracing-js-self",
        frozen_split=split,
    )
    supervision = Supervision(
        outcome="success" if admission == "admitted" else "unknown",
        verification="external_tests" if admission == "admitted" else "none",
        annotations_json=json.dumps(
            {
                "semantic_harness": {
                    "controlled_selections": True,
                    "admission": admission,
                    "traceFormat": "semantic-trace/3",
                }
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
    return episode, SelectedHarnessReceipt(
        records=len(records),
        invocations=len(activations),
        tracks=len(tracks),
        actions=len(actions),
        observations=observed,
        admission=admission,
        codec_fingerprint=getattr(codec, "fingerprint", None),
    )
