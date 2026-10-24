"""Versioned, fail-closed tracing-js-self semantic execution adapter.

The public JSON groups are *training targets*, not a prompt. A serving controller
must expose only the observed context and previously committed generated groups.
The private sidecar never becomes a PhysicalRecord, control or model metadata.
This module has no torch dependency.
"""

from __future__ import annotations

import hashlib
import json
import re
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
from .witness_v3 import WitnessV3Error
from .witness_v3 import parse_record as parse_v3_record

_PUBLIC = "semantic-execution/3"
_TRACE = "semantic-trace/3"
_LANE = re.compile(r"t[0-7]\Z")


class SemanticExecutionError(ValueError):
    """An execution export cannot enter the training corpus."""


@dataclass(frozen=True)
class SemanticExecutionReceipt:
    groups: int
    generated: int
    observed: int
    actions: int
    dependencies: int
    codec_fingerprint: str | None
    admission: str


def _fail(message: str) -> None:
    raise SemanticExecutionError(message)


def _record(line: Any) -> tuple[str, dict[str, Any]]:
    try:
        return parse_v3_record(line)
    except WitnessV3Error as error:
        raise SemanticExecutionError(str(error)) from error


def _text_bytes(text: str) -> bytes:
    if not isinstance(text, str):
        _fail("Semantic transport requires a string record")
    try:
        return text.encode("utf-8", errors="strict")
    except UnicodeEncodeError as error:
        raise SemanticExecutionError("Unescaped surrogate in transport record") from error


def _host_request(fields: dict[str, Any]) -> tuple[list[Any], str]:
    """Materialize bounded semantic arguments, never hidden JS getters."""
    operation = fields.get("operation")
    signatures = {
        "demo.readNumber": ("demo.readNumber/1", 1),
        "demo.readAsync": ("demo.readAsync/1", 1),
        "demo.store": ("demo.store/1", 2),
        "platform.dateNow": ("platform.dateNow/1", 0),
        "console.log": ("console.log/1", None),
        "console.warn": ("console.warn/1", None),
        "console.error": ("console.error/1", None),
    }
    if operation not in signatures:
        _fail("Unregistered v3 effect capability")
    signature, count = signatures[operation]
    supplied = fields.get("arguments")
    if not isinstance(supplied, list) or (count is not None and len(supplied) != count):
        _fail("Invalid v3 effect argument arity")
    values = []
    for value in supplied:
        if not isinstance(value, list) or not value:
            _fail("Invalid external scalar")
        kind = value[0]
        if kind == "str" and len(value) == 2 and isinstance(value[1], str):
            values.append(value[1])
        elif kind == "num" and len(value) == 2 and type(value[1]) in (int, float):
            values.append(value[1])
        elif kind == "bool" and len(value) == 2 and type(value[1]) is bool:
            values.append(value[1])
        elif kind == "null" and len(value) == 1:
            values.append(None)
        else:
            _fail("Unsupported v3 external transport scalar")
    if operation in ("demo.store", "demo.readNumber", "demo.readAsync") and not isinstance(values[0], str):
        _fail("Invalid version-three external key")
    return values, signature

def _validate(
    public: Mapping[str, Any],
    *,
    codec: Any | None,
    max_positions: int,
    max_history: int,
) -> tuple[list[dict[str, Any]], list[tuple[str, dict[str, Any]]], dict[str, int]]:
    if public.get("schema") != _PUBLIC or public.get("traceFormat") != _TRACE:
        _fail("Only semantic-execution/3 and semantic-trace/3 are supported")
    if public.get("profile") != "training-js/1" or public.get("contextFormat") != "semantic-context/3":
        _fail("Unsupported source or observed context contract")
    if public.get("completionPolicy") != "immediate-root":
        _fail("Unsupported declared root completion policy")
    groups = public.get("groups")
    if not isinstance(groups, list) or len(groups) < 4:
        _fail("Execution needs observed source/input and generated continuation")
    if len(groups) > max_history:
        _fail("Episode exceeds model history; continuation segmentation is required")
    prefix: list[tuple[str, dict[str, Any]]] = []
    observed = generated = 0
    actions: dict[str, int] = {}
    deliveries: set[str] = set()
    opened: tuple[str, str, str] | None = None
    fragment_parts: list[str] = []
    phase = 0  # 0 = source, 1 = input, 2 = semantic continuation
    completed_continuation = False
    last_completion: str | None = None
    for index, group in enumerate(groups):
        if not isinstance(group, dict) or group.get("id") != "m" + str(index + 1):
            _fail("Meridian identities must be consecutive and canonical")
        mode = group.get("mode")
        if mode not in ("observed", "generated") or group.get("loss") != (
            "none" if mode == "observed" else "semantic"
        ):
            _fail("Mixed or incorrectly scored execution meridian")
        events = group.get("events")
        if not isinstance(events, list) or len(events) != 1:
            _fail("Serial training-js/1 exporter expects one event per meridian")
        event = events[0]
        if not isinstance(event, dict) or not isinstance(event.get("track"), str):
            _fail("Invalid meridian member")
        if not _LANE.fullmatch(event["track"]):
            _fail("Track outside the fixed t0..t7 registry")
        payload = _text_bytes(event.get("record"))
        if len(payload) > 1024:
            _fail("Trace exceeds the declared physical byte cap")
        if codec is not None and len(codec.encode(payload)) > max_positions:
            _fail("Trace exceeds the pinned raw-codec output-position limit")
        kind, fields = _record(event["record"])
        if completed_continuation:
            _fail("Semantics follow the terminal end record")
        if opened is not None:
            tag, owner, opening_mode = opened
            if event["track"] != owner or mode != opening_mode:
                _fail("Fragment changes lane or observed/generated mode")
            if fields.get("id") != tag:
                _fail("Value-fragment identity changed before close")
            if kind == "value-part":
                if not isinstance(fields.get("data"), str):
                    _fail("Fragment payload must be escaped text")
                fragment_parts.append(fields["data"])
                continue
            if kind != "value-close":
                _fail("Fragment was interrupted before its close marker")
            completed_record = "".join(fragment_parts)
            complete_kind, complete_fields = _record(completed_record)
            opened = None
            fragment_parts = []
            kind, fields = complete_kind, complete_fields
        elif kind == "value-open":
            if not isinstance(fields.get("id"), str):
                _fail("Missing deterministic fragment identity")
            opened = (fields["id"], event["track"], mode)
            fragment_parts = []
            continue
        elif kind in ("value-part", "value-close"):
            _fail("Fragment without preceding value-open")
        if phase < 2:
            if mode != "observed" or event["track"] != "t0":
                _fail("Source and initial graph must be observed on the root lane")
            if kind != ("code" if phase == 0 else "input"):
                _fail("Expected source code followed by the initial graph")
            if phase == 0:
                text = fields.get("source")
                if not isinstance(text, str) or not text:
                    _fail("Missing exact source bytes")
            prefix.append((kind, fields))
            phase += 1
            observed += 1
            continue
        if mode == "observed":
            if kind not in ("observe", "code"):
                _fail("Nonenvironment operation cannot be observed")
            observed += 1
        else:
            if kind in ("observe", "code", "input"):
                _fail("Environment observations cannot be model-generated")
            generated += 1
        if kind in ("return", "throw"):
            last_completion = kind
        if kind == "end":
            if mode != "generated" or fields.get("kind") not in ("return", "throw"):
                _fail("Invalid or incomplete declared root completion")
            if fields["kind"] != last_completion:
                _fail("Version-three end lacks a selected matching root completion")
            completed_continuation = True
        if kind == "request":
            fx = fields.get("effect")
            if not isinstance(fx, str) or not fx or fx in actions:
                _fail("Duplicate or missing effect identity")
            _host_request(fields)
            actions[fx] = index
        if kind == "observe":
            fx = fields.get("effect")
            if not isinstance(fx, str) or fx not in actions or actions[fx] >= index:
                _fail("Tool observation has no committed matching request")
            if fx in deliveries:
                _fail("Duplicate external result delivery")
            deliveries.add(fx)
    if opened or phase != 2 or not completed_continuation or not generated:
        _fail("Unclosed context, fragmented value or incomplete execution")
    return groups, prefix, {"generated": generated, "observed": observed, "actions": len(actions)}


def adapt_semantic_execution(
    export: Mapping[str, Any],
    *,
    codec: Any | None = None,
    max_positions: int = 1024,
    max_history: int = 1024,
    allow_local: bool = False,
    admission_verifier: Callable[[Mapping[str, Any]], bool] | None = None,
) -> tuple[Episode, SemanticExecutionReceipt]:
    """Validate a tracer export and convert it to native causal Episode records.

    An export labelled merely locally_verified is a diagnostic. Training consumers
    must opt in explicitly; a successful schema conversion does NOT certify
    independent JavaScript conformance or free-running model correctness.
    """
    if not isinstance(export, Mapping) or not isinstance(export.get("public"), Mapping):
        _fail("A semantic execution export needs an explicit public payload")
    sidecar = export.get("sidecar")
    if not isinstance(sidecar, Mapping) or sidecar.get("schema") != "semantic-supervision/2":
        _fail("Only semantic-supervision/2 can accompany version-three episodes")
    status = sidecar.get("admission", "locally_verified")
    if status != "admitted" and not allow_local:
        _fail("Unadmitted execution requires allow_local=True for diagnostic conversion")
    if status == "admitted":
        if sidecar.get("referenceStatus") != "agreed":
            _fail("Admitted execution lacks independent outcome agreement")
        if admission_verifier is None or not admission_verifier(export):
            _fail("Admitted execution requires an independent external admission verifier")
    public = export["public"]
    groups, prefix, counts = _validate(
        public, codec=codec, max_positions=max_positions, max_history=max_history
    )
    source = prefix[0][1]["source"]
    family = sidecar.get("sourceLineage")
    if not isinstance(family, str) or not family:
        family = hashlib.sha256(_text_bytes(source)).hexdigest()[:20]
    revision = hashlib.sha256(_text_bytes(source)).hexdigest()
    split = sidecar.get("split")
    if split not in (None, "train", "validation", "test", "quarantine"):
        _fail("Unknown frozen corpus family split")
    lineage = Lineage(
        source_id="tracing-js-self:" + revision[:16],
        revision=revision,
        family_id=family,
        source_uri="https://github.com/werg/tracing-js-self",
        frozen_split=split,
    )
    # Reassemble *logical* semantic operations while retaining one physical
    # model event/meridian for each fragment. A request only becomes executable
    # when its final value-close commits. Source and input fragments remain user
    # observations regardless of their absolute physical meridian index.
    spans: list[tuple[int, int, str, dict[str, Any]]] = []
    opened_at: int | None = None
    fragments: list[str] = []
    for index, group in enumerate(groups):
        kind, fields = _record(group["events"][0]["record"])
        if kind == "value-open":
            opened_at = index
            fragments = []
        elif kind == "value-part":
            fragments.append(fields["data"])
        elif kind == "value-close":
            if opened_at is None:
                _fail("Fragment close has no opening event")
            resolved_kind, resolved_fields = _record("".join(fragments))
            spans.append((opened_at, index, resolved_kind, resolved_fields))
            opened_at = None
            fragments = []
        else:
            spans.append((index, index, kind, fields))
    if opened_at is not None:
        _fail("Unclosed physical request or source fragment")

    records: list[PhysicalRecord] = []
    meridians: list[Meridian] = []
    dependencies: list[Dependency] = []
    actions: list[LogicalAction] = []
    dispatched: dict[str, tuple[str, str]] = {}
    previous_committed_record: str | None = None

    for ordinal, (start, end, kind, fields) in enumerate(spans):
        mode = groups[start]["mode"]
        role = "assistant" if mode == "generated" else ("user" if ordinal < 2 else "environment")
        action_id: str | None = None
        result_of: str | None = None
        execution_id: str | None = None
        record_ids = tuple("semantic:r" + str(i + 1) for i in range(start, end + 1))
        if kind == "request":
            effect = fields["effect"]
            action_id = "semantic:" + effect
            execution_id = action_id + ":execution"
            dispatched[effect] = (action_id, execution_id)
            request_bytes = b"".join(
                _text_bytes(groups[i]["events"][0]["record"]) for i in range(start, end + 1)
            )
            host_arguments, signature = _host_request(fields)
            actions.append(
                LogicalAction(
                    action_id=action_id,
                    call_id=effect,
                    tool_name=fields["operation"],
                    arguments_json=json.dumps(
                        {"args": host_arguments}, separators=(",", ":"), ensure_ascii=True
                    ),
                    record_ids=record_ids,
                    serialized_sha256=digest_bytes(request_bytes),
                    execution_id=execution_id,
                    schema_id=signature,
                )
            )
        if kind == "observe":
            result_of, execution_id = dispatched[fields["effect"]]

        for i in range(start, end + 1):
            item = groups[i]["events"][0]
            name = record_ids[i - start]
            record = PhysicalRecord(
                record_id=name,
                role=role,
                payload=_text_bytes(item["record"]),
                turn_index=i,
                logical_id="semantic:logical:" + str(start + 1) if end > start else "",
                part_index=i - start,
                part_count=end - start + 1,
                policy_mask=mode == "generated",
                provenance="teacher" if mode == "generated" else "executed",
                action_id=action_id,
                result_of=result_of,
                execution_id=execution_id if result_of else None,
            )
            records.append(record)
            meridians.append(Meridian(i, (TrackEvent(name, item["track"], "text", i, role),)))

        # All value dependencies name the closing fragment, never a partially
        # materialized JSON request, array, or function return. Observations
        # are evidence, not policy-scored dependent predictions.
        name = record_ids[-1]
        # Native recurrence can only consume completed prior logical work.
        # The edge is derived from causal commit order and observed boundaries,
        # not from a private teacher operation or future source result.
        if previous_committed_record is not None and mode == "generated":
            dependencies.append(Dependency(previous_committed_record, name, "semantic"))
        previous_committed_record = name
    tracks = tuple(TrackSpec("t" + str(i), "assistant") for i in range(8))
    episode_id = (
        "semantic:" + hashlib.sha256(b"\n".join(record.payload for record in records)).hexdigest()
    )
    supervision = Supervision(
        outcome="success" if status == "admitted" else "unknown",
        verification="external_tests" if status == "admitted" else "none",
        annotations_json=json.dumps(
            {"semantic_execution": {"continuation": "teacher_conditioned", "admission": status}},
            separators=(",", ":"),
        ),
    )
    episode = Episode(
        episode_id,
        tuple(records),
        lineage,
        stream="semantic_execution",
        actions=tuple(actions),
        supervision=supervision,
        dependencies=tuple(dependencies),
        tracks=tracks,
        meridians=tuple(meridians),
    )
    return episode, SemanticExecutionReceipt(
        groups=len(groups),
        generated=counts["generated"],
        observed=counts["observed"],
        actions=counts["actions"],
        dependencies=len(dependencies),
        codec_fingerprint=getattr(codec, "fingerprint", None),
        admission=status,
    )


