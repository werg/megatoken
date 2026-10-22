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

_PUBLIC = "semantic-execution/1"
_TRACE = "semantic-trace/1"
_LANE = re.compile(r"t[0-7]\Z")
_ID = re.compile(r"e[1-9]\d*\Z")
_WORDS = frozenset(
    (
        "enter",
        "value",
        "branch",
        "loop",
        "alloc",
        "read",
        "write",
        "delete",
        "call",
        "return",
        "throw",
        "await",
        "resume",
        "settle",
        "request",
        "observe",
        "code",
        "end",
        "value-open",
        "value-part",
        "value-close",
    )
)


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
    if not isinstance(line, str) or "\n" in line or "\r" in line:
        _fail("Trace record is not a single UTF-8 line")
    head, separator, body = line.partition(" ")
    if not separator or head not in _WORDS:
        _fail("Unknown or malformed semantic-trace/1 record")
    try:
        fields = json.loads(body)
    except (ValueError, TypeError) as error:
        raise SemanticExecutionError("Malformed trace JSON") from error
    if not isinstance(fields, dict) or not all(isinstance(x, str) for x in fields):
        _fail("Trace fields must have string keys")

    # Reject duplicate JSON keys as well as nonstandard NaN/Infinity literals.
    def strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in pairs:
            if key in output:
                _fail("Duplicate trace JSON key")
            output[key] = value
        return output

    try:
        checked = json.loads(
            body,
            object_pairs_hook=strict_pairs,
            parse_constant=lambda value: _fail("Nonfinite JSON literal: " + value),
        )
    except (ValueError, TypeError) as error:
        raise SemanticExecutionError("Invalid strict trace JSON") from error
    if checked != fields:
        _fail("Inconsistent trace JSON")
    return head, fields


def _text_bytes(text: str) -> bytes:
    if not isinstance(text, str):
        _fail("Semantic transport requires a string record")
    try:
        return text.encode("utf-8", errors="strict")
    except UnicodeEncodeError as error:
        raise SemanticExecutionError("Unescaped surrogate in transport record") from error


def _validate(
    public: Mapping[str, Any],
    *,
    codec: Any | None,
    max_positions: int,
    max_history: int,
) -> tuple[list[dict[str, Any]], list[tuple[str, dict[str, Any]]], dict[str, int]]:
    if public.get("schema") != _PUBLIC or public.get("traceFormat") != _TRACE:
        _fail("Unknown execution or semantic trace format")
    if (
        public.get("profile") != "training-js/1"
        or public.get("contextFormat") != "semantic-context/1"
    ):
        _fail("Unsupported JavaScript execution profile")
    if public.get("completionPolicy") != "immediate-return":
        _fail("Completion policy not independently implemented")
    groups = public.get("groups")
    if not isinstance(groups, list) or len(groups) < 4:
        _fail("Execution needs observed source/input and generated continuation")
    if len(groups) > max_history:
        _fail("Episode exceeds model history; continuation segmentation is required")
    prefix: list[tuple[str, dict[str, Any]]] = []
    observed = generated = 0
    completed: dict[str, int] = {}
    actions: dict[str, int] = {}
    deliveries: set[str] = set()
    opened: tuple[str, str, str] | None = None
    fragment_parts: list[str] = []
    phase = 0  # 0 = source, 1 = input, 2 = semantic continuation
    completed_continuation = False
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
            if kind != ("code" if phase == 0 else "value"):
                _fail("Expected source code followed by the initial graph")
            if phase == 0:
                if not isinstance(fields.get("text"), str) or not fields["text"]:
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
            if kind in ("observe", "code"):
                _fail("Environment observations cannot be model-generated")
            generated += 1
        if kind == "end":
            if mode != "generated" or fields.get("kind") not in ("return", "throw"):
                _fail("Invalid or incomplete declared root completion")
            completed_continuation = True
        event_id = fields.get("id")
        if event_id is not None:
            if not isinstance(event_id, str) or not _ID.fullmatch(event_id):
                _fail("Noncanonical semantic event ID")
            if event_id in completed:
                _fail("Duplicate semantic producer ID")
        for prior in (
            *fields.get("dependsOn", ()),
            *((fields.get("control"),) if fields.get("control") else ()),
        ):
            if not isinstance(prior, str) or prior not in completed or completed[prior] >= index:
                _fail("Missing or same-meridian/future semantic producer")
        if kind == "request":
            fx = fields.get("effect")
            if not isinstance(fx, str) or not fx or fx in actions:
                _fail("Duplicate or missing effect identity")
            if not isinstance(fields.get("operation"), str) or not isinstance(
                fields.get("arguments"), list
            ):
                _fail("A request requires a declared operation and materialized arguments")
            actions[fx] = index
        if kind == "observe":
            fx = fields.get("effect")
            if not isinstance(fx, str) or fx not in actions or actions[fx] >= index:
                _fail("Tool observation has no committed matching request")
            if fx in deliveries:
                _fail("Duplicate external result delivery")
            deliveries.add(fx)
        if event_id is not None:
            completed[event_id] = index
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
    if not isinstance(sidecar, Mapping) or sidecar.get("schema") != "semantic-supervision/1":
        _fail("A supervision sidecar is required for dataset admission")
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
    source = prefix[0][1]["text"]
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
        repository="werg/tracing-js-self",
        frozen_split=split,
    )
    records: list[PhysicalRecord] = []
    meridians: list[Meridian] = []
    producers: dict[str, str] = {}
    dependencies: list[Dependency] = []
    actions: list[LogicalAction] = []
    dispatched: dict[str, tuple[str, str]] = {}
    open_id: str | None = None
    open_parts: list[str] = []
    for index, group in enumerate(groups):
        (item,) = group["events"]
        line = item["record"]
        kind, fields = _record(line)
        mode = group["mode"]
        name = "semantic:r" + str(index + 1)
        action_id: str | None = None
        result_of: str | None = None
        execution_id: str | None = None
        if kind == "request":
            effect = fields["effect"]
            action_id = "semantic:" + effect
            execution_id = action_id + ":execution"
            dispatched[effect] = (action_id, execution_id)
            actions.append(
                LogicalAction(
                    action_id=action_id,
                    call_id=effect,
                    tool_name=fields["operation"],
                    arguments_json=json.dumps({"args": fields["arguments"]}, separators=(",", ":")),
                    record_ids=(name,),
                    serialized_sha256=digest_bytes(_text_bytes(line)),
                    execution_id=execution_id,
                )
            )
        if kind == "observe":
            result_of, execution_id = dispatched[fields["effect"]]
        role = "assistant" if mode == "generated" else ("environment" if index >= 2 else "user")
        record = PhysicalRecord(
            record_id=name,
            role=role,
            payload=_text_bytes(line),
            turn_index=index,
            policy_mask=mode == "generated",
            provenance="teacher" if mode == "generated" else "executed",
            action_id=action_id,
            result_of=result_of,
            execution_id=execution_id if result_of else None,
        )
        records.append(record)
        meridians.append(Meridian(index, (TrackEvent(name, item["track"], "text", index, role),)))
        if kind == "value-open":
            open_id = fields["id"]
            open_parts = []
        elif kind == "value-part":
            if open_id is None:
                _fail("Unexpected value-part while mapping dependencies")
            open_parts.append(fields["data"])
        elif kind == "value-close":
            if open_id is None:
                _fail("Unexpected value-close while mapping dependencies")
            restored_kind, restored_fields = _record("".join(open_parts))
            _add_dependencies(restored_kind, restored_fields, name, producers, dependencies)
            producers[open_id] = name
            open_id = None
        elif kind not in ("code",):
            _add_dependencies(kind, fields, name, producers, dependencies)
            identifier = fields.get("id")
            if isinstance(identifier, str):
                producers[identifier] = name
    if open_id is not None:
        _fail("Unfinished physical semantic value")
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


def _add_dependencies(
    kind: str,
    fields: dict[str, Any],
    reader: str,
    producers: dict[str, str],
    dependencies: list[Dependency],
) -> None:
    if kind in ("code", "value-part", "value-open"):
        return
    earlier = fields.get("dependsOn", [])
    if not isinstance(earlier, list) or not all(isinstance(x, str) for x in earlier):
        _fail("Malformed semantic producer dependencies")
    control = fields.get("control")
    if control is not None:
        earlier = [*earlier, control]
    for dependency in dict.fromkeys(earlier):
        writer = producers.get(dependency)
        if writer is None or writer == reader:
            _fail("Unresolved or future writer in execution dependency")
        dependencies.append(Dependency(writer, reader, "semantic"))
