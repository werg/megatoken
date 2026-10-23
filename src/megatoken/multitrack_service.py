"""Stateful JSONL serving for joint text and media meridians.

The protocol of the one serving runtime (``MultiTrackRuntime``): the ``serve`` command
speaks it on standard input and output, and ``generate`` is a one-shot session of it.
Transport media as explicitly shaped little-endian FP32 buffers. The trained
encoder/realizer weights are loaded from one served Megatoken bundle. No tool
command is executed by the server; execution remains an external capability.

Observed events carry their ``role``; a finished ``user`` message or ``assistant``
turn, or a ``tool``, ``document`` or ``environment`` result, makes a memory lookup
due, as in training. ``open`` and ``observe`` take the ``effort`` their lookups run at
(default 6); a generation's lookup uses the generation's own effort. An event or output
may carry ``terminal: false`` when it is only part of a longer message; only a message's
last part closes a turn. A text event is one physical chunk and is kept whole;
``observe_text`` (``track``, ``role``, ``text`` or ``payload_b64``, and the optional
``content_kind``, ``language``, ``filename``, ``mime_type``, ``locale`` hints) cuts a
message into chunks by the bundle's chunk policy and ends it with the last one.

A text output takes ``max_tokens``, ``min_tokens``, ``kind`` (``bytes``, ``utf8`` or
``ascii``), ``terminal`` and public ``tools`` (or ``tools_json``, with ``wire_formats``):
its tool call may copy from the text the session already holds. The causal control of
an output is ``control_seed``, ``control_reference`` with its ``reference_origin``, and
its tools, or the raw bytes ``control_b64`` (the tools then only describe the output).
A text answer carries ``text`` when its bytes are valid UTF-8.

Knowledge bases (only when served with a ``kb_root`` directory, which every path
must lie inside): ``kb_open`` mounts a base file built for this model under a name,
``kb_create`` starts an empty base a user can fill, ``kb_save`` writes a base to a file,
``kb_close`` forgets one, ``kb_list`` describes the open ones. ``open`` takes ``bases``, a
map of base name to blend weight (0 leaves one out), and ``trained_weight``.
With ``kb_root``, ``open(scope="user-or-workspace")`` automatically creates or reopens
private memory and enables retrospective writes. ``memory_writes=false`` disables it;
surprise writes require explicit ``write_policy.surprise=true``. An open response names
its automatic store as ``overlay`` so subsequent management operations can address it.

Source collections are mounted by name (``collections``). ``open`` may name the ones
a session searches (default: all), ``max_sources`` and ``threshold``; every answer
that ran lookups lists the items they retrieved and observed (``retrieved``), each
with the original file it came from.

``add_files`` prepares files (``paths``) once and adds them to a mounted source
``collection`` and/or an open writable knowledge ``base`` (with optional ``metadata``
filing its entries), when the service was given a preparer. Sessions that mount either
find the new items from their next lookup.
"""

from __future__ import annotations

import base64
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, TextIO

import numpy as np
import torch

from megatoken.contracts import ROLE_TO_ID
from megatoken.controls import build_control_payload
from megatoken.generation.structure import ActionSpec
from megatoken.io import strict_json_loads
from megatoken.model.media import MediaOutputSpec
from megatoken.model.multitrack import ObservedEvent
from megatoken.multitrack_runtime import MultiTrackRuntime, TrackSession
from megatoken.tool_schema import compile_tool_spec, recent_copy_sources


def _bytes(value: str) -> bytes:
    if not isinstance(value, str):
        raise TypeError("Base64 payload must be text")
    return base64.b64decode(value, validate=True)


def _float_tensor(
    payload: str,
    shape: object,
    device: torch.device,
    *,
    max_elements: int = 8_000_000,
) -> torch.Tensor:
    if (
        not isinstance(shape, list)
        or not shape
        or any(not isinstance(n, int) or isinstance(n, bool) or n < 1 for n in shape)
    ):
        raise ValueError("Media data requires a nonempty integer shape")
    size = math.prod(shape)
    if size > max_elements:
        raise ValueError("Media payload exceeds the input memory bound")
    data = _bytes(payload)
    if len(data) != size * 4:
        raise ValueError("FP32 media payload does not match declared shape")
    return torch.from_numpy(np.frombuffer(data, dtype="<f4").copy()).to(device).reshape(shape)


def _terminal(value: Mapping[str, object], default: bool | None) -> bool | None:
    """The optional ``terminal`` flag: whether the event ends its message."""
    flag = value.get("terminal", default)
    if flag is not None and not isinstance(flag, bool):
        raise ValueError("terminal must be a boolean")
    return flag


def _payload(value: Mapping[str, object]) -> bytes:
    """Exact text bytes: ``text`` (UTF-8) or ``payload_b64`` (anything), not both."""
    if "text" in value and "payload_b64" in value:
        raise ValueError("provide text or payload_b64, not both")
    if "text" in value:
        if not isinstance(value["text"], str):
            raise ValueError("text must be a UTF-8 string")
        return value["text"].encode("utf-8")
    return _bytes(value.get("payload_b64", ""))


def _read_event(value: Mapping[str, object], device: torch.device) -> ObservedEvent:
    modality = value.get("modality", "text")
    track = value.get("track")
    if not isinstance(track, str) or not track:
        raise ValueError("Observed track requires a nonempty name")
    event_id = value.get("event_id", "")
    if not isinstance(event_id, str):
        raise ValueError("Observed event address must be a string")
    role_name = value.get("role", "user")
    if not isinstance(role_name, str) or role_name not in ROLE_TO_ID:
        raise ValueError("Unknown observation role")
    if modality == "text":
        return ObservedEvent(
            track=track,
            modality=modality,
            payload=_payload(value),
            role_id=ROLE_TO_ID[role_name],
            terminal=_terminal(value, True),
            event_id=event_id,
        )
    # Media arrives as pinned teacher hidden states [1, tokens, 512], never raw media.
    return ObservedEvent(
        track=track,
        modality=modality,
        payload=_float_tensor(value.get("payload_f32_b64", ""), value.get("shape"), device),
        role_id=ROLE_TO_ID[role_name],
        extractor=str(value.get("extractor", "")),
        terminal=_terminal(value, True),
        event_id=event_id,
    )


WIRE_FORMATS = {"canonical", "openai", "anthropic"}
OBSOLETE_SAMPLING = {"max_repairs", "repair_probability", "temperature", "top_p", "top_k"}


def _tools_json(value: Mapping[str, object]) -> str | None:
    """The public tool descriptions of an output, as the JSON text the control carries."""
    if "tools" in value and "tools_json" in value:
        raise ValueError("provide tools or tools_json, not both")
    if "tools_json" in value:
        if not isinstance(value["tools_json"], str):
            raise ValueError("tools_json must be a JSON string")
        return value["tools_json"]
    if "tools" in value:
        return json.dumps(value["tools"], ensure_ascii=False, allow_nan=False)
    return None


def _control_text(value: Mapping[str, object], name: str) -> bytes:
    text = value.get(name, "")
    if not isinstance(text, str):
        raise ValueError(f"{name} must be a UTF-8 string")
    return text.encode("utf-8")


def _output(
    value: Mapping[str, object], max_capacity: int, history: list[bytes]
) -> tuple[ActionSpec | MediaOutputSpec, bytes]:
    """One requested output: what it may be, and the causal control bytes it is drawn under.

    ``history`` is the text the session holds, the public sources a tool call may copy from.
    """
    tools_json = _tools_json(value)
    structured = {"control_seed", "control_reference", "reference_origin"} & value.keys()
    if "control_b64" in value and structured:
        raise ValueError("provide control_b64 or control_seed/control_reference, not both")
    if "control_b64" in value:
        control = _bytes(value["control_b64"])
    else:
        control = build_control_payload(
            seed=_control_text(value, "control_seed"),
            reference=_control_text(value, "control_reference"),
            reference_origin=value.get("reference_origin"),
            tools_json="[]" if tools_json is None else tools_json,
            reference_event_ids=value.get("reference_event_ids", ()),
        )
    modality = value.get("modality", "text")
    if modality == "text":
        if OBSOLETE_SAMPLING & value.keys():
            raise ValueError(
                "Decoder sampling and repair controls are not part of the facet policy"
            )
        length = int(value.get("max_tokens", min(256, max_capacity)))
        if not 1 <= length <= max_capacity:
            raise ValueError("Output exceeds the text capacity")
        if tools_json is not None:
            formats = tuple(value.get("wire_formats", ["canonical"]))
            if not formats or any(name not in WIRE_FORMATS for name in formats):
                raise ValueError("unknown tool wire format")
            spec = compile_tool_spec(tools_json, length, wire_formats=formats)
            return replace(spec, copy_sources=recent_copy_sources(history)), control
        spec = ActionSpec(
            text_kind=value.get("kind", "bytes"),
            min_tokens=int(value.get("min_tokens", 0)),
            max_tokens=length,
            require_terminal=_terminal(value, None),
        )
        return spec, control
    if tools_json is not None:
        raise ValueError("Public tools describe a text output")
    return (
        MediaOutputSpec(
            modality=str(modality),
            sample_count=int(value.get("sample_count", 0)),
            sample_rate=24000 if modality == "audio" else 0,
            frames=int(value.get("frames", 1)),
            height=int(value.get("height", 0)),
            width=int(value.get("width", 0)),
            fps_num=int(value.get("fps_num", 0)),
            fps_den=int(value.get("fps_den", 1)),
            steps=int(value.get("steps", 30)),
        ),
        control,
    )


def _retrieved(session: TrackSession, delivered: int) -> tuple[list[dict], int]:
    """The items retrieved since the last answer, and the new count."""
    sources = session.memory.sources
    log = [] if sources is None else sources.log
    items = [
        {
            "lookup": entry.lookup,
            "collection": entry.collection,
            "item": entry.item.item_id,
            "event_id": entry.event_id,
            "modality": entry.item.modality,
            "origin": entry.item.origin,
            "score": entry.score,
            "publication_revision": entry.publication_revision,
            "available_at_ns": entry.available_at_ns,
            "snapshot_revision": entry.snapshot_revision,
            "searched_at_ns": entry.searched_at_ns,
        }
        for entry in log[delivered:]
    ]
    return items, len(log)


def _output_response(group, specs):
    outputs = []
    for i, spec in enumerate(specs):
        item = {
            "track": group.tape.tracks[i],
            "event_id": group.tape.event_ids[i],
            "modality": group.tape.modalities[i],
            "valid": group.tape.valid[i],
        }
        if isinstance(spec, ActionSpec):
            item["payload_b64"] = base64.b64encode(group.tape.payloads[i]).decode("ascii")
            item["emitted_ids"] = list(group.tape.emitted_ids[i])
            item["terminal"] = group.tape.terminal[i]
            try:
                item["text"] = group.tape.payloads[i].decode("utf-8", errors="strict")
            except UnicodeError:
                pass
        else:
            tensor = group.media_outputs[i]
            assert tensor is not None
            media = tensor[0].detach().float().contiguous().cpu().numpy().astype("<f4")
            item["payload_f32_b64"] = base64.b64encode(media.tobytes()).decode("ascii")
            item["shape"] = list(media.shape)
            # The tape binds the realized bytes by SHA256 and realizer identity.
            item["tape_payload"] = group.tape.payloads[i].decode("utf-8")
        outputs.append(item)
    return outputs


def _inside(root: Path | None, name: object) -> Path:
    """A path the client names, which must lie inside the knowledge base root."""
    if root is None:
        raise ValueError("Knowledge base operations need the server's --kb-root")
    if not isinstance(name, str) or not name:
        raise ValueError("A knowledge base path must be a nonempty string")
    path = (root / name).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Path lies outside the knowledge base root")
    return path


def _knowledge_op(runtime: MultiTrackRuntime, root: Path | None, op: str, request: Mapping):
    """The knowledge base operations of the protocol; the answer to send."""
    memory = runtime.memory
    name = request.get("name")
    if op == "kb_list":
        return {"bases": {key: base.summary() for key, base in memory.bases.items()}}
    if not isinstance(name, str):
        raise ValueError("Knowledge base operations need a name")
    if op == "kb_open":
        base = memory.open_base(
            name, _inside(root, request.get("path")), writable=bool(request.get("writable"))
        )
    elif op == "kb_create":
        base = memory.create_base(name)
    elif op == "kb_overlay":
        base = memory.create_overlay(
            name,
            request.get("scope"),
            _inside(root, request.get("path")),
            max_edges=int(request.get("max_edges", 2048)),
            byte_budget=request.get("byte_budget"),
        )
    elif op == "kb_forget":
        return {
            "forgotten": memory.forget(
                name,
                ids=request.get("ids"),
                session=request.get("session"),
                scope=request.get("scope"),
            )
        }
    elif op == "kb_promote":
        return {
            "promoted": memory.promote(
                name,
                request["destination"],
                request["ids"],
                reviewed=request.get("reviewed") is True,
            )
        }
    elif op == "kb_save":
        base = memory.bases[name]
        base.save(_inside(root, request.get("path")))
    elif op == "kb_close":
        memory.close_base(name)
        return {"closed": name}
    else:
        raise ValueError("Unknown knowledge base operation")
    return {"name": name, **base.summary()}


def _add_files(runtime: MultiTrackRuntime, root: Path | None, request: Mapping):
    """``add_files``: one preparation of each file, added to a collection and/or a base."""
    paths = request.get("paths")
    if not isinstance(paths, list) or not paths or not all(isinstance(p, str) for p in paths):
        raise ValueError("add_files needs a nonempty list of file paths")
    metadata = request.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("File metadata must be a JSON object")
    targets = {key: request.get(key) for key in ("collection", "base") if key in request}
    if not targets or not all(isinstance(v, str) for v in targets.values()):
        raise ValueError("add_files needs a collection name, a base name, or both")
    # With a knowledge base root every file must lie inside it.
    files = paths if root is None else [_inside(root, path) for path in paths]
    added = runtime.add_files(files, metadata=metadata, **targets)
    answer: dict[str, object] = {}
    if "collection" in targets:
        answer["collection"] = {"name": targets["collection"], "added": added.collection}
    if "base" in targets:
        summary = runtime.memory.bases[targets["base"]].summary()
        answer["base"] = {"added": len(added.base), **summary}
    return answer


class MultiTrackService:
    """One runtime per weights bundle, many independent shared-track sessions.

    ``handle`` answers one request object; ``serve`` reads a request per line. The runtime
    carries the mounted collections, and the preparer (called once, on first use) that
    prepares the files ``add_files`` writes: the existing media preparation.
    """

    def __init__(
        self,
        runtime: MultiTrackRuntime,
        *,
        max_sessions: int = 32,
        kb_root: str | Path | None = None,
        tools: Sequence[dict] = (),
        adapters: Mapping[str, Any] | None = None,
    ) -> None:
        """``kb_root`` enables the knowledge base operations: every path lies inside it."""
        if max_sessions < 1:
            raise ValueError("max_sessions must be positive")
        self.runtime = runtime
        self.model = runtime.generator.generator
        self.device = self.model.core.initial_facets.device
        self.max_sessions = max_sessions
        self.root = None if kb_root is None else Path(kb_root).resolve()
        if self.root is not None:
            if runtime.memory.memory_root is not None and runtime.memory.memory_root != self.root:
                raise ValueError("Service and runtime must use the same memory root")
            runtime.memory.memory_root = self.root
        self.sessions: dict[str, TrackSession] = {}
        self.delivered: dict[str, int] = {}
        self.execution_tools = tuple(tools)
        self.execution_adapters = dict(adapters or {})
        self.executors = {}

    def serve(self, input_stream: TextIO, output_stream: TextIO) -> None:
        # Undecodable bytes become U+FFFD, so a bad line fails as its own malformed request
        # instead of ending the iteration (and with it every open session).
        reconfigure = getattr(input_stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(errors="replace")
        for line in input_stream:
            try:
                response = self.handle(strict_json_loads(line))
            except Exception as exc:
                response = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            output_stream.write(self.encode(response) + "\n")
            output_stream.flush()

    @staticmethod
    def encode(response: dict[str, Any]) -> str:
        """One response line; a result that is not strict JSON (a NaN density) fails its own request."""

        def dump(value: dict[str, Any]) -> str:
            text = json.dumps(value, ensure_ascii=False, allow_nan=False)
            try:
                text.encode("utf-8")
            except UnicodeEncodeError:
                # A lone surrogate (an echoed request id, say) cannot be written as UTF-8;
                # escaping it keeps the line strict JSON.
                text = json.dumps(value, ensure_ascii=True, allow_nan=False)
            return text

        try:
            return dump(response)
        except (ValueError, TypeError) as exc:
            failure = {
                "ok": False,
                "error": f"{type(exc).__name__}: response is not strict JSON: {exc}",
            }
            if "id" in response:
                failure["id"] = response["id"]
            return dump(failure)

    def handle(self, request: Any) -> dict[str, Any]:
        """The answer to one request; an error is the request's own and never ends serving."""
        request_id = request.get("id") if isinstance(request, dict) else None
        try:
            if not isinstance(request, dict):
                raise ValueError("Request must be a JSON object")
            response = {"ok": True, **self._answer(request)}
        except Exception as exc:
            response = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        if request_id is not None:
            response["id"] = request_id
        return response

    def _answer(self, request: Mapping[str, Any]) -> dict[str, Any]:
        runtime, op = self.runtime, request.get("op")
        if op == "open":
            answer = self._open(request)
            session_id = answer["session"]
        elif op == "add_files":
            return _add_files(runtime, self.root, request)
        elif isinstance(op, str) and op.startswith("kb_"):
            return _knowledge_op(runtime, self.root, op, request)
        else:
            session_id = request.get("session")
            if session_id not in self.sessions:
                raise ValueError("Unknown or closed session")
            session = self.sessions[session_id]
            if op == "step":
                from megatoken.execution import SessionExecutor

                if session_id not in self.executors:
                    self.executors[session_id] = SessionExecutor(
                        runtime,
                        session,
                        self.execution_tools,
                        self.execution_adapters,
                        max_selections=int(request.get("max_selections", 128)),
                    )
                answer = self.executors[session_id].tick(
                    effort=int(request.get("effort", 6)), seed=int(request.get("seed", 0))
                )
                answer["session"] = session_id
            elif op == "track_action":
                description = runtime.track_action(
                    session,
                    request["operation"],
                    address=request.get("track"),
                    **request.get("values", {}),
                )
                answer = {
                    "session": session_id,
                    "revision": session.revision,
                    "track": asdict(description),
                }
            elif op == "observe":
                raw = request.get("events", [])
                if not isinstance(raw, list) or not raw:
                    raise ValueError("Observation meridian needs an event list")
                events = [_read_event(value, self.device) for value in raw]
                runtime.observe(session, events, effort=int(request.get("effort", 6)))
                answer = self._observed(session_id, session)
            elif op == "observe_text":
                track, role = request.get("track"), request.get("role", "user")
                if not isinstance(track, str) or not track:
                    raise ValueError("Observed track requires a nonempty name")
                if not isinstance(role, str):
                    raise ValueError("Unknown observation role")
                hints = {
                    name: request[name]
                    for name in ("content_kind", "language", "filename", "mime_type", "locale")
                    if name in request
                }
                chunks = runtime.observe_text(
                    session,
                    track,
                    _payload(request),
                    role=role,
                    effort=int(request.get("effort", 6)),
                    **hints,
                )
                answer = {**self._observed(session_id, session), "chunks": chunks}
            elif op == "enqueue":
                from megatoken.multitrack_runtime import ReadyWork

                spec, control = _output(
                    request["output"], self.model.config.max_capacity, session.history
                )
                runtime.enqueue(
                    session,
                    ReadyWork(
                        request["event_id"],
                        request["track"],
                        spec,
                        control,
                        tuple(request.get("prerequisites", ())),
                        int(request.get("task_revision", 0)),
                        int(request.get("priority", 0)),
                    ),
                )
                answer = {"session": session_id, "queued": len(session.queue)}
            elif op == "schedule_ready":
                selected_revision = runtime.schedule_ready(
                    session,
                    max_events=int(request.get("max_events", 4)),
                    effort=int(request.get("effort", 6)),
                )
                answer = {
                    "session": session_id,
                    "selection_revision": selected_revision,
                    "queued": len(session.queue),
                }
            elif op == "release_generation":
                runtime.release_generation(session, int(request["selection_revision"]))
                answer = {"session": session_id, "revision": session.revision}
            elif op == "lifecycle_tools":
                answer = {
                    "session": session_id,
                    "tools": session.tracks.legal_tools(tuple(runtime.output_specs)),
                }
            elif op == "execute_lifecycle":
                group = session.selected_groups[int(request["selection_revision"])]
                answer = {
                    "session": session_id,
                    "outcomes": runtime.execute_lifecycle(session, group),
                    "revision": session.revision,
                }
            elif op in {"poll_generation", "stop_delivery", "delivered"}:
                selected_revision = int(request["selection_revision"])
                if op == "stop_delivery":
                    work = runtime.stop_delivery(session, selected_revision)
                elif op == "delivered":
                    runtime.delivered(
                        session,
                        selected_revision,
                        request["event_ids"],
                        extents=request.get("extents"),
                    )
                    work = runtime.poll_generation(session, selected_revision)
                else:
                    work = runtime.poll_generation(session, selected_revision)
                answer = {
                    "session": session_id,
                    "revision": session.revision,
                    "selection_revision": selected_revision,
                    "status": work.status,
                    "error": work.error,
                    "delivery_progress": dict(work.delivery_progress),
                    "delivery_log": list(work.delivery_log),
                    "selection": json.loads(work.selection.tape.to_json()),
                }
                if work.result is not None and work.status in {"rendered", "delivered"}:
                    answer["outputs"] = _output_response(work.result, work.selection.specs)
                    answer["tape"] = json.loads(work.result.tape.to_json())
            elif op in {"generate", "speculate", "start_generation"}:
                answer = self._generate(session_id, session, request, speculative=op == "speculate")
            elif op == "accept":
                pending = session.pending
                if pending is None:
                    raise ValueError("No speculative group is pending")
                group = runtime.accept(session, pending)
                answer = {
                    "session": session_id,
                    "revision": session.revision,
                    "meridian_id": group.tape.meridian_id,
                }
            elif op == "cancel":
                runtime.cancel(session)
                answer = {"session": session_id, "revision": session.revision}
            elif op == "status":
                return {
                    "session": session_id,
                    "revision": session.revision,
                    "meridians": int(session.state.meridians_seen[0]),
                    "events_seen": int(session.state.events_seen[0]),
                    "retained_meridians": len(session.state.meridian_sizes),
                    "retained_events": sum(session.state.meridian_sizes),
                    "track_registry": list(session.state.track_registry),
                    "tracks": [asdict(t) for t in session.tracks.tracks.values()],
                    "pending": session.pending is not None,
                }
            elif op == "flush":
                self.runtime.memory.flush(session.memory)
                return {"flushed": session_id}
            elif op == "close":
                executor = self.executors.pop(session_id, None)
                if executor is not None:
                    executor.close()
                self.runtime.memory.close_session(session.memory)
                del self.sessions[session_id]
                self.delivered.pop(session_id, None)
                return {"closed": session_id}
            else:
                raise ValueError("Unknown multitrack operation")
        if op in {
            "open",
            "observe",
            "observe_text",
            "generate",
            "accept",
            "poll_generation",
            "start_generation",
            "delivered",
        }:
            retrieved, self.delivered[session_id] = _retrieved(
                self.sessions[session_id], self.delivered.get(session_id, 0)
            )
            if retrieved:
                answer["retrieved"] = retrieved
        return answer

    def _open(self, request: Mapping[str, Any]) -> dict[str, Any]:
        if len(self.sessions) >= self.max_sessions:
            raise ValueError("Session capacity reached")
        metadata = request.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError("Session metadata must be a JSON object")
        bases = request.get("bases", {})
        if not isinstance(bases, dict):
            raise ValueError("Mounted bases must be a JSON object of name to weight")
        options = {
            key: request[key]
            for key in ("collections", "max_sources", "threshold", "overlay", "scope", "memory_writes")
            if key in request
        }
        if "write_policy" in request:
            from megatoken.serving_writes import WritePolicy

            options["write_policy"] = WritePolicy(**request["write_policy"])
        session = self.runtime.open(
            metadata,
            bases={str(k): float(v) for k, v in bases.items()},
            trained=float(request.get("trained_weight", 1.0)),
            effort=int(request.get("effort", 6)),
            **options,
        )
        session_id = session.memory.session_id
        self.sessions[session_id] = session
        return {
            "session": session_id,
            "architecture_version": self.model.config.architecture_version,
            "memory_writes": session.memory.writes is not None,
            "overlay": None if session.memory.writes is None else session.memory.writes.base.name,
        }

    @staticmethod
    def _observed(session_id: str, session: TrackSession) -> dict[str, Any]:
        return {
            "session": session_id,
            "revision": session.revision,
            "next_meridian": int(session.state.meridians_seen[0]),
            "events_seen": int(session.state.events_seen[0]),
            "retained_events": sum(session.state.meridian_sizes),
        }

    def _generate(
        self,
        session_id: str,
        session: TrackSession,
        request: Mapping[str, Any],
        *,
        speculative: bool,
    ) -> dict[str, Any]:
        requested = request.get("outputs", [])
        if not isinstance(requested, list) or not requested:
            raise ValueError("Generation needs nonempty output tracks")
        names = tuple(item["track"] for item in requested)
        outputs = [
            _output(item, self.model.config.max_capacity, session.history) for item in requested
        ]
        specs, controls = (tuple(column) for column in zip(*outputs, strict=True))
        options = tuple(
            tuple(
                _output({**item, **option}, self.model.config.max_capacity, session.history)[0]
                for option in item["options"]
            )
            if "options" in item
            else (spec,)
            for item, spec in zip(requested, specs, strict=True)
        )
        rng = (
            None
            if request.get("seed") is None
            else torch.Generator(device=self.device).manual_seed(int(request["seed"]))
        )
        arguments = dict(
            tracks=names,
            event_ids=(
                tuple(item["event_id"] for item in requested)
                if any("event_id" in item for item in requested)
                else None
            ),
            specs=specs,
            modality_options=options,
            controls=controls,
            effort=int(request.get("effort", 6)),
            generator=rng,
        )
        if request.get("op") == "start_generation":
            revision = self.runtime.start_generation(session, **arguments)
            return {
                "session": session_id,
                "revision": session.revision,
                "selection_revision": revision,
                "status": "rendering",
            }
        if speculative:
            group = self.runtime.speculate(session, **arguments).result
        else:
            group = self.runtime.generate(session, **arguments)
        specs = tuple(
            next(p for p in choices if self.runtime.generator._spec_id(p) == fingerprint)
            for choices, fingerprint in zip(options, group.tape.spec_fingerprints, strict=True)
        )
        return {
            "session": session_id,
            "revision": session.revision,
            "provisional": speculative,
            "log_prob": float(group.log_prob),
            "outputs": _output_response(group, specs),
            "tape": json.loads(group.tape.to_json()),
        }


def serve_multitrack_jsonl(
    runtime: MultiTrackRuntime,
    input_stream: TextIO,
    output_stream: TextIO,
    *,
    max_sessions: int = 32,
    kb_root: str | Path | None = None,
) -> None:
    """Serve ``runtime`` over JSON lines; it never opens a listening socket."""
    MultiTrackService(runtime, max_sessions=max_sessions, kb_root=kb_root).serve(
        input_stream, output_stream
    )
