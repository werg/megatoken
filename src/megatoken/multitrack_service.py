"""Stateful JSONL serving for v7 multitrack semantic meridians.

Transport media as explicitly shaped little-endian FP32 buffers. The trained
encoder/realizer weights are loaded from one served Megatoken bundle. No tool
command is executed by the server; execution remains an external capability.
"""

from __future__ import annotations

import base64
import json
import math
import uuid
from collections.abc import Mapping
from typing import TextIO

import numpy as np
import torch

from megatoken.contracts import ROLE_TO_ID
from megatoken.generation.structure import ActionSpec
from megatoken.inference import InferenceEngine
from megatoken.io import strict_json_loads
from megatoken.model.media import MediaOutputSpec
from megatoken.model.multitrack import MultiTrackGenerator, ObservedEvent
from megatoken.multitrack_runtime import MultiTrackRuntime, TrackSession


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
    if not isinstance(shape, list) or not shape or any(
        not isinstance(n, int) or isinstance(n, bool) or n < 1 for n in shape
    ):
        raise ValueError("Media data requires a nonempty integer shape")
    size = math.prod(shape)
    if size > max_elements:
        raise ValueError("Media payload exceeds the input memory bound")
    data = _bytes(payload)
    if len(data) != size * 4:
        raise ValueError("FP32 media payload does not match declared shape")
    return torch.from_numpy(np.frombuffer(data, dtype="<f4").copy()).to(device).reshape(shape)


def _read_event(value: Mapping[str, object], device: torch.device) -> ObservedEvent:
    modality = value.get("modality", "text")
    track = value.get("track")
    if not isinstance(track, str) or not track:
        raise ValueError("Observed track requires a nonempty name")
    role_name = value.get("role", "user")
    if not isinstance(role_name, str) or role_name not in ROLE_TO_ID:
        raise ValueError("Unknown observation role")
    if modality == "text":
        payload = _bytes(value.get("payload_b64", ""))
    else:
        payload = _float_tensor(value.get("payload_f32_b64", ""), value.get("shape"), device)
    return ObservedEvent(
        track=track, modality=modality, payload=payload,
        role_id=ROLE_TO_ID[role_name],
        sample_rate=int(value.get("sample_rate", 16000)),
    )


def _output_spec(value: Mapping[str, object], max_capacity: int):
    modality = value.get("modality", "text")
    if modality == "text":
        length = int(value.get("max_tokens", min(256, max_capacity)))
        if not 1 <= length <= max_capacity:
            raise ValueError("Output exceeds the text capacity")
        return ActionSpec(max_tokens=length)
    return MediaOutputSpec(
        modality=str(modality),
        sample_count=int(value.get("sample_count", 0)),
        frames=int(value.get("frames", 1)),
        height=int(value.get("height", 64)),
        width=int(value.get("width", 64)),
    )


def _output_response(group, specs):
    outputs = []
    for i, spec in enumerate(specs):
        item = {
            "track": group.tape.tracks[i],
            "modality": group.tape.modalities[i],
            "valid": group.tape.valid[i],
        }
        if isinstance(spec, ActionSpec):
            item["payload_b64"] = base64.b64encode(group.tape.payloads[i]).decode("ascii")
            item["emitted_ids"] = list(group.tape.emitted_ids[i])
        else:
            tensor = group.media_outputs[i]
            assert tensor is not None
            item["payload_f32_b64"] = base64.b64encode(group.tape.payloads[i]).decode("ascii")
            item["shape"] = list(tensor.shape[1:])
        outputs.append(item)
    return outputs


def serve_multitrack_jsonl(
    engine: InferenceEngine,
    input_stream: TextIO,
    output_stream: TextIO,
    *,
    max_sessions: int = 32,
) -> None:
    """One runtime per weights bundle, many independent shared-track sessions."""
    if max_sessions < 1:
        raise ValueError("max_sessions must be positive")
    runtime = MultiTrackRuntime(MultiTrackGenerator(engine.model))
    sessions: dict[str, TrackSession] = {}

    for line in input_stream:
        request_id = None
        try:
            request = strict_json_loads(line)
            if not isinstance(request, dict):
                raise ValueError("Request must be a JSON object")
            request_id = request.get("id")
            op = request.get("op")
            if op == "open":
                if len(sessions) >= max_sessions:
                    raise ValueError("Session capacity reached")
                session_id = uuid.uuid4().hex
                sessions[session_id] = runtime.open()
                answer = {"session": session_id, "architecture_version": 7}
            else:
                session_id = request.get("session")
                if session_id not in sessions:
                    raise ValueError("Unknown or closed session")
                session = sessions[session_id]
                if op == "observe":
                    raw = request.get("events", [])
                    if not isinstance(raw, list) or not raw:
                        raise ValueError("Observation meridian needs an event list")
                    events = [_read_event(value, engine.device) for value in raw]
                    runtime.observe(session, events)
                    answer = {
                        "session": session_id, "revision": session.revision,
                        "retained_events": sum(session.state.meridian_sizes),
                    }
                elif op in {"generate", "speculate"}:
                    requested = request.get("outputs", [])
                    if not isinstance(requested, list) or not requested:
                        raise ValueError("Generation needs nonempty output tracks")
                    names = tuple(item["track"] for item in requested)
                    specs = tuple(
                        _output_spec(item, engine.config.max_capacity) for item in requested
                    )
                    controls = tuple(_bytes(item.get("control_b64", "")) for item in requested)
                    rng = (
                        None if request.get("seed") is None
                        else torch.Generator(device=engine.device).manual_seed(int(request["seed"]))
                    )
                    args = dict(
                        tracks=names, specs=specs, controls=controls,
                        effort=int(request.get("effort", 6)), generator=rng,
                    )
                    if op == "generate":
                        group = runtime.generate(session, **args)
                    else:
                        group = runtime.speculate(session, **args).result
                    answer = {
                        "session": session_id, "revision": session.revision,
                        "provisional": op == "speculate",
                        "log_prob": float(group.log_prob),
                        "outputs": _output_response(group, specs),
                        "tape": json.loads(group.tape.to_json()),
                    }
                elif op == "accept":
                    pending = session.pending
                    if pending is None:
                        raise ValueError("No speculative group is pending")
                    group = runtime.accept(session, pending)
                    answer = {
                        "session": session_id, "revision": session.revision,
                        "meridian_id": group.tape.meridian_id,
                    }
                elif op == "cancel":
                    runtime.cancel(session)
                    answer = {"session": session_id, "revision": session.revision}
                elif op == "status":
                    answer = {
                        "session": session_id, "revision": session.revision,
                        "meridians": len(session.state.meridian_sizes),
                        "retained_events": sum(session.state.meridian_sizes),
                        "track_registry": list(session.state.track_registry),
                        "pending": session.pending is not None,
                    }
                elif op == "close":
                    del sessions[session_id]
                    answer = {"closed": session_id}
                else:
                    raise ValueError("Unknown multitrack operation")
            response = {"ok": True, **answer}
        except Exception as exc:
            response = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        if request_id is not None:
            response["id"] = request_id
        output_stream.write(json.dumps(response, ensure_ascii=False, allow_nan=False) + "\n")
        output_stream.flush()
