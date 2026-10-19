"""Served-only exports and stateful inference using the training policy itself."""

from __future__ import annotations

import base64
import binascii
import json
import os
import shutil
import tempfile
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

import torch
from jsonschema.exceptions import SchemaError, ValidationError
from safetensors.torch import load_file, save_file

from megatoken.contracts import ROLE_TO_ID
from megatoken.controls import build_control_payload
from megatoken.data.chunking import ChunkingConfig, segment_bytes
from megatoken.generation.structure import TAPE_VERSION, ActionSpec, ActionTape, Observation
from megatoken.io import fsync_directory, sha256_file, strict_json_loads, write_json
from megatoken.loading import build_model, load_codec, parameter_report
from megatoken.model.config import ModelConfig
from megatoken.tensors import byte_tensor


def export_served(
    checkpoint: str | Path, output: str | Path, *, codec_path: str = "byte"
) -> dict[str, Any]:
    from megatoken.training.checkpoint import load_checkpoint

    payload, metadata = load_checkpoint(checkpoint)
    config = ModelConfig.from_dict(payload["model_config"])
    codec = load_codec(codec_path, vocab_size=config.vocab_size)
    if codec.fingerprint != payload["codec_fingerprint"]:
        raise ValueError("Export codec does not match the trained checkpoint")
    chunking = ChunkingConfig.from_identity(payload["chunking"], codec)
    chunking = chunking.with_hard_limit(config.max_capacity)
    precision = payload["train_config"]["precision"]
    model = build_model(config, codec, precision=precision)
    model.load_state_dict(payload["model"], strict=True)
    state = model.served_state_dict()
    destination = Path(output)
    if destination.exists():
        raise FileExistsError("A served bundle is immutable; choose a new output directory")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".served-", dir=destination.parent))
    try:
        # Contiguous clones avoid exporting transposed/tied aliases as separate
        # parameter owners. The served constructor restores declared sharing.
        save_file(
            {name: value.detach().cpu().contiguous().clone() for name, value in state.items()},
            str(temporary / "model.safetensors"),
        )
        served_config = config.to_dict()
        served_config["checkpoint_stages"] = False
        write_json(temporary / "model_config.json", served_config)
        bundled_codec = "byte"
        if codec_path != "byte":
            source = Path(codec_path)
            if source.is_dir():
                source_metadata = json.loads((source / "codec.json").read_text())
                source = source / source_metadata["tokenizer_file"]
            shutil.copyfile(source, temporary / "tokenizer.json")
            bundled_codec = "tokenizer.json"
        durable_files = [temporary / "model.safetensors"]
        if bundled_codec != "byte":
            durable_files.append(temporary / bundled_codec)
        for artifact in durable_files:
            with artifact.open("rb") as handle:
                os.fsync(handle.fileno())
        manifest = {
            "format_version": 8,
            "model_sha256": sha256_file(temporary / "model.safetensors"),
            "config_sha256": sha256_file(temporary / "model_config.json"),
            "codec_file": bundled_codec,
            "codec_fingerprint": codec.fingerprint,
            "chunking": chunking.identity(codec),
            "policy_likelihood": "canonical_facet_gaussian_density",
            "precision": precision,
            "model_version": payload["global_step"],
            "routing_version": config.routing_version,
            "tape_version": TAPE_VERSION,
            "checkpoint_sha256": metadata["state_sha256"],
            "parameters": parameter_report(model)["totals"]["served"],
            "training_heads": False,
            "donor_assets_required": False,
        }
        if bundled_codec != "byte":
            manifest["tokenizer_sha256"] = sha256_file(temporary / bundled_codec)
        write_json(temporary / "manifest.json", manifest)
        fsync_directory(temporary)
        os.rename(temporary, destination)
        fsync_directory(destination.parent)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return {"bundle": str(destination), **manifest}


@dataclass
class Session:
    session_id: str
    state: Any
    records: list[tuple[str, bytes]] = field(default_factory=list)
    tapes: list[ActionTape] = field(default_factory=list)
    events: list[Observation | ActionTape] = field(default_factory=list)
    model_id: str = ""


class InferenceEngine:
    def __init__(self, bundle: str | Path, *, device: str = "auto", compile: bool = False) -> None:
        directory = Path(bundle)
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest.get("format_version") != 8 or manifest.get("training_heads") is not False:
            raise ValueError(
                "Unsupported or training-only bundle: v8 direct realization requires "
                "v8 weights and state; earlier bundles cannot be loaded"
            )
        if manifest.get("policy_likelihood") != "canonical_facet_gaussian_density":
            raise ValueError("Unsupported served action likelihood")
        if manifest.get("tape_version") != TAPE_VERSION:
            raise ValueError("Served tape protocol differs from the current control-aware policy")
        if manifest.get("precision") not in {"fp32", "bf16"}:
            raise ValueError("Unsupported served precision")
        if sha256_file(directory / "model.safetensors") != manifest["model_sha256"]:
            raise ValueError("Served weights checksum mismatch")
        if sha256_file(directory / "model_config.json") != manifest["config_sha256"]:
            raise ValueError("Served configuration checksum mismatch")
        self.config = ModelConfig.from_dict(
            json.loads((directory / "model_config.json").read_text())
        )
        if manifest.get("routing_version") != self.config.routing_version:
            raise ValueError("Served routing version differs from the model configuration")
        codec_name = manifest["codec_file"]
        if codec_name != "byte":
            if Path(codec_name).name != codec_name:
                raise ValueError("Unsafe served codec path")
            if sha256_file(directory / codec_name) != manifest["tokenizer_sha256"]:
                raise ValueError("Served tokenizer checksum mismatch")
        self.codec = load_codec(
            "byte" if codec_name == "byte" else directory / codec_name,
            vocab_size=self.config.vocab_size,
        )
        if self.codec.fingerprint != manifest["codec_fingerprint"]:
            raise ValueError("Served codec mapping differs from the training codec")
        self.chunking = ChunkingConfig.from_identity(manifest["chunking"], self.codec)
        if self.chunking.hard_max_tokens > self.config.max_capacity:
            raise ValueError("Served chunk policy exceeds the configured model capacity")
        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device == "auto"
            else torch.device(device)
        )
        self.model = build_model(
            self.config,
            self.codec,
            device=self.device,
            precision=manifest["precision"],
            training_heads=False,
        )
        self.model.load_state_dict(
            load_file(str(directory / "model.safetensors"), device=str(self.device)), strict=True
        )
        if parameter_report(self.model)["totals"]["served"] != manifest["parameters"]:
            raise ValueError("Served parameter ownership count differs from the bundle")
        self.model.core.set_model_version(manifest["model_version"])
        self.model.eval()
        if compile:
            self.model.core.compile_kernels()
        self.manifest = manifest

    def new_session(self) -> Session:
        return Session(
            uuid.uuid4().hex,
            self.model.core.initial_state(1, device=self.device),
            model_id=self.manifest["model_sha256"],
        )

    def _check_session(self, session: Session) -> None:
        if session.model_id != self.manifest["model_sha256"]:
            raise ValueError(
                "Session belongs to another served model; replay actual history into a new session"
            )

    @torch.inference_mode()
    def observe(
        self,
        session: Session,
        payload: bytes,
        *,
        role: str = "user",
        content_kind: str = "auto",
        language: str | None = None,
        filename: str | None = None,
        mime_type: str | None = None,
        locale: str | None = None,
    ) -> None:
        # This API already receives one authoritative event and its role.
        # Chat delimiters inside a tool result or user example are content.
        chunks = segment_bytes(
            payload,
            self.codec,
            self.chunking,
            content_kind=content_kind,
            language=language,
            filename=filename,
            mime_type=mime_type,
            locale=locale,
            chat_template="none",
        )
        self._append_observations(session, [chunk.payload for chunk in chunks], role)

    @torch.inference_mode()
    def observe_physical(
        self,
        session: Session,
        payload: bytes,
        *,
        role: str = "user",
        token_ids: Sequence[int] | None = None,
    ) -> None:
        """Ingest an already prepared physical record without changing its boundary."""
        ids = self.codec.encode(payload) if token_ids is None else list(token_ids)
        if self.codec.decode(ids) != payload:
            raise ValueError("Physical observation token IDs do not reproduce its exact bytes")
        if len(ids) > self.chunking.hard_max_tokens or len(payload) > self.chunking.max_bytes:
            raise ValueError("Physical observation exceeds the served chunk policy")
        self._append_observations(session, [payload], role)

    def _append_observations(self, session: Session, payloads: list[bytes], role: str) -> None:
        self._check_session(session)
        if role not in ROLE_TO_ID:
            raise ValueError(f"Unknown physical-record role {role!r}")
        # The same versioned semantic boundaries define training and live
        # recurrent writes. Record each physical observation for exact replay.
        state = session.state
        events = []
        for payload in payloads:
            ids, lengths = byte_tensor([payload], self.device)
            state = self.model.core.observe(
                ids,
                state,
                lengths=lengths,
                role_ids=torch.tensor([ROLE_TO_ID[role]], device=self.device),
            )
            events.append(Observation(payload, ROLE_TO_ID[role]))
        # An invalid observation cannot leave a partially appended transcript.
        session.state = state
        session.records.extend((role, event.payload) for event in events)
        session.events.extend(events)

    @torch.inference_mode()
    def generate(
        self,
        session: Session,
        *,
        spec: ActionSpec | None = None,
        effort: int = 6,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        control_seed: bytes = b"",
        control_reference: bytes = b"",
        reference_origin: str | None = None,
        tools_json: str = "[]",
    ) -> Any:
        """Sample once with the same explicit causal controls used in training.

        ``seed`` selects the random draw. ``control_seed`` is model input bytes;
        neither reference controls nor public tool descriptions are observations.
        The exact framed payload is retained by the returned action tape.
        """
        self._check_session(session)
        control_payload = build_control_payload(
            seed=control_seed,
            reference=control_reference,
            reference_origin=reference_origin,
            tools_json=tools_json,
        )
        if seed is not None:
            if generator is not None:
                raise ValueError("provide seed or generator, not both")
            generator = torch.Generator(device=self.device).manual_seed(seed)
        result = self.model.sample(
            session.state,
            spec=spec,
            effort=effort,
            generator=generator,
            control_payload=control_payload,
        )
        session.state = result.state
        session.tapes.append(result.tape)
        session.events.append(result.tape)
        session.records.append(("assistant", result.payload))
        return result


def _request_payload(request: dict[str, Any]) -> bytes:
    text, binary = "text" in request, "payload_b64" in request
    if text == binary:
        raise ValueError("provide exactly one of text or payload_b64")
    if text:
        if not isinstance(request["text"], str):
            raise ValueError("text must be a UTF-8 string")
        return request["text"].encode("utf-8")
    if not isinstance(request["payload_b64"], str):
        raise ValueError("payload_b64 must be a base64 string")
    try:
        return base64.b64decode(request["payload_b64"], validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("payload_b64 is not valid base64") from exc


def _request_tools_json(request: dict[str, Any]) -> str | None:
    if "tools" in request and "tools_json" in request:
        raise ValueError("provide tools or tools_json, not both")
    if "tools_json" in request:
        if not isinstance(request["tools_json"], str):
            raise ValueError("tools_json must be a JSON string")
        return request["tools_json"]
    if "tools" in request:
        return json.dumps(request["tools"], ensure_ascii=False, allow_nan=False)
    return None


def _request_control_bytes(request: dict[str, Any], name: str) -> bytes:
    value = request.get(name, "")
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a UTF-8 string")
    return value.encode("utf-8")


def _request_spec(
    engine: InferenceEngine,
    session: Session,
    request: dict[str, Any],
    tools_json: str | None = None,
) -> ActionSpec:
    obsolete = {"max_repairs", "repair_probability", "temperature", "top_p", "top_k"}
    if obsolete.intersection(request):
        raise ValueError("Decoder sampling and repair controls are not part of the facet policy")
    maximum = int(request.get("max_tokens", min(512, engine.config.max_capacity)))
    if not 0 <= maximum <= engine.config.max_capacity:
        raise ValueError("max_tokens exceeds this model's configured capacity")
    if tools_json is not None:
        from dataclasses import replace

        from megatoken.tool_schema import compile_tool_spec, recent_copy_sources

        formats = tuple(request.get("wire_formats", ["canonical"]))
        if not formats or any(
            value not in {"canonical", "openai", "anthropic"} for value in formats
        ):
            raise ValueError("unknown tool wire format")
        spec = compile_tool_spec(
            tools_json,
            maximum,
            wire_formats=formats,
        )
        sources = recent_copy_sources([payload for _, payload in session.records])
        return replace(spec, copy_sources=sources)
    return ActionSpec(
        text_kind=request.get("kind", "bytes"),
        min_tokens=int(request.get("min_tokens", 0)),
        max_tokens=maximum,
        require_terminal=request.get("require_terminal"),
    )


def serve_jsonl(
    engine: InferenceEngine, input_stream: TextIO, output_stream: TextIO, *, max_sessions: int = 32
) -> None:
    """A local stateful JSON-lines protocol; it never opens a listening socket.

    Operations are open, observe, generate, status, and close. All responses
    include ``ok`` and echo an optional request ``id``. Generated bytes always
    appear as base64; ``text`` appears only when the bytes are valid UTF-8.
    Errors are per-request and do not erase existing sessions. The complete
    augmented tape is returned for every generation, including invalid attempts.
    """
    if max_sessions < 1:
        raise ValueError("max_sessions must be positive")
    sessions: dict[str, Session] = {}
    for line in input_stream:
        request_id = None
        try:
            request = strict_json_loads(line)
            if not isinstance(request, dict):
                raise ValueError("each protocol line must be a JSON object")
            request_id = request.get("id")
            operation = request.get("op")
            if operation == "open":
                if len(sessions) >= max_sessions:
                    raise ValueError("session limit reached; close an existing session")
                session = engine.new_session()
                sessions[session.session_id] = session
                response = {
                    "session": session.session_id,
                    "model_version": engine.manifest["model_version"],
                }
            else:
                session_id = request.get("session")
                if session_id not in sessions:
                    raise ValueError("unknown or closed session")
                session = sessions[session_id]
                if operation == "observe":
                    engine.observe(
                        session,
                        _request_payload(request),
                        role=request.get("role", "user"),
                        content_kind=request.get("content_kind", "auto"),
                        language=request.get("language"),
                        filename=request.get("filename"),
                        mime_type=request.get("mime_type"),
                        locale=request.get("locale"),
                    )
                    response = {"session": session_id, "records": len(session.records)}
                elif operation == "generate":
                    tools_json = _request_tools_json(request)
                    spec = _request_spec(engine, session, request, tools_json)
                    result = engine.generate(
                        session,
                        spec=spec,
                        effort=int(request.get("effort", 6)),
                        seed=request.get("seed"),
                        control_seed=_request_control_bytes(request, "control_seed"),
                        control_reference=_request_control_bytes(request, "control_reference"),
                        reference_origin=request.get("reference_origin"),
                        tools_json="[]" if tools_json is None else tools_json,
                    )
                    response = {
                        "session": session_id,
                        "payload_b64": base64.b64encode(result.payload).decode(),
                        "emitted_ids": list(result.emitted_ids),
                        "valid": result.tape.valid,
                        "terminal": result.tape.terminal,
                        "scored_positions": result.scored_positions,
                        "latent_dimensions": result.tape.latent_dimensions,
                        "likelihood": "canonical_facet_gaussian_density",
                        "log_prob": float(result.log_prob),
                        "tape": json.loads(result.tape.to_json()),
                    }
                    try:
                        response["text"] = result.payload.decode("utf-8", errors="strict")
                    except UnicodeError:
                        pass
                elif operation == "status":
                    response = {
                        "session": session_id,
                        "records": len(session.records),
                        "decisions": len(session.tapes),
                        "events": len(session.events),
                    }
                elif operation == "close":
                    del sessions[session_id]
                    response = {"session": session_id, "closed": True}
                else:
                    raise ValueError(f"unknown protocol operation {operation!r}")
            response = {"ok": True, "id": request_id, **response}
        except (ValueError, TypeError, KeyError, RuntimeError, SchemaError, ValidationError) as exc:
            response = {
                "ok": False,
                "id": request_id,
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
        output_stream.write(
            json.dumps(response, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"
        )
        output_stream.flush()
