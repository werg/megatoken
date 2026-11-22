"""Deterministic JSONL peer backed by the real V8 native runtime.

This fixture replaces only the final lexical decoder output. All source binding,
TrackSession state, native selection evidence, checkpoint serialization and
restore replay are produced by the production Python adapter. It is an
engineering recovery test and provides no learned model or source qualification.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
import resource
import sys
import time
from pathlib import Path

import torch

from megatoken.generation.codec import ByteCodec
from megatoken.generation.policy import FacetedGenerator
from megatoken.model import FacetedCore, ModelConfig
from megatoken.model.multitrack import MultiTrackGenerator
from megatoken.multitrack_runtime import MultiTrackRuntime
from megatoken.semantic_model_v8 import NativeSemanticInferenceV8
from scripts.serve_semantic_v8 import MAX_REQUEST_BYTES, dispatch

MAX_TOKENS = 512
MAX_FRAGMENTS = 64
EFFORT = 1
SEED = 17
DECODER_LOG = "ORCH_DECODER "


def _build_model():
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    codec = ByteCodec.byte_fallback()
    model = FacetedGenerator(
        FacetedCore(ModelConfig.tiny(max_history_chunks=128, max_capacity=512)), codec
    ).eval()
    return model, codec


def _model_identity(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _describe():
    model, codec = _build_model()
    return {
        "checkpointIdentity": _model_identity(model),
        "tokenizerIdentity": codec.fingerprint,
        "trainingIdentity": None,
        "seed": SEED,
        "generationSettings": {
            "device": "cpu",
            "maxTokens": MAX_TOKENS,
            "maxFragments": MAX_FRAGMENTS,
            "effort": EFFORT,
        },
    }


def _fixture_records():
    fixture = Path(__file__).with_name("semantic-orchestration-teacher-v8.json.gz.b64")
    document = json.loads(gzip.decompress(base64.b64decode(fixture.read_bytes())))
    decisions = [
        event["record"]
        for event in document["sidecar"]["orchestrationTranscript"]["events"]
        if event["kind"] == "decision"
    ]
    child_targets = [
        row["fields"]["line"]
        for row in document["public"]["timeline"]
        if row["mode"] == "generated"
    ]
    if len(decisions) != 4 or len(child_targets) != 2:
        raise ValueError(
            "The paired recovery fixture must contain four decisions and two child targets"
        )
    return [decisions[0], decisions[1], *child_targets, decisions[2], decisions[3]]


class _FragmentingFixtureDecoder:
    """Replay a deterministic sequence without exceeding native spec bounds."""

    def __init__(self, records, codec, log):
        self.records = tuple(record.encode("utf-8") for record in records)
        self.codec = codec
        self.log = log
        self.next_index = 0
        self.active_index = None
        self.active_payload = b""
        self.offset = 0

    def __call__(self, _multi, _raw, specs, _control, **_kwargs):
        if len(specs) != 1:
            raise AssertionError(
                "The bounded recovery fixture expects one native track per generation"
            )
        spec = specs[0]
        maximum = int(spec.max_tokens)
        if not spec.allow_text or maximum < 1:
            raise AssertionError("Native generation spec has no text token capacity")
        if self.active_index is None or self.offset == len(self.active_payload):
            if self.next_index >= len(self.records):
                raise AssertionError("The deterministic decoder sequence is exhausted")
            self.active_index = self.next_index
            self.next_index += 1
            self.active_payload = self.records[self.active_index]
            self.offset = 0

        beginning = self.offset
        end = min(beginning + maximum, len(self.active_payload))
        while (
            end > beginning and len(self.codec.encode(self.active_payload[beginning:end])) > maximum
        ):
            end -= 1
        if end == beginning:
            raise AssertionError("No nonempty lexical fragment fits the native generation spec")
        payload = self.active_payload[beginning:end]
        token_ids = tuple(self.codec.encode(payload))
        if len(token_ids) < spec.min_tokens or len(token_ids) > maximum:
            raise AssertionError("Lexical fixture fragment violates native token bounds")
        self.offset = end
        terminal = end == len(self.active_payload)
        if spec.require_terminal is not None and terminal is not spec.require_terminal:
            raise AssertionError("Lexical fixture fragment violates native terminal constraints")
        if spec.text_kind == "ascii" and any(byte > 0x7F for byte in payload):
            raise AssertionError("Lexical fixture output violates its native ASCII alphabet")
        log = [
            self.active_index,
            beginning,
            len(payload),
            len(token_ids),
            terminal,
            hashlib.sha256(payload).hexdigest(),
        ]
        self.log.write(DECODER_LOG + json.dumps(log, separators=(",", ":")) + "\n")
        self.log.flush()
        return (payload,), (token_ids,), (True,), (terminal,), (None,)


def _serve(decoder_log_path, durability=None):
    model, codec = _build_model()
    identity = _model_identity(model)
    runtime = MultiTrackRuntime(MultiTrackGenerator.of(model))
    adapter = NativeSemanticInferenceV8(
        runtime,
        model_identity=identity,
        effort=EFFORT,
        max_tokens=MAX_TOKENS,
        max_fragments=MAX_FRAGMENTS,
        max_sessions=16,
        max_attempts=4096,
    )
    commit_bridge = execution_lease = None
    if durability is not None:
        from megatoken.semantic_commit_recovery_v8 import (
            DurableNativeCommitBridge,
            NativeExecutionLease,
        )
        from megatoken.semantic_request_journal_v8 import DurableNativeRequestJournal

        execution_lease = NativeExecutionLease(
            durability.execution_lock_fd,
            durability.execution_lock_path,
            durability.execution_session_id,
            durability.execution_stream_id,
        )
        commit_bridge = DurableNativeCommitBridge(
            adapter,
            DurableNativeRequestJournal(
                durability.request_journal_dir,
                max_request_bytes=64 << 20,
                max_result_bytes=64 << 20,
                max_log_bytes=512 << 20,
            ),
            execution_lease,
        )
    decoder_log_path.parent.mkdir(parents=True, exist_ok=True)
    decoder_log = decoder_log_path.open("x", encoding="utf-8")
    decoder = _FragmentingFixtureDecoder(_fixture_records(), codec, decoder_log)
    original_realize = MultiTrackGenerator._realize

    def realize(multi, raw, specs, control, **kwargs):
        return decoder(multi, raw, specs, control, **kwargs)

    MultiTrackGenerator._realize = realize
    try:
        while raw := sys.stdin.buffer.readline(MAX_REQUEST_BYTES + 1):
            if len(raw) > MAX_REQUEST_BYTES:
                raise ValueError("Native v8 request exceeds its byte budget")
            if not raw.strip():
                continue
            request = None
            started = time.perf_counter()
            try:
                from megatoken.io import strict_json_loads

                request = strict_json_loads(raw.decode("utf-8", errors="strict"))
                with torch.inference_mode():
                    result = dispatch(adapter, request, commit_bridge=commit_bridge)
                reply = {"id": request["id"], "ok": True, "result": result}
            except Exception as error:
                reply = {
                    "id": request.get("id") if isinstance(request, dict) else None,
                    "ok": False,
                    "error": {
                        "category": type(error).__name__,
                        "message": str(error),
                        "evidence": getattr(error, "evidence", None),
                    },
                }
            reply["runtimeMetrics"] = {
                "schema": "semantic-inference-metrics/8",
                "computeSeconds": time.perf_counter() - started,
                "peakMemoryBytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            }
            print(
                json.dumps(reply, ensure_ascii=False, separators=(",", ":"), allow_nan=False),
                flush=True,
            )
    finally:
        for session in list(adapter.sessions):
            adapter.cancel(session=session, reason="test-peer-closed")
        MultiTrackGenerator._realize = original_realize
        runtime.close()
        decoder_log.close()
        if execution_lease is not None:
            execution_lease.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--describe", action="store_true")
    parser.add_argument("--decoder-log", type=Path)
    parser.add_argument("--request-journal-dir", type=Path)
    parser.add_argument("--execution-lock-fd", type=int)
    parser.add_argument("--execution-lock-path", type=Path)
    parser.add_argument("--execution-session-id")
    parser.add_argument("--execution-stream-id")
    arguments = parser.parse_args()
    durability = (
        arguments.request_journal_dir,
        arguments.execution_lock_fd,
        arguments.execution_lock_path,
        arguments.execution_session_id,
        arguments.execution_stream_id,
    )
    if any(value is not None for value in durability) and not all(
        value is not None for value in durability
    ):
        parser.error("Every native durability argument must be supplied together")
    if arguments.describe:
        print(json.dumps(_describe(), ensure_ascii=False, separators=(",", ":"), allow_nan=False))
        return 0
    if arguments.decoder_log is None:
        parser.error("--decoder-log is required for the JSONL recovery fixture")
    _serve(arguments.decoder_log, arguments if arguments.request_journal_dir is not None else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
