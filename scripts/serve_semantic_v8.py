#!/usr/bin/env python3
"""Serve actual native v8 selections through bounded JSONL messages."""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import torch

from megatoken.io import strict_json_loads
from megatoken.model.multitrack import MultiTrackGenerator
from megatoken.multitrack_runtime import MultiTrackRuntime
from megatoken.semantic_model_v8 import NativeSemanticInferenceV8
from megatoken.served import ServedBundle

MAX_REQUEST_BYTES = 40 << 20


@torch.inference_mode()
def dispatch(adapter, request, *, commit_bridge=None):
    if commit_bridge is None:
        return _dispatch(adapter, request)
    # Ownership check, durable invalidation and the actual operation form one
    # serialized adapter transaction, including when an embedding uses threads.
    with adapter.lock:
        return _dispatch(adapter, request, commit_bridge=commit_bridge)


def _dispatch(adapter, request, *, commit_bridge=None):
    if not isinstance(request, dict) or not isinstance(request.get("id"), str):
        raise ValueError("V8 RPC requires a stable request ID")
    operation = request.get("op")
    if commit_bridge is not None:
        commit_bridge.assert_operation(operation, request)
    if operation in {
        "durable-commit",
        "recover-commit",
        "durable-bootstrap",
        "recover-bootstrap",
        "durable-close",
        "recover-close",
    }:
        if commit_bridge is None:
            raise ValueError(
                "Native durable operations require a configured execution lease and journal"
            )
        fields = {"id", "op", "identity", "requestBase64", "preCheckpointDigest"}
        if operation in {"durable-bootstrap", "recover-bootstrap"}:
            fields |= {"createIdentity", "createRequestBase64", "createPreCheckpointDigest"}
        if set(request) != fields:
            raise ValueError("Native durable operation RPC fields differ")
        return getattr(commit_bridge, operation.replace("-", "_"))(
            **{key: value for key, value in request.items() if key not in {"id", "op"}}
        )
    if operation in {"metrics", "status"}:
        return adapter.metrics()
    methods = {
        name: getattr(adapter, name)
        for name in (
            "initialize",
            "select",
            "commit",
            "reject",
            "complete",
            "cancel",
            "checkpoint",
            "restore",
            "open_orchestration",
            "select_orchestration",
            "commit_orchestration",
            "bootstrap_orchestration_child",
            "status_orchestration",
            "cancel_orchestration",
        )
    }
    operation = operation.replace("-", "_") if isinstance(operation, str) else operation
    if operation not in methods:
        raise ValueError("Unknown native v8 RPC operation")
    return methods[operation](
        **{key: value for key, value in request.items() if key not in {"id", "op"}}
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--served", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--effort", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--max-fragments", type=int, default=64)
    parser.add_argument("--max-sessions", type=int, default=16)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--profile-output", type=Path)
    parser.add_argument("--request-journal-dir", type=Path)
    parser.add_argument("--execution-lock-fd", type=int)
    parser.add_argument("--execution-lock-path", type=Path)
    parser.add_argument("--execution-session-id")
    parser.add_argument("--execution-stream-id")
    parser.add_argument("--execution-session-generation", type=int, default=1)
    args = parser.parse_args()
    durability = (
        args.request_journal_dir,
        args.execution_lock_fd,
        args.execution_lock_path,
        args.execution_session_id,
        args.execution_stream_id,
    )
    if any(value is not None for value in durability) and not all(
        value is not None for value in durability
    ):
        parser.error(
            "Native durability requires the journal directory and every inherited execution-lock argument"
        )
    if args.request_journal_dir is None and args.execution_session_generation != 1:
        parser.error("Native execution generation requires configured durability")
    execution_lease = None
    if args.request_journal_dir is not None:
        from megatoken.semantic_commit_recovery_v8 import NativeExecutionLease

        execution_lease = NativeExecutionLease(
            args.execution_lock_fd,
            args.execution_lock_path,
            args.execution_session_id,
            args.execution_stream_id,
        )
    if args.profile_output is not None and args.profile_output.exists():
        parser.error("Native v8 profiles are immutable; choose a new output path")
    if args.threads < 1:
        parser.error("Native v8 CPU threads must be positive")
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    started = time.perf_counter()
    bundle = ServedBundle(args.served, device=args.device)
    runtime = MultiTrackRuntime(MultiTrackGenerator.of(bundle.model), chunking=bundle.chunking)
    adapter = NativeSemanticInferenceV8(
        runtime,
        model_identity=bundle.manifest["model_sha256"],
        training_identity=bundle.manifest.get("semantic_v8_lineage_digest"),
        effort=args.effort,
        max_tokens=args.max_tokens,
        max_fragments=args.max_fragments,
        max_sessions=args.max_sessions,
    )
    commit_bridge = None
    if execution_lease is not None:
        from megatoken.semantic_commit_recovery_v8 import DurableNativeCommitBridge
        from megatoken.semantic_request_journal_v8 import DurableNativeRequestJournal

        commit_bridge = DurableNativeCommitBridge(
            adapter,
            DurableNativeRequestJournal(
                args.request_journal_dir,
                max_request_bytes=64 << 20,
                max_result_bytes=64 << 20,
                max_log_bytes=512 << 20,
            ),
            execution_lease,
            session_generation=args.execution_session_generation,
        )
    synchronize()
    cold_seconds = time.perf_counter() - started
    try:
        while raw := sys.stdin.buffer.readline(MAX_REQUEST_BYTES + 1):
            if len(raw) > MAX_REQUEST_BYTES:
                raise ValueError("Native v8 request exceeds its byte budget")
            if not raw.strip():
                continue
            request = None
            synchronize()
            request_started = time.perf_counter()
            try:
                request = strict_json_loads(raw.decode("utf-8", errors="strict"))
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
                        **(
                            {"durabilityFenced": True}
                            if commit_bridge is not None and commit_bridge.fenced
                            else {}
                        ),
                    },
                }
            synchronize()
            reply["runtimeMetrics"] = {
                "schema": "semantic-inference-metrics/8",
                "computeSeconds": time.perf_counter() - request_started,
                "peakMemoryBytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                * (1 if sys.platform == "darwin" else 1024),
            }
            print(
                json.dumps(reply, ensure_ascii=False, separators=(",", ":"), allow_nan=False),
                flush=True,
            )
    finally:
        for identity in list(adapter.sessions):
            adapter.cancel(session=identity, reason="process-closed")
        runtime.close()
        if execution_lease is not None:
            execution_lease.close()
        if args.profile_output is not None:
            args.profile_output.parent.mkdir(parents=True, exist_ok=True)
            with args.profile_output.open("x", encoding="utf-8") as output:
                json.dump(
                    {**adapter.metrics(), "coldLoadSeconds": cold_seconds, "runtimeClosed": True},
                    output,
                    indent=2,
                    allow_nan=False,
                )
                output.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
