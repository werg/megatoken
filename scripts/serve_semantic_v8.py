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

from megatoken.model.multitrack import MultiTrackGenerator
from megatoken.multitrack_runtime import MultiTrackRuntime
from megatoken.semantic_model_v8 import NativeSemanticInferenceV8
from megatoken.served import ServedBundle

MAX_REQUEST_BYTES = 40 << 20


@torch.inference_mode()
def dispatch(adapter, request):
    if not isinstance(request, dict) or not isinstance(request.get("id"), str):
        raise ValueError("V8 RPC requires a stable request ID")
    operation = request.get("op")
    if operation in {"metrics", "status"}:
        return adapter.metrics()
    methods = {
        name: getattr(adapter, name)
        for name in ("initialize", "select", "commit", "reject", "complete", "cancel")
    }
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
    args = parser.parse_args()
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
        effort=args.effort,
        max_tokens=args.max_tokens,
        max_fragments=args.max_fragments,
        max_sessions=args.max_sessions,
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
                request = json.loads(raw)
                result = dispatch(adapter, request)
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
