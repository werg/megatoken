#!/usr/bin/env python3
"""Serve semantic model selections through Megatoken's actual frozen inference path.

Protocol: one JSON request per input line, one JSON reply per output line.
The JS side controls source/invocation admission and broker observations.
The model chooses *only* selected semantic witness bytes. No training, reference
execution, or teacher correction occurs in this process.
"""

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
from megatoken.semantic_model_adapter import SelectedSemanticInference
from megatoken.served import ServedBundle


@torch.inference_mode()
def dispatch(adapter, message):
    op = message.get("op")
    if op == "initialize":
        return adapter.initialize(
            seed=message["seed"], checkpointIdentity=message["checkpointIdentity"]
        )
    if op == "select":
        return adapter.select(
            message["prefix"], representation=message.get("representation", "explicit")
        )
    if op == "select-controlled":
        return adapter.select(
            message["prefix"],
            representation=message["representation"],
            controlledRecord=message["record"],
        )
    if op == "outcome":
        return adapter.select_outcome(message["prefix"])
    if op == "open-orchestration":
        return adapter.open_orchestration(
            **{key: value for key, value in message.items() if key not in {"id", "op"}}
        )
    if op == "step-orchestration":
        return adapter.step_orchestration(
            **{key: value for key, value in message.items() if key not in {"id", "op"}}
        )
    if op == "status-orchestration":
        return adapter.status_orchestration(session=message["session"], track=message["track"])
    if op == "cancel-orchestration":
        return adapter.cancel_orchestration(
            session=message["session"], track=message["track"], reason=message["reason"]
        )
    if op == "commit-request":
        return adapter.commit_request(
            message["session"],
            message["invocation"],
            message["selection"],
            message["record"],
            message["validation"],
        )
    if op == "complete":
        return adapter.close_invocation(
            message["session"], message["invocation"], message["prefix"]
        )
    if op == "cancel":
        return adapter.cancel_invocation(message["session"], message["invocation"])
    if op == "checkpoint":
        return adapter.checkpoint(message["session"], message["prefixes"])
    if op == "restore":
        return adapter.restore(message["checkpoint"])
    raise ValueError("Unknown semantic RPC operation")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--served", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--effort", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--max-sessions", type=int, default=32)
    parser.add_argument("--controlled-lifecycle-fixtures", action="store_true")
    args = parser.parse_args()
    bundle = ServedBundle(args.served, device=args.device)
    runtime = MultiTrackRuntime(MultiTrackGenerator.of(bundle.model), chunking=bundle.chunking)
    adapter = SelectedSemanticInference(
        runtime,
        effort=args.effort,
        max_tokens=args.max_tokens,
        max_sessions=args.max_sessions,
        model_identity=bundle.manifest["model_sha256"],
        allow_controlled_lifecycle=args.controlled_lifecycle_fixtures,
    )
    try:
        for line in sys.stdin:
            if not line.strip():
                continue
            request_id = None
            started = time.perf_counter()
            try:
                message = json.loads(line)
                if not isinstance(message, dict) or not isinstance(message.get("id"), str):
                    raise ValueError("RPC needs a stable request ID")
                request_id = message["id"]
                response = dispatch(adapter, message)
                reply = {"id": request_id, "ok": True, "result": response}
            except Exception as error:
                # Invalid model output and unavailable resources are failures,
                # never replaced with a reference or a guessed semantic value.
                reply = {
                    "id": request_id,
                    "ok": False,
                    "error": {
                        "category": type(error).__name__,
                        "message": str(error),
                        "evidence": getattr(error, "evidence", None),
                    },
                }
            reply["runtimeMetrics"] = {
                "schema": "semantic-inference-metrics/4",
                "computeSeconds": time.perf_counter() - started,
                "peakMemoryBytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                * (1 if sys.platform == "darwin" else 1024),
                "scope": "native-process-rpc-wall-time-and-lifetime-peak-rss",
            }
            print(
                json.dumps(reply, ensure_ascii=False, separators=(",", ":"), allow_nan=False),
                flush=True,
            )
    finally:
        runtime.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
