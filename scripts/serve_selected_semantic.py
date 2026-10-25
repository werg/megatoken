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
    parser.add_argument(
        "--profile-output", type=Path, help="Write an immutable measured serving report after EOF"
    )
    parser.add_argument(
        "--profile-origin", choices=("controlled", "fixed", "trained"), default="fixed"
    )
    args = parser.parse_args()
    profile = None
    if args.profile_output is not None:
        if args.profile_output.exists():
            parser.error("Profile reports are immutable; choose a new output path")
        if args.controlled_lifecycle_fixtures and args.profile_origin != "controlled":
            parser.error("Controlled lifecycle fixtures require controlled profile origin")
        from megatoken.semantic_profile import ServingProfile

        profile = ServingProfile(args.served, args.device, origin=args.profile_origin)
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
    if profile is not None:
        profile.loaded(bundle)
    try:
        for line in sys.stdin:
            if not line.strip():
                continue
            request_id = None
            message = None
            if profile is not None:
                profile.synchronize()
            started = time.perf_counter()
            try:
                message = json.loads(line)
                if not isinstance(message, dict) or not isinstance(message.get("id"), str):
                    raise ValueError("RPC needs a stable request ID")
                request_id = message["id"]
                if (
                    profile is not None
                    and profile.origin != "controlled"
                    and message.get("op") == "select-controlled"
                ):
                    raise ValueError("Controlled selections require controlled profile origin")
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
            if profile is not None:
                profile.synchronize()
                profile.record(message, reply, time.perf_counter() - started, adapter)
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
        if profile is not None:
            from megatoken.semantic_profile import state_counts

            profile.before_close = state_counts(adapter)
        closed = False
        try:
            runtime.close()
            closed = True
        finally:
            if profile is not None:
                args.profile_output.parent.mkdir(parents=True, exist_ok=True)
                with args.profile_output.open("x", encoding="utf-8") as handle:
                    json.dump(
                        profile.report(runtime_closed=closed), handle, indent=2, allow_nan=False
                    )
                    handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
