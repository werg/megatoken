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
import sys
from pathlib import Path

from megatoken.model.multitrack import MultiTrackGenerator
from megatoken.multitrack_runtime import MultiTrackRuntime
from megatoken.semantic_model_adapter import SelectedSemanticInference
from megatoken.served import ServedBundle


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--served", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--effort", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--max-sessions", type=int, default=32)
    args = parser.parse_args()
    bundle = ServedBundle(args.served, device=args.device)
    runtime = MultiTrackRuntime(MultiTrackGenerator.of(bundle.model), chunking=bundle.chunking)
    adapter = SelectedSemanticInference(
        runtime, effort=args.effort, max_tokens=args.max_tokens,
        max_sessions=args.max_sessions,
    )
    try:
        for line in sys.stdin:
            if not line.strip():
                continue
            request_id = None
            try:
                message = json.loads(line)
                if not isinstance(message, dict) or not isinstance(message.get("id"), str):
                    raise ValueError("RPC needs a stable request ID")
                request_id = message["id"]
                op = message.get("op")
                if op == "select":
                    response = adapter.select(message["prefix"])
                elif op == "complete":
                    response = adapter.close_invocation(message["session"], message["invocation"])
                elif op == "cancel":
                    response = adapter.cancel_invocation(message["session"], message["invocation"])
                else:
                    raise ValueError("Unknown semantic RPC operation")
                reply = {"id": request_id, "ok": True, "result": response}
            except Exception as error:
                # Invalid model output and unavailable resources are failures,
                # never replaced with a reference or a guessed semantic value.
                reply = {
                    "id": request_id, "ok": False,
                    "error": {"category": type(error).__name__, "message": str(error)},
                }
            print(json.dumps(reply, ensure_ascii=False, separators=(",", ":"), allow_nan=False), flush=True)
    finally:
        runtime.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
