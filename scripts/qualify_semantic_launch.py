#!/usr/bin/env python3
"""Qualify real semantic launch inputs without starting a learner or optimizer."""

import argparse
import json
from pathlib import Path

from megatoken.training.semantic_launch import REPOSITORIES, qualify_launch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--repositories-root",
        type=Path,
        required=True,
        help="Directory containing the three clean repository checkouts",
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--bootstrap-checkpoint",
        type=Path,
        help="Previous-stage checkpoint used to initialize a NEW coordinator; not worker resume",
    )
    parser.add_argument(
        "--bootstrap-receipt",
        type=Path,
        help="Local semantic-bootstrap-provenance.json emitted by the exact coordinator initialization",
    )
    parser.add_argument(
        "--probe-coordinator",
        action="store_true",
        help="Authenticated GET /v1/manifest and /v1/status; no lease or writes",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Qualify CPU execution only; never implies GPU memory fit",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Qualification reports are immutable; choose a new output path")
    report = qualify_launch(
        args.config,
        repositories={name: args.repositories_root / name for name in REPOSITORIES},
        resume=args.resume,
        bootstrap_checkpoint=args.bootstrap_checkpoint,
        bootstrap_receipt=args.bootstrap_receipt,
        probe_coordinator=args.probe_coordinator,
        require_cuda=not args.allow_cpu,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "preflightReady": report["preflightReady"],
                "report": str(args.output),
                "blocked": [k for k, v in report["gates"].items() if v["status"] != "passed"],
                "optimizerSteps": 0,
            }
        )
    )
    return 0 if report["preflightReady"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
