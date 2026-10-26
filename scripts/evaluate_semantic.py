#!/usr/bin/env python3
"""Freeze semantic cohorts or evaluate production runner outputs without training."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from megatoken.generation.codec import LING_TOKENIZER_SHA256, ByteCodec
from megatoken.io import write_text_atomic
from megatoken.training.semantic_evaluation import (
    compare_experiments,
    evaluate_cohort,
    freeze_cohort,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--declaration", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    evaluate = commands.add_parser("run")
    evaluate.add_argument("--cohort", type=Path, required=True)
    evaluate.add_argument(
        "--runner-argv",
        required=True,
        help="JSON array of executable and arguments; no shell expansion",
    )
    evaluate.add_argument(
        "--origin", choices=("controlled", "fixed-weights", "trained"), required=True
    )
    evaluate.add_argument("--checkpoint-identity")
    evaluate.add_argument("--seed", type=int, choices=(17, 29, 43), required=True)
    evaluate.add_argument("--tokenizer-json", type=Path, required=True)
    evaluate.add_argument("--timeout", type=float, default=60)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--require-engineering-success", action="store_true")
    compare = commands.add_parser("compare")
    compare.add_argument("--report", type=Path, action="append", required=True)
    compare.add_argument("--budget", type=Path, action="append", required=True)
    compare.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    if options.command == "freeze":
        declaration = json.loads(options.declaration.read_text())
        result = freeze_cohort(**declaration)
    elif options.command == "compare":
        result = compare_experiments(
            [json.loads(path.read_text()) for path in options.report],
            [json.loads(path.read_text()) for path in options.budget],
        )
    else:
        argv = json.loads(options.runner_argv)
        if (
            not isinstance(argv, list)
            or not argv
            or any(not isinstance(value, str) for value in argv)
        ):
            parser.error("Runner arguments must be a nonempty string array")
        codec = ByteCodec.from_tokenizer_json(
            options.tokenizer_json, expected_sha256=LING_TOKENIZER_SHA256
        )

        def runner(request):
            try:
                completed = subprocess.run(
                    argv,
                    input=json.dumps(request),
                    capture_output=True,
                    text=True,
                    timeout=options.timeout,
                    check=False,
                )
            except subprocess.TimeoutExpired as error:
                raise TimeoutError("Production selected-runner deadline exceeded") from error
            if completed.returncode:
                raise RuntimeError("Production selected runner failed: " + completed.stderr[-2000:])
            return json.loads(completed.stdout)

        result = evaluate_cohort(
            json.loads(options.cohort.read_text()),
            runner,
            codec=codec,
            seed=options.seed,
            origin=options.origin,
            checkpoint_identity=options.checkpoint_identity,
        )
    options.output.parent.mkdir(parents=True, exist_ok=True)
    # These artifacts bind JavaScript-compatible insertion-order JSON. Sorting
    # nested request or outcome keys would invalidate the frozen source receipts.
    write_text_atomic(
        options.output, json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            result.get("summary", {"schema": result["schema"], "digest": result.get("digest")}),
            separators=(",", ":"),
        )
    )
    if options.command == "run" and options.require_engineering_success:
        if options.origin != "controlled":
            parser.error("Engineering success requires explicitly controlled selections")
        if not all(
            row["wholeTaskSuccess"]
            and row["traceValid"]
            and not any(row.get("hardGates", {}).values())
            and row.get("costs", {}).get("leakedResources") == 0
            for row in result["cases"]
        ):
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
