#!/usr/bin/env python3
"""Explicit hardware gate; invoked manually or by the Blackwell CI job.

The test image and a real Blackwell device are required. Results describe actual
measured work; the full primary vocabulary additionally requires a pinned codec.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from megatoken.benchmark import benchmark
from megatoken.runtime import doctor


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("validation"))
    parser.add_argument("--codec", type=Path, help="Pinned primary codec directory")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument(
        "--compile", action="store_true", help="Also validate tensor operator compilation"
    )
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    args.output.mkdir(parents=True, exist_ok=True)

    def save(name: str, value: dict) -> None:
        (args.output / f"{name}.json").write_text(
            json.dumps(value, indent=2, allow_nan=False) + "\n"
        )

    diagnostics = doctor(require_cuda=True, require_blackwell=True)
    save("doctor", diagnostics)
    if not diagnostics["ok"]:
        print("Blackwell preflight failed: " + "; ".join(diagnostics["errors"]), file=sys.stderr)
        return 1
    # Do not allow pytest.importorskip to turn a missing CUDA backend into a pass.
    import triton  # noqa: F401

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_kernels_grouped.py",
            "tests/test_kernels_recurrent.py",
            "tests/test_model_policy_integration.py",
            "tests/test_generation_distributions.py",
            f"--junitxml={args.output / 'cuda-tests.xml'}",
        ],
        check=True,
    )
    mode = "default" if args.compile else None
    for backend in ("torch", "triton"):
        for backward in (False, True):
            name = f"tiny-{backend}-{'training' if backward else 'inference'}"
            report = benchmark(
                config="tiny",
                codec="byte",
                device="cuda",
                precision="bf16",
                backend=backend,
                backward=backward,
                batch=2 if backward else 1,
                steps=args.steps,
                warmup=1,
                capacity=16,
                record_bytes=128,
                history_records=1,
                compile_mode=mode,
            )
            save(name, report)
        full_vocabulary = args.codec is not None
        report = benchmark(
            config="v13",
            codec=str(args.codec) if full_vocabulary else "byte",
            device="cuda",
            precision="bf16",
            backend=backend,
            backward=True,
            batch=1,
            steps=args.steps,
            warmup=1,
            capacity=64,
            record_bytes=256,
            history_records=1,
            core_only=not full_vocabulary,
            compile_mode=mode,
        )
        save(f"v13-{backend}-{'generator' if full_vocabulary else 'core'}", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
