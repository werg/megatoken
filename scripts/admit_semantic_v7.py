#!/usr/bin/env python3
"""Admit only strict v7 execution using a clean exact checkout and real tokenizer."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import megatoken
from megatoken.data.semantic_admission_v7 import native_admission_v7
from megatoken.generation.codec import LING_TOKENIZER_SHA256, ByteCodec
from megatoken.io import write_json


def checked_revision(root, expected):
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if (
        actual != expected
        or subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=root, text=True
        ).strip()
    ):
        raise ValueError("Native v7 admission requires its clean exact consumer commit")
    extra = subprocess.check_output(
        ["git", "ls-files", "--others", "-z", "--", "src", "scripts"], cwd=root, text=True
    ).split("\0")
    if any(
        Path(name).suffix in {".py", ".pyc", ".pyo", ".so", ".pyd"}
        and "__pycache__" not in Path(name).parts
        for name in extra
        if name
    ):
        raise ValueError("Untracked native code could shadow the pinned v7 implementation")
    if not Path(megatoken.__file__).resolve().is_relative_to(root / "src"):
        raise ValueError("Python imported a different consumer checkout")
    return actual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer-json", type=Path, required=True)
    parser.add_argument("--consumer-revision", required=True)
    parser.add_argument("--max-positions", type=int, default=8192)
    parser.add_argument("--max-history", type=int, default=1024)
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error("Input and receipt paths must differ")
    root = Path(__file__).resolve().parents[1]
    revision = checked_revision(root, args.consumer_revision)
    codec = ByteCodec.from_tokenizer_json(
        args.tokenizer_json, expected_sha256=LING_TOKENIZER_SHA256
    )
    receipt = native_admission_v7(
        json.loads(args.input.read_text()),
        codec=codec,
        consumer_revision=revision,
        max_positions=args.max_positions,
        max_history=args.max_history,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, receipt)
    print(json.dumps(receipt, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
