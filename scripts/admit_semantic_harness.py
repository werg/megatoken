#!/usr/bin/env python3
"""Independently verify a v4 semantic episode using the pinned production tokenizer."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from megatoken.data.semantic_admission import native_admission
from megatoken.generation.codec import LING_TOKENIZER_SHA256, ByteCodec
from megatoken.io import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer-json", type=Path, required=True)
    parser.add_argument("--consumer-revision")
    parser.add_argument("--max-positions", type=int, default=1024)
    parser.add_argument("--max-history", type=int, default=1024)
    options = parser.parse_args()
    if options.input.resolve() == options.output.resolve():
        parser.error("Input and native receipt must use distinct paths")
    revision = (
        options.consumer_revision
        or subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True
        ).strip()
    )
    codec = ByteCodec.from_tokenizer_json(
        options.tokenizer_json, expected_sha256=LING_TOKENIZER_SHA256
    )
    result = native_admission(
        json.loads(options.input.read_text()),
        codec=codec,
        consumer_revision=revision,
        max_positions=options.max_positions,
        max_history=options.max_history,
    )
    options.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(options.output, result)
    print(json.dumps(result, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
