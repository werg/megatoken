#!/usr/bin/env python3
"""Validate exact prepared v7 records, controls, tokenizer and family splits."""

import argparse
import json
from pathlib import Path

from megatoken.generation.codec import LING_TOKENIZER_SHA256, ByteCodec
from megatoken.training.semantic_v7 import validate_prepared_semantic_v7


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, action="append", required=True)
    parser.add_argument("--producer-revision", required=True)
    parser.add_argument("--consumer-revision", required=True)
    parser.add_argument("--tokenizer-json", type=Path, required=True)
    parser.add_argument("--max-positions", type=int, default=8192)
    parser.add_argument("--max-history", type=int, default=1024)
    args = parser.parse_args()
    codec = ByteCodec.from_tokenizer_json(
        args.tokenizer_json, expected_sha256=LING_TOKENIZER_SHA256
    )
    result = validate_prepared_semantic_v7(
        args.prepared,
        codec=codec,
        producer_revision=args.producer_revision,
        consumer_revision=args.consumer_revision,
        max_positions=args.max_positions,
        max_history=args.max_history,
    )
    print(json.dumps(result, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
