#!/usr/bin/env python3
"""Re-admit v7 source/native evidence and export frozen-family native training shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from admit_semantic_v7 import checked_revision

from megatoken.data.semantic_corpus_v7 import (
    prepare_semantic_corpus_v7,
    source_verifier_for_checkout_v7,
)
from megatoken.generation.codec import LING_TOKENIZER_SHA256, ByteCodec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tracer-root", type=Path, required=True)
    parser.add_argument("--producer-revision", required=True)
    parser.add_argument("--consumer-revision", required=True)
    parser.add_argument("--tokenizer-json", type=Path, required=True)
    parser.add_argument("--max-positions", type=int, default=8192)
    parser.add_argument("--max-history", type=int, default=1024)
    args = parser.parse_args()
    checked_revision(Path(__file__).resolve().parents[1], args.consumer_revision)
    codec = ByteCodec.from_tokenizer_json(
        args.tokenizer_json, expected_sha256=LING_TOKENIZER_SHA256
    )
    verifier = source_verifier_for_checkout_v7(args.tracer_root, args.producer_revision)

    def candidates():
        for path in args.input:
            if path.suffix == ".jsonl":
                for line in path.read_text().splitlines():
                    if line.strip():
                        yield json.loads(line)
            else:
                value = json.loads(path.read_text())
                if isinstance(value, list):
                    yield from value
                else:
                    yield value

    manifest, report = prepare_semantic_corpus_v7(
        candidates(),
        args.output,
        codec=codec,
        producer_revision=args.producer_revision,
        consumer_revision=args.consumer_revision,
        source_verifier=verifier,
        max_positions=args.max_positions,
        max_history=args.max_history,
    )
    print(
        json.dumps(
            {"manifestFingerprint": manifest["fingerprint"], "report": report},
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
