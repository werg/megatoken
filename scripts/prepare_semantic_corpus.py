#!/usr/bin/env python3
"""Revalidate admitted interactions and export native whole-family training data."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

from megatoken.data.semantic_corpus import prepare_semantic_corpus
from megatoken.generation.codec import LING_TOKENIZER_SHA256, ByteCodec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        action="append",
        required=True,
        help="Episode JSON or candidate JSONL; repeat for multiple inputs",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tracer-root", type=Path, required=True)
    parser.add_argument("--producer-revision", required=True)
    parser.add_argument("--consumer-revision", required=True)
    parser.add_argument("--tokenizer-json", type=Path, required=True)
    parser.add_argument("--max-positions", type=int, default=1024)
    parser.add_argument("--max-history", type=int, default=1024)
    options = parser.parse_args()
    tracer = options.tracer_root.resolve()
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=tracer, text=True).strip()
    if actual != options.producer_revision:
        parser.error("Tracer checkout must match the exact producer revision")
    verifier = tracer / "tools/verify-effectful-admission.mjs"
    if not verifier.is_file():
        parser.error("Pinned tracer has no independent source-admission verifier")
    codec = ByteCodec.from_tokenizer_json(
        options.tokenizer_json, expected_sha256=LING_TOKENIZER_SHA256
    )

    def source_verifier(document):
        with tempfile.TemporaryDirectory(prefix="semantic-source-") as directory:
            path = Path(directory) / "candidate.json"
            path.write_text(json.dumps(document, ensure_ascii=False, separators=(",", ":")))
            result = subprocess.run(
                ["node", str(verifier), str(path)],
                cwd=tracer,
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            if result.returncode:
                raise ValueError(
                    "Independent source admission rejected candidate: " + result.stderr[-2000:]
                )
            return json.loads(result.stdout)

    def candidates():
        for path in options.input:
            if path.suffix == ".jsonl":
                with path.open() as source:
                    for line in source:
                        if line.strip():
                            yield json.loads(line)
            else:
                yield json.loads(path.read_text())

    manifest, report = prepare_semantic_corpus(
        candidates(),
        options.output,
        codec=codec,
        producer_revision=options.producer_revision,
        consumer_revision=options.consumer_revision,
        source_verifier=source_verifier,
        max_positions=options.max_positions,
        max_history=options.max_history,
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
