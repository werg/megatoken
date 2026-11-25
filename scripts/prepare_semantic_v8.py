#!/usr/bin/env python3
"""Freshly recheck source/native V8 evidence and export ordinary training shards."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

from admit_semantic_v8 import checked_revision, recheck_source_binding

from megatoken.data.semantic_corpus_v8 import prepare_semantic_corpus_v8
from megatoken.data.semantic_json import stable_algorithm_hash
from megatoken.generation.codec import LING_TOKENIZER_SHA256, ByteCodec


def _write_source_recheck_artifact(path: Path, document: dict) -> None:
    """Preserve JS property order and lone UTF-16 code units in source evidence."""
    path.write_bytes(
        json.dumps(document, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode(
            "utf-8"
        )
    )


def _documents(paths):
    for path in paths:
        if path.suffix == ".jsonl":
            for line in path.read_text().splitlines():
                if line.strip():
                    yield json.loads(line)
        else:
            value = json.loads(path.read_text())
            yield from value if isinstance(value, list) else [value]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--producer-root", type=Path, required=True)
    parser.add_argument("--producer-revision", required=True)
    parser.add_argument("--consumer-revision", required=True)
    parser.add_argument("--tokenizer-json", type=Path, required=True)
    parser.add_argument("--max-positions", type=int, default=8192)
    parser.add_argument("--max-history", type=int, default=1024)
    parser.add_argument("--source-recheck-timeout-seconds", type=int, default=30)
    args = parser.parse_args()
    consumer = Path(__file__).resolve().parents[1]
    checked_revision(consumer, args.consumer_revision)
    codec = ByteCodec.from_tokenizer_json(
        args.tokenizer_json, expected_sha256=LING_TOKENIZER_SHA256
    )

    def verify(document):
        with tempfile.TemporaryDirectory(prefix="semantic-source-v8-") as directory:
            artifact = Path(directory) / "artifact.json"
            _write_source_recheck_artifact(artifact, document)
            receipt = recheck_source_binding(
                document,
                artifact,
                args.producer_root,
                args.producer_revision,
                timeout_seconds=args.source_recheck_timeout_seconds,
            )
            if receipt is None:
                raise ValueError("Pinned V8 source replay did not return a receipt")
            check_js = """
import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {pathToFileURL} from 'node:url';
const root=resolve(process.argv[1]), path=resolve(process.argv[2]);
const {verifyRegionDatasetBinding}=await import(pathToFileURL(resolve(root,'src/semantic/family-registry-v8.js')));
const result=verifyRegionDatasetBinding(JSON.parse(readFileSync(path,'utf8')));
process.stdout.write(JSON.stringify(result));
"""
            family = subprocess.run(
                [
                    "node",
                    "--input-type=module",
                    "-e",
                    check_js,
                    str(args.producer_root),
                    str(artifact),
                ],
                text=True,
                capture_output=True,
                timeout=60,
                check=False,
            )
            if family.returncode:
                raise ValueError(
                    "Pinned producer rejected V8 family/split binding: " + family.stderr[-2000:]
                )
            verified = json.loads(family.stdout)
            binding = document["sidecar"]["datasetBinding"]
            if (
                verified.get("valid") is not True
                or verified.get("registryDigest") != binding["registryDigest"]
            ):
                raise ValueError("Pinned producer did not validate the exact V8 family registry")
            receipt.update(
                familyBindingValidated=True,
                registryDigest=verified["registryDigest"],
                datasetBindingDigest=stable_algorithm_hash(binding),
            )
            return receipt

    manifest, report = prepare_semantic_corpus_v8(
        _documents(args.input),
        args.output,
        codec=codec,
        producer_revision=args.producer_revision,
        consumer_revision=args.consumer_revision,
        source_verifier=verify,
        tokenizer_evidence={
            "schema": "semantic-tokenizer-admission/1",
            "status": "passed",
            "identity": codec.fingerprint,
            "sha256": LING_TOKENIZER_SHA256,
            "verification": "pinned-tokenizer-bytes",
        },
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
