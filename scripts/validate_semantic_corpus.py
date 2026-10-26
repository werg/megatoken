#!/usr/bin/env python3
"""Recheck every generated source fixture and its actual production native preparation."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from megatoken.data.semantic_corpus import prepare_semantic_corpus, source_verifier_for_checkout
from megatoken.generation.codec import LING_TOKENIZER_SHA256, ByteCodec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tracer-root", type=Path, required=True)
    parser.add_argument("--tokenizer-json", type=Path, required=True)
    options = parser.parse_args()
    manifest = json.loads((options.corpus / "manifest.json").read_text())
    if manifest["schema"] != "semantic-corpus-manifest/4" or not manifest["cases"]:
        raise ValueError("Missing actual version-four source collection manifest")
    if any(case["status"] != "admitted" for case in manifest["cases"]):
        raise ValueError("Every required source fixture must pass independent admission")
    consumer = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True
    ).strip()
    codec = ByteCodec.from_tokenizer_json(
        options.tokenizer_json, expected_sha256=LING_TOKENIZER_SHA256
    )
    documents = [
        json.loads((options.corpus / case["path"]).read_text()) for case in manifest["cases"]
    ]
    if manifest.get("representation") != "compact" or any(
        document["sidecar"].get("representation") != "compact" for document in documents
    ):
        raise ValueError("Only a complete canonical semantic collection is admissible")
    native, report = prepare_semantic_corpus(
        documents,
        options.output,
        codec=codec,
        producer_revision=manifest["producerRevision"],
        consumer_revision=consumer,
        source_verifier=source_verifier_for_checkout(
            options.tracer_root, manifest["producerRevision"]
        ),
    )
    if report["counts"].get("admitted") != len(documents) or report["refusals"]:
        raise ValueError(
            "Native preparation refused an independently collected required source case"
        )
    print(
        json.dumps(
            {
                "cases": len(manifest["cases"]),
                "manifestFingerprint": native["fingerprint"],
                "optimizerSteps": 0,
                "report": report,
            },
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
