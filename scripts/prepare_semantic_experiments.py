#!/usr/bin/env python3
"""Prepare one canonical semantic execution curriculum with certified lifecycle labels.

Consumes one source-certified training-v4 or benchmark-v1 recipe manifest.
The same selected-harness/2 reasoning trace is used for every source family,
native admission, serving and training. No legacy representation comparisons.
No optimizer is created.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from megatoken.data.host_operations import CONTRACT_DIGEST
from megatoken.data.semantic_corpus import prepare_semantic_corpus, source_verifier_for_checkout
from megatoken.data.semantic_recipe_preparation import (
    benchmark_recipe_preparation_summary,
    validate_recipe_manifest,
)
from megatoken.generation.codec import LING_TOKENIZER_SHA256, ByteCodec
from megatoken.io import write_json
from megatoken.training.semantic_budget import validate_stage_supply


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--tracer-root", type=Path, required=True)
    parser.add_argument("--tokenizer-json", type=Path, required=True)
    parser.add_argument("--producer-revision", required=True)
    parser.add_argument("--consumer-revision", required=True)
    parser.add_argument("--recipe", choices=("training-v4", "benchmark-v1"), default="training-v4")
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    options.output.mkdir(parents=True, exist_ok=True)
    codec = ByteCodec.from_tokenizer_json(
        options.tokenizer_json, expected_sha256=LING_TOKENIZER_SHA256
    )
    source_verify = source_verifier_for_checkout(options.tracer_root, options.producer_revision)
    verified: dict[str, dict] = {}

    def verify(document):
        key = hashlib.sha256(
            json.dumps(document, ensure_ascii=False, separators=(",", ":")).encode()
        ).hexdigest()
        if key not in verified:
            verified[key] = source_verify(document)
        return verified[key]

    root = options.corpus.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("catalogDigest") != CONTRACT_DIGEST:
        raise ValueError("Prepared recipe uses a different host operation catalog")
    summary = validate_recipe_manifest(
        manifest, recipe_name=options.recipe, representation="compact",
        producer_revision=options.producer_revision,
    )
    datasets = {"common-support": {"compact": []}, "effectful-tracks-events": {"compact": []}}
    reports, lifecycle, prepared_counts = [], [], {}
    for cohort, role in (
        ("common-support", "common-support"),
        ("effectful-tracks-events", "effectful"),
    ):
        cases = [case for case in manifest["cases"] if case["role"] == role]
        if not cases:
            raise ValueError("Source curriculum is missing a required semantic behavior")
        output = options.output / (cohort + "-compact")
        _, report = prepare_semantic_corpus(
            (json.loads((root / case["path"]).read_text()) for case in cases),
            output,
            codec=codec,
            producer_revision=options.producer_revision,
            consumer_revision=options.consumer_revision,
            source_verifier=verify,
            representation="compact",
        )
        if report["accepted"] != len(cases) or report["refusals"]:
            raise ValueError("Canonical semantic preparation refused a required source case")
        datasets[cohort]["compact"].append(str(output.resolve()))
        reports.append(report)
        prepared_counts[f"{cohort}/compact"] = report["accepted"]
        print(f"Prepared {cohort}: {report['accepted']} admitted episodes", file=sys.stderr)

    children = manifest["recipe"]["lifecycleInputs"]
    if {child["split"] for child in children} != {"train", "validation"} or len(children) != 2:
        raise ValueError("The source recipe needs real train/validation lifecycle children")
    for child in children:
        output = options.output / ("lifecycle-" + child["split"])
        command = [
            sys.executable,
            str(Path(__file__).with_name("collect_semantic_lifecycle.py")),
            "--input", str(root / child["path"]),
            "--output", str(output),
            "--tracer-root", str(options.tracer_root.resolve()),
            "--producer-revision", options.producer_revision,
            "--consumer-revision", options.consumer_revision,
            "--tokenizer-json", str(options.tokenizer_json.resolve()),
        ]
        completed = subprocess.run(command, check=True, capture_output=True, text=True, timeout=180)
        receipt = json.loads(completed.stdout)["receipt"]
        if receipt["split"] != child["split"]:
            raise ValueError("Lifecycle child changed its source-family split")
        lifecycle.append(receipt)
        datasets["effectful-tracks-events"]["compact"].append(str(output.resolve()))
        print(f"Collected {child['split']} lifecycle teaching", file=sys.stderr)

    supply = []
    for cohort, arms in datasets.items():
        roots = arms["compact"]
        supply.append(validate_stage_supply(
            roots,
            phase="SE" if cohort == "effectful-tracks-events" else "SC",
            representation="compact",
            codec=codec,
            producer_revision=options.producer_revision,
            consumer_revision=options.consumer_revision,
        ))
    receipt = {
        "schema": "semantic-experiment-preparation/4",
        "producerRevision": options.producer_revision,
        "consumerRevision": options.consumer_revision,
        "tokenizerIdentity": codec.fingerprint,
        "datasets": datasets,
        "preparation": reports,
        "lifecycle": lifecycle,
        "stageSupply": supply,
        "optimizerSteps": 0,
        "learnedSuccess": None,
    }
    write_json(options.output / "experiment-preparation.json", receipt)
    if options.recipe == "benchmark-v1":
        write_json(
            options.output / "benchmark-recipe-preparation.json",
            benchmark_recipe_preparation_summary(summary, prepared_counts),
        )
    write_json(
        options.output / "dataset-bindings.json",
        {"datasets": datasets, "codec": str(options.tokenizer_json.resolve())},
    )
    print(json.dumps({"stageSupply": supply, "optimizerSteps": 0}, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
