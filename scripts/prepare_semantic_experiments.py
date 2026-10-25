#!/usr/bin/env python3
"""Prepare every semantic experiment arm and verify train/validation supply.

Consumes the two source-certified training-v4 recipe manifests. It rechecks
source/native admission, collects real runtime lifecycle labels for each split,
and verifies the configured curriculum without creating an optimizer.
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
from megatoken.generation.codec import LING_TOKENIZER_SHA256, ByteCodec
from megatoken.io import write_json
from megatoken.training.semantic_budget import validate_stage_supply


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--explicit-corpus", type=Path, required=True)
    parser.add_argument("--compact-corpus", type=Path, required=True)
    parser.add_argument("--tracer-root", type=Path, required=True)
    parser.add_argument("--tokenizer-json", type=Path, required=True)
    parser.add_argument("--producer-revision", required=True)
    parser.add_argument("--consumer-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    options.output.mkdir(parents=True, exist_ok=True)
    codec = ByteCodec.from_tokenizer_json(
        options.tokenizer_json, expected_sha256=LING_TOKENIZER_SHA256
    )
    source_verify = source_verifier_for_checkout(options.tracer_root, options.producer_revision)
    verified = {}

    def verify(document):
        key = hashlib.sha256(
            json.dumps(document, ensure_ascii=False, separators=(",", ":")).encode()
        ).hexdigest()
        if key not in verified:
            verified[key] = source_verify(document)
        return verified[key]

    manifests = {}
    for representation in ("explicit", "compact"):
        root = getattr(options, representation + "_corpus").resolve()
        manifest = json.loads((root / "manifest.json").read_text())
        if (
            manifest.get("schema") != "semantic-corpus-manifest/4"
            or manifest.get("producerRevision") != options.producer_revision
            or manifest.get("catalogDigest") != CONTRACT_DIGEST
            or manifest.get("representation") != representation
            or manifest.get("recipe", {}).get("name") != "training-v4"
            or manifest["recipe"].get("retainEveryAuthoredFamily") is not True
            or any(case.get("status") != "admitted" for case in manifest["cases"])
        ):
            raise ValueError("Complete source/native admitted training-v4 recipe manifest required")
        declared = {case["family"] for case in manifest["cases"]}
        if declared != set(manifest["recipe"]["families"]):
            raise ValueError("Prepared recipe dropped or relabeled an authored source family")
        manifests[representation] = (root, manifest)
    left, right = (manifests[name][1] for name in ("explicit", "compact"))
    if left["recipe"]["templateDigest"] != right["recipe"]["templateDigest"] or {
        (case["id"], case["family"], case["split"], case["role"]) for case in left["cases"]
    } != {(case["id"], case["family"], case["split"], case["role"]) for case in right["cases"]}:
        raise ValueError("Explicit and compact recipe families/splits are not paired")

    datasets = {"common-support": {}, "effectful-tracks-events": {}}
    reports, lifecycle = [], []
    for representation, (root, manifest) in manifests.items():
        for cohort, role in (
            ("common-support", "common-support"),
            ("effectful-tracks-events", "effectful"),
        ):
            cases = [case for case in manifest["cases"] if case["role"] == role]
            if not cases:
                raise ValueError("Recipe is missing a declared experiment arm")
            representations = [representation]
            if representation == "explicit" and cohort == "common-support":
                representations.append("outcome-only")
            for output_representation in representations:
                output = options.output / (cohort + "-" + output_representation)
                _, report = prepare_semantic_corpus(
                    (json.loads((root / case["path"]).read_text()) for case in cases),
                    output,
                    codec=codec,
                    producer_revision=options.producer_revision,
                    consumer_revision=options.consumer_revision,
                    source_verifier=verify,
                    representation=output_representation,
                )
                if report["accepted"] != len(cases) or report["refusals"]:
                    raise ValueError("Configured source recipe preparation refused a required case")
                datasets[cohort][output_representation] = [str(output.resolve())]
                reports.append(report)
                print(
                    f"Prepared {cohort}/{output_representation}: {report['accepted']} admitted episodes",
                    file=sys.stderr,
                )
        children = manifest["recipe"]["lifecycleInputs"]
        if {child["split"] for child in children} != {"train", "validation"} or len(children) != 2:
            raise ValueError(
                "Recipe must supply distinct certified train and validation lifecycle children"
            )
        for child in children:
            output = options.output / ("lifecycle-" + representation + "-" + child["split"])
            command = [
                sys.executable,
                str(Path(__file__).with_name("collect_semantic_lifecycle.py")),
                "--input",
                str(root / child["path"]),
                "--output",
                str(output),
                "--tracer-root",
                str(options.tracer_root.resolve()),
                "--producer-revision",
                options.producer_revision,
                "--consumer-revision",
                options.consumer_revision,
                "--tokenizer-json",
                str(options.tokenizer_json.resolve()),
                "--representation",
                representation,
            ]
            completed = subprocess.run(
                command, check=True, capture_output=True, text=True, timeout=180
            )
            receipt = json.loads(completed.stdout)["receipt"]
            if receipt["split"] != child["split"]:
                raise ValueError("Collected lifecycle child changed its frozen source split")
            lifecycle.append(receipt)
            datasets["effectful-tracks-events"][representation].append(str(output.resolve()))
            print(
                f"Collected {representation}/{child['split']} lifecycle teaching", file=sys.stderr
            )
    supply = []
    for cohort, representations in datasets.items():
        for representation, roots in representations.items():
            phase = (
                "SE"
                if cohort == "effectful-tracks-events"
                else "SO"
                if representation == "outcome-only"
                else "SC"
            )
            supply.append(
                validate_stage_supply(
                    roots,
                    phase=phase,
                    representation=representation,
                    codec=codec,
                    producer_revision=options.producer_revision,
                    consumer_revision=options.consumer_revision,
                )
            )
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
    write_json(
        options.output / "dataset-bindings.json",
        {"datasets": datasets, "codec": str(options.tokenizer_json.resolve())},
    )
    print(json.dumps({"stageSupply": supply, "optimizerSteps": 0}, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
