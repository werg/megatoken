#!/usr/bin/env python3
"""Independently check the actual shared-process engineering orchestration gates.

These deterministic source fixtures retain their original family splits. They
are never relabeled held-out tasks and do not report learned model quality.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from megatoken.data.host_operations import canonical_json
from megatoken.data.semantic_harness_v4 import adapt_selected_harness_v4
from megatoken.generation.codec import LING_TOKENIZER_SHA256, ByteCodec
from megatoken.io import write_json
from megatoken.training.semantic_evaluation import (
    HARD_GATES,
    _orchestration_usage,
    _original_emission_tokens,
    _validate_orchestration,
)


def verify(directory, codec):
    required = {
        "native-services-explicit",
        "native-services-compact",
        "native-budget",
        "native-invalid-child",
        "native-model-rejection",
    }
    files = {path.stem: path for path in directory.glob("*.json")}
    if required - set(files):
        raise ValueError("Missing required actual native orchestration engineering evidence")
    rows = []
    for name, path in sorted(files.items()):
        document = json.loads(path.read_text())
        case, result = document["case"], document["result"]
        if (
            result.get("schema") != "semantic-evaluation-result/4"
            or result.get("optimizerSteps") != 0
            or result.get("hardGates") != dict.fromkeys(HARD_GATES, 0)
            or result["costs"]["leakedResources"] != 0
            or result["outcome"]["cleanup"]["pendingTracks"] != 0
            or result.get("cleanupFailure") is not None
        ):
            raise ValueError(
                name + ": native execution did not preserve every hard engineering gate"
            )
        metrics = result["nativeRuntimeMetrics"]
        if metrics["computeSeconds"] <= 0 or metrics["peakMemoryBytes"] <= 0:
            raise ValueError(name + ": missing actual native work and memory accounting")
        if result["costs"]["peakMemoryBytes"] < metrics["peakMemoryBytes"]:
            raise ValueError(name + ": evaluator omitted its model process memory")
        generated, observed, physical = _orchestration_usage(result, codec)
        expected_refusal = name in {
            "native-budget",
            "native-invalid-child",
            "native-model-rejection",
        }
        if not expected_refusal:
            if (
                result["origin"] != "controlled"
                or result["failure"] is not None
                or result["outcome"] != case["expected"]
            ):
                raise ValueError(
                    name + ": actual controlled task did not match its independent source outcome"
                )
            reference = case["referenceReceipt"]
            if reference["status"] != "agreed" or any(
                reference[key] != hashlib.sha256(canonical_json(value).encode()).hexdigest()
                for key, value in (
                    ("requestDigest", case["request"]),
                    ("outcomeDigest", case["expected"]),
                )
            ):
                raise ValueError(name + ": source reference receipt changed")
            adapt_selected_harness_v4(
                result["episode"],
                codec=codec,
                allow_local=True,
                max_positions=1024,
                max_history=1024,
            )
            _validate_orchestration(result, codec, "controlled")
        else:
            if (
                not result["failure"]
                or result["costs"]["dispatches"] != 0
                or any(
                    track["status"] not in {"completed", "cancelled"}
                    for track in result["orchestration"]["finalState"]["tracks"]
                )
            ):
                raise ValueError(name + ": deliberate refusal was repaired or leaked live work")
            if name == "native-budget" and "decision budget" not in result["failure"]["message"]:
                raise ValueError("Coordinator budget fixture did not reach its declared refusal")
            if name == "native-invalid-child" and not any(
                item.get("phase") == "semantic-validation"
                and item.get("record") == 'return {"value":["ref","missing"]}'
                for item in result["rejections"]
            ):
                raise ValueError("Invalid guest fixture lost its original rejected selection")
        original = None
        if name == "native-model-rejection":
            identity = result["inferenceIdentity"]
            if result["origin"] != "fixed-weights" or identity != {
                "schema": "semantic-inference-identity/4",
                "seed": 17,
                "checkpointIdentity": result["checkpointIdentity"],
                "tokenizerIdentity": codec.fingerprint,
            }:
                raise ValueError("Initialized model fixture lacks its actual seeded identity")
            original = _original_emission_tokens(result, codec, canonical_tokens=generated)
            if original <= 0:
                raise ValueError("Initialized model refusal lost its original emitted token cost")
        rows.append(
            {
                "caseId": case["id"],
                "sourceSplit": case["split"],
                "representation": result["representation"],
                "origin": result["origin"],
                "disposition": "expected-refusal" if expected_refusal else "complete",
                "generatedTokens": original if original is not None else generated,
                "canonicalGeneratedTokens": generated,
                "observedTokens": observed,
                "physicalRecords": physical,
                "nativeRuntimeMetrics": metrics,
                "costs": result["costs"],
                "costScope": result["costScope"],
                "evidenceSha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    return {
        "schema": "semantic-orchestration-engineering/4",
        "passed": True,
        "tokenizerIdentity": codec.fingerprint,
        "cases": rows,
        "optimizerSteps": 0,
        "learnedSuccess": None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    codec = ByteCodec.from_tokenizer_json(
        options.tokenizer_json, expected_sha256=LING_TOKENIZER_SHA256
    )
    result = verify(options.evidence_dir, codec)
    options.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(options.output, result)
    print(
        json.dumps(
            {
                "schema": result["schema"],
                "passed": result["passed"],
                "cases": len(result["cases"]),
                "optimizerSteps": 0,
            }
        )
    )


if __name__ == "__main__":
    main()
