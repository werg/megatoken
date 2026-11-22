#!/usr/bin/env python3
"""Verify ordinary training exposed every positive-loss V8 target in an inventory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from megatoken.data.semantic_json import stable_algorithm_hash
from megatoken.semantic_lineage_v8 import checkpoint_lineage_v8
from megatoken.training.checkpoint import load_checkpoint
from megatoken.training.semantic_v8 import validate_semantic_v8_target_exposure
from megatoken.training.target_inventory_v8 import compare_target_inventory


def read_json(path: Path):
    return json.loads(path.read_text())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--expected-source-ids", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from qualify_semantic_v8_target_inventory import target_inventory

    checkpoint, metadata = load_checkpoint(args.checkpoint, map_location="cpu")
    lineage = checkpoint_lineage_v8(checkpoint, metadata)
    policy = checkpoint["train_config"].get("semantic_v8_policy")
    if not isinstance(policy, dict):
        raise ValueError("Coverage checkpoint has no explicit V8 policy")
    registry_digest = policy["registryDigest"]
    prepared_manifest = read_json(args.prepared / "manifest.json")
    derived = target_inventory(checkpoint["train_config"], args.expected_source_ids)
    supplied = read_json(args.inventory)
    state = validate_semantic_v8_target_exposure(
        checkpoint.get("semantic_v8_target_exposure"),
        registry_digest=registry_digest,
    )
    comparison = compare_target_inventory(
        supplied,
        state["sources"],
        rederived_inventory=derived,
        registry_digest=registry_digest,
        prepared_manifest_fingerprint=prepared_manifest["fingerprint"],
        codec_fingerprint=prepared_manifest["codec_fingerprint"],
        training_config_digest=derived["trainingConfigDigest"],
        model_config_digest=derived["modelConfigDigest"],
    )
    if state["startingGlobalStep"] != 0 or state["optimizerUpdates"] != checkpoint["global_step"]:
        raise ValueError("Coverage checkpoint does not retain a fresh complete optimizer census")
    if checkpoint["global_step"] != derived["windowCount"]:
        raise ValueError(
            "Coverage training stopped before the one-pass optimizer-update bound: "
            f"expected {derived['windowCount']}, got {checkpoint['global_step']}"
        )
    if not comparison["allPositiveWeightTargetsExposed"]:
        raise ValueError(
            "Ordinary training did not expose the complete positive-loss target inventory: "
            f"{len(comparison['missingPositiveWeightPhysicalTargets'])} missing, "
            f"{len(comparison['unexpectedPhysicalTargets'])} unexpected"
        )
    body = {
        "schema": "semantic-v8-ordinary-target-coverage/1",
        "consumerRevision": metadata.get("consumer_revision"),
        "checkpointStateSha256": metadata["state_sha256"],
        "globalStep": checkpoint["global_step"],
        "trainingLineage": lineage,
        "preparedManifestFingerprint": prepared_manifest["fingerprint"],
        "inventory": {
            "digest": derived["digest"],
            "windowCount": derived["windowCount"],
            "sourceIds": derived["sourceIds"],
            "available": derived["available"],
            "positiveWeight": derived["positiveWeight"],
        },
        "optimizerTargetExposure": {
            "digest": state["digest"],
            "previousDigest": state["previousDigest"],
            "sourceCount": len(state["sources"]),
            "sources": state["sources"],
        },
        "comparison": comparison,
        "targetPresenceCoverage": "complete: every positive-weight logical/physical target record contributed at least once",
        "tokenOrMaskPositionCoverage": "unmeasured",
        "learnedSuccess": None,
    }
    body["digest"] = stable_algorithm_hash(body)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(body, indent=2, allow_nan=False) + "\n")
    print("V8_TARGET_COVERAGE_EVIDENCE_JSON " + json.dumps(body, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
