#!/usr/bin/env python3
"""Verify ordinary training exposed every positive-loss V8 target in an inventory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from megatoken.data import PreparedCorpus
from megatoken.data.semantic_json import stable_algorithm_hash
from megatoken.loading import build_model, load_model_config
from megatoken.semantic_lineage_v8 import checkpoint_lineage_v8
from megatoken.training.checkpoint import load_checkpoint
from megatoken.training.config import TrainConfig
from megatoken.training.positions_v8 import (
    compare_position_inventory,
    independent_position_inventory,
    installed_modality_ids,
    position_exposure_report,
)
from megatoken.training.semantic_v8 import (
    load_semantic_v8_codec,
    validate_semantic_v8_target_exposure,
)
from megatoken.training.target_inventory_v8 import compare_target_inventory


def read_json(path: Path):
    return json.loads(path.read_text())


def prepared_position_inventory(config_raw, *, source_ids, prepared_manifest_fingerprint):
    """Read admitted physical data directly, separately from the window preflight."""
    config = TrainConfig.from_dict(config_raw)
    model_config = load_model_config(
        config.model,
        **{
            **config.model_overrides,
            "expert_backend": config.expert_backend,
            "checkpoint_stages": config.gradient_checkpointing,
        },
    )
    codec = load_semantic_v8_codec(config.codec, vocab_size=model_config.vocab_size)
    model = build_model(
        model_config,
        codec,
        device="cpu",
        precision=config.precision,
        packs=config.packs,
    )
    with PreparedCorpus(config.datasets, split="train") as corpus:
        if corpus.codec_fingerprint != codec.fingerprint:
            raise ValueError("Position inventory corpus tokenizer differs from the checkpoint")
        return independent_position_inventory(
            (corpus.fetch(index)[0] for index in range(len(corpus))),
            codec=codec,
            config=config,
            model_config=model_config,
            modalities=installed_modality_ids(model),
            prepared_manifest_fingerprint=prepared_manifest_fingerprint,
            expected_source_ids=source_ids,
        )


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
    if checkpoint["global_step"] != derived["optimizerWindowCount"]:
        raise ValueError(
            "Coverage training stopped before the one-pass optimizer-update bound: "
            f"expected {derived['optimizerWindowCount']}, got {checkpoint['global_step']}"
        )
    if not comparison["exactOnePassCensus"]:
        raise ValueError(
            "Ordinary training differs from the exact one-pass target census: "
            f"{len(comparison['missingPositiveWeightPhysicalTargets'])} missing keys, "
            f"{len(comparison['scoredOccurrenceMismatches'])} occurrence mismatches, "
            f"{len(comparison['windowIndexMismatches'])} window-set mismatches, "
            f"{len(comparison['positiveWeightDenominatorMismatches'])} denominator mismatches"
        )
    position_inventory = prepared_position_inventory(
        checkpoint["train_config"],
        source_ids=derived["sourceIds"],
        prepared_manifest_fingerprint=prepared_manifest["fingerprint"],
    )
    position_state = checkpoint.get("semantic_v8_position_exposure")
    position_comparison = compare_position_inventory(
        position_inventory,
        position_state,
        target_exposure=state,
        global_step=checkpoint["global_step"],
    )
    if metadata.get("semantic_v8_position_digest") != position_state["digest"]:
        raise ValueError("Checkpoint metadata does not bind the executed position history")
    if not position_comparison["exactOnePassPreparedTextPositions"]:
        raise ValueError(
            "Ordinary training differs from independently reconstructed prepared positions: "
            + json.dumps(position_comparison, sort_keys=True, separators=(",", ":"))
        )
    body = {
        "schema": "semantic-v8-ordinary-target-coverage/1",
        "consumerRevision": policy["consumerRevision"],
        "checkpointStateSha256": metadata["state_sha256"],
        "globalStep": checkpoint["global_step"],
        "trainingLineage": lineage,
        "preparedManifestFingerprint": prepared_manifest["fingerprint"],
        "inventory": {
            "digest": derived["digest"],
            "windowCount": derived["windowCount"],
            "optimizerWindowCount": derived["optimizerWindowCount"],
            "phaseMeter": derived["phaseMeter"],
            "phaseBudget": derived["phaseBudget"],
            "onePassMeterCount": derived["onePassMeterCount"],
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
        "independentPreparedPositionInventory": position_inventory,
        "optimizerPositionExposure": position_state,
        "optimizerPositionSummary": position_exposure_report(position_state),
        "positionComparison": position_comparison,
        "targetPresenceCoverage": "complete: every positive-weight logical/physical target record contributed at least once",
        "tokenOrMaskPositionCoverage": "complete-for-prepared-text",
        "routerInternalPositionCoverage": "unmeasured",
        "learnedSuccess": None,
    }
    if comparison["allPositiveWeightTargetsExposed"] is not True:
        raise ValueError("Exact one-pass census is missing one or more target keys")
    body["digest"] = stable_algorithm_hash(body)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(body, indent=2, allow_nan=False) + "\n")
    print(
        "V8_TARGET_COVERAGE_EVIDENCE_JSON "
        + json.dumps(body, sort_keys=True, separators=(",", ":"))
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
