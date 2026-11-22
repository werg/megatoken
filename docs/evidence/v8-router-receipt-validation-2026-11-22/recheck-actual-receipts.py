"""Revalidate unchanged actual 2f3e6cf fixture receipts with the stricter reader.

No optimizer, source admission or decoder runs. Fixture source/tokenizer
boundaries remain injected. Original temporary prepared roots must exist.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import torch

from megatoken.data import PreparedCorpus
from megatoken.generation.codec import ByteCodec
from megatoken.loading import load_model_config
from megatoken.training.checkpoint import load_checkpoint
from megatoken.training.config import TrainConfig
from megatoken.training.positions_v8 import (
    advance_position_exposure,
    compare_position_inventory,
    empty_position_exposure,
    independent_position_inventory,
    position_exposure_report,
    position_objective,
    qualify_position_history,
    restore_position_exposure,
)
from megatoken.training.semantic_v8 import validate_semantic_v8_target_exposure


def read(path):
    return json.loads(path.read_bytes())


def independent(config_raw, expected):
    config = TrainConfig.from_dict(config_raw)
    model = load_model_config(
        config.model,
        **{
            **config.model_overrides,
            "expert_backend": config.expert_backend,
            "checkpoint_stages": config.gradient_checkpointing,
        },
    )
    codec = ByteCodec.byte_fallback()
    manifest = read(Path(config.datasets[0]) / "manifest.json")
    assert manifest["fingerprint"] == expected["preparedManifestFingerprint"]
    with PreparedCorpus(config.datasets, split="train") as corpus:
        inventory = independent_position_inventory(
            (corpus.fetch(index)[0] for index in range(len(corpus))),
            codec=codec,
            config=config,
            model_config=model,
            modalities=[0],
            prepared_manifest_fingerprint=manifest["fingerprint"],
            expected_source_ids=expected["sourceIds"],
        )
    assert inventory == expected
    return inventory


def target_state(summary):
    """Invert the actual logged target summary, retaining its original seal."""
    state = {
        key: summary[key]
        for key in (
            "schema",
            "registryDigest",
            "previousDigest",
            "startingGlobalStep",
            "lastGlobalStep",
            "optimizerUpdates",
            "digest",
        )
    }
    state["sources"] = {
        source_id: {
            "split": source["split"],
            "logicalTargets": {
                logical_id: {"records": records}
                for logical_id, records in source["physicalTargetRecords"].items()
            },
        }
        for source_id, source in summary["sources"].items()
    }
    # The original digest must match exactly. No receipt data is supplied from
    # the position history and no replacement digest is computed or installed.
    return validate_semantic_v8_target_exposure(state, registry_digest=state["registryDigest"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controls-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    complete_root = args.controls_root / "test_full_controlled_corpus_ex0"
    resume_root = args.controls_root / "test_v8_normal_update_checkpoi0"
    complete = read(complete_root / "position-proof.json")
    resume = read(resume_root / "position-resume-proof.json")
    actual_source = "2f3e6cf04aad60473e8ecc67e846e81e5b9dec83"
    assert complete["sourceCommit"] == resume["sourceCommit"] == actual_source
    inventory = independent(complete["config"], complete["independentPreparedPositionInventory"])
    before_stage = complete["optimizerPositionExposure"]
    restored = restore_position_exposure(
        {"train_config": complete["config"], "semantic_v8_position_exposure": before_stage},
        registry_digest=inventory["registryDigest"],
        objective=inventory["objective"],
        global_step=complete["optimizerUpdates"],
        checkpoint_metadata=complete["checkpointMetadata"],
    )
    assert restored == before_stage
    comparison = compare_position_inventory(
        inventory,
        restored,
        target_exposure=complete["targetExposure"],
        global_step=complete["optimizerUpdates"],
    )
    assert comparison == complete["comparison"]
    assert comparison["exactOnePassPreparedTextPositions"] is True
    stage = complete["phaseTransition"]["positionExposure"]
    stage_config_raw = read(complete_root / "fit-next-stage/config.json")
    stage_config = TrainConfig.from_dict(stage_config_raw)
    model = load_model_config(
        stage_config.model,
        **{
            **stage_config.model_overrides,
            "expert_backend": stage_config.expert_backend,
            "checkpoint_stages": stage_config.gradient_checkpointing,
        },
    )
    stage_objective = position_objective(stage_config, model, ByteCodec.byte_fallback(), [0])
    assert stage["objective"] == stage_objective
    restored_stage = restore_position_exposure(
        {"train_config": stage_config_raw, "semantic_v8_position_exposure": stage},
        registry_digest=inventory["registryDigest"],
        objective=stage_objective,
        global_step=stage["lastGlobalStep"],
        checkpoint_metadata=complete["phaseTransition"]["checkpointMetadata"],
    )
    assert restored_stage == stage and stage["previousStage"] == before_stage
    fresh_stage = restore_position_exposure(
        {"train_config": complete["config"], "semantic_v8_position_exposure": before_stage},
        registry_digest=inventory["registryDigest"],
        objective=stage_objective,
        global_step=before_stage["lastGlobalStep"],
        phase_transition=True,
        checkpoint_metadata=complete["checkpointMetadata"],
    )
    assert fresh_stage["previousStage"] == before_stage
    assert fresh_stage["routerInternalPositionCounts"] == {}
    assert fresh_stage["routerInternalAssignmentCounts"] == {}
    assert fresh_stage["routerInternalPositionAuditUpdates"] == 0

    latest = read(resume_root / "fit/checkpoints/latest.json")
    payload, metadata = load_checkpoint(resume_root / "fit/checkpoints" / latest["checkpoint"])
    assert payload["semantic_v8_position_exposure"] == resume["resumedPositionExposure"]
    assert metadata == resume["resumedCheckpointMetadata"]
    resume_inventory = independent(
        payload["train_config"], resume["independentPreparedPositionInventory"]
    )
    finished = {
        row["global_step"]: row
        for row in map(json.loads, (resume_root / "fit/metrics.jsonl").read_text().splitlines())
        if row.get("event") == "finished"
    }
    presence_before = target_state(finished[1]["semantic_v8_target_exposure"])
    presence_after = target_state(finished[2]["semantic_v8_target_exposure"])
    assert presence_after == payload["semantic_v8_target_exposure"]
    first = {
        **payload,
        "global_step": 1,
        "semantic_v8_position_exposure": resume["firstPositionExposure"],
        "semantic_v8_target_exposure": presence_before,
    }
    for saved, saved_metadata in ((first, resume["firstCheckpointMetadata"]), (payload, metadata)):
        result = restore_position_exposure(
            saved,
            registry_digest=resume_inventory["registryDigest"],
            objective=resume_inventory["objective"],
            global_step=saved["global_step"],
            checkpoint_metadata=saved_metadata,
        )
        assert result == saved["semantic_v8_position_exposure"]
    history = qualify_position_history(
        first,
        payload,
        inventory=resume_inventory,
        first_metadata=resume["firstCheckpointMetadata"],
        resumed_metadata=metadata,
    )
    assert history == resume["qualification"]
    actual_router_updates = 0
    for fit, objective, registry_digest, starting_step in (
        (complete_root / "fit", inventory["objective"], inventory["registryDigest"], 0),
        (complete_root / "fit-next-stage", stage_objective, inventory["registryDigest"], 3),
        (
            resume_root / "fit",
            resume_inventory["objective"],
            resume_inventory["registryDigest"],
            0,
        ),
    ):
        # Exercise the stricter per-update input validator with actual recorded
        # count deltas. This shadow contains no text exposure and grants none.
        shadow = empty_position_exposure(
            registry_digest=registry_digest,
            objective=objective,
            starting_global_step=starting_step,
        )
        for row in map(json.loads, (fit / "metrics.jsonl").read_text().splitlines()):
            if row.get("event") != "train":
                continue
            recorded = row["semantic_v8_position_exposure"]
            positions = recorded["routerInternalPositionCounts"]
            assignments = recorded["routerInternalAssignmentCounts"]
            shadow = advance_position_exposure(
                shadow,
                [],
                registry_digest=registry_digest,
                objective=objective,
                optimizer_step=row["global_step"],
                router_internal_positions={
                    name: count - shadow["routerInternalPositionCounts"].get(name, 0)
                    for name, count in positions.items()
                },
                router_internal_assignments={
                    name: count - shadow["routerInternalAssignmentCounts"].get(name, 0)
                    for name, count in assignments.items()
                },
            )
            assert shadow["routerInternalPositionCounts"] == positions
            assert shadow["routerInternalAssignmentCounts"] == assignments
            assert (
                shadow["routerInternalPositionAuditUpdates"]
                == recorded["routerInternalPositionAuditUpdates"]
            )
            actual_router_updates += 1
    assert actual_router_updates == 6
    report = {
        "schema": "semantic-v8-stricter-router-retained-actual-audit/1",
        "status": "passed",
        "validatorRevision": "19d011247e9f1883db632121ca6d8c591abc875d",
        "actualOptimizerRevision": actual_source,
        "auditScriptSha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "runtime": {"python": platform.python_version(), "torch": str(torch.__version__)},
        "optimizerUpdatesPerformedByAudit": 0,
        "actualOptimizerUpdatesRetained": {
            "completeFixture": 3,
            "subsequentStage": 1,
            "separateResume": 2,
        },
        "independentPreparedInventoriesRebuilt": 2,
        "actualPositionStatesValidated": 4,
        "metadataBindingsChecked": 4,
        "loggedTargetHistorySealsReconstructedExactly": 2,
        "actualRouterUpdateDeltasValidated": actual_router_updates,
        "completeComparisonUnchanged": True,
        "boundedResumeComparisonUnchanged": True,
        "archivedPreviousStagePreserved": True,
        "freshStageRouterCountsEmpty": True,
        "routerSummaries": {
            name: {
                key: value
                for key, value in position_exposure_report(state).items()
                if key.startswith("routerInternal")
            }
            for name, state in (
                ("complete", before_stage),
                ("subsequentStage", stage),
                ("firstResumeCheckpoint", resume["firstPositionExposure"]),
                ("resumedCheckpoint", resume["resumedPositionExposure"]),
            )
        },
        "scope": "Read-only stronger validation of unchanged actual tiny fixture receipts; source/tokenizer admission is injected and prepared support-population or learned quality is not claimed.",
        "routerInternalPositionCoverage": "unmeasured",
        "learnedSuccess": None,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
