"""Focused integrity controls for V8 optimizer exposure lifecycle evidence."""

import pytest

from megatoken.data.semantic_json import stable_algorithm_hash

from scripts.qualify_semantic_v8_lifecycle import qualify_target_exposure
from megatoken.training.semantic_v8 import (
    V8_TARGET_EXPOSURE_SCHEMA,
    advance_semantic_v8_target_exposure,
    empty_semantic_v8_target_exposure,
    merge_semantic_v8_target_exposures,
)

REGISTRY = "a" * 64
SOURCE = "support-train-01"


def _state_at_steps(loss_counts, *, registry=REGISTRY):
    state = empty_semantic_v8_target_exposure(registry)
    for step, denominator in enumerate(loss_counts, start=1):
        occurrence = {
            "sourceId": SOURCE,
            "registryDigest": registry,
            "split": "train",
            "logicalTargetId": "semantic:target:one",
            "physicalRecordId": "physical-one",
            "windowIndex": step - 1,
            "lossDenominators": {"language_model": denominator},
        }
        merged = merge_semantic_v8_target_exposures(
            [
                {
                    "schema": V8_TARGET_EXPOSURE_SCHEMA,
                    "registryDigest": registry,
                    "occurrences": [occurrence],
                }
            ],
            registry_digest=registry,
        )
        state = advance_semantic_v8_target_exposure(
            state, merged, optimizer_step=step, registry_digest=registry
        )
    return state


def _qualify(first_state, resumed_state, *, registry=REGISTRY):
    return qualify_target_exposure(
        {"global_step": 1, "semantic_v8_target_exposure": first_state},
        {"global_step": 2, "semantic_v8_target_exposure": resumed_state},
        registry_digest=registry,
        allowed_source_ids={SOURCE},
    )


def test_census_report_is_bound_to_exact_registry_and_preserves_unique_vs_scored():
    first = _state_at_steps([3])
    resumed = _state_at_steps([3, 5])
    evidence = _qualify(first, resumed)
    assert evidence["registryDigest"] == REGISTRY
    assert evidence["historyContinued"] is True
    assert evidence["firstCheckpointCounts"] == {
        "sourceCount": 1,
        "logicalTargetCount": 1,
        "physicalTargetRecordCount": 1,
        "scoredOccurrences": 1,
        "lossDenominators": {"language_model": 3},
    }
    assert evidence["resumedCounts"]["sourceCount"] == 1
    assert evidence["resumedCounts"]["logicalTargetCount"] == 1
    assert evidence["resumedCounts"]["physicalTargetRecordCount"] == 1
    assert evidence["resumedCounts"]["scoredOccurrences"] == 2
    assert evidence["positiveWeightLossDenominators"] == {"language_model": 8}
    assert evidence["allTargetsExposed"] is False


def test_checkpoint_census_with_changed_registry_is_refused():
    first = _state_at_steps([3])
    resumed = _state_at_steps([3, 5], registry="b" * 64)
    with pytest.raises(ValueError, match="changed registry"):
        _qualify(first, resumed)


def test_checkpoint_census_with_changed_step_history_is_refused():
    first = _state_at_steps([3])
    resumed = _state_at_steps([3, 5])
    with pytest.raises(ValueError, match="exactly optimizer step one"):
        qualify_target_exposure(
            {"global_step": 2, "semantic_v8_target_exposure": first},
            {"global_step": 3, "semantic_v8_target_exposure": resumed},
            registry_digest=REGISTRY,
            allowed_source_ids={SOURCE},
        )


def test_checkpoint_with_dropped_census_is_refused():
    first = _state_at_steps([3])
    with pytest.raises(ValueError, match="lost its target-exposure census"):
        qualify_target_exposure(
            {"global_step": 1, "semantic_v8_target_exposure": first},
            {"global_step": 2},
            registry_digest=REGISTRY,
            allowed_source_ids={SOURCE},
        )


def test_resume_that_replaces_step_one_source_history_is_refused():
    first = _state_at_steps([3])
    replacement = empty_semantic_v8_target_exposure(REGISTRY)
    other = {
        "sourceId": "support-train-02",
        "registryDigest": REGISTRY,
        "split": "train",
        "logicalTargetId": "semantic:target:two",
        "physicalRecordId": "physical-two",
        "windowIndex": 0,
        "lossDenominators": {"language_model": 4},
    }
    merged = merge_semantic_v8_target_exposures(
        [{"schema": V8_TARGET_EXPOSURE_SCHEMA, "registryDigest": REGISTRY, "occurrences": [other]}],
        registry_digest=REGISTRY,
    )
    replacement = advance_semantic_v8_target_exposure(
        replacement, merged, optimizer_step=1, registry_digest=REGISTRY
    )
    replacement = advance_semantic_v8_target_exposure(
        replacement, merged, optimizer_step=2, registry_digest=REGISTRY
    )
    # Even a forged parent link cannot excuse dropping the original scored source.
    replacement["previousDigest"] = first["digest"]
    replacement["digest"] = stable_algorithm_hash(
        {key: value for key, value in replacement.items() if key != "digest"}
    )
    with pytest.raises(ValueError, match="dropped a scored source"):
        qualify_target_exposure(
            {"global_step": 1, "semantic_v8_target_exposure": first},
            {"global_step": 2, "semantic_v8_target_exposure": replacement},
            registry_digest=REGISTRY,
            allowed_source_ids={SOURCE, "support-train-02"},
        )



def test_resumed_census_must_name_exact_step_one_parent_digest():
    first = _state_at_steps([3])
    mismatched = _state_at_steps([4, 4])
    with pytest.raises(ValueError, match="does not name the exact step-one parent"):
        _qualify(first, mismatched)
