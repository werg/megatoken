"""V8 optimizer-window exposure counts only loss-bearing native target occurrences."""

import json
from types import SimpleNamespace

import pytest

from megatoken.training.semantic_v8 import (
    advance_semantic_v8_target_exposure,
    empty_semantic_v8_target_exposure,
    merge_semantic_v8_target_exposures,
    restore_semantic_v8_target_exposure,
    semantic_v8_target_exposure_from_batches,
    semantic_v8_target_exposure_report,
    validate_semantic_v8_target_exposure,
)

REGISTRY = "a" * 64
SOURCE = "support-train-01"


def _extract_v8(batches, *, registry_digest, effective_loss_weights=None):
    if effective_loss_weights is None:
        effective_loss_weights = {"language_model": 1.0, "structure": 1.0}
    return semantic_v8_target_exposure_from_batches(
        batches,
        registry_digest=registry_digest,
        effective_loss_weights=effective_loss_weights,
    )


def _batch(
    *,
    split="train",
    registry=REGISTRY,
    logical_id="semantic:target:one",
    record_id="physical-one",
    window_index=0,
    denominators=None,
    policy_mask=True,
    include_target=True,
):
    physical = SimpleNamespace(
        logical_id=logical_id,
        record_id=record_id,
        policy_mask=policy_mask,
    )
    encoded = SimpleNamespace(record=physical)
    plans = (
        [SimpleNamespace(denominators=denominators or {"language_model": 4})]
        if include_target
        else []
    )
    step = SimpleNamespace(records=[encoded], plans=plans, offsets=[0])
    binding = {"sourceId": SOURCE, "registryDigest": registry, "split": split}
    episode = SimpleNamespace(
        stream="semantic_execution",
        lineage=SimpleNamespace(frozen_split=split),
        supervision=SimpleNamespace(
            annotations_json=json.dumps(
                {
                    "semantic_execution": {
                        "schema": "semantic-execution/8",
                        "sourceId": SOURCE,
                        "datasetBinding": binding,
                    }
                }
            )
        ),
    )
    item = SimpleNamespace(episode=episode, window=SimpleNamespace(index=window_index))
    return SimpleNamespace(windows=[item], steps=[step])


def test_overlapping_rank_windows_keep_unique_coverage_and_sum_scored_occurrences():
    first = _extract_v8(
        [_batch(window_index=2)], registry_digest=REGISTRY
    )
    overlap = _extract_v8(
        [_batch(window_index=5)], registry_digest=REGISTRY
    )
    update = merge_semantic_v8_target_exposures(
        [first, overlap], registry_digest=REGISTRY
    )
    state = empty_semantic_v8_target_exposure(REGISTRY)
    state = advance_semantic_v8_target_exposure(
        state, update, optimizer_step=1, registry_digest=REGISTRY
    )
    report = semantic_v8_target_exposure_report(state)
    record = state["sources"][SOURCE]["logicalTargets"]["semantic:target:one"][
        "records"
    ]["physical-one"]
    assert report["sourceCount"] == 1
    assert report["logicalTargetCount"] == 1
    assert report["physicalTargetRecordCount"] == 1
    assert report["scoredOccurrences"] == 2
    assert report["lossDenominators"] == {"language_model": 8}
    assert record["windowIndices"] == [2, 5]
    assert record["optimizerUpdates"] == 1


def test_partial_masks_sum_across_ranks_without_claiming_extra_unique_targets():
    first = _extract_v8(
        [_batch(denominators={"language_model": 4})], registry_digest=REGISTRY
    )
    second = _extract_v8(
        [_batch(denominators={"structure": 2})], registry_digest=REGISTRY
    )
    update = merge_semantic_v8_target_exposures(
        [first, second], registry_digest=REGISTRY
    )
    state = advance_semantic_v8_target_exposure(
        empty_semantic_v8_target_exposure(REGISTRY),
        update,
        optimizer_step=1,
        registry_digest=REGISTRY,
    )
    report = semantic_v8_target_exposure_report(state)
    assert report["logicalTargetCount"] == report["physicalTargetRecordCount"] == 1
    assert report["scoredOccurrences"] == 2
    assert report["lossDenominators"] == {"language_model": 4, "structure": 2}


def test_resume_accumulates_the_same_target_and_retains_exact_step_state():
    state = empty_semantic_v8_target_exposure(REGISTRY)
    step = merge_semantic_v8_target_exposures(
        [_extract_v8([_batch()], registry_digest=REGISTRY)],
        registry_digest=REGISTRY,
    )
    first = advance_semantic_v8_target_exposure(
        state, step, optimizer_step=1, registry_digest=REGISTRY
    )
    restored = validate_semantic_v8_target_exposure(first, registry_digest=REGISTRY)
    second = advance_semantic_v8_target_exposure(
        restored, step, optimizer_step=2, registry_digest=REGISTRY
    )
    target = second["sources"][SOURCE]["logicalTargets"]["semantic:target:one"][
        "records"
    ]["physical-one"]
    assert target["optimizerUpdates"] == 2
    assert target["scoredOccurrences"] == 2
    assert target["lossDenominators"] == {"language_model": 8}
    assert second["lastGlobalStep"] == 2


@pytest.mark.parametrize(
    ("split", "registry"),
    [("validation", REGISTRY), ("train", "b" * 64)],
)
def test_wrong_frozen_source_split_or_registry_refuses_exposure(split, registry):
    with pytest.raises(ValueError, match="source, registry or split"):
        _extract_v8(
            [_batch(split=split, registry=registry)], registry_digest=REGISTRY
        )


def test_zero_loss_or_observed_window_produces_no_target_exposure():
    observed = _extract_v8(
        [_batch(include_target=False)], registry_digest=REGISTRY
    )
    zero_loss = _extract_v8(
        [_batch(denominators={"language_model": 0})], registry_digest=REGISTRY
    )
    assert observed["occurrences"] == []
    assert zero_loss["occurrences"] == []


def test_unmasked_plan_and_out_of_order_resume_cannot_advance_exposure():
    with pytest.raises(ValueError, match="outside the target mask"):
        _extract_v8(
            [_batch(policy_mask=False)], registry_digest=REGISTRY
        )
    state = empty_semantic_v8_target_exposure(REGISTRY)
    update = merge_semantic_v8_target_exposures(
        [_extract_v8([_batch()], registry_digest=REGISTRY)],
        registry_digest=REGISTRY,
    )
    with pytest.raises(ValueError, match="next optimizer update"):
        advance_semantic_v8_target_exposure(
            state, update, optimizer_step=2, registry_digest=REGISTRY
        )


def test_v8_resume_fails_closed_if_census_is_missing_or_checkpoint_step_differs():
    payload = {
        "train_config": {"semantic_v8_policy": {"registryDigest": REGISTRY}},
        "semantic_v8_target_exposure": None,
    }
    with pytest.raises(ValueError, match="missing its target-exposure history"):
        restore_semantic_v8_target_exposure(
            payload, registry_digest=REGISTRY, global_step=4
        )

    state = empty_semantic_v8_target_exposure(REGISTRY, starting_global_step=2)
    payload["semantic_v8_target_exposure"] = state
    with pytest.raises(ValueError, match="differs from checkpoint step"):
        restore_semantic_v8_target_exposure(
            payload, registry_digest=REGISTRY, global_step=4
        )


def test_non_v8_checkpoint_can_start_only_at_explicit_fresh_stage_boundary():
    payload = {
        "train_config": {"semantic_v8_policy": None},
        "semantic_v8_target_exposure": None,
    }
    with pytest.raises(ValueError, match="fresh stage"):
        restore_semantic_v8_target_exposure(
            payload, registry_digest=REGISTRY, global_step=7
        )
    state = restore_semantic_v8_target_exposure(
        payload,
        registry_digest=REGISTRY,
        global_step=7,
        phase_transition=True,
    )
    assert state["startingGlobalStep"] == state["lastGlobalStep"] == 7
    assert state["optimizerUpdates"] == 0


def test_positive_denominator_with_zero_objective_weight_is_not_counted():
    exposure = _extract_v8(
        [_batch(denominators={"language_model": 4, "router_balance": 8})],
        registry_digest=REGISTRY,
        effective_loss_weights={"language_model": 0.0, "router_balance": 1.0},
    )
    assert exposure["occurrences"] == []
