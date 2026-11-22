"""Exact positive-loss inventory and optimizer target-presence comparison controls."""

from copy import deepcopy

import pytest

from megatoken.data.semantic_json import stable_algorithm_hash
from megatoken.training.target_inventory_v8 import (
    compare_target_inventory,
    validate_target_inventory,
)

REGISTRY = "a" * 64
MANIFEST = "b" * 64
CODEC = "c" * 64
TRAINING = "d" * 64
MODEL = "e" * 64


def _inventory():
    sources = {
        "source-a": {
            "split": "train",
            "logicalTargets": {
                "target-one": {
                    "records": {
                        "record-one": {
                            "windowIndices": [0, 1],
                            "lossDenominators": {"language_model": 8},
                            "scoredOccurrences": 2,
                        }
                    }
                }
            },
        }
    }
    body = {
        "schema": "semantic-v8-positive-loss-target-inventory/1",
        "registryDigest": REGISTRY,
        "sourceIds": [f"source-a" if index == 0 else f"train-{index:02d}" for index in range(48)],
        "scoredSourceIds": ["source-a"],
        "trainEpisodeCount": 48,
        "windowCount": 120,
        "optimizerWindowCount": 100,
        "phaseMeter": "P",
        "phaseBudget": 200000000,
        "onePassMeterCount": 5000,
        "preparedManifestFingerprint": MANIFEST,
        "codecFingerprint": CODEC,
        "trainingConfigDigest": TRAINING,
        "modelConfigDigest": MODEL,
        "windowConfig": {},
        "effectiveLossWeights": {"language_model": 1.0},
        "available": {"policyTargetRecordCount": 1, "lossDenominators": {"language_model": 8}},
        "positiveWeight": {
            "sourceCount": 1,
            "logicalTargetCount": 1,
            "physicalTargetRecordCount": 1,
            "scoredOccurrences": 2,
            "lossDenominators": {"language_model": 8},
        },
        "sources": sources,
        "optimizerUpdates": 0,
        "learnedSuccess": None,
    }
    return {**body, "digest": stable_algorithm_hash(body)}


def _identities():
    return {
        "registry_digest": REGISTRY,
        "prepared_manifest_fingerprint": MANIFEST,
        "codec_fingerprint": CODEC,
        "training_config_digest": TRAINING,
        "model_config_digest": MODEL,
    }


def _actual(*, extra=False):
    rows = {
        "source-a": {
            "split": "train",
            "logicalTargets": {
                "target-one": {
                    "records": {
                        "record-one": {
                            "windowIndices": [0],
                            "scoredOccurrences": 1,
                        }
                    }
                }
            },
        }
    }
    if extra:
        rows["source-b"] = {
            "split": "train",
            "logicalTargets": {
                "unlisted": {"records": {"forged": {"windowIndices": [0], "scoredOccurrences": 1}}}
            },
        }
    return rows


def test_exact_target_presence_passes_without_claiming_position_coverage():
    result = compare_target_inventory(_inventory(), _actual(), rederived_inventory=_inventory(), **_identities())
    assert result["allPositiveWeightTargetsExposed"] is True
    assert result["targetPresenceCoverage"] == 1.0
    assert result["tokenOrMaskPositionCoverage"] == "unmeasured"
    assert result["repeatedWindowExposureObserved"] is False
    assert result["learnedSuccess"] is None


def test_missing_inventory_targets_are_reported_without_inflating_coverage():
    result = compare_target_inventory(_inventory(), {}, rederived_inventory=_inventory(), **_identities())
    assert result["allPositiveWeightTargetsExposed"] is False
    assert result["targetPresenceCoverage"] == 0.0
    assert result["missingPositiveWeightPhysicalTargets"] == [
        ["source-a", "target-one", "record-one"]
    ]


def test_unexpected_optimizer_target_is_integrity_failure():
    with pytest.raises(ValueError, match="outside the rederived inventory"):
        compare_target_inventory(
            _inventory(), _actual(extra=True), rederived_inventory=_inventory(), **_identities()
        )


@pytest.mark.parametrize(
    "field",
    [
        "registryDigest",
        "preparedManifestFingerprint",
        "codecFingerprint",
        "trainingConfigDigest",
        "modelConfigDigest",
    ],
)
def test_inventory_bound_to_other_run_identity_is_refused(field):
    keys = {
        "registryDigest": "registry_digest",
        "preparedManifestFingerprint": "prepared_manifest_fingerprint",
        "codecFingerprint": "codec_fingerprint",
        "trainingConfigDigest": "training_config_digest",
        "modelConfigDigest": "model_config_digest",
    }
    identities = _identities()
    identities[keys[field]] = "f" * 64
    with pytest.raises(ValueError, match="identity"):
        validate_target_inventory(_inventory(), rederived_inventory=_inventory(), **identities)


def test_rehashed_forged_target_keys_still_fail_count_invariant():
    value = deepcopy(_inventory())
    value["sources"]["source-a"]["logicalTargets"]["target-two"] = {
        "records": {"record-two": {"windowIndices": [0], "scoredOccurrences": 1}}
    }
    body = {key: child for key, child in value.items() if key != "digest"}
    value["digest"] = stable_algorithm_hash(body)
    with pytest.raises(ValueError, match="re-derivation"):
        validate_target_inventory(value, rederived_inventory=_inventory(), **_identities())


def test_repeated_window_occurrence_is_reported_separately_from_unique_target_coverage():
    actual = _actual()
    record = actual["source-a"]["logicalTargets"]["target-one"]["records"]["record-one"]
    record["windowIndices"] = [0]
    record["scoredOccurrences"] = 2
    result = compare_target_inventory(_inventory(), actual, rederived_inventory=_inventory(), **_identities())
    assert result["allPositiveWeightTargetsExposed"] is True
    assert result["repeatedWindowExposureObserved"] is True


def test_rehashed_inventory_that_disagrees_with_rederived_planner_is_refused():
    value = _inventory()
    tampered = deepcopy(value)
    tampered["sources"]["source-a"]["logicalTargets"]["target-one"]["records"]["record-one"]["lossDenominators"]["language_model"] = 99
    body = {key: child for key, child in tampered.items() if key != "digest"}
    tampered["digest"] = stable_algorithm_hash(body)
    with pytest.raises(ValueError, match="re-derivation"):
        validate_target_inventory(tampered, rederived_inventory=value, **_identities())


def test_inventory_omitting_train_source_or_duplicate_source_id_is_refused():
    value = _inventory()
    missing = deepcopy(value)
    missing["sourceIds"] = missing["sourceIds"][:-1]
    body = {key: child for key, child in missing.items() if key != "digest"}
    missing["digest"] = stable_algorithm_hash(body)
    with pytest.raises(ValueError, match="identity, count"):
        validate_target_inventory(missing, rederived_inventory=missing, **_identities())

    duplicate = deepcopy(value)
    duplicate["sourceIds"][-1] = duplicate["sourceIds"][0]
    body = {key: child for key, child in duplicate.items() if key != "digest"}
    duplicate["digest"] = stable_algorithm_hash(body)
    with pytest.raises(ValueError, match="identity, count"):
        validate_target_inventory(duplicate, rederived_inventory=duplicate, **_identities())


def test_duplicate_window_ids_in_inventory_are_refused():
    value = deepcopy(_inventory())
    record = value["sources"]["source-a"]["logicalTargets"]["target-one"]["records"]["record-one"]
    record["windowIndices"] = [0, 0]
    body = {key: child for key, child in value.items() if key != "digest"}
    value["digest"] = stable_algorithm_hash(body)
    with pytest.raises(ValueError, match="duplicate or invalid window indices"):
        validate_target_inventory(value, rederived_inventory=value, **_identities())
