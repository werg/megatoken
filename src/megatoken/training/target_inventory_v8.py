"""Bounded comparison for positive-loss target inventory and optimizer census."""

from __future__ import annotations

from megatoken.data.semantic_json import stable_algorithm_hash


def validate_target_inventory(inventory, *, rederived_inventory, registry_digest, prepared_manifest_fingerprint,
                              codec_fingerprint, training_config_digest, model_config_digest):
    required = {
        "schema", "registryDigest", "sourceIds", "scoredSourceIds", "trainEpisodeCount",
        "windowCount", "preparedManifestFingerprint", "codecFingerprint",
        "trainingConfigDigest", "modelConfigDigest", "windowConfig", "effectiveLossWeights",
        "available", "positiveWeight", "sources", "optimizerUpdates", "learnedSuccess", "digest",
    }
    if not isinstance(inventory, dict) or set(inventory) != required:
        raise ValueError("V8 target inventory has an invalid field set")
    if not isinstance(rederived_inventory, dict) or inventory != rederived_inventory:
        raise ValueError("V8 target inventory differs from exact native planner re-derivation")
    body = {key: value for key, value in inventory.items() if key != "digest"}
    if (
        inventory["schema"] != "semantic-v8-positive-loss-target-inventory/1"
        or inventory["registryDigest"] != registry_digest
        or inventory["preparedManifestFingerprint"] != prepared_manifest_fingerprint
        or inventory["codecFingerprint"] != codec_fingerprint
        or inventory["trainingConfigDigest"] != training_config_digest
        or inventory["modelConfigDigest"] != model_config_digest
        or inventory["digest"] != stable_algorithm_hash(body)
        or inventory["optimizerUpdates"] != 0
        or inventory["learnedSuccess"] is not None
        or type(inventory["windowCount"]) is not int
        or inventory["windowCount"] < 1
        or type(inventory["trainEpisodeCount"]) is not int
        or inventory["trainEpisodeCount"] != len(inventory["sourceIds"])
        or len(inventory["sourceIds"]) != 48
        or len(set(inventory["sourceIds"])) != 48
        or inventory["sourceIds"] != sorted(inventory["sourceIds"])
        or inventory["scoredSourceIds"] != sorted(inventory["sources"])
        or not set(inventory["scoredSourceIds"]) <= set(inventory["sourceIds"])
    ):
        raise ValueError("V8 target inventory identity, count or seal is invalid")
    if (
        not isinstance(inventory["sourceIds"], list)
        or any(not isinstance(item, str) or not item for item in inventory["sourceIds"])
        or not isinstance(inventory["scoredSourceIds"], list)
        or len(set(inventory["scoredSourceIds"])) != len(inventory["scoredSourceIds"])
    ):
        raise ValueError("V8 target inventory source IDs are malformed or duplicated")
    expected = inventory["positiveWeight"]
    expected_keys = {
        (source_id, logical_id, record_id)
        for source_id, source in inventory["sources"].items()
        for logical_id, target in source["logicalTargets"].items()
        for record_id in target["records"]
    }
    if (
        any(source.get("split") != "train" for source in inventory["sources"].values())
        or any(
            not isinstance(record.get("windowIndices"), list)
            or record["windowIndices"] != sorted(set(record["windowIndices"]))
            or any(type(index) is not int or index < 0 for index in record["windowIndices"])
            for source in inventory["sources"].values()
            for target in source["logicalTargets"].values()
            for record in target["records"].values()
        )
        or not isinstance(expected, dict)
        or expected.get("sourceCount") != len(inventory["sources"])
        or expected.get("logicalTargetCount")
        != len({
            (source_id, logical_id)
            for source_id, source in inventory["sources"].items()
            for logical_id in source["logicalTargets"]
        })
        or expected.get("physicalTargetRecordCount") != len(expected_keys)
    ):
        raise ValueError("V8 positive-loss target inventory summary differs from its keys")
    return expected_keys


def compare_target_inventory(inventory, actual_sources, *, rederived_inventory, **identities):
    """Compare target presence only; token/position-level coverage is not retained."""
    expected_keys = validate_target_inventory(
        inventory, rederived_inventory=rederived_inventory, **identities
    )
    if not isinstance(actual_sources, dict):
        raise ValueError("Optimizer target exposure has no source map")
    actual_keys = {
        (source_id, logical_id, record_id)
        for source_id, source in actual_sources.items()
        for logical_id, target in source["logicalTargets"].items()
        for record_id in target["records"]
    }
    missing = sorted(expected_keys - actual_keys)
    unexpected = sorted(actual_keys - expected_keys)
    if unexpected:
        raise ValueError(
            f"Optimizer census contains {len(unexpected)} physical target keys outside the rederived inventory"
        )
    expected_logical = {(source_id, logical_id) for source_id, logical_id, _ in expected_keys}
    actual_logical = {(source_id, logical_id) for source_id, logical_id, _ in actual_keys}
    repeated = any(
        record.get("scoredOccurrences", 0) > len(record.get("windowIndices", []))
        for source in actual_sources.values()
        for target in source["logicalTargets"].values()
        for record in target["records"].values()
    )
    total = len(expected_keys)
    covered = total - len(missing)
    return {
        "schema": "semantic-v8-target-presence-comparison/1",
        "inventoryDigest": inventory["digest"],
        "expectedPositiveWeightPhysicalTargets": total,
        "coveredPositiveWeightPhysicalTargets": covered,
        "expectedPositiveWeightLogicalTargets": len(expected_logical),
        "coveredPositiveWeightLogicalTargets": len(expected_logical & actual_logical),
        "missingPositiveWeightPhysicalTargets": [list(row) for row in missing],
        "unexpectedPhysicalTargets": [list(row) for row in unexpected],
        "targetPresenceCoverage": 1.0 if total and not missing and not unexpected else (
            covered / total if total else 0.0
        ),
        "allPositiveWeightTargetsExposed": bool(total) and not missing and not unexpected,
        "repeatedWindowExposureObserved": repeated,
        "tokenOrMaskPositionCoverage": "unmeasured",
        "learnedSuccess": None,
    }
