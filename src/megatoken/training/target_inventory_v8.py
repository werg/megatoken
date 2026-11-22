"""Bounded comparison for positive-loss target inventory and optimizer census."""

from __future__ import annotations

import json
import re
from dataclasses import asdict

from megatoken.data.semantic_json import stable_algorithm_hash


def training_config_identity_v8(config, *, codec_fingerprint, prepared_manifest_fingerprints):
    """Bind objective settings to verified input contents, not storage locations.

    Callers obtain these fingerprints from the loaded codec and checked prepared
    manifests. The real configuration still retains its physical paths as audit
    provenance. Changing any input contents or effective objective changes this
    identity; copying the same immutable inputs to another directory does not.
    """
    fingerprints = list(prepared_manifest_fingerprints)
    if (
        config.semantic_v8_policy is None
        or not isinstance(codec_fingerprint, str)
        or re.fullmatch(r"[0-9a-f]{64}", codec_fingerprint) is None
        or not fingerprints
        or any(
            not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
            for value in fingerprints
        )
        or len(fingerprints) != len(set(fingerprints))
    ):
        raise ValueError("V8 objective requires exact codec and distinct prepared fingerprints")
    identity = asdict(config)
    for key in (
        "output",
        "max_steps",
        "checkpoint_every",
        "keep_checkpoints",
        "log_every",
        "device",
    ):
        identity.pop(key, None)
    identity["codec"] = {"fingerprint": codec_fingerprint}
    identity["datasets"] = {"manifestFingerprints": sorted(fingerprints)}
    return identity


def inventory_episode_source_id(episode, *, registry_digest):
    """Read the source binding already verified by the complete V8 preflight.

    ``lineage.source_id`` names the prepared-data adapter, not a source program.
    The optimizer census uses the annotation's source ID and dataset binding;
    inventory must use that same identity even for windows with no scored work.
    """
    try:
        annotations = json.loads(episode.supervision.annotations_json)
        semantic = annotations["semantic_execution"]
        source_id = semantic["sourceId"]
        binding = semantic["datasetBinding"]
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError("Prepared inventory episode has no bound V8 source identity") from error
    if (
        episode.stream != "semantic_execution"
        or semantic.get("schema") != "semantic-execution/8"
        or not isinstance(source_id, str)
        or not source_id
        or not isinstance(binding, dict)
        or binding.get("sourceId") != source_id
        or binding.get("registryDigest") != registry_digest
        or binding.get("split") != "train"
        or episode.lineage.frozen_split != "train"
    ):
        raise ValueError("Prepared inventory episode has a changed V8 source or split binding")
    return source_id


def validate_target_inventory(
    inventory,
    *,
    rederived_inventory,
    registry_digest,
    prepared_manifest_fingerprint,
    codec_fingerprint,
    training_config_digest,
    model_config_digest,
):
    required = {
        "schema",
        "registryDigest",
        "sourceIds",
        "scoredSourceIds",
        "trainEpisodeCount",
        "windowCount",
        "optimizerWindowCount",
        "phaseMeter",
        "phaseBudget",
        "onePassMeterCount",
        "preparedManifestFingerprint",
        "codecFingerprint",
        "trainingConfigDigest",
        "modelConfigDigest",
        "windowConfig",
        "effectiveLossWeights",
        "available",
        "positiveWeight",
        "sources",
        "optimizerUpdates",
        "learnedSuccess",
        "digest",
    }
    if not isinstance(inventory, dict) or set(inventory) != required:
        raise ValueError("V8 target inventory has an invalid field set")
    sources = inventory.get("sources")
    if not isinstance(sources, dict):
        raise ValueError("V8 target inventory sources must be an object")
    for source in sources.values():
        if not isinstance(source, dict) or not isinstance(source.get("logicalTargets"), dict):
            raise ValueError("V8 target inventory has malformed source target rows")
        for target in source["logicalTargets"].values():
            if not isinstance(target, dict) or not isinstance(target.get("records"), dict):
                raise ValueError("V8 target inventory has malformed logical target rows")
            for record in target["records"].values():
                if not isinstance(record, dict):
                    raise ValueError("V8 target inventory has malformed physical target rows")
                indices = record.get("windowIndices")
                if (
                    not isinstance(indices, list)
                    or any(type(index) is not int or index < 0 for index in indices)
                    or indices != sorted(set(indices))
                ):
                    raise ValueError("V8 target inventory has duplicate or invalid window indices")
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
        or type(inventory["optimizerWindowCount"]) is not int
        or not 1 <= inventory["optimizerWindowCount"] <= inventory["windowCount"]
        or inventory["phaseMeter"] != "P"
        or type(inventory["phaseBudget"]) is not int
        or inventory["phaseBudget"] < 1
        or type(inventory["onePassMeterCount"]) is not int
        or not 0 <= inventory["onePassMeterCount"] <= inventory["phaseBudget"]
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
        != len(
            {
                (source_id, logical_id)
                for source_id, source in inventory["sources"].items()
                for logical_id in source["logicalTargets"]
            }
        )
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
    expected_records = {
        (source_id, logical_id, record_id): record
        for source_id, source in inventory["sources"].items()
        for logical_id, target in source["logicalTargets"].items()
        for record_id, record in target["records"].items()
    }
    actual_records = {
        (source_id, logical_id, record_id): record
        for source_id, source in actual_sources.items()
        for logical_id, target in source["logicalTargets"].items()
        for record_id, record in target["records"].items()
    }
    occurrence_mismatches = []
    window_mismatches = []
    denominator_mismatches = []
    for key in sorted(expected_keys & actual_keys):
        expected_record = expected_records[key]
        actual_record = actual_records[key]
        if expected_record["scoredOccurrences"] != actual_record.get("scoredOccurrences"):
            occurrence_mismatches.append(
                [
                    list(key),
                    expected_record["scoredOccurrences"],
                    actual_record.get("scoredOccurrences"),
                ]
            )
        if expected_record["windowIndices"] != actual_record.get("windowIndices"):
            window_mismatches.append(
                [list(key), expected_record["windowIndices"], actual_record.get("windowIndices")]
            )
        if expected_record["lossDenominators"] != actual_record.get("lossDenominators"):
            denominator_mismatches.append(
                [
                    list(key),
                    expected_record["lossDenominators"],
                    actual_record.get("lossDenominators"),
                ]
            )
    repeated = any(
        record.get("scoredOccurrences", 0) > len(record.get("windowIndices", []))
        for source in actual_sources.values()
        for target in source["logicalTargets"].values()
        for record in target["records"].values()
    )
    total = len(expected_keys)
    covered = total - len(missing)
    exact_census = (
        bool(total)
        and not missing
        and not unexpected
        and not (occurrence_mismatches or window_mismatches or denominator_mismatches)
    )
    return {
        "schema": "semantic-v8-target-presence-comparison/1",
        "inventoryDigest": inventory["digest"],
        "expectedPositiveWeightPhysicalTargets": total,
        "coveredPositiveWeightPhysicalTargets": covered,
        "expectedPositiveWeightLogicalTargets": len(expected_logical),
        "coveredPositiveWeightLogicalTargets": len(expected_logical & actual_logical),
        "missingPositiveWeightPhysicalTargets": [list(row) for row in missing],
        "unexpectedPhysicalTargets": [list(row) for row in unexpected],
        "targetPresenceCoverage": 1.0
        if total and not missing and not unexpected
        else (covered / total if total else 0.0),
        "allPositiveWeightTargetsExposed": bool(total) and not missing and not unexpected,
        "exactOnePassCensus": exact_census,
        "scoredOccurrenceMismatches": occurrence_mismatches,
        "windowIndexMismatches": window_mismatches,
        "positiveWeightDenominatorMismatches": denominator_mismatches,
        "repeatedWindowExposureObserved": repeated,
        "tokenOrMaskPositionCoverage": "unmeasured",
        "learnedSuccess": None,
    }
