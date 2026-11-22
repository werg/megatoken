"""Fail-closed V8 semantic data policy for the ordinary trainer."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

from megatoken.data.semantic_contract_v8 import (
    CONTRACT_DIGEST,
    FIXTURE_DIGEST,
    INTEROPERABILITY_DIGEST,
)
from megatoken.data.semantic_json import stable_algorithm_hash
from megatoken.generation.codec import LING_TOKENIZER_SHA256

V8_TARGET_EXPOSURE_SCHEMA = "semantic-v8-optimizer-target-exposure/1"


def empty_semantic_v8_target_exposure(registry_digest, *, starting_global_step=0):
    if (
        not isinstance(registry_digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", registry_digest) is None
    ):
        raise ValueError("V8 target exposure requires the pinned family registry digest")
    if type(starting_global_step) is not int or starting_global_step < 0:
        raise ValueError("V8 target exposure starting step must be a nonnegative integer")
    return _seal_semantic_v8_target_exposure(
        {
            "schema": V8_TARGET_EXPOSURE_SCHEMA,
            "registryDigest": registry_digest,
            "startingGlobalStep": starting_global_step,
            "lastGlobalStep": starting_global_step,
            "optimizerUpdates": 0,
            "sources": {},
        }
    )


def _seal_semantic_v8_target_exposure(state):
    value = {key: child for key, child in state.items() if key != "digest"}
    return {**value, "digest": stable_algorithm_hash(value)}


def restore_semantic_v8_target_exposure(
    payload, *, registry_digest, global_step, phase_transition=False
):
    """Restore exact census history or begin only at an explicit non-V8 stage boundary."""
    if type(global_step) is not int or global_step < 0:
        raise ValueError("V8 target-exposure restore step must be a nonnegative integer")
    if payload is None:
        return empty_semantic_v8_target_exposure(registry_digest, starting_global_step=global_step)
    previous = payload.get("train_config", {}).get("semantic_v8_policy")
    saved = payload.get("semantic_v8_target_exposure")
    if previous is not None and saved is None:
        raise ValueError("V8 resume checkpoint is missing its target-exposure history")
    if previous is None:
        if not phase_transition:
            raise ValueError("Starting a V8 target-exposure census requires a fresh stage")
        if saved is not None:
            raise ValueError("A non-V8 checkpoint cannot carry V8 target-exposure state")
        return empty_semantic_v8_target_exposure(registry_digest, starting_global_step=global_step)
    state = validate_semantic_v8_target_exposure(saved, registry_digest=registry_digest)
    if state["lastGlobalStep"] != global_step:
        raise ValueError("V8 target-exposure resume state differs from checkpoint step")
    return state


def validate_semantic_v8_target_exposure(state, *, registry_digest):
    required = {
        "schema",
        "registryDigest",
        "startingGlobalStep",
        "lastGlobalStep",
        "optimizerUpdates",
        "sources",
        "digest",
    }
    if (
        not isinstance(state, dict)
        or set(state) != required
        or state.get("schema") != V8_TARGET_EXPOSURE_SCHEMA
        or state.get("registryDigest") != registry_digest
        or type(state.get("startingGlobalStep")) is not int
        or type(state.get("lastGlobalStep")) is not int
        or type(state.get("optimizerUpdates")) is not int
        or state["startingGlobalStep"] < 0
        or state["lastGlobalStep"] < state["startingGlobalStep"]
        or state["optimizerUpdates"] != state["lastGlobalStep"] - state["startingGlobalStep"]
        or not isinstance(state.get("sources"), dict)
        or state.get("digest")
        != stable_algorithm_hash({key: child for key, child in state.items() if key != "digest"})
    ):
        raise ValueError("V8 optimizer target-exposure state is malformed or changed registry")
    for source_id, source in state["sources"].items():
        if (
            not isinstance(source_id, str)
            or not source_id
            or not isinstance(source, dict)
            or set(source) != {"split", "logicalTargets"}
            or source["split"] != "train"
            or not isinstance(source["logicalTargets"], dict)
        ):
            raise ValueError("V8 target-exposure source identity is malformed")
        for logical_id, target in source["logicalTargets"].items():
            if (
                not isinstance(logical_id, str)
                or not logical_id
                or not isinstance(target, dict)
                or set(target) != {"records"}
                or not isinstance(target["records"], dict)
            ):
                raise ValueError("V8 target-exposure logical target is malformed")
            for record_id, record in target["records"].items():
                if (
                    not isinstance(record_id, str)
                    or not record_id
                    or not isinstance(record, dict)
                    or set(record)
                    != {
                        "windowIndices",
                        "lossDenominators",
                        "scoredOccurrences",
                        "optimizerUpdates",
                        "firstGlobalStep",
                        "lastGlobalStep",
                    }
                    or not isinstance(record["windowIndices"], list)
                    or any(type(index) is not int or index < 0 for index in record["windowIndices"])
                    or len(record["windowIndices"]) != len(set(record["windowIndices"]))
                    or not isinstance(record["lossDenominators"], dict)
                    or any(
                        not isinstance(name, str) or type(count) is not int or count < 0
                        for name, count in record["lossDenominators"].items()
                    )
                    or not any(record["lossDenominators"].values())
                    or type(record["scoredOccurrences"]) is not int
                    or record["scoredOccurrences"] < 1
                    or type(record["optimizerUpdates"]) is not int
                    or record["optimizerUpdates"] < 1
                    or record["optimizerUpdates"] > record["scoredOccurrences"]
                    or type(record["firstGlobalStep"]) is not int
                    or type(record["lastGlobalStep"]) is not int
                    or not state["startingGlobalStep"]
                    < record["firstGlobalStep"]
                    <= record["lastGlobalStep"]
                    <= state["lastGlobalStep"]
                    or record["optimizerUpdates"] > state["optimizerUpdates"]
                ):
                    raise ValueError("V8 target-exposure physical record is malformed")
    return state


def _validate_semantic_v8_target_occurrence(occurrence, registry_digest):
    fields = {
        "sourceId",
        "registryDigest",
        "split",
        "logicalTargetId",
        "physicalRecordId",
        "windowIndex",
        "lossDenominators",
    }
    if (
        not isinstance(occurrence, dict)
        or set(occurrence) != fields
        or occurrence.get("registryDigest") != registry_digest
        or occurrence.get("split") != "train"
        or any(
            not isinstance(occurrence.get(key), str) or not occurrence[key]
            for key in ("sourceId", "logicalTargetId", "physicalRecordId")
        )
        or type(occurrence.get("windowIndex")) is not int
        or occurrence["windowIndex"] < 0
        or not isinstance(occurrence.get("lossDenominators"), dict)
        or any(
            not isinstance(name, str) or type(count) is not int or count < 0
            for name, count in occurrence["lossDenominators"].items()
        )
        or not any(occurrence["lossDenominators"].values())
    ):
        raise ValueError("V8 target-exposure occurrence is malformed or not loss-bearing")
    return occurrence


def semantic_v8_target_exposure_from_batches(batches, *, registry_digest, effective_loss_weights):
    """Collect exact loss-bearing source targets from ordinary prepared Meridian batches."""
    occurrences = []
    if not isinstance(effective_loss_weights, dict) or any(
        not isinstance(name, str)
        or isinstance(weight, bool)
        or not isinstance(weight, (int, float))
        or not math.isfinite(weight)
        or weight < 0
        for name, weight in effective_loss_weights.items()
    ):
        raise ValueError("V8 optimizer target exposure has invalid effective loss weights")
    for batch in batches:
        if len(batch.windows) != 1:
            raise ValueError("V8 target exposure requires one episode per prepared meridian batch")
        item = batch.windows[0]
        episode = item.episode
        annotations = json.loads(episode.supervision.annotations_json)
        semantic = annotations.get("semantic_execution", {})
        binding = semantic.get("datasetBinding")
        source_id = semantic.get("sourceId")
        split = episode.lineage.frozen_split
        if (
            episode.stream != "semantic_execution"
            or semantic.get("schema") != "semantic-execution/8"
            or not isinstance(binding, dict)
            or binding.get("registryDigest") != registry_digest
            or binding.get("sourceId") != source_id
            or binding.get("split") != split
            or split != "train"
            or not isinstance(source_id, str)
            or not source_id
        ):
            raise ValueError("V8 optimizer target has a changed source, registry or split binding")
        for step in batch.steps:
            if not step.plans:
                continue
            for encoded, plan in zip(step.records, step.plans, strict=True):
                record = encoded.record
                if not record.policy_mask:
                    raise ValueError("V8 optimizer plan exposes a record outside the target mask")
                denominators = plan.denominators
                if not isinstance(denominators, Mapping) or any(
                    type(value) is not int or value < 0 for value in denominators.values()
                ):
                    raise ValueError("V8 optimizer plan has invalid loss denominators")
                positive = {
                    name: count
                    for name, count in denominators.items()
                    if count
                    and name != "router_balance"
                    and effective_loss_weights.get(name, 0) > 0
                }
                if not positive:
                    continue
                occurrence = {
                    "sourceId": source_id,
                    "registryDigest": registry_digest,
                    "split": split,
                    "logicalTargetId": record.logical_id,
                    "physicalRecordId": record.record_id,
                    "windowIndex": item.window.index,
                    "lossDenominators": positive,
                }
                occurrences.append(
                    _validate_semantic_v8_target_occurrence(occurrence, registry_digest)
                )
    return {
        "schema": V8_TARGET_EXPOSURE_SCHEMA,
        "registryDigest": registry_digest,
        "occurrences": occurrences,
    }


def merge_semantic_v8_target_exposures(exposures, *, registry_digest):
    """Aggregate every scored occurrence while separately deduplicating target coverage."""
    merged = {"schema": V8_TARGET_EXPOSURE_SCHEMA, "registryDigest": registry_digest, "sources": {}}
    for exposure in exposures:
        if (
            not isinstance(exposure, dict)
            or exposure.get("schema") != V8_TARGET_EXPOSURE_SCHEMA
            or exposure.get("registryDigest") != registry_digest
            or not isinstance(exposure.get("occurrences"), list)
        ):
            raise ValueError("Distributed V8 target exposure changed its registry")
        for occurrence in exposure["occurrences"]:
            row = _validate_semantic_v8_target_occurrence(occurrence, registry_digest)
            source = merged["sources"].setdefault(
                row["sourceId"], {"split": row["split"], "logicalTargets": {}}
            )
            if source["split"] != row["split"]:
                raise ValueError("Distributed V8 target source changed split")
            logical = source["logicalTargets"].setdefault(row["logicalTargetId"], {"records": {}})
            record = logical["records"].setdefault(
                row["physicalRecordId"],
                {
                    "windowIndices": [],
                    "lossDenominators": {},
                    "scoredOccurrences": 0,
                },
            )
            record["windowIndices"] = sorted(set(record["windowIndices"]) | {row["windowIndex"]})
            record["scoredOccurrences"] += 1
            for name, count in row["lossDenominators"].items():
                record["lossDenominators"][name] = record["lossDenominators"].get(name, 0) + count
    return merged


def advance_semantic_v8_target_exposure(state, step_exposure, *, optimizer_step, registry_digest):
    validate_semantic_v8_target_exposure(state, registry_digest=registry_digest)
    if (
        type(optimizer_step) is not int
        or optimizer_step != state["lastGlobalStep"] + 1
        or step_exposure.get("schema") != V8_TARGET_EXPOSURE_SCHEMA
        or step_exposure.get("registryDigest") != registry_digest
    ):
        raise ValueError("V8 target exposure can advance only after the next optimizer update")
    updated = json.loads(json.dumps(state))
    updated["lastGlobalStep"] = optimizer_step
    updated["optimizerUpdates"] += 1
    for source_id, source in step_exposure["sources"].items():
        if source.get("split") != "train":
            raise ValueError("V8 target exposure can only advance with training sources")
        dest_source = updated["sources"].setdefault(
            source_id, {"split": "train", "logicalTargets": {}}
        )
        if dest_source["split"] != source["split"]:
            raise ValueError("V8 target exposure source changed split")
        for logical_id, target in source["logicalTargets"].items():
            dest_target = dest_source["logicalTargets"].setdefault(logical_id, {"records": {}})
            for record_id, current in target["records"].items():
                old = dest_target["records"].get(record_id)
                if old is None:
                    dest_target["records"][record_id] = {
                        "windowIndices": list(current["windowIndices"]),
                        "lossDenominators": dict(current["lossDenominators"]),
                        "scoredOccurrences": current["scoredOccurrences"],
                        "optimizerUpdates": 1,
                        "firstGlobalStep": optimizer_step,
                        "lastGlobalStep": optimizer_step,
                    }
                    continue
                old["windowIndices"] = sorted(
                    set(old["windowIndices"]) | set(current["windowIndices"])
                )
                old["lossDenominators"] = {
                    name: old["lossDenominators"].get(name, 0)
                    + current["lossDenominators"].get(name, 0)
                    for name in set(old["lossDenominators"]) | set(current["lossDenominators"])
                }
                old["scoredOccurrences"] += current["scoredOccurrences"]
                old["optimizerUpdates"] += 1
                old["lastGlobalStep"] = optimizer_step
    result = _seal_semantic_v8_target_exposure(updated)
    return validate_semantic_v8_target_exposure(result, registry_digest=registry_digest)


def semantic_v8_target_exposure_report(state):
    source_rows = {}
    loss_denominators = {}
    scored_occurrences = 0
    for source_id, source in sorted(state["sources"].items()):
        targets = source["logicalTargets"]
        source_rows[source_id] = {
            "split": source["split"],
            "logicalTargetIds": sorted(targets),
            "physicalTargetRecords": {
                logical_id: {
                    record_id: dict(record)
                    for record_id, record in sorted(target["records"].items())
                }
                for logical_id, target in sorted(targets.items())
            },
        }
        for target in targets.values():
            for record in target["records"].values():
                scored_occurrences += record["scoredOccurrences"]
                for name, value in record["lossDenominators"].items():
                    loss_denominators[name] = loss_denominators.get(name, 0) + value
    logical_count = sum(len(source["logicalTargets"]) for source in state["sources"].values())
    physical_count = sum(
        len(target["records"])
        for source in state["sources"].values()
        for target in source["logicalTargets"].values()
    )
    return {
        "schema": V8_TARGET_EXPOSURE_SCHEMA,
        "registryDigest": state["registryDigest"],
        "startingGlobalStep": state["startingGlobalStep"],
        "lastGlobalStep": state["lastGlobalStep"],
        "optimizerUpdates": state["optimizerUpdates"],
        "sourceCount": len(state["sources"]),
        "logicalTargetCount": logical_count,
        "physicalTargetRecordCount": physical_count,
        "scoredOccurrences": scored_occurrences,
        "lossDenominators": loss_denominators,
        "sources": source_rows,
        "digest": state["digest"],
    }


def semantic_v8_target_step_summary(step_exposure):
    logical_ids = {
        (source_id, logical_id)
        for source_id, source in step_exposure["sources"].items()
        for logical_id in source["logicalTargets"]
    }
    physical_ids = {
        (source_id, logical_id, record_id)
        for source_id, source in step_exposure["sources"].items()
        for logical_id, target in source["logicalTargets"].items()
        for record_id in target["records"]
    }
    denominators = {}
    occurrences = 0
    for source in step_exposure["sources"].values():
        for target in source["logicalTargets"].values():
            for record in target["records"].values():
                occurrences += record["scoredOccurrences"]
                for name, count in record["lossDenominators"].items():
                    denominators[name] = denominators.get(name, 0) + count
    return {
        "sourceCount": len(step_exposure["sources"]),
        "logicalTargetCount": len(logical_ids),
        "physicalTargetRecordCount": len(physical_ids),
        "scoredOccurrences": occurrences,
        "lossDenominators": denominators,
    }


def validate_semantic_training_policy_v8(policy):
    fields = {"schema", "producerRevision", "consumerRevision", "registryDigest"}
    if (
        not isinstance(policy, dict)
        or set(policy) != fields
        or policy.get("schema") != "semantic-training-policy/8"
        or any(
            not isinstance(policy.get(key), str)
            or re.fullmatch(r"[0-9a-f]{40}", policy[key]) is None
            for key in ("producerRevision", "consumerRevision")
        )
        or not isinstance(policy.get("registryDigest"), str)
        or re.fullmatch(r"[0-9a-f]{64}", policy["registryDigest"]) is None
    ):
        raise ValueError("V8 training requires exact producer, consumer and family-registry pins")
    return policy


def load_semantic_v8_codec(path, *, vocab_size):
    """Load only the pinned raw tokenizer bytes used by V8 admission."""
    from megatoken.generation.codec import ByteCodec

    source = Path(path)
    if source.is_dir():
        metadata = json.loads((source / "codec.json").read_text())
        if metadata.get("tokenizer_sha256") != LING_TOKENIZER_SHA256:
            raise ValueError("V8 training codec metadata does not pin the approved tokenizer bytes")
        tokenizer = source / metadata["tokenizer_file"]
        if tokenizer.parent.resolve() != source.resolve():
            raise ValueError("V8 tokenizer must remain inside its declared codec directory")
    else:
        tokenizer = source
    return ByteCodec.from_tokenizer_json(
        tokenizer,
        expected_sha256=LING_TOKENIZER_SHA256,
        expected_vocab_size=vocab_size,
    )


def _verify_registry(registry, expected_digest):
    required = {
        "schema",
        "policy",
        "traceFormat",
        "coreSchema",
        "seed",
        "holdouts",
        "sources",
        "components",
        "diagnostics",
        "digest",
    }
    if (
        not isinstance(registry, dict)
        or set(registry) != required
        or registry["schema"] != "semantic-region-family-registry/1"
        or registry["policy"] != "semantic-region-family-components/1"
        or registry["traceFormat"] != "semantic-trace/8"
        or registry["coreSchema"] != "semantic-algorithm/2"
        or type(registry["seed"]) is not int
        or registry["digest"] != expected_digest
        or stable_algorithm_hash({k: v for k, v in registry.items() if k != "digest"})
        != expected_digest
    ):
        raise ValueError("V8 prepared data has no exact frozen source-family registry")
    sources = registry["sources"]
    components = registry["components"]
    if not isinstance(sources, list) or not sources or not isinstance(components, list):
        raise ValueError("V8 registry is incomplete")
    ids = [row.get("id") for row in sources if isinstance(row, dict)]
    if len(ids) != len(sources) or len(ids) != len(set(ids)):
        raise ValueError("V8 registry source inventory is malformed")
    membership = {}
    for component in components:
        if (
            not isinstance(component, dict)
            or not isinstance(component.get("sourceIds"), list)
            or component.get("split") not in {"train", "validation", "test"}
        ):
            raise ValueError("V8 registry component or split is malformed")
        for source_id in component["sourceIds"]:
            if source_id in membership:
                raise ValueError("V8 source is assigned to more than one component")
            membership[source_id] = component
    if set(membership) != set(ids):
        raise ValueError("V8 registry does not assign every inventoried source exactly once")
    return sources, membership


def validate_semantic_episode_v8(
    episode,
    *,
    codec,
    policy,
    registry,
    max_history=None,
    max_positions=None,
    expected_split=None,
):
    from megatoken.data.native_v8 import prepare_native_v8_training
    from megatoken.data.source_identity_v8 import validate_source_binding_v8

    policy = validate_semantic_training_policy_v8(policy)
    sources, components = _verify_registry(registry, policy["registryDigest"])
    annotations = json.loads(episode.supervision.annotations_json)
    semantic = annotations.get("semantic_execution")
    expected = {"session_metadata", "semantic_execution", "semantic_geometry"}
    if set(annotations) != expected or not isinstance(semantic, dict):
        raise ValueError("V8 semantic annotations are missing or contain undeclared data")
    geometry = annotations["semantic_geometry"]
    if (
        episode.stream != "semantic_execution"
        or semantic.get("schema") != "semantic-execution/8"
        or semantic.get("representation") != "mixed"
        or semantic.get("admission") != "admitted"
        or semantic.get("continuation") != "teacher_conditioned"
        or semantic.get("targetEligibility") != "source_admitted"
        or semantic.get("modelExecutionOutcome") != "unmeasured"
        or geometry.get("schema") != "semantic-native-geometry/8"
        or any(
            type(geometry.get(key)) is not int or geometry[key] < 1
            for key in ("maxHistory", "maxPositions")
        )
    ):
        raise ValueError(
            "Ordinary V8 training requires source-admitted, freshly prepared V8 targets"
        )
    if max_history is not None and geometry["maxHistory"] > max_history:
        raise ValueError("V8 data exceeds model history geometry")
    if max_positions is not None and geometry["maxPositions"] > max_positions:
        raise ValueError("V8 data exceeds model position geometry")
    binding = semantic.get("datasetBinding")
    document = semantic.get("publicArtifact")
    sidecar = document.get("sidecar") if isinstance(document, dict) else None
    has_orchestration = isinstance(sidecar, dict) and "orchestrationTranscript" in sidecar
    orchestration = None
    if has_orchestration:
        from megatoken.data.semantic_orchestration_training_v8 import (
            validate_orchestration_transcript_v8,
        )

        orchestration = validate_orchestration_transcript_v8(document)
    if (
        not isinstance(binding, dict)
        or binding.get("schema") != "semantic-region-dataset-binding/1"
        or binding.get("traceFormat") != "semantic-trace/8"
        or binding.get("registryDigest") != policy["registryDigest"]
        or binding.get("componentId") != episode.lineage.family_id
        or binding.get("split") != episode.lineage.frozen_split
        or binding.get("sourceId") != semantic.get("sourceId")
        or has_orchestration
        and binding.get("orchestrationTranscriptDigest") != orchestration["transcriptDigest"]
        or not has_orchestration
        and "orchestrationTranscriptDigest" in binding
        or binding.get("split") != semantic.get("split")
        or expected_split is not None
        and binding.get("split") != expected_split
        or tuple(sorted("semantic-family:" + item for item in semantic.get("members", [])))
        != tuple(sorted(episode.lineage.ancestor_ids))
    ):
        raise ValueError("V8 training row moved outside its frozen family split or source binding")
    source = next((row for row in sources if row["id"] == semantic.get("sourceId")), None)
    component = components.get(semantic.get("sourceId"))
    if (
        source is None
        or component is None
        or component["id"] != binding["componentId"]
        or component["split"] != binding["split"]
        or source["sourceDigest"] != binding.get("sourceDigest")
        or source.get("coreDigest") != binding.get("coreDigest")
    ):
        raise ValueError("V8 exact source or component is absent from its complete registry")
    source_receipt, native = semantic.get("sourceAdmission"), semantic.get("nativeAdmission")
    native_checks = native.get("checks") if isinstance(native, dict) else None
    native_counts = native.get("counts") if isinstance(native, dict) else None
    if not isinstance(native_checks, dict):
        native_checks = {}
    if not isinstance(native_counts, dict):
        native_counts = {}
    direct_source = validate_source_binding_v8(binding)
    if (
        not isinstance(source_receipt, dict)
        or source_receipt.get("schema") != "semantic-native-source-recheck/1"
        or source_receipt.get("producerRevision") != policy["producerRevision"]
        or source_receipt.get("sourceValidated") is not True
        or source_receipt.get("projectionValidated") is not True
        or source_receipt.get("replayValidated") is not True
        or source_receipt.get("sourceArtifactDigest") != binding.get("sourceArtifactDigest")
        or direct_source
        and (
            source_receipt.get("sourceIdentityDigest") != binding["sourceIdentityDigest"]
            or source_receipt.get("identityValidated") is not True
        )
        or source_receipt.get("publicDigest") != binding.get("episodeDigest")
        or source_receipt.get("contextDigest") != binding.get("contextsDigest")
        or source_receipt.get("familyBindingValidated") is not True
        or source_receipt.get("registryDigest") != policy["registryDigest"]
        or source_receipt.get("datasetBindingDigest") != stable_algorithm_hash(binding)
        or has_orchestration
        and (
            source_receipt.get("orchestrationValidated") is not True
            or source_receipt.get("orchestrationTranscriptDigest")
            != orchestration["transcriptDigest"]
        )
        or not has_orchestration
        and any(
            key in source_receipt
            for key in ("orchestrationValidated", "orchestrationTranscriptDigest")
        )
        or not isinstance(native, dict)
        or native.get("schema") != "semantic-native-admission/8"
        or native.get("traceFormat") != "semantic-trace/8"
        or native.get("contextFormat") != "semantic-context/8"
        or native.get("componentGate") != "native-consumer-only"
        or native.get("admitted") is not True
        or native.get("consumerRevision") != policy["consumerRevision"]
        or native.get("contractDigest") != CONTRACT_DIGEST
        or native.get("fixtureDigest") != FIXTURE_DIGEST
        or native.get("interoperabilityDigest") != INTEROPERABILITY_DIGEST
        or native.get("publicDigest") != binding.get("episodeDigest")
        or native.get("contextsDigest") != binding.get("contextsDigest")
        or native.get("sourceArtifactDigest") != binding.get("sourceArtifactDigest")
        or direct_source
        and native.get("sourceIdentityDigest") != binding["sourceIdentityDigest"]
        or native.get("datasetBindingDigest") != stable_algorithm_hash(binding)
        or has_orchestration
        and (
            native.get("orchestrationTranscriptDigest") != orchestration["transcriptDigest"]
            or native_checks.get("orchestration") != "passed"
            or native_counts.get("orchestrationRecords") != orchestration["recordCount"]
        )
        or not has_orchestration
        and (
            "orchestrationTranscriptDigest" in native
            or "orchestration" in native_checks
            or "orchestrationRecords" in native_counts
        )
        or native.get("tokenizer")
        != {"status": "passed", "identity": codec.fingerprint, "sha256": LING_TOKENIZER_SHA256}
        or semantic.get("tokenizerEvidence")
        != {
            "schema": "semantic-tokenizer-admission/1",
            "status": "passed",
            "identity": codec.fingerprint,
            "sha256": LING_TOKENIZER_SHA256,
            "verification": "pinned-tokenizer-bytes",
        }
        or native.get("nativeTrainingDigest") != semantic.get("nativeTrainingDigest")
        or native.get("sourceVerification") != source_receipt
        or native.get("optimizerSteps") != 0
        or native.get("learningClaim") is not None
        or any(
            native_checks.get(key) != "passed"
            for key in ("replay", "contexts", "continuation", "masks", "sourceBinding", "tokenizer")
        )
        or has_orchestration
        and native_checks.get("orchestration") != "passed"
    ):
        raise ValueError(
            "Fresh source/native V8 admission, tokenizer or training evidence is missing"
        )
    # Rebuild physical supervision and compare the content digest. This makes
    # receipt presence insufficient if a shard changes after admission.
    if not isinstance(document, dict):
        raise ValueError("V8 artifact bytes were not retained for native revalidation")
    rebuilt, training = prepare_native_v8_training(
        document,
        codec=codec,
        max_positions=geometry["maxPositions"],
        max_history=geometry["maxHistory"],
    )
    if orchestration is not None:
        from megatoken.data.semantic_orchestration_training_v8 import (
            validate_orchestration_records_v8,
        )

        if (
            training.get("orchestrationRecords") != orchestration["recordCount"]
            or training.get("orchestrationTranscriptDigest") != orchestration["transcriptDigest"]
            or not validate_orchestration_records_v8(episode, orchestration)
        ):
            raise ValueError(
                "V8 lifecycle targets or observations differ from the retained transcript"
            )
    if any(
        getattr(episode.lineage, key) != getattr(rebuilt.lineage, key)
        for key in (
            "source_id",
            "revision",
            "repository",
            "task_id",
            "parent_episode_id",
            "source_uri",
            "created_at",
            "source_row",
            "raw_sha256",
        )
    ):
        raise ValueError("V8 source lineage differs from the source-bound artifact")
    normalized_records = tuple(
        replace(record, provenance="unknown")
        if record.policy_mask
        and not (record.logical_id or record.record_id).startswith(
            "semantic:v8:orchestration:target:"
        )
        else record
        for record in episode.records
    )
    normalized = replace(
        episode,
        records=normalized_records,
        lineage=rebuilt.lineage,
        supervision=rebuilt.supervision,
    )
    if (
        training["nativeTrainingDigest"] != semantic["nativeTrainingDigest"]
        or normalized != rebuilt
    ):
        raise ValueError("V8 physical records or target masks differ from admitted artifact bytes")
    result = {
        "generatedRecords": training["generatedRecords"],
        "split": binding["split"],
        "sourceId": source["id"],
    }
    if orchestration is not None:
        result["orchestrationRecords"] = orchestration["recordCount"]
    return result


def validate_prepared_semantic_v8(roots, *, codec, policy, max_history=None, max_positions=None):
    from megatoken.data.storage import PreparedCorpus

    policy = validate_semantic_training_policy_v8(policy)
    with PreparedCorpus(roots, split=None) as corpus:
        if not len(corpus) or corpus.codec_fingerprint != codec.fingerprint:
            raise ValueError("V8 prepared supply is empty or uses a different codec")
        registry = None
        for part in corpus.parts:
            receipts = [
                row
                for row in part.manifest.get("sources", [])
                if row.get("schema") == "semantic-preparation/8"
            ]
            if not receipts:
                raise ValueError("V8 semantic roots require a V8 preparation manifest")
            for receipt in receipts:
                if (
                    receipt.get("producerRevision") != policy["producerRevision"]
                    or receipt.get("consumerRevision") != policy["consumerRevision"]
                    or receipt.get("registryDigest") != policy["registryDigest"]
                    or receipt.get("tokenizerIdentity") != codec.fingerprint
                    or receipt.get("tokenizerSha256") != LING_TOKENIZER_SHA256
                    or receipt.get("contractDigest") != CONTRACT_DIGEST
                    or receipt.get("fixtureDigest") != FIXTURE_DIGEST
                    or receipt.get("optimizerSteps") != 0
                ):
                    raise ValueError("V8 preparation manifest differs from the selected policy")
            saved = json.loads((part.root / "semantic-v8-families.json").read_text())
            _verify_registry(saved, policy["registryDigest"])
            if registry is not None and saved != registry:
                raise ValueError("V8 prepared roots use different complete family registries")
            registry = saved
        counts = {"train": 0, "validation": 0, "test": 0}
        for episode in corpus.iter_episodes():
            validate_semantic_episode_v8(
                episode,
                codec=codec,
                policy=policy,
                registry=registry,
                max_history=max_history,
                max_positions=max_positions,
                expected_split=episode.lineage.frozen_split,
            )
            counts[episode.lineage.frozen_split] += 1
        return {
            "schema": "semantic-stage-supply/8",
            "registryDigest": policy["registryDigest"],
            "episodes": counts,
            "optimizerSteps": 0,
        }


def preflight_semantic_training_v8(roots, *, codec, policy, max_history=None, max_positions=None):
    """Detect V8 rows and fail closed on missing, stale, or mismatched admission."""
    from megatoken.data import PreparedCorpus

    with PreparedCorpus(roots, split=None) as corpus:
        receipts = [
            receipt for part in corpus.parts for receipt in part.manifest.get("sources", [])
        ]
        has_v8_receipt = any(
            receipt.get("schema") == "semantic-preparation/8" for receipt in receipts
        )
        semantic_schemas = set()
        for episode in corpus.iter_episodes():
            semantic = json.loads(episode.supervision.annotations_json).get("semantic_execution")
            if semantic is not None:
                semantic_schemas.add(semantic.get("schema"))
        corpus_codec_fingerprint = corpus.codec_fingerprint

    if policy is None:
        if has_v8_receipt or "semantic-execution/8" in semantic_schemas:
            raise ValueError("V8 semantic rows require an explicit pinned training policy")
        return None

    policy = validate_semantic_training_policy_v8(policy)
    if semantic_schemas - {"semantic-execution/8"}:
        raise ValueError("V8 training cannot include semantic rows with an older or unknown schema")
    if not has_v8_receipt or "semantic-execution/8" not in semantic_schemas:
        raise ValueError("V8 training policy requires matching V8 prepared semantic rows")
    if corpus_codec_fingerprint != codec.fingerprint:
        raise ValueError("Corpus and training codec differ")
    return validate_prepared_semantic_v8(
        roots,
        codec=codec,
        policy=policy,
        max_history=max_history,
        max_positions=max_positions,
    )
