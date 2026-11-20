"""Fail-closed V8 semantic data policy for the ordinary trainer."""

from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path

from megatoken.data.semantic_contract_v8 import (
    CONTRACT_DIGEST,
    FIXTURE_DIGEST,
    INTEROPERABILITY_DIGEST,
)
from megatoken.data.semantic_json import stable_algorithm_hash
from megatoken.generation.codec import LING_TOKENIZER_SHA256


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
    if (
        not isinstance(binding, dict)
        or binding.get("schema") != "semantic-region-dataset-binding/1"
        or binding.get("traceFormat") != "semantic-trace/8"
        or binding.get("registryDigest") != policy["registryDigest"]
        or binding.get("componentId") != episode.lineage.family_id
        or binding.get("split") != episode.lineage.frozen_split
        or binding.get("sourceId") != semantic.get("sourceId")
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
            native.get("checks", {}).get(key) != "passed"
            for key in ("replay", "contexts", "continuation", "masks", "sourceBinding", "tokenizer")
        )
    ):
        raise ValueError(
            "Fresh source/native V8 admission, tokenizer or training evidence is missing"
        )
    # Rebuild physical supervision and compare the content digest. This makes
    # receipt presence insufficient if a shard changes after admission.
    document = semantic.get("publicArtifact")
    if not isinstance(document, dict):
        raise ValueError("V8 artifact bytes were not retained for native revalidation")
    rebuilt, training = prepare_native_v8_training(
        document,
        codec=codec,
        max_positions=geometry["maxPositions"],
        max_history=geometry["maxHistory"],
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
        replace(record, provenance="unknown") if record.policy_mask else record
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
    return {
        "generatedRecords": training["generatedRecords"],
        "split": binding["split"],
        "sourceId": source["id"],
    }


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
