"""Training gates for newly prepared v7 execution data and family splits."""

from __future__ import annotations

import json
import re

from megatoken.data.algorithm_v7 import parse_algorithm_record, stable_algorithm_hash
from megatoken.data.host_operations import CONTRACT_DIGEST
from megatoken.data.schema import encode_record
from megatoken.data.semantic_admission_v7 import (
    native_training_digest_v7,
    validate_native_visibility_v7,
)
from megatoken.generation.codec import LING_TOKENIZER_SHA256


def validate_semantic_training_policy_v7(policy):
    if (
        not isinstance(policy, dict)
        or set(policy) != {"schema", "producerRevision", "consumerRevision", "registryDigest"}
        or policy["schema"] != "semantic-training-policy/7"
        or any(
            not isinstance(policy[key], str) or re.fullmatch(r"[0-9a-f]{40}", policy[key]) is None
            for key in ("producerRevision", "consumerRevision")
        )
        or not isinstance(policy["registryDigest"], str)
        or re.fullmatch(r"[0-9a-f]{64}", policy["registryDigest"]) is None
    ):
        raise ValueError(
            "V7 training requires explicit exact producer, consumer and family-registry pins"
        )
    return policy


def validate_semantic_episode_v7(
    episode, *, codec, policy, max_history=None, max_positions=None, expected_split=None
):
    from megatoken.data.semantic_contract_v7 import FIXTURE_DIGEST, SCHEMA_DIGEST

    validate_semantic_training_policy_v7(policy)
    annotations = json.loads(episode.supervision.annotations_json)
    execution = annotations.get("semantic_execution", {})
    geometry = annotations.get("semantic_geometry", {})
    expected_annotations = {
        "session_metadata",
        "semantic_execution",
        "semantic_geometry",
        "semantic_algorithm"
        if execution.get("representation") == "algorithm"
        else "semantic_harness",
    }
    if set(annotations) != expected_annotations:
        raise ValueError("V7 prepared data contains undeclared model or private annotations")
    if (
        episode.stream != "semantic_execution"
        or not episode.supervision.verified_success
        or execution.get("schema") != "semantic-execution/7"
        or execution.get("representation") not in {"algorithm", "detailed"}
        or execution.get("continuation") != "teacher_conditioned"
        or execution.get("admission") != "admitted"
    ):
        raise ValueError(
            "Training requires newly prepared, independently admitted v7 teacher records"
        )
    if geometry.get("schema") != "semantic-native-geometry/7" or any(
        type(geometry.get(key)) is not int or geometry[key] < 1
        for key in ("maxHistory", "maxPositions")
    ):
        raise ValueError("V7 prepared data lost its exact physical geometry")
    if (
        max_history is not None
        and geometry["maxHistory"] > max_history
        or max_positions is not None
        and geometry["maxPositions"] > max_positions
    ):
        raise ValueError("V7 data exceeds this model's context geometry; regenerate prepared data")
    if (
        execution.get("split") != episode.lineage.frozen_split
        or execution.get("componentId") != episode.lineage.family_id
        or expected_split is not None
        and execution.get("split") != expected_split
    ):
        raise ValueError("V7 training row moved outside its frozen semantic-family split")
    source, native = execution.get("sourceAdmission", {}), execution.get("nativeAdmission", {})
    binding = execution.get("datasetBinding")
    binding_fields = {
        "schema",
        "representation",
        "episodeDigest",
        "contextsDigest",
        "registryDigest",
        "componentId",
        "split",
        "sourceId",
        "sourceDigest",
        "coreDigest",
    }
    if (
        not isinstance(binding, dict)
        or set(binding) != binding_fields
        or binding["schema"] != "semantic-dataset-binding/1"
        or binding["representation"] != execution["representation"]
        or binding["episodeDigest"] != execution.get("publicDigest")
        or binding["contextsDigest"] != execution.get("contextDigest")
        or binding["registryDigest"] != policy["registryDigest"]
        or any(
            binding[key] != execution.get(key)
            for key in ("registryDigest", "componentId", "sourceId", "split")
        )
        or source.get("datasetBindingDigest") != stable_algorithm_hash(binding)
        or native.get("datasetBindingDigest") != stable_algorithm_hash(binding)
    ):
        raise ValueError(
            "V7 dataset split/source membership differs from both independent admission receipts"
        )
    if (
        source.get("schema") != "semantic-source-admission/7"
        or source.get("producerRevision") != policy["producerRevision"]
        or source.get("referenceStatus") != "agreed"
        or source.get("optimizerSteps") != 0
        or source.get("episodeDigest") != binding["episodeDigest"]
        or source.get("contextsDigest") != binding["contextsDigest"]
        or source.get("registryDigest") != execution.get("registryDigest")
        or any(
            source.get(key) is not True
            for key in ("sourceValidated", "journalValidated", "replayValidated")
        )
    ):
        raise ValueError("V7 prepared source/family proof is missing")
    if (
        native.get("schema") != "semantic-native-admission/7"
        or native.get("consumerRevision") != policy["consumerRevision"]
        or native.get("componentGate") != "native-consumer-only"
        or native.get("traceFormat") != "semantic-trace/7"
        or native.get("contractDigest") != CONTRACT_DIGEST
        or native.get("tokenizerSha256") != LING_TOKENIZER_SHA256
        or native.get("optimizerSteps") != 0
        or native.get("schemaDigest") != SCHEMA_DIGEST
        or native.get("fixtureDigest") != FIXTURE_DIGEST
        or native.get("tokenizerIdentity") != codec.fingerprint
        or native.get("representation") != execution["representation"]
        or native.get("episodeDigest") != source.get("episodeDigest")
        or native.get("contextsDigest") != source.get("contextsDigest")
        or native.get("nativeTrainingDigest") != native_training_digest_v7(episode)
        or any(
            native.get(key) is not True
            for key in (
                "admitted",
                "lossMaskValidated",
                "visibilityValidated",
                "fragmentationValidated",
                "contextValidated",
                "continuationValidated",
            )
        )
    ):
        raise ValueError(
            "V7 prepared labels, model controls, source bytes or tokenizer differ from admission"
        )
    generated = 0
    for parts in episode.iter_logical_records():
        if parts[0].policy_mask:
            if any(part.provenance != "teacher" or part.role != "assistant" for part in parts):
                raise ValueError("V7 generated loss requires certified teacher provenance")
            line = b"".join(part.payload for part in parts).decode("utf-8")
            if execution["representation"] == "algorithm":
                parse_algorithm_record(line, role="generated")
            else:
                from megatoken.data.detailed_v7 import parse_record

                parse_record(line, role="generated")
            generated += 1
        for part in parts:
            encoded = encode_record(part, codec)
            if not part.policy_mask and encoded.P or encoded.tokens > geometry["maxPositions"]:
                raise ValueError("V7 observed data acquired loss or exceeded native framing")
    if generated == 0:
        raise ValueError("V7 prepared data has no generated teaching records")
    validate_native_visibility_v7(episode)
    return {
        "generatedRecords": generated,
        "representation": execution["representation"],
        "split": execution["split"],
    }


def validate_prepared_semantic_v7(
    roots, *, codec, producer_revision, consumer_revision, max_history=None, max_positions=None
):
    from megatoken.data.semantic_contract_v7 import FIXTURE_DIGEST, SCHEMA_DIGEST
    from megatoken.data.semantic_family_v7 import verify_family_registry_v7
    from megatoken.data.storage import PreparedCorpus

    counts, registry_digest, index_splits = {}, None, {}
    with PreparedCorpus(roots, split=None) as corpus:
        if not len(corpus) or corpus.codec_fingerprint != codec.fingerprint:
            raise ValueError("V7 prepared supply is empty or uses a different tokenizer")
        for part in corpus.parts:
            receipts = part.manifest.get("sources", [])
            if not receipts or any(
                receipt.get("schema") != "semantic-preparation/7"
                or receipt.get("producerRevision") != producer_revision
                or receipt.get("consumerRevision") != consumer_revision
                or receipt.get("schemaDigest") != SCHEMA_DIGEST
                or receipt.get("fixtureDigest") != FIXTURE_DIGEST
                or receipt.get("tokenizerIdentity") != codec.fingerprint
                for receipt in receipts
            ):
                raise ValueError("V7 manifest requires exact newly admitted producer/consumer pins")
            for receipt in receipts:
                if registry_digest is None:
                    registry_digest = receipt["registryDigest"]
                if receipt["registryDigest"] != registry_digest:
                    raise ValueError("V7 prepared roots use different semantic-family registries")
            registry = json.loads((part.root / "semantic-v7-families.json").read_text())
            if verify_family_registry_v7(registry) != registry_digest:
                raise ValueError("V7 prepared registry differs from its manifest")
            for indexed in part.connection.execute("SELECT episode_id, split FROM episodes"):
                if (
                    indexed["episode_id"] in index_splits
                    and index_splits[indexed["episode_id"]] != indexed["split"]
                ):
                    raise ValueError("V7 prepared indices place one episode in different splits")
                index_splits[indexed["episode_id"]] = indexed["split"]
        members = {row["id"]: row for row in registry["sources"]}
        components = {
            source_id: component
            for component in registry["components"]
            for source_id in component["sourceIds"]
        }
        policy = {
            "schema": "semantic-training-policy/7",
            "producerRevision": producer_revision,
            "consumerRevision": consumer_revision,
            "registryDigest": registry_digest,
        }
        for episode in corpus.iter_episodes():
            execution = json.loads(episode.supervision.annotations_json)["semantic_execution"]
            member, component = (
                members.get(execution.get("sourceId")),
                components.get(execution.get("sourceId")),
            )
            binding = execution.get("datasetBinding", {})
            if (
                member is None
                or component is None
                or execution.get("componentId") != component["id"]
                or execution.get("split") != component["split"]
                or binding.get("sourceDigest") != member["sourceDigest"]
                or binding.get("coreDigest") != member["coreDigest"]
                or episode.lineage.ancestor_ids
                != tuple("semantic-family:" + name for name in component["members"])
            ):
                raise ValueError(
                    "V7 prepared episode differs from its complete verified family registry"
                )
            row = validate_semantic_episode_v7(
                episode,
                codec=codec,
                policy=policy,
                max_history=max_history,
                max_positions=max_positions,
                expected_split=index_splits[episode.episode_id],
            )
            counts[row["split"]] = counts.get(row["split"], 0) + 1
        identity = corpus.identity
    return {
        "schema": "semantic-stage-supply/7",
        "corpusIdentity": identity,
        "registryDigest": registry_digest,
        "episodes": counts,
        "producerRevision": producer_revision,
        "consumerRevision": consumer_revision,
        "tokenizerIdentity": codec.fingerprint,
        "optimizerSteps": 0,
    }
