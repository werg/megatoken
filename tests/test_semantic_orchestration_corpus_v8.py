"""Ordinary V8 training binds source-qualified lifecycle targets to their receipts."""

from __future__ import annotations

import base64
import gzip
import json
from dataclasses import replace
from pathlib import Path

import pytest

from megatoken.data.semantic_corpus_v8 import (
    prepare_semantic_corpus_v8,
    semantic_behaviors_v8,
)
from megatoken.data.semantic_json import stable_algorithm_hash
from megatoken.data.storage import PreparedCorpus
from megatoken.generation.codec import LING_TOKENIZER_SHA256, ByteCodec
from megatoken.training.semantic_v8 import validate_semantic_episode_v8

PRODUCER_REVISION = "a" * 40
CONSUMER_REVISION = "b" * 40


def source_qualified_artifact():
    path = Path(__file__).parent / "fixtures" / "semantic-orchestration-teacher-v8.json.gz.b64"
    document = json.loads(gzip.decompress(base64.b64decode(path.read_bytes())))
    sidecar = document["sidecar"]
    source = document["public"]["timeline"][0]["fields"]["source"]
    source_digest = stable_algorithm_hash(source)
    family = sidecar["family"]
    registry = {
        "schema": "semantic-region-family-registry/1",
        "policy": "semantic-region-family-components/1",
        "traceFormat": "semantic-trace/8",
        "coreSchema": "semantic-algorithm/2",
        "seed": 17,
        "holdouts": {"families": [], "sourceDigests": [], "coreDigests": []},
        "sources": [
            {
                "id": "orchestration-source",
                "family": family,
                "ancestors": [],
                "source": source,
                "entry": "run",
                "sourceDigest": source_digest,
                "coreDigest": None,
                "coreStatus": "detailed",
            }
        ],
        "components": [
            {
                "id": "orchestration-component",
                "members": [family],
                "sourceIds": ["orchestration-source"],
                "split": "train",
                "holdout": None,
            }
        ],
        "diagnostics": {},
    }
    registry["digest"] = stable_algorithm_hash(registry)
    sidecar["splitIdentity"] = {
        "schema": "semantic-region-split-identity/1",
        "registry": registry,
        "registryDigest": registry["digest"],
        "sourceId": "orchestration-source",
        "sourceDigest": source_digest,
        "coreDigest": None,
        "componentId": "orchestration-component",
        "members": [family],
        "split": "train",
    }
    sidecar.update(
        split="train",
        sourceComponent="orchestration-component",
        familyRegistryDigest=registry["digest"],
    )
    binding = {
        "schema": "semantic-region-dataset-binding/1",
        "traceFormat": "semantic-trace/8",
        "episodeDigest": stable_algorithm_hash(document["public"]),
        "contextsDigest": stable_algorithm_hash(document["contexts"]),
        "sourceArtifactDigest": stable_algorithm_hash(sidecar["sourceArtifact"]),
        "registryDigest": registry["digest"],
        "componentId": "orchestration-component",
        "split": "train",
        "sourceId": "orchestration-source",
        "family": family,
        "sourceDigest": source_digest,
        "coreDigest": None,
        "orchestrationTranscriptDigest": stable_algorithm_hash(sidecar["orchestrationTranscript"]),
    }
    sidecar["datasetBinding"] = binding
    source_receipt = {
        "schema": "semantic-native-source-recheck/1",
        "producerRevision": PRODUCER_REVISION,
        "producerTreeClean": True,
        "artifactSha256": "f" * 64,
        "sourceArtifactDigest": binding["sourceArtifactDigest"],
        "publicDigest": binding["episodeDigest"],
        "contextDigest": binding["contextsDigest"],
        "sourceValidated": True,
        "projectionValidated": True,
        "replayValidated": True,
        "familyBindingValidated": True,
        "registryDigest": registry["digest"],
        "datasetBindingDigest": stable_algorithm_hash(binding),
        "orchestrationValidated": True,
        "orchestrationTranscriptDigest": binding["orchestrationTranscriptDigest"],
    }
    return document, registry, binding, source_receipt


def prepare_episode(tmp_path):
    document, registry, binding, source_receipt = source_qualified_artifact()
    codec = ByteCodec.byte_fallback()
    tokenizer_evidence = {
        "schema": "semantic-tokenizer-admission/1",
        "status": "passed",
        "identity": codec.fingerprint,
        "sha256": LING_TOKENIZER_SHA256,
        "verification": "pinned-tokenizer-bytes",
    }
    output = tmp_path / "prepared-v8"
    _, report = prepare_semantic_corpus_v8(
        [document],
        output,
        codec=codec,
        producer_revision=PRODUCER_REVISION,
        consumer_revision=CONSUMER_REVISION,
        source_verifier=lambda _: source_receipt,
        tokenizer_evidence=tokenizer_evidence,
        max_positions=8192,
        max_history=1024,
    )
    with PreparedCorpus([str(output)], split="train") as corpus:
        episode = next(corpus.iter_episodes())
    policy = {
        "schema": "semantic-training-policy/8",
        "producerRevision": PRODUCER_REVISION,
        "consumerRevision": CONSUMER_REVISION,
        "registryDigest": registry["digest"],
    }
    return document, registry, binding, codec, policy, episode, report


def test_source_qualified_lifecycle_is_real_ordinary_training_and_coverage(tmp_path):
    document, registry, binding, codec, policy, episode, report = prepare_episode(tmp_path)
    annotations = json.loads(episode.supervision.annotations_json)["semantic_execution"]
    native = annotations["nativeAdmission"]
    assert report["counts"]["orchestrationRecords"] == 4
    assert native["checks"]["orchestration"] == "passed"
    assert native["counts"]["orchestrationRecords"] == 4
    assert native["orchestrationTranscriptDigest"] == binding["orchestrationTranscriptDigest"]
    assert annotations["sourceAdmission"]["orchestrationValidated"] is True
    assert (
        annotations["datasetBinding"]["orchestrationTranscriptDigest"]
        == binding["orchestrationTranscriptDigest"]
    )
    result = validate_semantic_episode_v8(
        episode, codec=codec, policy=policy, registry=registry, expected_split="train"
    )
    assert result["orchestrationRecords"] == 4
    assert "semantic_child" in semantic_behaviors_v8(episode)

    naked = {"public": document["public"], "contexts": document["contexts"]}
    from megatoken.data.native_v8 import prepare_native_v8_training

    naked_episode, _ = prepare_native_v8_training(
        naked, codec=codec, max_positions=8192, max_history=1024
    )
    assert naked_episode.episode_id != episode.episode_id


def test_parent_call_edge_without_persisted_lifecycle_never_counts_as_child(tmp_path):
    _, _, _, _, _, episode, _ = prepare_episode(tmp_path)
    annotations = json.loads(episode.supervision.annotations_json)
    artifact = annotations["semantic_execution"]["publicArtifact"]
    artifact["sidecar"].pop("orchestrationTranscript")
    artifact["public"]["timeline"][1]["fields"]["parentInvocationId"] = "parent-call"
    changed = replace(
        episode,
        supervision=replace(
            episode.supervision,
            annotations_json=json.dumps(annotations, separators=(",", ":")),
        ),
    )
    assert "semantic_child" not in semantic_behaviors_v8(changed)


def test_census_requires_native_lifecycle_event_order(tmp_path):
    _, registry, _, codec, policy, episode, _ = prepare_episode(tmp_path)
    delivery_id = "semantic:v8:orchestration:delivery:6"
    completion_id = "semantic:v8:orchestration:completion:6"
    logical_groups = [list(parts) for parts in episode.iter_logical_records()]
    group_ids = [parts[0].logical_id or parts[0].record_id for parts in logical_groups]
    delivery_index = group_ids.index(delivery_id)
    completion_index = group_ids.index(completion_id)
    logical_groups[delivery_index], logical_groups[completion_index] = (
        logical_groups[completion_index],
        logical_groups[delivery_index],
    )
    rows = []
    for turn, parts in enumerate(logical_groups):
        rows.extend(replace(part, turn_index=turn) for part in parts)
    events = {event.record_id: event for meridian in episode.meridians for event in meridian.events}
    meridians = tuple(
        type(episode.meridians[0])(
            index,
            (replace(events[record.record_id], meridian_id=index),),
        )
        for index, record in enumerate(rows)
    )
    changed = replace(episode, records=tuple(rows), meridians=meridians)

    assert "semantic_child" not in semantic_behaviors_v8(changed)
    with pytest.raises(ValueError, match="lifecycle targets or observations"):
        validate_semantic_episode_v8(changed, codec=codec, policy=policy, registry=registry)


@pytest.mark.parametrize("mutation", ["ack-bytes", "instruction", "origin", "null-transcript"])
def test_policy_and_census_refuse_mutated_lifecycle_evidence(tmp_path, mutation):
    document, registry, _, codec, policy, episode, _ = prepare_episode(tmp_path)
    if mutation == "null-transcript":
        malformed = json.loads(json.dumps(document))
        malformed["sidecar"]["orchestrationTranscript"] = None
        from megatoken.data.native_v8 import prepare_native_v8_training

        with pytest.raises(ValueError, match="transcript"):
            prepare_native_v8_training(malformed, codec=codec, max_positions=8192, max_history=1024)
        return

    annotations = json.loads(episode.supervision.annotations_json)
    if mutation == "origin":
        artifact = annotations["semantic_execution"]["publicArtifact"]
        artifact["sidecar"]["sourceArtifact"]["sidecar"]["selectionOrigin"] = "controlled-fixture"
        changed = replace(
            episode,
            supervision=replace(
                episode.supervision,
                annotations_json=json.dumps(annotations, separators=(",", ":")),
            ),
        )
        assert "semantic_child" not in semantic_behaviors_v8(changed)
        with pytest.raises(ValueError, match="origin"):
            validate_semantic_episode_v8(changed, codec=codec, policy=policy, registry=registry)
        return

    if mutation == "ack-bytes":
        index = next(
            i
            for i, record in enumerate(episode.records)
            if (record.logical_id or record.record_id) == "semantic:v8:orchestration:ack:2"
        )
        rows = list(episode.records)
        rows[index] = replace(rows[index], payload=b"x" * len(rows[index].payload))
        changed = replace(episode, records=tuple(rows))
    else:
        groups = list(episode.meridians)
        group_index = next(
            i
            for i, group in enumerate(groups)
            if any(
                event.record_id
                in {
                    record.record_id
                    for record in episode.records
                    if (record.logical_id or record.record_id)
                    == "semantic:v8:orchestration:target:1"
                }
                for event in group.events
            )
        )
        group = groups[group_index]
        events = tuple(
            replace(event, instruction="changed instruction")
            if event.track_id == "track:0" and event.role == "assistant"
            else event
            for event in group.events
        )
        groups[group_index] = replace(group, events=events)
        changed = replace(episode, meridians=tuple(groups))

    assert "semantic_child" not in semantic_behaviors_v8(changed)
    with pytest.raises(ValueError, match="lifecycle targets or observations"):
        validate_semantic_episode_v8(changed, codec=codec, policy=policy, registry=registry)
