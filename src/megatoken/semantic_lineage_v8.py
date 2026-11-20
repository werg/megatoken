"""Retain ordinary V8 training identities through checkpoint and served export.

These records bind existing preparation/training evidence to an artifact. They
do not certify learned execution or make a model-quality claim.
"""

from __future__ import annotations

import copy
import re

from megatoken.data.semantic_contract_v8 import (
    CONTRACT_DIGEST,
    FIXTURE_DIGEST,
    INTEROPERABILITY_DIGEST,
)
from megatoken.data.semantic_json import stable_algorithm_hash
from megatoken.generation.codec import LING_TOKENIZER_SHA256
from megatoken.io import content_hash
from megatoken.training.semantic_v8 import validate_semantic_training_policy_v8


def _hash(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _seal(value):
    return {**value, "digest": stable_algorithm_hash(value)}


def build_training_lineage_v8(
    *, policy, tokenizer_identity, prepared_manifests, corpus_identity, stream_identity, chunking
):
    record = _seal(
        {
            "schema": "semantic-training-lineage/8",
            "policy": copy.deepcopy(policy),
            "contractDigest": CONTRACT_DIGEST,
            "fixtureDigest": FIXTURE_DIGEST,
            "interoperabilityDigest": INTEROPERABILITY_DIGEST,
            "tokenizerIdentity": tokenizer_identity,
            "tokenizerSha256": LING_TOKENIZER_SHA256,
            "preparedManifestFingerprints": list(prepared_manifests),
            "preparedCorpusDigest": corpus_identity,
            "streamDigest": stream_identity,
            "chunkingFingerprint": chunking["fingerprint"],
        }
    )
    validate_training_lineage_v8(record)
    return record


def validate_training_lineage_v8(
    record, *, policy=None, tokenizer_identity=None, stream_identity=None, chunking_fingerprint=None
):
    fields = {
        "schema", "policy", "contractDigest", "fixtureDigest", "interoperabilityDigest",
        "tokenizerIdentity", "tokenizerSha256", "preparedManifestFingerprints",
        "preparedCorpusDigest", "streamDigest", "chunkingFingerprint", "digest",
    }
    if not isinstance(record, dict) or set(record) != fields:
        raise ValueError("Incomplete V8 training lineage")
    validate_semantic_training_policy_v8(record["policy"])
    manifests = record["preparedManifestFingerprints"]
    if (
        record["schema"] != "semantic-training-lineage/8"
        or record["contractDigest"] != CONTRACT_DIGEST
        or record["fixtureDigest"] != FIXTURE_DIGEST
        or record["interoperabilityDigest"] != INTEROPERABILITY_DIGEST
        or record["tokenizerSha256"] != LING_TOKENIZER_SHA256
        or not isinstance(manifests, list)
        or not manifests
        or not all(_hash(value) for value in manifests)
        or manifests != sorted(manifests)
        or record["preparedCorpusDigest"] != content_hash(manifests)
        or not all(_hash(record[key]) for key in (
            "tokenizerIdentity", "streamDigest", "chunkingFingerprint", "digest"
        ))
        or record["digest"] != stable_algorithm_hash(
            {key: value for key, value in record.items() if key != "digest"}
        )
    ):
        raise ValueError("V8 training lineage identities or integrity differ")
    for expected, actual in (
        (policy, record["policy"]),
        (tokenizer_identity, record["tokenizerIdentity"]),
        (stream_identity, record["streamDigest"]),
        (chunking_fingerprint, record["chunkingFingerprint"]),
    ):
        if expected is not None and expected != actual:
            raise ValueError("V8 training lineage differs from the enclosing checkpoint")
    return record


def checkpoint_lineage_v8(payload, metadata):
    policy = payload["train_config"].get("semantic_v8_policy")
    record = payload.get("semantic_training_v8")
    if policy is None:
        if record is not None or metadata.get("semantic_v8_lineage_digest") is not None:
            raise ValueError("V8 checkpoint lineage has no matching training policy")
        return None
    validate_training_lineage_v8(
        record,
        policy=policy,
        tokenizer_identity=payload["codec_fingerprint"],
        stream_identity=metadata.get("dataset_fingerprint"),
        chunking_fingerprint=payload["chunking"]["fingerprint"],
    )
    if (
        metadata.get("semantic_v8_lineage_digest") != record["digest"]
        or metadata.get("dataset_fingerprint") != record["streamDigest"]
    ):
        raise ValueError("V8 checkpoint metadata omits or changes its training lineage")
    return copy.deepcopy(record)


def build_served_lineage_v8(training, manifest):
    validate_training_lineage_v8(training, tokenizer_identity=manifest["codec_fingerprint"])
    return _seal(
        {
            "schema": "semantic-served-lineage/8",
            "training": copy.deepcopy(training),
            "trainingCheckpointSha256": manifest["checkpoint_sha256"],
            "modelIdentity": manifest["model_sha256"],
            "configSha256": manifest["config_sha256"],
            "transducerSha256": manifest["transducer_sha256"],
            "modelVersion": manifest["model_version"],
        }
    )


def validate_served_lineage_v8(record, manifest):
    fields = {"schema", "training", "trainingCheckpointSha256", "modelIdentity",
              "configSha256", "transducerSha256", "modelVersion", "digest"}
    if not isinstance(record, dict) or set(record) != fields:
        raise ValueError("Incomplete served V8 training lineage")
    validate_training_lineage_v8(
        record["training"], tokenizer_identity=manifest["codec_fingerprint"],
        chunking_fingerprint=manifest["chunking"]["fingerprint"],
    )
    if (
        record["schema"] != "semantic-served-lineage/8"
        or not _hash(record["digest"])
        or record["digest"] != stable_algorithm_hash(
            {key: value for key, value in record.items() if key != "digest"}
        )
        or record["digest"] != manifest.get("semantic_v8_lineage_digest")
        or any(record[key] != manifest[target] for key, target in (
            ("trainingCheckpointSha256", "checkpoint_sha256"),
            ("modelIdentity", "model_sha256"),
            ("configSha256", "config_sha256"),
            ("transducerSha256", "transducer_sha256"),
            ("modelVersion", "model_version"),
        ))
        or any(not _hash(record[key]) for key in (
            "trainingCheckpointSha256", "modelIdentity", "configSha256", "transducerSha256"
        ))
        or type(record["modelVersion"]) is not int
        or record["modelVersion"] < 0
    ):
        raise ValueError("Served V8 lineage is attached to different weights or training state")
    return copy.deepcopy(record)
