"""Identity binding controls, separate from real source/tokenizer qualification."""

import copy
import json

import pytest
import torch

from megatoken.data.chunking import ChunkingConfig
from megatoken.data.semantic_json import stable_algorithm_hash
from megatoken.generation.codec import ByteCodec
from megatoken.io import content_hash, sha256_file
from megatoken.semantic_lineage_v8 import (
    build_served_lineage_v8,
    build_training_lineage_v8,
    checkpoint_lineage_v8,
    validate_served_lineage_v8,
    validate_training_lineage_v8,
)


def seal(record):
    record["digest"] = stable_algorithm_hash(
        {key: value for key, value in record.items() if key != "digest"}
    )
    return record


def fixture(codec=None, chunking=None):
    codec = codec or ByteCodec.byte_fallback()
    chunking = chunking or ChunkingConfig().with_hard_limit(8).identity(codec)
    policy = {
        "schema": "semantic-training-policy/8",
        "producerRevision": "a" * 40,
        "consumerRevision": "b" * 40,
        "registryDigest": "c" * 64,
    }
    manifests = ["1" * 64, "2" * 64]
    lineage = build_training_lineage_v8(
        policy=policy,
        tokenizer_identity=codec.fingerprint,
        prepared_manifests=manifests,
        corpus_identity=content_hash(manifests),
        stream_identity="3" * 64,
        chunking=chunking,
    )
    payload = {
        "train_config": {"semantic_v8_policy": policy},
        "codec_fingerprint": codec.fingerprint,
        "chunking": chunking,
        "semantic_training_v8": lineage,
    }
    metadata = {"dataset_fingerprint": "3" * 64, "semantic_v8_lineage_digest": lineage["digest"]}
    return lineage, payload, metadata


def test_checkpoint_lineage_binds_policy_prepared_corpus_stream_codec_and_chunking():
    lineage, payload, metadata = fixture()
    assert checkpoint_lineage_v8(payload, metadata) == lineage
    changes = [
        lambda p, m: p["train_config"]["semantic_v8_policy"].update(registryDigest="4" * 64),
        lambda p, m: p.update(codec_fingerprint="5" * 64),
        lambda p, m: p["chunking"].update(fingerprint="6" * 64),
        lambda p, m: m.update(dataset_fingerprint="7" * 64),
        lambda p, m: m.pop("semantic_v8_lineage_digest"),
        lambda p, m: p.pop("semantic_training_v8"),
        lambda p, m: p["train_config"].pop("semantic_v8_policy"),
    ]
    for change in changes:
        changed, saved = copy.deepcopy(payload), copy.deepcopy(metadata)
        change(changed, saved)
        with pytest.raises(ValueError, match="V8"):
            checkpoint_lineage_v8(changed, saved)
    for key, value in (
        ("preparedManifestFingerprints", ["9" * 64]),
        ("contractDigest", "8" * 64),
        ("tokenizerSha256", "7" * 64),
    ):
        changed = copy.deepcopy(lineage)
        changed[key] = value
        with pytest.raises(ValueError, match="V8"):
            validate_training_lineage_v8(seal(changed))


def test_served_lineage_binds_all_model_resources_and_the_training_checkpoint():
    lineage, payload, _ = fixture()
    manifest = {
        "codec_fingerprint": payload["codec_fingerprint"],
        "chunking": payload["chunking"],
        "checkpoint_sha256": "4" * 64,
        "model_sha256": "5" * 64,
        "config_sha256": "6" * 64,
        "transducer_sha256": "7" * 64,
        "model_version": 3,
    }
    record = build_served_lineage_v8(lineage, manifest)
    manifest["semantic_v8_lineage_digest"] = record["digest"]
    assert validate_served_lineage_v8(record, manifest) == record
    for key in (
        "trainingCheckpointSha256",
        "modelIdentity",
        "configSha256",
        "transducerSha256",
        "modelVersion",
    ):
        changed = copy.deepcopy(record)
        changed[key] = 4 if key == "modelVersion" else "9" * 64
        seal(changed)
        changed_manifest = {**manifest, "semantic_v8_lineage_digest": changed["digest"]}
        with pytest.raises(ValueError, match="different weights"):
            validate_served_lineage_v8(changed, changed_manifest)


def test_ordinary_v8_export_and_reload_retain_lineage_and_reject_missing_proof(
    tmp_path, monkeypatch
):
    import megatoken.training.semantic_v8 as semantic_v8
    from megatoken.served import ServedBundle, export_served
    from megatoken.training.checkpoint import (
        load_checkpoint,
        save_checkpoint,
        transducer_attachment,
    )
    from tests.test_served import checkpoint_model

    model, original = checkpoint_model(tmp_path / "original")
    payload, _ = load_checkpoint(original)
    codec = ByteCodec.byte_fallback()
    lineage, binding, metadata = fixture(codec, payload["chunking"])
    payload["train_config"]["semantic_v8_policy"] = binding["train_config"]["semantic_v8_policy"]
    payload["semantic_training_v8"] = lineage
    # This metadata test explicitly injects tokenizer verification. Production
    # load/export still requires the pinned raw tokenizer and full preparation.
    monkeypatch.setattr(semantic_v8, "load_semantic_v8_codec", lambda path, *, vocab_size: codec)
    checkpoint = save_checkpoint(
        tmp_path / "checkpoint",
        "lineage",
        payload,
        {**metadata, "global_step": 7},
        attachments=transducer_attachment(lambda: {"bank": model.transducer.bank.state_snapshot()}),
    )
    directory = tmp_path / "served"
    manifest = export_served(checkpoint, directory)
    assert manifest["semantic_v8_lineage_digest"]
    loaded = ServedBundle(directory, device="cpu")
    assert loaded.semantic_v8_lineage["training"] == lineage
    assert loaded.semantic_v8_lineage["modelVersion"] == 7
    for name, expected in model.served_state_dict().items():
        assert torch.equal(expected, loaded.model.state_dict()[name]), name
    saved = json.loads((directory / "manifest.json").read_text())
    changed = dict(saved)
    changed.pop("semantic_v8_lineage_digest")
    (directory / "manifest.json").write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="lineage binding is incomplete"):
        ServedBundle(directory, device="cpu")
    (directory / "manifest.json").write_text(json.dumps(saved))
    proof_path = directory / "semantic-v8-lineage.json"
    proof = json.loads(proof_path.read_text())
    proof["modelIdentity"] = "9" * 64
    seal(proof)
    proof_path.write_text(json.dumps(proof))
    saved.update(
        semantic_v8_lineage_sha256=sha256_file(proof_path),
        semantic_v8_lineage_digest=proof["digest"],
    )
    (directory / "manifest.json").write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="different weights"):
        ServedBundle(directory, device="cpu")


def test_v8_checkpoint_cannot_export_through_generic_byte_codec_relocation(tmp_path):
    from megatoken.loading import load_trained
    from megatoken.training.checkpoint import (
        load_checkpoint,
        save_checkpoint,
        transducer_attachment,
    )
    from tests.test_served import checkpoint_model

    model, original = checkpoint_model(tmp_path / "original")
    payload, _ = load_checkpoint(original)
    lineage, binding, metadata = fixture(chunking=payload["chunking"])
    payload["train_config"]["semantic_v8_policy"] = binding["train_config"]["semantic_v8_policy"]
    payload["semantic_training_v8"] = lineage
    checkpoint = save_checkpoint(
        tmp_path / "checkpoint",
        "lineage",
        payload,
        {**metadata, "global_step": 7},
        attachments=transducer_attachment(lambda: {"bank": model.transducer.bank.state_snapshot()}),
    )
    with pytest.raises((ValueError, FileNotFoundError)):
        load_trained(checkpoint, codec_path="byte")
