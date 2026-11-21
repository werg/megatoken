"""Small fail-closed checks for the ordinary V8 training admission policy."""

import base64
import gzip
import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from megatoken.data.semantic_corpus_v8 import prepare_semantic_corpus_v8
from megatoken.data.semantic_json import stable_algorithm_hash
from megatoken.generation.codec import ByteCodec
from megatoken.training.config import TrainConfig
from megatoken.training.semantic_v8 import (
    validate_prepared_semantic_v8,
    validate_semantic_episode_v8,
    validate_semantic_training_policy_v8,
)
from tests.test_source_identity_v8 import direct_source_sidecar


def pinned_tokenizer_path():
    fallback = Path(__file__).resolve().parents[1].parent / "checkpoint" / "tokenizer.json"
    return Path(os.environ.get("MEGATOKEN_TOKENIZER_JSON", fallback))


POLICY = {
    "schema": "semantic-training-policy/8",
    "producerRevision": "a" * 40,
    "consumerRevision": "b" * 40,
    "registryDigest": "0" * 64,
}


def test_v8_policy_is_exact_and_pinned():
    assert validate_semantic_training_policy_v8(POLICY) == POLICY
    for invalid in (
        {**POLICY, "schema": "semantic-training-policy/7"},
        {**POLICY, "extra": True},
        {**POLICY, "producerRevision": "dirty"},
    ):
        with pytest.raises(ValueError, match="exact producer"):
            validate_semantic_training_policy_v8(invalid)


def test_unverified_data_flag_cannot_admit_unqualified_v8_targets(tmp_path):
    registry = {
        "schema": "semantic-region-family-registry/1",
        "policy": "semantic-region-family-components/1",
        "traceFormat": "semantic-trace/8",
        "coreSchema": "semantic-algorithm/2",
        "seed": 17,
        "holdouts": {"families": [], "sourceDigests": [], "coreDigests": []},
        "sources": [{"id": "source-1", "sourceDigest": "d" * 64, "coreDigest": None}],
        "components": [{"id": "component-1", "sourceIds": ["source-1"], "split": "train"}],
        "diagnostics": {},
    }
    registry["digest"] = stable_algorithm_hash(registry)
    policy = {**POLICY, "registryDigest": registry["digest"]}
    config = TrainConfig(
        purpose="diagnostic",
        phase="SC",
        model="tiny",
        codec="injected-tokenizer",
        datasets=[str(tmp_path)],
        max_steps=1,
        allow_unverified_data=True,
        semantic_v8_policy=policy,
    )
    assert config.allow_unverified_data
    annotations = {
        "session_metadata": {},
        "semantic_geometry": {
            "schema": "semantic-native-geometry/8",
            "maxHistory": 8,
            "maxPositions": 32,
        },
        "semantic_execution": {
            "schema": "semantic-execution/8",
            "representation": "mixed",
            "admission": "admitted",
            "continuation": "teacher_conditioned",
            # Deliberately no targetEligibility or source/native proof.
            "modelExecutionOutcome": "unmeasured",
        },
    }
    episode = SimpleNamespace(
        stream="semantic_execution",
        supervision=SimpleNamespace(annotations_json=json.dumps(annotations)),
        lineage=SimpleNamespace(family_id="component-1", frozen_split="train", ancestor_ids=()),
    )
    with pytest.raises(ValueError, match="source-admitted"):
        validate_semantic_episode_v8(
            episode,
            codec=SimpleNamespace(fingerprint="e" * 64),
            policy=policy,
            registry=registry,
        )


def prepared_v8_fixture(
    tmp_path, *, max_positions=128, max_history=8, direct_source=False, document=None, codec=None
):
    """Real native rows with explicitly injected source/tokenizer audit boundaries."""
    if document is None:
        fixture = Path(__file__).parent / "fixtures" / "semantic-regions-v8.json.gz.b64"
        document = json.loads(gzip.decompress(base64.b64decode(fixture.read_bytes())))["examples"][
            0
        ]
    source = document["public"]["timeline"][0]["fields"]["source"]
    source_digest = hashlib.sha256(source.encode()).hexdigest()
    registry = {
        "schema": "semantic-region-family-registry/1",
        "policy": "semantic-region-family-components/1",
        "traceFormat": "semantic-trace/8",
        "coreSchema": "semantic-algorithm/2",
        "seed": 17,
        "holdouts": {"families": [], "sourceDigests": [], "coreDigests": []},
        "sources": [
            {
                "id": "source-1",
                "family": "family-1",
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
                "id": "component-1",
                "members": ["family-1"],
                "sourceIds": ["source-1"],
                "split": "train",
                "holdout": None,
            }
        ],
        "diagnostics": {},
    }
    registry["digest"] = stable_algorithm_hash(registry)
    source_artifact = {"sidecar": {"referenceInput": {"source": source, "entry": "run"}}}
    binding = {
        "schema": "semantic-region-dataset-binding/1",
        "traceFormat": "semantic-trace/8",
        "episodeDigest": stable_algorithm_hash(document["public"]),
        "contextsDigest": stable_algorithm_hash(document["contexts"]),
        "sourceArtifactDigest": stable_algorithm_hash(source_artifact),
        "registryDigest": registry["digest"],
        "componentId": "component-1",
        "split": "train",
        "sourceId": "source-1",
        "family": "family-1",
        "sourceDigest": source_digest,
        "coreDigest": None,
    }
    document["sidecar"] = {
        "schema": "semantic-region-evidence/1",
        "profile": "compositional-regions/1",
        "sourceArtifact": source_artifact,
        "splitIdentity": {
            "schema": "semantic-region-split-identity/1",
            "registry": registry,
            "registryDigest": registry["digest"],
            "sourceId": "source-1",
            "sourceDigest": source_digest,
            "coreDigest": None,
            "componentId": "component-1",
            "members": ["family-1"],
            "split": "train",
        },
        "split": "train",
        "sourceComponent": "component-1",
        "familyRegistryDigest": registry["digest"],
        "datasetBinding": binding,
    }
    if direct_source:
        document["sidecar"].pop("sourceArtifact")
        document["sidecar"].update(direct_source_sidecar())
        binding.update(
            sourceArtifactDigest=None, sourceIdentityDigest=document["sidecar"]["identityDigest"]
        )
    producer_receipt = {
        "schema": "semantic-native-source-recheck/1",
        "producerRevision": POLICY["producerRevision"],
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
    }
    if direct_source:
        producer_receipt.update(
            sourceIdentityDigest=binding["sourceIdentityDigest"], identityValidated=True
        )
    codec = codec or ByteCodec.byte_fallback()
    # Test an explicit injected tokenizer-verification boundary. Production
    # preparation obtains this evidence only after checking the pinned file.
    tokenizer_evidence = {
        "schema": "semantic-tokenizer-admission/1",
        "status": "passed",
        "identity": codec.fingerprint,
        "sha256": "40fb9d7d7795b8bd305aeff39ce9963f3f450915b9553f2938e009be9a1fed60",
        "verification": "pinned-tokenizer-bytes",
    }
    output = tmp_path / "prepared-v8"
    prepare_semantic_corpus_v8(
        [document],
        output,
        codec=codec,
        producer_revision=POLICY["producerRevision"],
        consumer_revision=POLICY["consumerRevision"],
        source_verifier=lambda _: producer_receipt,
        tokenizer_evidence=tokenizer_evidence,
        max_positions=max_positions,
        max_history=max_history,
    )
    policy = {**POLICY, "registryDigest": registry["digest"]}
    supply = validate_prepared_semantic_v8(
        [str(output)],
        codec=codec,
        policy=policy,
        max_history=max_history,
        max_positions=max_positions,
    )
    assert supply["episodes"]["train"] == 1
    assert supply["episodes"]["validation"] == 0
    return document, codec, policy, output


@pytest.mark.parametrize("direct_source", [False, True])
def test_freshly_prepared_v8_supply_passes_complete_native_batch_admission(tmp_path, direct_source):
    document, codec, policy, output = prepared_v8_fixture(tmp_path, direct_source=direct_source)
    registry = document["sidecar"]["splitIdentity"]["registry"]
    import torch

    from megatoken.data import PreparedCorpus, iter_windows
    from megatoken.data.media_assets import TensorAssetStore
    from megatoken.loading import build_model, load_model_config
    from megatoken.training.engine import prepare_batches, window_config
    from megatoken.training.stream import EpisodeWindow

    config = TrainConfig(
        purpose="diagnostic",
        phase="SC",
        model="tiny",
        codec="injected-tokenizer",
        datasets=[str(output)],
        max_steps=1,
        allow_unverified_data=True,
        semantic_v8_policy=policy,
    )
    torch.set_num_threads(1)
    model = build_model(load_model_config("tiny"), codec, device="cpu", precision="fp32")
    with PreparedCorpus([str(output)], split="train") as corpus:
        episode = next(corpus.iter_episodes())
        windows = [
            EpisodeWindow(episode, item, TensorAssetStore("text-only"))
            for item in iter_windows(episode, codec, window_config(config))
        ]
        batches = prepare_batches(model, windows, config)
    assert batches
    if direct_source:
        from dataclasses import replace

        original = json.loads(episode.supervision.annotations_json)
        for location in ("sourceAdmission", "nativeAdmission", "datasetBinding"):
            changed = json.loads(json.dumps(original))
            changed["semantic_execution"][location]["sourceIdentityDigest"] = "0" * 64
            forged = replace(
                episode,
                supervision=replace(episode.supervision, annotations_json=json.dumps(changed)),
            )
            with pytest.raises(ValueError, match="source/native"):
                validate_semantic_episode_v8(forged, codec=codec, policy=policy, registry=registry)
        changed = json.loads(json.dumps(original))
        changed["semantic_execution"]["sourceAdmission"]["identityValidated"] = False
        forged = replace(
            episode,
            supervision=replace(episode.supervision, annotations_json=json.dumps(changed)),
        )
        with pytest.raises(ValueError, match="source/native"):
            validate_semantic_episode_v8(forged, codec=codec, policy=policy, registry=registry)


def test_generic_unverified_flag_cannot_hide_or_relabel_a_semantic_schema():
    from megatoken.training.engine import prepare_batches

    config = TrainConfig(
        purpose="diagnostic", model="tiny", allow_unverified_data=True, max_steps=1
    )
    for schema in (None, "semantic-execution/9", "anything"):
        episode = SimpleNamespace(
            stream="semantic_execution",
            supervision=SimpleNamespace(
                annotations_json=json.dumps(
                    {"semantic_execution": {} if schema is None else {"schema": schema}}
                )
            ),
        )
        with pytest.raises(ValueError, match="missing or unknown"):
            prepare_batches(SimpleNamespace(), [SimpleNamespace(episode=episode)], config)


def test_v8_codec_rejects_byte_fallback_and_unpinned_file(tmp_path):
    from megatoken.training.semantic_v8 import load_semantic_v8_codec

    with pytest.raises(ValueError, match="pinned raw tokenizer"):
        TrainConfig(purpose="diagnostic", model="tiny", semantic_v8_policy=POLICY)
    invalid = tmp_path / "tokenizer.json"
    invalid.write_text("{}")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        load_semantic_v8_codec(invalid, vocab_size=256)


def test_readiness_uses_v8_policy_and_strict_codec_preflight(tmp_path, monkeypatch):
    from megatoken.training.readiness import check

    _, codec, policy, output = prepared_v8_fixture(tmp_path)
    monkeypatch.setattr(
        "megatoken.training.semantic_v8.load_semantic_v8_codec",
        lambda path, *, vocab_size: codec,
    )
    config = TrainConfig(
        purpose="diagnostic",
        phase="SC",
        model="tiny",
        codec="injected-tokenizer",
        datasets=[str(output)],
        output=str(tmp_path / "ready"),
        model_overrides={
            "vocab_size": codec.vocab_size,
            "max_history_chunks": 8,
            "max_capacity": 128,
        },
        generated_capacity=128,
        max_steps=1,
        semantic_v8_policy=policy,
    )

    report = check(config)

    assert report["ready"], report["errors"]
    assert report["checks"]["semantic_v8"]["episodes"]["train"] == 1


def test_readiness_refuses_missing_and_mismatched_v8_admission(tmp_path, monkeypatch):
    from megatoken.training.readiness import check

    _, codec, policy, output = prepared_v8_fixture(tmp_path)
    monkeypatch.setattr(
        "megatoken.training.semantic_v8.load_semantic_v8_codec",
        lambda path, *, vocab_size: codec,
    )
    config = TrainConfig(
        purpose="diagnostic",
        phase="SC",
        model="tiny",
        codec="injected-tokenizer",
        datasets=[str(output)],
        output=str(tmp_path / "ready"),
        model_overrides={
            "vocab_size": codec.vocab_size,
            "max_history_chunks": 8,
            "max_capacity": 128,
        },
        generated_capacity=128,
        max_steps=1,
        semantic_v8_policy=policy,
    )

    missing_policy = check(replace(config, phase="B0", codec="byte", semantic_v8_policy=None))
    assert not missing_policy["ready"]
    assert any("explicit pinned training policy" in error for error in missing_policy["errors"])

    mismatched = check(replace(config, semantic_v8_policy={**policy, "registryDigest": "0" * 64}))
    assert not mismatched["ready"]
    assert any("preparation manifest differs" in error for error in mismatched["errors"])


def test_readiness_refuses_v8_rows_without_preparation_manifest(tmp_path, monkeypatch):
    from megatoken.training.readiness import check

    _, codec, policy, output = prepared_v8_fixture(tmp_path)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["sources"] = [
        receipt
        for receipt in manifest["sources"]
        if receipt.get("schema") != "semantic-preparation/8"
    ]
    unsigned = {key: value for key, value in manifest.items() if key != "fingerprint"}
    from megatoken.io import content_hash

    manifest["fingerprint"] = content_hash(unsigned)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")))
    monkeypatch.setattr(
        "megatoken.training.semantic_v8.load_semantic_v8_codec",
        lambda path, *, vocab_size: codec,
    )
    config = TrainConfig(
        purpose="diagnostic",
        phase="SC",
        model="tiny",
        codec="injected-tokenizer",
        datasets=[str(output)],
        output=str(tmp_path / "ready"),
        model_overrides={
            "vocab_size": codec.vocab_size,
            "max_history_chunks": 8,
            "max_capacity": 128,
        },
        generated_capacity=128,
        max_steps=1,
        semantic_v8_policy=policy,
        allow_unverified_data=True,
    )

    report = check(config)

    assert not report["ready"]
    assert any("matching V8 prepared semantic rows" in error for error in report["errors"])


def test_readiness_loads_the_actual_pinned_tokenizer_when_available(tmp_path):
    from megatoken.generation.codec import LING_TOKENIZER_SHA256, LING_VOCAB_SIZE, ByteCodec
    from megatoken.training.readiness import check

    tokenizer = pinned_tokenizer_path()
    if not tokenizer.exists():
        pytest.skip("The separately provisioned pinned tokenizer is unavailable")
    codec = ByteCodec.from_tokenizer_json(
        tokenizer, expected_sha256=LING_TOKENIZER_SHA256, expected_vocab_size=LING_VOCAB_SIZE
    )
    _, _, policy, output = prepared_v8_fixture(tmp_path, codec=codec)
    config = TrainConfig(
        purpose="diagnostic",
        phase="SC",
        model="tiny",
        codec=str(tokenizer),
        datasets=[str(output)],
        output=str(tmp_path / "ready"),
        model_overrides={
            "vocab_size": codec.vocab_size,
            "max_history_chunks": 8,
            "max_capacity": 128,
        },
        generated_capacity=128,
        max_steps=1,
        semantic_v8_policy=policy,
    )

    report = check(config)

    assert report["ready"], report["errors"]
    assert report["checks"]["semantic_v8"]["episodes"]["train"] == 1
    mismatch = check(replace(config, semantic_v8_policy={**policy, "registryDigest": "0" * 64}))
    assert not mismatch["ready"]
    assert any("preparation manifest differs" in error for error in mismatch["errors"])


def test_semantic_v8_production_template_passes_real_config_validation(tmp_path):
    from megatoken.generation.codec import LING_TOKENIZER_SHA256, LING_VOCAB_SIZE, ByteCodec
    from megatoken.io import read_mapping

    root = Path(__file__).resolve().parents[1]
    tokenizer = pinned_tokenizer_path()
    if not tokenizer.exists():
        pytest.skip("The separately provisioned pinned tokenizer is unavailable")
    codec = ByteCodec.from_tokenizer_json(
        tokenizer, expected_sha256=LING_TOKENIZER_SHA256, expected_vocab_size=LING_VOCAB_SIZE
    )
    _, _, policy, prepared = prepared_v8_fixture(tmp_path, codec=codec)
    donors = [tmp_path / "deepseek-donor", tmp_path / "qwen-donor"]
    for donor in donors:
        donor.mkdir()
    hindsight = tmp_path / "hindsight-sc"
    hindsight.mkdir()
    mapping = read_mapping(root / "configs/train/semantic-v8-production.example.yaml")
    mapping.update(
        datasets=[str(prepared)],
        codec=str(tokenizer),
        hindsight_targets=str(hindsight),
        engram_imports=[str(path) for path in donors],
        semantic_v8_policy=policy,
        decoupled={**mapping["decoupled"], "coordinator_url": "https://coordinator.test"},
    )

    config = TrainConfig.from_dict(mapping, expand_data=False)

    assert config.purpose == "production" and config.phase == "SC"
    assert config.semantic_v8_policy == policy
    assert config.codec == str(tokenizer)
    assert config.hindsight_targets == str(hindsight)
    assert config.decoupled.workers == 8


@pytest.mark.parametrize("schema", ["semantic-execution/7", "semantic-execution/999"])
def test_readiness_refuses_old_or_unknown_semantic_schema_even_when_unverified(
    tmp_path, monkeypatch, schema
):
    from megatoken.training.readiness import check

    _, codec, policy, output = prepared_v8_fixture(tmp_path)
    monkeypatch.setattr(
        "megatoken.training.semantic_v8.load_semantic_v8_codec",
        lambda path, *, vocab_size: codec,
    )

    class FakeCorpus:
        codec_fingerprint = codec.fingerprint
        parts = [SimpleNamespace(manifest={"sources": [{"schema": "semantic-preparation/8"}]})]

        def __init__(self, roots, split=None):
            del roots, split

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def iter_episodes(self):
            return iter(
                [
                    SimpleNamespace(
                        supervision=SimpleNamespace(
                            annotations_json=json.dumps({"semantic_execution": {"schema": schema}})
                        )
                    )
                ]
            )

    monkeypatch.setattr("megatoken.data.PreparedCorpus", FakeCorpus)
    config = TrainConfig(
        purpose="diagnostic",
        phase="SC",
        model="tiny",
        codec="injected-tokenizer",
        datasets=[str(output)],
        output=str(tmp_path / "ready"),
        model_overrides={
            "vocab_size": codec.vocab_size,
            "max_history_chunks": 8,
            "max_capacity": 128,
        },
        generated_capacity=128,
        max_steps=1,
        semantic_v8_policy=policy,
        allow_unverified_data=True,
    )

    report = check(config)

    assert not report["ready"]
    assert any("older or unknown schema" in error for error in report["errors"])
