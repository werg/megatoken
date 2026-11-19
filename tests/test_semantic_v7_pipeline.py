"""Cross-repository v7 admission and real-tokenizer prepared-training regression."""

import copy
import json
import math
import os
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from megatoken.data.algorithm_v7 import stable_algorithm_hash
from megatoken.data.semantic_admission_v7 import (
    adapt_semantic_v7,
    native_admission_v7,
    validate_native_visibility_v7,
)
from megatoken.data.semantic_corpus_v7 import prepare_semantic_corpus_v7
from megatoken.data.storage import PreparedCorpus
from megatoken.generation.codec import LING_TOKENIZER_SHA256, ByteCodec
from megatoken.training.semantic_v7 import (
    validate_prepared_semantic_v7,
    validate_semantic_episode_v7,
)


@pytest.fixture(scope="module")
def paired_candidates():
    root = Path(__file__).resolve().parents[2]
    producer = Path(os.environ.get("TRACING_JS_SELF_ROOT", root / "tracing-js-self"))
    tokenizer = Path(
        os.environ.get("MEGATOKEN_TOKENIZER_JSON", root / "Ling-3.0-tiny-tokenizer.json")
    )
    if (
        not producer.joinpath("tools/verify-algorithm-admission.mjs").is_file()
        or not tokenizer.is_file()
    ):
        pytest.skip(
            "Pinned v7 producer and production tokenizer are required by the paired CI gate"
        )
    script = """
import {verifyAlgorithmTrainingCase} from './src/semantic/compiler.js';
import {SEMANTIC_GALLERY_FIXTURES} from './src/training/semantic-trace-gallery.js';
import {prepareSemanticFamilyRegistry,bindSemanticFamilySplit} from './src/semantic/dataset-v7.js';
const candidates=[];
for(const fixture of SEMANTIC_GALLERY_FIXTURES){
 const result=await verifyAlgorithmTrainingCase(structuredClone(fixture));
 if(!result.exported)throw new Error(fixture.id+': '+result.reason);
 candidates.push({id:fixture.id,...result});
}
const prepared=prepareSemanticFamilyRegistry(candidates,{seed:17,families:['async-cleanup']});
console.log(JSON.stringify(prepared.members.map(member=>({id:member.candidate.id,
 exported:bindSemanticFamilySplit(member.candidate,member,prepared.registry)}))));
"""
    generated = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=producer,
        text=True,
        capture_output=True,
        timeout=120,
        check=True,
    )
    codec = ByteCodec.from_tokenizer_json(tokenizer, expected_sha256=LING_TOKENIZER_SHA256)
    return producer, codec, json.loads(generated.stdout)


def test_every_gallery_profile_has_real_native_admission_and_causal_masks(paired_candidates):
    _, codec, candidates = paired_candidates
    representations = set()
    for candidate in candidates:
        try:
            receipt = native_admission_v7(
                candidate["exported"],
                codec=codec,
                consumer_revision="a" * 40,
                max_positions=1024,
                max_history=1024,
            )
        except ValueError as error:
            pytest.fail(f"Fresh gallery case {candidate['id']} failed native admission: {error}")
        representations.add(receipt["representation"])
        assert receipt["contextValidated"] and receipt["continuationValidated"]
        assert receipt["lossMaskValidated"] and receipt["visibilityValidated"]
        assert receipt["generatedTokens"] > 0 and receipt["observedTokens"] > 0
        assert receipt["continuations"] > 0 and receipt["optimizerSteps"] == 0
    assert representations == {"algorithm", "detailed"}


def test_production_tokenizer_and_version_are_mandatory(paired_candidates):
    _, codec, candidates = paired_candidates
    value = candidates[0]["exported"]
    with pytest.raises(ValueError, match="tokenizer"):
        native_admission_v7(value, codec=ByteCodec.byte_fallback(), consumer_revision="a" * 40)
    old = copy.deepcopy(value)
    old["public"]["schema"] = "semantic-execution/6"
    with pytest.raises(ValueError, match="Only newly"):
        native_admission_v7(old, codec=codec, consumer_revision="a" * 40)


def test_native_history_refuses_evicted_current_context(paired_candidates):
    _, codec, candidates = paired_candidates
    value = next(row["exported"] for row in candidates if row["id"] == "counter-0")
    episode, _ = adapt_semantic_v7(
        value, codec=codec, max_positions=128, max_history=1, allow_local=True
    )
    with pytest.raises(ValueError, match="context|definition|input"):
        validate_native_visibility_v7(episode)


def test_prepared_shards_enter_actual_training_gate_with_frozen_families(
    paired_candidates, tmp_path
):
    producer, codec, candidates = paired_candidates
    producer_revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=producer, text=True
    ).strip()
    consumer_revision = (
        "b" * 40
    )  # Unit-test receipt pin; production CLI requires a clean real checkout.
    verified = []

    def source_verifier(document):
        candidate_path = tmp_path / "source-input.json"
        candidate_path.write_text(json.dumps(document))
        result = subprocess.run(
            ["node", str(producer / "tools/verify-algorithm-admission.mjs"), str(candidate_path)],
            text=True,
            capture_output=True,
            check=True,
            timeout=120,
        )
        return json.loads(result.stdout)

    for row in candidates:
        if row["id"] not in {"compound-pure", "counter-0", "rejection-finally"}:
            continue
        document = copy.deepcopy(row["exported"])
        document["sidecar"]["nativeAdmission"] = native_admission_v7(
            document,
            codec=codec,
            consumer_revision=consumer_revision,
            max_positions=1024,
            max_history=1024,
        )
        document["sidecar"]["sourceAdmission"] = source_verifier(document)
        document["sidecar"]["admission"] = "admitted"
        document["sidecar"]["nativeAdmissionStatus"] = "admitted"
        verified.append(document)
    destination = tmp_path / "prepared"
    manifest, report = prepare_semantic_corpus_v7(
        verified,
        destination,
        codec=codec,
        producer_revision=producer_revision,
        consumer_revision=consumer_revision,
        source_verifier=source_verifier,
        max_positions=1024,
        max_history=1024,
    )
    assert report["counts"]["admitted"] == 3 and not report["refusals"]
    assert manifest["complete"]
    assert report["profiles"].get("algorithm", 0) > 0 and report["profiles"].get("detailed", 0) > 0
    result = validate_prepared_semantic_v7(
        [destination],
        codec=codec,
        producer_revision=producer_revision,
        consumer_revision=consumer_revision,
        max_history=1024,
        max_positions=1024,
    )
    assert sum(result["episodes"].values()) == 3
    from megatoken.data.semantic_family_v7 import verify_split_identity_v7

    compound_pure_source = verify_split_identity_v7(
        next(row["exported"] for row in candidates if row["id"] == "compound-pure")
    )["sourceId"]
    with PreparedCorpus([destination], split=None) as corpus:
        episode = next(
            item
            for item in corpus.iter_episodes()
            if json.loads(item.supervision.annotations_json)["semantic_execution"]["sourceId"]
            == compound_pure_source
        )
    policy = {
        "schema": "semantic-training-policy/7",
        "producerRevision": producer_revision,
        "consumerRevision": consumer_revision,
        "registryDigest": result["registryDigest"],
    }
    validate_semantic_episode_v7(episode, codec=codec, policy=policy)

    # Exercise the same public training entry used by the optimizer loop, with
    # the production Ling codec and the exact geometry recorded at admission.
    from megatoken.data.media_assets import TensorAssetStore
    from megatoken.data.packing import WindowConfig, iter_windows
    from megatoken.loading import build_model
    from megatoken.model import ModelConfig
    from megatoken.training.config import TrainConfig
    from megatoken.training.engine import batch_counters, prepare_batches
    from megatoken.training.losses import LossLedger
    from megatoken.training.meridians import forward_meridian_batch
    from megatoken.training.stream import EpisodeWindow
    from megatoken.training.transducer import TransducerTrainer

    torch.set_num_threads(1)
    torch.manual_seed(71)
    geometry = json.loads(episode.supervision.annotations_json)["semantic_geometry"]
    tokenizer = Path(
        os.environ.get(
            "MEGATOKEN_TOKENIZER_JSON",
            Path(__file__).resolve().parents[2] / "Ling-3.0-tiny-tokenizer.json",
        )
    )
    model_config = ModelConfig.tiny(
        max_capacity=geometry["maxPositions"],
        max_history_chunks=geometry["maxHistory"],
        vocab_size=codec.vocab_size,
    )
    model = build_model(model_config, codec, device="cpu", precision="fp32").eval()
    train_config = TrainConfig(
        purpose="diagnostic",
        phase="P1",
        max_steps=1,
        model="tiny",
        model_overrides={
            "max_capacity": geometry["maxPositions"],
            "max_history_chunks": geometry["maxHistory"],
            "vocab_size": codec.vocab_size,
        },
        codec=str(tokenizer),
        device="cpu",
        precision="fp32",
        semantic_v7_policy=policy,
        max_record_bytes=max(4096, *(len(record.payload) for record in episode.records)),
        generated_capacity=geometry["maxPositions"],
        max_history_tokens=geometry["maxHistory"],
        effort=1,
        gradient_checkpointing=False,
        compile=False,
    )
    windows = list(
        iter_windows(
            episode,
            codec,
            WindowConfig(max_window_tokens=8192, max_decisions=8, max_records=128),
        )
    )
    target_window = next(window for window in windows if window.P > 0)
    observed = [record for record in target_window.records if not record.record.policy_mask]
    assert observed and all(record.P == 0 for record in observed)
    item = EpisodeWindow(episode, target_window, TensorAssetStore("text-only"))
    (batch,) = prepare_batches(model, [item], train_config)
    assert batch.counts["lexical"] > 0
    assert batch_counters([batch]).P > 0
    ledger = LossLedger(batch.counts, batch.counts, train_config.loss_weights)
    with torch.no_grad():
        forward_meridian_batch(
            model,
            batch,
            ledger,
            train_config,
            torch.device("cpu"),
            model.core.model_version,
            transducer=TransducerTrainer(model.transducer),
            differentiate=False,
        )
    ledger.finish()
    assert math.isfinite(float(ledger.sums["lexical"]))
    assert float(ledger.sums["lexical"]) >= 0

    # A training-policy omission must stop at the entry gate, before batch
    # construction can hand any work to the model's forward path.
    import megatoken.training.engine as engine

    missing_policy = replace(train_config, semantic_v7_policy=None)

    def unexpected_batch_forward(*_args, **_kwargs):
        pytest.fail("semantic training reached batch construction without an explicit policy")

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(engine, "prepare_meridian_batch", unexpected_batch_forward)
        with pytest.raises(ValueError, match="explicit exact producer"):
            prepare_batches(model, [item], missing_policy)

    records = list(episode.records)
    target = next(index for index, record in enumerate(records) if record.policy_mask)
    records[target] = replace(records[target], payload=records[target].payload[:-1] + b" ")
    changed = replace(episode, records=tuple(records))
    with pytest.raises(ValueError, match="labels, model controls"):
        validate_semantic_episode_v7(changed, codec=codec, policy=policy)
    with pytest.raises(ValueError, match="split"):
        validate_semantic_episode_v7(
            episode,
            codec=codec,
            policy=policy,
            expected_split="test" if episode.lineage.frozen_split != "test" else "train",
        )
    annotation = json.loads(episode.supervision.annotations_json)
    wrong_split = "test" if episode.lineage.frozen_split != "test" else "train"
    annotation["semantic_execution"]["split"] = wrong_split
    moved = replace(
        episode,
        lineage=replace(episode.lineage, frozen_split=wrong_split),
        supervision=replace(episode.supervision, annotations_json=json.dumps(annotation)),
    )
    with pytest.raises(ValueError, match="split/source membership"):
        validate_semantic_episode_v7(moved, codec=codec, policy=policy)
    for section, key, value in [
        ("sourceAdmission", "producerRevision", "f" * 40),
        ("sourceAdmission", "optimizerSteps", 1),
        ("nativeAdmission", "consumerRevision", "e" * 40),
        ("nativeAdmission", "contractDigest", "f" * 64),
        ("nativeAdmission", "tokenizerSha256", "f" * 64),
        ("nativeAdmission", "componentGate", "invented"),
        ("nativeAdmission", "traceFormat", "semantic-trace/6"),
        ("nativeAdmission", "optimizerSteps", 1),
    ]:
        annotation = json.loads(episode.supervision.annotations_json)
        annotation["semantic_execution"][section][key] = value
        changed = replace(
            episode,
            supervision=replace(episode.supervision, annotations_json=json.dumps(annotation)),
        )
        with pytest.raises(ValueError):
            validate_semantic_episode_v7(changed, codec=codec, policy=policy)


def test_changed_future_context_is_not_admitted_even_after_public_hash_refresh(paired_candidates):
    _, codec, candidates = paired_candidates
    for representation in ("algorithm", "detailed"):
        document = copy.deepcopy(
            next(
                row["exported"]
                for row in candidates
                if row["exported"]["public"]["representation"] == representation
            )
        )
        document["contexts"][0]["visibleThrough"] += 100000
        document["sidecar"]["contextDigest"] = stable_algorithm_hash(document["contexts"])
        with pytest.raises(ValueError):
            native_admission_v7(document, codec=codec, consumer_revision="a" * 40)
