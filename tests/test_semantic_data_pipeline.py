"""Production tokenizer, physical context, preparation and report gates without optimization."""

import copy
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from megatoken.data.host_operations import canonical_json
from megatoken.data.semantic_admission import native_admission, validate_native_visibility
from megatoken.data.semantic_contract import CONTRACT
from megatoken.data.semantic_harness_v4 import adapt_selected_harness_v4
from megatoken.generation.codec import LING_TOKENIZER_SHA256, ByteCodec
from megatoken.training.config import TrainConfig
from megatoken.training.semantic_evaluation import (
    HARD_GATES,
    evaluate_cohort,
    freeze_cohort,
    verify_cohort,
)


def canonical_document(index: int):
    # The old contract's negative/archival fixture remains immutable, but
    # runnable v4 test inputs must declare their actual selected representation.
    document = copy.deepcopy(CONTRACT["fixtures"]["envelope"]["valid"][index])
    document["sidecar"]["representation"] = "compact"
    return document


@pytest.fixture
def production_codec():
    path = Path(os.environ.get("MEGATOKEN_TOKENIZER_JSON", ".validation/tokenizer.json"))
    if not path.is_file():
        pytest.skip("Dedicated semantic CI downloads the exact production tokenizer")
    return ByteCodec.from_tokenizer_json(path, expected_sha256=LING_TOKENIZER_SHA256)


def test_production_tokenizer_fragmentation_source_visibility_and_loss(production_codec):
    document = canonical_document(0)
    program, invocation = document["public"]["timeline"][:2]
    program["source"] += "\n/* " + "long source context must remain visible. " * 300 + "*/"
    digest = hashlib.sha256(program["source"].encode()).hexdigest()
    program["digest"] = invocation["sourceDigest"] = digest
    receipt = native_admission(
        document,
        codec=production_codec,
        consumer_revision="a" * 40,
        max_positions=256,
        max_history=256,
    )
    assert (
        receipt["physicalRecords"] > receipt["logicalRecords"] + receipt["contextTransportRecords"]
    )
    assert receipt["generatedTokens"] > 0 and receipt["observedTokens"] > receipt["generatedTokens"]
    assert receipt["lossMaskValidated"] and receipt["visibilityValidated"]
    episode, _ = adapt_selected_harness_v4(
        document, codec=production_codec, allow_local=True, max_positions=256, max_history=256
    )
    assert validate_native_visibility(episode) == 1
    # A falsely small cross-track window would hide earlier source fragments.
    records = tuple(
        replace(
            record,
            payload=record.payload.replace(b'"cross_events":256', b'"cross_events":1'),
            cached_ids=None,
            codec_fingerprint=None,
            source_span=None,
        )
        if record.logical_id.startswith("semantic:context:")
        else record
        for record in episode.records
    )
    with pytest.raises(ValueError, match="source context"):
        validate_native_visibility(replace(episode, records=records))


def test_production_context_eviction_and_checked_reread(production_codec):
    document = json.loads(
        (Path(__file__).parent / "fixtures/semantic-v4/checkpoint-recovery.json").read_text()
    )
    document["sidecar"]["representation"] = "compact"
    receipt = native_admission(document, codec=production_codec, consumer_revision="a" * 40)
    assert receipt["contextTransportRecords"] == 3
    assert receipt["fragmentationValidated"] and receipt["visibilityValidated"]
    timeline = document["public"]["timeline"]
    reread = next(item for item in timeline if item["kind"] == "source-read")
    reread["source"] += " changed"
    with pytest.raises(ValueError, match="reread"):
        native_admission(document, codec=production_codec, consumer_revision="a" * 40)


def test_source_visibility_alone_does_not_admit_evicted_captured_state(production_codec):
    document = json.loads(
        (Path(__file__).parent / "fixtures/semantic-v4/retained-events.json").read_text()
    )
    document["sidecar"]["representation"] = "compact"
    episode, _ = adapt_selected_harness_v4(
        document, codec=production_codec, allow_local=True, max_positions=1024, max_history=1024
    )
    validate_native_visibility(episode)
    source_ids = []
    for parts in episode.iter_logical_records():
        if not parts[0].policy_mask:
            payload = json.loads(b"".join(part.payload for part in parts))
            if payload.get("kind") == "program":
                source_ids.extend(part.record_id for part in parts)
    assert 0 < len(source_ids) <= 8
    records = []
    for record in episode.records:
        if record.logical_id.startswith("semantic:context:"):
            payload = json.loads(record.payload)
            payload["context_window"] = {
                "local_events": 1,
                "cross_events": 1,
                "pinned_events": source_ids,
            }
            record = replace(record, payload=json.dumps(payload).encode(), source_span=None)
        records.append(record)
    with pytest.raises(ValueError, match="earlier committed realm state"):
        validate_native_visibility(replace(episode, records=tuple(records)))


def test_verified_public_state_refresh_supports_real_bounded_history(production_codec):
    document = json.loads(
        (Path(__file__).parent / "fixtures/semantic-v4/retained-state-tail.json").read_text()
    )
    receipt = native_admission(
        document, codec=production_codec, consumer_revision="a" * 40, max_history=128
    )
    assert receipt["physicalRecords"] > 128
    assert receipt["visibilityValidated"] and receipt["lossMaskValidated"]
    states = [
        item
        for item in document["public"]["timeline"]
        if item["kind"] == "model-context" and "selectedState" in item
    ]
    assert len(states) >= 8
    episode, _ = adapt_selected_harness_v4(
        document, codec=production_codec, allow_local=True, max_positions=1024, max_history=128
    )
    snapshots = [
        parts
        for parts in episode.iter_logical_records()
        if not parts[0].policy_mask
        and json.loads(b"".join(part.payload for part in parts)).get("selectedState") is not None
    ]
    assert len(snapshots) == len(states)
    assert all(not record.policy_mask for parts in snapshots for record in parts)
    # Deleting state transport while retaining exact source rereads cannot
    # pretend the evicted captured-cell history remains available.
    removed = {part.record_id for parts in snapshots for part in parts}
    without_states = replace(
        episode,
        records=tuple(record for record in episode.records if record.record_id not in removed),
        meridians=tuple(
            group for group in episode.meridians if group.events[0].record_id not in removed
        ),
        dependencies=tuple(
            edge
            for edge in episode.dependencies
            if edge.writer_id not in removed and edge.reader_id not in removed
        ),
    )
    with pytest.raises(ValueError, match="earlier committed realm state"):
        validate_native_visibility(without_states)
    forged = copy.deepcopy(document)
    next(item for item in forged["public"]["timeline"] if item["kind"] == "model-context")[
        "visibleThrough"
    ] += 1
    with pytest.raises(ValueError, match="cutoff"):
        native_admission(
            forged, codec=production_codec, consumer_revision="a" * 40, max_history=128
        )


def test_production_configs_preserve_seeds_and_validate_without_training():
    for seed in (17, 29, 43):
        config = TrainConfig.load(
            Path(__file__).parents[1] / f"configs/semantic/train-seed-{seed}.json"
        )
        assert config.seed == seed
        assert config.phase == "SE"
        assert config.checkpoint_every > 0
        assert config.allow_unverified_data is False
        assert config.generated_capacity == 1024


def cohort_fixture():
    request = {"source": "function run(value){return value;}", "args": [7]}
    outcome = {
        "completion": {"kind": "return", "value": ["num", 7]},
        "state": {},
        "effects": [],
        "callbacks": [],
        "cleanup": [],
    }

    def digest(value):
        return hashlib.sha256(canonical_json(value).encode()).hexdigest()

    case = {
        "id": "identity",
        "family": "heldout-identity",
        "split": "test",
        "request": request,
        "expected": outcome,
        "referenceReceipt": {
            "status": "agreed",
            "requestDigest": digest(request),
            "outcomeDigest": digest(outcome),
        },
    }
    return freeze_cohort(
        [case],
        name="effectful-tracks-events",
        thresholds={"wholeTaskSuccess": 1.0, "traceValid": 1.0},
    )


def test_frozen_cohort_cli_preserves_source_receipt_bytes(tmp_path):
    cohort = cohort_fixture()
    declaration = tmp_path / "declaration.json"
    output = tmp_path / "cohort.json"
    declaration.write_text(
        json.dumps(
            {
                "cases": cohort["cases"],
                "name": cohort["name"],
                "thresholds": cohort["thresholds"],
            }
        )
    )
    root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts/evaluate_semantic.py"),
            "freeze",
            "--declaration",
            str(declaration),
            "--output",
            str(output),
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    restored = json.loads(output.read_text())
    verify_cohort(restored)
    assert restored == cohort
    assert list(restored["cases"][0]["request"]) == ["source", "args"]


def test_evaluation_keeps_expected_outcome_private_and_controlled_success_unlearned():
    cohort = cohort_fixture()
    seen = []
    evidence = {
        "schema": "semantic-evaluation-result/4",
        "origin": "controlled",
        "representation": "compact",
        "outcome": cohort["cases"][0]["expected"],
        "episode": canonical_document(0),
        "firstSemanticInconsistency": None,
        "hardGates": dict.fromkeys(HARD_GATES, 0),
        "costs": {
            "computeSeconds": 0.01,
            "peakMemoryBytes": 1000,
            "dispatches": 0,
            "queueDepth": 0,
            "leakedResources": 0,
        },
    }

    def runner(request):
        seen.append(request)
        return copy.deepcopy(evidence)

    report = evaluate_cohort(
        cohort, runner, codec=ByteCodec.byte_fallback(), seed=17, origin="controlled"
    )
    assert report["summary"]["wholeTaskSuccess"] == 1
    assert report["learnedSuccess"] is None and report["acceptance"] is False
    assert "expected" not in json.dumps(seen) and "referenceReceipt" not in json.dumps(seen)
    evidence["firstSemanticInconsistency"] = {"record": 3, "reason": "wrong computation"}
    failed = evaluate_cohort(
        cohort, runner, codec=ByteCodec.byte_fallback(), seed=17, origin="controlled"
    )
    assert failed["summary"]["traceValid"] == 0
    assert failed["summary"]["wholeTaskSuccess"] == 0
    edited = copy.deepcopy(cohort)
    edited["thresholds"]["traceValid"] = 0.5
    with pytest.raises(ValueError, match="changed"):
        evaluate_cohort(
            edited, runner, codec=ByteCodec.byte_fallback(), seed=17, origin="controlled"
        )


def test_exposure_budget_restart_preserves_whole_task_and_measured_compute_counts():
    from megatoken.training.semantic_budget import SemanticExperimentConfig, SemanticExposure

    config = SemanticExperimentConfig(
        schema="semantic-training-experiment/4",
        regime="fixed-examples",
        limit=2,
        representation="compact",
        cohort="frozen-cohort.json",
        engineering_receipt="engineering.json",
    )
    exposure = SemanticExposure()
    exposure.consume(examples=0, generatedTokens=8, observedTokens=100, computeSeconds=0.25)
    assert not exposure.finished(config)
    restored = SemanticExposure.restore(exposure.checkpoint())
    restored.consume(examples=2, generatedTokens=16, observedTokens=30, computeSeconds=0.5)
    assert restored.finished(config)
    receipt = restored.receipt(
        config, seed=17, checkpoint_identity="b" * 64, cohort_digest="c" * 64
    )
    assert receipt["examples"] == 2 and receipt["generatedTokens"] == 24
    assert receipt["computeSeconds"] == 0.75 and receipt["overshoot"] == 0
    bad = restored.checkpoint()
    del bad["generatedTokens"]
    with pytest.raises(ValueError, match="restart"):
        SemanticExposure.restore(bad)


def test_training_experiment_preflight_blocks_missing_engineering_gate(tmp_path):
    from megatoken.training.semantic_budget import SemanticExperimentConfig

    cohort = tmp_path / "cohort.json"
    cohort.write_text(json.dumps(cohort_fixture()))
    gate = tmp_path / "engineering.json"
    gate.write_text(json.dumps({"schema": "semantic-engineering-completion/4", "complete": False}))
    config = SemanticExperimentConfig(
        schema="semantic-training-experiment/4",
        regime="fixed-tokens",
        limit=100,
        representation="compact",
        cohort=str(cohort),
        engineering_receipt=str(gate),
    )
    with pytest.raises(ValueError, match="precede"):
        config.preflight()
    with pytest.raises(ValueError, match="phase, frozen cohort"):
        config.preflight(phase="SC")


def test_canonical_training_refuses_obsolete_labels_and_keeps_public_state(production_codec):
    from megatoken.training.semantic_budget import (
        validate_semantic_episode,
        validate_semantic_geometry,
    )

    document = canonical_document(1)
    episode, _ = adapt_selected_harness_v4(
        document, codec=production_codec, max_positions=1024,
        max_history=128, allow_local=True,
    )
    annotation = json.loads(episode.supervision.annotations_json)
    annotation["semantic_execution"].update(
        representation="compact", continuation="teacher_conditioned", admission="admitted"
    )
    episode = replace(
        episode,
        records=tuple(
            replace(row, provenance="teacher") if row.policy_mask else row
            for row in episode.records
        ),
        supervision=replace(
            episode.supervision,
            outcome="success",
            verification="local_assertions",
            annotations_json=json.dumps(annotation),
        ),
    )
    validate_semantic_episode(episode, phase="SC", representation="compact")
    validate_native_visibility(episode)
    with pytest.raises(ValueError, match="configured model"):
        validate_semantic_geometry(episode, max_history=64, max_positions=1024)
    with pytest.raises(ValueError, match="certified training representation"):
        validate_semantic_episode(episode, phase="SC", representation="explicit")
    with pytest.raises(ValueError, match="certified training representation"):
        validate_semantic_episode(episode, phase="SO", representation="outcome-only")

    forged = copy.deepcopy(document)
    forged["sidecar"]["representation"] = "explicit"
    with pytest.raises(ValueError, match="canonical|contract"):
        adapt_selected_harness_v4(forged, allow_local=True)


def test_rejected_interactive_cost_keeps_new_bytes_and_deduplicates_committed_native_failure():
    from megatoken.training.semantic_evaluation import _interactive_usage

    accepted = 'return {"value":["num",7]}'
    rejected = b"compute invalid"
    evidence = {
        "episode": {"public": {"timeline": [{"kind": "semantic", "line": accepted}]}},
        "rejections": [
            {"record": accepted, "phase": "native-commit-or-delivery"},
            {
                "record": rejected.decode(),
                "phase": "semantic-validation",
                "selectedEventId": "failed1",
            },
        ],
        "failure": {"evidence": {"payloadHex": rejected.hex(), "selectedEventId": "failed1"}},
    }
    generated, _, failures = _interactive_usage(evidence, ByteCodec.byte_fallback())
    assert generated == len(accepted.encode()) and failures == len(rejected)
    # Identical spelling does not identify a later, distinct failed prediction.
    evidence["rejections"].append(
        {"record": accepted, "phase": "semantic-validation", "selectedEventId": "failed2"}
    )
    assert _interactive_usage(evidence, ByteCodec.byte_fallback())[2] == len(rejected) + len(
        accepted.encode()
    )


def test_original_decoder_fragments_are_counted_without_bpe_reencoding(production_codec):
    from megatoken.training.semantic_evaluation import _original_emission_tokens

    pieces = [b"h", b"ello"]
    fragments = [
        {
            "selectedEventId": "actual:" + str(i),
            "payloadHex": raw.hex(),
            "emittedTokenIds": production_codec.encode(raw),
        }
        for i, raw in enumerate(pieces)
    ]
    evidence = {
        "selections": [
            {
                "record": "hello",
                "selection": {
                    "modelSelected": True,
                    "codecFingerprint": production_codec.fingerprint,
                    "fragments": fragments,
                },
            }
        ],
        "rejections": [
            {
                "record": "hello",
                "selection": {
                    "modelSelected": True,
                    "codecFingerprint": production_codec.fingerprint,
                    "fragments": fragments,
                },
            }
        ],
    }
    canonical = len(production_codec.encode(b"hello"))
    original = sum(len(fragment["emittedTokenIds"]) for fragment in fragments)
    assert original > canonical
    assert (
        _original_emission_tokens(evidence, production_codec, canonical_tokens=canonical)
        == original
    )
