"""Admission and train-membership controls for the bounded V8 fitter."""

import copy
import importlib.util
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_semantic_v8_training_policy import prepared_v8_fixture

_FIT_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "fit_semantic_v8.py"
_SPEC = importlib.util.spec_from_file_location("fit_semantic_v8", _FIT_SCRIPT)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
verified_train_example = _MODULE.verified_train_example
capture_boundaries = _MODULE.capture_boundaries


@pytest.mark.parametrize("mutation", [None, "state", "replay-count", "host-dispatch", "next-token"])
def test_accepted_recovery_collector_requires_exact_state_and_original_next_tokens(
    tmp_path, monkeypatch, mutation
):
    from tests.test_prediction_prefix_v8 import examples
    from tests.test_semantic_v8_serving_restore import adapter_for, commit, select, target

    example = examples("semantic-regions-v8.json.gz.b64")[0]
    adapter, _ = adapter_for([target(example, 0), target(example, 1)])
    commit(adapter, example, 0, select(adapter, example, 0))
    packet = adapter.checkpoint()
    next_selection = select(adapter, example, 1)
    outcome = {
        "acknowledgedCheckpoint": packet,
        "acknowledged": 1,
        "completed": False,
        "nextSelectionRequest": {"session": next_selection["selection"]["session"]},
        "nextSelectionResponse": {"ok": True, "result": next_selection},
    }
    responses = [
        {
            "id": "restore",
            "ok": True,
            "result": {"restored": True, "hostDispatches": 0, "reexecutedSelections": 1},
        },
        {"id": "checkpoint", "ok": True, "result": copy.deepcopy(packet)},
        {"id": "next", "ok": True, "result": copy.deepcopy(next_selection)},
    ]
    if mutation == "state":
        responses[1]["result"]["randomState"]["torch"] = "altered"
    elif mutation == "replay-count":
        responses[0]["result"]["reexecutedSelections"] = 0
    elif mutation == "host-dispatch":
        responses[0]["result"]["hostDispatches"] = 1
    elif mutation == "next-token":
        responses[2]["result"]["selection"]["payloadHex"] = "00"

    def injected_service(command, **kwargs):
        # The collector's process boundary is explicitly injected here. Separate
        # command-level evidence must use the actual service and fitted tensors.
        assert Path(command[1]).name == "serve_semantic_v8.py"
        requests = [json.loads(line) for line in kwargs["input"].splitlines()]
        assert requests[0]["checkpoint"] == packet
        assert [request["op"] for request in requests] == ["restore", "checkpoint", "select"]
        return SimpleNamespace(
            returncode=0, stdout="".join(json.dumps(row) + "\n" for row in responses), stderr=""
        )

    monkeypatch.setattr(subprocess, "run", injected_service)
    result = _MODULE.qualify_accepted_restoration(tmp_path / "served", outcome, tmp_path / "proof")
    assert result["status"] == ("passed" if mutation is None else "failed")
    assert result["sourceCorrespondenceClaim"] is False
    assert result["generalizationClaim"] is False
    assert (tmp_path / "proof" / "service-stdout.jsonl").exists()
    if mutation is None:
        assert result["acknowledgedCheckpointExact"] and result["nextSelectionExact"]
        assert result["nextSelectionAcknowledged"] is False


def test_selected_prefix_identity_counts_only_scored_targets_and_retains_prepared_chunking(
    tmp_path,
):
    from megatoken.data.storage import PreparedCorpus
    from megatoken.semantic_lineage_v8 import validate_training_lineage_v8
    from megatoken.training.config import TrainConfig

    _, codec, policy, prepared = prepared_v8_fixture(tmp_path)
    with PreparedCorpus([prepared], split="train", stream="semantic_execution") as corpus:
        episode = next(corpus.iter_episodes())
        expected_chunking = corpus.chunking
    config = TrainConfig(
        purpose="diagnostic",
        phase="P1",
        model="tiny",
        codec="injected-tokenizer",
        semantic_v8_policy=policy,
        max_steps=1,
    )
    chunking, lineage, selection = _MODULE.diagnostic_training_identity(
        [prepared], episode, codec=codec, config=config, records=1
    )
    targets = _MODULE.complete_targets(episode)
    assert selection["generatedTrainingTokensPerStep"] == len(codec.encode(targets[0][0].encode()))
    assert selection["selectedLogicalRecords"] == 1
    assert selection["admittedEpisodeLogicalRecords"] == len(targets) > 1
    assert selection["admittedEpisodeGeneratedTokens"] > selection["generatedTrainingTokensPerStep"]
    assert chunking.identity(codec) == expected_chunking
    validate_training_lineage_v8(lineage, chunking_fingerprint=expected_chunking["fingerprint"])
    _, second_lineage, second_selection = _MODULE.diagnostic_training_identity(
        [prepared], episode, codec=codec, config=config, records=2
    )
    assert second_lineage["streamDigest"] != lineage["streamDigest"]
    assert second_selection["selectedTargetSha256"][:1] == selection["selectedTargetSha256"]
    assert selection["ordinaryTrainerResumeClaim"] is False


def test_prefix_success_requires_original_acknowledged_records_and_keeps_completion_separate():
    targets = ["detail a"]
    result = {
        "status": "accepted-nonterminal",
        "completed": False,
        "acknowledged": 1,
        "records": targets,
    }
    assert _MODULE.exact_selected_prefix(result, targets)
    assert not result["completed"]
    assert not _MODULE.exact_selected_prefix({**result, "acknowledged": 0}, targets)
    assert not _MODULE.exact_selected_prefix({**result, "status": "failed"}, targets)
    assert not _MODULE.exact_selected_prefix({**result, "records": ["repaired"]}, targets)


def test_fitted_checkpoint_and_served_bundle_retain_selected_training_identity(
    tmp_path, monkeypatch
):
    import torch

    import megatoken.training.semantic_v8 as semantic_v8
    from megatoken.data.storage import PreparedCorpus
    from megatoken.loading import build_model
    from megatoken.model import ModelConfig
    from megatoken.semantic_lineage_v8 import checkpoint_lineage_v8
    from megatoken.training.checkpoint import load_checkpoint
    from megatoken.training.config import TrainConfig
    from megatoken.training.optimizer import MasterAdamW
    from megatoken.training.transducer import TransducerTrainer

    torch.set_num_threads(1)
    _, codec, policy, prepared = prepared_v8_fixture(tmp_path)
    with PreparedCorpus([prepared], split="train", stream="semantic_execution") as corpus:
        episode = next(corpus.iter_episodes())
    injected_tokenizer = tmp_path / "injected-tokenizer.json"
    injected_tokenizer.write_text("{}")
    config = TrainConfig(
        purpose="diagnostic",
        phase="P1",
        model="tiny",
        codec=str(injected_tokenizer),
        precision="fp32",
        semantic_v8_policy=policy,
        max_steps=1,
    )
    chunking, lineage, selection = _MODULE.diagnostic_training_identity(
        [prepared], episode, codec=codec, config=config, records=1
    )
    model = build_model(ModelConfig.tiny(vocab_size=codec.vocab_size), codec).eval()
    transducer = TransducerTrainer(
        model.transducer, bank_trainable=False, teacher_weight=0.0, navigation_weight=0.0
    )
    optimizer = MasterAdamW(model, lr=0.02, weight_decay=0)
    # This structural export regression explicitly injects tokenizer verification;
    # actual pinned-tokenizer qualification remains a separate command-level run.
    monkeypatch.setattr(semantic_v8, "load_semantic_v8_codec", lambda path, *, vocab_size: codec)
    try:
        loaded, manifest = _MODULE.save_bundle(
            model,
            optimizer,
            transducer,
            config,
            tmp_path,
            "fitted",
            steps=3,
            chunking=chunking,
            lineage=lineage,
            selection=selection,
        )
    finally:
        optimizer.close()
    payload, metadata = load_checkpoint(tmp_path / "checkpoints" / "fitted")
    assert checkpoint_lineage_v8(payload, metadata) == lineage
    assert payload["diagnostic_training_selection"] == selection
    assert payload["diagnostic_optimizer_steps"] == 3
    assert payload["global_step"] == 0 and metadata["ordinaryTrainerResumeClaim"] is False
    assert loaded.chunking.identity(codec) == chunking.identity(codec)
    assert loaded.semantic_v8_lineage["training"] == lineage
    assert manifest["semantic_v8_lineage_digest"]
    retained = json.loads((tmp_path / "fitted" / "semantic-v8-lineage.json").read_text())
    assert retained["training"]["streamDigest"] == lineage["streamDigest"]


def test_fitter_reads_the_verified_prepared_train_episode(tmp_path, monkeypatch):
    document, codec, policy, prepared = prepared_v8_fixture(tmp_path)
    monkeypatch.setattr(
        subprocess,
        "check_output",
        lambda command, **kwargs: "b" * 40 if command[1] == "rev-parse" else "",
    )

    episode, artifact, revision, supply, semantic = verified_train_example(
        [prepared], codec=codec, policy=policy
    )

    assert revision == policy["consumerRevision"]
    assert supply["episodes"] == {"train": 1, "validation": 0, "test": 0}
    assert episode.lineage.frozen_split == semantic["split"] == "train"
    assert semantic["sourceId"] == "source-1"
    assert artifact == {"public": document["public"], "contexts": document["contexts"]}
    actual_targets = {}
    for row in episode.records:
        if row.policy_mask:
            actual_targets.setdefault(row.logical_id, []).append((row.part_index, row.payload))
    assert [
        b"".join(payload for _, payload in sorted(parts)) for parts in actual_targets.values()
    ] == [
        row["fields"]["line"].encode()
        for row in document["public"]["timeline"]
        if row["mode"] == "generated"
    ]


def test_fitter_refuses_unknown_source_without_falling_back(tmp_path, monkeypatch):
    _, codec, policy, prepared = prepared_v8_fixture(tmp_path)
    monkeypatch.setattr(
        subprocess,
        "check_output",
        lambda command, **kwargs: "b" * 40 if command[1] == "rev-parse" else "",
    )

    with pytest.raises(ValueError, match="No verified V8 training row"):
        verified_train_example([prepared], codec=codec, policy=policy, source_id="not-in-registry")


def test_fitter_requires_policy_to_pin_current_consumer(tmp_path, monkeypatch):
    _, codec, policy, prepared = prepared_v8_fixture(tmp_path)
    monkeypatch.setattr(
        subprocess,
        "check_output",
        lambda command, **kwargs: "c" * 40 if command[1] == "rev-parse" else "",
    )

    with pytest.raises(ValueError, match="does not pin this exact consumer"):
        verified_train_example([prepared], codec=codec, policy=policy)


def test_fitter_captures_actual_prepared_episode_through_grouped_trainer(tmp_path, monkeypatch):
    import torch

    _, codec, policy, prepared = prepared_v8_fixture(tmp_path)
    monkeypatch.setattr(
        subprocess,
        "check_output",
        lambda command, **kwargs: "b" * 40 if command[1] == "rev-parse" else "",
    )
    from megatoken.data.storage import PreparedCorpus
    from megatoken.loading import build_model
    from megatoken.model import ModelConfig
    from megatoken.training.config import TrainConfig
    from tests.test_semantic_v8_continuation_parity import assert_same_state

    with PreparedCorpus([prepared], split="train", stream="semantic_execution") as corpus:
        episode = next(corpus.iter_episodes())
    config = TrainConfig(
        purpose="diagnostic",
        phase="P1",
        model="tiny",
        datasets=[str(prepared)],
        output=str(tmp_path / "capture"),
        codec="injected-tokenizer",
        device="cpu",
        semantic_v8_policy=policy,
        max_steps=1,
        use_stage_mixture=False,
        use_effort_mixture=False,
    )

    model = build_model(ModelConfig.tiny(vocab_size=codec.vocab_size), codec).eval()
    windows_seen = []
    prepare_batches = _MODULE.prepare_batches

    def count_windows(model, windows, config):
        windows_seen.append(len(windows))
        return prepare_batches(model, windows, config)

    monkeypatch.setattr(_MODULE, "prepare_batches", count_windows)
    before_random = torch.get_rng_state()
    boundaries, _ = capture_boundaries(model, episode, config, tmp_path / "capture")

    assert len(boundaries) == sum(record.policy_mask for record in episode.records)
    torch.set_rng_state(before_random)
    first, _ = capture_boundaries(model, episode, config, tmp_path / "capture", max_boundaries=1)
    assert len(first) == 1 and windows_seen[1] < windows_seen[0]
    assert_same_state(boundaries[0][0], first[0][0], {})
    assert boundaries[0][1]["control_payloads"] == first[0][1]["control_payloads"]
