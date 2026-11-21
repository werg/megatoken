"""Admission and train-membership controls for the bounded V8 fitter."""

import importlib.util
import subprocess
from pathlib import Path

import pytest
from test_semantic_v8_training_policy import prepared_v8_fixture

_FIT_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "fit_semantic_v8.py"
_SPEC = importlib.util.spec_from_file_location("fit_semantic_v8", _FIT_SCRIPT)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
verified_train_example = _MODULE.verified_train_example
capture_boundaries = _MODULE.capture_boundaries


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
    boundaries, _ = capture_boundaries(model, episode, config, tmp_path / "capture")

    assert len(boundaries) == sum(record.policy_mask for record in episode.records)
