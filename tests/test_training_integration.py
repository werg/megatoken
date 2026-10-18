"""Restart and phase semantics checked against real model/optimizer trajectories."""

from dataclasses import replace

import pytest
import torch

from megatoken.data import prepare_sources, write_demo_source
from megatoken.generation.codec import ByteCodec
from megatoken.training.checkpoint import load_checkpoint
from megatoken.training.config import TrainConfig
from megatoken.training.engine import train


@pytest.fixture
def training_config(tmp_path):
    torch.set_num_threads(1)
    source = write_demo_source(tmp_path / "source.jsonl", count=32, max_record_bytes=128)
    dataset = prepare_sources([source], tmp_path / "data", ByteCodec.byte_fallback())
    dataset.close()
    return TrainConfig(
        dataset=str(tmp_path / "data"),
        output=str(tmp_path / "run"),
        model="tiny",
        codec="byte",
        device="cpu",
        precision="fp32",
        microbatch_episodes=2,
        max_steps=1,
        accumulate_scored=80,
        accumulate_ingested=1000,
        max_accumulation_windows=2,
        generated_capacity=128,
        max_history_tokens=1024,
        max_record_bytes=512,
        checkpoint_every=0,
        eval_every=0,
        gradient_checkpointing=True,
        use_stage_mixture=False,
        use_effort_mixture=False,
    )


def assert_equal_tree(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_equal_tree(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            assert_equal_tree(a, b)
    else:
        assert left == right


def test_uninterrupted_matches_exact_resume(training_config):
    full = train(replace(training_config, output=training_config.output + "-full", max_steps=2))
    first = train(training_config)
    resumed = train(training_config, resume=first["checkpoint"])
    uninterrupted, _ = load_checkpoint(full["checkpoint"])
    recovered, _ = load_checkpoint(resumed["checkpoint"])
    for key in ("model", "optimizer", "rank_state", "phase_consumed", "counters", "global_step"):
        assert_equal_tree(uninterrupted[key], recovered[key])


def test_all_supervised_stages_preserve_optimizer_and_use_real_meters(training_config):
    previous = None
    step = 0
    for phase in ("B0", "P1", "M1", "M2", "LC", "S", "D"):
        config = replace(
            training_config, phase=phase, budget=1, output=training_config.output + "-" + phase
        )
        result = train(config, resume=previous, phase_transition=previous is not None)
        assert result["complete"] and result["global_step"] == step + 1
        assert (
            result["phase_consumed"]
            == result["counters"][{"B0": "I", "S": "P", "D": "P"}.get(phase, "D")]
        )
        payload, metadata = load_checkpoint(result["checkpoint"])
        assert metadata["pinned"] and metadata["checkpoint_name"].endswith("phase-end")
        steps = [
            int(state["step"]) for state in payload["optimizer"]["optimizer"]["state"].values()
        ]
        assert max(steps) == step + 1
        previous, step = result["checkpoint"], result["global_step"]
    # A completed phase is an idempotent no-op, with no checkpoint overwrite.
    assert train(config, resume=previous)["checkpoint"] == previous


def test_bf16_checkpointed_training_and_evaluation(training_config):
    result = train(replace(training_config, precision="bf16", eval_every=1, eval_episodes=1))
    payload, _ = load_checkpoint(result["checkpoint"])
    assert any(
        t.dtype == torch.bfloat16 for t in payload["model"].values() if isinstance(t, torch.Tensor)
    )
    assert all(t.dtype == torch.float32 for t in payload["optimizer"]["master_weights"].values())
    assert result["counters"]["P"] > 0


def test_resume_rejects_changed_objective(training_config):
    first = train(training_config)
    with pytest.raises(ValueError, match="prior_probability"):
        train(replace(training_config, prior_probability=1), resume=first["checkpoint"])


def test_legacy_action_without_public_tools_uses_raw_targets(training_config):
    from megatoken.data import PreparedDataset, WindowConfig, iter_windows
    from megatoken.loading import build_model, load_model_config
    from megatoken.training.engine import prepare_batches
    from megatoken.training.stream import EpisodeWindow

    codec = ByteCodec.byte_fallback()
    with PreparedDataset(training_config.dataset) as data:
        episode = replace(data[0], tools_json="[]")
    windows = [
        EpisodeWindow(episode, window) for window in iter_windows(episode, codec, WindowConfig())
    ]
    model = build_model(load_model_config("tiny"), codec)
    batches = prepare_batches(model, windows, training_config)
    assert any(
        record.record.action_id
        for batch in batches
        for item in batch.windows
        for record in item.window.records
    )
    assert batches and all(
        not hasattr(target, "spec") for batch in batches for target in batch.targets.values()
    )
