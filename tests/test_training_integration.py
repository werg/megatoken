"""Restart and phase semantics checked against real model/optimizer trajectories."""

from dataclasses import replace

import pytest
import torch
from retrieval_fixture import retrieval_train_config

from megatoken.training.checkpoint import load_checkpoint
from megatoken.training.engine import train


@pytest.fixture
def training_config(tmp_path):
    torch.set_num_threads(1)
    return retrieval_train_config(tmp_path, gradient_checkpointing=True)


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
    for phase in ("B0", "P1"):
        config = replace(
            training_config, phase=phase, budget=1, output=training_config.output + "-" + phase
        )
        result = train(config, resume=previous, phase_transition=previous is not None)
        assert result["complete"] and result["global_step"] == step + 1
        # Retrieval-memory training is always metered in grouped events (G).
        assert result["phase_consumed"] == result["counters"]["G"]
        payload, metadata = load_checkpoint(result["checkpoint"])
        assert metadata["pinned"] and metadata["checkpoint_name"].endswith("phase-end")
        steps = [int(state["step"]) for state in payload["optimizer"]["moments"].values()]
        assert max(steps) == step + 1
        previous, step = result["checkpoint"], result["global_step"]
    # A completed phase is an idempotent no-op, with no checkpoint overwrite.
    assert train(config, resume=previous)["checkpoint"] == previous


def test_bf16_checkpointed_training(training_config):
    result = train(replace(training_config, precision="bf16"))
    payload, _ = load_checkpoint(result["checkpoint"])
    assert any(
        t.dtype == torch.bfloat16 for t in payload["model"].values() if isinstance(t, torch.Tensor)
    )
    assert all(t.dtype == torch.float32 for t in payload["optimizer"]["master_weights"].values())
    assert result["counters"]["P"] > 0


def test_resume_rejects_changed_objective(training_config):
    first = train(training_config)
    with pytest.raises(ValueError, match="loss_weights"):
        train(
            replace(
                training_config,
                loss_weights={**training_config.loss_weights, "facet_prediction": 0.2},
            ),
            resume=first["checkpoint"],
        )
