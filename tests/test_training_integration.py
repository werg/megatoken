"""Restart and phase semantics checked against real model/optimizer trajectories."""

from dataclasses import replace

import pytest
import torch
from training_fixture import tiny_train_config

from megatoken.training.checkpoint import load_checkpoint
from megatoken.training.engine import train
from megatoken.training.schedule import PHASES


@pytest.fixture
def training_config(tmp_path):
    torch.set_num_threads(1)
    return tiny_train_config(tmp_path, gradient_checkpointing=True)


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
        # Each stage is metered in its own unit: B0 in ingested tokens, P1 in new tokens.
        meter = PHASES[phase].meter
        assert meter == {"B0": "I", "P1": "D"}[phase]
        assert result["phase_consumed"] == result["counters"][meter]
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


def test_resume_compatibility_covers_every_setting_that_changes_the_trajectory():
    from dataclasses import asdict

    from megatoken.training.config import DecoupledConfig, TrainConfig
    from megatoken.training.engine import _resume_compatible

    base = TrainConfig(phase="B0", decoupled=DecoupledConfig("http://a:1", "run", workers=2))
    saved = asdict(base)
    # Where the coordinator lives (and how patiently it is polled) does not matter ...
    moved = replace(
        base,
        decoupled=replace(
            base.decoupled, coordinator_url="http://b:2", timeout_seconds=9.0, io_workers=1
        ),
    )
    _resume_compatible(asdict(base), moved, False)
    assert saved == asdict(base)  # the comparison does not rewrite the saved configuration
    # ... what the learner computes does.
    for changed in (
        replace(base, decoupled=replace(base.decoupled, local_steps=7)),
        replace(base, decoupled=replace(base.decoupled, workers=3)),
        replace(base, decoupled=None),
        replace(base, accumulate_events=base.accumulate_events + 1),
        replace(base, max_media_events_per_window=base.max_media_events_per_window + 1),
    ):
        with pytest.raises(ValueError, match="Resume configuration changed"):
            _resume_compatible(asdict(base), changed, False)


def test_router_balance_recorders_are_released_when_a_forward_fails(training_config, monkeypatch):
    models = []

    def failing(model, *args, **kwargs):
        models.append(model)
        assert model.core.p_router.balance_recorder is not None
        raise RuntimeError("forward failed")

    monkeypatch.setattr("megatoken.training.engine.forward_meridian_batch", failing)
    with pytest.raises(RuntimeError, match="forward failed"):
        train(training_config)
    (model,) = models
    assert model.core.p_router.balance_recorder is None
    assert model.core.u_router.balance_recorder is None
