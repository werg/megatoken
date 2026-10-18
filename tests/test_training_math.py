from __future__ import annotations

import copy
import random

import numpy as np
import pytest
import torch

from megatoken.runtime import capture_rng, restore_rng, seed_everything
from megatoken.training.checkpoint import load_checkpoint, save_checkpoint
from megatoken.training.config import TrainConfig
from megatoken.training.losses import LossLedger, leave_one_out_advantages
from megatoken.training.optimizer import MasterAdamW
from megatoken.training.schedule import PHASES


def test_accumulated_variable_mask_loss_matches_full_batch():
    parameter = torch.nn.Parameter(torch.tensor([0.7, -0.2]))
    batches = [torch.tensor([[1.0, 2.0]]), torch.tensor([[2.0, 0.0], [0.0, 3.0]])]
    ledger = LossLedger(
        {"lexical": 3, "structure": 2},
        {"lexical": 3, "structure": 2},
        {"lexical": 1.0, "structure": 0.25},
    )
    for batch in batches:
        ledger.backward_loss(
            {
                "lexical": ((batch @ parameter).square().sum(), len(batch)),
                "structure": (parameter.square().sum(), 1),
            }
        ).backward()
    actual = parameter.grad.clone()
    parameter.grad = None
    reference = (torch.cat(batches) @ parameter).square().mean() + 0.25 * parameter.square().sum()
    reference.backward()
    torch.testing.assert_close(actual, parameter.grad)
    ledger.finish()


def test_plan_drift_is_fatal():
    ledger = LossLedger({"lexical": 2}, {"lexical": 2}, {"lexical": 1.0})
    ledger.backward_loss({"lexical": (torch.tensor(1.0), 1)})
    with pytest.raises(RuntimeError, match="plan drift"):
        ledger.finish()


def test_bf16_accumulates_fp32_masters_and_moments():
    model = torch.nn.Linear(3, 2).bfloat16()
    optimizer = MasterAdamW(model, lr=1e-3)
    for _ in range(3):
        model(torch.ones(2, 3, dtype=torch.bfloat16)).float().sum().backward()
    assert all(p.grad is None for p in model.parameters())
    assert all(p.dtype == p.grad.dtype == torch.float32 for p in optimizer.parameters)
    optimizer.step()
    for state in optimizer.optimizer.state.values():
        assert state["exp_avg"].dtype == state["exp_avg_sq"].dtype == torch.float32
    optimizer.close()


def test_optimizer_resume_preserves_next_update(tmp_path):
    seed_everything(23)
    model = torch.nn.Linear(3, 2).bfloat16()
    optimizer = MasterAdamW(model, lr=0.002)
    model(torch.ones(1, 3, dtype=torch.bfloat16)).float().square().sum().backward()
    optimizer.step()
    optimizer.zero_grad()
    checkpoint = save_checkpoint(
        tmp_path,
        "step-00000001",
        {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "rng": capture_rng()},
        {"global_step": 1, "codec": "test"},
    )
    payload, _ = load_checkpoint(checkpoint)
    resumed = copy.deepcopy(model)
    resumed_optimizer = MasterAdamW(resumed, lr=0.002)
    resumed.load_state_dict(payload["model"])
    resumed_optimizer.load_state_dict(payload["optimizer"])
    for network, optim in ((model, optimizer), (resumed, resumed_optimizer)):
        network(torch.full((2, 3), 0.3, dtype=torch.bfloat16)).float().square().sum().backward()
        optim.step()
    for a, b in zip(optimizer.parameters, resumed_optimizer.parameters, strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    optimizer.close()
    resumed_optimizer.close()


def test_checkpoint_corruption_and_retention(tmp_path):
    first = save_checkpoint(
        tmp_path, "phase-B0", {"value": torch.ones(2)}, {"global_step": 0}, pinned=True, keep=1
    )
    old = save_checkpoint(tmp_path, "step-1", {}, {"global_step": 1}, keep=1)
    new = save_checkpoint(tmp_path, "step-2", {}, {"global_step": 2}, keep=1)
    assert first.exists() and new.exists() and not old.exists()
    with (new / "state.pt").open("ab") as handle:
        handle.write(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        load_checkpoint(tmp_path)


def test_rng_resume_is_exact():
    seed_everything(123)
    state = capture_rng()
    expected = (random.random(), np.random.rand(), torch.randn(3))
    restore_rng(state)
    actual = (random.random(), np.random.rand(), torch.randn(3))
    assert expected[:2] == actual[:2]
    torch.testing.assert_close(expected[2], actual[2], rtol=0, atol=0)


def test_phase_meters_and_schedules():
    assert sum(PHASES[name].budget for name in ("P1", "M1", "M2", "LC")) == 520_000_000_000
    assert PHASES["B0"].meter == "I"
    assert PHASES["D"].meter == PHASES["R"].meter == "P"
    assert PHASES["B0"].learning_rate(0) == pytest.approx(2e-5)
    assert PHASES["P1"].learning_rate(320_000_000_000) == pytest.approx(2e-4)
    assert PHASES["P1"].learning_rate(400_000_000_000) == pytest.approx(4e-5)


def test_leave_one_out_groups_by_task():
    rewards = torch.tensor([[0.0, 1.0, 0.0, 1.0], [0.2, 0.2, 0.2, 0.2]])
    advantages = leave_one_out_advantages(rewards)
    torch.testing.assert_close(advantages[0], torch.tensor([-2 / 3, 2 / 3, -2 / 3, 2 / 3]))
    torch.testing.assert_close(advantages[1], torch.zeros(4))


def test_configuration_rejects_unknown_or_invalid_fields():
    with pytest.raises(ValueError, match="Unknown"):
        TrainConfig.from_dict({"leaning_rate": 0.01})
    with pytest.raises(ValueError, match="repair_fraction"):
        TrainConfig(repair_fraction=0.4)
