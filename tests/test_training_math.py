from __future__ import annotations

import copy
import random

import numpy as np
import pytest
import torch

from megatoken.runtime import capture_rng, restore_rng, seed_everything
from megatoken.training.checkpoint import load_checkpoint, save_checkpoint, transducer_attachment
from megatoken.training.config import TrainConfig
from megatoken.training.losses import LossLedger
from megatoken.training.optimizer import MasterAdamW, parameter_rules
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
        attachments=transducer_attachment(lambda: {"bank": {}}),
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


def test_learning_rules_follow_the_owning_module_not_the_parameter_name():
    from megatoken.generation import ByteCodec
    from megatoken.loading import build_model
    from megatoken.model import ModelConfig

    model = build_model(ModelConfig.tiny(), ByteCodec.byte_fallback())
    rules = parameter_rules(model, 0.1)
    # Embedding tables: half rate, no decay.
    for name in (
        "core.frontend.bytes.weight",
        "decoder.vocabulary.embedding.weight",
        "core.central.role_embedding.weight",
    ):
        assert (rules[name].lr_multiplier, rules[name].weight_decay) == (0.5, 0.0), name
    # Dense linears keep the full rate and decay even when their name says
    # "vocabulary" or "ngram". The n-gram projection is routed now, so its
    # matrix rule belongs to the unique global pool parameter.
    for name in (
        "decoder.vocabulary.projection.weight",
        "core.frontend.ngram_mixture.weight",
    ):
        assert (rules[name].lr_multiplier, rules[name].weight_decay) == (1.0, 0.1), name
    # Routed sites own scalar gains; their matrix rule belongs to the unique
    # registry owner shared by all same-shape linear projections.
    names = set(rules)
    assert "embedding_realizer.query.output_gain" in names
    assert "embedding_realizer.document.output_gain" in names
    pool_parameter = model.core.frontend.ngram_projection._bank.weight
    pool_name = next(
        name for name, parameter in model.named_parameters() if id(parameter) == id(pool_parameter)
    )
    assert pool_name.startswith("core.expert_pools.pools.linear_")
    assert (rules[pool_name].lr_multiplier, rules[pool_name].weight_decay) == (1.0, 0.1)
    query_gain_rule = rules["embedding_realizer.query.output_gain"]
    assert (query_gain_rule.lr_multiplier, query_gain_rule.weight_decay) == (1.0, 0.0)
    # SeedControlEncoder declares slow learning. Its local routed gains and
    # normalization stay at half rate, while the shared projection matrix
    # above keeps the rule of its unique ShapePools owner.
    for name in (
        "core.control_encoder.projection.output_gain",
        "core.control_encoder.output.output_gain",
        "core.control_encoder.norm.weight",
    ):
        assert (rules[name].lr_multiplier, rules[name].weight_decay) == (0.5, 0.0), name
    assert rules["embedding_realizer.norm.weight"] == rules["embedding_realizer.norm.bias"]
    assert rules["embedding_realizer.norm.bias"].weight_decay == 0.0


def test_freezing_decoder_preserves_bf16_owners_fp32_masters_and_adam_moments():
    torch.manual_seed(31)
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 2)).bfloat16()
    optimizer = MasterAdamW(model, lr=0.01, weight_decay=0.2)
    values = torch.ones(2, 3, dtype=torch.bfloat16)
    model(values).float().square().sum().backward()
    optimizer.step()
    optimizer.zero_grad()
    frozen = [name for name, _ in model.named_parameters() if name.startswith("1.")]
    optimizer.freeze(frozen)
    weights = {name: p.detach().clone() for name, p in model.named_parameters()}
    masters = {name: optimizer.masters[name].detach().clone() for name in frozen}
    moments = {
        name: copy.deepcopy(optimizer.optimizer.state[optimizer.masters[name]]) for name in frozen
    }
    model(values * 2).float().square().sum().backward()
    optimizer.step()
    assert not torch.equal(weights["0.weight"], model[0].weight)
    for name in frozen:
        torch.testing.assert_close(weights[name], optimizer.model_parameters[name], rtol=0, atol=0)
        torch.testing.assert_close(masters[name], optimizer.masters[name], rtol=0, atol=0)
        for key, value in moments[name].items():
            torch.testing.assert_close(
                value, optimizer.optimizer.state[optimizer.masters[name]][key], rtol=0, atol=0
            )
    optimizer.close()


def test_encoder_follower_cannot_rescale_main_model_gradient_clipping():
    torch.manual_seed(19)
    baseline = torch.nn.ModuleDict(
        {"encoder": torch.nn.Linear(2, 2), "core": torch.nn.Linear(2, 2)}
    )
    networks = [copy.deepcopy(baseline), copy.deepcopy(baseline)]
    optimizers = [MasterAdamW(model, lr=0.01) for model in networks]
    for optimizer in optimizers:
        optimizer.set_gradient_domains(
            {
                "encoder": [name for name in optimizer.masters if name.startswith("encoder.")],
                "model": [name for name in optimizer.masters if name.startswith("core.")],
            }
        )
    for step in range(3):
        reports = []
        for index, (model, optimizer) in enumerate(zip(networks, optimizers, strict=True)):
            optimizer.zero_grad()
            inputs = torch.tensor([[1.0 + step, -2.0]])
            loss = model["core"](inputs).square().sum()
            loss = loss + (1 if index == 0 else 1000) * model["encoder"](inputs).square().sum()
            loss.backward()
            reports.append(optimizer.step(max_norm=0.1))
        assert reports[0]["gradient_norm_model"] == reports[1]["gradient_norm_model"]
        assert reports[1]["gradient_norm_encoder"] > 100 * reports[0]["gradient_norm_encoder"]
        for name in ("core.weight", "core.bias"):
            torch.testing.assert_close(
                optimizers[0].masters[name], optimizers[1].masters[name], rtol=0, atol=0
            )
            for key in ("exp_avg", "exp_avg_sq"):
                torch.testing.assert_close(
                    optimizers[0].optimizer.state[optimizers[0].masters[name]][key],
                    optimizers[1].optimizer.state[optimizers[1].masters[name]][key],
                    rtol=0,
                    atol=0,
                )
    for optimizer in optimizers:
        optimizer.close()


def test_gradient_domains_require_disjoint_complete_ownership():
    optimizer = MasterAdamW(torch.nn.Linear(2, 2), lr=0.01)
    for invalid in (
        {"a": ["weight"]},
        {"a": ["weight", "bias"], "b": ["bias"]},
        {"a": ["missing", "bias"]},
    ):
        with pytest.raises(ValueError, match="partition"):
            optimizer.set_gradient_domains(invalid)
    optimizer.close()


def test_shared_expert_pool_has_one_training_owner_and_its_own_clip_domain():
    from types import SimpleNamespace

    from megatoken.model.expert_pools import ShapePools

    class TwoSites(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.first = torch.nn.Linear(3, 4)
            self.second = torch.nn.Linear(3, 4)
            self.expert_pools = ShapePools(
                SimpleNamespace(
                    facet_dim=4,
                    projection_experts=3,
                    p_experts=5,
                    u_experts=3,
                    adapter_experts=2,
                    adapter_rank=2,
                    adapter_topk=1,
                    expert_backend="torch",
                )
            )
            self.expert_pools.bind(self, "test")

    model = TwoSites()
    # Both sites refer to the same bank without registering it a second time.
    assert model.first._bank is model.second._bank
    assert model.first.selector is model.second.selector
    pool_parameter = model.first._bank.weight
    owners = [
        name
        for name, parameter in model.named_parameters(remove_duplicate=False)
        if id(parameter) == id(pool_parameter)
    ]
    assert len(owners) == 1

    optimizer = MasterAdamW(model, lr=0.01)
    shared_names = [name for name in optimizer.masters if name.startswith("expert_pools.")]
    rest_names = [name for name in optimizer.masters if name not in set(shared_names)]
    optimizer.set_gradient_domains({"shared": shared_names, "model": rest_names})
    values = torch.randn(2, 3)
    loss = model.first(values).square().sum() + 2 * model.second(values).square().sum()
    loss.backward()
    metrics = optimizer.step(max_norm=0.1)
    assert "gradient_norm_shared" in metrics
    assert "gradient_norm_model" in metrics
    assert any(
        id(parameter) == id(pool_parameter) for parameter in optimizer.model_parameters.values()
    )
    optimizer.close()


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
    assert PHASES["D"].meter == "Q"
    assert PHASES["B0"].learning_rate(0) == pytest.approx(2e-5)
    assert PHASES["P1"].learning_rate(320_000_000_000) == pytest.approx(2e-4)
    assert PHASES["P1"].learning_rate(400_000_000_000) == pytest.approx(4e-5)


def test_configuration_rejects_unknown_or_invalid_fields():
    with pytest.raises(ValueError, match="Unknown"):
        TrainConfig.from_dict({"leaning_rate": 0.01})
    with pytest.raises(ValueError, match="Unknown.*repair_fraction"):
        TrainConfig.from_dict({"repair_fraction": 0.4})
