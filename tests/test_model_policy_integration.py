"""Causal and replay gates spanning the model/realizer interface."""

import pytest
import torch

from megatoken.generation import ActionSpec, ByteCodec, FacetedGenerator
from megatoken.model import FacetedCore, ModelConfig


@pytest.mark.parametrize(
    "device,backend",
    [
        ("cpu", "torch"),
        pytest.param(
            "cuda",
            "torch",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"),
        ),
        pytest.param(
            "cuda",
            "triton",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"),
        ),
    ],
)
def test_bf16_complete_tape_replay_and_writer_to_reader_gradient(device, backend):
    if backend == "triton":
        pytest.importorskip("triton")
    torch.set_num_threads(1)
    torch.manual_seed(741)
    config = ModelConfig.tiny(
        max_capacity=8, checkpoint_stages=True, expert_backend=backend, recurrent_backend=backend
    )
    model = FacetedGenerator(FacetedCore(config), ByteCodec.byte_fallback()).to(
        device=device, dtype=torch.bfloat16
    )
    model.train()
    spec = ActionSpec(text_kind="ascii", min_tokens=3, max_tokens=4, max_repairs=0)
    initial = model.core.initial_state(1)
    prompt = torch.tensor([[65, 66, 67, 68]], device=device)
    with torch.no_grad():
        factual = model.core.ingest(prompt, initial)
        first = model.sample(
            factual, spec=spec, generator=torch.Generator(device=device).manual_seed(12)
        )
        second = model.sample(
            first.state, spec=spec, generator=torch.Generator(device=device).manual_seed(13)
        )
    replay_factual = model.core.ingest(prompt, model.core.initial_state(1))
    replay_first = model.score_tape(replay_factual, first.tape, spec=spec)
    replay_first.state.history_b.values.retain_grad()
    replay_first.state.recurrent[0].retain_grad()
    replay_second = model.score_tape(replay_first.state, second.tape, spec=spec)
    # CPU should be bitwise stable; CUDA dispatches may differ in harmless final
    # accumulation bits. This tight density tolerance is the replay gate.
    tolerance = 0.0 if device == "cpu" else 2e-3
    torch.testing.assert_close(first.log_prob, replay_first.log_prob, rtol=0, atol=tolerance)
    torch.testing.assert_close(second.log_prob, replay_second.log_prob, rtol=0, atol=tolerance)
    torch.testing.assert_close(
        second.state.facets, replay_second.state.facets, rtol=0, atol=tolerance
    )
    torch.testing.assert_close(
        second.state.carry, replay_second.state.carry, rtol=0, atol=tolerance
    )
    (-replay_second.log_prob).backward()
    assert replay_first.state.history_b.values.grad.float().norm() > 0
    assert replay_first.state.recurrent[0].grad.float().norm() > 0
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.grad is not None
    )


def test_u_auxiliary_losses_do_not_leak_across_predictive_decisions():
    torch.set_num_threads(1)
    torch.manual_seed(871)
    core = FacetedCore(ModelConfig.tiny())
    state = core.initial_state(1)
    prediction = core.begin_prediction(state, torch.zeros(1, 8))
    rows = core.apply_u(torch.ones(1, 7, 32), prediction, 8)
    first_balance, first_z = core.u_routing_losses()
    core.begin_prediction(state, torch.ones(1, 8))
    second_balance, second_z = core.u_routing_losses()
    assert second_balance.item() == second_z.item() == 0
    (rows.sum() + first_balance + first_z).backward()
    assert core.u_router.palette_codes.grad.norm() > 0
