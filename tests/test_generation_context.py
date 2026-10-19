"""Shared C/R computation, exact state ownership, and isolated encoder learning."""

from dataclasses import fields, replace

import pytest
import torch

from megatoken.generation import ActionSpec, FacetedGenerator
from megatoken.generation.structure import TapeMismatch
from megatoken.model import FacetedCore, ModelConfig
from megatoken.model.state import RealizationContext
from megatoken.tensors import byte_tensor


def make_model(**overrides):
    torch.set_num_threads(1)
    torch.manual_seed(392)
    return FacetedGenerator(FacetedCore(ModelConfig.tiny(**overrides)))


def test_direct_realization_and_next_prediction_have_separate_core_dependencies():
    model = make_model()
    values, lengths = byte_tensor([b"existing context"], "cpu")
    source = model.core.observe(values, model.core.initial_state(1), lengths=lengths)
    central_calls, contexts = [], []
    central_hook = model.core.central.register_forward_hook(
        lambda module, args, output: central_calls.append(args[0])
    )
    decoder_hook = model.decoder.realizer.register_forward_pre_hook(
        lambda module, args: contexts.append(args[1])
    )
    spec = ActionSpec(text_kind="ascii", min_tokens=3, max_tokens=3)
    selected = model.sample(source, spec=spec, generator=torch.Generator().manual_seed(3))
    assert len(central_calls) == 1  # Only next-step KV, not realization.
    assert central_calls[0] is selected.state.facets
    assert contexts[0].facets is selected.state.facets
    assert [field.name for field in fields(RealizationContext)] == ["facets"]
    with pytest.raises(ValueError, match="initial sentinel"):
        model.core.prepare_realization(model.core.initial_state(1))
    assert selected.state.records_seen.item() == source.records_seen.item() + 1
    assert model.decode_state(selected.state, spec=spec).payload == selected.payload
    model.core.predict_facets(selected.state)
    assert len(central_calls) == 1
    central_hook.remove()
    decoder_hook.remove()


def test_all_32_raw_slot_identities_reach_short_output_and_receive_gradients():
    model = make_model(n_facets=32, max_capacity=4)
    facets = torch.randn(1, 32, model.config.facet_dim, requires_grad=True)
    rows = model.decoder.realizer(
        model.core, RealizationContext(facets), 1,
        torch.zeros(1, model.config.facet_dim),
    )
    assert model.decoder.realizer.seed_count == 32
    assert rows.shape == (1, 1, model.config.facet_dim)
    rows.square().mean().backward()
    assert facets.grad is not None
    assert bool((facets.grad.norm(dim=-1) > 0).all())


def test_direct_decoder_is_history_independent_and_tape_rejects_forged_bytes():
    model = make_model()
    state = model.core.initial_state(1)
    spec = ActionSpec(text_kind="ascii", min_tokens=1, max_tokens=1)
    sampled = model.sample(state, spec=spec, generator=torch.Generator().manual_seed(7))
    changed = replace(
        sampled.state,
        final=sampled.state.final.flip(-1) * 3,
        history=replace(sampled.state.history, values=sampled.state.history.values + 2),
    )
    captured = []
    hook = model.decoder.realizer.register_forward_hook(
        lambda module, args, output: captured.append(output.detach())
    )
    original = model.decode_state(sampled.state, spec=spec)
    alternate = model.decode_state(changed, spec=spec)
    hook.remove()
    torch.testing.assert_close(captured[0], captured[1], rtol=0, atol=0)
    assert alternate.payload == original.payload
    assert model.score_tape(state, sampled.tape, spec=spec).payload == sampled.payload
    with pytest.raises(TapeMismatch, match="direct raw-facet realization"):
        model.score_tape(state, replace(sampled.tape, payload=b"forged"), spec=spec)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_direct_encoder_auxiliary_has_no_core_decoder_or_source_history_gradients(dtype):
    model = make_model(checkpoint_stages=True).to(dtype=dtype)
    first_plan = model.prepare_target(target_bytes=b"live", representation_mode="recurrent")
    first = model.supervised_step(model.core.initial_state(1), plan=first_plan)
    first.state.facets.retain_grad()
    first.state.history.values.retain_grad()
    plan = model.prepare_target(target_bytes=b"new observation", representation_mode="recurrent")
    flags = {name: parameter.requires_grad for name, parameter in model.named_parameters()}
    result = model.supervised_step(first.state, plan=plan)
    result.losses["reconstruction"][0].backward()
    encoder_ids = {id(parameter) for parameter in model.core.encoder_parameters()}
    assert any(
        parameter.grad is not None and parameter.grad.float().norm() > 0
        for parameter in model.core.encoder_parameters()
    )
    assert all(
        parameter.grad is None
        for parameter in model.parameters()
        if id(parameter) not in encoder_ids
    )
    assert first.state.facets.grad is first.state.history.values.grad is None
    assert flags == {name: parameter.requires_grad for name, parameter in model.named_parameters()}
    assert result.metrics["reconstruction_contextual_examples"] == 0
    assert result.metrics["reconstruction_direct_examples"] == 1


def test_bootstrap_reconstruction_directly_trains_encoder_and_realizer():
    model = make_model()
    plan = model.prepare_target(target_bytes=b"no C needed", representation_mode="bootstrap")
    result = model.supervised_step(model.core.initial_state(1), plan=plan)
    result.losses["reconstruction"][0].backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.core.encoder_parameters()
    )
    assert model.decoder.realizer.seed.weight.grad is not None
    assert all(p.grad is None for p in model.core.central_parameters())
    assert all(p.grad is None for p in model.core.predictor_parameters())


def test_recurrent_e_reconstruction_never_evaluates_central():
    model = make_model()
    plan = model.prepare_target(target_bytes=b"direct codec", representation_mode="recurrent")
    values, lengths = byte_tensor(plan.payloads, "cpu")
    encoded = model.core.encode_chunk(values, lengths)
    calls = []
    hook = model.core.central.register_forward_hook(
        lambda module, inputs, output: calls.append(inputs)
    )
    try:
        losses = model.encoder_reconstruction_losses(encoded, plan)
    finally:
        hook.remove()
    assert calls == []
    losses["lexical"][0].backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.core.encoder_parameters()
    )
    assert all(p.grad is None for p in model.core.central_parameters())


def test_encoder_auxiliary_does_not_contribute_to_main_router_objectives():
    model = make_model()
    state = model.core.initial_state(1)
    model.core.predict_facets(state)
    before = model.core.routing_aux_loss(clear=False)
    plan = model.prepare_target(target_bytes=b"codec", representation_mode="recurrent")
    values, lengths = byte_tensor(plan.payloads, "cpu")
    encoded = model.core.encode_chunk(values, lengths)
    losses = model.encoder_reconstruction_losses(encoded, plan)
    after = model.core.routing_aux_loss(clear=False)
    for original, candidate in zip(before, after, strict=True):
        torch.testing.assert_close(original, candidate, rtol=0, atol=0)
    losses["lexical"][0].backward()
    assert all(parameter.grad is None for parameter in model.core.p_router.parameters())
    assert all(parameter.grad is None for parameter in model.core.u_router.parameters())


def test_native_alignment_uses_only_actual_matching_pairs_and_never_conditions_prior():
    model = make_model()
    initial = model.core.initial_state(1)
    spec = ActionSpec(text_kind="ascii", min_tokens=3, max_tokens=3)
    sample = model.sample(initial, spec=spec, generator=torch.Generator().manual_seed(61))
    plain = model.prepare_target(sample.emitted_ids, representation_mode="recurrent")
    paired = model.prepare_target(
        sample.emitted_ids, representation_mode="recurrent", native_tapes=[sample.tape]
    )
    unpaired = model.supervised_step(
        initial, plan=plain, generator=torch.Generator().manual_seed(2)
    )
    result = model.supervised_step(initial, plan=paired, generator=torch.Generator().manual_seed(2))
    assert torch.equal(unpaired.state.facets, result.state.facets)
    assert plain.denominators["encoder_alignment"] == 0
    assert paired.denominators["encoder_alignment"] == model.latent_dimensions
    assert paired.scored_positions == plain.scored_positions == 6
    result.losses["encoder_alignment"][0].backward()
    encoder_ids = {id(parameter) for parameter in model.core.encoder_parameters()}
    assert any(parameter.grad is not None for parameter in model.core.encoder_parameters())
    assert all(
        parameter.grad is None
        for parameter in model.parameters()
        if id(parameter) not in encoder_ids
    )
    for invalid in (
        replace(sample.tape, payload=b"wrong"),
        replace(sample.tape, emitted_ids=sample.emitted_ids[::-1] + (1,)),
        replace(sample.tape, role_id=4),
        replace(sample.tape, valid=False),
    ):
        with pytest.raises(ValueError, match="actual valid output"):
            model.prepare_target(
                sample.emitted_ids, representation_mode="recurrent", native_tapes=[invalid]
            )
