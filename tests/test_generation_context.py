"""Shared C/R computation, exact state ownership, and isolated encoder learning."""

from dataclasses import replace

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


def test_selected_c_features_are_computed_once_and_reused_by_decoder_and_next_prediction():
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
    assert len(central_calls) == 1
    assert central_calls[0] is selected.state.facets
    assert contexts[0].facets is selected.state.facets
    assert contexts[0].early is selected.state.early
    assert contexts[0].final is selected.state.final
    assert selected.state.records_seen.item() == source.records_seen.item() + 1
    assert source.records_seen.item() == 1
    assert model.decode_state(selected.state, spec=spec).payload == selected.payload
    model.core.predict_facets(selected.state)
    assert len(central_calls) == 1
    central_hook.remove()
    decoder_hook.remove()


def test_all_32_slot_identities_and_early_final_features_reach_short_output():
    model = make_model(n_facets=32, max_capacity=4)
    tensors = [torch.randn(1, 32, model.config.facet_dim, requires_grad=True) for _ in range(3)]
    context = RealizationContext(*tensors)
    rows = model.decoder.realizer(model.core, context, 1, torch.zeros(1, model.config.facet_dim))
    assert model.decoder.realizer.seed_count == 32
    assert rows.shape == (1, 1, model.config.facet_dim)
    rows.square().mean().backward()
    for value in tensors:
        assert value.grad is not None
        assert bool((value.grad.norm(dim=-1) > 0).all())


def test_cached_final_features_affect_materialization_and_conditional_tape_validation():
    model = make_model()
    state = model.core.initial_state(1)
    spec = ActionSpec(text_kind="ascii", min_tokens=1, max_tokens=1)
    sampled = model.sample(state, spec=spec, generator=torch.Generator().manual_seed(7))
    changed = replace(sampled.state, final=sampled.state.final.flip(-1) * 3)
    captured = []
    hook = model.decoder.realizer.register_forward_hook(
        lambda module, args, output: captured.append(output[0, 0].detach())
    )
    model.decode_state(sampled.state, spec=spec)
    model.decode_state(changed, spec=spec)
    hook.remove()
    first, second = (value / value.norm() for value in captured)
    assert not torch.allclose(first, second)
    # Select a separating lexical readout, so this tests actual decoded bytes
    # rather than relying on a random vocabulary's argmax happening to change.
    with torch.no_grad():
        model.decoder.vocabulary.projection.weight.zero_()
        model.decoder.vocabulary.projection.weight[0] = first - second
        model.decoder.vocabulary.embedding.weight.zero_()
        model.decoder.vocabulary.embedding.weight[ord("a"), 0] = 1
        model.decoder.vocabulary.embedding.weight[ord("b"), 0] = -1
    original = model.decode_state(sampled.state, spec=spec)
    altered = model.decode_state(changed, spec=spec)
    assert original.payload == b"a" and altered.payload == b"b"
    tape = replace(
        sampled.tape,
        payload=original.payload,
        emitted_ids=original.emitted_ids,
        valid=original.valid,
        terminal=original.terminal,
        trace=original.trace,
        error=original.error,
    )
    assert model.score_tape(state, tape, spec=spec).payload == b"a"
    with pytest.raises(TapeMismatch, match="conditional action map"):
        model.score_tape(
            state,
            replace(tape, payload=altered.payload, emitted_ids=altered.emitted_ids),
            spec=spec,
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_contextual_encoder_auxiliary_has_no_core_decoder_or_source_history_gradients(dtype):
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
    assert result.metrics["reconstruction_contextual_examples"] == 1
    assert result.metrics["reconstruction_direct_examples"] == 0


def test_encoder_auxiliary_does_not_contribute_to_main_router_objectives():
    model = make_model()
    state = model.core.initial_state(1)
    model.core.predict_facets(state)
    before = model.core.routing_aux_loss(clear=False)
    plan = model.prepare_target(target_bytes=b"codec", representation_mode="recurrent")
    values, lengths = byte_tensor(plan.payloads, "cpu")
    encoded = model.core.encode_chunk(values, lengths)
    losses = model.encoder_reconstruction_losses(state, encoded, plan, torch.tensor([2]))
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
