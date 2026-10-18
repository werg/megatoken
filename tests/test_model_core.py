"""Raw facet recurrence, nonresidual boundary, causal context, and exact replay."""

from dataclasses import fields, replace

import pytest
import torch

from megatoken.model import CoreState, FacetedCore, ModelConfig, RealizationContext
from megatoken.model.audit import audit_parameters


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.set_num_threads(1)
    torch.manual_seed(41)


def assert_history_equal(first, second):
    torch.testing.assert_close(first.keys, second.keys, rtol=0, atol=0)
    torch.testing.assert_close(first.values, second.values, rtol=0, atol=0)
    if first.mask is None or second.mask is None:
        assert first.mask is second.mask
    else:
        torch.testing.assert_close(first.mask, second.mask, rtol=0, atol=0)


def assert_same_state(first, second):
    assert first.geometry == second.geometry
    assert first.model_version == second.model_version
    assert first.routing_version == second.routing_version
    assert first.has_pending_input == second.has_pending_input
    for name in (
        "facets",
        "records_seen",
        "last_record_ids",
        "input_record_ids",
        "role_ids",
        "episode_ids",
    ):
        torch.testing.assert_close(getattr(first, name), getattr(second, name), rtol=0, atol=0)
    if first.pending_mask is None or second.pending_mask is None:
        assert first.pending_mask is second.pending_mask
    else:
        torch.testing.assert_close(first.pending_mask, second.pending_mask, rtol=0, atol=0)
    assert_history_equal(first.history, second.history)


def test_versioned_geometry_and_explicit_compact_ablation():
    large, compact = ModelConfig(), ModelConfig.compact32()
    assert (large.name, large.n_facets, large.facet_dim, large.max_capacity) == (
        "recurrent32",
        32,
        512,
        1024,
    )
    assert (large.p_experts, large.u_experts, large.source_dim) == (64, 32, 256)
    assert (large.query_heads, large.kv_heads, large.attention_head_dim) == (4, 2, 128)
    assert (large.body_stages, large.n_stages, large.remix_hidden) == (6, 16, 1024)
    assert (large.history_slots, large.max_history_chunks) == (32, 1024)
    assert (compact.n_facets, compact.facet_dim, compact.max_capacity) == (32, 256, 1024)
    assert (
        2
        * large.kv_heads
        * large.attention_head_dim
        * large.history_slots
        * large.max_history_chunks
        * 2
        == 32 << 20
    )
    assert large.p_experts * large.pair_dim == 128 * 512
    assert large.u_experts * large.facet_dim == 64 * 256
    assert not any(field.name.startswith("gdn") for field in fields(large))
    assert ModelConfig.from_dict(large.to_dict()) == large
    legacy = large.to_dict()
    del legacy["architecture_version"]
    with pytest.raises(ValueError, match="explicitly declare"):
        ModelConfig.from_dict(legacy)
    with pytest.raises(ValueError, match="Incompatible architecture"):
        ModelConfig(architecture_version=2)
    with pytest.raises(ValueError, match="Old architecture names"):
        ModelConfig(name="v13")


def test_initial_observation_only_encodes_and_skips_sentinel_body():
    core = FacetedCore(ModelConfig.tiny()).eval()
    ids = torch.tensor([[65, 0, 66, 67], [97, 98, 99, 100]])
    initial = core.initial_state(2)
    encoded = core.encode_chunk(ids)
    state = core.observe(ids, initial)
    torch.testing.assert_close(state.facets, encoded, rtol=0, atol=0)
    assert state.records_seen.tolist() == [0, 0]
    assert state.history.slots == 0 and state.has_pending_input
    assert core.dispatch_metrics()["p"]["routed_rows"].sum().item() == 0
    assert isinstance(state, CoreState)
    assert not torch.equal(encoded[0], encoded[1])
    # A future control or history never changes the byte encoder target.
    core.process(state, control=torch.randn(2, core.config.control_dim))
    torch.testing.assert_close(core.encode_chunk(ids), encoded, rtol=0, atol=0)


def test_nonresidual_remix_is_internal_and_final_facets_are_recurrent_input():
    core = FacetedCore(ModelConfig.tiny()).eval()
    state = core.observe(torch.randint(256, (1, 99)), core.initial_state(1))
    boundary = []
    hook = core.remix.register_forward_hook(lambda module, args, output: boundary.append(output))
    result = core.process(state)
    hook.remove()
    assert len(boundary) == 1 and not torch.equal(result.facets, boundary[0])
    assert core.prepare_realization(result.facets).facets is result.facets
    assert result.history.slots == core.config.n_facets
    assert result.records_seen.item() == 1
    next_input = result.facets
    core.process(result)
    assert result.facets is next_input  # processing is immutable, not an in-place writer
    # Removing both learned remix maps removes the signal completely: there is
    # no hidden identity addition across this boundary.
    with torch.no_grad():
        core.remix.channels.down.weight.zero_()
    raw = torch.randn(1, core.config.n_facets, core.config.facet_dim)
    assert core.remix(raw).count_nonzero() == 0


def test_raw_sample_replacement_has_no_neural_work_and_preserves_exact_tensor():
    core = FacetedCore(ModelConfig.tiny()).eval()
    observed = core.observe(torch.randint(256, (1, 9)), core.initial_state(1))
    predicted = core.process(observed)
    raw = torch.randn_like(predicted.facets, dtype=torch.float32, requires_grad=True)
    saved = raw.detach().clone()
    calls = []
    hooks = [
        module.register_forward_hook(lambda *args: calls.append(1)) for module in core.modules()
    ]
    replaced = core.replace_facets(predicted, raw, role_ids=torch.tensor([4]))
    for hook in hooks:
        hook.remove()
    assert calls == []
    assert replaced.facets is raw and replaced.history is predicted.history
    assert replaced.records_seen is predicted.records_seen
    assert replaced.input_record_ids is predicted.input_record_ids
    assert replaced.role_ids.tolist() == [4]
    assert [field.name for field in fields(RealizationContext)] == ["facets"]
    core.process(replaced).facets.square().mean().backward()
    assert raw.grad is not None and raw.grad.norm() > 0
    torch.testing.assert_close(raw, saved, rtol=0, atol=0)


def test_consecutive_observations_capture_prior_input_without_remix_or_prediction():
    core = FacetedCore(ModelConfig.tiny()).eval()
    first = core.observe(torch.tensor([[65, 66]]), core.initial_state(1))
    expected = core.consume_input(first)
    core.dispatch_metrics(reset=True)
    remix_calls = []
    hook = core.remix.register_forward_hook(lambda *args: remix_calls.append(1))
    next_ids = torch.tensor([[67, 68]])
    second = core.observe(next_ids, first, role_ids=torch.tensor([1]))
    hook.remove()
    assert remix_calls == []
    assert_history_equal(second.history, expected.history)
    torch.testing.assert_close(second.facets, core.encode_chunk(next_ids), rtol=0, atol=0)
    assert second.records_seen.item() == 1
    assert second.input_record_ids.item() == 1
    assert core.dispatch_metrics()["p"]["routed_rows"].sum().item() == 6 * 3 * core.config.n_facets
    result = core.process(second)
    assert result.records_seen.item() == 2
    assert result.history.slots == 2 * core.config.n_facets
    assert_history_equal(first.history, core.initial_state(1).history)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_recorded_raw_replay_skips_only_discarded_prediction_work(dtype):
    core = FacetedCore(ModelConfig.tiny()).to(dtype=dtype).eval()
    state = core.observe(torch.randint(256, (1, 55)), core.initial_state(1))
    state.facets.retain_grad()
    control = torch.randn(1, core.config.control_dim)
    raw = torch.randn_like(state.facets, dtype=torch.float32, requires_grad=True)
    roles = torch.tensor([4])
    expected = core.replace_facets(core.process(state, control, 3, next_role_ids=roles), raw)
    core.dispatch_metrics(reset=True)
    actual = core.advance_raw(state, raw, control, 3, role_ids=roles)
    assert_same_state(actual, expected)
    assert actual.facets is raw
    assert core.dispatch_metrics()["p"]["routed_rows"].sum().item() == 6 * 3 * core.config.n_facets
    core.process(actual).facets.square().mean().backward()
    assert raw.grad.norm() > 0
    assert state.facets.grad.norm() > 0


def test_action_is_captured_once_before_observation_without_reencoding(monkeypatch):
    core = FacetedCore(ModelConfig.tiny()).eval()
    source = core.observe(torch.tensor([[65, 66]]), core.initial_state(1))
    predicted = core.process(source)
    action = core.replace_facets(predicted, torch.randn_like(predicted.facets))
    expected = core.consume_input(action)
    encoded_payloads = []
    original = core.encode_chunk

    def recording_encoder(ids, lengths=None):
        encoded_payloads.append(ids.clone())
        return original(ids, lengths)

    monkeypatch.setattr(core, "encode_chunk", recording_encoder)
    feedback_ids = torch.tensor([[111, 107]])
    feedback = core.observe(feedback_ids, action, role_ids=torch.tensor([3]))
    assert len(encoded_payloads) == 1
    torch.testing.assert_close(encoded_payloads[0], feedback_ids)
    assert_history_equal(feedback.history, expected.history)
    assert feedback.records_seen.item() == 2
    assert feedback.history.slots == 2 * core.config.n_facets
    assert core.process(feedback).records_seen.item() == 3


def test_cache_contains_input_body_not_the_future_prediction():
    core = FacetedCore(ModelConfig.tiny()).eval()
    state = core.observe(torch.randint(256, (1, 13)), core.initial_state(1))
    before = core.process(state)
    with torch.no_grad():
        core.stages[-1].always.down.weight.add_(1)
    after = core.process(state)
    assert_history_equal(before.history, after.history)
    assert not torch.equal(before.facets, after.facets)
    assert state.history.slots == 0 and state.records_seen.item() == 0


@pytest.mark.parametrize("checkpoint_stages", [False, True])
def test_earlier_input_and_cache_receive_future_prediction_gradients(checkpoint_stages):
    core = FacetedCore(ModelConfig.tiny(checkpoint_stages=checkpoint_stages))
    first = core.observe(torch.randint(256, (1, 97)), core.initial_state(1))
    first.facets.retain_grad()
    second = core.observe(torch.randint(256, (1, 19)), first)
    second.history.keys.retain_grad()
    second.history.values.retain_grad()
    prediction = core.process(second).facets
    (prediction * torch.randn_like(prediction)).sum().backward()
    assert first.facets.grad.norm() > 0
    assert second.history.keys.grad.norm() > 0
    assert second.history.values.grad.norm() > 0
    assert core.frontend.bytes.weight.grad.norm() > 0
    assert core.live_attention.key_value.weight.grad.norm() > 0


def test_current_encoder_input_does_not_erase_different_earlier_context():
    core = FacetedCore(ModelConfig.tiny()).eval()
    shared = torch.tensor([[88, 89]])
    left = core.observe(shared, core.observe(torch.tensor([[65]]), core.initial_state(1)))
    right = core.observe(shared, core.observe(torch.tensor([[66]]), core.initial_state(1)))
    torch.testing.assert_close(left.facets, right.facets, rtol=0, atol=0)
    assert not torch.equal(core.process(left).facets, core.process(right).facets)


def test_detach_cuts_both_raw_recurrence_and_history():
    core = FacetedCore(ModelConfig.tiny())
    state = core.observe(torch.randint(256, (1, 49)), core.initial_state(1))
    state.facets.retain_grad()
    first = core.process(state)
    detached = first.detach()
    assert not detached.facets.requires_grad
    assert not detached.history.keys.requires_grad
    assert not detached.history.values.requires_grad
    core.process(detached).facets.square().mean().backward()
    assert state.facets.grad is None


def test_detached_observation_encoder_preserves_pending_body_gradient():
    core = FacetedCore(ModelConfig.tiny())
    pending = core.process(core.initial_state(1))
    pending.facets.retain_grad()
    observed = core.observe(torch.randint(256, (1, 43)), pending, encoder_grad=False)
    assert not observed.facets.requires_grad
    assert observed.history.values.requires_grad
    core.process(observed).facets.square().mean().backward()
    assert pending.facets.grad.norm() > 0
    assert all(parameter.grad is None for parameter in core.encoder_parameters())
    assert core.remix.channels.down.weight.grad.norm() > 0


def test_detached_control_frontend_keeps_control_encoder_trainable():
    core = FacetedCore(ModelConfig.tiny())
    control = core.encode_control(torch.randint(256, (2, 43)), encoder_grad=False)
    control.square().mean().backward()
    assert all(parameter.grad is None for parameter in core.encoder_parameters())
    assert all(parameter.requires_grad for parameter in core.encoder_parameters())
    assert all(
        parameter.grad is not None and parameter.grad.norm() > 0
        for parameter in core.control_encoder.parameters()
    )


def test_exact_exposure_and_checkpoint_recompute_accounting():
    core = FacetedCore(ModelConfig.tiny(checkpoint_stages=True))
    state = core.observe(torch.randint(256, (1, 43)), core.initial_state(1))
    assert core.dispatch_metrics(reset=True)["p"]["routed_rows"].sum().item() == 0
    prediction = core.process(state, effort=3).facets
    count = core.dispatch_metrics()["p"]["routed_rows"].sum().item()
    assert count == 6 * 3 * core.config.n_facets + 10 * core.config.n_facets * 3 // 2
    first_aux = core.routing_aux_loss(clear=False)
    (prediction.sum() + sum(first_aux)).backward()
    assert core.dispatch_metrics()["p"]["routed_rows"].sum().item() == count
    for before, after in zip(first_aux, core.routing_aux_loss(), strict=True):
        torch.testing.assert_close(before, after, rtol=0, atol=0)
    assert all(loss.item() == 0 for loss in core.routing_aux_loss())


def test_bounded_history_reset_duplicates_and_model_version_contracts():
    core = FacetedCore(ModelConfig.tiny(max_history_chunks=2))
    state = core.observe(torch.randint(256, (2, 7)), core.initial_state(2))
    state = core.process(core.process(state))
    previous = state.history
    state = core.process(state)
    assert state.history.slots == 2 * core.config.n_facets
    torch.testing.assert_close(
        state.history.keys[:, :, : core.config.n_facets],
        previous.keys[:, :, core.config.n_facets :],
        rtol=0,
        atol=0,
    )
    with pytest.raises(ValueError, match="duplicate"):
        core.observe(torch.randint(256, (2, 7)), state, record_ids=state.input_record_ids)
    reset = core.reset_state(state, torch.tensor([True, False]), torch.tensor([9, 10]))
    assert reset.records_seen.tolist() == [0, 3]
    assert reset.episode_ids.tolist() == [9, 0]
    assert reset.history.keys[0].count_nonzero() == 0
    assert not reset.history.mask[0].any()
    torch.testing.assert_close(reset.facets[0], core.initial_facets)
    torch.testing.assert_close(reset.history.keys[1], state.history.keys[1])
    incompatible = replace(state, geometry=ModelConfig.compact32().state_geometry)
    with pytest.raises(ValueError, match="geometry"):
        core.process(incompatible)
    core.set_model_version(1)
    with pytest.raises(RuntimeError, match="Stale cache"):
        core.process(state)


@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16))
def test_no_hidden_randomness_and_complete_core_replay(dtype):
    core = FacetedCore(ModelConfig.tiny()).to(dtype=dtype).train()
    ids = torch.randint(256, (1, 47))
    rng = torch.random.get_rng_state().clone()
    with torch.no_grad():
        observed = core.observe(ids, core.initial_state(1))
        first = core.process(observed, effort=3)
        first_rows = core.lexical_refine(
            core.apply_u(first.facets, core.prepare_realization(first.facets), 4)
        )
    replay = core.observe(ids, core.initial_state(1))
    second = core.process(replay, effort=3)
    second_rows = core.lexical_refine(
        core.apply_u(second.facets, core.prepare_realization(second.facets), 4)
    )
    torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)
    assert_same_state(first, second)
    torch.testing.assert_close(first_rows, second_rows, rtol=0, atol=0)
    core.eval()
    changed_metadata = replace(replay, episode_ids=torch.tensor([999]))
    torch.testing.assert_close(
        core.process(changed_metadata, effort=3).facets, second.facets, rtol=0, atol=0
    )


def test_bf16_all_served_paths_and_unconditional_initial_state_are_differentiable():
    core = FacetedCore(ModelConfig.tiny()).bfloat16()
    ids = torch.randint(256, (1, 99))
    control = core.encode_control(ids)
    state = core.observe(ids, core.initial_state(1))
    prediction = core.process(state, control).facets
    rows = core.apply_u(prediction.float(), core.prepare_realization(prediction.float()), 4)
    rows = core.lexical_refine(rows)
    unconditional = core.process(core.initial_state(1)).facets
    balance, stabilization = core.routing_aux_loss()
    loss = (
        (rows.float() * torch.randn_like(rows.float())).mean()
        + unconditional.float().square().mean()
        + 0.005 * balance
        + 1e-4 * stabilization
    )
    loss.backward()
    missing = [name for name, parameter in core.named_parameters() if parameter.grad is None]
    assert missing == []
    assert all(torch.isfinite(parameter.grad).all() for parameter in core.parameters())
    assert prediction.dtype == torch.bfloat16


def test_decoder_owner_freeze_does_not_freeze_core_body_or_remix():
    core = FacetedCore(ModelConfig.tiny())
    encoder, decoder = list(core.encoder_parameters()), list(core.realizer_parameters())
    assert len({id(parameter) for parameter in encoder + decoder}) == len(encoder + decoder)
    for parameter in encoder + decoder:
        parameter.requires_grad_(False)
    state = core.observe(torch.randint(256, (1, 51)), core.initial_state(1))
    core.process(state).facets.square().mean().backward()
    assert all(parameter.grad is None for parameter in encoder + decoder)
    assert core.remix.channels.down.weight.grad.norm() > 0
    assert core.live_attention.key_value.weight.grad.norm() > 0


def test_real_owner_counts_and_shared_banks():
    config = ModelConfig.tiny()
    core = FacetedCore(config)
    audit = audit_parameters(core)
    assert audit["total_parameters"] == sum(parameter.numel() for parameter in core.parameters())
    assert sum(parameter.numel() for parameter in core.p_bank.parameters()) == (
        3 * config.p_experts * config.pair_dim * config.p_hidden
    )
    assert len({id(stage.always.gate_up.weight) for stage in core.stages}) == 16
    assert sum(name == "u_always.gate_up.weight" for name, _ in core.named_parameters()) == 1
    assert not any("gdn" in name or "carry" in name for name, _ in core.named_parameters())
    assert core.history_pool is None  # primary keeps every input facet in KV context
    model = torch.nn.Module()
    model.first = core.u_always
    model.alias = core.u_always
    assert audit_parameters(model)["aliases"] == 2


@pytest.mark.parametrize("effort_index,effort", enumerate((1, 2, 3, 6)))
def test_fixed_pair_reduction_matches_incident_facet_average_and_gradient(effort_index, effort):
    """Replace unordered CUDA atomics without changing the mathematical update."""
    core = FacetedCore(ModelConfig.tiny())
    for stage in range(core.config.n_stages):
        for update, count in enumerate(core._row_counts(core.config.n_facets, effort)):
            contribution = torch.randn(2, 2 * count, 3, dtype=torch.float64, requires_grad=True)
            destinations = torch.cat(
                (core.pair_left[stage, update, :count], core.pair_right[stage, update, :count])
            )
            reference = contribution.new_zeros(2, core.config.n_facets, 3).index_add(
                1, destinations, contribution
            )
            degree = torch.bincount(destinations, minlength=core.config.n_facets).clamp_min(1)
            reference = reference / degree[None, :, None]
            indices = core.pair_reduction_indices[stage, update, effort_index]
            mask = core.pair_reduction_mask[stage, update, effort_index]
            actual = (contribution[:, indices] * mask[None, :, :, None]).sum(2)
            actual = actual / core.pair_reduction_degree[stage, update, effort_index][None, :, None]
            torch.testing.assert_close(actual, reference, rtol=1e-15, atol=1e-15)
            upstream = torch.randn_like(actual)
            (expected_grad,) = torch.autograd.grad(
                reference, contribution, upstream, retain_graph=True
            )
            (actual_grad,) = torch.autograd.grad(actual, contribution, upstream)
            torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)
