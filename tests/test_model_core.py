"""Shared central processing, exact raw identity, reusable caches, and gradients."""

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
    for name in (
        "facets",
        "early",
        "final",
        "records_seen",
        "last_record_ids",
        "role_ids",
        "episode_ids",
    ):
        left, right = getattr(first, name), getattr(second, name)
        if left is None or right is None:
            assert left is right
        else:
            torch.testing.assert_close(left, right, rtol=0, atol=0)
    assert_history_equal(first.history, second.history)


def test_versioned_geometry_and_explicit_width_presets():
    large, compact = ModelConfig(), ModelConfig.compact32()
    assert (large.name, large.n_facets, large.facet_dim, large.max_capacity) == (
        "32x512",
        32,
        512,
        1024,
    )
    assert (compact.name, compact.n_facets, compact.facet_dim) == ("32x256", 32, 256)
    assert (large.encoder_layers, large.central_layers, large.prediction_layers) == (2, 4, 12)
    assert (large.central_early_layer, large.n_stages, large.central_hidden) == (2, 16, 1024)
    assert (large.p_experts, large.u_experts, large.source_dim) == (64, 32, 256)
    assert (large.query_heads, large.kv_heads, large.attention_head_dim) == (4, 2, 128)
    assert (large.history_slots, large.max_history_chunks) == (32, 1024)
    assert large.codec_noise_std == large.prior_noise_std == 0.1
    assert (
        2
        * large.kv_heads
        * large.attention_head_dim
        * large.history_slots
        * large.max_history_chunks
        * 2
        == 32 << 20
    )
    assert ModelConfig.from_dict(large.to_dict()) == large
    legacy = large.to_dict()
    del legacy["architecture_version"]
    with pytest.raises(ValueError, match="explicitly declare"):
        ModelConfig.from_dict(legacy)
    with pytest.raises(ValueError, match="Incompatible architecture"):
        ModelConfig(architecture_version=3)
    with pytest.raises(ValueError, match="Old architecture names"):
        ModelConfig(name="recurrent32")
    assert not any(
        field.name in {"pending_mask", "has_pending_input", "input_record_ids"}
        for field in fields(CoreState)
    )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 0.0, 0.001, 3.0])
def test_invalid_codec_and_prior_noise_scales_are_rejected(value):
    with pytest.raises(ValueError, match="codec_noise_std"):
        ModelConfig.tiny(codec_noise_std=value)
    with pytest.raises(ValueError, match="prior_noise_std"):
        ModelConfig.tiny(prior_noise_std=value)


def test_observation_processes_new_input_immediately_and_skips_initial_sentinel():
    core = FacetedCore(ModelConfig.tiny()).eval()
    ids = torch.tensor([[65, 0, 66, 67], [97, 98, 99, 100]])
    initial = core.initial_state(2)
    encoded = core.encode_chunk(ids)
    calls = []
    hook = core.central.register_forward_pre_hook(lambda module, args: calls.append(args[0]))
    state = core.observe(ids, initial)
    hook.remove()
    assert len(calls) == 1 and calls[0] is state.facets
    torch.testing.assert_close(state.facets, encoded, rtol=0, atol=0)
    assert state.records_seen.tolist() == [1, 1]
    assert state.last_record_ids.tolist() == [0, 0]
    assert state.history.slots == core.config.n_facets
    assert state.early.shape == state.final.shape == state.facets.shape
    assert core.dispatch_metrics()["p"]["routed_rows"].sum().item() == 0
    assert initial.history.slots == 0 and initial.final is None
    projected = core.central.memory(state.final)
    torch.testing.assert_close(state.history.keys, projected.keys, rtol=0, atol=0)
    torch.testing.assert_close(state.history.values, projected.values, rtol=0, atol=0)


def test_realizer_and_predictor_reuse_the_same_central_features_without_work():
    core = FacetedCore(ModelConfig.tiny()).eval()
    state = core.observe(torch.randint(256, (1, 55)), core.initial_state(1))
    calls = []
    hook = core.central.register_forward_hook(lambda *args: calls.append(1))
    context = core.prepare_realization(state)
    mean = core.predict_facets(state)
    other = core.predict_from_processed(state)
    hook.remove()
    assert calls == []
    assert (
        context.facets is state.facets
        and context.early is state.early
        and context.final is state.final
    )
    torch.testing.assert_close(mean, other, rtol=0, atol=0)
    assert state.records_seen.item() == 1 and state.history.slots == core.config.n_facets
    assert [field.name for field in fields(RealizationContext)] == ["facets", "early", "final"]
    auxiliary = core.prepare_realization(state.facets)
    assert auxiliary.facets is state.facets and auxiliary.early is auxiliary.final is None


def test_predictor_depends_on_processed_final_and_history_without_raw_skip():
    core = FacetedCore(ModelConfig.tiny()).eval()
    state = core.observe(torch.randint(256, (1, 27)), core.initial_state(1))
    expected = core.predict_facets(state)
    changed_final = replace(state, final=state.final + torch.randn_like(state.final))
    assert not torch.equal(core.predict_facets(changed_final), expected)
    changed_history = replace(
        state, history=replace(state.history, values=state.history.values + 2)
    )
    assert not torch.equal(core.predict_facets(changed_history), expected)
    changed_raw = replace(state, facets=state.facets + torch.randn_like(state.facets))
    torch.testing.assert_close(core.predict_facets(changed_raw), expected, rtol=0, atol=0)


def test_nonresidual_remix_precedes_twelve_residual_prediction_layers():
    core = FacetedCore(ModelConfig.tiny()).eval()
    state = core.observe(torch.randint(256, (1, 99)), core.initial_state(1))
    boundary = []
    hook = core.remix.register_forward_hook(
        lambda module, args, output: boundary.append((args[0], output))
    )
    result = core.predict_facets(state)
    hook.remove()
    assert len(boundary) == 1 and boundary[0][0] is state.final
    assert not torch.equal(result, boundary[0][1])
    assert len(core.stages) == 12
    with torch.no_grad():
        core.remix.channels.down.weight.zero_()
    assert core.remix(torch.randn_like(state.facets)).count_nonzero() == 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_raw_selection_preserves_exact_fp32_tensor_and_caches_one_c_pass(dtype):
    core = FacetedCore(ModelConfig.tiny()).to(dtype=dtype).eval()
    state = core.observe(torch.randint(256, (1, 55)), core.initial_state(1))
    raw = torch.randn_like(state.facets, dtype=torch.float32, requires_grad=True)
    saved = raw.detach().clone()
    calls = []
    hook = core.central.register_forward_pre_hook(lambda module, args: calls.append(args[0]))
    selected = core.advance_raw(state, raw, role_ids=torch.tensor([4]))
    core.predict_facets(selected, effort=3)
    core.prepare_realization(selected)
    hook.remove()
    assert len(calls) == 1 and calls[0] is raw
    assert selected.facets is raw and selected.facets.dtype == torch.float32
    assert selected.early.dtype == selected.final.dtype == dtype
    assert selected.records_seen.item() == 2 and selected.history.slots == 2 * core.config.n_facets
    core.predict_facets(selected).float().square().mean().backward()
    assert raw.grad is not None and raw.grad.norm() > 0
    torch.testing.assert_close(raw, saved, rtol=0, atol=0)


def test_observation_and_fixed_raw_share_exact_central_transition():
    core = FacetedCore(ModelConfig.tiny()).eval()
    prior = core.observe(torch.tensor([[65, 66]]), core.initial_state(1))
    ids, roles = torch.tensor([[67, 68]]), torch.tensor([1])
    expected = core.advance_raw(prior, core.encode_chunk(ids), role_ids=roles)
    actual = core.observe(ids, prior, role_ids=roles)
    assert_same_state(actual, expected)
    assert actual.records_seen.item() == 2
    assert core.dispatch_metrics()["p"]["routed_rows"].sum().item() == 0


def test_action_then_observation_never_reencodes_or_reprocesses_action(monkeypatch):
    core = FacetedCore(ModelConfig.tiny()).eval()
    source = core.observe(torch.tensor([[65, 66]]), core.initial_state(1))
    raw = core.predict_facets(source)
    action = core.advance_raw(source, raw)
    seen = []
    hook = core.central.register_forward_pre_hook(lambda module, args: seen.append(args[0]))
    feedback_ids = torch.tensor([[111, 107]])
    feedback = core.observe(feedback_ids, action, role_ids=torch.tensor([3]))
    hook.remove()
    assert len(seen) == 1 and seen[0] is feedback.facets and seen[0] is not raw
    assert feedback.records_seen.item() == 3
    assert feedback.history.slots == 3 * core.config.n_facets
    torch.testing.assert_close(
        feedback.history.keys[:, :, : action.history.slots], action.history.keys, rtol=0, atol=0
    )


def test_control_changes_prediction_without_changing_selected_processing():
    core = FacetedCore(ModelConfig.tiny()).eval()
    state = core.observe(torch.randint(256, (2, 31)), core.initial_state(2))
    raw, early, final, history = state.facets, state.early, state.final, state.history
    first = core.predict_facets(state)
    second = core.predict_facets(state, torch.randn(2, core.config.control_dim))
    assert not torch.equal(first, second)
    assert (
        state.facets is raw
        and state.early is early
        and state.final is final
        and state.history is history
    )
    with pytest.raises(TypeError):
        core.advance_raw(state, raw, control=torch.zeros(2, core.config.control_dim))


@pytest.mark.parametrize("checkpoint_stages", [False, True])
def test_earlier_raw_and_cache_receive_future_prediction_gradients(checkpoint_stages):
    core = FacetedCore(ModelConfig.tiny(checkpoint_stages=checkpoint_stages))
    raw = torch.randn(1, 4, 32, requires_grad=True)
    first = core.advance_raw(core.initial_state(1), raw)
    first.history.keys.retain_grad()
    first.history.values.retain_grad()
    second = core.observe(torch.randint(256, (1, 19)), first)
    prediction = core.predict_facets(second)
    (prediction * torch.randn_like(prediction)).sum().backward()
    assert raw.grad.norm() > 0
    assert first.history.keys.grad.norm() > 0 and first.history.values.grad.norm() > 0
    assert core.frontend.bytes.weight.grad.norm() > 0
    assert core.central.memory.key_value.weight.grad.norm() > 0


def test_same_observation_retains_different_earlier_context_in_central_features():
    core = FacetedCore(ModelConfig.tiny()).eval()
    shared = torch.tensor([[88, 89]])
    left = core.observe(shared, core.observe(torch.tensor([[65]]), core.initial_state(1)))
    right = core.observe(shared, core.observe(torch.tensor([[66]]), core.initial_state(1)))
    torch.testing.assert_close(left.facets, right.facets, rtol=0, atol=0)
    assert not torch.equal(left.final, right.final)
    assert not torch.equal(core.predict_facets(left), core.predict_facets(right))


def test_detach_cuts_raw_and_all_cached_features_and_history():
    core = FacetedCore(ModelConfig.tiny())
    raw = torch.randn(1, 4, 32, requires_grad=True)
    state = core.advance_raw(core.initial_state(1), raw)
    detached = state.detach()
    for value in (
        detached.facets,
        detached.early,
        detached.final,
        detached.history.keys,
        detached.history.values,
    ):
        assert not value.requires_grad
    core.predict_facets(detached).square().mean().backward()
    assert raw.grad is None


def test_detached_observation_encoder_preserves_central_and_live_recurrence_gradients():
    core = FacetedCore(ModelConfig.tiny())
    raw = core.predict_facets(core.initial_state(1))
    raw.retain_grad()
    selected = core.advance_raw(core.initial_state(1), raw)
    observed = core.observe(torch.randint(256, (1, 43)), selected, encoder_grad=False)
    assert not observed.facets.requires_grad and observed.final.requires_grad
    core.predict_facets(observed).square().mean().backward()
    assert raw.grad.norm() > 0
    assert all(parameter.grad is None for parameter in core.encoder_parameters())
    assert core.central.layers[0].ffn.down.weight.grad.norm() > 0


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


def test_exact_predictor_exposure_and_checkpoint_recompute_accounting():
    core = FacetedCore(ModelConfig.tiny(checkpoint_stages=True))
    state = core.observe(torch.randint(256, (1, 43)), core.initial_state(1))
    assert core.dispatch_metrics(reset=True)["p"]["routed_rows"].sum().item() == 0
    prediction = core.predict_facets(state, effort=3)
    count = core.dispatch_metrics()["p"]["routed_rows"].sum().item()
    assert count == 12 * core.config.n_facets * 3 // 2
    first_aux = core.routing_aux_loss(clear=False)
    (prediction.sum() + sum(first_aux)).backward()
    assert core.dispatch_metrics()["p"]["routed_rows"].sum().item() == count
    for before, after in zip(first_aux, core.routing_aux_loss(), strict=True):
        torch.testing.assert_close(before, after, rtol=0, atol=0)
    assert all(loss.item() == 0 for loss in core.routing_aux_loss())


def test_central_and_prediction_checkpointing_preserve_gradients_and_cache():
    reference = FacetedCore(ModelConfig.tiny())
    checked = FacetedCore(ModelConfig.tiny(checkpoint_stages=True))
    checked.load_state_dict(reference.state_dict())
    raw = torch.randn(1, 4, 32)
    for core in (reference, checked):
        selected = core.advance_raw(core.initial_state(1), raw)
        mean = core.predict_facets(selected, effort=3)
        next_state = core.advance_raw(selected, mean)
        loss = core.predict_facets(next_state, effort=1).square().mean() + sum(
            core.routing_aux_loss()
        )
        loss.backward()
    for (name, parameter), (other_name, other) in zip(
        reference.named_parameters(), checked.named_parameters(), strict=True
    ):
        assert name == other_name
        if parameter.grad is None or other.grad is None:
            assert parameter.grad is other.grad
        else:
            torch.testing.assert_close(parameter.grad, other.grad, rtol=1e-5, atol=1e-6)


def test_auxiliary_routing_isolation_preserves_outer_graph_after_nested_failure():
    core = FacetedCore(ModelConfig.tiny())
    state = core.observe(torch.randint(256, (1, 23)), core.initial_state(1))
    mean = core.predict_facets(state)
    original = core.routing_aux_loss(clear=False)
    with core.discard_routing_losses():
        rows = core.apply_u(mean, core.prepare_realization(mean), 4)
        inner = core.routing_aux_loss(clear=False)
        with pytest.raises(RuntimeError, match="auxiliary failed"):
            with core.discard_routing_losses():
                core.apply_u(rows, core.prepare_realization(mean), 8)
                raise RuntimeError("auxiliary failed")
        for before, after in zip(inner, core.routing_aux_loss(), strict=True):
            torch.testing.assert_close(before, after, rtol=0, atol=0)
    for before, after in zip(original, core.routing_aux_loss(), strict=True):
        torch.testing.assert_close(before, after, rtol=0, atol=0)
    sum(original).backward()
    assert core.p_router.palette_codes.grad.norm() > 0
    assert all(parameter.grad is None for parameter in core.u_router.parameters())


def test_bounded_history_reset_duplicates_and_model_versions():
    core = FacetedCore(ModelConfig.tiny(max_history_chunks=2))
    state = core.observe(torch.randint(256, (2, 7)), core.initial_state(2))
    for _ in range(2):
        state = core.advance_raw(state, core.predict_facets(state))
    assert state.records_seen.tolist() == [3, 3]
    assert state.history.slots == 2 * core.config.n_facets
    with pytest.raises(ValueError, match="duplicate"):
        core.observe(torch.randint(256, (2, 7)), state, record_ids=state.last_record_ids)
    reset = core.reset_state(state, torch.tensor([True, False]), torch.tensor([9, 10]))
    assert reset.records_seen.tolist() == [0, 3]
    assert reset.episode_ids.tolist() == [9, 0]
    assert reset.history.keys[0].count_nonzero() == 0 and not reset.history.mask[0].any()
    torch.testing.assert_close(reset.facets[0], core.initial_facets)
    torch.testing.assert_close(reset.final[1], state.final[1], rtol=0, atol=0)
    torch.testing.assert_close(reset.history.keys[1], state.history.keys[1], rtol=0, atol=0)
    torch.testing.assert_close(
        core.predict_facets(reset)[:1],
        core.predict_facets(core.initial_state(1)),
        rtol=2e-5,
        atol=2e-6,
    )
    with pytest.raises(ValueError, match="geometry"):
        core.predict_facets(replace(state, geometry=ModelConfig.compact32().state_geometry))
    core.set_model_version(1)
    with pytest.raises(RuntimeError, match="Stale cache"):
        core.predict_facets(state)


@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16))
def test_no_hidden_randomness_and_complete_central_replay(dtype):
    core = FacetedCore(ModelConfig.tiny()).to(dtype=dtype).train()
    ids = torch.randint(256, (1, 47))
    raw = torch.randn(1, 4, 32)
    rng = torch.random.get_rng_state().clone()
    with torch.no_grad():
        first = core.advance_raw(core.observe(ids, core.initial_state(1)), raw)
        first_mean = core.predict_facets(first, effort=3)
    second = core.advance_raw(core.observe(ids, core.initial_state(1)), raw)
    second_mean = core.predict_facets(second, effort=3)
    torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)
    assert_same_state(first, second)
    torch.testing.assert_close(first_mean, second_mean, rtol=0, atol=0)
    core.eval()
    torch.testing.assert_close(
        core.predict_facets(replace(second, episode_ids=torch.tensor([999])), effort=3),
        second_mean,
        rtol=0,
        atol=0,
    )


def test_bf16_every_owner_and_unconditional_initial_state_are_differentiable():
    core = FacetedCore(ModelConfig.tiny()).bfloat16()
    ids = torch.randint(256, (1, 99))
    control = core.encode_control(ids)
    state = core.observe(ids, core.initial_state(1))
    prediction = core.predict_facets(state, control)
    selected = core.advance_raw(state, prediction.float())
    rows = core.apply_u(prediction.float(), core.prepare_realization(selected), 4)
    rows = core.lexical_refine(rows)
    unconditional = core.predict_facets(core.initial_state(1))
    balance, stabilization = core.routing_aux_loss()
    loss = (
        (rows.float() * torch.randn_like(rows.float())).mean()
        + unconditional.float().square().mean()
        + 0.005 * balance
        + 1e-4 * stabilization
    )
    loss.backward()
    assert [name for name, parameter in core.named_parameters() if parameter.grad is None] == []
    assert all(torch.isfinite(parameter.grad).all() for parameter in core.parameters())
    assert prediction.dtype == torch.bfloat16


def test_owner_groups_are_disjoint_complete_and_allow_fixed_decoder_latent_rl():
    core = FacetedCore(ModelConfig.tiny())
    groups = [
        list(core.encoder_parameters()),
        list(core.central_parameters()),
        list(core.predictor_parameters()),
        list(core.realizer_parameters()),
    ]
    owners = [parameter for group in groups for parameter in group]
    assert len({id(parameter) for parameter in owners}) == len(owners)
    assert {id(parameter) for parameter in owners} == {
        id(parameter) for parameter in core.parameters()
    }
    for group in (groups[0], groups[1], groups[3]):
        for parameter in group:
            parameter.requires_grad_(False)
    state = core.observe(torch.randint(256, (1, 51)), core.initial_state(1))
    core.predict_facets(state).square().mean().backward()
    assert all(
        parameter.grad is None for group in (groups[0], groups[1], groups[3]) for parameter in group
    )
    assert core.remix.channels.down.weight.grad.norm() > 0
    assert core.stages[0].attention.query.weight.grad.norm() > 0
    assert core.p_bank.gate_up.grad.norm() > 0


def test_real_owner_counts_shared_kv_and_expert_banks():
    config = ModelConfig.tiny()
    core = FacetedCore(config)
    audit = audit_parameters(core)
    assert audit["total_parameters"] == sum(parameter.numel() for parameter in core.parameters())
    assert (
        sum(parameter.numel() for parameter in core.p_bank.parameters())
        == 3 * config.p_experts * config.pair_dim * config.p_hidden
    )
    assert len({id(stage.always.gate_up.weight) for stage in core.stages}) == 12
    assert len({id(stage.attention.query.weight) for stage in core.stages}) == 12
    assert all(layer.kv is None for layer in core.central.layers)
    assert core.central.history_pool is None
    assert (
        sum(name == "central.memory.key_value.weight" for name, _ in core.named_parameters()) == 1
    )
    assert sum(name == "u_always.gate_up.weight" for name, _ in core.named_parameters()) == 1
    assert not any("gdn" in name or "carry" in name for name, _ in core.named_parameters())


@pytest.mark.parametrize("effort_index,effort", enumerate((1, 2, 3, 6)))
def test_fixed_pair_reduction_matches_incident_facet_average_and_gradient(effort_index, effort):
    """Replace unordered CUDA atomics without changing the mathematical update."""
    core = FacetedCore(ModelConfig.tiny())
    for stage in range(core.config.prediction_layers):
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


def test_completed_old_history_uses_coarse_slots_with_live_writer_gradients():
    config = ModelConfig.tiny(
        max_history_chunks=4, recent_history_chunks=2, archived_slots_per_chunk=1
    )
    core = FacetedCore(config)
    state = core.initial_state(1)
    raw = torch.randn(1, config.n_facets, config.facet_dim, requires_grad=True)
    state = core.advance_raw(state, raw)
    assert state.history.slots == 4 and state.history.archived_chunks == 0
    state = core.advance_raw(state, torch.randn_like(raw))
    assert state.history.slots == 8 and state.history.archived_chunks == 0
    state = core.advance_raw(state, torch.randn_like(raw))
    assert state.history.slots == 9 and state.history.archived_chunks == 1
    core.predict_facets(state, effort=1).square().mean().backward()
    assert raw.grad is not None and raw.grad.norm() > 0

    # Exactly two older records remain coarse, plus two complete recent ones;
    # a fifth action evicts only the oldest coarse record.
    for expected in (10, 10):
        state = core.advance_raw(state.detach(), torch.randn_like(raw).detach())
        assert state.history.slots == expected
        assert state.history.archived_chunks == 2
    assert state.records_seen.item() == 5
    assert state.history.keys.isfinite().all()
