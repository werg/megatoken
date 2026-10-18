import pytest
import torch

from megatoken.model import FacetedCore, ModelConfig
from megatoken.model.audit import audit_parameters


@pytest.fixture(autouse=True)
def deterministic_small_cpu():
    torch.set_num_threads(1)
    torch.manual_seed(7)


def predict(core, state, *, root=None, effort=6):
    root = torch.zeros(state.batch_size, core.config.root_dim) if root is None else root
    prediction = core.begin_prediction(state, root, effort=effort)
    for stage in (7, 8, 10, 15, 16):
        prediction = core.advance_prediction(prediction, stage)
    return prediction


def test_factual_memory_writes_once_and_prediction_never_writes():
    core = FacetedCore(ModelConfig.tiny()).eval()
    state = core.initial_state(2)
    state = core.ingest(torch.randint(256, (2, 19)), state)
    assert state.records_seen.tolist() == [1, 1]
    assert state.history_a.keys.shape == (2, 2, 4, 8)
    assert state.history_b.keys.shape == (2, 2, 4, 8)
    assert state.literal.keys.shape[1] == 1
    assert len(state.recurrent) == 4
    assert all(s.dtype == torch.float32 and s.shape == (2, 3, 2, 8, 8) for s in state.recurrent)
    saved = [s.clone() for s in state.recurrent]
    saved_facets, saved_keys = state.facets.clone(), state.history_b.keys.clone()
    first = predict(core, state)
    second = predict(core, state, root=torch.ones(2, 8))
    assert not torch.equal(first.facets, second.facets)
    for old, new in zip(saved, state.recurrent, strict=True):
        torch.testing.assert_close(old, new, rtol=0, atol=0)
    torch.testing.assert_close(saved_facets, state.facets, rtol=0, atol=0)
    torch.testing.assert_close(saved_keys, state.history_b.keys, rtol=0, atol=0)


def test_later_loss_reaches_earlier_actual_writers():
    core = FacetedCore(ModelConfig.tiny())
    earlier = core.ingest(torch.tensor([[65, 66, 67, 68, 69, 70, 71, 72]]), core.initial_state(1))
    earlier.history_a.keys.retain_grad()
    earlier.recurrent[0].retain_grad()
    later = core.ingest(torch.tensor([[73, 74, 75, 76]]), earlier)
    prediction = predict(core, later)
    loss = (prediction.facets * torch.randn_like(prediction.facets)).sum()
    loss.backward()
    assert earlier.history_a.keys.grad.norm() > 0
    assert earlier.recurrent[0].grad.norm() > 0
    assert core.frontend.bytes.weight.grad[65].norm() > 0


def test_private_carry_does_not_contaminate_factual_cache():
    core = FacetedCore(ModelConfig.tiny()).eval()
    state = core.ingest(torch.tensor([[1, 2, 3]]), core.initial_state(1))
    other = core.reconcile_carry(state, torch.ones(1, 32), torch.ones(1, 8))
    assert not torch.equal(state.carry, other.carry)
    a = core.ingest(torch.tensor([[4, 5, 6]]), state)
    b = core.ingest(torch.tensor([[4, 5, 6]]), other)
    torch.testing.assert_close(a.facets, b.facets, rtol=0, atol=0)
    torch.testing.assert_close(a.history_b.values, b.history_b.values, rtol=0, atol=0)


def test_exact_exposure_effort_and_checkpoint_recompute_accounting():
    core = FacetedCore(ModelConfig.tiny(checkpoint_stages=True))
    state = core.ingest(torch.randint(256, (1, 16)), core.initial_state(1))
    factual_rows = core.dispatch_metrics(reset=True)["p"]["routed_rows"].sum()
    assert factual_rows.item() == 6 * 3 * core.config.n_facets
    prediction = predict(core, state, effort=3)
    count = core.dispatch_metrics()["p"]["routed_rows"].sum().item()
    assert count == 10 * core.config.n_facets * 3 // 2
    prediction.facets.sum().backward()
    assert core.dispatch_metrics()["p"]["routed_rows"].sum().item() == count


def test_route_epochs_refresh_only_future_choices():
    core = FacetedCore(ModelConfig.tiny())
    state = core.ingest(torch.randint(256, (1, 11)), core.initial_state(1))
    prediction = core.begin_prediction(state, torch.zeros(1, 8))
    initial_program = prediction.program
    prediction = core.advance_prediction(prediction, 7)
    prediction = core.advance_prediction(prediction, 8, torch.randn(1, 32))
    assert prediction.program is initial_program
    prediction = core.advance_prediction(prediction, 10)
    refreshed = core.refresh_prediction(prediction, torch.randn(1, 32))
    assert refreshed.program.stages == (11, 12, 13, 14, 15, 16)
    assert refreshed.previous_programs == (initial_program,)
    assert refreshed.factual is state
    with pytest.raises(ValueError, match="once"):
        core.refresh_prediction(refreshed, torch.zeros(1, 32))
    refreshed = core.advance_prediction(refreshed, 15)
    route_context = refreshed.route_context.clone()
    refreshed = core.advance_prediction(refreshed, 16, torch.randn(1, 32))
    torch.testing.assert_close(route_context, refreshed.route_context)


def test_reset_duplicate_record_and_model_version_contracts():
    core = FacetedCore(ModelConfig.tiny())
    state = core.ingest(torch.randint(256, (2, 7)), core.initial_state(2))
    with pytest.raises(ValueError, match="duplicate"):
        core.ingest(torch.randint(256, (2, 7)), state, record_ids=state.last_record_ids)
    reset = core.reset_state(state, torch.tensor([True, False]), torch.tensor([9, 10]))
    assert reset.records_seen.tolist() == [0, 1]
    assert reset.episode_ids.tolist() == [9, 0]
    assert not reset.history_a.mask[0].any()
    torch.testing.assert_close(reset.recurrent[0][1], state.recurrent[0][1])
    core.set_model_version(1)
    with pytest.raises(RuntimeError, match="Stale cache"):
        core.begin_prediction(state, torch.zeros(2, 8))


def test_bf16_and_all_served_core_paths_are_differentiable():
    core = FacetedCore(ModelConfig.tiny()).bfloat16()
    ids = torch.randint(256, (1, 15))
    control = core.encode_control(ids)
    state = core.ingest(ids, core.initial_state(1))
    state = core.reconcile_carry(
        state, state.facets.mean(1), torch.ones(1, 8, dtype=torch.bfloat16)
    )
    prediction = core.begin_prediction(state, torch.ones(1, 8, dtype=torch.bfloat16), control)
    for stage in (7, 8, 10, 15, 16):
        prediction = core.advance_prediction(
            prediction, stage, torch.ones(1, 32, dtype=torch.bfloat16)
        )
    rows = core.apply_u(prediction.facets, prediction, 16)
    rows = core.lexical_refine(rows)
    loss = (rows.float() * torch.randn_like(rows.float())).mean() + prediction.routing_aux_loss
    loss.backward()
    missing = [name for name, parameter in core.named_parameters() if parameter.grad is None]
    assert missing == []
    assert all(torch.isfinite(p.grad).all() for p in core.parameters())
    assert state.recurrent[0].dtype == torch.float32


def test_real_owner_counts_and_shared_banks():
    config = ModelConfig.tiny()
    core = FacetedCore(config)
    audit = audit_parameters(core)
    assert audit["total_parameters"] == sum(p.numel() for p in core.parameters())
    assert (
        sum(p.numel() for p in core.p_bank.parameters())
        == 3 * config.p_experts * config.pair_dim * config.p_hidden
    )
    assert len({id(stage.always.gate_up.weight) for stage in core.stages}) == 16
    assert sum(name == "u_always.gate_up.weight" for name, _ in core.named_parameters()) == 1
    model = torch.nn.Module()
    model.first = core.u_always
    model.alias = core.u_always
    alias_audit = audit_parameters(model)
    assert alias_audit["aliases"] == 2


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
