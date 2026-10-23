from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from megatoken.model.expert_pools import (
    RoutedLinear,
    RoutedSwiGLU,
    ShapePools,
    checkpoint_context_fn,
    current_adapter_weights,
    expert_scope,
    flatten_expert_scope,
    iter_sites,
)


def config():
    return SimpleNamespace(
        facet_dim=4,
        projection_experts=3,
        p_experts=5,
        u_experts=3,
        adapter_experts=2,
        adapter_rank=2,
        adapter_topk=1,
        expert_backend="torch",
    )


def test_bind_shares_pool_without_registering_duplicate_references():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(3, 5)
            self.b = nn.Linear(3, 5)
            self.norm = nn.LayerNorm(3)

    model = Model()
    pools = ShapePools(config())
    pools.bind(model, "test")
    pools.bind(model, "test")
    assert isinstance(model.a, RoutedLinear)
    assert isinstance(model.b, RoutedLinear)
    assert model.a._bank is model.b._bank
    names = [name for name, _ in model.named_modules()]
    assert not any("_bank" in name for name in names)
    assert len(list(pools.pools.parameters())) == len(
        {id(parameter) for parameter in pools.pools.parameters()}
    )
    assert isinstance(model.norm, nn.LayerNorm)
    report = pools.manifest()
    assert len(report["sites"]) == 2
    assert len(report["pools"]) == 1


def test_routed_linear_preserves_shape_and_routes_gradients():
    module = nn.Sequential(nn.Linear(3, 5, bias=False))
    pools = ShapePools(config())
    pools.bind(module, "linear")
    x = torch.randn(2, 4, 3, requires_grad=True)
    y = module(x)
    assert y.shape == (2, 4, 5)
    y.square().sum().backward()
    site = module[0]
    assert site.router.weight.grad is not None
    assert site._bank.weight.grad is not None


def test_unbatched_vector_routes_as_one_row():
    module = nn.Sequential(nn.Linear(4, 3, bias=False))
    pools = ShapePools(config())
    pools.bind(module, "single")
    result = module(torch.randn(4))
    assert result.shape == (3,)


def test_zero_initialized_linear_retains_its_initial_bias_and_can_open():
    linear = nn.Linear(4, 3)
    with torch.no_grad():
        linear.weight.zero_()
        linear.bias.copy_(torch.tensor([1.0, -2.0, 0.5]))
    module = nn.Sequential(linear)
    pools = ShapePools(config())
    pools.bind(module, "zero_head")
    site = module[0]
    assert site.output_gain.item() == 0
    assert torch.equal(site.bias, torch.tensor([1.0, -2.0, 0.5]))
    output = module(torch.randn(2, 4))
    assert torch.equal(output, site.bias[None].expand_as(output))
    output.sum().backward()
    assert site.output_gain.grad is not None
    assert site._bank.weight.grad is not None


def test_swiglu_sites_share_width_bucket():
    from megatoken.model.layers import SwiGLU

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = SwiGLU(3, 6)
            self.b = SwiGLU(4, 7)

    model = Model()
    pools = ShapePools(config())
    pools.bind(model, "ffn")
    assert isinstance(model.a, RoutedSwiGLU)
    assert isinstance(model.b, RoutedSwiGLU)
    assert model.a._bank is model.b._bank
    assert model.a(torch.randn(2, 3)).shape == (2, 3)


def test_scope_expands_per_event_rows_and_rejects_cross_session_shapes():
    module = nn.Sequential(nn.Linear(4, 4, bias=False))
    pools = ShapePools(config())
    pools.bind(module, "scope")
    weights = torch.tensor([[[1.0, 0.0], [0.0, 1.0]], [[0.3, 0.7], [0.6, 0.4]]])
    x = torch.randn(2, 2, 3, 4)
    with expert_scope(weights):
        y = module(x)
        assert current_adapter_weights(x).shape == (2, 2, 3, 2)
    assert y.shape == x.shape
    assert current_adapter_weights(x) is None
    with expert_scope(weights):
        with pytest.raises(ValueError, match="does not align"):
            module(torch.randn(3, 4, 4))


def test_explicit_scope_flatten_helper_matches_flattened_events():
    weights = torch.rand(2, 3, 4)
    flattened = flatten_expert_scope(weights)
    assert flattened.shape == (6, 4)
    module = nn.Sequential(nn.Linear(4, 4, bias=False))
    pools = ShapePools(config())
    pools.bind(module, "flat")
    rows = torch.randn(6, 5, 4)
    with expert_scope(flattened):
        assert module(rows).shape == rows.shape


def test_scope_restores_after_exception_and_deepcopy_rebinds_bank():
    model = nn.Sequential(nn.Linear(4, 4, bias=False))
    pools = ShapePools(config())
    pools.bind(model, "copy")
    original_bank = model[0]._bank
    cloned = copy.deepcopy(nn.ModuleDict({"pools": pools, "model": model}))
    assert cloned["model"][0]._bank is cloned["pools"].pools[next(iter(cloned["pools"].pools))]
    assert cloned["model"][0]._bank is not original_bank
    with pytest.raises(RuntimeError):
        with expert_scope(torch.ones(1, 2)):
            raise RuntimeError("stop")
    assert module_scope_is_empty()


def module_scope_is_empty():
    from megatoken.model.expert_pools import _CURRENT_ADAPTER_WEIGHTS

    return _CURRENT_ADAPTER_WEIGHTS.get() is None


def test_gru_cell_is_exposed_as_two_routed_gate_projections():
    model = nn.GRUCell(3, 4)
    pools = ShapePools(config())
    pools.bind(model, "gru")
    # The supplied module is itself a leaf, so bind it through a parent.
    if isinstance(model, nn.GRUCell):
        parent = nn.Module()
        parent.cell = model
        pools.bind(parent, "gru")
        model = parent.cell
    x, h = torch.randn(2, 3), torch.randn(2, 4)
    result = model(x, h)
    assert result.shape == h.shape


def test_shared_ffn_accessor_and_site_iteration_are_not_registered_aliases():
    module = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
    pools = ShapePools(config())
    pools.bind(module, "iter")
    sites = list(iter_sites(module))
    assert len(sites) == 2
    assert sites[0][1]._bank is sites[1][1]._bank
    bank = pools.ffn_bank(4)
    assert bank is pools.ffn_bank(3)


def test_site_exposes_replayable_balance_route():
    module = nn.Sequential(nn.Linear(4, 4, bias=False))
    pools = ShapePools(config())
    pools.bind(module, "balanced")
    site = module[0]

    class Recorder:
        def __init__(self):
            self.routes = []

        def capture(self, router, route):
            self.routes.append((router, route))

    recorder = Recorder()
    site.balance_recorder = recorder
    module(torch.randn(2, 3, 4))
    assert len(recorder.routes) == 1
    router, route = recorder.routes[0]
    assert router is site
    assert route.expert_ids.shape == (6, 1)
    probabilities = site.balance_probabilities(route.router_context, route.router_stage)
    assert probabilities.shape == (6, site.experts)
    assert torch.allclose(probabilities.sum(-1), torch.ones(6))


def test_checkpoint_replay_context_restores_scope_and_marks_replay():
    from megatoken.model.expert_pools import _EXPERT_REPLAY

    weights = torch.ones(2, 2)
    with expert_scope(weights):
        forward, recompute = checkpoint_context_fn()
    with forward:
        assert current_adapter_weights(torch.randn(2, 3, 4)).shape == (2, 3, 2)
        assert not _EXPERT_REPLAY.get()
    with recompute:
        assert current_adapter_weights(torch.randn(2, 3, 4)).shape == (2, 3, 2)
        assert _EXPERT_REPLAY.get()
    assert not _EXPERT_REPLAY.get()


def test_late_pool_and_site_binding_follow_existing_device_and_dtype():
    pools = ShapePools(config()).to(dtype=torch.float64)
    module = nn.Sequential(nn.Linear(4, 4))
    pools.bind(module, "late")
    assert next(pools.pools.parameters()).dtype == torch.float64
    assert module[0].router.weight.dtype == torch.float64
    assert module[0].bias.dtype == torch.float64
