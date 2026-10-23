from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from megatoken.model.expert_pools import (
    RoutedConv1d,
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
        router_dim=4,
        router_experts=3,
        max_tracks=4,
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
    assert report["routing"]["router_bank"]["experts"] == config().router_experts
    assert report["routing"]["router_bank"]["input_dim"] == config().router_dim
    assert report["routing"]["pool_selector_count"] == 1


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
    weights = torch.rand(2, 3, 2)
    flattened = flatten_expert_scope(weights)
    assert flattened.shape == (6, 2)
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
    assert router is site.router
    assert route.expert_ids.shape == (2, 3)
    probabilities = site.balance_probabilities(route.router_context, route.router_stage)
    assert probabilities.shape == (2, site.experts)
    assert torch.allclose(probabilities.sum(-1), torch.ones(2))


def test_conv1d_kernel_taps_share_linear_pool_and_one_palette():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv1d(4, 8, kernel_size=3, bias=False)
            self.projection = nn.Linear(4, 8, bias=False)

    model = Model()
    pools = ShapePools(config())
    pools.bind(model, "conv_test")
    assert isinstance(model.conv, RoutedConv1d)
    assert isinstance(model.projection, RoutedLinear)
    assert model.conv._bank is model.projection._bank

    # The production frontend pads two positions on the left before a valid
    # convolution. Its routed replacement must retain the original sequence
    # length and channel-first output shape.
    x = torch.randn(2, 4, 7, requires_grad=True)
    pools.metrics(reset=True)
    with pools.routing_epoch(x.transpose(1, 2).mean(dim=1), role="e"):
        y = model.conv(torch.nn.functional.pad(x, (2, 0)))
        z = model.projection(torch.randn(2, 5, 4))
    assert y.shape == (2, 8, 7)
    assert z.shape == (2, 5, 8)
    (y.square().mean() + z.square().mean()).backward()
    assert pools.pools[next(iter(pools.pools))].weight.grad is not None
    metrics = pools.metrics()
    assert metrics["router_evaluations"] == 1
    assert metrics["palette_evaluations"] == 1
    # 3 convolution taps x 2 x 7 rows, plus the shared projection's 2 x 5.
    assert metrics["expert_rows"] == 52
    assert pools.manifest(model)["coverage"]["unconverted"] == ()


def test_conv1d_covers_stride_dilation_and_explicit_padding_geometry():
    module = nn.Sequential(nn.Conv1d(4, 8, kernel_size=3, stride=2, dilation=2, padding=1))
    pools = ShapePools(config())
    pools.bind(module, "conv_geometry")
    assert module[0](torch.randn(2, 4, 9)).shape == (2, 8, 4)


def test_routing_subset_reuses_per_example_palette_and_adapter_scope():
    module = nn.Sequential(nn.Linear(4, 4, bias=False))
    pools = ShapePools(config())
    pools.bind(module, "subset")
    with torch.no_grad():
        module[0]._bank.adapter_up.fill_(0.01)
    context = torch.randn(3, 4)
    rows = torch.randn(3, 5, 4)
    weights = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [0.25, 0.75]],
        requires_grad=True,
    )
    with expert_scope(weights):
        with pools.routing_epoch(context, role="R") as plan:
            batched = module(rows)
            palette_evaluations = pools.palette_evaluations
            pool_name = module[0]._pool_name
            batch_palette = plan.palette_cache[pool_name][0].clone()
            pieces = []
            for index in range(rows.shape[0]):
                with pools.routing_subset([index]) as subset:
                    assert current_adapter_weights(rows[index : index + 1]).shape == (1, 5, 2)
                    torch.testing.assert_close(
                        subset.palette_cache[pool_name][0],
                        batch_palette[index : index + 1],
                        rtol=0,
                        atol=0,
                    )
                    pieces.append(module(rows[index : index + 1]))
            sequential = torch.cat(pieces, dim=0)
            assert pools.router_evaluations == 1
            assert pools.palette_evaluations == palette_evaluations == 1
            assert current_adapter_weights(rows).shape == (3, 5, 2)
    torch.testing.assert_close(sequential, batched, rtol=1e-6, atol=2e-7)
    sequential.sum().backward()
    assert weights.grad is not None and weights.grad.abs().sum() > 0


def test_grouped_conv1d_fails_explicitly_instead_of_leaving_dense_weights():
    module = nn.Sequential(nn.Conv1d(4, 8, kernel_size=3, groups=2))
    pools = ShapePools(config())
    with pytest.raises(NotImplementedError, match="grouped Conv1d"):
        pools.bind(module, "grouped_conv")


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


def test_checkpoint_replay_clones_route_palette_snapshot_without_counter_replay():
    from megatoken.model.expert_pools import _CURRENT_ROUTING_PLAN, _EXPERT_REPLAY

    module = nn.Sequential(nn.Linear(4, 4, bias=False))
    pools = ShapePools(config())
    pools.bind(module, "checkpoint_palette")
    context = torch.randn(2, 4)
    with pools.routing_epoch(context, adapter_weights=torch.ones(2, 2)) as plan:
        rows = torch.randn(2, 3, 4)
        first = module(rows)
        assignments = plan.palette_cache[next(iter(plan.palette_cache))][0].clone()
        before = (pools.router_evaluations, pools.palette_evaluations, pools.expert_rows)
        _, recompute = checkpoint_context_fn()
        with recompute:
            replay = _CURRENT_ROUTING_PLAN.get()
            assert replay is not plan
            assert replay.palette_cache.keys() == plan.palette_cache.keys()
            replay_output = module(rows)
            assert torch.equal(
                replay.palette_cache[next(iter(replay.palette_cache))][0], assignments
            )
            assert _EXPERT_REPLAY.get()
        assert torch.equal(first, replay_output)
        assert (pools.router_evaluations, pools.palette_evaluations, pools.expert_rows) == before


def test_late_pool_and_site_binding_follow_existing_device_and_dtype():
    pools = ShapePools(config()).to(dtype=torch.float64)
    module = nn.Sequential(nn.Linear(4, 4))
    pools.bind(module, "late")
    assert next(pools.pools.parameters()).dtype == torch.float64
    assert module[0].router.weight.dtype == torch.float64
    assert module[0].bias.dtype == torch.float64


def test_shared_hierarchical_routing_computes_once_and_caches_per_pool():
    module = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4), nn.Linear(4, 8))
    pools = ShapePools(config())
    pools.bind(module, "hierarchy")
    context = torch.randn(2, 4)
    x = torch.randn(2, 7, 4)
    with pools.routing_epoch(context, track_ids=torch.tensor([0, 1]), role="C", stage=2):
        for layer in module:
            x = layer(x)
    assert pools.router_evaluations == 1
    # The two 4x4 sites share one linear pool; the 4x8 projection uses another.
    assert pools.palette_evaluations == 2
    assert pools.expert_rows == 2 * 7 * 3
    assert all(
        not name.startswith("router.")
        for site in (module[0], module[1], module[2])
        for name, _ in site.named_parameters()
    )
    x.sum().backward()
    assert pools.router_bank.gate_up.grad is not None
    assert (
        pools.pool_codes[next(key for key in pools.pool_codes if "linear_4_4" in key)].grad
        is not None
    )


def test_router_fold_reads_semantics_not_addresses_and_controller_uses_shared_experts():
    from megatoken.model.config import RESERVED_TRACKS

    pools = ShapePools(config())
    context = torch.ones(2, 64)
    first = pools.router_features(context, track_ids=torch.tensor([0, 1]), role="R")
    second = pools.router_features(context, track_ids=torch.tensor([1, 0]), role="R")
    assert first.shape == (2, 4)
    torch.testing.assert_close(first, second)
    unknown = pools.router_features(context, track_ids=torch.tensor([-1, -1]), role="R")
    assert unknown.shape == first.shape
    last_reserved = config().max_tracks + len(RESERVED_TRACKS) - 1
    assert (
        pools.router_features(context, track_ids=torch.tensor([last_reserved, 0]), role="R").shape
        == first.shape
    )
    torch.testing.assert_close(
        pools.router_features(context, track_ids=torch.tensor([900, 301]), role="R"), first
    )
    purposes = torch.randn(2, pools.router_semantic_projection.in_features)
    assert not torch.allclose(
        pools.router_features(context, track_semantics=purposes, role="R"), first
    )
    output = pools.controller_features(context, track_ids=torch.tensor([0, 1]), role="R")
    assert output.shape == (2, 4)
    assert pools.router_evaluations == 1


def test_standalone_rank_two_calls_keep_one_context_per_batch_and_adapt_router():
    module = nn.Sequential(nn.Linear(4, 4, bias=False))
    pools = ShapePools(config())
    pools.bind(module, "standalone")
    recorder = type(
        "Recorder", (), {"capture": lambda self, selector, route: setattr(self, "route", route)}
    )()
    module[0].balance_recorder = recorder
    weights = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    rows = torch.randn(2, 4, requires_grad=True)
    with expert_scope(weights):
        output = module(rows)
    assert output.shape == rows.shape
    assert recorder.route.router_context.shape[0] == 2
    output.sum().backward()
    assert pools.router_bank.gate_up_adapter_up.grad is not None
