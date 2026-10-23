"""The knowledge-routed adapter mixture: lookups choose it, every C layer, P stage and
realization applies it, and it trains and syncs like the rest of the model."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from media_support import observe_text
from test_meridian_training import grouped_batch, tiny_model, train_config

from megatoken.contracts import ROLE_TO_ID
from megatoken.generation.codec import ByteCodec
from megatoken.loading import build_model, parameter_report
from megatoken.model.adapters import ADAPTER_DROPOUT, OWNERS
from megatoken.model.config import ModelConfig
from megatoken.model.expert_pools import expert_scope, iter_sites
from megatoken.model.multitrack import MultiTrackGenerator, ObservedEvent
from megatoken.model.transducer import metadata_payload, real_boundary
from megatoken.training.diloco_protocol import plan_fragments
from megatoken.training.distributed import DistributedContext
from megatoken.training.losses import LossLedger
from megatoken.training.meridians import forward_meridian_batch
from megatoken.training.optimizer import MasterAdamW
from megatoken.training.router_balance import RouterBalanceWindow, balance_routers
from megatoken.training.transducer import Meridian, TransducerTrainer
from megatoken.transducer_service import TransducerInferenceEngine


def memory_model(edges: int = 12, *, strength: float = 0.3):
    """A tiny model whose bank can find something, with adapters strong enough to see."""
    torch.manual_seed(0)
    model = build_model(ModelConfig.tiny(), ByteCodec.byte_fallback())
    g = model.transducer.bank.geometry
    if edges:
        model.transducer.bank.add_edges(
            departure=torch.randn(edges, g.location_dim),
            key=torch.randn(edges, g.query_dim),
            value=torch.randn(edges, g.value_dim),
            destination=torch.randn(edges, g.location_dim),
            bias=torch.zeros(edges),
            write_mass=torch.ones(edges),
            step=0,
        )
    # Adapter up-projections are zero-initialized in production so the newly
    # installed model begins as the base expert model. Tests that need to see
    # the correction explicitly make the shared pool factors nonzero.
    for name, parameter in model.core.expert_pools.named_parameters():
        if "adapter_up" in name:
            torch.nn.init.normal_(parameter, std=strength)
    return model


def test_adapter_controllers_share_one_compact_router_feature():
    model = tiny_model()
    core = model.core
    value = torch.randn(core.config.transducer_value_rows * core.config.facet_dim)
    before = core.expert_pools.router_evaluations
    mixture = core.adapters.select(value, torch.tensor(0.75))
    assert mixture.logits.shape == (1, len(OWNERS), core.config.adapter_experts)
    assert core.expert_pools.router_evaluations == before + 1
    assert all(
        owner.router.in_features == core.config.router_dim
        for owner in core.adapters.owners.values()
    )


def looked_up(model, *, training: bool = False):
    """Metadata, then one lookup: the session as the first answer would find it."""
    multi = MultiTrackGenerator(model)
    state, agenda = model.transducer.begin(
        multi, multi.tracks.initial_state(1), metadata_payload({})
    )
    assert state.adapters is None
    state, found = model.transducer.boundary(multi, state, agenda, training=training)
    return multi, state, agenda, found


def observe(multi, state, track: str = "a", payload: bytes = b"hello"):
    return observe_text(multi.tracks, state, [payload], tracks=(track,))


def adapted_realizer_site(model, multi, state):
    """Exercise one actual R site with its retrieval-conditioned adapter plan."""
    sites = list(iter_sites(model.decoder))
    assert sites
    _, site = sites[0]
    track_ids, _ = multi.tracks._track_indices(state, ("a",))
    plan = model.core.adapters.plan("r", state.adapters, track_ids)
    weights = plan.weights[0, :, 0]
    values = torch.randn(1, site.input_dim, device=state.raw.device)
    with expert_scope(weights):
        return site(values)


def test_no_adapters_apply_without_a_lookup_or_when_it_found_nothing():
    with torch.no_grad():
        model = memory_model(edges=0).eval()
        multi, state, _, found = looked_up(model)
        # An empty bank answers with the sink alone: the mixture exists but has no strength.
        assert float(found.sink_weight) == 1.0 and float(state.adapters.strength) == 0.0
        off = replace(state, adapters=None)
        on_next, off_next = observe(multi, state), observe(multi, off)
        for left, right in (
            (on_next.final, off_next.final),
            (on_next.history.keys, off_next.history.keys),
            (on_next.history.values, off_next.history.values),
        ):
            assert torch.equal(left, right)
        for track in (("a",), ("a", "b")):
            on = multi.tracks.predict_group(state, tracks=track)
            assert torch.equal(on, multi.tracks.predict_group(off, tracks=track))
        plain_site = list(iter_sites(model.decoder))[0][1]
        values = torch.randn(1, plain_site.input_dim)
        track_ids, _ = multi.tracks._track_indices(on_next, ("a",))
        weights = model.core.adapters.plan("r", on_next.adapters, track_ids).weights[0, :, 0]
        with expert_scope(weights):
            applied = plain_site(values)
        torch.testing.assert_close(applied, plain_site(values))


def test_the_mixture_follows_each_lookup_and_persists_across_meridians_and_tracks():
    with torch.no_grad():
        model = memory_model().eval()
        multi, state, agenda, found = looked_up(model)
        first = state.adapters
        assert float(found.sink_weight) < 1.0 and float(first.strength) > 0.0
        off = replace(state, adapters=None)
        # Every track and every meridian after the lookup runs under it.
        for track in ("a", "b", "c"):
            state, off = observe(multi, state, track), observe(multi, off, track)
            assert state.adapters is first
            assert not torch.equal(state.final, off.final)
            for tracks in (("a",), ("b", "c")):
                on = multi.tracks.predict_group(state, tracks=tracks)
                assert not torch.equal(on, multi.tracks.predict_group(off, tracks=tracks))
        state, second_found = model.transducer.boundary(multi, state, agenda)
        second = state.adapters
        assert second is not first and second_found is not None
        assert not torch.equal(second.logits, first.logits)
        assert observe(multi, state).adapters is second


def test_serving_and_training_apply_the_same_mixture():
    model = memory_model().eval()
    multi = MultiTrackGenerator(model)
    system, user, tool = (ROLE_TO_ID[name] for name in ("system", "user", "tool"))
    script = [
        ("system", system),
        ("tool", tool),
        ("user", user),
        ("tool", tool),
        ("system", system),
    ]
    engine = TransducerInferenceEngine(multi)
    session = engine.new_session({"task": "chat"})
    for track, role in script:
        engine.observe(session, [ObservedEvent(track, "text", b"x", role_id=role)])
    trainer = TransducerTrainer(model.transducer)
    meridians = [
        Meridian((track,), (role,), ("text",), real_boundary([role], [True]))
        for track, role in script
    ]
    sequence = trainer.sequence(meridians, step=0)
    state = sequence.begin(multi, multi.tracks.initial_state(1), metadata_payload({"task": "chat"}))
    with torch.no_grad():
        for index, (track, role) in enumerate(script):
            state = sequence.before_meridian(multi, state, index)
            state = multi.tracks.observe_group(
                state, [ObservedEvent(track, "text", b"x", role_id=role)]
            )
        assert sequence.lookups == session.agenda.lookups == 4
        served, trained = session.state, state
        assert float(trained.adapters.strength) > 0
        torch.testing.assert_close(served.adapters.logits, trained.adapters.logits)
        torch.testing.assert_close(served.adapters.strength, trained.adapters.strength)
        torch.testing.assert_close(served.final, trained.final)
        torch.testing.assert_close(served.history.keys, trained.history.keys)
        torch.testing.assert_close(
            multi.tracks.predict_group(served, tracks=("a", "b")),
            multi.tracks.predict_group(trained, tracks=("a", "b")),
        )


def test_the_mixture_strength_is_what_the_lookup_found_and_dropout_is_seeded():
    model = memory_model()
    native = model.transducer
    found = SimpleNamespace(
        value=torch.randn(native.bank.geometry.value_dim), sink_weight=torch.tensor(0.5)
    )

    def strength(overlay, training=False, generator=None):
        mixture = native.mixture(found, overlay=overlay, training=training, generator=generator)
        return float(mixture.strength)

    # One minus the sink's share; a hindsight overlay is knowledge, not sink.
    assert strength(0.0) == 0.5 and strength(0.5) == 0.75 and strength(1.0) == 1.0

    def drops(seed):
        generator = torch.Generator().manual_seed(seed)
        return [strength(0.0, True, generator) == 0.0 for _ in range(1000)]

    assert drops(7) == drops(7) and drops(7) != drops(8)
    assert 0.05 < sum(drops(7)) / 1000 < 0.15 and ADAPTER_DROPOUT == 0.1
    assert all(strength(0.0, False, torch.Generator().manual_seed(7)) for _ in range(100))


def test_gradients_reach_every_controller_and_shared_pool_adapter(monkeypatch):
    monkeypatch.setattr("megatoken.model.transducer.ADAPTER_DROPOUT", 0.0)
    model = memory_model(strength=0.05).train()
    multi, state, _, _ = looked_up(model, training=True)
    assert float(state.adapters.strength.detach()) > 0
    state = observe(multi, state)
    predicted = multi.tracks.predict_group(state, tracks=("a",))
    # A routed R site exercises the real expert-pool adapter path while keeping
    # this test independent of sampled text validity and decode no-grad wrappers.
    realized = adapted_realizer_site(model, multi, state)
    (state.final.sum() + predicted.square().sum() + realized.float().sum()).backward()
    for name, owner in model.core.adapters.owners.items():
        for parameter_name, parameter in owner.named_parameters():
            assert parameter.grad is not None, (name, parameter_name)
            assert parameter.grad.abs().sum() > 0, (name, parameter_name)
    adapter_factors = [
        parameter
        for name, parameter in model.core.expert_pools.named_parameters()
        if "adapter_" in name and "adapter_up" in name
    ]
    assert adapter_factors and any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in adapter_factors
    )


def test_c_p_and_r_controllers_route_only_their_role_sites():
    # The controller stays role-local while the adapter factors are globally
    # shared. Changing C/R preferences cannot alter a prediction from fixed
    # state; changing P preferences can.
    with torch.no_grad():
        model = memory_model().eval()
        multi, state, _, _ = looked_up(model)
        state = observe(multi, state)
        for name in ("c", "p", "r"):
            owner = model.core.adapters.owners[name]
            saved = {
                "bias": owner.bias.clone(),
                "router": owner.router.weight.clone(),
                "track": owner.track_preference.weight.clone(),
                "stage": owner.stage_preference.weight.clone(),
                "topk": owner.topk,
            }
            owner.topk = 1
            owner.router.weight.zero_()
            owner.track_preference.weight.zero_()
            owner.stage_preference.weight.zero_()
            owner.bias.zero_()
            owner.bias[0] = 10.0
            baseline = multi.tracks.predict_group(state, tracks=("a",))
            owner.bias[0] = -10.0
            owner.bias[1] = 10.0
            changed = not torch.equal(baseline, multi.tracks.predict_group(state, tracks=("a",)))
            owner.bias.copy_(saved["bias"])
            owner.router.weight.copy_(saved["router"])
            owner.track_preference.weight.copy_(saved["track"])
            owner.stage_preference.weight.copy_(saved["stage"])
            owner.topk = saved["topk"]
            assert changed == (name == "p")


def test_global_router_balance_covers_the_adapter_routers(tmp_path):
    torch.set_num_threads(1)
    model = tiny_model()
    config = train_config(tmp_path, "recurrent")
    batch = grouped_batch(model, config)
    routers = balance_routers(model)
    core_routers = {"p", "u", *(f"adapter_{name}" for name in OWNERS)}
    assert core_routers <= routers.keys()
    selector_names = [name for name in routers if name.startswith("shape_")]
    assert "shape_bootstrap" in selector_names
    assert any(name.startswith("shape_pool_") for name in selector_names)
    assert len({id(routers[name]) for name in selector_names}) == len(selector_names)
    window = RouterBalanceWindow()
    for router in routers.values():
        router.balance_recorder = window
    ledger = LossLedger(batch.counts, batch.counts, config.loss_weights)
    forward_meridian_batch(
        model,
        batch,
        ledger,
        config,
        torch.device("cpu"),
        0,
        differentiate=False,
        transducer=TransducerTrainer(model.transducer),
    )
    ledger.finish()
    model.zero_grad()
    diagnostics = window.backward(DistributedContext(), routers=routers, coefficient=1.0)
    assert selector_names and all(
        f"balance_{name}_assignments" in diagnostics for name in selector_names
    )
    assert sum(diagnostics[f"balance_{name}_assignments"] for name in selector_names) > 0
    topk = model.config.adapter_topk
    for name in OWNERS:
        count = diagnostics[f"balance_adapter_{name}_assignments"]
        assert count > 0 and count % topk == 0
        assert diagnostics[f"balance_adapter_{name}_loss"] > 0
        owner = model.core.adapters.owners[name]
        # The balance reaches only the controller; shared adapter factors see
        # task loss, but no balance-only gradient.
        assert owner.router.weight.grad.abs().sum() > 0
    assert all(
        parameter.grad is None
        for name, parameter in model.core.expert_pools.named_parameters()
        if "adapter_" in name
    )


def test_adapters_are_ordinary_parameters_of_every_owner_and_sync_path():
    model = memory_model()
    core = model.core
    adapters = {id(p): name for name, p in core.adapters.named_parameters()}
    for name, group in (
        ("e", core.encoder_parameters()),
        ("c", core.central_parameters()),
        ("p", core.predictor_parameters()),
        ("r", core.realizer_parameters()),
    ):
        owned = {id(p) for p in core.adapters.owners[name].parameters()}
        assert owned <= {id(p) for p in group}
        others = set(adapters) - owned
        assert not others & {id(p) for p in group}
    # Controllers remain role-owned; pool factors are globally owned and are
    # deliberately frozen by the media-bridge stage.
    assert not set(adapters) & {id(p) for p in model.media_bridge_parameters()}
    assert {id(p) for p in core.shared_parameters()} & {
        id(p) for p in model.media_bridge_parameters()
    } == set()
    optimizer = MasterAdamW(model, 1e-3)
    try:
        names = {n for n in optimizer.masters if ".adapters." in n}
        assert len(names) == len(adapters)
        for name in names:
            rule = optimizer.rules[name]
            if "preference" in name:  # embedding tables: half rate, no decay
                assert (rule.lr_multiplier, rule.weight_decay) == (0.5, 0.0)
            elif name.endswith(".bias"):
                assert (rule.lr_multiplier, rule.weight_decay) == (1.0, 0.0)
            else:
                assert rule.lr_multiplier == 1.0 and rule.weight_decay > 0
        covered = dict.fromkeys(names, 0)
        for fragment in plan_fragments(optimizer.masters, target_bytes=1024):
            for part in fragment.slices:
                if part.name in covered:
                    covered[part.name] += part.length
        assert covered == {n: optimizer.masters[n].numel() for n in names}
        pool_names = {name for name in optimizer.masters if "expert_pools" in name}
        factor_names = {name for name in pool_names if "adapter_" in name}
        assert factor_names
        shared_covered = dict.fromkeys(factor_names, 0)
        for fragment in plan_fragments(optimizer.masters, target_bytes=1024):
            for part in fragment.slices:
                if part.name in shared_covered:
                    shared_covered[part.name] += part.length
        assert shared_covered == {name: optimizer.masters[name].numel() for name in factor_names}
    finally:
        optimizer.close()
    served = model.served_state_dict()
    assert {f"core.adapters.{n}" for n in core.adapters.state_dict()} <= served.keys()
    report = parameter_report(model)["owners"]
    assert all(owner["category"] == "served" for owner in report if ".adapters." in owner["name"])


def test_adapter_size_is_part_of_the_architecture_identity():
    small = ModelConfig.tiny(adapter_experts=4, adapter_rank=2)
    assert small.to_dict() != ModelConfig.tiny().to_dict()
    assert ModelConfig.from_dict(small.to_dict()) == small
    model = build_model(small, ByteCodec.byte_fallback())
    pool = model.core.expert_pools.ffn_bank(small.facet_dim)
    assert pool.adapter_experts == 4 and pool.adapter_rank == 2
    assert tuple(pool.gate_up_adapter_down.shape[:3]) == (
        pool.experts,
        4,
        2,
    )
    with pytest.raises(ValueError, match="adapter_topk"):
        ModelConfig.tiny(adapter_experts=1)
