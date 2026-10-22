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
    for owner in model.core.adapters.owners.values():
        torch.nn.init.normal_(owner.up, std=strength)
    return model


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
        raw = on_next.raw
        ids, _ = multi.tracks._track_indices(on_next, ("a",))
        assert torch.equal(model.core.adapters.realization(state.adapters, ids, raw), raw)


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


def test_gradients_reach_every_owners_experts_and_routers(monkeypatch):
    monkeypatch.setattr("megatoken.model.transducer.ADAPTER_DROPOUT", 0.0)
    model = memory_model(strength=0.05).train()
    multi, state, _, _ = looked_up(model, training=True)
    assert float(state.adapters.strength.detach()) > 0
    state = observe(multi, state)
    predicted = multi.tracks.predict_group(state, tracks=("a",))
    ids, _ = multi.tracks._track_indices(state, ("a",))
    realized = model.core.adapters.realization(state.adapters, ids, state.raw)
    (state.final.sum() + predicted.square().sum() + realized.square().sum()).backward()
    for name, owner in model.core.adapters.owners.items():
        for parameter_name, parameter in owner.named_parameters():
            assert parameter.grad is not None, (name, parameter_name)
            assert parameter.grad.abs().sum() > 0, (name, parameter_name)


def test_each_owner_adapts_its_own_sites_and_only_its_own():
    # C's adapters change what C writes, not P's prediction from a fixed state, and so on.
    with torch.no_grad():
        model = memory_model().eval()
        multi, state, _, _ = looked_up(model)
        state = observe(multi, state)
        baseline = multi.tracks.predict_group(state, tracks=("a",))
        for name in OWNERS:
            owner = model.core.adapters.owners[name]
            saved = owner.up.clone()
            owner.up.zero_()
            changed = not torch.equal(baseline, multi.tracks.predict_group(state, tracks=("a",)))
            owner.up.copy_(saved)
            assert changed == (name == "p")


def test_global_router_balance_covers_the_adapter_routers(tmp_path):
    torch.set_num_threads(1)
    model = tiny_model()
    config = train_config(tmp_path, "recurrent")
    batch = grouped_batch(model, config)
    routers = balance_routers(model.core)
    assert {"p", "u", "adapter_c", "adapter_p", "adapter_r"} == routers.keys()
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
    topk = model.config.adapter_topk
    for name in OWNERS:
        count = diagnostics[f"balance_adapter_{name}_assignments"]
        assert count > 0 and count % topk == 0
        assert diagnostics[f"balance_adapter_{name}_loss"] > 0
        owner = model.core.adapters.owners[name]
        # The balance reaches only the controller; the experts see the task loss alone.
        assert owner.router.weight.grad.abs().sum() > 0
        assert owner.down.grad is None and owner.up.grad is None


def test_adapters_are_ordinary_parameters_of_every_owner_and_sync_path():
    model = memory_model()
    core = model.core
    adapters = {id(p): name for name, p in core.adapters.named_parameters()}
    for name, group in (
        ("c", core.central_parameters()),
        ("p", core.predictor_parameters()),
        ("r", core.realizer_parameters()),
    ):
        owned = {id(p) for p in core.adapters.owners[name].parameters()}
        assert owned <= {id(p) for p in group}
        others = set(adapters) - owned
        assert not others & {id(p) for p in group}
    # Not part of the frozen media bridge, so every DiLoCo worker trains them alike.
    assert not set(adapters) & {id(p) for p in model.media_bridge_parameters()}
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
    owner = model.core.adapters.owners["c"]
    assert tuple(owner.down.shape) == (4, 2, small.facet_dim)
    with pytest.raises(ValueError, match="adapter_topk"):
        ModelConfig.tiny(adapter_experts=1)
