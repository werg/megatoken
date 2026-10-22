"""v10 transducer integration: native lookups, hindsight, training, DiLoCo and serving."""

from __future__ import annotations

import pytest
import torch

from megatoken.generation.codec import ByteCodec
from megatoken.loading import build_model
from megatoken.memory.transducer import TransducerBank, TransducerGeometry
from megatoken.memory.transducer_growth import PROVISIONAL_BASE, TransducerGrowth
from megatoken.memory.transducer_optimizer import BlockScaleDescent
from megatoken.model.attention import AttentionFeatures, AttentionRead
from megatoken.model.config import RESERVED_TRACKS, RETRIEVAL_QUERY_TRACK, ModelConfig
from megatoken.model.multitrack import MultiTrackGenerator, ObservedEvent
from megatoken.model.transducer import OrdinaryEvents, metadata_payload
from megatoken.training.transducer import (
    BoundaryScheduler,
    HindsightSchedule,
    Meridian,
    TransducerTrainer,
    plan_boundaries,
)
from megatoken.training.transducer_diloco import (
    TransducerCoordinator,
    install_changes,
    learner_round,
)
from megatoken.training.transducer_sync import TransducerHost, TransducerSync, install_published

GEOMETRY = TransducerGeometry(
    location_dim=8,
    query_dim=16,
    value_dim=64,
    neighbors=4,
    selected=4,
    exploration=1,
    max_lookups=8,
    max_edges=64,
    byte_budget=64 * 10_000,
)


def tiny_model():
    torch.manual_seed(0)
    model = build_model(ModelConfig.tiny(), ByteCodec.byte_fallback())
    return model.train()


def give_gradient(bank: TransducerBank, edge: int, name: str, gradient: torch.Tensor) -> None:
    """As if a training lookup had selected ``edge`` and backward reached its master."""
    leaf = bank.masters.leaf(name, [edge], bank.decode(name, [edge]))
    leaf.grad = gradient.reshape(leaf.shape).float()


def add_random(bank: TransducerBank, count: int, *, step: int = 0, ids=None, provisional=False):
    g = bank.geometry
    return bank.add_edges(
        departure=torch.randn(count, g.location_dim),
        key=torch.randn(count, g.query_dim),
        value=torch.randn(count, g.value_dim),
        destination=torch.randn(count, g.location_dim),
        bias=torch.zeros(count),
        write_mass=torch.ones(count),
        step=step,
        ids=ids,
        provisional=provisional,
    )


def test_sink_logit_is_exact_with_one_time_per_slot():
    torch.manual_seed(0)
    read = AttentionRead(64, 4, 2, 16, sink=True)
    torch.nn.init.normal_(read.sink_key)
    torch.nn.init.normal_(read.sink_value)
    query = torch.randn(2, 5, 64)
    features = AttentionFeatures(torch.randn(2, 2, 7, 16), torch.randn(2, 2, 7, 16), None, None)
    times = torch.tensor([3, 9])
    per_row = read.read(query, features, temporal_positions=times)
    per_slot = read.read(query, features, temporal_positions=times[:, None].expand(2, 5))
    torch.testing.assert_close(per_row, per_slot, atol=1e-6, rtol=1e-5)


def test_reserved_tracks_never_use_user_slots():
    model = tiny_model()
    multi = MultiTrackGenerator(model)
    state = multi.tracks.initial_state(1)
    ids, registry = multi.tracks._track_indices(state, RESERVED_TRACKS)
    cap = model.config.max_tracks
    assert ids[0].tolist() == [cap + i for i in range(len(RESERVED_TRACKS))] and registry == ()


def test_native_lookup_hindsight_and_parameter_ownership():
    model = tiny_model()
    native = model.transducer
    names = [name for name, _ in model.named_parameters()]
    assert len(names) == len(set(names))
    assert not any(name.startswith(("transducer._core", "transducer.core")) for name in names)
    multi = MultiTrackGenerator(model)
    state = multi.tracks.initial_state(1)
    state, agenda = native.begin(multi, state, metadata_payload({"task": "chat"}))
    assert state.frontier_tracks == ("__metadata__",) and state.track_registry == ()
    state, first = native.boundary(multi, state, agenda, training=True)
    assert float(first.sink_weight.detach()) == 1.0 and agenda.lookups == 1
    assert state.frontier_tracks == ("__transducer_knowledge__",)
    width = model.config.facet_dim
    events = OrdinaryEvents(
        torch.randn(1, 3, model.config.n_facets, width),
        torch.ones(1, 3, dtype=torch.long),
        ("text",) * 3,
        ("a", "b", "a"),
        torch.tensor([[1, 1, 2]]),
        torch.tensor([[0, 0, 1]]),
    )
    hindsight = native.hindsight_pass(events, [0, 1, None], mode="group-inverse")
    assert hindsight.values.shape == (3, native.bank.geometry.value_dim)
    assert hindsight.sink_only.tolist() == [False, False, True]
    with pytest.raises(ValueError, match="not ordinary"):
        native.hindsight_pass(
            OrdinaryEvents(
                events.raw[:, :1],
                events.roles[:, :1],
                ("text",),
                (RETRIEVAL_QUERY_TRACK,),
                events.meridians[:, :1],
                events.groups[:, :1],
            ),
            [0],
            mode="full",
        )
    state, second = native.boundary(
        multi, state, agenda, training=True, hindsight=hindsight.values[0], overlay=0.5
    )
    (state.final.square().sum() + second.value.square().sum()).backward()
    for module in (native.query_realizer, native.readout, native.hindsight_realizer):
        assert any(p.grad is not None for p in module.parameters())
    model.eval()
    with pytest.raises(RuntimeError, match="serving path"):
        native.hindsight_pass(events, [0], mode="full")


def test_boundary_plan_starts_after_metadata_and_respects_real_boundaries_and_cap():
    plain = Meridian(("a",), (1,), ("text",), False)
    turn = Meridian(("a",), (2,), ("text",), True)
    lookups = plan_boundaries([plain] * 20, max_lookups=256, generator=None)
    assert lookups[0] == lookups[8] == lookups[16] == 1 and sum(lookups) == 3
    assert plan_boundaries([plain, turn, plain], max_lookups=256, generator=None) == [1, 0, 1]
    assert sum(plan_boundaries([turn] * 10, max_lookups=4, generator=None)) == 4
    sampled = plan_boundaries([plain] * 64, max_lookups=256, generator=torch.Generator())
    starts = [i for i, x in enumerate(sampled) if x]
    gaps = [b - a for a, b in zip(starts, starts[1:], strict=False)]
    assert all(4 <= gap <= 12 for gap in gaps)


def test_scheduler_defers_a_boundary_during_a_lookup_and_waits_after_cancel():
    scheduler = BoundaryScheduler()
    assert scheduler.first_metadata()
    assert not scheduler.ordinary_commit(real_boundary=True)  # outstanding: deferred
    assert scheduler.finish(committed=True)  # the deferred lookup starts now
    assert not scheduler.finish(committed=False)  # cancelled
    for _ in range(7):
        assert not scheduler.ordinary_commit()
    assert scheduler.ordinary_commit()
    assert scheduler.lookups == 1


def test_hindsight_schedule():
    schedule = HindsightSchedule()
    assert [schedule.stage(s) for s in (0, 4096, 8192)] == ["full", "gap1", "gap2"]
    assert schedule.overlay(100, 3) == 1.0
    assert schedule.overlay(16384 + 32768, 3) == pytest.approx(0.5)
    after = 16384 + 65536
    assert schedule.overlay(after, 8) == 1.0 and schedule.overlay(after, 9) == 0.0
    assert not schedule.computes_hindsight(after, 9, learned=True)
    assert schedule.computes_hindsight(after, 9, learned=False)
    assert schedule.reads(9000, groups=3, lookups=3) == [1, 2, None]
    assert schedule.reads(0, groups=3, lookups=3) == [0, 1, 2]


def test_trainer_writes_out_helpful_hindsight_and_steps_the_bank():
    model = tiny_model()
    trainer = TransducerTrainer(
        model.transducer,
        schedule=HindsightSchedule(full_steps=0, gap1_steps=0, warmup_steps=4, overlay_steps=4),
    )
    multi = MultiTrackGenerator(model)
    meridians = [Meridian((f"t{i % 2}",), (1,), ("text",), i % 3 == 2) for i in range(8)]
    payloads = [f"event {i}".encode() for i in range(8)]

    def raw(index):
        state = multi.tracks.initial_state(1)
        return multi.tracks.observe_text_group(state, [payloads[index]], tracks=("x",)).raw

    for step in (1, 2):
        sequence = trainer.sequence(meridians, step=step, generator=torch.Generator())
        sequence.hindsight(meridians, raw)
        state = sequence.begin(multi, multi.tracks.initial_state(1), metadata_payload({}))
        for index, meridian in enumerate(meridians):
            state = sequence.before_meridian(multi, state, index)
            state = multi.tracks.observe_text_group(
                state, [payloads[index]], tracks=meridian.tracks
            )
        # A loss that hindsight lowers: gradient -h on every lookup's value.
        loss = state.final.sum() * 0
        for record in trainer.records:
            loss = loss - record.lookup.value @ record.hindsight.detach()
        loss.backward()
        usable = sum(not r.sink_only for r in trainer.records)
        trainer.after_backward(step)
        metrics = trainer.barrier(step)
        if step == 1:
            assert usable and metrics["transducer_writeouts"] == usable
    assert trainer.bank.edge_count > 0
    restored = TransducerTrainer(tiny_model().transducer)
    restored.load_state_dict(trainer.state_dict())
    assert restored.bank.edge_ids().tolist() == trainer.bank.edge_ids().tolist()


def learner(canonical: TransducerBank, worker: int):
    bank = TransducerBank(canonical.geometry)
    bank.load_snapshot(canonical.state_snapshot())
    optimizer = BlockScaleDescent(bank, worker=worker, address_lr=0.5)
    growth = TransducerGrowth(bank, optimizer=optimizer, provisional=True)
    return bank, optimizer, growth


def test_coordinator_merges_deltas_writes_and_provisional_rows_then_learners_resync():
    torch.manual_seed(0)
    canonical = TransducerBank(GEOMETRY)
    add_random(canonical, 6)
    optimizer = BlockScaleDescent(canonical)
    growth = TransducerGrowth(canonical, optimizer=optimizer)
    coordinator = TransducerCoordinator(canonical, optimizer, growth)
    template = torch.randn(4, GEOMETRY.query_dim)
    rounds = []
    learners = []
    for worker in range(2):
        bank, opt, gro = learner(canonical, worker)
        learners.append((bank, opt, gro))
        give_gradient(bank, 1, "key", torch.randn(GEOMETRY.query_dim))
        opt.add_write(torch.tensor([2]), torch.tensor([1.0]), torch.randn(GEOMETRY.value_dim))
        opt.step(1)
        # Both learners write out the same entry; worker 1 also writes a distinct one.
        count = 1 + worker
        added = bank.add_edges(
            departure=template[:count, : GEOMETRY.location_dim],
            key=template[:count] if worker == 0 else torch.cat((template[:1], -template[1:2])),
            value=torch.randn(count, GEOMETRY.value_dim),
            destination=torch.randn(count, GEOMETRY.location_dim),
            bias=torch.zeros(count),
            write_mass=torch.ones(count),
            step=1,
            ids=torch.arange(PROVISIONAL_BASE, PROVISIONAL_BASE + count),
            provisional=True,
        )
        gro.provisional_ids.extend(added.tolist())
        for evidence in (gro.evidence, gro.round_evidence):
            evidence.visits[evidence.rows(added.tolist(), 1)] = 3.0
        rounds.append(
            learner_round(opt, gro, worker=worker, sequence=1, base_version=0, examples=1, step=1)
        )
    old_key = canonical.decode("key", [1])[0]
    for submission in rounds:
        assert coordinator.submit(submission)
    assert not coordinator.submit(rounds[0])
    result = coordinator.merge()
    assert result["submitted"] == 2 and result["adopted"] == 2  # one shared, one distinct
    assert not torch.equal(canonical.decode("key", [1])[0], old_key)
    assert float(canonical.table.write_mass[canonical.table.slots([2])][0]) > 1.0
    adopted = [edge for edge in canonical.edge_ids().tolist() if edge > 6]
    assert adopted == [7, 8] and canonical.table.next_id == 9
    visits = growth.evidence.visits[growth.evidence.rows(adopted, coordinator.step)]
    # Both learners' visits to the shared entry count; age-discounted by one round.
    assert visits.tolist() == pytest.approx([6.0, 3.0], rel=1e-2)
    # Stale and duplicate rounds are refused; later sequences are accepted.
    assert not coordinator.submit(rounds[1])
    for bank, opt, gro in learners:
        install_changes(bank, opt, gro, coordinator.changes_since(0))
        assert bank.edge_ids().tolist() == canonical.edge_ids().tolist()
        for name in ("key", "value", "departure"):
            torch.testing.assert_close(
                bank.decode(name, bank.edge_ids()), canonical.decode(name, canonical.edge_ids())
            )


def test_coordinator_host_persists_and_publishes(tmp_path):
    torch.manual_seed(0)
    settings = {"geometry": dict(vars(GEOMETRY)), "evidence_from": 0}
    host = TransducerHost(tmp_path, settings, workers=1, min_quorum=1, grace_seconds=0)
    bank, opt, gro = learner(host.bank, 0)
    add_random(
        bank,
        2,
        step=1,
        ids=torch.tensor([PROVISIONAL_BASE, PROVISIONAL_BASE + 1]),
        provisional=True,
    )
    gro.provisional_ids.extend([PROVISIONAL_BASE, PROVISIONAL_BASE + 1])
    submission = learner_round(opt, gro, worker=0, sequence=1, base_version=0, examples=1, step=1)
    upload = tmp_path / "round.pt"
    torch.save(submission.state_dict(), upload)
    host.submit(0, 1, 0, upload, RuntimeError)
    assert host.try_merge() and host.version() == 1
    version, path = host.changes(0)

    class Trainer:
        pass

    trainer = Trainer()
    trainer.bank, trainer.optimizer, trainer.growth = bank, opt, gro
    sync = TransducerSync()
    install_published(trainer, sync, "changes", path)
    assert sync.version == 1 and bank.edge_ids().tolist() == host.bank.edge_ids().tolist()
    host.persist()
    again = TransducerHost(tmp_path, settings, workers=1, min_quorum=1, grace_seconds=0)
    assert again.version() == 1 and again.bank.edge_count == host.bank.edge_count


def test_serving_session_cancels_cleanly_and_restores():
    from megatoken.transducer_service import TransducerInferenceEngine

    model = tiny_model()
    add_random(model.transducer.bank, 12)
    engine = TransducerInferenceEngine(MultiTrackGenerator(model))
    session = engine.new_session({"task": "chat"})
    assert session.agenda.lookups == 1
    before = (session.state.meridians_seen.clone(), session.agenda.lookups)
    events = [ObservedEvent("user", "text", b"hi")]
    for _ in range(7):
        assert engine.observe(session, events, cancelled=lambda: True) is None
    assert engine.observe(session, events, real_boundary=True, cancelled=lambda: True) is None
    assert session.agenda.lookups == before[1]
    assert int(session.state.meridians_seen[0]) == int(before[0][0]) + 8
    lookup = engine.observe(session, events, real_boundary=True)
    assert lookup is not None and session.agenda.lookups == 2
    restored = engine.restore_session(engine.state_snapshot(session))
    torch.testing.assert_close(restored.agenda.state, session.agenda.state)
    assert restored.scheduler.lookups == 2


def test_write_outs_are_sent_once_and_learner_checkpoints_reload():
    torch.manual_seed(0)
    canonical = TransducerBank(GEOMETRY)
    add_random(canonical, 3)
    bank, opt, gro = learner(canonical, 0)
    first = add_random(
        bank,
        2,
        step=1,
        ids=torch.tensor([PROVISIONAL_BASE, PROVISIONAL_BASE + 1]),
        provisional=True,
    )
    gro.provisional_ids.extend(first.tolist())
    one = learner_round(opt, gro, worker=0, sequence=1, base_version=0, examples=1, step=1)
    later = add_random(bank, 1, step=2, ids=torch.tensor([PROVISIONAL_BASE + 2]), provisional=True)
    gro.provisional_ids.extend(later.tolist())
    two = learner_round(opt, gro, worker=0, sequence=2, base_version=0, examples=1, step=2)
    assert one.provisional["ids"].tolist() == first.tolist()
    assert two.provisional["ids"].tolist() == later.tolist()
    # A learner snapshot holding write-outs reloads.
    copy = TransducerBank(GEOMETRY)
    copy.load_snapshot(bank.state_snapshot())
    assert copy.edge_count == bank.edge_count == 6


def test_learner_writes_into_rows_stamped_ahead_of_its_clock():
    canonical = TransducerBank(GEOMETRY)
    add_random(canonical, 2, step=640)  # the coordinator's clock is ahead
    bank, opt, _ = learner(canonical, 0)
    opt.add_write(torch.tensor([1]), torch.tensor([1.0]), torch.randn(GEOMETRY.value_dim))
    assert opt.step(3)["written_values"] == 1


def test_malformed_rounds_are_refused_and_failed_merges_are_consumed():
    canonical = TransducerBank(GEOMETRY)
    add_random(canonical, 3)
    optimizer = BlockScaleDescent(canonical)
    coordinator = TransducerCoordinator(
        canonical, optimizer, TransducerGrowth(canonical, optimizer=optimizer)
    )
    bank, opt, gro = learner(canonical, 0)
    give_gradient(bank, 1, "bias", torch.tensor(1.0))
    opt.step(1)
    good = learner_round(opt, gro, worker=0, sequence=1, base_version=0, examples=1, step=1)
    ids, rows = good.deltas["bias"]
    bad = learner_round(opt, gro, worker=0, sequence=2, base_version=0, examples=1, step=2)
    bad.deltas = {"bias": (ids, torch.full_like(rows, float("nan")))}
    with pytest.raises(ValueError):
        coordinator.submit(bad)
    assert coordinator.submit(good)
    coordinator._adopt = lambda weights: (_ for _ in ()).throw(RuntimeError("boom"))
    with pytest.raises(RuntimeError):
        coordinator.merge()
    assert not coordinator.pending and coordinator.version == 0
    assert not coordinator.submit(good)  # consumed, never reapplied
