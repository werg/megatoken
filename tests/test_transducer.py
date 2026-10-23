"""v10 transducer integration: native lookups, hindsight, training, DiLoCo and serving."""

from __future__ import annotations

import io
import types
from dataclasses import replace

import pytest
import torch
from media_support import observe_text

from megatoken.contracts import ROLE_TO_ID
from megatoken.generation.codec import ByteCodec
from megatoken.loading import build_model
from megatoken.memory.transducer import TransducerBank, TransducerGeometry
from megatoken.memory.transducer_growth import PROVISIONAL_BASE, TransducerGrowth
from megatoken.memory.transducer_optimizer import BlockScaleDescent
from megatoken.model.attention import AttentionFeatures, AttentionRead
from megatoken.model.config import (
    KNOWLEDGE_TRACK,
    RESERVED_TRACKS,
    TRANSDUCER_QUERY_TRACK,
    ModelConfig,
)
from megatoken.model.expert_pools import expert_scope
from megatoken.model.multitrack import MultiTrackGenerator, ObservedEvent
from megatoken.model.transducer import OrdinaryEvents, metadata_payload
from megatoken.training.transducer import (
    HindsightSchedule,
    Meridian,
    TransducerTrainer,
    plan_boundaries,
    question_lookup,
)
from megatoken.training.transducer_diloco import (
    LOCAL_STEPS,
    TransducerCoordinator,
    install_changes,
    learner_round,
)
from megatoken.training.transducer_sync import (
    TransducerHost,
    TransducerSync,
    _pack,
    install_published,
)

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


def test_reserved_tracks_share_the_equality_registry_without_semantic_slots():
    model = tiny_model()
    multi = MultiTrackGenerator(model)
    state = multi.tracks.initial_state(1)
    ids, registry = multi.tracks._track_indices(state, RESERVED_TRACKS)
    assert registry == tuple(sorted(RESERVED_TRACKS))
    assert ids[0].tolist() == [registry.index(t) for t in RESERVED_TRACKS]


def test_native_lookup_hindsight_and_parameter_ownership():
    model = tiny_model()
    native = model.transducer
    names = [name for name, _ in model.named_parameters()]
    assert len(names) == len(set(names))
    assert not any(name.startswith(("transducer._core", "transducer.core")) for name in names)
    coverage = model.core.expert_pools.manifest(model)["coverage"]
    assert coverage["unconverted"] == ()
    multi = MultiTrackGenerator(model)
    state = multi.tracks.initial_state(1)
    state, agenda = native.begin(multi, state, metadata_payload({"task": "chat"}))
    assert state.frontier_tracks == ("__metadata__",) and state.track_registry == ("__metadata__",)
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
                (TRANSDUCER_QUERY_TRACK,),
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
    retrospective = native.retrospective_values(events, [0, None])
    assert retrospective.values.shape == (2, native.bank.geometry.value_dim)
    assert retrospective.sink_only.tolist() == [False, True]


def test_transducer_query_and_knowledge_use_prior_committed_adapter_scopes(monkeypatch):
    model = tiny_model()
    native = model.transducer
    multi = MultiTrackGenerator(model)
    state = multi.tracks.initial_state(1)
    state, agenda = native.begin(multi, state, metadata_payload({"task": "adapter"}))
    active = model.core.adapters.select(
        torch.randn(model.config.transducer_geometry.value_dim), torch.ones(())
    )
    state = replace(state, adapters=active)

    calls = []
    original_plan = model.core.adapters.plan

    def record(owner, mixture, track_ids, track_semantics=None):
        calls.append((owner, mixture, track_ids.detach().clone()))
        return original_plan(owner, mixture, track_ids, track_semantics)

    monkeypatch.setattr(model.core.adapters, "plan", record)
    committed, found = native.boundary(multi, state, agenda, training=True)
    assert found is not None

    query_id = committed.track_registry.index(TRANSDUCER_QUERY_TRACK)
    knowledge_id = committed.track_registry.index(KNOWLEDGE_TRACK)
    query_calls = [call for call in calls if int(call[2][0, 0]) == query_id]
    knowledge_calls = [call for call in calls if int(call[2][0, 0]) == knowledge_id]
    assert any(owner == "p" and mixture is active for owner, mixture, _ in query_calls)
    assert any(owner == "r" and mixture is active for owner, mixture, _ in query_calls)
    assert any(owner == "r" and mixture is active for owner, mixture, _ in knowledge_calls)
    assert committed.adapters is not active


def test_transducer_readout_adapter_factors_receive_gradients_from_prior_mix():
    model = tiny_model()
    native = model.transducer
    multi = MultiTrackGenerator(model)
    state = multi.tracks.initial_state(1)
    state, agenda = native.begin(multi, state, metadata_payload({"task": "adapter-grad"}))
    state = replace(
        state,
        adapters=model.core.adapters.select(
            torch.randn(model.config.transducer_geometry.value_dim), torch.ones(())
        ),
    )
    committed, found = native.boundary(multi, state, agenda, training=True)
    assert found is not None
    committed.final.square().sum().backward()
    row_adapter = native.readout.row_adapter[1]
    assert row_adapter._bank.adapter_up.grad is not None
    assert torch.count_nonzero(row_adapter._bank.adapter_up.grad) > 0


def test_hindsight_pass_clears_any_live_adapter_scope():
    model = tiny_model()
    native = model.transducer
    width = model.config.facet_dim
    events = OrdinaryEvents(
        torch.randn(1, 3, model.config.n_facets, width),
        torch.ones(1, 3, dtype=torch.long),
        ("text",) * 3,
        ("a", "b", "a"),
        torch.tensor([[1, 1, 2]]),
        torch.tensor([[0, 0, 1]]),
    )
    mixture = model.core.adapters.select(
        torch.randn(model.config.transducer_geometry.value_dim), torch.ones(())
    )
    track_id = model.config.max_tracks + RESERVED_TRACKS.index(TRANSDUCER_QUERY_TRACK)
    plan = model.core.adapters.plan("r", mixture, torch.tensor([[track_id]]))
    assert plan is not None
    weights = plan.weights[0, :, 0]
    with expert_scope(weights):
        scoped = native.hindsight_pass(events, [0, 1, None], mode="group-inverse").values
    unscoped = native.hindsight_pass(events, [0, 1, None], mode="group-inverse").values
    torch.testing.assert_close(scoped, unscoped)


def test_hindsight_source_routing_uses_each_events_track(monkeypatch):
    model = tiny_model()
    native = model.transducer
    width = model.config.facet_dim
    events = OrdinaryEvents(
        torch.randn(1, 3, model.config.n_facets, width),
        torch.ones(1, 3, dtype=torch.long),
        ("text",) * 3,
        ("a", "b", "a"),
        torch.tensor([[1, 1, 2]]),
        torch.tensor([[0, 0, 1]]),
    )
    seen = []
    original = model.core.expert_pools.router_features

    def record(context, *, track_ids=None, track_semantics=None, role="other", stage=0):
        if role == "r":
            routed_tracks = track_ids.detach().clone() if track_ids is not None else None
            seen.append((context.shape, routed_tracks))
        return original(
            context, track_ids=track_ids, track_semantics=track_semantics, role=role, stage=stage
        )

    monkeypatch.setattr(model.core.expert_pools, "router_features", record)
    native.hindsight_pass(events, [0, 1, None], mode="group-inverse")
    assert seen[0][0] == (3, width)
    assert seen[0][1].tolist() == [0, 1, 0]


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
        return observe_text(multi.tracks, state, [payloads[index]], tracks=("x",)).raw

    for step in (1, 2):
        sequence = trainer.sequence(meridians, step=step, generator=torch.Generator())
        sequence.hindsight(meridians, raw)
        state = sequence.begin(multi, multi.tracks.initial_state(1), metadata_payload({}))
        for index, meridian in enumerate(meridians):
            state = sequence.before_meridian(multi, state, index)
            state = observe_text(multi.tracks, state, [payloads[index]], tracks=meridian.tracks)
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
    growth = TransducerGrowth(canonical)
    coordinator = TransducerCoordinator(canonical, growth)
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
    host.submit(0, 1, 0, upload)
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


def test_serving_session_cancels_cleanly():
    from megatoken.transducer_service import TransducerInferenceEngine

    model = tiny_model()
    add_random(model.transducer.bank, 12)
    engine = TransducerInferenceEngine(MultiTrackGenerator(model))
    session = engine.new_session({"task": "chat"})
    assert session.agenda.lookups == 1
    before = (session.state.meridians_seen.clone(), session.agenda.lookups)
    events = [ObservedEvent("user", "text", b"hi")]
    result = [ObservedEvent("tool", "text", b"ok", role_id=ROLE_TO_ID["tool"])]
    for _ in range(7):
        assert engine.observe(session, events, cancelled=lambda: True) is None
    assert engine.observe(session, result, cancelled=lambda: True) is None
    assert session.agenda.lookups == before[1]
    assert int(session.state.meridians_seen[0]) == int(before[0][0]) + 8
    lookup = engine.observe(session, result)
    assert lookup is not None and session.agenda.lookups == 2


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
    coordinator = TransducerCoordinator(canonical, TransducerGrowth(canonical))
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
    assert not coordinator.pending
    assert not coordinator.submit(good)  # consumed, never reapplied
    # The half-applied merge still moved the version and recorded the rows it touched.
    assert coordinator.version == 1
    changes = coordinator.changes_since(0)
    assert 1 in changes.rows["ids"].tolist()
    install_changes(bank, opt, gro, changes)
    torch.testing.assert_close(bank.decode("bias", [1]), canonical.decode("bias", [1]))


def test_phase_handoff_seeds_a_fresh_coordinator_whose_merges_succeed(tmp_path):
    torch.manual_seed(0)
    settings = {"geometry": dict(vars(GEOMETRY)), "evidence_from": 0}

    def merge_rounds(host, count):
        bank, opt, gro = learner(host.bank, 0)
        for sequence in range(1, count + 1):
            give_gradient(bank, 1, "key", torch.randn(GEOMETRY.query_dim))
            opt.step(sequence * LOCAL_STEPS)
            base = host.version()
            submission = learner_round(
                opt, gro, worker=0, sequence=sequence, base_version=base, examples=1,
                step=sequence * LOCAL_STEPS,
            )  # fmt: skip
            upload = tmp_path / f"round-{sequence}.pt"
            torch.save(submission.state_dict(), upload)
            host.submit(0, sequence, base, upload)
            assert host.try_merge() and host.version() == base + 1
            install_changes(bank, opt, gro, host.coordinator.changes_since(base))

    first = TransducerHost(tmp_path / "one", settings, workers=1, min_quorum=1, grace_seconds=0)
    add_random(first.bank, 6)
    merge_rounds(first, 2)
    # What export_coordinator writes and the next phase's coordinator is seeded from.
    exported = torch.load(io.BytesIO(_pack(first.checkpoint_state())), weights_only=True)
    second = TransducerHost(tmp_path / "two", settings, workers=1, min_quorum=1, grace_seconds=0)
    second.seed(exported)
    assert second.version() == 0 and second.bank.edge_ids().tolist() == [1, 2, 3, 4, 5, 6]
    merge_rounds(second, 2)
    assert second.skipped_merges == 0


def _worker_with_pending_round(tmp_path, error):
    from concurrent.futures import Future

    from megatoken.training.diloco_client import DecoupledWorker

    worker = DecoupledWorker.__new__(DecoupledWorker)
    worker.transducer = TransducerSync(sequence=4)
    worker.remote_transducer_version = 0
    worker.dropped_transducer_rounds = 0
    worker.last_progress_error = ""
    worker.control_io = types.SimpleNamespace(submit=lambda function, *args: args)
    upload = tmp_path / "round.pt"
    upload.write_bytes(b"round")
    future = Future()
    future.set_exception(error)
    worker.transducer.pending = {
        "filename": str(upload),
        "sequence": 4,
        "version": 0,
        "future": future,
    }
    return worker, upload


def test_rejected_rounds_are_dropped_and_counted(tmp_path):
    from megatoken.training.diloco_client import RemoteError

    worker, upload = _worker_with_pending_round(tmp_path, RemoteError(400, '{"error": "bad"}'))
    errors, ready = worker._transducer_step(1, None)
    assert (errors, ready) == (0, None)
    assert worker.transducer.pending is None and not upload.exists()
    assert worker.dropped_transducer_rounds == 1 and not worker.transducer.force_snapshot


def test_duplicate_rounds_skip_ahead_of_the_ledger_and_stale_ones_resnapshot(tmp_path):
    from megatoken.training.diloco_client import RemoteError

    duplicate = RemoteError(409, '{"error": "Duplicate transducer round", "sequence": 9}')
    worker, upload = _worker_with_pending_round(tmp_path, duplicate)
    worker._transducer_step(1, None)
    assert worker.transducer.sequence == 9 and worker.transducer.pending is None
    assert not worker.transducer.force_snapshot and not upload.exists()

    stale = RemoteError(409, '{"error": "Stale transducer base", "stale": true}')
    worker, upload = _worker_with_pending_round(tmp_path, stale)
    worker._transducer_step(1, None)
    assert worker.transducer.force_snapshot and worker.transducer.sequence == 4
    assert worker.transducer.download == (-1,)  # a full snapshot was requested


def test_duplicate_and_stale_rounds_are_told_apart_by_the_host(tmp_path):
    from megatoken.training.transducer_sync import DuplicateRound, StaleRound

    settings = {"geometry": dict(vars(GEOMETRY)), "evidence_from": 0}
    host = TransducerHost(tmp_path, settings, workers=1, min_quorum=1, grace_seconds=0)
    add_random(host.bank, 2)
    bank, opt, gro = learner(host.bank, 0)
    give_gradient(bank, 1, "bias", torch.tensor(1.0))
    opt.step(1)
    upload = tmp_path / "round.pt"

    def send(sequence, base):
        submission = learner_round(
            opt, gro, worker=0, sequence=sequence, base_version=base, examples=1, step=1
        )
        torch.save(submission.state_dict(), upload)
        return host.submit(0, sequence, base, upload)

    send(3, 0)
    with pytest.raises(DuplicateRound) as duplicate:
        send(3, 0)
    assert duplicate.value.sequence == 3
    with pytest.raises(StaleRound):
        send(4, 7)


def test_learned_values_switch_reaches_the_learner_through_changes_and_snapshots(tmp_path):
    canonical = TransducerBank(GEOMETRY)
    add_random(canonical, 3)
    growth = TransducerGrowth(canonical)
    coordinator = TransducerCoordinator(canonical, growth)
    changes_bank, changes_opt, changes_gro = learner(canonical, 0)
    snapshot_bank, snapshot_opt, snapshot_gro = learner(canonical, 1)
    for evidence in (changes_gro.evidence, snapshot_gro.evidence):
        evidence.sketch_count[evidence.rows([1], 0)] = 5.0
    coordinator.switch_requested = True
    bank, opt, gro = learner(canonical, 2)
    give_gradient(bank, 1, "bias", torch.tensor(1.0))
    opt.step(1)
    assert coordinator.submit(
        learner_round(opt, gro, worker=2, sequence=1, base_version=0, examples=1, step=1)
    )
    coordinator.merge()
    assert canonical.learned_values
    install_changes(changes_bank, changes_opt, changes_gro, coordinator.changes_since(0))
    assert changes_bank.learned_values
    assert float(changes_gro.evidence.sketch_count.sum()) == 0.0
    upload = tmp_path / "published.pt"
    torch.save({"version": 1, "bank": canonical.state_snapshot()}, upload)
    trainer = types.SimpleNamespace(bank=snapshot_bank, optimizer=snapshot_opt, growth=snapshot_gro)
    install_published(trainer, TransducerSync(), "snapshot", upload)
    assert snapshot_bank.learned_values
    assert float(snapshot_gro.evidence.sketch_count.sum()) == 0.0


def _learned_canonical(edges=3):
    """A canonical bank already in the learned-values phase, with its coordinator."""
    canonical = TransducerBank(GEOMETRY)
    add_random(canonical, edges)
    growth = TransducerGrowth(canonical)
    canonical.switch_to_learned()
    growth.switch_to_learned()
    return canonical, growth, TransducerCoordinator(canonical, growth)


def _value_round(canonical, *, worker=0, sequence=1, step=1):
    bank, opt, gro = learner(canonical, worker)
    bank.learned_values = False  # the learner crosses the switch as install_changes does
    opt.switch_to_learned()
    gro.switch_to_learned()
    give_gradient(bank, 1, "value", torch.randn(GEOMETRY.value_dim))
    opt.step(step)
    return learner_round(
        opt, gro, worker=worker, sequence=sequence, base_version=0, examples=1, step=step
    )


def test_value_deltas_travel_packed_and_merge_exactly_as_decoded():
    from megatoken.memory.transducer_packing import pack_fp4, row_seeds
    from megatoken.training.transducer_diloco import COORDINATOR_WORKER

    torch.manual_seed(0)
    canonical, _, coordinator = _learned_canonical()
    submission = _value_round(canonical, step=5)
    assert "value" not in submission.deltas and submission.value_delta["ids"].tolist() == [1]
    dense = GEOMETRY.value_dim * 4  # bytes of one FP32 value row
    packed = sum(t.numel() * t.element_size() for t in submission.value_delta.values())
    assert packed * 4 <= dense + 8  # at least 4x smaller than the dense FP32 row
    # The merge is exactly the decoded rows that were sent, added to the current row.
    sent_ids, sent = submission.decoded_deltas(GEOMETRY.value_dim)["value"]
    current = canonical.decode("value", sent_ids)
    seeds = row_seeds(sent_ids, "value", 5, COORDINATOR_WORKER)
    expected = pack_fp4(current + sent, seeds).decode(GEOMETRY.value_dim)
    assert coordinator.submit(submission)
    coordinator.merge()
    torch.testing.assert_close(canonical.decode("value", sent_ids), expected, rtol=0, atol=0)
    # The wire format round-trips through the torch archive the transport uses.
    buffer = io.BytesIO()
    torch.save(submission.state_dict(), buffer)
    buffer.seek(0)
    again = type(submission).from_state(torch.load(buffer, weights_only=True))
    torch.testing.assert_close(
        again.decoded_deltas(GEOMETRY.value_dim)["value"][1], sent, rtol=0, atol=0
    )


def test_a_round_that_cannot_stage_is_consumed_instead_of_wedging_every_merge():
    torch.manual_seed(0)
    canonical = TransducerBank(GEOMETRY)
    add_random(canonical, 3)
    host_growth = TransducerGrowth(canonical)
    coordinator = TransducerCoordinator(canonical, host_growth)
    # Built in the learned phase, queued, then the bank is still in the written phase at
    # merge time: its value deltas cannot merge as means.
    learned, _, _ = _learned_canonical()
    bad = _value_round(learned, worker=0, sequence=1)
    assert coordinator.submit(bad)
    with pytest.raises(ValueError, match="Written-phase values"):
        coordinator.merge()
    assert not coordinator.pending and coordinator.ledger == {0: 1}
    assert coordinator.version == 0  # nothing was applied
    # The next round merges.
    bank, opt, gro = learner(canonical, 1)
    give_gradient(bank, 1, "bias", torch.tensor(1.0))
    opt.step(1)
    good = learner_round(opt, gro, worker=1, sequence=1, base_version=0, examples=1, step=1)
    assert coordinator.submit(good)
    assert coordinator.merge()["submitted"] == 1 and coordinator.version == 1


def test_a_failed_staging_is_counted_as_a_skipped_merge(tmp_path):
    settings = {"geometry": dict(vars(GEOMETRY)), "evidence_from": 0}
    host = TransducerHost(tmp_path, settings, workers=1, min_quorum=1, grace_seconds=0)
    add_random(host.bank, 3)
    learned, _, _ = _learned_canonical()
    host.coordinator.pending.append(_value_round(learned))
    assert host.merge_or_skip() is False
    assert host.skipped_merges == 1 and not host.coordinator.pending


def test_malformed_packed_rows_are_refused_at_submission():
    canonical, _, coordinator = _learned_canonical()
    submission = _value_round(canonical)
    submission.value_delta["codes"] = submission.value_delta["codes"][:, :-1]
    with pytest.raises(ValueError, match="packed value delta"):
        coordinator.submit(submission)
    canonical = TransducerBank(GEOMETRY)
    add_random(canonical, 3)
    coordinator = TransducerCoordinator(canonical, TransducerGrowth(canonical))
    bank, opt, gro = learner(canonical, 0)
    opt.add_write(torch.tensor([2]), torch.tensor([1.0]), torch.randn(GEOMETRY.value_dim))
    opt.step(1)
    writes = learner_round(opt, gro, worker=0, sequence=1, base_version=0, examples=1, step=1)
    writes.write_means["value_groups"] = writes.write_means["value_groups"][:, :-1]
    with pytest.raises(ValueError, match="hindsight writes"):
        coordinator.submit(writes)


def test_the_coordinator_clock_follows_the_learners_not_the_merge_count():
    canonical = TransducerBank(GEOMETRY)
    add_random(canonical, 3)
    coordinator = TransducerCoordinator(canonical, TransducerGrowth(canonical))
    for worker in range(2):  # two workers, each round merged alone (partial quorum)
        bank, opt, gro = learner(canonical, worker)
        give_gradient(bank, 1, "bias", torch.tensor(1.0))
        opt.step(LOCAL_STEPS)
        submission = learner_round(
            opt, gro, worker=worker, sequence=1, base_version=0, examples=1, step=LOCAL_STEPS
        )
        assert coordinator.submit(submission)
        coordinator.merge()
    # Two merges of 32-step rounds: the clock is the learners' 32, not 2 x 32.
    assert coordinator.version == 2 and coordinator.step == LOCAL_STEPS
    bank, opt, gro = learner(canonical, 0)
    give_gradient(bank, 1, "bias", torch.tensor(1.0))
    opt.step(1)
    late = learner_round(opt, gro, worker=0, sequence=2, base_version=2, examples=1, step=7)
    coordinator.submit(late)
    coordinator.merge()
    assert coordinator.step == LOCAL_STEPS  # never backwards
    # The clock survives a checkpoint.
    restored = TransducerCoordinator(canonical, TransducerGrowth(canonical))
    restored.load_state_dict(coordinator.state_dict())
    assert restored.step == LOCAL_STEPS and restored.version == 3


def test_a_seeded_coordinator_restamps_rows_to_its_own_clock(tmp_path):
    settings = {"geometry": dict(vars(GEOMETRY)), "evidence_from": 0}
    first = TransducerHost(tmp_path / "one", settings, workers=1, min_quorum=1, grace_seconds=0)
    add_random(first.bank, 4, step=640)
    first.bank.table.last_write[:] = 700
    first.growth.evidence.rows([1, 2], 900)
    exported = torch.load(io.BytesIO(_pack(first.checkpoint_state())), weights_only=True)
    second = TransducerHost(tmp_path / "two", settings, workers=1, min_quorum=1, grace_seconds=0)
    second.seed(exported)
    table = second.bank.table
    slots = table.slots(second.bank.edge_ids().tolist())
    assert table.birth[slots].tolist() == [0] * 4 and table.last_write[slots].tolist() == [0] * 4
    assert int(second.growth.evidence.last_step.max()) == 0


def test_the_annotated_lookup_is_the_one_after_the_first_finished_user_message():
    system = Meridian(("s",), (0,), ("text",), False)
    asked = Meridian(("u",), (1,), ("text",), True, asked=True)
    answer = Meridian(("a",), (2,), ("text",), True)

    def planned(meridians):
        before = plan_boundaries(meridians, max_lookups=256, generator=None)
        return before, question_lookup(meridians, before)

    # Lookup 0 follows the metadata; lookup 1 follows the question, before the answer.
    assert planned([asked, answer]) == ([1, 1], 1)
    # A system message before it is not a boundary and does not move the question's lookup.
    assert planned([system, asked, answer]) == ([1, 0, 1], 1)
    # Later user messages are not the annotated one.
    assert planned([asked, answer, asked, answer])[1] == 1
    # A question that is the window's last meridian has no lookup after it in the window,
    # and a window without a finished question has none to annotate.
    assert planned([answer, asked])[1] is None
    assert planned([system, answer])[1] is None
