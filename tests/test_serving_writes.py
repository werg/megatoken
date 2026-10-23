"""Conversation memory: ordering, bounded storage, rollback and real observed-text scoring."""

import pytest
import torch

from megatoken.generation.codec import ByteCodec
from megatoken.kb import model_identity
from megatoken.loading import build_model
from megatoken.memory.knowledge import WRITE_HALF_LIFE, KnowledgeBase
from megatoken.model import ModelConfig
from megatoken.model.multitrack import MultiTrackGenerator, ObservedEvent
from megatoken.model.transducer import HindsightValues
from megatoken.serving_writes import SurpriseMeter, WritePolicy
from megatoken.transducer_service import TransducerInferenceEngine


@pytest.fixture
def engine():
    torch.manual_seed(9)
    return TransducerInferenceEngine(
        MultiTrackGenerator(build_model(ModelConfig.tiny(), ByteCodec.byte_fallback()))
    )


def add_row(base, value=None):
    g = base.geometry
    value = torch.ones(1, g.value_dim) if value is None else value
    return base.add_edges(
        departure=torch.ones(1, g.location_dim),
        key=torch.ones(1, g.query_dim),
        value=value,
        destination=torch.ones(1, g.location_dim),
        bias=torch.zeros(1),
        write_mass=torch.ones(1),
        step=0,
    ).tolist()


def test_blend_uses_master_decay_and_survives_save(engine, tmp_path):
    base = engine.create_overlay("user", "alice", tmp_path / "u.kb")
    ids = add_row(base)
    old = base.decode("value", ids).clone()
    base.blend(ids, torch.full_like(old, 3), 2, WRITE_HALF_LIFE)
    expected = (old * 0.5 + 6) / 2.5
    torch.testing.assert_close(base.masters[ids[0]], expected[0])
    base.save(base.path)
    reopened = KnowledgeBase.load(
        base.path, geometry=engine.bank.geometry, identity=engine.identity, writable=True
    )
    assert reopened.geometry.max_edges == 2048 and reopened.scope == "alice"
    reopened.blend(ids, torch.full_like(old, 7), 1, WRITE_HALF_LIFE)
    torch.testing.assert_close(reopened.masters[ids[0]], (expected[0] * 2.5 + 7) / 3.5)


def test_retire_zeroes_free_columns_and_disk(engine, tmp_path):
    base = engine.create_overlay("user", "alice", tmp_path / "u.kb")
    ids = add_row(base)
    slots = base.table.slots(ids)
    base.sources[ids[0]] = "secret"
    base.provenance[ids[0]] = {"scope": "alice", "session": "old"}
    base.masters[ids[0]] = torch.ones(base.geometry.value_dim)
    old_version = base.version
    assert engine.forget("user", scope="alice") == ids
    assert base.version > old_version
    for name in base.table.spec:
        if name != "ids":
            assert torch.count_nonzero(getattr(base.table, name)[slots].float()) == 0
    assert not base.sources and not base.masters and not base.provenance
    assert b"secret" not in base.path.read_bytes()
    assert (
        KnowledgeBase.load(
            base.path, geometry=engine.bank.geometry, identity=engine.identity
        ).edge_count
        == 0
    )


def test_transaction_restores_blends_evictions_and_next_id(engine, tmp_path):
    base = engine.create_overlay("user", "alice", tmp_path / "u.kb", max_edges=1)
    ids = add_row(base)
    old = base.table.state()
    version = base.version
    with pytest.raises(RuntimeError, match="fail"):
        with base.transaction():
            base.blend(ids, torch.zeros(1, base.geometry.value_dim), 1, 1)
            base.make_room(1)
            add_row(base)
            raise RuntimeError("fail")
    assert base.version == version and base.edge_ids().tolist() == ids
    for name, value in old.items():
        torch.testing.assert_close(base.table.state()[name].float(), value.float())
    assert base.table.next_id == 2


def test_exported_hindsight_does_not_change_base_identity(engine):
    model = engine.multi.generator
    identity = model_identity(model)
    with torch.no_grad():
        for parameter in model.transducer.hindsight_realizer.parameters():
            parameter.add_(1)
    assert model_identity(model) == identity


def fake_hindsight(engine, monkeypatch, seen):
    def values(events, reads):
        seen.append((engine.bank.version, list(reads), events.groups.clone()))
        return HindsightValues(
            torch.ones(len(reads), engine.bank.geometry.value_dim),
            torch.zeros(len(reads), dtype=torch.bool),
        )

    monkeypatch.setattr(engine.native, "retrospective_values", values)


def test_writes_follow_commits_and_flush_skips_unseen_future(engine, tmp_path, monkeypatch):
    seen = []
    base = engine.create_overlay("user", "alice", tmp_path / "u.kb")
    fake_hindsight(engine, monkeypatch, seen)
    session = engine.new_session(
        {}, overlay="user", scope="alice", write_policy=WritePolicy(lag=0, autosave_every=1)
    )
    user = (ObservedEvent("main", "text", b"new fact"),)
    first = engine.observe(session, user)
    assert first is not None and not seen and base.edge_count == 0
    second = engine.observe(session, user)
    assert second is not None and session.agenda.lookups == 3
    assert seen and base.edge_count == 1 and session.writes.journal
    assert {p["scope"] for p in base.provenance.values()} == {"alice"}
    # A mutation after commit invalidates the old lookup for any second commit.
    with pytest.raises(RuntimeError, match="[Ss]tale"):
        engine.bank.commit(session.agenda, second)
    engine.close_session(session)
    assert session.closed and not session.writes.events
    # No h for the most recent lookup: it still has no real future group.
    assert max(read for _, reads, _ in seen for read in reads) <= 1
    assert (
        KnowledgeBase.load(
            base.path, geometry=engine.bank.geometry, identity=engine.identity
        ).edge_count
        >= 1
    )


def test_failed_write_rolls_back_accepted_history_and_recorder(engine, tmp_path, monkeypatch):
    base = engine.create_overlay("user", "alice", tmp_path / "u.kb")
    fake_hindsight(engine, monkeypatch, [])
    session = engine.new_session({}, overlay="user", scope="alice", write_policy=WritePolicy(lag=0))
    events = (ObservedEvent("main", "text", b"fact"),)
    engine.observe(session, events)
    state, lookups = session.state, session.agenda.lookups
    previous = dict(vars(session.scheduler))

    def fail(**kwargs):
        raise RuntimeError("publication failed")

    monkeypatch.setattr(base, "add_edges", fail)
    with pytest.raises(RuntimeError, match="publication failed"):
        engine.observe(session, events)
    assert session.state is state and session.agenda.lookups == lookups
    assert vars(session.scheduler) == previous and session.writes.turn == 1
    assert base.edge_count == 0 and len(session.writes.events) == 1


def test_scopes_caps_and_forget_cannot_resurrect(engine, tmp_path, monkeypatch):
    base = engine.create_overlay("user", "alice", tmp_path / "u.kb", max_edges=1)
    with pytest.raises(ValueError, match="scope"):
        engine.new_session({}, overlay="user", scope="bob")
    with pytest.raises(ValueError, match="scope"):
        engine.new_session({}, bases={"user": 1}, scope="bob")
    fake_hindsight(engine, monkeypatch, [])
    session = engine.new_session(
        {},
        overlay="user",
        scope="alice",
        write_policy=WritePolicy(lag=0, max_writes=2, max_events=2),
    )
    for _ in range(7):
        engine.observe(session, (ObservedEvent("main", "text", b"x"),))
    assert base.edge_count <= 1 and session.writes.writes == 2
    assert sum(e.frontier_count for e, _ in session.writes.events) <= 2
    engine.forget("user", session=session.session_id)
    engine.flush(session)
    assert base.edge_count == 0 and not session.writes.events


def test_observed_byte_surprise_and_exponential_threshold(engine):
    session = engine.new_session({})
    meter = SurpriseMeter()
    event = ObservedEvent("main", "text", b"hello")
    score = meter.score(engine.multi, session.state, event, effort=1)
    assert score > 0 and torch.isfinite(torch.tensor(score))
    assert not meter.accept(2, WritePolicy())
    assert meter.accept(4, WritePolicy())
    assert meter.mean == pytest.approx(2.1)
    assert not meter.accept(float("nan"), WritePolicy())


def test_surprise_gate_checks_trained_and_overlay_keys(engine, tmp_path, monkeypatch):
    base = engine.create_overlay("user", "alice", tmp_path / "u.kb")
    session = engine.new_session(
        {},
        overlay="user",
        scope="alice",
        write_policy=WritePolicy(retrospective=False, surprise=True),
    )
    scores = iter([1.0, 10.0, 100.0])
    monkeypatch.setattr(session.writes.meter, "score", lambda *a, **k: next(scores))
    for _ in range(3):
        engine.observe(session, (ObservedEvent("main", "text", b"new fact"),))
    assert base.edge_count == 1
    assert session.writes.journal[0]["rule"] == "surprise"
    engine.close_session(session)


def test_promotion_requires_review_and_copies_exact_rows(engine, tmp_path):
    base = engine.create_overlay("user", "alice", tmp_path / "u.kb")
    ids = add_row(base)
    target = engine.create_base("shared")
    with pytest.raises(ValueError, match="review"):
        engine.promote("user", "shared", ids)
    added = engine.promote("user", "shared", ids, reviewed=True)
    for name in ("departure", "key", "value", "destination"):
        torch.testing.assert_close(base.decode(name, ids), target.decode(name, added))


def test_real_retrospective_pass_and_close_autosave(engine, tmp_path):
    base = engine.create_overlay("user", "alice", tmp_path / "u.kb")
    session = engine.new_session({}, overlay="user", scope="alice", write_policy=WritePolicy(lag=4))
    engine.observe(session, (ObservedEvent("main", "text", b"fact"),))
    engine.observe(session, (ObservedEvent("main", "text", b"result"),))
    assert session.writes.writes == 0
    engine.close_session(session)
    assert base.edge_count > 0
    assert (
        KnowledgeBase.load(
            base.path, geometry=engine.bank.geometry, identity=engine.identity
        ).edge_count
        == base.edge_count
    )


def test_failed_save_rolls_back_eviction_and_recorder(engine, tmp_path, monkeypatch):
    base = engine.create_overlay("user", "alice", tmp_path / "u.kb", max_edges=1)
    add_row(base, torch.zeros(1, base.geometry.value_dim))
    before = base.table.state()
    fake_hindsight(engine, monkeypatch, [])
    session = engine.new_session(
        {}, overlay="user", scope="alice", write_policy=WritePolicy(lag=0, autosave_every=1)
    )
    engine.observe(session, (ObservedEvent("main", "text", b"x"),))
    state, lookups = session.state, session.agenda.lookups

    def fail(path):
        raise OSError("disk full")

    monkeypatch.setattr(base, "save", fail)
    with pytest.raises(OSError, match="disk full"):
        engine.observe(session, (ObservedEvent("main", "text", b"y"),))
    assert session.state is state and session.agenda.lookups == lookups
    assert session.writes.writes == 0 and session.writes.turn == 1
    for name, value in before.items():
        torch.testing.assert_close(base.table.state()[name].float(), value.float())


def test_speculative_work_does_not_write(engine, tmp_path):
    from megatoken.generation.structure import ActionSpec
    from megatoken.multitrack_runtime import MultiTrackRuntime

    runtime = MultiTrackRuntime(engine.multi)
    base = runtime.memory.create_overlay("user", "alice", tmp_path / "u.kb")
    session = runtime.open(overlay="user", scope="alice")
    before = (base.version, session.memory.writes.turn, session.revision)
    runtime.speculate(session, tracks=("main",), specs=(ActionSpec(max_tokens=2),))
    assert (base.version, session.memory.writes.turn, session.revision) == before
    runtime.close()


def test_protocol_overlay_flush_forget_and_scope(engine, tmp_path):
    from megatoken.multitrack_runtime import MultiTrackRuntime
    from megatoken.multitrack_service import MultiTrackService

    runtime = MultiTrackRuntime(engine.multi)
    service = MultiTrackService(runtime, kb_root=tmp_path)
    created = service.handle(dict(op="kb_overlay", name="private", scope="alice", path="a.kb"))
    assert created["ok"]
    opened = service.handle(
        dict(op="open", overlay="private", scope="alice", write_policy={"lag": 0})
    )
    assert opened["ok"]
    for text in ("fact", "result"):
        assert service.handle(
            dict(op="observe_text", session=opened["session"], track="main", text=text)
        )["ok"]
    assert service.handle(dict(op="flush", session=opened["session"]))["ok"]
    assert service.handle(dict(op="kb_forget", name="private", session=opened["session"]))["ok"]
    assert service.handle(dict(op="close", session=opened["session"]))["ok"]
    assert runtime.memory.bases["private"].edge_count == 0
    runtime.close()


def test_legacy_knowledge_base_loads_without_changing_model_identity(engine, tmp_path):
    from megatoken.memory.knowledge import FORMAT

    base = engine.create_base("old")
    ids = add_row(base)
    path = tmp_path / "old.kb"
    torch.save(
        dict(
            format=FORMAT,
            name="old",
            identity=engine.identity,
            next_edge_id=base.table.next_id,
            rows=base.table.state(),
            sources={},
        ),
        path,
    )
    opened = engine.open_base("legacy", path, writable=True)
    assert opened.edge_ids().tolist() == ids and opened.scope is None
    assert not opened.masters and not opened.provenance


def test_scope_has_one_writable_store_even_after_unmount(engine, tmp_path):
    base = engine.create_overlay("user", "alice", tmp_path / "u.kb")
    session = engine.new_session({}, overlay="user", scope="alice")
    engine.close_base("user")
    with pytest.raises(ValueError, match="scope"):
        engine.open_base("again", base.path, writable=True)
    engine.close_session(session)
    assert engine.open_base("again", base.path, writable=True).scope == "alice"


@pytest.mark.parametrize("stream_failure", [False, True])
def test_cli_shutdown_flushes_open_sessions(engine, tmp_path, monkeypatch, stream_failure):
    import io
    import json
    import sys
    from types import SimpleNamespace

    from megatoken import cli
    from megatoken.multitrack_runtime import MultiTrackRuntime

    runtime = MultiTrackRuntime(engine.multi)
    monkeypatch.setattr(cli, "_runtime", lambda *a, **kw: (None, runtime))
    args = SimpleNamespace(packs=None, collection=[], kb_root=str(tmp_path), max_sessions=2)

    def requests():
        yield json.dumps(dict(op="kb_overlay", name="private", scope="alice", path="u.kb")) + "\n"
        yield json.dumps(dict(op="open", overlay="private", scope="alice",
                              write_policy={"lag": 4})) + "\n"
        (session_id,) = runtime.memory.sessions
        for text in ("fact", "result"):
            yield json.dumps(dict(op="observe_text", session=session_id, track="main", text=text)) + "\n"
        assert runtime.memory.bases["private"].edge_count == 0
        if stream_failure:
            raise RuntimeError("input disconnected")

    output = io.StringIO()
    monkeypatch.setattr(sys, "stdin", requests())
    monkeypatch.setattr(sys, "stdout", output)
    if stream_failure:
        with pytest.raises(RuntimeError, match="input disconnected"):
            cli._serve(args)
    else:
        cli._serve(args)
    assert all(json.loads(line)["ok"] for line in output.getvalue().splitlines())
    assert not runtime.memory.sessions
    saved = KnowledgeBase.load(tmp_path / "u.kb", geometry=engine.bank.geometry,
                               identity=runtime.memory.identity)
    assert saved.edge_count > 0


def test_scoped_sessions_write_by_default_and_reopen_after_restart(engine, tmp_path):
    from megatoken.multitrack_runtime import MultiTrackRuntime

    runtime = MultiTrackRuntime(engine.multi, memory_root=tmp_path)
    session = runtime.open(scope="alice")
    writer = session.memory.writes
    assert writer is not None and writer.policy.retrospective and not writer.policy.surprise
    base = writer.base
    assert base.scope == "alice" and base.path.is_relative_to(tmp_path)
    runtime.observe(session, (ObservedEvent("main", "text", b"fact"),))
    runtime.observe(session, (ObservedEvent("main", "text", b"result"),))
    runtime.close()
    assert base.edge_count > 0

    restarted = MultiTrackRuntime(engine.multi, memory_root=tmp_path)
    again = restarted.open(scope="alice")
    restored = again.memory.writes.base
    assert restored.path == base.path and restored.edge_ids().tolist() == base.edge_ids().tolist()
    assert all(p["scope"] == "alice" for p in restored.provenance.values())
    assert any(m.base is restored for m in again.memory.agenda.mounted)
    restarted.close()


def test_scoped_default_reuses_existing_store_and_separates_users(engine, tmp_path):
    from megatoken.multitrack_runtime import MultiTrackRuntime

    runtime = MultiTrackRuntime(engine.multi, memory_root=tmp_path)
    alice = runtime.open(scope="alice")
    again = runtime.open(scope="alice")
    bob = runtime.open(scope="bob")
    assert alice.memory.writes.base is again.memory.writes.base
    assert alice.memory.writes.base is not bob.memory.writes.base
    assert alice.memory.writes.base.path != bob.memory.writes.base.path
    assert bob.memory.writes.base not in [m.base for m in alice.memory.agenda.mounted]
    runtime.close()


def test_default_memory_requires_scope_and_allows_explicit_disable(engine, tmp_path):
    from megatoken.multitrack_runtime import MultiTrackRuntime
    from megatoken.multitrack_service import MultiTrackService

    runtime = MultiTrackRuntime(engine.multi)
    assert runtime.open().memory.writes is None
    with pytest.raises(ValueError, match="persistent"):
        runtime.open(scope="alice")
    assert runtime.open(scope="alice", memory_writes=False).memory.writes is None
    service = MultiTrackService(runtime, kb_root=tmp_path)
    disabled = service.handle(dict(op="open", scope="bob", memory_writes=False))
    assert disabled["ok"] and not disabled["memory_writes"]
    assert not (tmp_path / "scopes").exists()
    for bad in ("false", 0, None):
        assert not service.handle(dict(op="open", scope="alice", memory_writes=bad))["ok"]
    opened = service.handle(dict(op="open", scope="alice"))
    assert opened["ok"] and opened["memory_writes"] and opened["overlay"]
    assert runtime.memory.bases[opened["overlay"]].scope == "alice"
    runtime.close()


def test_default_scope_reuses_custom_store_without_a_root(engine, tmp_path):
    custom = engine.create_overlay("custom", "alice", tmp_path / "custom.kb")
    session = engine.new_session({}, scope="alice")
    assert session.writes.base is custom
    engine.close_base("custom")
    another = engine.new_session({}, scope="alice")
    assert another.writes.base is custom
    engine.close()


def test_scope_name_cannot_escape_storage_root(engine, tmp_path):
    engine.memory_root = tmp_path
    session = engine.new_session({}, scope="../../outside")
    assert session.writes.base.path.is_relative_to(tmp_path / "scopes")
    assert "outside" not in session.writes.base.path.name
    engine.close()
