"""One serving session mounts a knowledge base and a source collection together."""

from __future__ import annotations

import io
import json
from dataclasses import replace

import torch
from retrieval_support import COLLECTION, FakePreparer, reading_root, tiny_model

from megatoken.contracts import ROLE_TO_ID
from megatoken.model.multitrack import MultiTrackGenerator, ObservedEvent
from megatoken.multitrack_runtime import MultiTrackRuntime
from megatoken.multitrack_service import serve_multitrack_jsonl


def _runtime(tmp_path, preparer=None):
    root = reading_root(tmp_path)
    model = tiny_model()
    # Adapter up-projections start at zero in a fresh model; explicitly enable
    # the shared per-expert factors so this test can observe their effect.
    for name, parameter in model.core.expert_pools.named_parameters():
        if "adapter_up" in name:
            torch.nn.init.normal_(parameter, std=0.5)
    runtime = MultiTrackRuntime(
        MultiTrackGenerator(model),
        collections={"main": root / COLLECTION},
        preparer=(lambda: preparer) if preparer is not None else None,
    )
    runtime.memory.collections["main"].set_threshold(-1.0)
    return runtime


def _tool():
    return (ObservedEvent("main", "text", b"result", role_id=ROLE_TO_ID["tool"]),)


def _notes(tmp_path):
    (tmp_path / "notes.txt").write_text("The vault key is kept in the north cabinet. " * 5)
    (tmp_path / "photo.png").write_bytes(b"a photo")
    return [tmp_path / "notes.txt", tmp_path / "photo.png"]


def test_one_call_prepares_each_file_once_and_adds_it_to_a_collection_and_a_base(tmp_path):
    preparer = FakePreparer(tmp_path / "scratch")
    reads = []
    media = preparer.media
    preparer.media = lambda modality, path, **kw: (
        reads.append(str(path)) or media(modality, path, **kw)
    )
    runtime = _runtime(tmp_path, preparer)
    runtime.memory.create_base("mine")
    before = runtime.memory.collections["main"].count
    added = runtime.add_files(_notes(tmp_path), collection="main", base="mine")
    assert added.collection == 2 and len(added.base) == 2
    assert runtime.memory.collections["main"].count == before + 2
    assert runtime.memory.bases["mine"].edge_count == 2
    # The photo was read by the teacher once, not once per destination.
    assert reads == [str(tmp_path / "photo.png")]


def test_a_session_mounts_a_base_and_a_collection_and_a_lookup_uses_both(tmp_path):
    runtime = _runtime(tmp_path, FakePreparer(tmp_path / "scratch"))
    runtime.memory.create_base("mine")
    runtime.add_files(_notes(tmp_path), collection="main", base="mine")
    session = runtime.open({"task": "q"}, bases={"mine": 1.0}, collections=["main"], max_sources=2)
    names = [m.name for m in session.memory.agenda.mounted]
    assert names == ["trained", "mine"] and session.memory.sources is not None
    first_log = len(session.memory.sources.log)
    lookup = runtime.memory.observe(session.memory, _tool())
    # The base's entries took part in the search ...
    assert {name for _, name, _ in lookup.pairs} == {"mine"}
    # ... the adapter mixture came from the blended value they gave ...
    mixture = session.state.adapters
    assert mixture is not None and float(mixture.strength) > 0
    assert float(mixture.strength) == float(1 - lookup.sink_weight)
    plain = replace(session.state, adapters=None)
    multi = runtime.generator
    event = ObservedEvent("main", "text", b"x", role_id=ROLE_TO_ID["user"])
    with torch.no_grad():
        applied = multi.tracks.observe_group(session.state, [event])
        without = multi.tracks.observe_group(plain, [event])
    assert not torch.equal(applied.final, without.final)
    # ... and the collection's items were observed after the knowledge event.
    log = session.memory.sources.log
    assert len(log) > first_log and {entry.collection for entry in log} == {"main"}
    # A session may leave either mount out.
    bare = runtime.open({"task": "q"}, bases={"mine": 1.0}, collections=[])
    assert bare.memory.sources is None and len(bare.memory.agenda.mounted) == 2
    assert runtime.open({"task": "q"}).memory.sources is not None


def test_the_service_adds_files_to_both_and_opens_a_session_on_both(tmp_path):
    root = tmp_path / "kb"
    root.mkdir()
    _notes(root)
    collection_root = reading_root(tmp_path)
    model = tiny_model()
    runtime = MultiTrackRuntime(
        MultiTrackGenerator(model),
        collections={"main": collection_root / COLLECTION},
        preparer=lambda: FakePreparer(root / "prepared"),
    )

    def requests(results):
        yield json.dumps({"op": "kb_create", "id": 1, "name": "mine"}) + "\n"
        yield (
            json.dumps(
                {
                    "op": "add_files",
                    "id": 2,
                    "collection": "main",
                    "base": "mine",
                    "paths": ["notes.txt", "photo.png"],
                }
            )
            + "\n"
        )
        yield json.dumps({"op": "add_files", "id": 3, "paths": ["notes.txt"]}) + "\n"
        yield json.dumps({"op": "add_files", "id": 4, "base": "mine", "paths": ["../x.txt"]}) + "\n"
        yield (
            json.dumps(
                {"op": "open", "id": 5, "effort": 1, "bases": {"mine": 1.0}, "max_sources": 2}
            )
            + "\n"
        )

    results = io.StringIO()
    serve_multitrack_jsonl(runtime, requests(results), results, kb_root=root)
    by_id = {row["id"]: row for row in map(json.loads, results.getvalue().splitlines())}
    assert by_id[2]["collection"] == {"name": "main", "added": 2}
    assert by_id[2]["base"]["added"] == 2 and by_id[2]["base"]["edges"] == 2
    assert not by_id[3]["ok"] and "collection name, a base name" in by_id[3]["error"]
    assert not by_id[4]["ok"] and "outside the knowledge base root" in by_id[4]["error"]
    assert by_id[5]["ok"] and by_id[5]["retrieved"]


def test_private_writes_with_collection_and_failure_restore_sources(tmp_path, monkeypatch):
    import pytest

    from megatoken.model.transducer import HindsightValues
    from megatoken.serving_writes import WritePolicy

    runtime = _runtime(tmp_path, FakePreparer(tmp_path / "scratch"))
    memory = runtime.memory
    base = memory.create_overlay("private", "alice", tmp_path / "private.kb")

    def values(events, reads):
        return HindsightValues(
            torch.ones(len(reads), memory.bank.geometry.value_dim),
            torch.zeros(len(reads), dtype=torch.bool),
        )

    monkeypatch.setattr(memory.native, "retrospective_values", values)
    session = runtime.open(
        overlay="private", scope="alice", max_sources=1, write_policy=WritePolicy(lag=0)
    )
    runtime.observe(session, _tool())
    sources = session.memory.sources
    before = (
        list(sources.log),
        {n: set(v) for n, v in sources.observed.items()},
        sources.last,
        dict(sources.search_snapshots),
    )
    state = session.state
    add_edges = base.add_edges

    def fail(**kwargs):
        raise RuntimeError("write failed")

    monkeypatch.setattr(base, "add_edges", fail)
    with pytest.raises(RuntimeError, match="write failed"):
        runtime.observe(session, _tool())
    assert session.state is state and sources.log == before[0]
    assert sources.observed == before[1] and sources.last is before[2]
    assert sources.search_snapshots == before[3]
    monkeypatch.setattr(base, "add_edges", add_edges)
    runtime.observe(session, _tool())
    assert base.edge_count > 0
    assert [m.name for m in session.memory.agenda.mounted] == ["trained", "private"]
    runtime.close()
