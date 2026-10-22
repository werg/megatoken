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
    # Adapter weights start at zero in a fresh model; give them an effect to observe.
    for owner in model.core.adapters.owners.values():
        torch.nn.init.normal_(owner.up, std=0.5)
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

    class Engine:
        device = torch.device("cpu")
        config = model.config

    engine = Engine()
    engine.model = model

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
    serve_multitrack_jsonl(
        engine,
        requests(results),
        results,
        collections={"main": collection_root / COLLECTION},
        kb_root=root,
        preparer=lambda: FakePreparer(root / "prepared"),
    )
    by_id = {row["id"]: row for row in map(json.loads, results.getvalue().splitlines())}
    assert by_id[2]["collection"] == {"name": "main", "added": 2}
    assert by_id[2]["base"]["added"] == 2 and by_id[2]["base"]["edges"] == 2
    assert not by_id[3]["ok"] and "collection name, a base name" in by_id[3]["error"]
    assert not by_id[4]["ok"] and "outside the knowledge base root" in by_id[4]["error"]
    assert by_id[5]["ok"] and by_id[5]["retrieved"]
