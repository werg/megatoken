"""The query path of a lookup in serving, and the same lookup in training."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from media_support import prepared_record
from retrieval_support import (
    COLLECTION,
    forward,
    reading_batches,
    reading_root,
    tiny_model,
)
from training_fixture import tiny_train_config

from megatoken.contracts import ROLE_TO_ID
from megatoken.data.media_assets import TensorAssetStore
from megatoken.model import sources as source_path
from megatoken.model.config import KNOWLEDGE_TRACK
from megatoken.model.multitrack import MultiTrackGenerator, ObservedEvent
from megatoken.multitrack_runtime import MultiTrackRuntime
from megatoken.retrieval.collection import SourceCollection
from megatoken.retrieval.index import unit
from megatoken.teacher import PINNED_TEACHER
from megatoken.training import meridians
from megatoken.training.meridians import _metadata


@pytest.fixture(autouse=True)
def single_thread():
    torch.set_num_threads(1)


def serving_runtime(model, root: Path, **options):
    mounted = {"main": root / COLLECTION}
    return MultiTrackRuntime(MultiTrackGenerator(model), collections=mounted, **options)


def first_lookup(log):
    return [(entry.item.item_id, entry.score) for entry in log if entry.lookup == 0]


def test_training_and_serving_retrieve_the_same_items(tmp_path, monkeypatch):
    root = reading_root(tmp_path)
    config = tiny_train_config(tmp_path / "run", phase="P1", representation_mode="recurrent")
    model = tiny_model()
    episode, batches = reading_batches(model, root, config)
    sessions = []
    mount = meridians.mount
    monkeypatch.setattr(
        meridians, "mount", lambda *args: sessions.append(mount(*args)) or sessions[-1]
    )
    model.eval()
    with torch.no_grad():
        forward(model, batches, config, differentiate=False)
    served = serving_runtime(model, root).open(
        json.loads(_metadata(episode, None)), effort=config.effort
    )
    trained, answered = first_lookup(sessions[0].log), first_lookup(served.memory.sources.log)
    assert len(trained) == 2 == len(answered)
    assert [item for item, _ in trained] == [item for item, _ in answered]
    assert [score for _, score in trained] == pytest.approx([score for _, score in answered])


def test_a_session_never_observes_an_item_twice(tmp_path):
    root = reading_root(tmp_path)
    runtime = serving_runtime(tiny_model(), root)
    session = runtime.open({"request": "code words"}, effort=1)
    for _ in range(6):
        event = ObservedEvent("main", "text", b"x", role_id=ROLE_TO_ID["tool"])
        runtime.observe(session, (event,), effort=1)
    log = session.memory.sources.log
    assert [entry.lookup for entry in log] == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6]
    assert len({entry.item.row for entry in log}) == len(log)
    # The original file of every item is there for the caller to return.
    assert all(entry.item.origin.startswith("corpus/") for entry in log)


def test_a_cancelled_lookup_observes_nothing_and_records_nothing(tmp_path):
    root = reading_root(tmp_path)
    runtime = serving_runtime(tiny_model(), root)
    outcomes = set()

    def stop_after(limit):
        calls = 0

        def cancelled():
            nonlocal calls
            calls += 1
            return calls > limit

        return cancelled

    for limit in range(1, 14):
        session = runtime.memory.new_session({}, effort=1, cancelled=stop_after(limit))
        meridians_seen = int(session.state.meridians_seen[0])
        log = session.sources.log
        if session.agenda.lookups == 0:
            # Metadata only: no query, knowledge or item event, and nothing recorded.
            assert meridians_seen == 1 and log == [] and not any(session.sources.observed.values())
        else:
            # The query, the knowledge block and the two items committed together.
            assert meridians_seen == 5 and len(log) == 2
        outcomes.add(session.agenda.lookups)
    assert outcomes == {0, 1}


def test_media_items_are_observed_in_their_own_medium_on_the_knowledge_track(tmp_path, monkeypatch):
    store = TensorAssetStore(tmp_path / "prepared")
    collection = SourceCollection.create(tmp_path / "mixed", name="mixed", teacher=PINNED_TEACHER)
    for seed, modality in ((1, "image"), (2, "text")):
        record = prepared_record(store, modality, seed=seed, target=False)
        collection.add_record(record, store, origin=f"{seed}.{modality}")
    key = unit(np.random.default_rng(0).standard_normal(768))
    collection.add_text(b"a short passage", key, origin="c.txt")
    collection.reindex()
    collection.set_threshold(-1.0)
    collection.close()
    seen = []
    event = source_path.source_event
    monkeypatch.setattr(
        source_path, "source_event", lambda *args: seen.append(event(*args)) or seen[-1]
    )
    runtime = MultiTrackRuntime(
        MultiTrackGenerator(tiny_model()), collections={"mixed": tmp_path / "mixed"}
    )
    session = runtime.open({}, effort=1, max_sources=3)
    assert {event.modality for event in seen} == {"image", "compressed_text", "text"}
    assert all(
        event.track == KNOWLEDGE_TRACK and event.role_id == ROLE_TO_ID["document"] for event in seen
    )
    media = [event for event in seen if event.modality != "text"]
    assert all(event.payload.shape[0] == 1 and event.extractor == PINNED_TEACHER for event in media)
    # Each item is its own meridian after the metadata, the query and the knowledge block.
    assert int(session.state.meridians_seen[0]) == 3 + 3
    assert session.state.frontier_tracks == (KNOWLEDGE_TRACK,)


def test_the_jsonl_service_mounts_collections_reports_originals_and_adds_files(tmp_path):
    import io
    import json

    from retrieval_support import FakePreparer

    from megatoken.multitrack_service import serve_multitrack_jsonl

    root = reading_root(tmp_path)
    model = tiny_model()
    (tmp_path / "new.txt").write_text("A fact that arrived while serving.")

    runtime = MultiTrackRuntime(
        MultiTrackGenerator(model),
        collections={"main": root / COLLECTION},
        preparer=lambda: FakePreparer(tmp_path / "scratch"),
    )

    def requests(results):
        yield (
            json.dumps(
                {
                    "op": "add_files",
                    "id": 0,
                    "collection": "main",
                    "paths": [str(tmp_path / "new.txt")],
                }
            )
            + "\n"
        )
        yield json.dumps({"op": "open", "id": 1, "effort": 1, "max_sources": 3}) + "\n"

    results = io.StringIO()
    serve_multitrack_jsonl(runtime, requests(results), results)
    added, opened = [json.loads(line) for line in results.getvalue().splitlines()]
    assert added["ok"] and added["collection"] == {"name": "main", "added": 1}
    assert opened["ok"] and len(opened["retrieved"]) == 3
    first = opened["retrieved"][0]
    assert set(first) == {
        "lookup",
        "collection",
        "item",
        "event_id",
        "modality",
        "origin",
        "score",
        "publication_revision",
        "available_at_ns",
        "snapshot_revision",
        "searched_at_ns",
    }
    assert first["publication_revision"] <= first["snapshot_revision"]
    assert first["available_at_ns"] <= first["searched_at_ns"]
    assert first["event_id"].startswith("source:")
    assert len({entry["event_id"] for entry in opened["retrieved"]}) == 3
    assert first["collection"] == "main" and first["lookup"] == 0


def test_retrieval_is_deterministic_and_reads_only_the_past(tmp_path):
    root = reading_root(tmp_path)
    model = tiny_model()
    runtime = serving_runtime(model, root)
    metadata = {"request": "What is the code word of task 1?"}
    plain = runtime.open(metadata, effort=1)
    busy = runtime.open(metadata, effort=1)
    # Whatever happens afterwards, the first lookup saw only the metadata.
    runtime.observe(
        busy, (ObservedEvent("main", "text", b"later", role_id=ROLE_TO_ID["tool"]),), effort=1
    )
    assert first_lookup(plain.memory.sources.log) == first_lookup(busy.memory.sources.log)
    again = serving_runtime(model, root).open(metadata, effort=1)
    assert first_lookup(again.memory.sources.log) == first_lookup(plain.memory.sources.log)


def test_the_generate_command_reads_a_mounted_collection_after_the_question(tmp_path, capsys):
    from test_served import checkpoint_model

    from megatoken.cli import main
    from megatoken.served import export_served

    _, checkpoint = checkpoint_model(tmp_path / "checkpoint")
    bundle = tmp_path / "served"
    export_served(checkpoint, bundle)
    root = reading_root(tmp_path / "corpus")
    argv = ["generate", "--model", str(bundle), "--device", "cpu", "--effort", "1"]
    argv += ["--prompt", "What is the code word of task 1?", "--max-tokens", "4"]
    assert main([*argv, "--collection", f"main={root / COLLECTION}"]) == 0
    answer = json.loads(capsys.readouterr().out)
    # The threshold is open: the lookup after the metadata, and the one after the finished
    # question, each observed passages. The session says which original files they came from.
    assert {item["lookup"] for item in answer["retrieved"]} >= {0, 1}
    assert all(item["collection"] == "main" and item["origin"] for item in answer["retrieved"])
    assert answer["tape"]["tracks"] == ["assistant"]
