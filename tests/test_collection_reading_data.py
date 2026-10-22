"""The collection_reading producer: episodes without documents, collections per split."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch
from retrieval_support import CODEC, FakePreparer, forward, reading_batches, tiny_model
from training_fixture import tiny_train_config

from megatoken.data import PreparedCorpus
from megatoken.data.open_media import frozen_split
from megatoken.retrieval.annotation import RetrievalAnnotation
from megatoken.retrieval.collection import SourceCollection

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/prepare_collection_reading.py"


@pytest.fixture(scope="module")
def producer():
    spec = importlib.util.spec_from_file_location("prepare_collection_reading", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def task_ids(split: str, count: int) -> list[str]:
    return [f"t{i}" for i in range(2000) if frozen_split(f"t{i}") == split][:count]


def tasks(tmp_path):
    photo = tmp_path / "photo.png"
    photo.write_bytes(b"pixels")
    rows = []
    for number, identity in enumerate(task_ids("train", 5) + task_ids("validation", 2)):
        rows.append(
            {
                "id": identity,
                "question": f"What is the code word of task {number}?",
                "answer": f"word{number}",
                "documents": [
                    f"Task {number} note: the code word is word{number}.",
                    f"A distractor for task {number}.",
                ],
                "supporting": [0],
            }
        )
    rows.append(
        {
            "id": task_ids("train", 6)[5],
            "question": "Describe the image that matches this caption: a red kite",
            "request": "a red kite",
            "answer": "A red kite over a field.",
            "documents": [{"image": str(photo)}],
            "supporting": [0],
        }
    )
    return rows


def test_episodes_hold_the_question_and_the_collection_holds_the_support(producer, tmp_path):
    root = tmp_path / "prepared"
    rows = tasks(tmp_path)
    report = producer.build(rows, FakePreparer(tmp_path / "scratch"), root, CODEC)
    assert report["collection_reading"] == 8 and set(report["collections"]) == {
        "train",
        "validation",
    }
    assert report["collections"]["train"]["items"] == 5 * 2 + 1
    assert report["collections"]["validation"]["items"] == 2 * 2
    for split, count in (("train", 6), ("validation", 2)):
        with PreparedCorpus([str(root)], split=split, stream="collection_reading") as corpus:
            assert len(corpus) == count
            episode, assets = corpus.fetch(0)
            assert [record.role for record in episode.records] == ["user", "assistant"]
            annotation = RetrievalAnnotation.of(episode)
            assert annotation.collections == (f"collections/{split}",)
            with SourceCollection.open(root / annotation.collections[0]) as collection:
                # The annotated item is in this split's own collection.
                (row,) = collection.find(annotation.positives).values()
                assert collection.item(row).origin and collection.query_center is not None
    # The image task's request is its caption, and its support is the image file.
    with SourceCollection.open(root / "collections/train") as collection:
        images = [i for i in map(collection.item, range(1, 12)) if i.modality == "image"]
        assert [i.origin for i in images] == [str(tmp_path / "photo.png")]


def test_a_produced_root_trains_the_query_losses(producer, tmp_path):
    torch.set_num_threads(1)
    root = tmp_path / "prepared"
    producer.build(tasks(tmp_path), FakePreparer(tmp_path / "scratch"), root, CODEC)
    config = tiny_train_config(tmp_path / "run", phase="P1", representation_mode="recurrent")
    model = tiny_model()
    model.train()
    _, batches = reading_batches(model, root, config)
    assert all(b.counts["query_alignment"] == 1 for b in batches)
    (metrics,), ledger = forward(model, batches[:1], config)
    assert ledger.seen["query_alignment"] == 1 == ledger.seen["retrieval_ranking"]
    assert metrics["retrieval_queries"] == 1
    assert model.embedding_realizer.query.weight.grad.abs().sum() > 0
