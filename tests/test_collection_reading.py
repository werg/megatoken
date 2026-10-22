"""Reading from a collection: the query path in training and serving, its losses and recall."""

from __future__ import annotations

import json

import pytest
import torch
from retrieval_support import (
    STREAM,
    forward,
    reading_batches,
    reading_root,
    tiny_model,
)
from training_fixture import tiny_train_config

from megatoken.data import PreparedCorpus
from megatoken.generation.codec import ByteCodec
from megatoken.training import meridians
from megatoken.training.meridians import _metadata

CODEC = ByteCodec.byte_fallback()


@pytest.fixture(autouse=True)
def single_thread():
    torch.set_num_threads(1)


def test_an_annotated_window_trains_the_query_head_and_observes_what_it_retrieves(
    tmp_path, monkeypatch
):
    root = reading_root(tmp_path)
    config = tiny_train_config(tmp_path / "run", phase="P1", representation_mode="recurrent")
    model = tiny_model()
    model.train()
    _, batches = reading_batches(model, root, config)
    (batch,) = batches
    assert batch.counts["query_alignment"] == 1 and batch.counts["retrieval_ranking"] == 1
    sessions = []
    mount = meridians.mount
    monkeypatch.setattr(
        meridians, "mount", lambda *args: sessions.append(mount(*args)) or sessions[-1]
    )
    model.core.set_model_version(0)
    (metrics,), ledger = forward(model, batches, config)
    assert ledger.seen["query_alignment"] == 1 and ledger.seen["retrieval_ranking"] == 1
    # The loss reaches the embedding head's query role, and only that role.
    head = model.embedding_realizer
    assert head.query.weight.grad.abs().sum() > 0 and head.document.weight.grad is None
    # With the threshold open the first lookup observed two passages, in the collection's order.
    (session,) = sessions
    assert [entry.lookup for entry in session.log][:2] == [0, 0]
    assert all(entry.item.text.startswith(b"passage") for entry in session.log)
    assert 0 <= metrics["retrieval_student_recall"] <= 1 and metrics["retrieval_queries"] == 1
    # The same question is the request of the session's metadata, kept whatever the dropout.
    episode = PreparedCorpus([str(root)], stream=STREAM).fetch(0)[0]
    for seed in (None, 1, 2, 3):
        request = json.loads(_metadata(episode, seed))["request"]
        assert request == "What is the code word of task 0?"
