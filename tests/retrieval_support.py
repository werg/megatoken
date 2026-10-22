"""Test doubles for reading from a collection: a prepared root whose episodes carry
only a question and its answer, with the supporting passages in a collection."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
from media_support import prepared_record
from training_fixture import MODEL_OVERRIDES

from megatoken.data import (
    DedupConfig,
    DedupStore,
    Episode,
    Lineage,
    PhysicalRecord,
    PreparedCorpus,
    iter_windows,
)
from megatoken.data.media_assets import TensorAssetStore
from megatoken.data.schema import Supervision, sequential_meridians
from megatoken.data.storage import export_prepared
from megatoken.generation.codec import ByteCodec
from megatoken.io import canonical_json
from megatoken.loading import build_model
from megatoken.model import ModelConfig
from megatoken.omni.embeddinggemma2 import TeacherOutput
from megatoken.retrieval.annotation import KEY, RetrievalAnnotation
from megatoken.retrieval.collection import SourceCollection, text_item_id
from megatoken.retrieval.index import unit
from megatoken.teacher import EMBEDDING_WIDTH, HIDDEN_WIDTH, PINNED_TEACHER
from megatoken.training.engine import prepare_batches, window_config
from megatoken.training.losses import LossLedger
from megatoken.training.meridians import forward_meridian_batch
from megatoken.training.stream import EpisodeWindow
from megatoken.training.transducer import TransducerTrainer

CODEC = ByteCodec.byte_fallback()
STREAM = "collection_reading"
COLLECTION = "collections/main"
DISTRACTORS = 6


def passage(task: int, number: int) -> bytes:
    return (
        f"passage {task}.{number}: the code word of task {task} is word{task * 3 + number}".encode()
    )


def reading_root(
    root: Path, *, tasks: int = 3, first: int = 0, heldout: bool = False, threshold: float = -1.0
) -> Path:
    """A prepared root with ``tasks`` reading episodes and one collection of all their passages.

    Task t asks for the code word of t; passage t.0 supports it and the others are
    distractors. Teacher embeddings are random unit vectors, the support passage's
    key lies near its question's.
    """
    rng = np.random.default_rng(1234 + first)
    data = root / "data"
    store = TensorAssetStore(data)
    collection = SourceCollection.create(data / COLLECTION, name="main", teacher=PINNED_TEACHER)
    queries, episodes = [], []
    for task in range(first, first + tasks):
        query = unit(rng.standard_normal(EMBEDDING_WIDTH))
        queries.append(query)
        for number in range(DISTRACTORS):
            key = (
                unit(query + 0.4 * rng.standard_normal(EMBEDDING_WIDTH))
                if number == 0
                else unit(rng.standard_normal(EMBEDDING_WIDTH))
            )
            collection.add_text(passage(task, number), key, origin=f"corpus/{task}-{number}.txt")
        question = f"What is the code word of task {task}?".encode()
        asset = store.put(
            query,
            modality="text",
            kind="embedding",
            extractor=PINNED_TEACHER,
            source_family=hashlib.sha256(question).hexdigest(),
        )
        annotation = RetrievalAnnotation(
            (COLLECTION,), question.decode(), asset, (text_item_id(passage(task, 0)),)
        )
        records = (
            PhysicalRecord(f"q{task}", "user", question),
            PhysicalRecord(
                f"a{task}", "assistant", f"word{task * 3}".encode(), turn_index=1, policy_mask=True
            ),
        )
        episodes.append(
            sequential_meridians(
                Episode(
                    f"reading-{task}",
                    records,
                    Lineage("reading", "v1", f"family-{task}"),
                    stream=STREAM,
                    supervision=Supervision(
                        annotations_json=canonical_json({KEY: annotation.to_json()}).decode()
                    ),
                )
            )
        )
    collection.reindex()
    collection.set_threshold(threshold)
    collection.set_query_center(np.mean(queries, axis=0))
    collection.close()
    codec = ByteCodec.byte_fallback()
    dedup = DedupStore(
        root / "spool.sqlite",
        DedupConfig(train_fraction=0 if heldout else 1, validation_fraction=1 if heldout else 0),
    )
    try:
        with dedup.transaction():
            for episode in episodes:
                dedup.add(episode, codec)
        dedup.freeze()
        export_prepared(
            dedup, data, codec=codec, source_receipts=[], pipeline_fingerprint="reading-test-v1"
        )
    finally:
        dedup.close()
    return data


def tiny_model(seed: int = 3):
    torch.manual_seed(seed)
    return build_model(ModelConfig.tiny(**MODEL_OVERRIDES), CODEC)


def reading_batches(model, root: Path, config):
    corpus = PreparedCorpus([str(root)], stream=STREAM)
    episode, assets = corpus.fetch(0)
    windows = [
        EpisodeWindow(episode, window, assets)
        for window in iter_windows(episode, CODEC, window_config(config))
    ]
    return episode, prepare_batches(model, windows, config)


def forward(model, batches, config, *, differentiate=True):
    trainer = TransducerTrainer(model.transducer)
    metrics = []
    for batch in batches:
        ledger = LossLedger(dict(batch.counts), dict(batch.counts), config.loss_weights)
        metrics.append(
            forward_meridian_batch(
                model,
                batch,
                ledger,
                config,
                torch.device("cpu"),
                0,
                transducer=trainer,
                differentiate=differentiate,
            )
        )
        ledger.finish()
    return metrics, ledger


class FakeTeacher:
    """A teacher whose embedding of a text is a fixed random unit vector of that text."""

    limit = 5000

    def text_pieces(self, text, *, prompt=""):
        return [text[i : i + self.limit] for i in range(0, len(text), self.limit)]

    def encode(self, modality, source, *, prompt=""):
        seed = int(hashlib.sha256((prompt + source).encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        embedding = torch.nn.functional.normalize(torch.from_numpy(rng.standard_normal(768)), dim=0)
        features = torch.from_numpy(rng.standard_normal((4, HIDDEN_WIDTH)).astype(np.float32))
        return TeacherOutput(features, embedding.float(), modality, PINNED_TEACHER)

    def encode_batch(self, items, *, prompt=""):
        return [self.encode(modality, source, prompt=prompt) for modality, source in items]


class FakePreparer:
    def __init__(self, root):
        self.teacher, self.store = FakeTeacher(), TensorAssetStore(root)

    def media(self, modality, path, *, target=True):
        seed = int(hashlib.sha256(Path(path).read_bytes()).hexdigest()[:6], 16)
        return prepared_record(self.store, modality, seed=seed, target=False)
