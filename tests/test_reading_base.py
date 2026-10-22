"""Ingest heads trained inside the hindsight pass; the reading-from-a-base stream and its measurement."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from media_support import prepared_record
from torch.nn import functional as F
from training_fixture import prepare_dataset, tiny_train_config

from megatoken.contracts import ROLE_TO_ID
from megatoken.data import Lineage, PhysicalRecord, WindowConfig, iter_windows
from megatoken.data.knowledge import (
    STREAM,
    declared_items,
    episode_items,
    episode_metadata,
    knowledge_episode,
)
from megatoken.data.media_assets import MEDIA_RECORD_TYPE, TensorAssetStore
from megatoken.data.schema import Episode, sequential_meridians
from megatoken.generation.codec import ByteCodec
from megatoken.loading import build_model
from megatoken.model import ModelConfig
from megatoken.model.multitrack import MultiTrackGenerator
from megatoken.model.transducer import metadata_payload
from megatoken.omni.embeddinggemma2 import TeacherOutput
from megatoken.teacher import HIDDEN_WIDTH, PINNED_TEACHER
from megatoken.training.engine import prepare_batches
from megatoken.training.heldout import MOUNTED, UNMOUNTED, HeldOutEvaluator
from megatoken.training.losses import LossLedger
from megatoken.training.meridians import _boundary_meridian, forward_meridian_batch
from megatoken.training.reading_base import ReadingBase
from megatoken.training.stream import EpisodeWindow
from megatoken.training.transducer import (
    INGEST_KEY_TEMPERATURE,
    Meridian,
    TransducerTrainer,
    plan_boundaries,
)
from megatoken.training.transducer_diloco import learner_round

CODEC = ByteCodec.byte_fallback()


class CountingTeacher:
    """One feature row per 64 bytes; pieces of at most ``limit`` characters stand in for the
    real teacher's 8192-token pieces."""

    def __init__(self, limit=600):
        self.limit = limit

    def text_pieces(self, text, *, prompt=""):
        return [text[i : i + self.limit] for i in range(0, len(text), self.limit)]

    def encode(self, modality, source, *, prompt=""):
        generator = torch.Generator().manual_seed(len(source))
        return TeacherOutput(
            torch.randn(max(1, len(source) // 64), HIDDEN_WIDTH, generator=generator),
            F.normalize(torch.ones(768), dim=0),
            modality,
            PINNED_TEACHER,
        )


def tiny_model():
    torch.manual_seed(6)
    return build_model(ModelConfig.tiny(), CODEC).train()


def reading_episode(store: TensorAssetStore, index: int, *, items: int = 3, modality="text"):
    """A question about items that are in a base, not in the context."""
    records = [
        (prepared_record(store, modality, seed=100 * index + k, target=False), "doc", k, f"s{k}")
        for k in range(items)
    ]
    return knowledge_episode(
        f"reading-{index}",
        f"Question {index}: which key opens the vault?",
        f"Answer {index}: key {index * 7 + 3}",
        records,
        Lineage(STREAM, "v1", f"family-{index}"),
    )


def window_of(episode, store, config=None):
    window = next(iter_windows(episode, CODEC, WindowConfig(max_records=8, max_decisions=4)))
    return EpisodeWindow(episode, window, store)


def train_config(tmp_path, **overrides):
    values = dict(
        phase="P1",
        representation_mode="recurrent",
        generated_capacity=64,
        max_record_bytes=128,
        microbatch_episodes=1,
        max_accumulation_windows=1,
    )
    return tiny_train_config(tmp_path, **{**values, **overrides})


# ------------------------------------------------------------ the episodes


def test_a_reading_episode_holds_the_question_and_the_answer(tmp_path):
    store = TensorAssetStore(tmp_path)
    episode = reading_episode(store, 1)
    assert [(r.role, r.policy_mask) for r in episode.records] == [
        ("user", False),
        ("assistant", True),
    ]
    assert episode.stream == STREAM
    assert not any(r.media_type == MEDIA_RECORD_TYPE for r in episode.records)  # none in context
    assert [row["part"] for row in declared_items(episode)] == [0, 1, 2]
    items = episode_items(episode, store, episode_metadata(episode))
    assert [item.part for item in items] == [0, 1, 2] and {i.document for i in items} == {
        "reading-1:doc"
    }
    assert all(i.modality == "compressed_text" for i in items)
    assert episode_metadata(episode)["task"] == STREAM
    # The finished question is a boundary: a lookup falls between question and answer.
    steps = [_boundary_meridian(_step(r)) for r in episode.records]
    assert plan_boundaries(steps, max_lookups=16, generator=None) == [1, 1]
    with pytest.raises(ValueError, match="at least one item"):
        knowledge_episode("x", "q", "a", [], Lineage(STREAM, "v1", "f"))


def _step(record):
    return SimpleNamespace(
        tracks=(record.role,),
        roles=(ROLE_TO_ID[record.role],),
        modalities=("text",),
        records=(SimpleNamespace(record=record),),
    )


# --------------------------------------------------- the ingest heads' losses


def hindsight_sequence(step: int, shape):
    """A training window through the hindsight pass: one meridian per ``(role, modalities)``."""
    model = tiny_model()
    c = model.config
    meridians = [
        Meridian(
            tuple(f"t{i}.{j}" for j in range(len(kinds))),
            (ROLE_TO_ID[role],) * len(kinds),
            kinds,
            role in ("document", "assistant"),
        )
        for i, (role, kinds) in enumerate(shape)
    ]
    trainer = TransducerTrainer(model.transducer)
    sequence = trainer.sequence(meridians, step=step, generator=torch.Generator().manual_seed(0))
    torch.manual_seed(1)
    sequence.hindsight(
        meridians, lambda index: torch.randn(1, len(shape[index][1]), c.n_facets, c.facet_dim)
    )
    return model, trainer, sequence


TEXT, ITEM = ("text",), ("compressed_text",)
DOCUMENT = [
    ("user", TEXT),
    ("document", ITEM),
    ("document", ITEM),
    ("document", ITEM),
    ("assistant", TEXT),
]


def test_a_lookup_matches_the_item_its_hindsight_group_is_or_contains():
    # Full pass: lookup t reads its own group; group 0 holds the question and one item.
    _, _, full = hindsight_sequence(0, DOCUMENT)
    assert full.items.matches == ((0, 0), (1, 1), (2, 2))
    # Gap 2: lookup t reads the group after; the last two have no item or lie past the end.
    _, _, gap = hindsight_sequence(8192, DOCUMENT)
    assert gap.items.matches == ((0, 1), (1, 2))
    assert gap.items.values.shape == (3, gap.native.bank.geometry.value_dim)
    assert gap.items.keys.shape == (3, gap.native.bank.geometry.query_dim)
    # A group of two items has no single item to learn from; neither has one without items.
    two = [("user", TEXT), ("document", ITEM + ITEM), ("assistant", TEXT)]
    assert hindsight_sequence(0, two)[2].items is None
    assert hindsight_sequence(0, [("user", TEXT), ("assistant", TEXT)])[2].items is None


def test_ingest_losses_distil_the_hindsight_value_and_contrast_the_keys():
    model, _, sequence = hindsight_sequence(8192, DOCUMENT)
    multi = MultiTrackGenerator(model)
    state = sequence.begin(multi, multi.tracks.initial_state(1), metadata_payload({"task": "t"}))
    for index in range(len(DOCUMENT)):
        state = sequence.before_meridian(multi, state, index, effort=1)
    losses = sequence.ingest_losses()
    assert set(losses) == {"ingest_value", "ingest_key"}
    window, pairs = sequence.items, list(sequence.items.matches)
    target = torch.stack([sequence.values.values[t].detach() for t, _ in pairs])
    item = torch.tensor([k for _, k in pairs])
    relative = (window.values[item] - target).square().sum(-1) / target.square().sum(-1)
    torch.testing.assert_close(losses["ingest_value"], relative.mean())
    queries = torch.stack([sequence.queries[t] for t, _ in pairs])
    logits = queries @ window.keys.T / INGEST_KEY_TEMPERATURE
    torch.testing.assert_close(losses["ingest_key"], F.cross_entropy(logits, item))
    # Only the ingest heads learn from them: the hindsight head is the teacher, the query is a target.
    (losses["ingest_value"] + losses["ingest_key"]).backward()
    heads = model.transducer.ingest
    assert heads.value.output.weight.grad is not None and heads.key.project.weight.grad is not None
    assert model.transducer.hindsight_realizer.output.weight.grad is None
    assert model.transducer.query_realizer.query.weight.grad is None


def test_only_lookups_actually_made_are_paired_and_one_item_has_nothing_to_contrast():
    model, _, sequence = hindsight_sequence(0, DOCUMENT)
    multi = MultiTrackGenerator(model)
    state = sequence.begin(multi, multi.tracks.initial_state(1), metadata_payload({"task": "t"}))
    sequence.before_meridian(multi, state, 0, effort=1)  # only the first lookup happens
    assert len(sequence.queries) == 1
    # The one lookup made is matched with its item, and still contrasted with the other items.
    assert set(sequence.ingest_losses()) == {"ingest_value", "ingest_key"}
    # With a single item in the window there is nothing to contrast it with.
    _, _, alone = hindsight_sequence(0, [("user", TEXT), ("document", ITEM), ("assistant", TEXT)])
    multi = MultiTrackGenerator(model)
    state = alone.begin(multi, multi.tracks.initial_state(1), metadata_payload({"task": "t"}))
    alone.before_meridian(multi, state, 0, effort=1)
    assert set(alone.ingest_losses()) == {"ingest_value"}


# ------------------------------------------------------- the reading base


def test_each_episodes_items_are_ingested_into_a_fixed_base_and_mounted(tmp_path):
    model = tiny_model()
    multi = MultiTrackGenerator(model)
    store = TensorAssetStore(tmp_path)
    reading = ReadingBase(model.transducer, capacity=2, refresh_every=100)
    first = reading_episode(store, 1)
    mounted = reading.mount(multi, first, store, step=0)
    assert [m.name for m in mounted] == ["trained", "reading:reading-1"]
    base = reading.segments["reading-1"]
    assert base.edge_count == 3 and not base.writable
    # The three parts of one document form a chain written by the current heads.
    assert len(set(base.sources.values())) == 3
    for index in (2, 3):
        reading.mount(multi, reading_episode(store, index), store, step=1)
    assert list(reading.segments) == ["reading-2", "reading-3"]  # the oldest left
    assert reading.mount(multi, reading_episode(store, 3), store, step=2) is not None
    kept = reading.segments["reading-3"]
    assert reading.mount(multi, reading_episode(store, 3), store, step=3) and (
        reading.segments["reading-3"] is kept
    )
    # After refresh_every steps every entry is rewritten with the heads as they are now.
    reading.mount(multi, reading_episode(store, 3), store, step=100)
    assert list(reading.segments) == ["reading-3"] and reading.segments["reading-3"] is not kept
    # An episode that declares no items searches the trained base alone.
    plain = Episode("plain", (PhysicalRecord("q", "user", b"hi"),), Lineage("s", "v", "f"))
    assert reading.mount(multi, sequential_meridians(plain), store, step=0) is None


def test_a_window_that_mounts_a_base_learns_nothing_into_the_trained_bank(tmp_path):
    model = tiny_model()
    store = TensorAssetStore(tmp_path)
    episode = reading_episode(store, 1)
    config = train_config(tmp_path)
    model.core.set_model_version(0)
    (batch,) = prepare_batches(model, [window_of(episode, store)], config)
    trainer = TransducerTrainer(model.transducer)
    snapshot = model.transducer.bank.state_snapshot()
    ledger = LossLedger(batch.counts, batch.counts, config.loss_weights)
    metrics = forward_meridian_batch(
        model, batch, ledger, config, torch.device("cpu"), 0, transducer=trainer
    )
    ledger.finish()
    assert metrics["transducer_lookups"] == 2
    assert [r.writes for r in trainer.records] == [False, False]
    # The lookup before the answer searched the ingested items (and nothing was written).
    assert {name for r in trainer.records for _, name, _ in r.lookup.pairs} >= {"reading:reading-1"}
    assert all(r.lookup.edge_ids.numel() == 0 for r in trainer.records)
    trainer.after_backward(0)
    assert trainer.candidates == [] and trainer.barrier(1)["transducer_writeouts"] == 0
    # The trained bank is exactly as it was, and what a checkpoint holds has no ingested base.
    after = model.transducer.bank.state_snapshot()
    assert after["rows"]["ids"].numel() == snapshot["rows"]["ids"].numel() == 0
    # Nothing of it reaches DiLoCo: a learner's round for this window is empty.
    submitted = learner_round(
        trainer.optimizer,
        trainer.growth,
        worker=0,
        sequence=1,
        base_version=0,
        examples=1,
        step=1,
    )
    assert all(ids.numel() == 0 for ids, _ in submitted.deltas.values())
    assert submitted.write_ids.numel() == 0 and not submitted.deltas.get("value")
    assert set(trainer.state_dict()) == {"bank", "optimizer", "growth", "sequences", "switch"}
    assert trainer.state_dict()["bank"]["rows"]["ids"].numel() == 0


def test_ingest_losses_reach_the_heads_through_a_whole_training_window(tmp_path):
    from megatoken.data.compression import CompressionPolicy, DocumentCompressor
    from megatoken.data.schema import stable_id

    model = tiny_model()
    store = TensorAssetStore(tmp_path)
    compressor = DocumentCompressor(CountingTeacher(), store, CompressionPolicy(min_bytes=1))
    document = ("The key to the vault is kept in the north cabinet. " * 36).encode()
    records = (
        PhysicalRecord("question", "user", b"where is the key?"),
        *compressor.records(document, logical_id="doc", role="document"),
        PhysicalRecord("answer", "assistant", b"the north cabinet", turn_index=1, policy_mask=True),
    )
    assert sum(r.media_type == MEDIA_RECORD_TYPE for r in records) >= 3
    episode = sequential_meridians(
        Episode(stable_id("e", "1"), records, Lineage("long_document_reading", "v1", "f"))
    )
    config = train_config(tmp_path, max_history_tokens=512, max_record_bytes=4096)
    model.core.set_model_version(0)
    (batch, *_) = prepare_batches(model, [window_of(episode, store)], config)
    ledger = LossLedger(batch.counts, batch.counts, config.loss_weights)
    metrics = forward_meridian_batch(
        model,
        batch,
        ledger,
        config,
        torch.device("cpu"),
        0,
        transducer=TransducerTrainer(model.transducer),
    )
    assert torch.isfinite(metrics["ingest_value_loss"]) and "ingest_key_loss" in metrics
    heads = model.transducer.ingest
    assert heads.value.queries.grad is not None and heads.key.pool.grad is not None


# ----------------------------------------------- the engine, the producer, the measurement


def reading_dataset(root, count, *, heldout=False):
    """Prepared reading episodes; their items live under the prepared root, as the producer writes."""
    store = TensorAssetStore(root / "data")
    episodes = [reading_episode(store, i) for i in range(count)]
    return prepare_dataset(root, episodes, heldout=heldout)


def test_the_engine_trains_reading_windows_and_checkpoints_only_the_trained_bank(tmp_path):
    from megatoken.training.checkpoint import load_checkpoint, resolve_checkpoint
    from megatoken.training.engine import train

    torch.set_num_threads(1)
    dataset = reading_dataset(tmp_path / "a", 3)
    config = train_config(tmp_path / "b", datasets=[str(dataset)], max_steps=2)
    outcome = train(config)
    assert outcome["global_step"] == 2
    _, metadata = load_checkpoint(outcome["checkpoint"])
    saved = torch.load(
        resolve_checkpoint(outcome["checkpoint"]) / metadata["transducer_checkpoint"]["file"],
        weights_only=True,
    )
    assert set(saved) >= {"bank", "optimizer", "growth"}
    assert saved["bank"]["rows"]["ids"].numel() == 0  # the reading windows wrote no trained edge


def test_held_out_loss_is_reported_with_the_base_mounted_and_unmounted(tmp_path):
    torch.set_num_threads(1)
    dataset = reading_dataset(tmp_path / "a", 2, heldout=True)
    config = train_config(tmp_path / "b", datasets=[str(dataset)], eval_windows=4)
    evaluator = HeldOutEvaluator(
        config, CODEC, WindowConfig(max_records=8, max_decisions=4), require=True
    )
    model = tiny_model()
    model.core.set_model_version(0)
    sums = evaluator.sums(model, TransducerTrainer(model.transducer), 0, rank=0, world=1)
    report = evaluator.report([sums])
    assert sums[MOUNTED]["windows"] == sums[UNMOUNTED]["windows"] == sums["all"]["windows"] > 0
    gain = report["reading_base"]
    assert gain["mounted_loss"] != gain["unmounted_loss"]
    assert gain["gain"] == pytest.approx(gain["unmounted_loss"] - gain["mounted_loss"])
    # The twin runs re-score windows the stream already counts: they are not pooled twice.
    assert report["streams"]["all"]["windows"] == sums["all"]["windows"]
    assert report["pooled_weighted_loss"] == pytest.approx(
        report["streams"]["all"]["weighted_loss"]
    )
    evaluator.close()


def test_the_producer_builds_documents_and_media_episodes_with_declared_items(tmp_path):
    path = Path(__file__).resolve().parents[1] / "scripts/prepare_base_reading.py"
    spec = importlib.util.spec_from_file_location("prepare_base_reading", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from megatoken.data.compression import CompressionPolicy, DocumentCompressor

    store = TensorAssetStore(tmp_path)
    compressor = DocumentCompressor(
        CountingTeacher(limit=300), store, CompressionPolicy(min_bytes=1)
    )
    task = {
        "id": "q1",
        "question": "Which river?",
        "documents": ["Paris\n" + "The Seine flows through Paris. " * 40, "Rome\nThe Tiber."],
        "answer": "The Seine",
    }
    episode = module.document_episode(task, compressor)
    assert episode.stream == STREAM and len(episode.records) == 2
    parts = declared_items(episode)
    assert len(parts) > 1 and [p["part"] for p in parts] == list(range(len(parts)))
    assert {p["document"] for p in parts} == {"document"}

    class Media:
        def media(self, modality, path, *, target=True):
            return prepared_record(store, modality, seed=3, target=False)

    photo = module.image_episode(
        Media(), {"id": "d1", "image": tmp_path / "d1.jpg", "description": "A red door. It is old."}
    )
    speech = module.speech_episode(
        Media(), {"id": "s1", "audio": tmp_path / "s1.flac", "text": "one two three four five six"}
    )
    assert photo.records[0].payload.endswith(b"A red door.")  # the cue names the photo
    assert speech.records[0].payload.endswith(b"one two three four five")
    assert photo.records[1].payload == b"A red door. It is old."
    assert [r["document"] for r in declared_items(photo)] == [""]
