"""Atomic-meridian training windows and global-denominator integration."""

import pytest
import torch

from megatoken.data import Episode, Lineage, PhysicalRecord, WindowConfig, iter_windows
from megatoken.data.schema import Meridian, TrackEvent, TrackSpec
from megatoken.generation.codec import ByteCodec
from megatoken.loading import build_model
from megatoken.model import ModelConfig
from megatoken.model.multitrack import MultiTrackGenerator
from megatoken.training.config import TrainConfig
from megatoken.training.engine import batch_counters, forward_batch, prepare_batches
from megatoken.training.losses import LossLedger
from megatoken.training.meridians import MeridianPreparedBatch
from megatoken.training.stream import EpisodeWindow


def fixture_episode():
    names = ("a", "b", "worker_a", "worker_b", "answer")
    values = (b"red", b"blue", b"one", b"two", b"three")
    records = tuple(
        PhysicalRecord(name, "user" if i < 2 else "assistant", values[i],
                       turn_index=max(0, i - 1), policy_mask=i >= 2)
        for i, name in enumerate(names)
    )
    tracks = tuple(TrackSpec(name, record.role) for name, record in zip(names, records))
    meridians = tuple(
        Meridian(g, tuple(TrackEvent(names[i], names[i], "text", g, records[i].role)
                          for i in indices))
        for g, indices in enumerate(((0, 1), (2, 3), (4,)))
    )
    return Episode("joint", records, Lineage("fixture", "v1", "family"),
                   tracks=tracks, meridians=meridians)


def train_config(mode):
    return TrainConfig(
        phase="B0" if mode == "bootstrap" else "P1", representation_mode=mode,
        model="tiny", device="cpu", precision="fp32", generated_capacity=16,
        max_record_bytes=128, effort=1, use_stage_mixture=False,
        use_effort_mixture=False, gradient_checkpointing=False,
    )


def grouped_batch(model, mode):
    episode = fixture_episode()
    windows = list(iter_windows(
        episode, ByteCodec.byte_fallback(),
        WindowConfig(max_records=8, max_decisions=4, max_window_tokens=4096),
    ))
    assert len(windows) == 1
    batch, = prepare_batches(
        model, [EpisodeWindow(episode, windows[0])], train_config(mode),
    )
    assert isinstance(batch, MeridianPreparedBatch)
    return batch


def test_meridians_are_indivisible_window_groups():
    episode = fixture_episode()
    windows = list(iter_windows(
        episode, ByteCodec.byte_fallback(),
        WindowConfig(max_records=1, max_decisions=1, max_window_tokens=8),
    ))
    assert [(w.start, w.end) for w in windows] == [(0, 2), (2, 4), (4, 5)]


@pytest.mark.parametrize("mode", ("bootstrap", "recurrent"))
def test_grouped_global_ledger_and_selected_recurrence(mode, monkeypatch):
    torch.set_num_threads(1)
    torch.manual_seed(27)
    model = build_model(ModelConfig.tiny(), ByteCodec.byte_fallback())
    batch = grouped_batch(model, mode)
    assert len(batch.steps) == 3 and not batch.steps[0].generated
    assert len(batch.steps[1].plans) == 2 and len(batch.steps[2].plans) == 1
    assert batch.counts["lexical"] == len(b"onetwothree")
    assert batch.counts["reconstruction"] == batch.counts["lexical"]
    assert batch.counts["facet_prediction"] == (
        3 * model.config.n_facets * model.config.facet_dim
        if mode == "bootstrap" else 0
    )
    seen = []
    original = MultiTrackGenerator.supervised_meridian

    def watch(self, state, **kwargs):
        result = original(self, state, **kwargs)
        result.state.raw.retain_grad()
        seen.append((state, result.state))
        return result

    monkeypatch.setattr(MultiTrackGenerator, "supervised_meridian", watch)
    config = train_config(mode)
    ledger = LossLedger(batch.counts, batch.counts, config.loss_weights)
    metrics = forward_batch(model, batch, ledger, config, torch.device("cpu"), 0)
    ledger.finish()
    assert metrics["grouped_supervised_meridians"] == 2
    assert metrics["grouped_selected_events"] == 3
    assert seen[1][0] is seen[0][1]
    assert seen[0][1].history.slots == 4 * model.config.n_facets
    assert seen[0][1].raw.grad is not None
    assert model.core.frontend.facet_queries.grad is not None
    assert batch_counters([batch]).P == 2 * len(b"onetwothree")


def test_media_payload_is_not_silently_trained_as_text():
    from dataclasses import replace

    ep = fixture_episode()
    changed = replace(ep, meridians=(
        Meridian(0, (
            TrackEvent("a", "a", "audio", 0, "user"),
            TrackEvent("b", "b", "text", 0, "user"),
        )), *ep.meridians[1:],
    ))
    model = build_model(ModelConfig.tiny(), ByteCodec.byte_fallback())
    window = next(iter_windows(changed, ByteCodec.byte_fallback()))
    with pytest.raises(ValueError, match="pinned media asset"):
        prepare_batches(
            model, [EpisodeWindow(changed, window)], train_config("bootstrap"),
        )

def test_grouped_episode_runs_in_existing_training_checkpoint_loop(tmp_path):
    from dataclasses import replace

    from megatoken.data import DedupConfig, DedupStore
    from megatoken.data.storage import export_prepared
    from megatoken.training.checkpoint import load_checkpoint
    from megatoken.training.engine import train

    torch.set_num_threads(1)
    codec = ByteCodec.byte_fallback()
    store = DedupStore(
        tmp_path / "spool.sqlite", DedupConfig(train_fraction=1, validation_fraction=0),
    )
    try:
        with store.transaction():
            store.add(fixture_episode(), codec)
        store.freeze()
        dataset = tmp_path / "data"
        export_prepared(
            store, dataset, codec=codec, source_receipts=[],
            pipeline_fingerprint="meridian-integration-v1",
        )
    finally:
        store.close()
    config = replace(
        train_config("recurrent"),
        dataset=str(dataset), output=str(tmp_path / "run"),
        max_steps=1, max_accumulation_windows=1, microbatch_episodes=1,
        accumulate_scored=1, checkpoint_every=0, eval_every=0,
    )
    outcome = train(config)
    payload, _ = load_checkpoint(outcome["checkpoint"])
    assert outcome["global_step"] == 1
    assert outcome["counters"]["P"] > 0
    assert payload["global_step"] == 1
