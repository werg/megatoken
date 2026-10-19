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

