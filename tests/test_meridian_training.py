"""Atomic-meridian training windows and global-denominator integration."""

import pytest
import torch
from training_fixture import MODEL_OVERRIDES, TEXT_ONLY_ASSETS, tiny_train_config

from megatoken.data import Episode, Lineage, PhysicalRecord, WindowConfig, iter_windows
from megatoken.data.schema import Meridian, TrackEvent, TrackSpec
from megatoken.generation.codec import ByteCodec
from megatoken.loading import build_model
from megatoken.model import ModelConfig
from megatoken.model.multitrack import MultiTrackGenerator
from megatoken.training.engine import batch_counters, prepare_batches
from megatoken.training.losses import LossLedger
from megatoken.training.meridians import MeridianPreparedBatch, forward_meridian_batch
from megatoken.training.stream import EpisodeWindow
from megatoken.training.transducer import TransducerTrainer


def fixture_episode():
    names = ("a", "b", "worker_a", "worker_b", "answer")
    values = (b"red", b"blue", b"one", b"two", b"three")
    records = tuple(
        PhysicalRecord(
            name,
            "user" if i < 2 else "assistant",
            values[i],
            turn_index=max(0, i - 1),
            policy_mask=i >= 2,
        )
        for i, name in enumerate(names)
    )
    tracks = tuple(
        TrackSpec(name, record.role) for name, record in zip(names, records, strict=True)
    )
    meridians = tuple(
        Meridian(
            g, tuple(TrackEvent(names[i], names[i], "text", g, records[i].role) for i in indices)
        )
        for g, indices in enumerate(((0, 1), (2, 3), (4,)))
    )
    return Episode(
        "joint", records, Lineage("fixture", "v1", "family"), tracks=tracks, meridians=meridians
    )


def train_config(root, mode):
    return tiny_train_config(
        root,
        phase="B0" if mode == "bootstrap" else "P1",
        representation_mode=mode,
    )


def tiny_model():
    return build_model(ModelConfig.tiny(**MODEL_OVERRIDES), ByteCodec.byte_fallback())


def grouped_batch(model, config):
    episode = fixture_episode()
    windows = list(
        iter_windows(
            episode,
            ByteCodec.byte_fallback(),
            WindowConfig(max_records=8, max_decisions=4, max_window_tokens=4096),
        )
    )
    assert len(windows) == 1
    (batch,) = prepare_batches(
        model,
        [EpisodeWindow(episode, windows[0], TEXT_ONLY_ASSETS)],
        config,
    )
    assert isinstance(batch, MeridianPreparedBatch)
    return batch


def test_meridians_are_indivisible_window_groups():
    episode = fixture_episode()
    windows = list(
        iter_windows(
            episode,
            ByteCodec.byte_fallback(),
            WindowConfig(max_records=1, max_decisions=1, max_window_tokens=8),
        )
    )
    assert [(w.start, w.end) for w in windows] == [(0, 2), (2, 4), (4, 5)]


@pytest.mark.parametrize("mode", ("bootstrap", "recurrent"))
def test_grouped_global_ledger_and_selected_recurrence(mode, monkeypatch, tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(27)
    model = tiny_model()
    config = train_config(tmp_path, mode)
    batch = grouped_batch(model, config)
    assert len(batch.steps) == 3 and not batch.steps[0].generated
    assert len(batch.steps[1].plans) == 2 and len(batch.steps[2].plans) == 1
    assert batch.counts["lexical"] == len(b"onetwothree")
    assert batch.counts["reconstruction"] == batch.counts["lexical"]
    assert batch.counts["facet_prediction"] == (
        3 * model.config.n_facets * model.config.facet_dim if mode == "bootstrap" else 0
    )
    seen = []
    original = MultiTrackGenerator.supervised_meridian

    def watch(self, state, **kwargs):
        result = original(self, state, **kwargs)
        result.state.raw.retain_grad()
        seen.append((state, result.state))
        return result

    monkeypatch.setattr(MultiTrackGenerator, "supervised_meridian", watch)
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
    ledger.finish()
    assert metrics["grouped_supervised_meridians"] == 2
    assert metrics["grouped_selected_events"] == 3
    # The bank lookup between meridians extends the state the next one starts from.
    assert seen[1][0].history.slots >= seen[0][1].history.slots
    # Four facet slots per meridian step, plus the bank's metadata and lookup slots.
    assert seen[0][1].history.slots > 4 * model.config.n_facets
    assert seen[0][1].raw.grad is not None
    assert model.core.frontend.facet_queries.grad is not None
    assert batch_counters([batch]).P == 2 * len(b"onetwothree")


def test_media_payload_is_not_silently_trained_as_text(tmp_path):
    from dataclasses import replace

    ep = fixture_episode()
    changed = replace(
        ep,
        meridians=(
            Meridian(
                0,
                (
                    TrackEvent("a", "a", "audio", 0, "user"),
                    TrackEvent("b", "b", "text", 0, "user"),
                ),
            ),
            *ep.meridians[1:],
        ),
    )
    model = tiny_model()
    window = next(iter_windows(changed, ByteCodec.byte_fallback()))
    with pytest.raises(ValueError, match="pinned canonical media descriptor"):
        prepare_batches(
            model,
            [EpisodeWindow(changed, window, TEXT_ONLY_ASSETS)],
            train_config(tmp_path, "bootstrap"),
        )


def test_grouped_episode_runs_in_existing_training_checkpoint_loop(tmp_path):
    from training_fixture import tiny_train_config

    from megatoken.training.checkpoint import load_checkpoint
    from megatoken.training.engine import train

    torch.set_num_threads(1)
    config = tiny_train_config(
        tmp_path,
        phase="P1",
        representation_mode="recurrent",
        microbatch_episodes=1,
        max_accumulation_windows=1,
    )
    outcome = train(config)
    payload, _ = load_checkpoint(outcome["checkpoint"])
    assert outcome["global_step"] == 1
    assert outcome["counters"]["P"] > 0
    assert payload["global_step"] == 1


def test_executed_tool_result_uses_teacher_consistent_group_continuation(monkeypatch, tmp_path):
    from dataclasses import replace

    from megatoken.data.schema import LogicalAction, digest_bytes
    from megatoken.tensors import byte_tensor

    original = fixture_episode()
    records = (
        *original.records[:2],
        replace(original.records[2], action_id="call-1"),
        original.records[3],
        replace(
            original.records[4],
            role="tool",
            policy_mask=False,
            result_of="call-1",
            execution_id="execution-1",
        ),
    )
    tracks = (*original.tracks[:4], TrackSpec("answer", "tool"))
    meridians = (
        *original.meridians[:2],
        Meridian(2, (TrackEvent("answer", "answer", "text", 2, "tool"),)),
    )
    action = LogicalAction(
        "call-1",
        "id-1",
        "noop",
        "{}",
        ("worker_a",),
        digest_bytes(b"one"),
        execution_id="execution-1",
    )
    episode = replace(
        original,
        records=records,
        tracks=tracks,
        meridians=meridians,
        actions=(action,),
    )
    model = tiny_model()
    config = train_config(tmp_path, "recurrent")
    window = next(
        iter_windows(
            episode,
            ByteCodec.byte_fallback(),
            WindowConfig(max_records=8, max_decisions=4),
        )
    )
    (batch,) = prepare_batches(
        model,
        [EpisodeWindow(episode, window, TEXT_ONLY_ASSETS)],
        config,
    )
    selected = []
    original_step = MultiTrackGenerator.supervised_meridian

    def capture(self, state, **kwargs):
        selected.append((kwargs["teacher_forced_action"], original_step(self, state, **kwargs)))
        return selected[-1][1]

    monkeypatch.setattr(MultiTrackGenerator, "supervised_meridian", capture)
    ledger = LossLedger(batch.counts, batch.counts, config.loss_weights)
    forward_meridian_batch(
        model,
        batch,
        ledger,
        config,
        torch.device("cpu"),
        0,
        transducer=TransducerTrainer(model.transducer),
    )
    ledger.finish()
    assert len(selected) == 1 and selected[0][0]
    encoded = []
    for data in (b"one", b"two"):
        ids, lengths = byte_tensor([data], torch.device("cpu"))
        encoded.append(model.core.encode_chunk(ids, lengths).detach())
    actual = torch.stack(encoded, dim=1)
    torch.testing.assert_close(selected[0][1].state.raw, actual, atol=1e-6, rtol=1e-6)



def test_semantic_execution_continues_from_teacher_recorded_semantics(monkeypatch, tmp_path):
    """Every dependent semantic group commits the recorded target after scoring P."""
    from dataclasses import replace

    from megatoken.data.schema import Supervision

    torch.set_num_threads(1)
    torch.manual_seed(47)
    example = replace(
        fixture_episode(),
        stream="semantic_execution",
        supervision=Supervision(
            annotations_json='{"semantic_execution":{"continuation":"teacher_conditioned"}}'
        ),
    )
    model = tiny_model()
    config = replace(train_config(tmp_path, "recurrent"), allow_unverified_data=True)
    window = next(
        iter_windows(
            example,
            ByteCodec.byte_fallback(),
            WindowConfig(max_records=8, max_decisions=4, max_window_tokens=4096),
        )
    )
    (batch,) = prepare_batches(
        model,
        [EpisodeWindow(example, window, TEXT_ONLY_ASSETS)],
        config,
    )
    choices = []
    original = MultiTrackGenerator.supervised_meridian

    def capture(self, state, **kwargs):
        choices.append(kwargs["teacher_forced_action"])
        return original(self, state, **kwargs)

    monkeypatch.setattr(MultiTrackGenerator, "supervised_meridian", capture)
    ledger = LossLedger(batch.counts, batch.counts, config.loss_weights)
    forward_meridian_batch(
        model,
        batch,
        ledger,
        config,
        torch.device("cpu"),
        0,
        transducer=TransducerTrainer(model.transducer),
    )
    ledger.finish()
    assert choices == [True, True]
