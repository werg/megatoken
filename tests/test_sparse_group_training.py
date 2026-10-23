"""Sparse physical records still train the complete selected native group."""

import json
from dataclasses import replace

import torch
from training_fixture import MODEL_OVERRIDES, TEXT_ONLY_ASSETS, tiny_train_config

from megatoken.data import Episode, Lineage, PhysicalRecord, WindowConfig, iter_windows
from megatoken.data.schema import Meridian, Supervision, TrackEvent, TrackSpec, canonical_json
from megatoken.generation.codec import ByteCodec
from megatoken.generation.structure import ActionSpec
from megatoken.loading import build_model
from megatoken.model import ModelConfig
from megatoken.model.multitrack import MultiTrackGenerator, ObservedEvent
from megatoken.training.engine import prepare_batches
from megatoken.training.history import NativeHistory
from megatoken.training.losses import LossLedger
from megatoken.training.meridians import forward_meridian_batch
from megatoken.training.stream import EpisodeWindow
from megatoken.training.transducer import TransducerTrainer


def test_sparse_native_peer_stays_in_prediction_and_recurrence(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    torch.manual_seed(19)
    codec = ByteCodec.byte_fallback()
    model = build_model(ModelConfig.tiny(**MODEL_OVERRIDES), codec)
    multi = MultiTrackGenerator.of(model)
    state = multi.tracks.initial_state(1)
    selected = multi.generate_group(
        state,
        tracks=("missing", "present"),
        event_ids=("failed-event", "written-event"),
        effort=1,
        specs=(ActionSpec(max_tokens=4), ActionSpec(max_tokens=4)),
    )
    tape = selected.tape
    # The first choice failed rendering and therefore has no physical record.
    sparse = replace(
        tape,
        payloads=(b"", tape.payloads[1]),
        emitted_ids=((), tape.emitted_ids[1]),
        valid=(False, tape.valid[1]),
    )
    record = PhysicalRecord(
        "written-record",
        "assistant",
        sparse.payloads[1],
        policy_mask=True,
        emitted_ids=sparse.emitted_ids[1],
        codec_fingerprint=codec.fingerprint,
    )
    episode = Episode(
        "sparse-group",
        (record,),
        Lineage("fixture", "v1", "sparse-group"),
        tracks=(TrackSpec("missing", "assistant"), TrackSpec("present", "assistant")),
        meridians=(
            Meridian(0, (TrackEvent("written-record", "present", "text", 0, "assistant"),)),
        ),
        supervision=Supervision(
            annotations_json=canonical_json(
                {
                    "native_action_tapes": [
                        {
                            "record_cursor": 0,
                            "record_ids": [None, "written-record"],
                            "tape": json.loads(sparse.to_json()),
                        }
                    ]
                }
            )
        ),
    )
    native = NativeHistory.from_episode(episode, model)
    assert native.group_for_records((0,)) == sparse

    config = tiny_train_config(tmp_path, phase="P1", representation_mode="recurrent")
    window = next(
        iter_windows(
            episode,
            codec,
            WindowConfig(max_records=4, max_decisions=4, max_window_tokens=4096),
        )
    )
    (batch,) = prepare_batches(model, [EpisodeWindow(episode, window, TEXT_ONLY_ASSETS)], config)
    assert batch.counts["modality"] == 2
    assert batch.counts["lexical"] == len(record.payload)

    observed = []
    original = MultiTrackGenerator.supervised_meridian

    def capture(self, current, **kwargs):
        observed.append(kwargs.copy())
        return original(self, current, **kwargs)

    monkeypatch.setattr(MultiTrackGenerator, "supervised_meridian", capture)
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

    call = observed[0]
    assert call["full_group"] == sparse
    assert call["tracks"] == sparse.tracks
    assert call["event_ids"] == sparse.event_ids
    assert call["plan_indices"] == (1,)
    assert len(call["plans"]) == 1
    assert metrics["grouped_selected_events"] == 2
    assert metrics["grouped_supervised_meridians"] == 1
    assert model.core.predictor_parameters()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.core.predictor_parameters()
    )


def test_trailing_failed_selection_is_sampled_and_trained_without_dummy_text(tmp_path):
    torch.set_num_threads(1)
    model = build_model(ModelConfig.tiny(), ByteCodec.byte_fallback())
    multi = MultiTrackGenerator.of(model)
    multi.tracks.record_policy_context = True
    state = multi.tracks.observe_group(
        multi.tracks.initial_state(1),
        (ObservedEvent("user", "text", b"Do useful work", event_id="source"),),
    )
    selected = multi.select_group(state, tracks=("reply",), effort=1)
    episode = Episode(
        "failed-tail",
        (PhysicalRecord("source", "user", b"Do useful work"),),
        Lineage("fixture", "v1", "failed-tail"),
        tracks=(TrackSpec("user", "user"),),
        meridians=(Meridian(0, (TrackEvent("source", "user", "text", 0, "user"),)),),
        supervision=Supervision(
            annotations_json=canonical_json(
                {
                    "native_action_tapes": [
                        {
                            "record_cursor": 1,
                            "record_ids": [None],
                            "tape": json.loads(selected.tape.to_json()),
                            "outcome_credit": {
                                "reward": -1,
                                "baseline": 0,
                                "selected_events": 1,
                                "verification": "external_tests",
                            },
                        }
                    ],
                }
            )
        ),
    )
    windows = list(iter_windows(episode, model.codec))
    tail = windows[-1]
    assert tail.start == tail.end == 1
    assert tail.records == () and tail.Q == 1 and tail.P == 0
    config = tiny_train_config(tmp_path, phase="P1", representation_mode="recurrent")
    (batch,) = prepare_batches(model, [EpisodeWindow(episode, tail, TEXT_ONLY_ASSETS)], config)
    assert batch.counts["policy_outcome"] == 1
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
    assert any(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in model.core.predictor_parameters()
    )
