"""Counterfactual targets cannot change a causal student's earlier prediction."""

import copy
import json
from dataclasses import replace

import torch
from training_fixture import TEXT_ONLY_ASSETS, tiny_train_config

from megatoken.contracts import DEFAULT_LOSS_WEIGHTS
from megatoken.data import (
    Episode,
    Lineage,
    LogicalAction,
    PhysicalRecord,
    WindowConfig,
    iter_windows,
)
from megatoken.data.schema import canonical_json, digest_bytes, sequential_meridians
from megatoken.generation import ByteCodec
from megatoken.loading import build_model
from megatoken.model import ModelConfig
from megatoken.model.multitrack import MultiTrackGenerator
from megatoken.training.engine import prepare_batches
from megatoken.training.losses import LossLedger
from megatoken.training.meridians import forward_meridian_batch
from megatoken.training.stream import EpisodeWindow
from megatoken.training.transducer import HindsightSchedule, Meridian, TransducerTrainer


def task(choice):
    tools = [
        {
            "name": name,
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        }
        for name in ("serial_reads", "delegated_reads")
    ]
    payload = canonical_json({"name": choice, "arguments": {}}).encode()
    return sequential_meridians(
        Episode(
            "counterfactual",
            (
                PhysicalRecord("request", "user", b"Choose a procedure from the available costs."),
                PhysicalRecord(
                    "costs",
                    "environment",
                    b'{"read_delay_seconds":0.04,"delegation_overhead_seconds":0.001}',
                ),
                PhysicalRecord(
                    "choice", "assistant", payload, policy_mask=True, action_id="choose"
                ),
                PhysicalRecord(
                    "result", "tool", b'{"done":true}', result_of="choose", execution_id="executed"
                ),
            ),
            Lineage("fixture", "v1", "counterfactual"),
            tools_json=json.dumps(tools),
            actions=(
                LogicalAction(
                    "choose", "choose", choice, "{}", ("choice",), digest_bytes(payload), "executed"
                ),
            ),
        )
    )


def test_future_tool_label_cannot_change_the_prior_prediction(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    torch.manual_seed(31)
    codec = ByteCodec.byte_fallback()
    model = build_model(ModelConfig.tiny(), codec)
    model.train()
    config = replace(
        tiny_train_config(tmp_path, phase="P1"), purpose="diagnostic", generated_capacity=128
    )
    multi = MultiTrackGenerator.of(model)
    original = multi.tracks.predict_group
    captured = []

    def predict(*args, **kwargs):
        value = original(*args, **kwargs)
        if kwargs.get("event_ids") == ("choice",):
            captured.append(value.detach().clone())
        return value

    monkeypatch.setattr(multi.tracks, "predict_group", predict)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Student tool training must not read future teaching inputs")

    monkeypatch.setattr(model.transducer, "hindsight_pass", forbidden)
    predictions = []
    for label in ("serial_reads", "delegated_reads"):
        episode = task(label)
        (window,) = list(
            iter_windows(
                episode, codec, WindowConfig(max_records=8, max_decisions=4, max_window_tokens=4096)
            )
        )
        (batch,) = prepare_batches(
            model, [EpisodeWindow(episode, window, TEXT_ONLY_ASSETS)], config
        )
        from types import SimpleNamespace

        trainer = TransducerTrainer(
            model.transducer, teacher_targets=SimpleNamespace(load=forbidden)
        )
        captured.clear()
        torch.manual_seed(19)
        metrics = forward_meridian_batch(
            model,
            batch,
            LossLedger(batch.counts, batch.counts, DEFAULT_LOSS_WEIGHTS),
            config,
            torch.device("cpu"),
            model.core.model_version,
            transducer=trainer,
            differentiate=False,
        )
        assert metrics["causal_student_windows"] == 1 and trainer.records[0].hindsight is None
        assert captured
        predictions.append(captured[-1])
    assert torch.equal(*predictions)


def test_teacher_assistance_is_opt_in_and_student_windows_start_immediately():
    model = build_model(ModelConfig.tiny(), ByteCodec.byte_fallback())
    trainer = TransducerTrainer(model.transducer)
    events = [Meridian(("assistant",), (1,), ("text",), False)]
    student = trainer.sequence(events, step=0)
    assert student.alpha == 0 and student.stored is None
    from types import SimpleNamespace

    trainer.teacher_targets = SimpleNamespace()
    assisted = [trainer.sequence(events, step=0, teacher_supervision=True) for _ in range(8)]
    assert sum(window.alpha == 0 for window in assisted) == 7
    assert sum(window.alpha == 1 for window in assisted) == 1
    assert all(HindsightSchedule().overlay(step, 3) == 0 for step in (0, 100, 100000))


def test_frozen_memory_cannot_accumulate_or_apply_sparse_updates():
    model = build_model(ModelConfig.tiny(), ByteCodec.byte_fallback())
    trainer = TransducerTrainer(model.transducer, bank_trainable=False)
    g = trainer.bank.geometry
    trainer.bank.add_edges(
        departure=torch.randn(1, g.location_dim),
        key=torch.randn(1, g.query_dim),
        value=torch.randn(1, g.value_dim),
        destination=torch.randn(1, g.location_dim),
        bias=torch.zeros(1),
        write_mass=torch.ones(1),
        step=0,
    )
    before = copy.deepcopy(trainer.bank.state_snapshot())
    sequence = trainer.sequence(
        [Meridian(("assistant",), (1,), ("text",), False)], step=0, teacher_supervision=True
    )
    assert sequence.alpha == 0 and sequence.stored is None
    assert trainer.after_backward(1)["transducer_writes"] == 0
    assert trainer.barrier(1)["transducer_updated_fields"] == 0
    after = trainer.bank.state_snapshot()

    def equal(left, right):
        if isinstance(left, dict):
            assert left.keys() == right.keys()
            for key in left:
                equal(left[key], right[key])
        elif isinstance(left, torch.Tensor):
            assert torch.equal(left, right)
        else:
            assert left == right

    equal(before, after)


def test_causal_training_prediction_matches_actual_serving(monkeypatch, tmp_path):
    import megatoken.training.meridians as training
    from megatoken.contracts import MAIN_TRACK
    from megatoken.controls import resolve_event_control
    from megatoken.model.transducer import metadata_payload
    from megatoken.multitrack_runtime import MultiTrackRuntime
    from megatoken.tool_schema import compile_tool_spec, recent_copy_sources

    torch.set_num_threads(1)
    torch.manual_seed(51)
    codec = ByteCodec.byte_fallback()
    model = build_model(ModelConfig.tiny(max_capacity=512), codec).eval()
    multi = MultiTrackGenerator.of(model)
    sample = task("serial_reads")
    config = replace(tiny_train_config(tmp_path, phase="P1"), generated_capacity=128)
    (window,) = list(
        iter_windows(
            sample, codec, WindowConfig(max_records=8, max_decisions=4, max_window_tokens=4096)
        )
    )
    (batch,) = prepare_batches(model, [EpisodeWindow(sample, window, TEXT_ONLY_ASSETS)], config)
    monkeypatch.setattr(training, "_metadata", lambda *_: metadata_payload({}))
    original = multi.tracks.predict_group
    predictions = []

    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        if kwargs.get("event_ids") == ("choice",):
            predictions.append(result.detach().clone())
        return result

    monkeypatch.setattr(multi.tracks, "predict_group", capture)
    with torch.no_grad():
        forward_meridian_batch(
            model,
            batch,
            LossLedger(batch.counts, batch.counts, DEFAULT_LOSS_WEIGHTS),
            config,
            torch.device("cpu"),
            model.core.model_version,
            transducer=TransducerTrainer(model.transducer),
            differentiate=False,
        )
    assert len(predictions) == 1
    trained = predictions.pop()
    runtime = MultiTrackRuntime(multi)
    try:
        session = runtime.open(metadata={}, effort=1, memory_writes=False)
        for record in sample.records[:2]:
            runtime.observe_text(session, MAIN_TRACK, record.payload, role=record.role, effort=1)
        spec = replace(
            compile_tool_spec(sample.tools_json, 128),
            allow_text=True,
            copy_sources=recent_copy_sources([r.payload for r in sample.records[:2]]),
        )
        runtime.generate(
            session,
            tracks=(MAIN_TRACK,),
            event_ids=("choice",),
            specs=(spec,),
            controls=(resolve_event_control(sample, sample.meridians[2].events[0]),),
            effort=1,
            generator=torch.Generator().manual_seed(17),
        )
        assert len(predictions) == 1
        torch.testing.assert_close(predictions[0], trained, rtol=0, atol=0)
    finally:
        runtime.close()
