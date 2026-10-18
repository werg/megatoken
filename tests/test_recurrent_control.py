"""Training-only noisy trajectory references and causal-control isolation."""

from dataclasses import replace

import pytest
import torch

from megatoken.controls import encode_controls
from megatoken.data import WindowConfig, iter_windows
from megatoken.data.schema import Episode, Lineage, PhysicalRecord, Supervision
from megatoken.generation import ByteCodec, FacetedGenerator
from megatoken.model import FacetedCore, ModelConfig
from megatoken.training import engine
from megatoken.training.config import TrainConfig
from megatoken.training.engine import forward_batch, make_training_control, prepare_batches
from megatoken.training.losses import LossLedger
from megatoken.training.stream import EpisodeWindow


def _window(*, verified: bool, suffix: bytes = b"success", causal: bytes = b"public policy"):
    episode = Episode(
        "controlled-episode",
        (
            PhysicalRecord("request", "user", b"edit the file"),
            PhysicalRecord("action", "assistant", b"write", turn_index=1, policy_mask=True),
            PhysicalRecord("answer", "assistant", suffix, turn_index=2, policy_mask=True),
        ),
        Lineage("reference-fixture", "r1", "family"),
        control_seed=causal,
        supervision=(
            Supervision(outcome="success", verification="local_assertions")
            if verified
            else Supervision()
        ),
    )
    codec = ByteCodec.byte_fallback()
    window, = iter_windows(episode, codec, WindowConfig(max_records=5, max_decisions=5))
    return EpisodeWindow(episode, window)


@pytest.fixture
def model():
    torch.set_num_threads(1)
    torch.manual_seed(561)
    return FacetedGenerator(FacetedCore(ModelConfig.tiny()), ByteCodec.byte_fallback())


def test_verified_reference_is_detached_and_noisy_but_public_control_is_trainable(
    model, monkeypatch
):
    teacher_window = _window(verified=True, suffix=b"recorded good result")
    causal_window = _window(verified=False, suffix=b"ordinary continuation")
    monkeypatch.setattr(engine, "_RECURRENT_REFERENCE_FRACTION", 1.0)
    monkeypatch.setattr(engine, "_RECURRENT_REFERENCE_NOISE_STD", 0.0)

    expected_causal = engine.make_control(
        model, [teacher_window, causal_window], torch.device("cpu"), encoder_grad=False
    )
    control, count = make_training_control(
        model, [teacher_window, causal_window], torch.device("cpu"), recurrent=True
    )
    assert count == 1 and control is not None
    torch.testing.assert_close(control[1], expected_causal[1], rtol=0, atol=0)
    expected_teacher = encode_controls(
        model.core,
        [
            engine.episode_control_payload(teacher_window.episode)
            + b"\n<reference>\n"
            + engine._successful_trajectory_reference(teacher_window.episode)
        ],
        encoder_grad=False,
    )
    torch.testing.assert_close(control[0], expected_teacher[0], rtol=0, atol=0)
    assert not control[0].requires_grad

    model.zero_grad(set_to_none=True)
    model.core.control_injection(control).square().sum().backward()
    assert model.core.control_injection.weight.grad.norm() > 0
    # The causal branch still trains the public control head, whereas the
    # trajectory-derived reference has been fully detached before injection.
    assert any(p.grad is not None and p.grad.norm() > 0
               for p in model.core.control_encoder.parameters())
    assert all(p.grad is None for p in model.core.encoder_parameters())


def test_training_reference_noise_only_applies_to_verified_success(model, monkeypatch):
    teacher = _window(verified=True, suffix=b"first verified result")
    alternative = _window(verified=True, suffix=b"a different verified result")
    unknown = _window(verified=False, causal=b"")
    monkeypatch.setattr(engine, "_RECURRENT_REFERENCE_FRACTION", 1.0)
    monkeypatch.setattr(engine, "_RECURRENT_REFERENCE_NOISE_STD", 0.0)
    a, used_a = make_training_control(model, [teacher], torch.device("cpu"), recurrent=True)
    b, used_b = make_training_control(model, [alternative], torch.device("cpu"), recurrent=True)
    assert used_a == used_b == 1 and not torch.allclose(a, b)
    none, count = make_training_control(model, [unknown], torch.device("cpu"), recurrent=True)
    assert none is None and count == 0
    none, count = make_training_control(model, [unknown], torch.device("cpu"), recurrent=False)
    assert none is None and count == 0

    monkeypatch.setattr(engine, "_RECURRENT_REFERENCE_NOISE_STD", 0.2)
    torch.manual_seed(13)
    noisy, count = make_training_control(model, [teacher], torch.device("cpu"), recurrent=True)
    assert count == 1
    assert not torch.equal(noisy, a)
    assert not noisy.requires_grad


def test_reference_is_one_window_level_control_for_recurrent_supervision(model, monkeypatch):
    teacher = _window(verified=True)
    monkeypatch.setattr(engine, "_RECURRENT_REFERENCE_FRACTION", 1.0)
    monkeypatch.setattr(engine, "_RECURRENT_REFERENCE_NOISE_STD", 0.0)
    config = TrainConfig(
        phase="P1",
        device="cpu",
        precision="fp32",
        representation_mode="recurrent",
        generated_capacity=16,
        effort=1,
        microbatch_episodes=1,
        use_stage_mixture=False,
        use_effort_mixture=False,
    )
    (batch,) = prepare_batches(model, [teacher], config)
    observed_controls = []
    original = model.supervised_step

    def capture(*args, **kwargs):
        observed_controls.append(kwargs["control"])
        return original(*args, **kwargs)

    monkeypatch.setattr(model, "supervised_step", capture)
    ledger = LossLedger(batch.counts, batch.counts, config.loss_weights)
    metrics = forward_batch(model, batch, ledger, config, torch.device("cpu"), 0)
    ledger.finish()
    assert metrics["noisy_trajectory_control_examples"] == 1
    assert len(observed_controls) == 2
    assert observed_controls[0] is observed_controls[1]
    assert observed_controls[0] is not None
    assert not observed_controls[0].requires_grad


def test_recurrent_noisy_reference_never_changes_causal_control_protocol(model, monkeypatch):
    teacher = _window(verified=True)
    monkeypatch.setattr(engine, "_RECURRENT_REFERENCE_FRACTION", 1.0)
    # A complete training trajectory may contain future targets. This feature
    # is never written into the episode's causal control fields or an action tape.
    payload = engine.episode_control_payload(teacher.episode)
    injected, count = make_training_control(model, [teacher], torch.device("cpu"), recurrent=True)
    assert count == 1 and injected is not None
    assert engine.episode_control_payload(teacher.episode) == payload
    assert b"recorded good result" not in payload
    assert b"\n<reference>\n" in (
        engine.episode_control_payload(teacher.episode)
        + b"\n<reference>\n"
        + engine._successful_trajectory_reference(teacher.episode)
    )
