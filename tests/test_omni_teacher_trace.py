"""Predecoder acoustic trace is exact and never feeds a transcript into native R."""
from types import SimpleNamespace

import pytest
import torch

from megatoken.omni.teacher import capture_pocket_teacher


class FakeFlow:
    ldim = 3
    emb_mean = torch.tensor([1., 2., 3.])
    emb_std = torch.tensor([2., 2., 2.])

    def _sample_next_latent(self, *args, **kwargs):
        return torch.tensor([[.5, 1., 1.5]]), torch.tensor([[False]])


class FakeTeacher:
    def __init__(self):
        self.flow_lm = FakeFlow()
        self.observed_text = None

    def generate_audio(self, voice_state, spoken_text, *, copy_state):
        assert voice_state == "approved separate reference"
        assert copy_state is True
        self.observed_text = spoken_text
        self.flow_lm._sample_next_latent()
        self.flow_lm._sample_next_latent()
        return torch.zeros(1, 3840)


def test_teacher_emits_exact_native_predecoder_coordinates():
    teacher = FakeTeacher()
    original = teacher.flow_lm._sample_next_latent
    recorded = capture_pocket_teacher(
        teacher, voice_state="approved separate reference",
        spoken_text="The blue truck turns left.",
    )
    assert recorded.waveform.shape == (1, 3840)
    assert recorded.normalized_latents.shape == (2, 3)
    assert torch.equal(recorded.mimi_latents[0], torch.tensor([2., 4., 6.]))
    assert recorded.eos_flags == (False, False)
    assert teacher.observed_text == "The blue truck turns left."
    assert teacher.flow_lm._sample_next_latent == original


def test_teacher_cleans_capture_on_generator_failure():
    teacher = FakeTeacher()
    original = teacher.flow_lm._sample_next_latent
    def fail(*args, **kwargs):
        raise RuntimeError("teacher interrupted")
    teacher.generate_audio = fail
    with pytest.raises(RuntimeError, match="interrupted"):
        capture_pocket_teacher(
            teacher, voice_state="approved separate reference",
            spoken_text="Hello",
        )
    assert teacher.flow_lm._sample_next_latent == original
