"""Pocket Mimi decoder state must scale with the codec's internal frame clock."""

import sys
import types
from types import SimpleNamespace

import torch
from torch import nn

from megatoken.omni.donors import PocketFacetsStudent


class DummyFlowLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.dim = 16
        self.ldim = 32
        self.transformer = nn.Linear(16, 16)
        self.flow_net = nn.Linear(32, 32)
        self.conditioner = nn.Linear(16, 16)
        self.register_buffer("emb_mean", torch.zeros(32))
        self.register_buffer("emb_std", torch.ones(32))


class DummyMimi(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(32, 32)
        self.quantizer = SimpleNamespace(dimension=32)
        self.frame_rate = 12.5
        self.encoder_frame_rate = 200.0
        self.sample_rate = 24000
        self.requested_state_capacity = None

    def decode_from_latent(self, latents, state):
        self.requested_state_capacity = state
        return latents.new_zeros((latents.shape[0], 1, 1920 * latents.shape[1]))


class DummyPocket(nn.Module):
    def __init__(self):
        super().__init__()
        self.flow_lm = DummyFlowLM()
        self.mimi = DummyMimi()
        self.temp = 0.3
        self.noise_clamp = 5.0
        self.eos_threshold = 0.5


def test_pocket_decode_uses_native_mimi_upsampler_clock(monkeypatch):
    stateful = types.ModuleType("pocket_tts.modules.stateful_module")
    stateful.init_states = lambda module, batch, max_seq_len: max_seq_len
    root = types.ModuleType("pocket_tts")
    root.__path__ = []
    modules = types.ModuleType("pocket_tts.modules")
    modules.__path__ = []
    monkeypatch.setitem(sys.modules, "pocket_tts", root)
    monkeypatch.setitem(sys.modules, "pocket_tts.modules", modules)
    monkeypatch.setitem(sys.modules, "pocket_tts.modules.stateful_module", stateful)
    student = PocketFacetsStudent(DummyPocket(), facet_width=8)
    audio = student.decode(torch.zeros(1, 10, 32))
    assert audio.shape == (1, 19200)
    assert student.mimi.requested_state_capacity == 10 * 16 + 16
    assert student.flow_lm.conditioner is None
    assert "pocket.flow_lm.transformer.weight" not in student.state_dict()
    assert "flow_lm.transformer.weight" in student.state_dict()
