"""PocketRealizer: Mimi decoder clock, Pocket's own flow objectives, frozen-donor ownership."""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from megatoken.model.media import MediaOutputSpec
from megatoken.omni.realizers import PocketRealizer

FACETS, WIDTH, DIM, LDIM = 4, 8, 16, 4


class FlowNet(nn.Module):
    """Pocket's flow net: (hidden, t, x) for OT, (hidden, s, t, x) for LSD."""

    def __init__(self, record=None):
        super().__init__()
        self.ot = nn.Linear(DIM + 1 + LDIM, LDIM)
        self.lsd = nn.Linear(DIM + 2 + LDIM, LDIM)
        self.record = record

    def forward(self, hidden, *args):
        if self.record is not None:
            self.record.append(args)
        if len(args) == 2:
            t, x = args
            return self.ot(torch.cat((hidden, t, x), dim=-1)).tanh()
        s, t, x = args
        return self.lsd(torch.cat((hidden, s, t, x), dim=-1)).tanh()


class ZeroFlowNet(nn.Module):
    """Predicts zero so the loss is the exact target energy; records its arguments."""

    def __init__(self, record):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.record = record

    def forward(self, hidden, *args):
        self.record.append(args)
        return torch.zeros_like(args[-1]) + 0 * self.anchor


class Transformer(nn.Module):
    def __init__(self, record=None):
        super().__init__()
        self.linear = nn.Linear(DIM, DIM)
        self.record = record

    def forward(self, x, model_state=None):
        assert model_state is None
        if self.record is not None:
            self.record.append(x.detach().clone())
        return self.linear(x)


class FlowLM(nn.Module):
    def __init__(self, flow_type="flow_matching", flow_net=None, transformer=None):
        super().__init__()
        self.dim, self.ldim, self.flow_type = DIM, LDIM, flow_type
        self.transformer = transformer or Transformer()
        self.input_linear = nn.Linear(LDIM, DIM)
        self.out_norm = nn.LayerNorm(DIM)
        self.flow_net = flow_net or FlowNet()
        self.bos_emb = nn.Parameter(torch.randn(LDIM))
        self.conditioner = nn.Linear(DIM, DIM)
        self.register_buffer("emb_mean", torch.linspace(-0.5, 0.5, LDIM))
        self.register_buffer("emb_std", torch.linspace(0.5, 1.5, LDIM))
        self.states: list = []
        self.eos_at = 3

    def _sample_next_latent(self, previous, embeddings, states, steps, temp, clamp, eos_threshold):
        self.states.append((previous.shape, embeddings.shape, steps))
        if embeddings.shape[1]:
            states["facets"] = embeddings.mean(dim=1)[:, :LDIM]
        noise = torch.randn(previous.shape[0], LDIM, device=previous.device)
        step = len(self.states) - 1
        eos = torch.full((previous.shape[0], 1), step >= self.eos_at)
        return (states["facets"] + 0.1 * noise).tanh(), eos


class Mimi(nn.Module):
    def __init__(self, encoder_rate=200.0):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))
        self.quantizer = SimpleNamespace(dimension=LDIM)
        self.frame_rate = 12.5
        self.encoder_frame_rate = encoder_rate
        self.sample_rate = 24000
        self.requested_state_capacity = None
        self.encoded_input = None

    def encode_to_latent(self, waveform):
        self.encoded_input = waveform
        frames = waveform.shape[-1] // 1920
        return torch.ones(waveform.shape[0], frames, LDIM, dtype=waveform.dtype)

    def decode_from_latent(self, latents, state):
        self.requested_state_capacity = state
        return latents.new_ones((latents.shape[0], 1, 1920 * latents.shape[1]))


class Pocket(nn.Module):
    def __init__(self, flow_lm=None, mimi=None):
        super().__init__()
        self.flow_lm = flow_lm or FlowLM()
        self.mimi = mimi or Mimi()
        self.temp, self.noise_clamp, self.eos_threshold = 0.3, 5.0, 0.5


@pytest.fixture(autouse=True)
def stub_pocket_runtime(monkeypatch):
    """The real ``pocket_tts`` is an optional extra; stub its two stateful helpers."""
    stateful = types.ModuleType("pocket_tts.modules.stateful_module")
    stateful.init_states = lambda module, batch, max_seq_len: {"capacity": max_seq_len}
    stateful.increment_steps = lambda module, states, increment: None
    root = types.ModuleType("pocket_tts")
    root.__path__ = []
    modules = types.ModuleType("pocket_tts.modules")
    modules.__path__ = []
    monkeypatch.setitem(sys.modules, "pocket_tts", root)
    monkeypatch.setitem(sys.modules, "pocket_tts.modules", modules)
    monkeypatch.setitem(sys.modules, "pocket_tts.modules.stateful_module", stateful)


def realizer(**pocket) -> PocketRealizer:
    return PocketRealizer(Pocket(**pocket), FACETS, WIDTH)


def raw(batch=1, seed=0):
    return torch.randn(batch, FACETS, WIDTH, generator=torch.Generator().manual_seed(seed))


def test_decode_state_capacity_follows_the_mimi_encoder_clock():
    r = realizer()
    audio = r.decode(torch.zeros(1, 10, LDIM))
    assert audio.shape == (1, 19200)
    # 200 Hz encoder / 12.5 Hz latents = 16x: the decoder advances at the encoder rate.
    assert r.mimi.requested_state_capacity == {"capacity": 10 * 16 + 16}
    r = realizer(mimi=Mimi(encoder_rate=100.0))
    r.decode(torch.zeros(1, 10, LDIM))
    assert r.mimi.requested_state_capacity == {"capacity": 10 * 8 + 16}
    with pytest.raises(ValueError, match="integer"):
        realizer(mimi=Mimi(encoder_rate=30.0)).decode(torch.zeros(1, 2, LDIM))


def test_decode_rejects_non_mono_output():
    class Stereo(Mimi):
        def decode_from_latent(self, latents, state):
            return latents.new_ones((latents.shape[0], 2, 100))

    with pytest.raises(ValueError, match="mono"):
        realizer(mimi=Stereo()).decode(torch.zeros(1, 2, LDIM))


def test_encode_is_native_mono_and_returns_fp32_latents():
    r = realizer()
    latents = r.encode(torch.zeros(2, 1920 * 5))
    assert latents.shape == (2, 5, LDIM) and latents.dtype == torch.float32
    assert r.mimi.encoded_input.shape == (2, 1, 1920 * 5)
    with pytest.raises(ValueError, match="mono"):
        r.encode(torch.zeros(1, 1, 1920))


def test_donor_is_outside_the_module_tree_and_frozen():
    pocket = Pocket()
    r = PocketRealizer(pocket, FACETS, WIDTH)
    assert pocket.flow_lm.conditioner is None  # teacher-only token table dropped
    names = [name for name, _ in r.named_parameters()]
    assert names and all(name.startswith("condition.") for name in names)
    assert all(key.startswith("condition.") for key in r.state_dict())
    assert not any("flow_lm" in key or "mimi" in key for key in r.state_dict())
    assert all(not p.requires_grad for p in pocket.parameters())
    # Module.train() must not wake the donor, however often the trainer calls it.
    r.train()
    assert r.training and r.condition.training
    assert not pocket.training and not pocket.flow_lm.training and not pocket.mimi.training
    r.eval().train(True)
    assert not pocket.flow_lm.training


def test_identity_requires_a_locked_pack():
    r = realizer()
    with pytest.raises(ValueError, match="locked"):
        _ = r.identity
    r.checkpoint_identity = "ab" * 32
    assert r.identity == "pocket:" + "ab" * 32


def test_flow_matching_loss_is_pockets_ot_objective():
    record: list = []
    r = realizer(flow_lm=FlowLM(flow_net=ZeroFlowNet(record)))
    latents = torch.randn(2, 5, LDIM, generator=torch.Generator().manual_seed(1))
    mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]], dtype=torch.bool)
    loss = r.loss(raw(2), latents, mask, generator=torch.Generator().manual_seed(7))
    generator = torch.Generator().manual_seed(7)
    lm = r.flow_lm
    clean = (latents - lm.emb_mean) / lm.emb_std
    noise = torch.randn(clean.shape, generator=generator)
    t = torch.rand((2, 5, 1), generator=generator)
    sig = 0.001
    ((t_seen, x_t),) = record
    torch.testing.assert_close(t_seen, t)
    torch.testing.assert_close(x_t, (1 - (1 - sig) * t) * noise + t * clean)
    # Prediction is zero, so the loss is the velocity-target energy over valid frames.
    error = (clean - (1 - sig) * noise).square() * mask[..., None]
    torch.testing.assert_close(loss, error.sum() / (mask.sum() * LDIM))


def test_flow_matching_loss_ignores_padded_frames_and_trains_only_the_condition():
    r = realizer()
    latents = torch.randn(1, 5, LDIM)
    mask = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.bool)
    first = r.loss(raw(), latents, mask, generator=torch.Generator().manual_seed(3))
    padded = latents.clone()
    padded[:, 3:] = 100.0
    second = r.loss(raw(), padded, mask, generator=torch.Generator().manual_seed(3))
    torch.testing.assert_close(first, second)
    first.backward()
    assert all(p.grad is not None for p in r.condition.parameters())
    assert all(p.grad is None for p in r.pocket.parameters())


def test_loss_is_teacher_forced_with_bos_and_no_future_leak():
    record: list = []
    r = realizer(flow_lm=FlowLM(transformer=Transformer(record)))
    lm = r.flow_lm
    latents = torch.randn(1, 4, LDIM)
    r.loss(raw(), latents, None, generator=torch.Generator().manual_seed(0))
    changed = latents.clone()
    changed[:, -1] += 5.0  # the last frame is a target only, never an input
    r.loss(raw(), changed, None, generator=torch.Generator().manual_seed(0))
    first, second = record
    assert first.shape == (1, FACETS + 4, DIM)
    torch.testing.assert_close(first, second)
    clean = (latents - lm.emb_mean) / lm.emb_std
    torch.testing.assert_close(first[:, FACETS], lm.input_linear(lm.bos_emb)[None])
    torch.testing.assert_close(first[:, FACETS + 1 :], lm.input_linear(clean[:, :-1]))


def test_loss_rejects_wrong_geometry_and_flow_type():
    r = realizer()
    with pytest.raises(ValueError, match="continuous"):
        r.loss(raw(), torch.zeros(1, 3, LDIM + 1), None)
    with pytest.raises(ValueError, match="batch"):
        r.loss(raw(2), torch.zeros(1, 3, LDIM), None)
    with pytest.raises(ValueError, match="complete raw facet"):
        r.loss(torch.randn(1, FACETS - 1, WIDTH), torch.zeros(1, 3, LDIM), None)
    with pytest.raises(ValueError, match="LSD or OT"):
        realizer(flow_lm=FlowLM(flow_type="rectified")).loss(raw(), torch.zeros(1, 3, LDIM), None)


class LinearLSDNet(nn.Module):
    """f(h, s, t, x) = a x + c t: closed-form JVP, so the LSD objective is checkable."""

    a, c = 0.5, 0.25

    def forward(self, hidden, s, t, x):
        return self.a * x + self.c * t


def test_lsd_loss_is_diagonal_plus_stop_gradient_self_distillation():
    r = realizer(flow_lm=FlowLM("lsd", flow_net=LinearLSDNet()))
    latents = torch.randn(1, 4, LDIM, generator=torch.Generator().manual_seed(4))
    mask = torch.tensor([[1, 1, 1, 0]], dtype=torch.bool)
    torch.manual_seed(11)
    loss = r.loss(raw(), latents, mask)
    torch.manual_seed(11)
    lm, a, c = r.flow_lm, LinearLSDNet.a, LinearLSDNet.c
    clean = (latents - lm.emb_mean) / lm.emb_std
    noise = torch.randn(clean.shape)
    t = torch.sigmoid(torch.randn(1, 4, 1) + 0.4)
    logits = torch.randn(1, 4, 2) + 0.4
    s = torch.sigmoid(logits.min(-1, keepdim=True).values)
    end = torch.sigmoid(logits.max(-1, keepdim=True).values)
    x_t, x_s = t * clean + (1 - t) * noise, s * clean + (1 - s) * noise
    diagonal = a * x_t + c * t
    vt = a * x_s + c * end
    target = a * (x_s + (end - s) * vt) + c * end
    weight = mask[..., None]

    def mse(p, q):
        return ((p - q).square() * weight).sum() / (mask.sum() * LDIM)

    expected = 0.75 * mse(diagonal, clean - noise) + 0.25 * mse(vt + (end - s) * c, target)
    torch.testing.assert_close(loss, expected)


def test_lsd_loss_backpropagates_through_the_jvp_to_the_condition_only():
    r = realizer(flow_lm=FlowLM("lsd"))
    loss = r.loss(raw(), torch.randn(1, 4, LDIM), None)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in r.condition.parameters()
    )
    assert all(p.grad is None for p in r.pocket.parameters())


def test_realize_trims_to_the_requested_sample_count():
    r = realizer()
    spec = MediaOutputSpec("audio", sample_rate=24000, sample_count=5000, steps=3)
    audio = r.realize(raw(2), spec, generator=torch.Generator().manual_seed(0))
    assert audio.shape == (2, 5000)
    # 5000 samples at 12.5 latent Hz -> 3 frames; the facet prefix is fed on step zero only.
    shapes = r.flow_lm.states
    assert shapes[0][1] == (2, FACETS, DIM) and shapes[1][1][1] == 0
    assert shapes[0][2] == 3
    assert torch.equal(audio, torch.ones(2, 5000))


def test_realize_pads_early_eos_with_silence():
    r = realizer()
    spec = MediaOutputSpec("audio", sample_rate=24000, sample_count=20000)
    audio = r.realize(raw(), spec, generator=torch.Generator().manual_seed(0))
    assert audio.shape == (1, 20000)
    frames = r.flow_lm.eos_at + 1  # EOS never ends speech before four frames
    assert torch.equal(audio[:, : frames * 1920], torch.ones(1, frames * 1920))
    assert not audio[:, frames * 1920 :].any()


def test_realize_is_replayable_under_an_explicit_generator():
    r = realizer()
    spec = MediaOutputSpec("audio", sample_rate=24000, sample_count=6000)
    a = r.latent(raw(), max_frames=3, steps=2, generator=torch.Generator().manual_seed(5))
    b = r.latent(raw(), max_frames=3, steps=2, generator=torch.Generator().manual_seed(5))
    c = r.latent(raw(), max_frames=3, steps=2, generator=torch.Generator().manual_seed(6))
    assert torch.equal(a, b) and not torch.equal(a, c)
    assert r.realize(raw(), spec).shape == (1, 6000)
    with pytest.raises(ValueError, match="Unbounded"):
        r.latent(raw(), max_frames=0, steps=2)
    with pytest.raises(ValueError, match="Unbounded"):
        r.latent(raw(), max_frames=2, steps=129)


@pytest.mark.parametrize("frames", [7, 14, 28, 55, 109, 218])
def test_frame_count_is_exact_for_whole_frame_lengths(frames):
    # Float division rounds these exact multiples of 1920 samples up to one frame too many.
    r = realizer()
    r.flow_lm.eos_at = 10**6  # never stop early: the step count is the requested frame count
    spec = MediaOutputSpec("audio", sample_rate=24000, sample_count=1920 * frames)
    assert r.realize(raw(), spec).shape == (1, 1920 * frames)
    assert len(r.flow_lm.states) == frames


def test_generator_device_plumbing_draws_the_seed_on_the_generators_device():
    r = realizer()
    generator = torch.Generator().manual_seed(5)
    a = r.latent(raw(), max_frames=2, steps=2, generator=generator)
    b = r.latent(raw(), max_frames=2, steps=2, generator=torch.Generator().manual_seed(5))
    assert torch.equal(a, b)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_latent_accepts_a_generator_on_the_models_device():
    r = realizer().to_device("cuda")
    cuda_raw = raw().cuda()
    a = r.latent(cuda_raw, max_frames=3, steps=2, generator=torch.Generator("cuda").manual_seed(5))
    b = r.latent(cuda_raw, max_frames=3, steps=2, generator=torch.Generator("cuda").manual_seed(5))
    assert a.is_cuda and torch.equal(a, b)


def test_from_checkpoint_needs_a_local_pinned_yaml(tmp_path, monkeypatch):
    runtime = sys.modules["pocket_tts"]
    runtime.TTSModel = SimpleNamespace(load_model=lambda **kwargs: Pocket())
    with pytest.raises(FileNotFoundError):
        PocketRealizer.from_checkpoint(
            config=str(tmp_path / "missing.yaml"),
            checkpoint=None,
            facet_count=FACETS,
            facet_width=WIDTH,
        )
    config = tmp_path / "pocket.yaml"
    config.write_text("{}")
    r = PocketRealizer.from_checkpoint(
        config=str(config), checkpoint=None, facet_count=FACETS, facet_width=WIDTH
    )
    assert r.sample_rate == 24000 and r.latent_width == LDIM


def test_mismatched_flow_and_mimi_latent_widths_are_rejected():
    mimi = Mimi()
    mimi.quantizer = SimpleNamespace(dimension=LDIM + 1)
    with pytest.raises(ValueError, match="geometry"):
        realizer(mimi=mimi)


def test_pocket_yaml_resources_resolve_against_the_yaml_and_must_be_locked(tmp_path):
    import hashlib
    import json

    from megatoken.omni.loader import read_locked_manifest
    from megatoken.omni.realizers import resolve_pocket_config

    pack = tmp_path / "pocket"
    (pack / "english").mkdir(parents=True)
    (pack / "english" / "model.safetensors").write_bytes(b"weights")
    (pack / "english" / "tokenizer.json").write_text("{}")
    (pack / "pocket.yaml").write_text(
        "weights_path: english/model.safetensors\n"
        "flow_lm:\n  lookup_table:\n    tokenizer_path: english/tokenizer.json\n"
    )
    mapping, resources = resolve_pocket_config(pack / "pocket.yaml")
    assert mapping["weights_path"] == str(pack / "english" / "model.safetensors")
    assert mapping["flow_lm"]["lookup_table"]["tokenizer_path"] == str(
        pack / "english" / "tokenizer.json"
    )
    assert len(resources) == 2

    def lock(names):
        files = {
            f"pocket/{name}": hashlib.sha256((pack / name).read_bytes()).hexdigest()
            for name in names
        }
        item = {"revision": "a" * 40, "config": "pocket/pocket.yaml", "files": files}
        path = tmp_path / "lock.json"
        path.write_text(json.dumps({"version": 1, "packs": {"pocket": item}}))
        return str(path)

    read_locked_manifest(
        lock(["pocket.yaml", "english/model.safetensors", "english/tokenizer.json"])
    )
    with pytest.raises(ValueError, match="Unhashed Pocket resource"):
        read_locked_manifest(lock(["pocket.yaml", "english/tokenizer.json"]))
    (pack / "pocket.yaml").write_text("weights_path: hf://kyutai/pocket-tts/model.safetensors\n")
    with pytest.raises(ValueError, match="Remote Pocket resource"):
        resolve_pocket_config(pack / "pocket.yaml")
