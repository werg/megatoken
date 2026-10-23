"""SanaRealizer: scheduler flow convention, learned-sigma, video layout, frozen donors."""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from megatoken.model.media import MediaOutputSpec
from megatoken.omni.realizers import SanaRealizer

FACETS, WIDTH, CAPTION = 32, 8, 6
SHIFT, TRAIN_STEPS, TIMESTEP_SCALE = 3.0, 10, 2.0


class Transformer(nn.Module):
    """Records what the DiT is called with; predicts zero (or a chosen head)."""

    def __init__(self, learned_sigma=False):
        super().__init__()
        self.projection = nn.Linear(1, 1)
        self.config = SimpleNamespace(
            caption_channels=CAPTION, timestep_scale=TIMESTEP_SCALE, patch_size=1
        )
        self.learned_sigma = learned_sigma
        self.calls: list[dict] = []

    @property
    def dtype(self):
        return self.projection.weight.dtype

    def forward(
        self, noisy, *, encoder_hidden_states, encoder_attention_mask, timestep, return_dict
    ):
        assert encoder_hidden_states.shape[1:] == (FACETS, CAPTION)
        assert encoder_attention_mask.all() and encoder_attention_mask.dtype == torch.bool
        assert timestep.shape == (noisy.shape[0],) and return_dict is False
        self.calls.append(dict(noisy=noisy, timestep=timestep, cond=encoder_hidden_states))
        out = noisy * 0 + encoder_hidden_states.mean(dim=(1, 2)).view(-1, *([1] * (noisy.ndim - 1)))
        if self.learned_sigma:
            out = torch.cat((out, torch.full_like(out, 1e6)), dim=1)  # junk variance half
        return (out,)


class Config(dict):
    """diffusers' FrozenDict: a mapping with attribute access."""

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


def flow_config(**overrides):
    # The shipped Sana / Sana-Video DPMSolverMultistepScheduler in its flow mode.
    return Config(
        {
            "num_train_timesteps": TRAIN_STEPS,
            "prediction_type": "flow_prediction",
            "use_flow_sigmas": True,
            "use_dynamic_shifting": False,
            "flow_shift": SHIFT,
            **overrides,
        }
    )


class FlowScheduler:
    def __init__(self):
        self.config = flow_config()
        self.timesteps = torch.tensor([10.0, 5.0, 1.0])
        self.sigmas = torch.tensor([1.0, 0.5, 0.1, 0.0])

    def scale_noise(self, *args):
        raise AssertionError("The training objective must not use scheduler noise helpers")


class Vae(nn.Module):
    """Shape-faithful DC-AE / LTX-2 stand-in: 4 latent channels, 2x spatial pooling."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))
        self.config = SimpleNamespace(scaling_factor=0.5, latent_channels=4)
        self.spatial_compression_ratio = 2
        self.temporal_compression_ratio = 4
        self.latents_mean = torch.tensor([0.1, 0.2, 0.3, 0.4])
        self.latents_std = torch.tensor([1.0, 2.0, 3.0, 4.0])
        self.seen = None

    def encode(self, x, return_dict=True):
        self.seen = x
        assert return_dict
        spatial = nn.functional.avg_pool3d if x.ndim == 5 else nn.functional.avg_pool2d
        pooled = spatial(x, (1, 2, 2) if x.ndim == 5 else 2)
        latent = torch.cat((pooled, pooled[:, :1]), dim=1)
        return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: latent))


class Pipeline:
    def __init__(self, frames=0, learned_sigma=False, grow=0):
        self.grow = grow
        self.transformer = Transformer(learned_sigma)
        self.vae = Vae()
        self.scheduler = FlowScheduler()
        self.text_encoder, self.tokenizer = object(), object()
        self.frames, self.calls = frames, []

    def __call__(self, **kwargs):
        # The real pipeline refuses a text negative prompt next to negative embeds,
        # and its default negative prompt is "" (diffusers 0.41). The condition has
        # no trained null variant, so guidance is off and no null prompt is passed.
        assert "prompt" not in kwargs and kwargs["negative_prompt"] is None
        assert kwargs["guidance_scale"] == 1.0
        assert "negative_prompt_embeds" not in kwargs
        embeds = kwargs["prompt_embeds"]
        assert embeds.shape[1:] == (FACETS, CAPTION)
        assert kwargs["prompt_attention_mask"].all()
        self.calls.append(kwargs)
        batch = embeds.shape[0]
        h, w = kwargs["height"] + self.grow, kwargs["width"] + self.grow
        if self.frames:
            return (torch.ones(batch, self.frames, 3, h, w),)
        return (torch.ones(batch, 3, h, w),)


def realizer(modality="image", **kwargs) -> SanaRealizer:
    return SanaRealizer(Pipeline(**kwargs), modality, FACETS, WIDTH)


def raw(batch=1, seed=0):
    return torch.randn(batch, FACETS, WIDTH, generator=torch.Generator().manual_seed(seed))


def shifted(sigma):
    return SHIFT * sigma / (1 + (SHIFT - 1) * sigma)


def test_identity_names_and_dropped_donor_text_path():
    pipeline = Pipeline()
    image = SanaRealizer(pipeline, "image", FACETS, WIDTH)
    assert (image.name, image.modality) == ("sana", "image")
    assert pipeline.text_encoder is None and pipeline.tokenizer is None
    assert SanaRealizer(Pipeline(), "video", FACETS, WIDTH).name == "sana_video"
    with pytest.raises(ValueError, match="still images or video"):
        SanaRealizer(Pipeline(), "audio", FACETS, WIDTH)


def test_donor_is_outside_the_module_tree_and_frozen():
    r = realizer()
    names = [name for name, _ in r.named_parameters()]
    assert names and all(name.startswith("condition.") for name in names)
    assert all(key.startswith("condition.") for key in r.state_dict())
    assert r.condition.project.out_features == CAPTION
    assert all(not p.requires_grad for p in (*r.transformer.parameters(), *r.vae.parameters()))
    r.train()
    assert r.training and not r.transformer.training and not r.vae.training
    r.eval().train(True)
    assert not r.transformer.training


def test_image_loss_follows_the_scheduler_flow_convention():
    r = realizer()
    latents = torch.randn(2, 4, 8, 8, generator=torch.Generator().manual_seed(1))
    loss = r.loss(raw(2), latents, None, generator=torch.Generator().manual_seed(9))
    g = torch.Generator().manual_seed(9)
    noise = torch.randn(latents.shape, generator=g)
    sigma = shifted(torch.rand((2,), generator=g))
    (call,) = r.transformer.calls
    # Flow convention: sigma = 1 is pure noise, sigma = 0 is data; target noise - latents.
    s = sigma.view(2, 1, 1, 1)
    torch.testing.assert_close(call["noisy"], s * noise + (1 - s) * latents)
    # Static shift, continuous time over the training grid, image DiT timestep scale.
    torch.testing.assert_close(call["timestep"], sigma * TRAIN_STEPS * TIMESTEP_SCALE)
    cond = r.condition(raw(2)).mean(dim=(1, 2)).view(2, 1, 1, 1)
    expected = (cond - (noise - latents)).square().mean()
    torch.testing.assert_close(loss, expected)


def test_loss_has_gradients_for_the_condition_only_and_is_schedule_independent():
    pipeline = Pipeline()
    r = SanaRealizer(pipeline, "image", FACETS, WIDTH)
    latents = torch.randn(1, 4, 8, 8)
    first = r.loss(raw(), latents, None, generator=torch.Generator().manual_seed(9031))
    first.backward()
    grads = [p.grad for p in r.condition.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)
    assert all(p.grad is None for p in (*r.transformer.parameters(), *r.vae.parameters()))
    # A completed free rollout mutates the inference schedule; training time must not follow.
    pipeline.scheduler.timesteps = torch.tensor([4.0])
    pipeline.scheduler.sigmas = torch.tensor([0.4, 0.0])
    second = r.loss(raw(), latents, None, generator=torch.Generator().manual_seed(9031))
    assert torch.equal(first.detach(), second.detach())


def test_learned_sigma_head_uses_only_the_mean_channels():
    latents = torch.randn(1, 4, 8, 8)
    plain, junk = realizer(), realizer(learned_sigma=True)
    junk.condition.load_state_dict(plain.condition.state_dict())
    plain = plain.loss(raw(), latents, None, generator=torch.Generator().manual_seed(2))
    learned = junk.loss(raw(), latents, None, generator=torch.Generator().manual_seed(2))
    # (junk variance channels are 1e6 and would dominate any loss that read them)
    torch.testing.assert_close(plain, learned)


def test_video_latents_are_permuted_to_channel_first_and_the_mask_to_time():
    r = realizer("video")
    latents = torch.randn(1, 5, 4, 2, 2)  # time-major [B, F, C, h, w]
    mask = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.bool)
    loss = r.loss(raw(), latents, mask, generator=torch.Generator().manual_seed(3))
    (call,) = r.transformer.calls
    assert call["noisy"].shape == (1, 4, 5, 2, 2)  # [B, C, F, h, w]
    g = torch.Generator().manual_seed(3)
    channel_first = latents.permute(0, 2, 1, 3, 4)
    noise = torch.randn(channel_first.shape, generator=g)
    sigma = shifted(torch.rand((1,), generator=g))
    torch.testing.assert_close(call["timestep"], sigma * TRAIN_STEPS)
    cond = r.condition(raw()).mean(dim=(1, 2)).view(1, 1, 1, 1, 1)
    error = (cond - (noise - channel_first)).square()
    valid = mask.view(1, 1, 5, 1, 1).expand_as(error)
    torch.testing.assert_close(loss, (error * valid).sum() / valid.sum())
    assert valid.sum().item() == 4 * 3 * 2 * 2


def test_padded_video_frames_do_not_contribute_to_the_loss():
    r = realizer("video")
    latents = torch.randn(1, 4, 4, 2, 2)
    mask = torch.tensor([[1, 1, 0, 0]], dtype=torch.bool)
    changed = latents.clone()
    changed[:, 2:] = 50.0
    a = r.loss(raw(), latents, mask, generator=torch.Generator().manual_seed(1))
    b = r.loss(raw(), changed, mask, generator=torch.Generator().manual_seed(1))
    assert torch.allclose(a, b)
    unmasked = r.loss(raw(), changed, None, generator=torch.Generator().manual_seed(1))
    assert not torch.allclose(a, unmasked)


def test_loss_rejects_non_flow_scheduler_and_bad_shift():
    r = realizer()
    r.pipe.scheduler.config.flow_shift = -1.0
    with pytest.raises(ValueError, match="shift"):
        r.loss(raw(), torch.zeros(1, 4, 8, 8), None)
    for wrong in (
        {"prediction_type": "epsilon"},
        {"use_flow_sigmas": False},
        {"use_dynamic_shifting": True},
    ):
        r = realizer()
        r.pipe.scheduler = SimpleNamespace(config=flow_config(**wrong))
        with pytest.raises(ValueError, match="static-shift flow scheduler"):
            r.loss(raw(), torch.zeros(1, 4, 8, 8), None)


def test_image_realize_is_a_one_frame_clip_through_the_donor_pipeline():
    r = realizer()
    spec = MediaOutputSpec("image", height=32, width=32, steps=2)
    out = r.realize(raw(2), spec, generator=torch.Generator().manual_seed(0))
    assert out.shape == (2, 1, 3, 32, 32)
    call = r.pipe.calls[0]
    assert call["num_inference_steps"] == 2 and call["output_type"] == "pt"
    assert "frames" not in call and call["use_resolution_binning"] is False
    torch.testing.assert_close(call["prompt_embeds"], r.condition(raw(2)))


def test_video_realize_requests_frames_and_keeps_the_clip_layout():
    r = realizer("video", frames=9)
    spec = MediaOutputSpec("video", frames=9, height=32, width=32, fps_num=8)
    out = r.realize(raw(), spec)
    assert out.shape == (1, 9, 3, 32, 32)
    assert r.pipe.calls[0]["frames"] == 9


def test_realize_rejects_output_that_differs_from_the_requested_geometry():
    # The pipeline rounded the request to its own grid.
    r = realizer("image", grow=8)
    with pytest.raises(ValueError, match="size differs"):
        r.realize(raw(), MediaOutputSpec("image", height=32, width=32))
    v = realizer("video", frames=8)  # a clip one frame short of the request
    with pytest.raises(ValueError, match="frame count"):
        v.realize(raw(), MediaOutputSpec("video", frames=9, height=32, width=32, fps_num=8))


def test_realize_rejects_a_spec_off_the_pipeline_grid_before_generating():
    r = realizer("video", frames=9)
    for spec in (
        MediaOutputSpec("video", frames=8, height=32, width=32, fps_num=8),
        MediaOutputSpec("video", frames=9, height=33, width=32, fps_num=8),
    ):
        with pytest.raises(ValueError, match="Sana geometry"):
            r.realize(raw(), spec)
    assert r.pipe.calls == []


def test_realize_rejects_a_decode_with_the_wrong_layout():
    r = realizer("video", frames=0)  # the pipeline returns a still [B, 3, H, W]
    with pytest.raises(ValueError, match="frames"):
        r.realize(raw(), MediaOutputSpec("video", frames=9, height=32, width=32, fps_num=8))
    r = realizer("image", frames=4)  # the pipeline returns a clip
    with pytest.raises(ValueError, match="RGB image"):
        r.realize(raw(), MediaOutputSpec("image", height=32, width=32))


def test_image_encode_scales_with_the_vae_factor():
    r = realizer()
    latent = r.encode(torch.full((1, 3, 8, 8), 0.75))
    assert latent.shape == (1, 4, 4, 4) and latent.dtype == torch.float32
    # Pixels [0, 1] -> [-1, 1], then the checkpoint scaling factor.
    torch.testing.assert_close(latent, torch.full_like(latent, 0.5 * 0.5))


def test_video_encode_normalizes_per_channel_and_returns_time_major():
    r = realizer("video")
    latent = r.encode(torch.full((1, 3, 3, 8, 8), 0.75))  # [B, F, 3, H, W]
    assert r.vae.seen.shape == (1, 3, 3, 8, 8) and r.vae.seen.shape[1] == 3  # VAE reads [B,3,F,H,W]
    assert latent.shape == (1, 3, 4, 4, 4)  # [B, F, C, h, w]
    mean, std = r.vae.latents_mean, r.vae.latents_std
    expected = ((0.5 - mean) / std).view(1, 1, 4, 1, 1).expand_as(latent)
    torch.testing.assert_close(latent, expected)


def test_from_checkpoint_pins_a_local_snapshot_and_its_matching_vae(tmp_path, monkeypatch):
    class AutoencoderDC(Vae):
        pass

    pipeline = Pipeline()
    pipeline.vae = AutoencoderDC()
    loads = []

    def from_pretrained(snapshot, **kwargs):
        loads.append(kwargs)
        return pipeline

    monkeypatch.setitem(
        sys.modules,
        "diffusers",
        types.SimpleNamespace(
            SanaPipeline=SimpleNamespace(from_pretrained=from_pretrained),
            SanaVideoPipeline=SimpleNamespace(from_pretrained=from_pretrained),
        ),
    )
    kwargs = dict(facet_count=FACETS, facet_width=WIDTH)
    with pytest.raises(FileNotFoundError):
        SanaRealizer.from_checkpoint(snapshot=str(tmp_path / "gone"), modality="image", **kwargs)
    assert SanaRealizer.from_checkpoint(snapshot=str(tmp_path), modality="image", **kwargs)
    assert loads[0]["local_files_only"] is True  # never a Hub fallback
    with pytest.raises(ValueError, match="AutoencoderKLLTX2Video"):
        SanaRealizer.from_checkpoint(snapshot=str(tmp_path), modality="video", **kwargs)
