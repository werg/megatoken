"""Checkpoint API-facing donor contracts, using shape-faithful stub components."""
from types import SimpleNamespace

import torch
from torch import nn

from megatoken.omni.donors import SanaFacetsStudent


class DummyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(1, 1)
        self.config = SimpleNamespace(
            caption_channels=6,
            timestep_scale=1.0,
        )

    @property
    def dtype(self):
        return self.projection.weight.dtype

    def forward(self, noisy, *, encoder_hidden_states,
                encoder_attention_mask, timestep, return_dict):
        assert encoder_hidden_states.shape[1] == 32
        assert encoder_attention_mask.all()
        assert timestep.shape == (noisy.shape[0],)
        cond = encoder_hidden_states.mean(dim=(1, 2))
        while cond.ndim < noisy.ndim:
            cond = cond.unsqueeze(-1)
        return ((noisy * self.projection.weight[0, 0] + cond),)


class FlowScheduler:
    def __init__(self):
        self.config = SimpleNamespace(num_train_timesteps=10)
        self.timesteps = torch.tensor([10., 5., 1.])
        self.sigmas = torch.tensor([1., .5, .1, 0.])

    def scale_noise(self, *args):
        raise AssertionError("Native flow objective must not use an extra noise path")


class DummyPipeline:
    def __init__(self, frames):
        self.transformer = DummyTransformer()
        self.vae = nn.Linear(1, 1)
        self.scheduler = FlowScheduler()
        self.frames = frames

    def __call__(self, **kwargs):
        assert "prompt" not in kwargs
        assert kwargs["prompt_embeds"].shape == (1, 32, 6)
        if self.frames:
            return (torch.ones(1, self.frames, 3, 32, 32),)
        return (torch.ones(1, 3, 32, 32),)


def test_pretrained_sana_still_uses_native_one_frame_clip_geometry():
    student = SanaFacetsStudent(DummyPipeline(frames=0), "image", facet_width=8)
    realized = student.realize(torch.randn(1, 32, 8), height=32, width=32, steps=1)
    assert realized.shape == (1, 1, 3, 32, 32)


def test_pretrained_sana_uses_actual_flow_sigma_loss_with_gradients():
    student = SanaFacetsStudent(DummyPipeline(frames=0), "image", facet_width=8)
    raw = torch.randn(1, 32, 8)
    target = torch.randn(1, 4, 8, 8)
    numerator, count = student.native_loss(raw, target)
    assert count.item() == target.numel()
    (numerator / count).backward()
    assert student.condition[1].weight.grad is not None
    assert torch.isfinite(student.condition[1].weight.grad).all()
    assert all(p.grad is None for p in student.vae.parameters())


def test_video_pretrained_output_and_padded_time_loss():
    student = SanaFacetsStudent(DummyPipeline(frames=9), "video", facet_width=8)
    raw = torch.randn(1, 32, 8)
    visual = student.realize(raw, height=32, width=32, frames=9, steps=1)
    assert visual.shape == (1, 9, 3, 32, 32)
    target = torch.randn(1, 4, 3, 8, 8)
    valid = torch.tensor([[[[[True]], [[True]], [[False]]]]])
    numerator, count = student.native_loss(raw, target, valid_mask=valid)
    assert count.item() == 4 * 2 * 8 * 8
    assert numerator.isfinite()


def test_native_flow_loss_independent_of_prior_inference_schedule():
    pipe = DummyPipeline(frames=0)
    student = SanaFacetsStudent(pipe, "image", facet_width=8)
    raw = torch.randn(1, 32, 8)
    target = torch.randn(1, 4, 8, 8)
    gen = torch.Generator().manual_seed(9031)
    first, count = student.native_loss(raw, target, generator=gen)
    # A completed free rollout changes the scheduler's inference schedule.
    pipe.scheduler.timesteps = torch.tensor([4.])
    pipe.scheduler.sigmas = torch.tensor([0.4, 0.0])
    second, count2 = student.native_loss(
        raw, target, generator=torch.Generator().manual_seed(9031),
    )
    assert count.item() == count2.item()
    assert torch.allclose(first, second, rtol=0, atol=0)
