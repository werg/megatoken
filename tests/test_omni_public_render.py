"""Public render geometry never reads teacher-only codec targets."""
import pytest
import torch
from torch import nn

from megatoken.model.config import ModelConfig
from megatoken.model.media import (
    MediaOutputSpec,
    MediaRealizers,
    validate_public_media_spec,
)
from megatoken.model.state import RealizationContext


def test_public_media_shape_cannot_use_training_teacher_metadata():
    teacher = MediaOutputSpec(
        "audio", renderer="pocket", sample_rate=24000,
        sample_count=1920, latent_shape=(1, 32),
        shape_provenance="teacher_only",
    )
    with pytest.raises(ValueError, match="Teacher-only"):
        validate_public_media_spec(teacher)
    with pytest.raises(ValueError, match="codec geometry"):
        validate_public_media_spec(
            MediaOutputSpec("image", renderer="sana", height=64, width=64,
                            latent_shape=(32, 2, 2))
        )
    validate_public_media_spec(
        MediaOutputSpec("video", renderer="sana_video", frames=17,
                        height=64, width=64)
    )


def test_public_pocket_waveform_trimming_is_deterministic():
    class ShortUtterance(nn.Module):
        sample_rate = 24000
        frame_rate = 12.5

        def realize(self, raw, *, max_frames, steps, generator):
            assert raw.shape[1] == 32
            assert max_frames == 2
            return torch.ones(raw.shape[0], 2000)

    config = ModelConfig.tiny(n_facets=32)
    realizer = MediaRealizers(config)
    realizer.register_pretrained("pocket", ShortUtterance())
    spec = MediaOutputSpec(
        "audio", renderer="pocket", sample_rate=24000, sample_count=2400,
    )
    raw = RealizationContext(torch.zeros(1, 32, config.facet_dim))
    output = realizer.realize(raw, spec)
    assert output.shape == (1, 2400)
    assert torch.count_nonzero(output[:, 2000:]) == 0
    assert torch.equal(output[:, :2000], torch.ones(1, 2000))
    with pytest.raises(ValueError, match="native rate"):
        realizer.realize(
            raw, MediaOutputSpec("audio", renderer="pocket", sample_rate=16000,
                                 sample_count=1600),
        )
