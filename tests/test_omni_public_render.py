"""Public media output specs, resident realizers and deterministic realization."""

import pytest
import torch

from megatoken.data.media_assets import OutputGeometry
from megatoken.generation.codec import ByteCodec
from megatoken.generation.policy import FacetedGenerator
from megatoken.model import FacetedCore, ModelConfig
from megatoken.model.media import (
    MediaOutputSpec,
    MediaRealizers,
    MediaTrainTarget,
    validate_public_media_spec,
)
from megatoken.model.multitrack import MultiTrackGenerator
from megatoken.model.state import RealizationContext
from megatoken.teacher import PINNED_TEACHER
from tests.media_support import LATENT, FakeRealizer, features, geometry, install_fake_realizers


class NoisyRealizer(FakeRealizer):
    """A realizer whose output also depends on its sampler generator."""

    def realize(self, raw, spec, generator=None):
        base = super().realize(raw, spec, generator)
        return base + 0.1 * torch.randn(base.shape, generator=generator)


def _context(config: ModelConfig, seed: int = 0) -> RealizationContext:
    generator = torch.Generator().manual_seed(seed)
    return RealizationContext(
        torch.randn(1, config.n_facets, config.facet_dim, generator=generator)
    )


def test_public_media_shape_cannot_use_training_teacher_metadata():
    teacher = MediaOutputSpec.from_geometry(geometry("audio"))
    assert teacher.provenance == "teacher_only"  # a gold target's extent
    with pytest.raises(ValueError, match="Teacher-only"):
        validate_public_media_spec(teacher)
    # The same extent, but requested by the caller, is public.
    validate_public_media_spec(MediaOutputSpec("audio", sample_count=48, sample_rate=24000))
    validate_public_media_spec(MediaOutputSpec("video", frames=17, height=64, width=64, fps_num=8))
    for modality in ("image", "video"):
        spec = MediaOutputSpec.from_geometry(geometry(modality))
        with pytest.raises(ValueError, match="Teacher-only"):
            validate_public_media_spec(spec)


def test_output_spec_carries_only_physical_bounds():
    with pytest.raises(ValueError, match="provenance"):
        MediaOutputSpec("image", height=16, width=16, provenance="oracle")
    with pytest.raises(ValueError, match="24 kHz"):
        MediaOutputSpec("audio", sample_rate=16000, sample_count=1600)
    with pytest.raises(ValueError, match="visual geometry"):
        MediaOutputSpec("audio", sample_rate=24000, sample_count=100, height=16)
    with pytest.raises(ValueError, match="audio clock"):
        MediaOutputSpec("image", height=16, width=16, sample_count=10)
    with pytest.raises(ValueError, match="one frame"):
        MediaOutputSpec("image", height=16, width=16, frames=2)
    with pytest.raises(ValueError, match="frame clock"):
        MediaOutputSpec("video", frames=4, height=16, width=16)
    with pytest.raises(ValueError, match="audio, image or video"):
        MediaOutputSpec("text")
    # A prepared record's geometry converts field for field, provenance included.
    spec = MediaOutputSpec.from_geometry(
        OutputGeometry("video", frames=4, height=16, width=16, fps_num=8, provenance="requested"),
        steps=5,
    )
    assert (spec.frames, spec.height, spec.fps_num, spec.steps) == (4, 16, 8, 5)
    validate_public_media_spec(spec)


def test_free_generation_rejects_teacher_only_extent():
    torch.manual_seed(3)
    core = FacetedCore(ModelConfig.tiny())
    model = install_fake_realizers(FacetedGenerator(core, ByteCodec.byte_fallback()))
    service = MultiTrackGenerator(model)
    state = service.tracks.initial_state(1)
    with pytest.raises(ValueError, match="Teacher-only"):
        service.generate_group(
            state,
            tracks=("voice",),
            specs=(MediaOutputSpec.from_geometry(geometry("audio")),),
            effort=1,
        )


def test_one_resident_realizer_per_modality():
    config = ModelConfig.tiny()
    realizers = MediaRealizers()
    for modality in ("audio", "image"):
        with pytest.raises(ValueError, match="not installed"):
            realizers.resident(modality)
    audio = FakeRealizer("audio", config.n_facets, config.facet_dim)
    realizers.register(audio)
    assert realizers.resident("audio") is audio
    assert list(realizers.packs) == ["pocket"]
    with pytest.raises(ValueError, match="One locked realizer"):
        realizers.register(FakeRealizer("audio", config.n_facets, config.facet_dim))
    wrong = FakeRealizer("image", config.n_facets, config.facet_dim)
    wrong.name = "pocket"  # an image realizer may not squat the audio slot
    with pytest.raises(ValueError, match="One locked realizer"):
        realizers.register(wrong)
    with pytest.raises(ValueError, match="not installed"):
        realizers.realize(_context(config), MediaOutputSpec("image", height=16, width=16))


def test_realization_shapes_follow_the_requested_spec():
    config = ModelConfig.tiny()
    model = install_fake_realizers(FacetedGenerator(FacetedCore(config), ByteCodec.byte_fallback()))
    context = _context(config)
    audio = model.media_realizers.realize(
        context, MediaOutputSpec("audio", sample_count=2400, sample_rate=24000)
    )
    image = model.media_realizers.realize(context, MediaOutputSpec("image", height=16, width=24))
    video = model.media_realizers.realize(
        context, MediaOutputSpec("video", frames=3, height=16, width=24, fps_num=8)
    )
    assert audio.shape == (1, 2400)
    assert image.shape == (1, 1, 3, 16, 24)
    assert video.shape == (1, 3, 3, 16, 24)
    assert image.min() >= 0 and image.max() <= 1


def test_realization_is_deterministic_under_a_fixed_generator():
    config = ModelConfig.tiny()
    realizers = MediaRealizers()
    realizers.register(NoisyRealizer("audio", config.n_facets, config.facet_dim))
    context = _context(config, seed=1)
    spec = MediaOutputSpec("audio", sample_count=480, sample_rate=24000)

    def sample(seed):
        return realizers.realize(context, spec, torch.Generator().manual_seed(seed))

    torch.testing.assert_close(sample(7), sample(7), rtol=0, atol=0)
    assert not torch.equal(sample(7), sample(8))
    # Realization is a function of raw facets: different facets, different media.
    other = realizers.realize(_context(config, seed=2), spec, torch.Generator().manual_seed(7))
    assert not torch.equal(sample(7), other)


def test_realizer_loss_requires_the_pack_that_encoded_the_latents():
    config = ModelConfig.tiny()
    realizers = MediaRealizers()
    realizer = FakeRealizer("image", config.n_facets, config.facet_dim)
    realizers.register(realizer)
    spec = MediaOutputSpec("image", height=16, width=16)

    def target(identity):
        return MediaTrainTarget(
            "image",
            spec,
            features(),
            PINNED_TEACHER,
            torch.zeros(1, *LATENT["image"]),
            None,
            identity,
            "bootstrap",
        )

    context = _context(config)
    loss = realizers.loss(context, target(realizer.identity))
    assert loss.isfinite() and loss.ndim == 0
    with pytest.raises(ValueError, match="different realizer pack"):
        realizers.loss(context, target("sana:" + "0" * 64))
