"""Native media E/R adapters, causal controls and heterogeneous meridian replay."""

import math

import pytest
import torch

from megatoken.controls import ControlBundle, ControlReference, build_control_payload
from megatoken.data.schema import MediaSpan, Meridian, TrackEvent
from megatoken.generation.codec import ByteCodec
from megatoken.generation.latents import DiagonalGaussian
from megatoken.generation.policy import FacetedGenerator
from megatoken.generation.structure import ActionSpec
from megatoken.model import FacetedCore, ModelConfig
from megatoken.model.media import MediaOutputSpec
from megatoken.model.multitrack import (
    GroupActionTape,
    MultiTrackGenerator,
    ObservedEvent,
)


def _system():
    torch.manual_seed(91)
    core = FacetedCore(ModelConfig.tiny(max_history_chunks=8))
    model = FacetedGenerator(core, ByteCodec.byte_fallback())
    service = MultiTrackGenerator(model)
    return core, model, service


def test_heterogeneous_ingestion_one_meridian():
    core, _, service = _system()
    state = service.tracks.initial_state(1)
    observations = [
        ObservedEvent("user", "text", b"Describe the clip"),
        ObservedEvent("mic", "audio", torch.randn(1, 2400) * 0.1),
        ObservedEvent("camera", "video", torch.rand(1, 2, 3, 16, 16)),
        ObservedEvent("photo", "image", torch.rand(1, 3, 16, 16)),
    ]
    next_state = service.tracks.observe_group(state, observations)
    assert next_state.raw.shape == (1, 4, core.config.n_facets, core.config.facet_dim)
    assert next_state.history.slots == 4 * core.config.n_facets
    assert next_state.meridian_sizes == (4,)
    assert next_state.frontier_tracks == ("user", "mic", "camera", "photo")


def test_mixed_media_generation_and_replay():
    core, _, service = _system()
    state = service.tracks.observe_text_group(
        service.tracks.initial_state(1), (b"Generate a picture and sound",), tracks=("prompt",)
    )
    specifications = (
        ActionSpec(max_tokens=8),
        MediaOutputSpec(modality="audio", sample_count=256),
        MediaOutputSpec(modality="image", frames=1, height=16, width=16),
        MediaOutputSpec(modality="video", frames=2, height=16, width=16),
    )
    result = service.generate_group(
        state, tracks=("reply", "voice", "still", "clip"),
        specs=specifications, effort=1, generator=torch.Generator().manual_seed(13),
    )
    assert result.tape.modalities == ("text", "audio", "image", "video")
    assert result.media_outputs[0] is None
    assert result.media_outputs[1].shape == (1, 256)
    assert result.media_outputs[2].shape == (1, 1, 3, 16, 16)
    assert result.media_outputs[3].shape == (1, 2, 3, 16, 16)
    tape = GroupActionTape.from_json(result.tape.to_json())
    replay = service.replay_group(state, tape, specs=specifications, effort=1)
    assert replay.tape.payloads == result.tape.payloads
    torch.testing.assert_close(replay.state.final, result.state.final)
    with pytest.raises(ValueError, match="specifications"):
        service.replay_group(state, tape, specs=(
            ActionSpec(max_tokens=8),
            MediaOutputSpec("audio", sample_count=128),
            *specifications[2:],
        ), effort=1)


def test_grouped_multimodal_bootstrap_has_trainable_gradients():
    core, model, service = _system()
    context = service.tracks.initial_state(1)
    image = torch.rand(1, 1, 3, 16, 16)
    training = service.supervised_group(
        context,
        tracks=("assistant_text", "assistant_image"),
        targets=(b"Red square", image),
        specs=(ActionSpec(max_tokens=32), MediaOutputSpec("image", height=16, width=16)),
        effort=1,
        generator=torch.Generator().manual_seed(4),
    )
    assert training.loss.isfinite()
    assert set(training.components) == {"prediction", "reconstruction", "facet_kl", "routing"}
    training.loss.backward()
    assert core.media_stems.vision.projection.weight.grad is not None
    assert model.media_realizers.visual_filter[0].weight.grad is not None
    assert core.central.layers[0].ffn.gate_up.weight.grad is not None


def test_text_and_media_realize_raw_without_extra_meridian_c_pass():
    core, model, service = _system()
    initial = service.tracks.observe_text_group(
        service.tracks.initial_state(1), (b"preceding context",), tracks=("prompt",)
    )
    calls = []
    hook = core.central.register_forward_hook(
        lambda module, inputs, output: calls.append(inputs[0])
    )
    try:
        result = service.supervised_group(
            initial,
            tracks=("reply", "picture", "sound"),
            targets=(b"hello", torch.rand(1, 1, 3, 16, 16), torch.randn(1, 128) * 0.1),
            specs=(
                ActionSpec(max_tokens=8),
                MediaOutputSpec("image", height=16, width=16),
                MediaOutputSpec("audio", sample_count=128),
            ),
            effort=1,
            generator=torch.Generator().manual_seed(17),
        )
    finally:
        hook.remove()
    assert len(calls) == 1  # Teacher-forced next state only, not R or reconstruction.
    torch.testing.assert_close(calls[0], result.state.raw.flatten(1, 2), rtol=0, atol=0)
    result.loss.backward()
    assert core.media_stems.audio.projection.weight.grad is not None
    assert core.media_stems.vision.projection.weight.grad is not None
    assert model.media_realizers.audio_filter[0].weight.grad is not None
    assert model.media_realizers.visual_filter[0].weight.grad is not None


def test_media_realizers_are_direct_raw_facet_functions():
    core, model, _ = _system()
    selected = core.observe(torch.tensor([[65, 66]]), core.initial_state(1))
    raw_context = core.prepare_realization(selected)
    variants = (
        MediaOutputSpec("audio", sample_count=128),
        MediaOutputSpec("image", height=16, width=16),
        MediaOutputSpec("video", frames=2, height=16, width=16),
    )
    def forbid(*args, **kwargs):
        raise AssertionError("R must not execute the central core")
    original = core.central.forward
    core.central.forward = forbid
    try:
        for spec in variants:
            output = model.media_realizers.realize(raw_context, spec)
            assert output.isfinite().all()
            replay = model.media_realizers.realize(
                core.prepare_realization(selected.facets.clone()), spec
            )
            torch.testing.assert_close(output, replay, rtol=0, atol=0)
    finally:
        core.central.forward = original


def test_single_control_source_supports_structured_media_references():
    reference = ControlReference(
        asset_id="speaker-consented-1", sha256="a" * 64, media_type="audio/wav",
        purpose="voice", origin="external_evidence",
    )
    control = ControlBundle(
        instructions=b"Say hello",
        settings_json='{"language":"en","style":"calm"}',
        references=(reference,),
    )
    payload = control.payload(track_override={"track": "speech"})
    assert b"Say hello" in payload
    assert b"<settings>" in payload
    assert b"<assets>" in payload
    assert b"<track>" in payload
    assert payload == build_control_payload(
        seed=b"Say hello",
        settings_json='{"style":"calm","language":"en"}',
        references=(reference,),
        track_override={"track":"speech"},
    )
    with pytest.raises(ValueError, match="provenance"):
        ControlReference("x", "a" * 64, "audio/wav", "voice", "future_target")


def test_meridians_separate_causal_and_media_time():
    span = MediaSpan("capture", 100, 400)
    recorded = TrackEvent("a", "mic", "audio", 1, "user", "pair", span)
    transcript = TrackEvent("b", "text", "text", 2, "assistant", "pair", span)
    assert Meridian(1, (recorded,)).meridian_id == 1
    assert Meridian(2, (transcript,)).events[0].media_span == span
    with pytest.raises(ValueError, match="at most one"):
        Meridian(1, (recorded, TrackEvent("c", "mic", "audio", 1, "user")))


def test_grouped_bootstrap_kl_is_normalized_once_per_latent_coordinate():
    core, model, service = _system()
    state = service.tracks.initial_state(1)
    target = torch.rand(1, 1, 3, 16, 16)
    spec = MediaOutputSpec("image", height=16, width=16)
    result = service.supervised_group(
        state, tracks=("picture",), targets=(target,), specs=(spec,),
        effort=1, generator=torch.Generator().manual_seed(101),
    )
    with torch.no_grad():
        mean = service.tracks.predict_group(
            state, tracks=("picture",), modalities=("image",), effort=1,
        )
        encoded = core.encode_media("image", target)[:, None]
        posterior = DiagonalGaussian(
            encoded.float(),
            torch.full_like(encoded.float(), math.log(core.config.codec_noise_std)),
        )
        expected = posterior.kl(model.facet_distribution(mean)).sum() / encoded.numel()
    torch.testing.assert_close(result.components["facet_kl"], expected, rtol=1e-5, atol=1e-5)
    assert float(expected) > 0


def test_precomputed_media_features_reuse_exact_encoder_reduction():
    core, _, _ = _system()
    signal = torch.randn(1, 2400) * 0.1
    features = core.media_stems.audio(signal)
    direct = core.media_stems.encode_ordered_features(core, features)
    through_stem = core.encode_media("audio", signal)
    torch.testing.assert_close(direct, through_stem, atol=0, rtol=0)
    with pytest.raises(ValueError, match="source_dim"):
        core.media_stems.encode_ordered_features(core, features[..., :-1])
    with pytest.raises(ValueError, match="mask"):
        core.media_stems.encode_ordered_features(
            core, features, torch.zeros(features.shape[:2], dtype=torch.bool)
        )
