"""Native media E/R adapters, causal controls and heterogeneous meridian replay."""

import pytest
import torch

from megatoken.controls import ControlBundle, ControlReference, build_control_payload
from megatoken.data.schema import MediaSpan, Meridian, TrackEvent
from megatoken.generation.codec import ByteCodec
from megatoken.generation.policy import FacetedGenerator
from megatoken.generation.structure import ActionSpec
from megatoken.model import FacetedCore, ModelConfig
from megatoken.model.media import MediaOutputSpec
from megatoken.model.multitrack import (
    GroupActionTape,
    MultiTrackCore,
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
