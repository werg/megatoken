"""Pinned-feature media E, resident realizers, causal controls and meridian replay."""

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
from megatoken.model.media import MediaOutputSpec, MediaTrainTarget
from megatoken.model.multitrack import (
    GroupActionTape,
    MultiTrackGenerator,
    ObservedEvent,
)
from megatoken.model.state import RealizationContext
from megatoken.teacher import PINNED_TEACHER
from megatoken.training.config import DEFAULT_LOSS_WEIGHTS
from megatoken.training.losses import LossLedger
from tests.media_support import LATENT, features, install_fake_realizers, observe_text

AUDIO = MediaOutputSpec("audio", sample_count=256, sample_rate=24000)
IMAGE = MediaOutputSpec("image", height=16, width=16)
VIDEO = MediaOutputSpec("video", frames=2, height=16, width=16, fps_num=8)


def _system():
    torch.manual_seed(91)
    core = FacetedCore(ModelConfig.tiny(max_history_chunks=8))
    model = install_fake_realizers(FacetedGenerator(core, ByteCodec.byte_fallback()))
    service = MultiTrackGenerator(model)
    return core, model, service


def _media_target(model, modality, spec, *, seed=0, mode="bootstrap"):
    """A resolved media event as the data pipeline hands it to the grouped loss."""
    generator = torch.Generator().manual_seed(seed)
    return MediaTrainTarget(
        modality,
        spec,
        features(seed=seed),
        PINNED_TEACHER,
        torch.randn(1, *LATENT[modality], generator=generator),
        None,
        model.media_realizers.resident(modality).identity,
        mode,
    )


def _text_target(model, payload, mode="bootstrap"):
    return model.prepare_target(target_bytes=payload, representation_mode=mode)


def test_heterogeneous_ingestion_one_meridian():
    core, _, service = _system()
    state = service.tracks.initial_state(1)
    observations = [
        ObservedEvent("user", "text", b"Describe the clip"),
        ObservedEvent("mic", "audio", features(20, seed=1), extractor=PINNED_TEACHER),
        ObservedEvent("camera", "video", features(33, seed=2), extractor=PINNED_TEACHER),
        ObservedEvent("photo", "image", features(7, seed=3), extractor=PINNED_TEACHER),
    ]
    next_state = service.tracks.observe_group(state, observations)
    assert next_state.raw.shape == (1, 4, core.config.n_facets, core.config.facet_dim)
    assert next_state.history.slots == 4 * core.config.n_facets
    assert next_state.meridian_sizes == (4,)
    assert next_state.frontier_tracks == ("user", "mic", "camera", "photo")


def test_media_events_are_only_pinned_teacher_features():
    _, _, service = _system()
    state = service.tracks.initial_state(1)
    # Raw samples, pixels and unlabelled tensors are not observations.
    with pytest.raises(TypeError, match="pinned"):
        ObservedEvent("mic", "audio", features(), extractor="")
    with pytest.raises(TypeError, match="pinned"):
        ObservedEvent("mic", "audio", b"pcm", extractor=PINNED_TEACHER)
    with pytest.raises(ValueError, match="ordered"):
        ObservedEvent("mic", "audio", torch.zeros(1, 2400), extractor=PINNED_TEACHER)
    with pytest.raises(ValueError, match="ordered"):
        ObservedEvent("cam", "video", torch.rand(1, 2, 3, 16, 16), extractor=PINNED_TEACHER)
    with pytest.raises(TypeError, match="exact bytes"):
        ObservedEvent("user", "text", b"hi", extractor=PINNED_TEACHER)
    with pytest.raises(ValueError, match="Unsupported"):
        ObservedEvent("doc", "document", features(), extractor=PINNED_TEACHER)
    # A different teacher or a wrong feature width is refused at the model boundary.
    with pytest.raises(ValueError, match="different teacher"):
        service.tracks.observe_group(
            state, [ObservedEvent("mic", "audio", features(), extractor="another-teacher")]
        )
    with pytest.raises(ValueError, match="512"):
        service.tracks.observe_group(
            state,
            [ObservedEvent("mic", "audio", features()[..., :-1], extractor=PINNED_TEACHER)],
        )


def _backward(result):
    """Backward the training objective exactly as the engine does: global-count weighted terms."""
    counts = {name: count for name, (_, count) in result.losses.items()}
    ledger = LossLedger(counts, counts, dict(DEFAULT_LOSS_WEIGHTS))
    ledger.backward_loss(result.losses).backward()


def test_modality_reaches_the_model_through_embedding_not_a_separate_stem():
    core, _, service = _system()
    with torch.no_grad():  # the modality embedding starts at zero; give it a trained value
        core.central.modality_embedding.weight.normal_(generator=torch.Generator().manual_seed(1))
    event = features(10, seed=4)
    state = service.tracks.initial_state(1)
    by_modality = {
        modality: service.tracks.observe_group(
            state, [ObservedEvent("t", modality, event, extractor=PINNED_TEACHER)]
        )
        for modality in ("audio", "image", "video")
    }
    # The same features encode to the same raw facets whatever the modality;
    # C and P are the modality-aware stages.
    torch.testing.assert_close(by_modality["audio"].raw, by_modality["image"].raw, atol=0, rtol=0)
    torch.testing.assert_close(by_modality["audio"].raw, by_modality["video"].raw, atol=0, rtol=0)
    assert not torch.allclose(by_modality["audio"].final, by_modality["image"].final)


def test_mixed_media_generation_and_replay():
    core, model, service = _system()
    state = observe_text(
        service.tracks,
        service.tracks.initial_state(1),
        (b"Generate a picture and sound",),
        tracks=("prompt",),
    )
    specifications = (ActionSpec(max_tokens=8), AUDIO, IMAGE, VIDEO)
    result = service.generate_group(
        state,
        tracks=("reply", "voice", "still", "clip"),
        specs=specifications,
        effort=1,
        generator=torch.Generator().manual_seed(13),
    )
    assert result.tape.modalities == ("text", "audio", "image", "video")
    assert result.media_outputs[0] is None
    assert result.media_outputs[1].shape == (1, 256)
    assert result.media_outputs[2].shape == (1, 1, 3, 16, 16)
    assert result.media_outputs[3].shape == (1, 2, 3, 16, 16)
    packs = result.tape.pack_fingerprints
    assert packs[0] == "text"
    assert packs[1:] == tuple(
        model.media_realizers.resident(m).identity for m in ("audio", "image", "video")
    )
    tape = GroupActionTape.from_json(result.tape.to_json())
    replay = service.replay_group(state, tape, specs=specifications, effort=1)
    assert replay.tape.payloads == result.tape.payloads
    torch.testing.assert_close(replay.state.final, result.state.final)
    for original, again in zip(result.media_outputs[1:], replay.media_outputs[1:], strict=True):
        torch.testing.assert_close(original, again, rtol=0, atol=0)
    with pytest.raises(ValueError, match="specifications"):
        service.replay_group(
            state,
            tape,
            specs=(
                ActionSpec(max_tokens=8),
                MediaOutputSpec("audio", sample_count=128, sample_rate=24000),
                *specifications[2:],
            ),
            effort=1,
        )


def test_grouped_multimodal_bootstrap_has_trainable_gradients():
    core, model, service = _system()
    context = service.tracks.initial_state(1)
    training = service.supervised_meridian(
        context,
        tracks=("assistant_text", "assistant_image"),
        plans=(_text_target(model, b"Red square"), _media_target(model, "image", IMAGE)),
        effort=1,
        generator=torch.Generator().manual_seed(4),
    )
    assert all(value.isfinite() for value, _ in training.losses.values())
    assert training.losses["visual_prediction"][1] == 1
    assert training.losses["visual_reconstruction"][1] == 1
    _backward(training)
    realizer = model.media_realizers.resident("image")
    # The teacher-feature projection, the realizer condition and the shared core all learn.
    assert core.media_encoder.projection.router.weight.grad.abs().sum() > 0
    assert core.media_encoder.projection._bank.weight.grad.abs().sum() > 0
    assert realizer.condition.project.router.weight.grad.abs().sum() > 0
    assert realizer.condition.project._bank.weight.grad.abs().sum() > 0
    assert core.central.layers[0].ffn._bank.gate_up.grad is not None
    # The frozen donor never receives gradients.
    assert all(p.grad is None and not p.requires_grad for p in realizer.donor.parameters())


def test_text_and_media_realize_raw_without_extra_meridian_c_pass():
    core, model, service = _system()
    initial = observe_text(
        service.tracks, service.tracks.initial_state(1), (b"preceding context",), tracks=("prompt",)
    )
    calls = []
    hook = core.central.register_forward_hook(
        lambda module, inputs, output: calls.append(inputs[0])
    )
    try:
        result = service.supervised_meridian(
            initial,
            tracks=("reply", "picture", "sound"),
            plans=(
                _text_target(model, b"hello"),
                _media_target(model, "image", IMAGE, seed=1),
                _media_target(model, "audio", AUDIO, seed=2),
            ),
            effort=1,
            generator=torch.Generator().manual_seed(17),
        )
    finally:
        hook.remove()
    assert len(calls) == 1  # Teacher-forced next state only, not R or reconstruction.
    torch.testing.assert_close(calls[0], result.state.raw.flatten(1, 2), rtol=0, atol=0)
    _backward(result)
    assert core.media_encoder.projection.router.weight.grad.abs().sum() > 0
    assert core.media_encoder.projection._bank.weight.grad.abs().sum() > 0
    for modality in ("audio", "image"):
        condition = model.media_realizers.resident(modality).condition
        assert condition.project.router.weight.grad.abs().sum() > 0
        assert condition.project._bank.weight.grad.abs().sum() > 0


def test_media_realizers_are_direct_raw_facet_functions():
    core, model, _ = _system()
    selected = core.observe(torch.tensor([[65, 66]]), core.initial_state(1))
    raw_context = core.prepare_realization(selected)
    variants = (AUDIO, IMAGE, VIDEO)

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
            # Every slot reaches the realizer: perturbing one facet changes the media.
            perturbed = selected.facets.clone()
            perturbed[:, -1] += torch.randn(
                perturbed[:, -1].shape, generator=torch.Generator().manual_seed(3)
            )
            changed = model.media_realizers.realize(core.prepare_realization(perturbed), spec)
            assert not torch.equal(output, changed)
    finally:
        core.central.forward = original
    # A realizer demands one complete raw facet block, not a pooled summary.
    incomplete = RealizationContext(selected.facets[:, :-1])
    with pytest.raises(ValueError, match="complete raw facet block"):
        model.media_realizers.realize(incomplete, IMAGE)


def test_single_control_source_supports_structured_media_references():
    reference = ControlReference(
        asset_id="speaker-consented-1",
        sha256="a" * 64,
        media_type="audio/wav",
        purpose="voice",
        origin="external_evidence",
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
        track_override={"track": "speech"},
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
    target = _media_target(model, "image", IMAGE, seed=9)
    result = service.supervised_meridian(
        state,
        tracks=("picture",),
        plans=(target,),
        effort=1,
        generator=torch.Generator().manual_seed(101),
    )
    with torch.no_grad():
        mean = service.tracks.predict_group(
            state,
            tracks=("picture",),
            modalities=("image",),
            effort=1,
        )
        encoded = core.encode_media(target.source, extractor=target.extractor)[:, None]
        posterior = DiagonalGaussian(
            encoded.float(),
            torch.full_like(encoded.float(), math.log(core.config.codec_noise_std)),
        )
        expected = posterior.kl(model.facet_distribution(mean)).sum() / encoded.numel()
    total, count = result.losses["facet_prediction"]
    torch.testing.assert_close(total / count, expected, rtol=1e-5, atol=1e-5)
    assert float(expected) > 0


def test_media_features_read_every_teacher_token_at_once():
    core, _, _ = _system()
    n = core.config.n_facets
    # 32 learned queries read every token: no pair reductions, any length.
    reductions = []
    hook = core.frontend.reduction.register_forward_hook(
        lambda module, inputs, output: reductions.append(inputs[0].shape[1])
    )
    try:
        short = core.encode_media(features(3, seed=1), extractor=PINNED_TEACHER)
        long = core.encode_media(features(8 * n, seed=1), extractor=PINNED_TEACHER)
    finally:
        hook.remove()
    assert reductions == []
    assert short.shape == long.shape == (1, n, core.config.facet_dim)
    # The module is exactly the public encode_media path.
    direct = core.media_encoder(core, features(8 * n, seed=1), extractor=PINNED_TEACHER)
    torch.testing.assert_close(direct, long, atol=0, rtol=0)


def test_media_feature_mask_and_pin_validation():
    core, _, _ = _system()
    source = features(10, seed=6)
    valid = torch.zeros(1, 10, dtype=torch.bool)
    valid[:, :6] = True
    base = core.encode_media(source, valid, extractor=PINNED_TEACHER)
    changed = source.clone()
    changed[:, 6:] += 100.0
    torch.testing.assert_close(
        base, core.encode_media(changed, valid, extractor=PINNED_TEACHER), atol=0, rtol=0
    )
    with pytest.raises(ValueError, match="mask"):
        core.encode_media(source, torch.zeros_like(valid), extractor=PINNED_TEACHER)
    with pytest.raises(ValueError, match="mask"):
        core.encode_media(source, valid[:, :5], extractor=PINNED_TEACHER)
    with pytest.raises(ValueError, match="different teacher"):
        core.encode_media(source, extractor="unpinned")
    with pytest.raises(ValueError, match="512"):
        core.encode_media(source[..., :-1], extractor=PINNED_TEACHER)
    assert core.media_encoder.extractor == PINNED_TEACHER == core.config.media_extractor
