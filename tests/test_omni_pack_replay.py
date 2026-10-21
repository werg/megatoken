"""Group tapes bind to exact realizer weight identities and the realized bytes."""

import dataclasses
import json

import numpy as np
import pytest
import torch

from megatoken.generation.codec import ByteCodec
from megatoken.generation.policy import FacetedGenerator
from megatoken.generation.structure import ActionSpec
from megatoken.model import FacetedCore, ModelConfig
from megatoken.model.media import MediaOutputSpec
from megatoken.model.multitrack import GROUP_TAPE_VERSION, GroupActionTape, MultiTrackGenerator
from tests.media_support import FakeRealizer, install_fake_realizers


def _tape(pack: str):
    return GroupActionTape(
        version=GROUP_TAPE_VERSION,
        meridian_id=3,
        tracks=("image",),
        roles=(2,),
        controls=(b"draw",),
        modalities=("image",),
        spec_fingerprints=("sha-output-spec",),
        codec_fingerprint="text-codec",
        facet_shape=(1, 32, 8),
        raw_f32=np.zeros((1, 32, 8), dtype="<f4").tobytes(),
        behavior_log_prob=0.0,
        event_log_probs=(0.0,),
        renderer_seeds=(37,),
        pack_fingerprints=(pack,),
        payloads=(b"rendered",),
        emitted_ids=((),),
        valid=(True,),
    )


def _service(seed: int = 5):
    torch.manual_seed(seed)
    core = FacetedCore(ModelConfig.tiny())
    model = install_fake_realizers(FacetedGenerator(core, ByteCodec.byte_fallback()))
    return model, MultiTrackGenerator(model)


def test_realizer_pack_identity_survives_tape_json():
    original = _tape("a" * 64)
    replayed = GroupActionTape.from_json(original.to_json())
    assert replayed.pack_fingerprints == original.pack_fingerprints
    assert replayed.raw_f32 == original.raw_f32
    assert replayed.renderer_seeds == original.renderer_seeds
    with pytest.raises(ValueError, match="exact event identities"):
        dataclasses.replace(original, pack_fingerprints=("image", "extra"))
    with pytest.raises(ValueError, match="realizer identities"):
        dataclasses.replace(original, pack_fingerprints=("",))
    with pytest.raises(ValueError, match="Renderer noise seeds"):
        dataclasses.replace(original, renderer_seeds=(-1,))


def test_realizer_requires_pinned_checkpoint_identity():
    model, _ = _service()
    spec = MediaOutputSpec("image", height=16, width=16)
    resident = model.media_realizers.resident("image")
    assert MultiTrackGenerator._pack_ids(model, (spec,)) == (resident.identity,)
    assert resident.identity.startswith("sana:")
    resident.checkpoint_identity = ""
    with pytest.raises(ValueError, match="locked pack manifest"):
        MultiTrackGenerator._pack_ids(model, (spec,))
    # Text events are bound to no realizer; an uninstalled modality is an error.
    assert MultiTrackGenerator._pack_ids(model, (ActionSpec(max_tokens=4),)) == ("text",)
    model.media_realizers.packs.pop("sana")
    with pytest.raises(ValueError, match="not installed"):
        MultiTrackGenerator._pack_ids(model, (spec,))


def test_tape_payload_is_a_digest_bound_to_the_realizer_and_output():
    model, service = _service()
    state = service.tracks.observe_text_group(
        service.tracks.initial_state(1), (b"draw a square",), tracks=("prompt",)
    )
    spec = MediaOutputSpec("image", height=16, width=16)
    result = service.generate_group(
        state,
        tracks=("picture",),
        specs=(spec,),
        effort=1,
        generator=torch.Generator().manual_seed(2),
    )
    digest = json.loads(result.tape.payloads[0])
    assert digest["kind"] == "media-sha256-v1"
    assert digest["modality"] == "image"
    assert digest["realizer"] == model.media_realizers.resident("image").identity
    assert digest["shape"] == [1, 3, 16, 16] and digest["dtype"] == "f32le"
    assert result.tape.pack_fingerprints == (digest["realizer"],)
    assert len(result.tape.payloads[0]) < 512  # never a multi-megabyte tape field


def test_replay_rejects_a_different_realizer_pack():
    _, service = _service()
    state = service.tracks.initial_state(1)
    spec = MediaOutputSpec("audio", sample_count=128, sample_rate=24000)
    result = service.generate_group(
        state,
        tracks=("voice",),
        specs=(spec,),
        effort=1,
        generator=torch.Generator().manual_seed(8),
    )
    replay = service.replay_group(state, result.tape, specs=(spec,), effort=1)
    assert replay.tape.payloads == result.tape.payloads

    # Same architecture, different frozen realizer weights/identity.
    other = service.generator.media_realizers.resident("audio")
    other.checkpoint_identity = "f" * 64
    with pytest.raises(ValueError, match="realizer identity mismatch"):
        service.replay_group(state, result.tape, specs=(spec,), effort=1)
    other.checkpoint_identity = FakeRealizer("audio", 1, 1).checkpoint_identity

    # Same identity but a different realization (weights drifted): digest mismatch.
    with torch.no_grad():
        other.donor.weight.add_(1.0)
    with pytest.raises(ValueError, match="realization replay mismatch"):
        service.replay_group(state, result.tape, specs=(spec,), effort=1)
