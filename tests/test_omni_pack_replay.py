"""Raw group tapes bind to exact, verified renderer/codec weight identities."""
from types import SimpleNamespace

import numpy as np
import pytest
from torch import nn

from megatoken.model.media import MediaOutputSpec
from megatoken.model.multitrack import GROUP_TAPE_VERSION, GroupActionTape, MultiTrackGenerator


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


def test_renderer_pack_identity_survives_tape_json():
    original = _tape("a" * 64)
    replayed = GroupActionTape.from_json(original.to_json())
    assert replayed.pack_fingerprints == original.pack_fingerprints
    assert replayed.raw_f32 == original.raw_f32
    assert replayed.renderer_seeds == original.renderer_seeds
    with pytest.raises(ValueError, match="pack identities"):
        GroupActionTape(
            **{**original.__dict__, "pack_fingerprints": ("image", "extra")}
        )


def test_realizer_requires_pinned_checkpoint_identity():
    donor = nn.Linear(1, 1)
    residents = nn.ModuleDict({"sana": donor})
    generator = SimpleNamespace(media_realizers=SimpleNamespace(pretrained=residents))
    spec = MediaOutputSpec("image", renderer="sana", height=64, width=64)
    with pytest.raises(ValueError, match="verified lockfile"):
        MultiTrackGenerator._pack_ids(generator, (spec,))
    donor.checkpoint_identity = "c" * 64
    assert MultiTrackGenerator._pack_ids(generator, (spec,)) == ("c" * 64,)
