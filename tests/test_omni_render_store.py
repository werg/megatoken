"""Large media group tapes store immutable digests, not raw video byte arrays."""
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from megatoken.model.multitrack import GROUP_TAPE_VERSION, GroupActionTape
from megatoken.omni.render_store import RenderReceipt, RenderedOutputStore
from megatoken.omni.session import OmniSession, PublishedGroup


def test_content_addressed_media_is_complete_and_tamper_evident(tmp_path):
    store = RenderedOutputStore(tmp_path)
    data = b"synthetic-video-f32-bytes" * 1024
    receipt = store.put(data)
    assert store.put(data) == receipt
    assert store.get(receipt) == data
    artifact = tmp_path / "rendered" / (receipt.sha256 + ".bin")
    artifact.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="altered"):
        store.get(receipt)


def test_compact_group_tape_omits_media_payload_and_retains_exact_hash(tmp_path):
    store = RenderedOutputStore(tmp_path)
    media = b"pixel-array" * 4096
    receipt = store.put(media)
    tape = GroupActionTape(
        version=GROUP_TAPE_VERSION, meridian_id=0,
        tracks=("clip",), modalities=("video",), roles=(2,),
        controls=(b"generate video",),
        spec_fingerprints=("requested-720p-video",),
        codec_fingerprint="text-codec",
        facet_shape=(1, 32, 8),
        raw_f32=np.zeros((1, 32, 8), dtype="<f4").tobytes(),
        behavior_log_prob=0.0, event_log_probs=(0.0,),
        renderer_seeds=(42,), pack_fingerprints=("pretrained-identity",),
        payloads=(b"",), emitted_ids=((),), valid=(True,),
        media_hashes=(receipt.sha256,), media_sizes=(receipt.byte_count,),
    )
    serialized = tape.to_json()
    assert len(serialized) < len(media)
    restored = GroupActionTape.from_json(serialized)
    assert restored == tape

    class Tracks:
        def _validate(self, state):
            assert state.batch_size == 1

    session = OmniSession(
        SimpleNamespace(tracks=Tracks()),
        SimpleNamespace(batch_size=1), output_store=store,
    )
    session.published[0] = PublishedGroup(
        restored, specs=(), delivery_cursors={"clip": 0},
    )
    session.deliver(0, "clip", len(media))
    assert session.delivered_bytes(0, "clip", start=17, count=200) == media[17:217]
    with pytest.raises(ValueError, match="already delivered"):
        session.delivered_bytes(0, "clip", count=len(media) + 1)
