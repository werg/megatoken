"""Endpoint targets use last *displayed* frame, never codec padding."""
import numpy as np
import pytest

from megatoken.data.media_assets import TensorAssetStore
from megatoken.omni.endpoints import derive_displayed_endpoints


def test_visible_video_endpoints_follow_sampled_indices_not_padding(tmp_path):
    store = TensorAssetStore(tmp_path)
    source = np.zeros((5, 3, 8, 8), dtype=np.float32)
    for index in range(5):
        source[index] += index
    asset = store.put(
        source, modality="video", fps_num=24000, fps_den=1001,
        valid_count=3,
        normalization="rgb01-fp32", source_family="scene-1",
        artifact_version="v0",
    )
    first, last, mapping = derive_displayed_endpoints(
        store, asset, sampled_source_indices=(11, 13, 18),
    )
    assert first.modality == last.modality == "image"
    assert np.array_equal(store.load(first), source[0])
    assert np.array_equal(store.load(last), source[2])
    assert mapping.first_display_index == 0
    assert mapping.last_display_index == 2
    assert mapping.sampled_source_indices == (11, 13, 18)
    assert mapping.video_sha256 == asset.sha256
    assert mapping.fingerprint == mapping.fingerprint
    with pytest.raises(ValueError, match="valid visible interval"):
        derive_displayed_endpoints(
            store, asset, sampled_source_indices=(11, 13, 18, 21),
        )


def test_endpoint_preparation_rejects_noncausal_source_frame_order(tmp_path):
    store = TensorAssetStore(tmp_path)
    asset = store.put(
        np.zeros((3, 3, 8, 8), dtype=np.float32), modality="video",
        fps_num=30,
    )
    with pytest.raises(ValueError, match="source display order"):
        derive_displayed_endpoints(store, asset, sampled_source_indices=(3, 9, 2))
