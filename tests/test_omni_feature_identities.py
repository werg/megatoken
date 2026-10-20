"""Configured multi-view E identities work without legacy bridge singleton IDs."""
from types import SimpleNamespace

import numpy as np
import pytest

from megatoken.data.media_assets import MEDIA_RECORD_TYPE, MediaRecord, TensorAssetStore
from megatoken.training.meridians import _media_plan, _validate_feature_identity


def test_minicpm_pinned_detail_profile_is_valid_first_input_view(tmp_path):
    store = TensorAssetStore(tmp_path)
    feature = store.put(
        np.ones((9, 12), dtype=np.float32), modality="image",
        kind="features", extractor="minicpm-v47-merger-detail-4x",
        valid_count=7,
    )
    target = store.put(
        np.zeros((3, 8, 8), dtype=np.float32), modality="image",
    )
    config = SimpleNamespace(
        vision_feature_id="", vision_feature_dim=0,
        audio_feature_id="", audio_feature_dim=0,
        media_feature_views={feature.extractor: feature.shape[-1]},
        n_facets=32, facet_dim=512,
    )
    model = SimpleNamespace(config=config)
    entry = SimpleNamespace(
        payload=MediaRecord("image", feature, target).to_bytes(),
        record=SimpleNamespace(media_type=MEDIA_RECORD_TYPE),
    )
    plan = _media_plan(
        model, entry, "image", SimpleNamespace(representation_training="bootstrap"),
    )
    assert plan.spec.modality == "image"
    assert plan.spec.height == 8
    with pytest.raises(ValueError, match="width"):
        _validate_feature_identity(
            SimpleNamespace(
                **{**config.__dict__, "media_feature_views": {feature.extractor: 13}}
            ), "image", feature,
        )


def test_codec_latent_input_view_uses_native_channel_width(tmp_path):
    store = TensorAssetStore(tmp_path)
    video_latents = store.put(
        np.ones((4, 6, 2, 2), dtype=np.float32),
        modality="video", kind="codec_latents",
        extractor="ltx2-pinned-v1", fps_num=24,
    )
    config = SimpleNamespace(
        vision_feature_id="", vision_feature_dim=0,
        audio_feature_id="", audio_feature_dim=0,
        media_feature_views={"ltx2-pinned-v1": 6},
    )
    _validate_feature_identity(config, "video", video_latents)
    config.media_feature_views["ltx2-pinned-v1"] = 4
    with pytest.raises(ValueError, match="width"):
        _validate_feature_identity(config, "video", video_latents)
