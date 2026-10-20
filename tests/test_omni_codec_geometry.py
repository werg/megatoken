"""Codec-target geometry is exact even without storing huge raw video frames."""

from types import SimpleNamespace

import numpy as np
import pytest

from megatoken.data.media_assets import (
    MEDIA_RECORD_TYPE,
    MediaRecord,
    OutputGeometry,
    TensorAssetStore,
)
from megatoken.training.meridians import _media_plan


def test_large_cached_video_geometry_round_trip_and_training_spec(tmp_path):
    store = TensorAssetStore(tmp_path)
    source = store.put(
        np.ones((16, 24), dtype=np.float32),
        modality="video",
        kind="features",
        extractor="minicpm-pinned-16x",
        fps_num=24,
    )
    codec = store.put(
        np.zeros((5, 32, 22, 40), dtype=np.float32),
        modality="video",
        kind="codec_latents",
        extractor="ltx2-v2-checkpoint",
        fps_num=24,
        valid_count=5,
    )
    original = MediaRecord(
        "video",
        source,
        codec,
        output_geometry=OutputGeometry(
            "video",
            frames=81,
            height=720,
            width=1280,
            fps_num=24,
            provenance="teacher_only",
        ),
    )
    raw = original.to_bytes()
    parsed = MediaRecord.from_bytes(raw)
    assert parsed.to_bytes() == raw
    assert parsed.output_geometry.frames == 81
    assert parsed.output_geometry.height == 720
    model = SimpleNamespace(
        config=SimpleNamespace(
            vision_feature_id=source.extractor,
            vision_feature_dim=source.shape[-1],
            n_facets=32,
            facet_dim=512,
        )
    )
    entry = SimpleNamespace(payload=raw, record=SimpleNamespace(media_type=MEDIA_RECORD_TYPE))
    plan = _media_plan(model, entry, "video", SimpleNamespace(representation_training="recurrent"))
    assert plan.spec.frames == 81
    assert (plan.spec.height, plan.spec.width) == (720, 1280)
    assert plan.spec.latent_shape == (32, 5, 22, 40)
    assert plan.spec.shape_provenance == "teacher_only"


def test_codec_features_require_actual_display_geometry(tmp_path):
    store = TensorAssetStore(tmp_path)
    source = store.put(
        np.ones((16, 24), dtype=np.float32),
        modality="video",
        kind="features",
        extractor="minicpm-pinned-16x",
        fps_num=24,
    )
    codec = store.put(
        np.zeros((3, 4, 2, 2), dtype=np.float32),
        modality="video",
        kind="codec_latents",
        extractor="ltx2-native-codec",
        fps_num=24,
    )
    model = SimpleNamespace(
        config=SimpleNamespace(
            vision_feature_id=source.extractor,
            vision_feature_dim=source.shape[-1],
            n_facets=32,
            facet_dim=512,
        )
    )
    payload = MediaRecord("video", source, codec).to_bytes()
    entry = SimpleNamespace(payload=payload, record=SimpleNamespace(media_type=MEDIA_RECORD_TYPE))
    with pytest.raises(ValueError, match="displayed output geometry"):
        _media_plan(
            model,
            entry,
            "video",
            SimpleNamespace(representation_training="bootstrap"),
        )


def test_backward_compatible_media_v2_records_do_not_gain_undeclared_geometry(tmp_path):
    store = TensorAssetStore(tmp_path)
    source = store.put(np.ones((3, 8, 8), dtype=np.float32), modality="image")
    reference = MediaRecord("image", source)
    encoded = reference.to_bytes()
    assert b"output_geometry" not in encoded
    assert MediaRecord.from_bytes(encoded).to_bytes() == encoded
