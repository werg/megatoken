"""A cached U tensor is unusable with a different pretrained codec revision."""
from types import SimpleNamespace

import numpy as np
import pytest

from megatoken.data.media_assets import (
    MEDIA_RECORD_TYPE, MediaRecord, OutputGeometry, TensorAssetStore,
)
from megatoken.training.meridians import _media_plan


def test_training_refuses_mismatched_codec_coordinates(tmp_path):
    store = TensorAssetStore(tmp_path)
    source = store.put(
        np.zeros((3, 8, 8), dtype=np.float32), modality="image",
    )
    target = store.put(
        np.zeros((4, 1, 1), dtype=np.float32),
        modality="image", kind="codec_latents",
        extractor="sana-600m-checkpoint",
    )
    expected = "a" * 64
    pack = SimpleNamespace(checkpoint_identity=expected)
    model = SimpleNamespace(
        config=SimpleNamespace(n_facets=32, facet_dim=512),
        media_realizers=SimpleNamespace(pretrained={"sana": pack}),
    )
    config = SimpleNamespace(representation_training="bootstrap")

    def run(identity):
        reference = MediaRecord(
            "image", source, target,
            output_geometry=OutputGeometry(
                "image", height=8, width=8, provenance="teacher_only",
            ),
            codec_sha256=identity,
        )
        roundtrip = MediaRecord.from_bytes(reference.to_bytes())
        assert roundtrip.codec_sha256 == identity
        return _media_plan(
            model,
            SimpleNamespace(
                payload=roundtrip.to_bytes(),
                record=SimpleNamespace(media_type=MEDIA_RECORD_TYPE),
            ),
            "image", config,
        )

    plan = run(expected)
    assert plan.spec.renderer == "sana"
    with pytest.raises(ValueError, match="codec identity"):
        run("b" * 64)
    with pytest.raises(ValueError, match="codec identity"):
        run("")
