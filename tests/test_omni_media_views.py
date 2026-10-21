"""Media clock and optional-view dropout contracts."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from megatoken.data.media_assets import MediaRecord, TensorAssetStore
from megatoken.model.config import ModelConfig
from megatoken.model.media import MediaOutputSpec, MediaStems
from megatoken.training.meridians import MediaPlan


def test_raw_24k_source_keeps_actual_rate_until_e_encoding(tmp_path):
    store = TensorAssetStore(tmp_path)
    waveform = store.put(
        np.zeros(4800, dtype=np.float32),
        modality="audio",
        sample_rate=24000,
    )
    latents = store.put(
        np.zeros((3, 32), dtype=np.float32),
        modality="audio",
        kind="codec_latents",
        extractor="pocket-mimi-pinned-v1",
        sample_rate=24000,
    )
    spec = MediaOutputSpec(
        "audio",
        renderer="pocket",
        sample_rate=24000,
        sample_count=5760,
        latent_shape=(3, 32),
        shape_provenance="teacher_only",
    )
    plan = MediaPlan(
        MediaRecord("audio", waveform, latents),
        spec,
        "bootstrap",
        32,
        8,
    )
    target = plan.resolve(store, torch.device("cpu"))
    assert target.source_sample_rate == 24000
    cfg = ModelConfig.tiny()
    stem = MediaStems(cfg)
    # The legacy waveform E requires 16 kHz. It must never silently
    # reinterpret a 24 kHz source as though it had a 16 kHz clock.
    with pytest.raises(ValueError, match="16 kHz"):
        stem.audio(target.source, sample_rate=target.source_sample_rate)


def test_input_view_dropout_maintains_at_least_one_available_source(monkeypatch):
    cfg = ModelConfig.tiny(
        media_feature_views={
            "whistle-pinned-embed-v1": 5,
            "pocket-mimi-acoustic-v1": 4,
        },
        media_view_dropout=0.999,
    )
    stems = MediaStems(cfg)
    core = SimpleNamespace(
        config=cfg,
        initial_facets=torch.zeros(1, cfg.n_facets, cfg.facet_dim),
    )
    monkeypatch.setattr(
        MediaStems,
        "encode_ordered_features",
        staticmethod(lambda core, features, mask=None: mask),
    )
    views = (
        ("whistle-pinned-embed-v1", torch.ones(1, 3, 5), torch.ones(1, 3, dtype=torch.bool)),
        ("pocket-mimi-acoustic-v1", torch.ones(1, 2, 4), torch.ones(1, 2, dtype=torch.bool)),
    )
    stems.train()
    torch.manual_seed(17)
    training_mask = stems.encode_views(core, "audio", views)
    assert training_mask.shape == (1, 5)
    assert 0 < int(training_mask.sum()) <= 5
    stems.eval()
    inference_mask = stems.encode_views(core, "audio", views)
    assert inference_mask.all()
    with pytest.raises(ValueError, match="dropout"):
        ModelConfig.tiny(media_view_dropout=1.0)
