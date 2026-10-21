"""Unified hidden-state media contracts and frozen projection warmup (no HF downloads)."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from megatoken.data.media_assets import MediaRecord, TensorAssetStore
from megatoken.generation.codec import ByteCodec
from megatoken.loading import build_model
from megatoken.model.config import ModelConfig
from megatoken.model.media import MediaOutputSpec, MediaTrainTarget
from megatoken.model.multitrack import ObservedEvent
from megatoken.omni.embeddinggemma2_hidden import (
    EmbeddingGemma2HiddenExtractor,
    hidden_identity,
)
from megatoken.training.embeddinggemma2_text_sidecar import TextHiddenSidecar
from megatoken.training.embeddinggemma2_warmup import (
    freeze_other_owners,
    media_warmup_batch,
    reconstruction_loss,
    text_hidden_alignment,
    warmup_parameter_names,
)
from megatoken.training.meridians import MediaPlan
from megatoken.training.optimizer import MasterAdamW

IDENTITY = hidden_identity("d" * 40)


def _build():
    config = ModelConfig.tiny(embeddinggemma2_hidden_dim=512, embeddinggemma2_hidden_id=IDENTITY)
    return build_model(config, ByteCodec.byte_fallback(), precision="fp32")


def _audio_target():
    return MediaTrainTarget(
        "audio",
        MediaOutputSpec("audio", sample_count=128),
        torch.randn(1, 7, 512),
        torch.zeros(1, 128),
        "bootstrap",
        source_kind="features",
        extractor=IDENTITY,
    )


def test_hidden_identity_requires_actual_revision():
    assert IDENTITY.startswith("embeddinggemma2:" + "d" * 40)
    with pytest.raises(ValueError):
        hidden_identity("main")


class _FakeLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding_projection = nn.Linear(512, 768, bias=False)


class _FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = _FakeLanguageModel()
        self.register_buffer("hidden", torch.randn(1, 4, 512))

    def forward(self, **kwargs):
        return SimpleNamespace(
            last_hidden_state=self.language_model.embedding_projection(self.hidden)
        )


class _FakeProcessor:
    def apply_chat_template(self, conversation, **kwargs):
        assert conversation[0]["role"] == "user"
        return {
            "input_ids": torch.ones(1, 4, dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1, 0]], dtype=torch.long),
        }


def test_teacher_taps_unpooled_internal_512_before_768_projection():
    fake = _FakeModel()
    extractor = EmbeddingGemma2HiddenExtractor(fake, _FakeProcessor(), revision="d" * 40)
    result = extractor.encode("video", "sample.mp4")
    assert result.features.shape == (1, 4, 512)
    assert result.mask.tolist() == [[True, True, True, False]]
    torch.testing.assert_close(result.features, fake.hidden)
    assert result.features.shape[-1] != fake.language_model.embedding_projection.out_features
    assert all(not p.requires_grad for p in fake.parameters())
    with pytest.raises(ValueError, match="context"):
        extractor.encode("text", "long text", max_tokens=2)


def test_unified_media_rejects_legacy_raw_and_unpinned_features():
    model = _build()
    core = model.core
    hidden = torch.randn(1, 7, 512)
    assert core.encode_media_features("image", hidden, extractor=IDENTITY).shape == (1, 4, 32)
    assert core.encode_document_hidden(hidden, extractor=IDENTITY).shape == (1, 4, 32)
    with pytest.raises(ValueError, match="requires pinned hidden-state"):
        core.encode_media("image", torch.rand(1, 3, 8, 8))
    with pytest.raises(ValueError, match="fingerprint"):
        core.encode_media_features("image", hidden, extractor="an-unrelated-model")
    with pytest.raises(ValueError, match="one ordered hidden-state"):
        core.encode_media_views(
            "video",
            (
                (IDENTITY, hidden, torch.ones(1, 7, dtype=torch.bool)),
                (IDENTITY, hidden, torch.ones(1, 7, dtype=torch.bool)),
            ),
        )
    with pytest.raises(ValueError, match="pinned"):
        ObservedEvent("doc", "document", hidden)
    observed = ObservedEvent("doc", "document", hidden, extractor=IDENTITY)
    assert observed.modality == "document"


def test_unified_hidden_mask_cannot_encode_masked_future_tokens():
    core = _build().core
    features = torch.randn(1, 6, 512)
    valid = torch.tensor([[True, True, False, False, False, False]])
    a = core.encode_media_views("audio", ((IDENTITY, features, valid),))
    changed = features.clone()
    changed[:, 2:] = 100.0
    b = core.encode_media_views("audio", ((IDENTITY, changed, valid),))
    torch.testing.assert_close(a, b)
    with pytest.raises(ValueError, match="nonempty"):
        core.encode_media_views("audio", ((IDENTITY, features, torch.zeros_like(valid)),))


def test_projection_only_warmup_freezes_all_except_media_edges():
    model = _build()
    media_names = warmup_parameter_names(model, "media")
    text_names = warmup_parameter_names(model, "text")
    assert any("unified_projection" in n for n in media_names)
    assert any(n.startswith("media_realizers.") for n in media_names)
    assert not any("audio_filter" in n or "visual_filter" in n for n in media_names)
    assert set(media_names).isdisjoint(text_names)
    optimizer = MasterAdamW(model, 1e-3, weight_decay=0.0)
    initial_master_names = set(optimizer.masters)
    freeze_other_owners(optimizer, model, "media")
    assert set(optimizer.masters) == initial_master_names  # Required by DiLoCo
    loss = reconstruction_loss(model, _audio_target())
    loss.backward()
    bridge = model.core.media_stems.unified_projection.weight
    assert bridge.grad is not None and bridge.grad.abs().sum() > 0
    assert not any(p.grad is not None for p in model.core.frontend.parameters())
    assert not any(p.grad is not None for p in model.core.stages.parameters())
    optimizer.step()
    assert set(optimizer.masters) == initial_master_names


def test_same_source_identity_reconstruction_and_finite_gradients(tmp_path):
    model = _build()
    store = TensorAssetStore(tmp_path)
    target = store.put(np.zeros(128, dtype=np.float32), modality="audio", sample_rate=16000)
    source = store.put(
        np.random.default_rng(2).normal(size=(7, 512)).astype("<f4"),
        modality="audio",
        kind="features",
        extractor=IDENTITY,
        source_family=target.sha256,
        sample_rate=16000,
    )
    reference = MediaRecord("audio", source=source, target=target)
    plan = MediaPlan(reference, MediaOutputSpec("audio", sample_count=128), "bootstrap", 4, 32)
    batch = SimpleNamespace(steps=[SimpleNamespace(plans=[plan])])
    numerator, count = media_warmup_batch(model, batch, store, torch.device("cpu"))
    assert count == 1 and numerator.isfinite()
    numerator.backward()
    wrong = store.put(
        np.ones((7, 512), dtype=np.float32),
        modality="audio",
        kind="features",
        extractor=IDENTITY,
        source_family="a" * 64,
        sample_rate=16000,
    )
    bad = MediaPlan(
        MediaRecord("audio", source=wrong, target=target),
        plan.spec,
        "bootstrap",
        4,
        32,
    )
    with pytest.raises(ValueError, match="exact original target"):
        media_warmup_batch(
            model,
            SimpleNamespace(steps=[SimpleNamespace(plans=[bad])]),
            store,
            torch.device("cpu"),
        )


def test_text_hidden_sidecar_rejects_substituted_bytes(tmp_path):
    raw_text = b"one original document chunk"
    digest = hashlib.sha256(raw_text).hexdigest()
    features = np.ones((3, 512), dtype="<f4")
    hidden_bytes = features.tobytes()
    checksum = hashlib.sha256(hidden_bytes).hexdigest()
    (tmp_path / "teacher").mkdir()
    (tmp_path / "teacher" / "hidden.f32le").write_bytes(hidden_bytes)
    row = {
        "format": "megatoken.embeddinggemma2.hidden.v1",
        "source_sha256": digest,
        "modality": "document",
        "hidden_tap": "language_model.embedding_projection:input",
        "valid_tokens": 3,
        "asset": {
            "kind": "features",
            "extractor": IDENTITY,
            "shape": [3, 512],
            "sha256": checksum,
            "path": "teacher/hidden.f32le",
        },
    }
    (tmp_path / "sidecar.jsonl").write_text(json.dumps(row) + "\n")
    cache = TextHiddenSidecar(str(tmp_path / "sidecar.jsonl"), str(tmp_path), extractor=IDENTITY)
    assert raw_text in cache
    assert b"substituted" not in cache
    model = _build()
    loss = text_hidden_alignment(
        model,
        text_bytes=raw_text,
        hidden=cache.get(raw_text, torch.device("cpu")),
        identity=IDENTITY,
    )
    assert loss.isfinite()
    (tmp_path / "teacher" / "hidden.f32le").write_bytes(b"substituted")
    with pytest.raises(ValueError, match="mutated"):
        cache.get(raw_text, torch.device("cpu"))
