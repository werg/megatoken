"""Unified hidden-state media contracts and frozen projection warmup (no HF downloads)."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from megatoken.data.media_assets import MEDIA_RECORD_TYPE, MediaRecord, TensorAssetStore
from megatoken.generation.codec import ByteCodec
from megatoken.loading import build_model
from megatoken.model.config import ModelConfig
from megatoken.model.media import MediaOutputSpec, MediaTrainTarget
from megatoken.model.multitrack import ObservedEvent
from megatoken.omni.embeddinggemma2_hidden import (
    EmbeddingGemma2HiddenExtractor,
    hidden_identity,
)
from megatoken.training.config import TrainConfig
from megatoken.training.embeddinggemma2_text_sidecar import TextHiddenSidecar
from megatoken.training.embeddinggemma2_warmup import (
    freeze_other_owners,
    media_warmup_batch,
    reconstruction_loss,
    text_hidden_alignment,
    warmup_parameter_names,
)
from megatoken.training.meridians import MediaPlan, _media_plan, _validate_media_source
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


def test_generated_and_observed_media_share_pinned_unified_validation(tmp_path):
    model = _build()
    store = TensorAssetStore(tmp_path)
    target = store.put(np.zeros(128, dtype=np.float32), modality="audio", sample_rate=16000)
    teacher = store.put(
        np.ones((5, 512), dtype=np.float32),
        modality="audio",
        kind="features",
        extractor=IDENTITY,
        source_family=target.sha256,
        sample_rate=16000,
    )
    reference = MediaRecord("audio", teacher, target)
    _validate_media_source(model, reference, "audio")
    entry = SimpleNamespace(
        payload=reference.to_bytes(),
        record=SimpleNamespace(media_type=MEDIA_RECORD_TYPE),
    )
    config = SimpleNamespace(representation_training="bootstrap")
    plan = _media_plan(model, entry, "audio", config)
    assert plan.reference.source.sha256 == teacher.sha256

    raw = store.put(np.zeros(128, dtype=np.float32), modality="audio", sample_rate=16000)
    with pytest.raises(ValueError, match="one pinned hidden-state"):
        _validate_media_source(model, MediaRecord("audio", raw, target), "audio")
    with pytest.raises(ValueError, match="one pinned hidden-state"):
        _media_plan(
            model,
            SimpleNamespace(
                payload=MediaRecord("audio", raw, target).to_bytes(),
                record=entry.record,
            ),
            "audio",
            config,
        )


def test_identity_episode_attests_observed_features_not_future_target(tmp_path):
    from dataclasses import asdict

    from scripts.prepare_embeddinggemma2_identity import make_identity_episode

    store = TensorAssetStore(tmp_path)
    target = store.put(np.zeros(128, dtype=np.float32), modality="audio", sample_rate=16000)
    teacher = store.put(
        np.ones((3, 512), dtype=np.float32),
        modality="audio",
        kind="features",
        extractor=IDENTITY,
        source_family=target.sha256,
        sample_rate=16000,
    )
    (tmp_path / "teacher.json").write_text(
        json.dumps(
            {
                "format": "megatoken.embeddinggemma2.hidden.v1",
                "asset": asdict(teacher),
                "modality": "audio",
                "hidden_tap": "language_model.embedding_projection:input",
                "identity_target_sha256": target.sha256,
                "teacher_revision": "d" * 40,
                "preprocessing_id": "chat-template-v1",
                "valid_tokens": 3,
            }
        )
    )
    episode = make_identity_episode(
        store,
        {
            "modality": "audio",
            "teacher_descriptor": "teacher.json",
            "target_asset": asdict(target),
            "family_id": "one-source",
        },
        source_path=tmp_path / "pairs.jsonl",
        index=1,
    )
    assert episode.meridians[1].events[0].asset_sha256 == teacher.sha256
    assert episode.meridians[1].events[0].asset_sha256 != target.sha256


def test_joint_training_can_retain_text_alignment_without_warmup_lanes():
    common = {
        "phase": "P1",
        "budget": 8,
        "retrieval_sidecar": "query.jsonl",
        "retrieval_document_sidecar": "documents.jsonl",
        "retrieval_memory_recipe": "retrieval.yaml",
        "embeddinggemma2_text_sidecar": "text-hidden.jsonl",
    }
    with pytest.raises(ValueError, match="512d"):
        TrainConfig.from_dict(common)
    config = TrainConfig.from_dict(
        {
            **common,
            "model_overrides": {
                "embeddinggemma2_hidden_dim": 512,
                "embeddinggemma2_hidden_id": IDENTITY,
            },
        }
    )
    assert config.embeddinggemma2_lane == "joint"
    assert config.embeddinggemma2_text_sidecar == "text-hidden.jsonl"


def test_parallel_warmup_requires_two_diloco_lanes():
    with pytest.raises(ValueError, match="DiLoCo"):
        TrainConfig.from_dict(
            {
                "phase": "B0",
                "budget": 8,
                "retrieval_sidecar": "query.jsonl",
                "retrieval_document_sidecar": "documents.jsonl",
                "retrieval_memory_recipe": "retrieval.yaml",
                "embeddinggemma2_warmup": True,
                "embeddinggemma2_media_workers": [0],
                "embeddinggemma2_text_recipe": "text.yaml",
                "embeddinggemma2_media_recipe": "media.yaml",
            }
        )


def test_checkpoint_restore_reinstalls_pinned_native_media_pack(tmp_path, monkeypatch):
    from megatoken import loading
    from megatoken.training import checkpoint

    manifest = tmp_path / "native-packs.json"
    manifest.write_text("immutable native pack declaration", encoding="utf-8")
    codec = ByteCodec.byte_fallback()
    versions = []
    created = []

    class FakeModel:
        def __init__(self):
            self.core = SimpleNamespace(set_model_version=versions.append)

        def load_state_dict(self, state, *, strict):
            assert strict and state == {"weights": "saved"}

    def build(config, model_codec, *, device, precision, native_pack_manifest):
        assert model_codec.fingerprint == codec.fingerprint
        assert precision == "fp32" and device == "cpu"
        created.append(native_pack_manifest)
        return FakeModel()

    payload = {
        "model_config": ModelConfig.tiny().to_dict(),
        "train_config": {
            "codec": "byte",
            "precision": "fp32",
            "native_pack_manifest": str(manifest),
        },
        "codec_fingerprint": codec.fingerprint,
        "native_pack_fingerprint": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "model": {"weights": "saved"},
        "global_step": 23,
    }
    monkeypatch.setattr(checkpoint, "load_checkpoint", lambda path: (payload, {}))
    monkeypatch.setattr(loading, "build_model", build)

    trained = loading.load_trained(tmp_path / "checkpoint", device="cpu")
    assert trained.codec.fingerprint == codec.fingerprint
    assert created == [str(manifest)]
    assert versions == [23]

    manifest.write_text("tampered pack declaration", encoding="utf-8")
    with pytest.raises(ValueError, match="pack manifest checksum"):
        loading.load_trained(tmp_path / "checkpoint", device="cpu")
    assert created == [str(manifest)]


def test_prepared_document_and_long_text_are_real_teacher_ingestion(tmp_path):
    from megatoken.data.schema import (
        Meridian,
        PhysicalRecord,
        TrackEvent,
        TrackSpec,
        encode_record,
    )
    from megatoken.data.text_hidden import TEXT_HIDDEN_RECORD_TYPE, TextHiddenReference
    from megatoken.training.meridians import MeridianStep, _actual_raw

    model = _build()
    codec = ByteCodec.byte_fallback()
    features = np.random.default_rng(11).normal(size=(7, 512)).astype("<f4")
    data = features.tobytes()
    digest = hashlib.sha256(data).hexdigest()
    (tmp_path / "teacher").mkdir()
    asset_file = tmp_path / "teacher" / f"{digest}.f32le"
    asset_file.write_bytes(data)
    source_sha = hashlib.sha256(b"full document bytes").hexdigest()
    content_sha = hashlib.sha256(b"parsed document content").hexdigest()

    for original_kind, event_kind in (
        ("document", "document"),
        ("text", "text_summary"),
    ):
        descriptor = {
            "format": "megatoken.embeddinggemma2.hidden.v1",
            "modality": original_kind,
            "source_sha256": source_sha,
            "teacher_input_sha256": content_sha,
            "hidden_tap": "language_model.embedding_projection:input",
            "preprocessing_id": "chat-template-v1",
            "teacher_revision": "d" * 40,
            "valid_tokens": 7,
            "asset": {
                "kind": "features",
                "path": f"teacher/{digest}.f32le",
                "sha256": digest,
                "extractor": IDENTITY,
                "shape": [7, 512],
            },
        }
        reference = TextHiddenReference.from_cache_descriptor(descriptor)
        assert reference.modality == event_kind
        payload = reference.to_bytes()
        assert TextHiddenReference.from_bytes(payload) == reference
        record = PhysicalRecord(
            "doc",
            "document",
            payload,
            media_type=TEXT_HIDDEN_RECORD_TYPE,
        )
        event = TrackEvent("doc", "doc", event_kind, 0, "document", asset_sha256=digest)
        assert Meridian(0, (event,)).events[0].modality == event_kind
        assert TrackSpec("doc", "document").track_id == "doc"
        encoded = encode_record(record, codec)
        assert encoded.token_ids == () and encoded.transcript_tokens == 0
        step = MeridianStep(0, ("doc",), (4,), (0,), (encoded,), modalities=("text",))
        observed = _actual_raw(
            model,
            step,
            SimpleNamespace(paired_tape=lambda cursor: None),
            TensorAssetStore(tmp_path),
            torch.device("cpu"),
            encoder_grad=False,
        )
        assert observed.shape == (1, 1, 4, 32)
        expected = model.core.encode_document_hidden(
            torch.from_numpy(features)[None], extractor=IDENTITY
        )
        torch.testing.assert_close(observed[:, 0], expected)
        with pytest.raises(ValueError, match="never lexical"):
            encode_record(
                PhysicalRecord(
                    "doc", "document", payload, policy_mask=True, media_type=TEXT_HIDDEN_RECORD_TYPE
                ),
                codec,
            )

    asset_file.write_bytes(b"tampered features")
    with pytest.raises(ValueError, match="checksum/width"):
        reference.load(tmp_path)


def test_compressed_document_cannot_escape_pinned_teacher_cache(tmp_path):
    from megatoken.data.text_hidden import TextHiddenReference

    reference = TextHiddenReference(
        modality="document",
        extractor=IDENTITY,
        feature_sha256="a" * 64,
        source_sha256="b" * 64,
        teacher_input_sha256="c" * 64,
        tokens=7,
    )
    with pytest.raises(ValueError, match="missing"):
        reference.load(tmp_path)
    with pytest.raises(ValueError, match="canonical"):
        TextHiddenReference.from_bytes(json.dumps(json.loads(reference.to_bytes())).encode())
    with pytest.raises(ValueError, match="immutable"):
        TextHiddenReference(
            modality="document",
            extractor="mutable:latest",
            feature_sha256="a" * 64,
            source_sha256="b" * 64,
            teacher_input_sha256="c" * 64,
            tokens=7,
        )
