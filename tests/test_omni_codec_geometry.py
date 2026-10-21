"""Output geometry and codec-latent contracts of a prepared media record."""

import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from media_support import LATENT, geometry, prepared_record

from megatoken.data.media_assets import (
    MEDIA_RECORD_TYPE,
    MediaRecord,
    OutputGeometry,
    TensorAssetStore,
)
from megatoken.media_teacher import PINNED_EXTRACTOR
from megatoken.model.media import MediaOutputSpec, validate_public_media_spec
from megatoken.training.meridians import MediaPlan, _media_plan

MODEL = SimpleNamespace(
    config=SimpleNamespace(media_extractor=PINNED_EXTRACTOR, n_facets=32, facet_dim=512)
)
TRAIN = SimpleNamespace(representation_training="recurrent")


def replace_namespace(namespace, **fields):
    return SimpleNamespace(**{**vars(namespace), **fields})


def _entry(record: MediaRecord):
    return SimpleNamespace(
        payload=record.to_bytes(), record=SimpleNamespace(media_type=MEDIA_RECORD_TYPE)
    )


def _large_video(store, geom):
    base = prepared_record(store, "video", seed=5, tokens=64)
    return replace(base, output_geometry=geom)


def test_large_video_geometry_round_trip_and_training_spec(tmp_path):
    """Physical geometry is stored exactly; the huge frames themselves are never stored."""
    geom = OutputGeometry("video", frames=81, height=720, width=1280, fps_num=24000, fps_den=1001)
    original = _large_video(TensorAssetStore(tmp_path), geom)
    raw = original.to_bytes()
    parsed = MediaRecord.from_bytes(raw)
    assert parsed == original and parsed.to_bytes() == raw
    assert parsed.output_geometry == geom
    assert parsed.target.shape == LATENT["video"]
    assert len(raw) < 16384
    plan = _media_plan(MODEL, _entry(parsed), "video", TRAIN)
    assert isinstance(plan, MediaPlan)
    assert plan.spec == MediaOutputSpec.from_geometry(geom)
    assert (plan.spec.frames, plan.spec.height, plan.spec.width) == (81, 720, 1280)
    assert (plan.spec.fps_num, plan.spec.fps_den) == (24000, 1001)
    # A target's extent is teacher-only until a policy declares it requested/predicted.
    assert plan.spec.provenance == "teacher_only"
    with pytest.raises(ValueError, match="Teacher-only"):
        validate_public_media_spec(plan.spec)
    served = replace(plan.spec, provenance="requested")
    validate_public_media_spec(served)


@pytest.mark.parametrize("modality", ("audio", "image", "video"))
def test_every_modality_yields_its_own_output_spec(tmp_path, modality):
    record = prepared_record(TensorAssetStore(tmp_path), modality, seed=6)
    plan = _media_plan(MODEL, _entry(record), modality, TRAIN)
    expected = geometry(modality)
    assert plan.spec.modality == modality
    assert plan.spec.sample_count == expected.sample_count
    assert (plan.spec.height, plan.spec.width) == (expected.height, expected.width)
    assert plan.spec.frames == expected.frames


def test_observed_record_has_no_realization_plan(tmp_path):
    """Without a target there is no geometry to realize, so it cannot be a generated event."""
    record = prepared_record(TensorAssetStore(tmp_path), "image", target=False)
    assert record.output_geometry is None
    with pytest.raises(ValueError, match="realization target"):
        _media_plan(MODEL, _entry(record), "image", TRAIN)


def test_plan_rejects_other_modality_and_other_teacher(tmp_path):
    record = prepared_record(TensorAssetStore(tmp_path), "image")
    with pytest.raises(ValueError, match="modality"):
        _media_plan(MODEL, _entry(record), "video", TRAIN)
    other = SimpleNamespace(
        config=replace_namespace(MODEL.config, media_extractor=PINNED_EXTRACTOR[::-1])
    )
    with pytest.raises(ValueError, match="different teacher"):
        _media_plan(other, _entry(record), "image", TRAIN)
    untyped = SimpleNamespace(payload=record.to_bytes(), record=SimpleNamespace(media_type="x/y"))
    with pytest.raises(ValueError, match="canonical media descriptor"):
        _media_plan(MODEL, untyped, "image", TRAIN)


def test_target_and_geometry_come_together_and_share_one_source(tmp_path):
    store = TensorAssetStore(tmp_path)
    full = prepared_record(store, "image", seed=7)
    with pytest.raises(ValueError, match="come together"):
        MediaRecord("image", full.source, full.target)
    with pytest.raises(ValueError, match="come together"):
        MediaRecord("image", full.source, None, full.output_geometry)
    foreign = prepared_record(store, "image", seed=8)
    with pytest.raises(ValueError, match="different media"):
        MediaRecord("image", full.source, foreign.target, full.output_geometry)
    with pytest.raises(ValueError, match="geometry modality"):
        MediaRecord("image", full.source, full.target, geometry("video"))
    # The observed view drops the target and its geometry entirely.
    observed = full.observed()
    assert observed == MediaRecord("image", full.source)
    assert full.target.sha256.encode() not in observed.to_bytes()


def test_geometry_bounds_are_exact_per_modality():
    OutputGeometry("audio", sample_count=24000 * 120, sample_rate=24000)
    with pytest.raises(ValueError, match="24 kHz"):
        OutputGeometry("audio", sample_count=100, sample_rate=16000)
    with pytest.raises(ValueError, match="visual geometry"):
        OutputGeometry("audio", sample_count=100, sample_rate=24000, height=16)
    with pytest.raises(ValueError, match="frame clock"):
        OutputGeometry("image", height=16, width=16, fps_num=8)
    with pytest.raises(ValueError, match="frame rate"):
        OutputGeometry("video", frames=4, height=16, width=16)
    with pytest.raises(ValueError, match="missing or excessive"):
        OutputGeometry("video", frames=194, height=16, width=16, fps_num=8)
    with pytest.raises(ValueError, match="provenance"):
        OutputGeometry("image", height=16, width=16, provenance="guessed")


def test_codec_latents_declare_the_realizer_native_clock_and_rank(tmp_path):
    store = TensorAssetStore(tmp_path)
    family = "a" * 64
    realizer = "pocket:" + "b" * 64

    def latent(array, modality, extractor=realizer, **fields):
        return store.put(
            array,
            modality=modality,
            kind="codec_latents",
            extractor=extractor,
            source_family=family,
            **fields,
        )

    latent(np.zeros((6, 4)), "audio", sample_rate=24000)
    with pytest.raises(ValueError, match="24 kHz"):
        latent(np.zeros((6, 4)), "audio", sample_rate=16000)
    with pytest.raises(ValueError, match="native rank"):
        latent(np.zeros((4, 2, 2)), "audio", sample_rate=24000)
    with pytest.raises(ValueError, match="native rank"):
        latent(np.zeros((3, 4, 2, 2)), "image")
    with pytest.raises(ValueError, match="frame rate"):
        latent(np.zeros((3, 4, 2, 2)), "video")
    with pytest.raises(ValueError, match="realizer identity"):
        latent(np.zeros((4, 2, 2)), "image", extractor="no-checkpoint-digest")
    with pytest.raises(ValueError, match="Valid frames"):
        latent(np.zeros((6, 4)), "audio", sample_rate=24000, valid_count=7)
    # Latents are never teacher features, and features never carry a clock or padding.
    with pytest.raises(ValueError, match="Codec latents"):
        latent(np.zeros((12, 512)), "text")
    with pytest.raises(ValueError, match="clock-free"):
        store.put(
            np.zeros((12, 512)),
            modality="audio",
            kind="features",
            extractor=PINNED_EXTRACTOR,
            source_family=family,
            sample_rate=16000,
        )
    with pytest.raises(ValueError, match="512"):
        store.put(
            np.zeros((12, 16)),
            modality="audio",
            kind="features",
            extractor=PINNED_EXTRACTOR,
            source_family=family,
        )


@pytest.mark.parametrize(("modality", "valid"), (("audio", 2), ("video", 1), ("image", 0)))
def test_valid_count_masks_codec_padding_in_the_resolved_target(tmp_path, modality, valid):
    store = TensorAssetStore(tmp_path)
    if modality == "image":
        # Stills have no padded frames to mask: the whole latent is supervised.
        record = prepared_record(store, modality, seed=9)
    else:
        record = prepared_record(store, modality, seed=9, valid_frames=valid)
    plan = _media_plan(MODEL, _entry(record), modality, TRAIN)
    target = plan.resolve(store, torch.device("cpu"))
    assert target.source.shape == (1, 12, 512)
    assert target.target.shape == (1, *LATENT[modality])
    assert target.realizer == record.target.extractor
    assert target.mode == "recurrent"
    if valid:
        frames = LATENT[modality][0]
        assert target.target_mask.shape == (1, frames)
        assert target.target_mask.sum().item() == valid
        assert target.target_mask[0, :valid].all()
        assert not target.target_mask[0, valid:].any()
    else:
        assert target.target_mask is None


def test_media_record_wire_format_is_strict_v3(tmp_path):
    record = prepared_record(TensorAssetStore(tmp_path), "image", seed=10)
    raw = record.to_bytes()
    value = json.loads(raw)
    assert value["version"] == 3
    assert set(value) == {"version", "modality", "source", "target", "output_geometry"}
    value["version"] = 2
    with pytest.raises(ValueError, match="unsupported version"):
        MediaRecord.from_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())
    value["version"] = 3
    with pytest.raises(ValueError, match="canonical"):
        MediaRecord.from_bytes(json.dumps(value, indent=1).encode())
    del value["target"]
    with pytest.raises(ValueError, match="unsupported version or fields"):
        MediaRecord.from_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())
    # An observed record serializes explicit nulls: no field is ever implied.
    observed = prepared_record(TensorAssetStore(tmp_path), "image", target=False).to_bytes()
    assert json.loads(observed)["target"] is None
    assert MediaRecord.from_bytes(observed).to_bytes() == observed
