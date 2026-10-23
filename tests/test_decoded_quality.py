import json

import numpy as np
import pytest

from megatoken.data.schema import Episode, Lineage, PhysicalRecord
from megatoken.omni.concordance import SampledEvent
from megatoken.omni.quality import DecodedQualityArchive, canonical_decoded_array
from megatoken.omni.timing import impulse_events, shifted_timing_control


def test_shifted_actual_waveform_fails_output_timing_check():
    audio = np.zeros(2000, np.float32)
    audio[500:520] = 0.8
    video = np.zeros((20, 3, 4, 4), np.float32)
    video[5] = 1
    report = shifted_timing_control(audio, video, sample_rate=1000, fps=10)
    assert report["matched_events"] == report["reference_event_count"] == 1
    assert report["shifted_matched_events"] == 0
    assert report["negative_control_separated"]
    silence = shifted_timing_control(audio * 0, video * 0, sample_rate=1000, fps=10)
    assert not silence["timing_evidence_available"]
    assert not silence["negative_control_separated"]
    assert impulse_events(audio[None], modality="audio", rate=1000) == (("impulse", 0.5),)


def test_quality_archive_retains_realized_text_and_missing_decode_failures(tmp_path):
    episode = Episode(
        "example",
        (
            PhysicalRecord("words", "assistant", b"the expected words", policy_mask=True),
            PhysicalRecord("invalid", "assistant", b"also expected", policy_mask=True),
        ),
        Lineage("source", "rev", "held-out", frozen_split="validation"),
    )
    archive = DecodedQualityArchive(tmp_path / "quality", evaluator_metadata={"reference": "human"})
    report = archive(episode, {"words": SampledEvent("words", "text", "unrelated output")})
    agreement = report["reference_agreement"]["reference_word_error"]
    assert agreement["count"] == 1 and agreement["score"] > 0
    assert report["generated_targets_without_decoded_sample"] == ["invalid"]
    assert all(r["separates"] for r in report["negative_controls"])
    asset = tmp_path / "quality" / report["sampled_assets"]["words"]["path"]
    assert asset.read_text() == "unrelated output"
    assert (
        json.loads((asset.parent / "report.json").read_text())["human_review"]["status"]
        == "pending"
    )
    with pytest.raises(ValueError, match="overwrite"):
        archive(episode, {})


def test_external_verifier_requires_identity(tmp_path):
    with pytest.raises(ValueError, match="recognizer identity"):
        DecodedQualityArchive(
            tmp_path / "quality",
            evaluator_metadata={"reference": "human"},
            speech_recognizer=lambda _: "answer",
        )


def test_real_image_single_frame_layout_is_normalized_before_endpoint_scoring():
    pixels = np.zeros((1, 1, 3, 8, 12), np.float32)
    assert canonical_decoded_array("image", pixels).shape == (3, 8, 12)
    assert canonical_decoded_array("video", np.zeros((1, 9, 3, 8, 12))).shape == (9, 3, 8, 12)
    with pytest.raises(ValueError, match="layout"):
        canonical_decoded_array("image", np.zeros((1, 2, 3, 8, 12)))
