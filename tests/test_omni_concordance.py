"""Decoded peer evaluation never invents unavailable modalities/verifiers."""
import numpy as np
import pytest

from megatoken.omni.bundles import Relation
from megatoken.omni.concordance import SampledEvent, score_sampled_relations


def test_actual_decoded_endpoints_and_audio_video_timing():
    first = np.zeros((3, 8, 8), dtype=np.float32)
    last = np.ones((3, 8, 8), dtype=np.float32)
    video = np.stack((first, (first + last) / 2, last))
    events = {
        "video": SampledEvent("video", "video", video, (("bark", 0.4),)),
        "opening": SampledEvent("opening", "image", first.copy()),
        "ending": SampledEvent("ending", "image", last.copy()),
        "audio": SampledEvent("audio", "audio", np.zeros(128), (("bark", 0.42),)),
    }
    relations = (
        Relation("first_frame_of", "opening", "video"),
        Relation("last_frame_of", "ending", "video"),
        Relation("synchronized_with", "audio", "video"),
    )
    report = score_sampled_relations(relations, events)
    assert not report.unmeasured
    assert report.measured["first_frame_of_rmse"]["score"] == 0
    assert report.measured["last_frame_of_rmse"]["score"] == 0
    assert report.measured["av_event_timing_recall"]["score"] == 1


def test_speech_agreement_uses_actual_audio_not_hidden_teacher():
    utterance = np.zeros(24000, dtype=np.float32)
    events = {
        "transcript": SampledEvent("transcript", "text", "hello world"),
        "speech": SampledEvent("speech", "audio", utterance),
    }
    pairs = (Relation("same_spoken_content", "transcript", "speech"),)
    missing = score_sampled_relations(pairs, events)
    assert "sampled_speech_text_wer" not in missing.measured
    assert missing.unmeasured and "speech_recognizer_required" in missing.unmeasured[0]
    measured = score_sampled_relations(
        pairs, events, speech_recognizer=lambda audio: "hello world",
    )
    assert measured.measured["sampled_speech_text_wer"]["score"] == 0.0


def test_edit_preservation_only_outside_requested_change():
    base = np.zeros((3, 8, 8), dtype=np.float32)
    new = base.copy()
    new[:, 1:3, 1:3] = 1
    mask = np.zeros((8, 8), dtype=bool)
    mask[1:3, 1:3] = True
    sampled = {
        "base": SampledEvent("base", "image", base),
        "edited": SampledEvent("edited", "image", new, changed_region=mask),
    }
    result = score_sampled_relations(
        (Relation("preserves", "edited", "base"),), sampled,
    )
    assert result.measured["edit_unchanged_rmse"]["score"] == 0.0
    with pytest.raises(ValueError, match="sampled outputs"):
        SampledEvent("target", "image", base, provenance="teacher")
