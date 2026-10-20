"""Simultaneous translation must not see a later source window or copy its timing."""
import pytest

from megatoken.omni.translation import SourceChunk, TranslationClock, TranslationCommit


def test_prefix_limited_translation_wait_and_natural_speech_timing():
    clock = TranslationClock()
    clock.append_source(SourceChunk("german-A", 0.0, 1.0, 1.1, 16000, 16000, "de"))
    clock.append_source(SourceChunk("german-B", 1.0, 2.0, 2.2, 16000, 16000, "de"))
    with pytest.raises(ValueError, match="later source"):
        clock.commit(TranslationCommit(
            1.2, ("german-B",), "english-B-text", "english-B-speech",
            1.2, 2.0, "en",
        ))
    clock.commit(TranslationCommit(
        1.3, ("german-A",), "english-A-text", "english-A-speech",
        1.4, 2.8, "en",
    ))
    # Target speech is naturally longer than its German source; no implicit
    # time stretch or copied word timing occurs.
    assert clock.translations[0].target_speech_end == 2.8
    assert clock.backlog(1.5) == ()
    assert clock.backlog(2.3) == ("german-B",)
    clock.wait(2.0, 2.2, "waiting for additional source evidence")
    clock.commit(TranslationCommit(
        2.3, ("german-B",), "english-B-text", "english-B-speech",
        2.5, 4.0, "en",
    ))
    assert clock.backlog(2.4) == ()
    assert clock.latency()["1"] == pytest.approx(0.5)


def test_translation_rejects_source_reuse_and_temporal_overlap():
    clock = TranslationClock()
    clock.append_source(SourceChunk("one", 0, 1, 1, 24000, 24000, "en"))
    with pytest.raises(ValueError, match="overlap"):
        clock.append_source(SourceChunk("two", .5, 1.5, 2.0, 24000, 24000, "en"))
    first = TranslationCommit(1, ("one",), "text1", "speech1", 1, 2, "de")
    clock.commit(first)
    with pytest.raises(ValueError, match="already translated"):
        clock.commit(TranslationCommit(
            1.1, ("one",), "text2", "speech2", 1.2, 2.0, "de",
        ))
