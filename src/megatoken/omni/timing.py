"""Output-only impulse timing checks and deliberately shifted media controls.

This detector is for controlled flash/click calibration. It does not label
speech, lips, actions or semantic events in arbitrary video. Silence or the
absence of a visible impulse provides no timing evidence, never perfect credit.
"""

from __future__ import annotations

import math

import numpy as np

from megatoken.omni.evaluation import timed_event_matches


def impulse_events(array, *, modality, rate, threshold=0.65, refractory_seconds=0.1):
    if (
        not math.isfinite(rate)
        or rate <= 0
        or not 0 < threshold < 1
        or not math.isfinite(refractory_seconds)
        or refractory_seconds <= 0
    ):
        raise ValueError("Impulse detection requires a positive clock and bounded threshold")
    values = np.asarray(array, dtype=np.float64)
    if not values.size or not np.isfinite(values).all():
        raise ValueError("Impulse detection requires nonempty finite decoded media")
    if modality == "audio":
        if values.ndim == 2 and values.shape[0] == 1:
            values = values[0]
        if values.ndim != 1:
            raise ValueError("Use one original mono output stream for acoustic timing")
        energy = np.abs(values)
    elif modality == "video":
        if values.ndim == 5 and values.shape[0] == 1:
            values = values[0]
        if values.ndim != 4:
            raise ValueError("Visible impulse detection needs time-first decoded frames")
        # Full-frame flash brightness, independent of caption/prompt labels.
        energy = values.mean(axis=tuple(range(1, values.ndim)))
    else:
        raise ValueError("Timing detector supports actual audio or video")
    floor, ceiling = np.quantile(energy, 0.1), float(energy.max())
    if ceiling - floor < 1e-8:
        return ()
    above = energy > floor + threshold * (ceiling - floor)
    starts = np.flatnonzero(above & ~np.r_[False, above[:-1]])
    events = []
    for index in starts:
        time = index / rate
        if not events or time - events[-1][1] >= refractory_seconds:
            events.append(("impulse", time))
    return tuple(events)


def shifted_timing_control(
    audio, video, *, sample_rate, fps, shift_seconds=0.4, tolerance_seconds=0.12
):
    """Actually shift the decoded waveform, then repeat output event detection."""
    if shift_seconds <= 2 * tolerance_seconds or tolerance_seconds < 0:
        raise ValueError("A timing negative control must exceed the matching tolerance")
    samples = np.asarray(audio)
    if samples.ndim == 2 and samples.shape[0] == 1:
        samples = samples[0]
    shift = round(shift_seconds * sample_rate)
    if shift < 1:
        raise ValueError("The declared shift must contain at least one actual audio sample")
    shifted = np.zeros_like(samples)
    if shift < len(samples):
        shifted[shift:] = samples[:-shift]
    video_events = impulse_events(video, modality="video", rate=fps)
    audio_events = impulse_events(samples, modality="audio", rate=sample_rate)
    shifted_events = impulse_events(shifted, modality="audio", rate=sample_rate)
    positive, denominator = timed_event_matches(
        video_events, audio_events, tolerance_seconds=tolerance_seconds
    )
    negative, _ = timed_event_matches(
        video_events, shifted_events, tolerance_seconds=tolerance_seconds
    )
    measured = bool(video_events and audio_events)
    return {
        "detector": "output-only full-frame flash / acoustic impulse v1",
        "audio_events": audio_events,
        "video_events": video_events,
        "shifted_audio_events": shifted_events,
        "reference_event_count": denominator,
        "matched_events": positive,
        "shifted_matched_events": negative,
        "timing_evidence_available": measured,
        "negative_control_separated": measured and positive > negative,
        "shift_seconds": shift_seconds,
        "tolerance_seconds": tolerance_seconds,
        "semantic_av_synchronization_established": False,
    }
