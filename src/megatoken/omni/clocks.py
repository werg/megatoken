"""Explicit media clock windows and prefix-safe availability."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction


@dataclass(frozen=True)
class TimeWindow:
    start: Fraction
    end: Fraction
    available_until: Fraction

    def __post_init__(self):
        if not 0 <= self.start <= self.end <= self.available_until:
            raise ValueError("Window would see future source media")


def available_windows(windows, decision_time):
    limit = Fraction(decision_time)
    for w in windows:
        if w.available_until > limit:
            raise ValueError("Encoder window leaks unavailable future samples")
    return tuple(windows)


def video_endpoints(frame_count, displayed_first=0, displayed_last=None):
    if frame_count < 1:
        raise ValueError("Empty displayed video")
    last = frame_count - 1 if displayed_last is None else displayed_last
    if not 0 <= displayed_first <= last < frame_count:
        raise ValueError("Invalid displayed endpoint")
    return displayed_first, last
