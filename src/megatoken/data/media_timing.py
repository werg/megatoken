"""Original presentation clocks and bounded, causal media chunk planning.

Integer timestamps come from the container, never from average frame rate.
Conversion to a decoder's uniform clock is explicit and retains the original
displayed-frame correspondence, including variable frame durations.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class VideoTimeline:
    time_base: Fraction
    pts: tuple[int, ...]
    ends: tuple[int, ...]

    def __post_init__(self):
        if (
            self.time_base <= 0
            or not self.pts
            or len(self.pts) != len(self.ends)
            or any(a >= b for a, b in zip(self.pts, self.pts[1:], strict=False))
            or any(a >= b for a, b in zip(self.pts, self.ends, strict=True))
        ):
            raise ValueError("Video needs ordered, explicitly timed displayed frames")

    def uniform_indices(self, start: int, end: int, rate: Fraction, *, max_frames: int = 193):
        if (
            type(start) is not int
            or type(end) is not int
            or start < self.pts[0]
            or end > self.ends[-1]
            or start >= end
            or rate <= 0
        ):
            raise ValueError("Requested video interval is outside its original presentation clock")
        count = int((end - start) * self.time_base * rate)
        if count < 1 or count > max_frames:
            raise ValueError(
                "Video interval exceeds event frame bounds; divide it before extraction"
            )
        times = [Fraction(start) + Fraction(i, 1) / (rate * self.time_base) for i in range(count)]
        # A displayed frame is available from its PTS through its declared end;
        # gaps are rejected rather than filled with invented observations.
        import bisect

        indices = tuple(bisect.bisect_right(self.pts, time) - 1 for time in times)
        if any(time >= self.ends[index] for time, index in zip(times, indices, strict=True)):
            raise ValueError("Requested interval crosses an unannotated presentation gap")
        return indices


def video_timeline(path: str | Path) -> VideoTimeline:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_frames",
            "-show_streams",
            "-show_entries",
            "stream=time_base:frame=best_effort_timestamp,duration,pkt_duration",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(result.stdout)
    frames = report["frames"]
    pts = tuple(int(frame["best_effort_timestamp"]) for frame in frames)
    ends = []
    for index, frame in enumerate(frames):
        duration = int(frame.get("duration", frame.get("pkt_duration", 0)))
        if duration <= 0 and index + 1 < len(pts):
            duration = pts[index + 1] - pts[index]
        if duration <= 0:
            raise ValueError("Container omits the final displayed-frame duration")
        ends.append(pts[index] + duration)
    return VideoTimeline(Fraction(report["streams"][0]["time_base"]), pts, tuple(ends))


def decode_video_interval(path, timeline, start, end, rate, *, max_frames=193):
    from torchcodec.decoders import VideoDecoder

    indices = timeline.uniform_indices(start, end, rate, max_frames=max_frames)
    decoder = VideoDecoder(str(path))
    if len(decoder) != len(timeline.pts):
        raise ValueError("Decoded frame count differs from the original container timeline")
    frames = decoder.get_frames_at(list(indices)).data
    values = frames.float().numpy() / 255.0
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite decoded video")
    return values, indices


def sample_intervals(sample_count: int, sample_rate: int, *, max_seconds: int = 20):
    """Nonoverlapping exact source prefixes; silence is retained as samples."""
    if min(sample_count, sample_rate, max_seconds) <= 0:
        raise ValueError("Audio chunk planning requires a positive original clock and limit")
    bound = sample_rate * max_seconds
    return tuple(
        (start, min(sample_count, start + bound)) for start in range(0, sample_count, bound)
    )
