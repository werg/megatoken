"""Deterministic displayed-video endpoint supervision; no decoder-side bypass."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from hashlib import sha256

import numpy as np


@dataclass(frozen=True)
class EndpointMap:
    """Audit of visible endpoints AFTER temporal sampling/crop/resizing."""

    video_sha256: str
    opening_sha256: str
    closing_sha256: str
    first_display_index: int
    last_display_index: int
    sampled_source_indices: tuple[int, ...]
    crop_xyxy: tuple[int, int, int, int] = ()
    fps_num: int = 0
    fps_den: int = 1
    spatial_transform: str = "already_displayed"

    def __post_init__(self):
        if (
            len(self.video_sha256) != 64
            or len(self.opening_sha256) != 64
            or len(self.closing_sha256) != 64
        ):
            raise ValueError("Endpoint mapping requires pinned media digests")
        if not (
            0
            <= self.first_display_index
            <= self.last_display_index
            < len(self.sampled_source_indices)
        ):
            raise ValueError("Endpoint mapping outside displayed samples")
        if not self.fps_num > 0 or not self.fps_den > 0:
            raise ValueError("Endpoint mapping needs actual frame clock")
        if any(index < 0 for index in self.sampled_source_indices):
            raise ValueError("Negative original source frame index")
        if len(self.crop_xyxy) not in (0, 4):
            raise ValueError("Crop mapping must have four image coordinates")

    @property
    def fingerprint(self):
        return sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


def displayed_endpoints(
    clip: np.ndarray,
    *,
    sampled_source_indices: tuple[int, ...],
    fps_num: int,
    fps_den: int,
    first_display_index: int = 0,
    crop_xyxy: tuple[int, int, int, int] = (),
    spatial_transform: str = "already_displayed",
) -> tuple[np.ndarray, np.ndarray, dict]:
    """The opening and closing displayed frames [3, H, W] of a final clip [F, 3, H, W].

    Returns both frames and the audit fields of their ``EndpointMap`` (the
    caller adds the media digests once the frames are prepared as images).
    Never a codec latent or a padded frame.
    """
    if clip.ndim != 4 or clip.shape[1] != 3:
        raise ValueError("Displayed source video must be [frames, 3, height, width]")
    if len(sampled_source_indices) != len(clip):
        raise ValueError("Frame mapping must match exactly the displayed frames")
    if type(first_display_index) is not int or not 0 <= first_display_index < len(clip):
        raise ValueError("Opening target lies outside visible frames")
    if any(np.diff(np.asarray(sampled_source_indices, dtype=np.int64)) < 0):
        raise ValueError("Temporal sampling map must be in source display order")
    if crop_xyxy and not (
        len(crop_xyxy) == 4
        and 0 <= crop_xyxy[0] < crop_xyxy[2] <= clip.shape[-1]
        and 0 <= crop_xyxy[1] < crop_xyxy[3] <= clip.shape[-2]
    ):
        raise ValueError("Invalid crop audit geometry")
    last = len(clip) - 1
    audit = dict(
        first_display_index=first_display_index,
        last_display_index=last,
        sampled_source_indices=tuple(sampled_source_indices),
        crop_xyxy=tuple(crop_xyxy),
        fps_num=fps_num,
        fps_den=fps_den,
        spatial_transform=spatial_transform,
    )
    return clip[first_display_index].copy(), clip[last].copy(), audit
