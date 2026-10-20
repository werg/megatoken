"""Deterministic displayed-video endpoint supervision; no decoder-side bypass."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json

import numpy as np

from megatoken.data.media_assets import TensorAssetStore


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
        if len(self.video_sha256) != 64 or len(self.opening_sha256) != 64 or len(self.closing_sha256) != 64:
            raise ValueError("Endpoint mapping requires pinned media digests")
        if not (0 <= self.first_display_index <= self.last_display_index <
                len(self.sampled_source_indices)):
            raise ValueError("Endpoint mapping outside displayed samples")
        if not self.fps_num > 0 or not self.fps_den > 0:
            raise ValueError("Endpoint mapping needs actual frame clock")
        if any(index < 0 for index in self.sampled_source_indices):
            raise ValueError("Negative original source frame index")
        if len(self.crop_xyxy) not in (0, 4):
            raise ValueError("Crop mapping must have four image coordinates")
    @property
    def fingerprint(self):
        return sha256(json.dumps(asdict(self),sort_keys=True).encode()).hexdigest()


def derive_displayed_endpoints(
    store: TensorAssetStore,
    video_asset,
    *,
    sampled_source_indices: tuple[int, ...],
    first_display_index: int = 0,
    valid_frames: int | None = None,
    crop_xyxy: tuple[int, int, int, int] = (),
    spatial_transform: str = "already_displayed",
):
    """Store actual image targets from the already-final TCHW displayed clip.

    No arbitrary codec latent or padded frame is allowed as an endpoint. The
    exact video SHA and source/display frame mapping stay linked to the images.
    """
    if video_asset.modality != "video" or video_asset.kind != "raw":
        raise ValueError("Only fully displayed RGB source videos supply endpoint images")
    video = store.load(video_asset)
    if video.ndim != 4 or video.shape[1] != 3:
        raise ValueError("Displayed source video must be [frames,3,height,width]")
    available = valid_frames if valid_frames is not None else (video_asset.valid_count or len(video))
    if type(available) is not int or not 1 <= available <= len(video):
        raise ValueError("Invalid valid displayed frame count")
    if len(sampled_source_indices) != available:
        raise ValueError("Frame mapping must match exactly the valid visible interval")
    if (type(first_display_index) is not int or
            not 0 <= first_display_index < available):
        raise ValueError("Opening target lies outside visible frames")
    last = available - 1
    if first_display_index > last:
        raise ValueError("Inverted endpoint mapping")
    if any(np.diff(np.asarray(sampled_source_indices, dtype=np.int64)) < 0):
        raise ValueError("Temporal sampling map must be in source display order")
    if crop_xyxy:
        if len(crop_xyxy) != 4 or not (0 <= crop_xyxy[0] < crop_xyxy[2] <= video.shape[-1]
                                      and 0 <= crop_xyxy[1] < crop_xyxy[3] <= video.shape[-2]):
            raise ValueError("Invalid crop audit geometry")
    params = {
        "modality": "image",
        "kind": "raw",
        "normalization": video_asset.normalization,
        "artifact_version": video_asset.artifact_version,
        "source_family": video_asset.source_family,
    }
    opening = store.put(video[first_display_index].copy(), **params)
    closing = store.put(video[last].copy(), **params)
    mapping = EndpointMap(
        video_sha256=video_asset.sha256,
        opening_sha256=opening.sha256,
        closing_sha256=closing.sha256,
        first_display_index=first_display_index,
        last_display_index=last,
        sampled_source_indices=sampled_source_indices,
        crop_xyxy=crop_xyxy,
        fps_num=video_asset.fps_num, fps_den=video_asset.fps_den,
        spatial_transform=spatial_transform,
    )
    return opening, closing, mapping
