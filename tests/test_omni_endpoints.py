"""Endpoint targets use the first and last *displayed* frame, never codec padding."""

from hashlib import sha256

import numpy as np
import pytest

from megatoken.omni.endpoints import EndpointMap, displayed_endpoints


def _clip(frames=3, height=8, width=8):
    clip = np.zeros((frames, 3, height, width), dtype=np.float32)
    for index in range(frames):
        clip[index] += index
    return clip


def test_endpoints_are_the_first_and_last_displayed_frames():
    clip = _clip(3)
    opening, closing, audit = displayed_endpoints(
        clip, sampled_source_indices=(11, 13, 18), fps_num=24000, fps_den=1001
    )
    assert np.array_equal(opening, clip[0]) and np.array_equal(closing, clip[2])
    assert opening.shape == closing.shape == (3, 8, 8)
    # Endpoints are copies: editing one never mutates the clip.
    opening += 5
    assert clip[0].max() == 0
    assert audit == dict(
        first_display_index=0,
        last_display_index=2,
        sampled_source_indices=(11, 13, 18),
        crop_xyxy=(),
        fps_num=24000,
        fps_den=1001,
        spatial_transform="already_displayed",
    )


def test_opening_frame_may_start_later_and_crop_is_audited():
    clip = _clip(5)
    opening, closing, audit = displayed_endpoints(
        clip,
        sampled_source_indices=(0, 2, 4, 6, 8),
        fps_num=30,
        fps_den=1,
        first_display_index=1,
        crop_xyxy=(1, 2, 7, 8),
        spatial_transform="center_crop_resize",
    )
    assert opening[0, 0, 0] == 1 and closing[0, 0, 0] == 4
    assert audit["first_display_index"] == 1 and audit["last_display_index"] == 4
    assert audit["crop_xyxy"] == (1, 2, 7, 8)
    assert audit["spatial_transform"] == "center_crop_resize"


def test_audit_becomes_a_fingerprinted_endpoint_map_bound_to_media_digests():
    """The preparer digests the video and both endpoint frames, then binds them to the audit."""
    clip = _clip(3)
    opening, closing, audit = displayed_endpoints(
        clip, sampled_source_indices=(11, 13, 18), fps_num=24000, fps_den=1001
    )
    video, first, last = (sha256(x.tobytes()).hexdigest() for x in (clip, opening, closing))
    assert len({video, first, last}) == 3
    mapping = EndpointMap(video, first, last, **audit)
    assert mapping.fingerprint == EndpointMap(video, first, last, **audit).fingerprint
    assert mapping.fingerprint != EndpointMap(video, last, first, **audit).fingerprint
    assert mapping.fingerprint != EndpointMap("0" * 64, first, last, **audit).fingerprint
    with pytest.raises(ValueError, match="pinned media digests"):
        EndpointMap("short", first, last, **audit)


def test_endpoints_reject_inconsistent_display_mapping():
    clip = _clip(3)
    kw = dict(fps_num=30, fps_den=1)
    with pytest.raises(ValueError, match="source display order"):
        displayed_endpoints(clip, sampled_source_indices=(3, 9, 2), **kw)
    # One source index per displayed frame: padding frames cannot be mapped.
    with pytest.raises(ValueError, match="match exactly"):
        displayed_endpoints(clip, sampled_source_indices=(11, 13, 18, 21), **kw)
    with pytest.raises(ValueError, match="match exactly"):
        displayed_endpoints(clip, sampled_source_indices=(11, 13), **kw)
    with pytest.raises(ValueError, match="outside visible frames"):
        displayed_endpoints(clip, sampled_source_indices=(1, 2, 3), first_display_index=3, **kw)
    with pytest.raises(ValueError, match="outside visible frames"):
        displayed_endpoints(clip, sampled_source_indices=(1, 2, 3), first_display_index=-1, **kw)
    with pytest.raises(ValueError, match="crop"):
        displayed_endpoints(clip, sampled_source_indices=(1, 2, 3), crop_xyxy=(0, 0, 9, 8), **kw)
    with pytest.raises(ValueError, match="frames, 3, height, width"):
        displayed_endpoints(clip[:, 0], sampled_source_indices=(1, 2, 3), **kw)


def test_endpoint_map_requires_a_real_clock_and_ordered_frames():
    digests = ["a" * 64] * 3
    common = dict(first_display_index=0, last_display_index=1, sampled_source_indices=(0, 1))
    EndpointMap(*digests, fps_num=24, **common)
    with pytest.raises(ValueError, match="frame clock"):
        EndpointMap(*digests, **common)
    with pytest.raises(ValueError, match="outside displayed samples"):
        EndpointMap(*digests, fps_num=24, **{**common, "last_display_index": 2})
    with pytest.raises(ValueError, match="Negative"):
        EndpointMap(*digests, fps_num=24, **{**common, "sampled_source_indices": (-1, 1)})
    with pytest.raises(ValueError, match="four image coordinates"):
        EndpointMap(*digests, fps_num=24, crop_xyxy=(0, 0, 4), **common)
