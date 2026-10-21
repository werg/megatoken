#!/usr/bin/env python3
"""Produce immutable endpoint images from final displayed/prepared video tensors.

Runs only when users explicitly prepare real assets. This script is infrastructure
only; no data is acquired and no Megatoken weights are trained.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from megatoken.data.media_assets import TensorAsset, TensorAssetStore
from megatoken.omni.endpoints import derive_displayed_endpoints


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--video-asset", required=True, help="Existing TensorAsset JSON")
    parser.add_argument(
        "--source-frame-indices",
        required=True,
        help="JSON array recording original sampled frame indices",
    )
    parser.add_argument("--first-index", type=int, default=0)
    parser.add_argument("--valid-frames", type=int)
    parser.add_argument("--crop-xyxy", default="[]")
    parser.add_argument("--spatial-transform", default="already_displayed")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    asset = TensorAsset.from_dict(json.loads(Path(args.video_asset).read_text()))
    indices = tuple(json.loads(Path(args.source_frame_indices).read_text()))
    crop = tuple(json.loads(args.crop_xyxy))
    first, last, mapping = derive_displayed_endpoints(
        TensorAssetStore(args.root),
        asset,
        sampled_source_indices=indices,
        first_display_index=args.first_index,
        valid_frames=args.valid_frames,
        crop_xyxy=crop,
        spatial_transform=args.spatial_transform,
    )
    Path(args.output).write_text(
        json.dumps(
            {
                "first_frame": asdict(first),
                "last_frame": asdict(last),
                "mapping": asdict(mapping),
                "mapping_fingerprint": mapping.fingerprint,
            },
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(args.output)


if __name__ == "__main__":
    main()
