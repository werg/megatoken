#!/usr/bin/env python3
"""Prepare real recorded image/video peers and endpoint tasks from bounded prefixes."""

from __future__ import annotations

import argparse
import json
import subprocess
import urllib.request
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image

from megatoken.data import DedupConfig, DedupStore
from megatoken.data.concordance import (
    AlignedBundle,
    BundleArtifact,
    Relation,
    TaskView,
    compile_task,
    generated,
    group,
    observed,
)
from megatoken.data.media_prepare import MediaPreparer
from megatoken.data.open_media import identity_episode
from megatoken.data.schema import Lineage
from megatoken.data.sources import TermsReview
from megatoken.data.storage import export_prepared
from megatoken.generation.codec import ByteCodec
from megatoken.io import content_hash, sha256_file, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--packs", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--train-video", type=Path, required=True)
    p.add_argument("--terms", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    terms = json.loads(args.terms.read_text())
    TermsReview(**terms).require_approval()
    args.root.mkdir(parents=True, exist_ok=False)
    args.source_root.mkdir(parents=True, exist_ok=False)
    url = "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/ES2003a/video/ES2003a.Closeup1.avi"
    prefix = args.source_root / "ES2003a.Closeup1.first-8MiB.avi"
    request = urllib.request.Request(url, headers={"Range": "bytes=0-8388607"})
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = response.read(8 << 20)
        transport = {
            "url": url,
            "response_status": response.status,
            "content_range": response.headers.get("Content-Range"),
            "received_prefix_bytes": len(payload),
            "complete_original_claimed": False,
        }
    if payload[:4] != b"RIFF":
        raise ValueError("Publisher response is not an AVI container prefix")
    prefix.write_bytes(payload)
    write_json(
        args.source_root / "download.json",
        {**transport, "sha256": sha256_file(prefix), "terms": terms},
    )
    media = MediaPreparer(args.packs, args.root, device=args.device)
    episodes, receipts = [], []
    for meeting, original, split in (
        ("ES2002a", args.train_video, "train"),
        ("ES2003a", prefix, "validation"),
    ):
        clip = args.source_root / f"{meeting}.9-displayed-frames.mkv"
        command = [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-i",
            str(original),
            "-an",
            "-vf",
            "fps=8,scale=256:192",
            "-frames:v",
            "9",
            "-c:v",
            "ffv1",
            str(clip),
        ]
        subprocess.run(command, check=True, capture_output=True)
        from torchcodec.decoders import VideoDecoder

        frames = VideoDecoder(str(clip), device="cpu").get_frames_in_range(0, 9).data
        if len(frames) != 9:
            raise ValueError(
                "The available container prefix does not provide nine displayed frames"
            )
        image_paths = []
        for name, frame in (("first", frames[0]), ("last", frames[-1])):
            path = args.source_root / f"{meeting}.{name}.png"
            Image.fromarray(np.asarray(frame.permute(1, 2, 0))).save(path)
            image_paths.append(path)
        video = media.media("video", clip)
        opening, closing = [media.media("image", path) for path in image_paths]
        lineage = Lineage(
            "ami-original-video-prefix",
            "ami-public-signals",
            f"ami:{meeting[:-1]}",
            source_uri=str(original),
            raw_sha256=sha256_file(original),
            frozen_split=split,
        )
        artifacts = (
            BundleArtifact(
                "request",
                "request",
                "text",
                payload=b"Generate a short view of a recorded meeting room.",
            ),
            BundleArtifact(
                "plan",
                "content plan",
                "text",
                payload=b"A meeting room at the beginning of a recorded session.",
                provenance="simulated",
            ),
            BundleArtifact("video", "video", "video", media=video),
            BundleArtifact("first", "first frame", "image", media=opening),
            BundleArtifact("last", "last frame", "image", media=closing),
        )
        bundle = AlignedBundle(
            f"visual-prefix:{meeting}",
            lineage,
            artifacts,
            terms["license"],
            "nine-displayed-frames-8fps-256x192-v1",
            "publisher-recording/authored-request",
        )
        relations = (
            Relation("first_frame_of", "first", "video"),
            Relation("last_frame_of", "last", "video"),
        )
        episodes.append(
            compile_task(
                bundle,
                TaskView(
                    "joint-visual-peers",
                    "endpoints",
                    (
                        group(
                            observed("request", "request", "Use the supplied generation request.")
                        ),
                        group(
                            generated(
                                "plan", "scene", "Choose and commit the scene before realizing it."
                            )
                        ),
                        group(
                            generated("first", "scene", "Generate the opening frame."),
                            generated("last", "scene", "Generate the closing frame."),
                            generated("video", "scene", "Generate corresponding motion."),
                        ),
                    ),
                    relations=relations,
                ),
            )
        )
        episodes.append(
            compile_task(
                bundle,
                TaskView(
                    "observed-endpoints",
                    "endpoint_conditioned_video",
                    (
                        group(
                            observed("first", "evidence", "Use the supplied opening frame."),
                            observed("last", "evidence", "Use the supplied closing frame."),
                        ),
                        group(generated("video", "scene", "Continue between the supplied frames.")),
                    ),
                    relations=relations,
                ),
            )
        )
        for descriptor in (video, opening, closing):
            episode = identity_episode(descriptor, lineage.family_id)
            episodes.append(replace(episode, lineage=replace(lineage, task_id=episode.episode_id)))
        receipts.append(
            {
                "meeting": meeting,
                "split": split,
                "source_path": str(original),
                "source_sha256": sha256_file(original),
                "displayed_video_sha256": sha256_file(clip),
                "transform_command": command,
                "target_speech_or_av_alignment_claimed": False,
                "first_display_index": 0,
                "last_display_index": 8,
                "endpoint_provenance": "exact displayed frames after recorded temporal/spatial transform",
            }
        )
    codec = ByteCodec.byte_fallback()
    receipt = {
        "terms": terms,
        "sources": receipts,
        "download": transport,
        "request_and_scene_plan_provenance": "authored over real publisher recordings",
    }
    with DedupStore(
        args.root / "dedup.sqlite", DedupConfig(train_fraction=1, validation_fraction=0)
    ) as store:
        for episode in episodes:
            store.add(episode, codec)
        store.freeze()
        export_prepared(
            store,
            args.root,
            codec=codec,
            source_receipts=[receipt],
            pipeline_fingerprint=content_hash(receipt),
        )
    write_json(args.root / "source-receipts.json", receipt)


if __name__ == "__main__":
    main()
