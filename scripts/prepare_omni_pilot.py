#!/usr/bin/env python3
"""Create deterministic multi-track learning tasks with native audio/visual assets.

Procedural verification corpus, not a substitute for natural speech/video data.
Every task requires more than one evidence track and contains a paired controlled
tone/image realization target. Assets and episodes have reproducible ancestry.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from megatoken.data import DedupConfig, DedupStore, Episode, Lineage, PhysicalRecord
from megatoken.data.media_assets import MEDIA_RECORD_TYPE, MediaRecord, TensorAssetStore
from megatoken.data.schema import Meridian, TrackEvent, TrackSpec
from megatoken.data.storage import export_prepared
from megatoken.generation.codec import ByteCodec


def _picture(red: int, blue: int) -> np.ndarray:
    picture = np.zeros((3, 16, 16), dtype=np.float32)
    picture[0, : min(red, 16), :7] = 1
    picture[2, : min(blue, 16), 9:] = 1
    return picture


def _tone(count: int, samples: int = 1024) -> np.ndarray:
    index = np.arange(samples, dtype=np.float32)
    hz = 220 + 19 * count
    envelope = np.minimum(1.0, index / 100)
    return (0.2 * np.sin(2 * math.pi * hz * index / 16000) * envelope).astype(np.float32)


def make_episode(
    store: TensorAssetStore,
    *,
    seed: int,
    index: int,
    variant: str = "base",
) -> Episode:
    rng = np.random.default_rng(seed + index * 1201)
    red, blue = (int(n) for n in rng.integers(4, 12, 2))
    used_red, used_blue = (int(n) for n in rng.integers(1, 4, 2))
    if variant == "audio":
        # Change only the acoustic evidence; keep instructions and image fixed.
        blue = blue + 1 if blue < 11 else blue - 1
    elif variant == "visual":
        # Change only visual evidence; keep instructions and sound fixed.
        red = red + 1 if red < 11 else red - 1
    elif variant != "base":
        raise ValueError("Unknown counterfactual variant")
    remain_red, remain_blue = red - used_red, blue - used_blue
    # Hide both initial counts from text. Red is observable only in
    # the visual bar; blue is observable only in the acoustic pitch.
    visual_modality = "video" if index % 2 else "image"
    if visual_modality == "video":
        source_image = store.put(
            np.stack((_picture(red, 0), _picture(red, 0))),
            modality="video",
            fps_num=8,
        )
        target_image = store.put(
            np.stack(
                (
                    _picture(remain_red, remain_blue),
                    _picture(remain_red - 1, remain_blue + 1),
                )
            ),
            modality="video",
            fps_num=8,
        )
    else:
        source_image = store.put(_picture(red, 0), modality="image")
        target_image = store.put(
            _picture(remain_red, remain_blue),
            modality="image",
        )
    source_audio = store.put(
        _tone(blue),
        modality="audio",
        sample_rate=16000,
    )
    target_audio = store.put(
        _tone(remain_red + remain_blue),
        modality="audio",
        sample_rate=16000,
    )
    source_visual = MediaRecord(visual_modality, source_image)
    source_acoustic = MediaRecord("audio", source_audio)
    target_visual = MediaRecord(visual_modality, target_image, target_image)
    target_acoustic = MediaRecord("audio", target_audio, target_audio)
    answer = f"red={remain_red};blue={remain_blue};total={remain_red + remain_blue}".encode()

    records = (
        PhysicalRecord(
            "inventory", "user", b"The camera bar shows red stock; tone pitch encodes blue stock."
        ),
        PhysicalRecord(
            "shipment", "user", f"Remove {used_red} red and {used_blue} blue parts.".encode()
        ),
        PhysicalRecord("camera", "user", source_visual.to_bytes(), media_type=MEDIA_RECORD_TYPE),
        PhysicalRecord("mic", "user", source_acoustic.to_bytes(), media_type=MEDIA_RECORD_TYPE),
        PhysicalRecord("answer", "assistant", answer, turn_index=1, policy_mask=True),
        PhysicalRecord(
            "picture",
            "assistant",
            target_visual.to_bytes(),
            turn_index=1,
            policy_mask=True,
            media_type=MEDIA_RECORD_TYPE,
        ),
        PhysicalRecord(
            "voice",
            "assistant",
            target_acoustic.to_bytes(),
            turn_index=1,
            policy_mask=True,
            media_type=MEDIA_RECORD_TYPE,
        ),
    )
    tracks = tuple(
        TrackSpec(
            name,
            "user" if ix < 4 else "assistant",
        )
        for ix, name in enumerate(
            ("inventory", "shipment", "camera", "mic", "answer", "picture", "voice"),
        )
    )
    meridians = (
        Meridian(
            0,
            (
                TrackEvent("inventory", "inventory", "text", 0, "user"),
                TrackEvent("shipment", "shipment", "text", 0, "user"),
                TrackEvent(
                    "camera", "camera", visual_modality, 0, "user", asset_sha256=source_image.sha256
                ),
                TrackEvent("mic", "mic", "audio", 0, "user", asset_sha256=source_audio.sha256),
            ),
        ),
        Meridian(
            1,
            (
                TrackEvent("answer", "answer", "text", 1, "assistant"),
                TrackEvent(
                    "picture",
                    "picture",
                    visual_modality,
                    1,
                    "assistant",
                    asset_sha256=target_image.sha256,
                ),
                TrackEvent(
                    "voice", "voice", "audio", 1, "assistant", asset_sha256=target_audio.sha256
                ),
            ),
        ),
    )
    return Episode(
        f"pilot-{seed}-{index}-{variant}",
        records,
        Lineage("omni-procedural-pilot", f"seed-{seed}", f"pilot-parent-{seed}-{index}"),
        stream="omni_pilot",
        tracks=tracks,
        meridians=meridians,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--seed", type=int, default=91)
    parser.add_argument(
        "--counterfactual-variants",
        action="store_true",
        help="For each parent, add audio-only and visual-only evidence swaps in the same split family.",
    )
    args = parser.parse_args()
    if not 8 <= args.count <= 100000:
        raise ValueError("Pilot count must be between 8 and 100000")
    root = args.output
    if (root / "manifest.json").exists():
        raise FileExistsError("Prepared corpus is immutable; use a new output directory")
    root.mkdir(parents=True, exist_ok=True)
    assets = TensorAssetStore(root)
    spool = DedupStore(
        root / "preparation.sqlite",
        DedupConfig(train_fraction=0.9, validation_fraction=0.1),
    )
    try:
        with spool.transaction():
            for index in range(args.count):
                for variant in (
                    ("base", "audio", "visual") if args.counterfactual_variants else ("base",)
                ):
                    spool.add(
                        make_episode(assets, seed=args.seed, index=index, variant=variant),
                        ByteCodec.byte_fallback(),
                    )
        spool.freeze()
        result = export_prepared(
            spool,
            root,
            codec=ByteCodec.byte_fallback(),
            source_receipts=[
                {
                    "name": "procedural_omni_pilot",
                    "generator": "prepare_omni_pilot-v1",
                    "seed": args.seed,
                    "episodes": args.count * (3 if args.counterfactual_variants else 1),
                    "parent_families": args.count,
                    "counterfactual_variants": args.counterfactual_variants,
                    "natural_media": False,
                }
            ],
            pipeline_fingerprint=(
                f"omni-pilot-v2-seed-{args.seed}-cf-{int(args.counterfactual_variants)}"
            ),
        )
    finally:
        spool.close()
    print(
        f"Prepared {args.count} parent families and "
        f"{args.count * (3 if args.counterfactual_variants else 1)} candidate joint tasks: {root}"
    )
    print(f"Manifest schema: {result.get('schema_version', 'unknown')}")


if __name__ == "__main__":
    main()
