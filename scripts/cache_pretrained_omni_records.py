#!/usr/bin/env python3
"""Cache a PINNED pretrained omnimodal E feature view without acquiring data.

This executable performs extraction only when explicitly run. Both source and
all donor checkpoint files must already be local; no default HF revisions,
downloads, or silent resampling. The primary output is a canonical MediaRecord.
Examples:
  --family whistle --weights assets/whistle.cact --expected-width 128 --slot primary
  --family minicpm47 --weights assets/MiniCPM-V-4.7 --slot detail
  --family pocket-mimi --weights assets/pocket-tts --slot acoustic --input-record ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from megatoken.data.media_assets import MediaRecord, TensorAssetStore
from megatoken.data.pretrained_extract import (
    exact_audio_rate,
    extract_minicpm_v47,
    extract_pocket_mimi,
    extract_siglip2,
    extract_whistle,
    file_digest,
    pinned_export,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--record", required=True, help="Canonical base MediaRecord path")
    parser.add_argument("--input-record", help="Optional paired raw donor view (e.g. 24k).")
    parser.add_argument("--output", required=True, help="New unique MediaRecord path.")
    parser.add_argument(
        "--family",
        required=True,
        choices=(
            "whistle",
            "pocket-mimi",
            "minicpm47",
            "siglip2",
        ),
    )
    parser.add_argument("--weights", required=True, help="Local donor checkpoint/file.")
    parser.add_argument("--expected-sha256", required=True, help="Pinned SHA256 for .cact file.")
    parser.add_argument(
        "--slot", choices=("primary", "acoustic", "detail", "codec"), default="primary"
    )
    parser.add_argument("--expected-width", type=int, default=0)
    parser.add_argument("--downsample-mode", choices=("4x", "16x"), default="16x")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--preprocessing-id", required=True)
    parser.add_argument(
        "--paired-source-id",
        default="",
        help=(
            "Required when input-record differs from record; externally verified "
            "original recording ID proving the two sample-rate views share ancestry."
        ),
    )
    args = parser.parse_args()
    store = TensorAssetStore(args.root)
    original = MediaRecord.from_bytes(Path(args.record).read_bytes())
    donor_record = (
        MediaRecord.from_bytes(Path(args.input_record).read_bytes())
        if args.input_record
        else original
    )
    if donor_record.modality != original.modality or (
        args.input_record and not args.paired_source_id
    ):
        raise ValueError("Extra source view needs declared shared source ancestry")
    source = donor_record.source
    if source.kind != "raw":
        raise ValueError("Native pretrained extraction needs its original raw physical view")
    weights = Path(args.weights).resolve()
    if not weights.exists():
        raise ValueError("Donor checkpoint must already exist locally")
    if len(args.expected_sha256) != 64 or not all(
        c in "0123456789abcdef" for c in args.expected_sha256
    ):
        raise ValueError("Pinned weights must use exact lowercase SHA256")
    from megatoken.model.native_latents import directory_sha256

    digest = file_digest(weights) if weights.is_file() else directory_sha256(weights)
    if digest != args.expected_sha256:
        raise ValueError("Extracting with altered/unverified pretrained weights")
    raw = store.load(source)
    if args.family == "whistle":
        exact_audio_rate(source, 16000)
        features = extract_whistle(raw, weights=str(weights), expected_width=args.expected_width)
    elif args.family == "pocket-mimi":
        exact_audio_rate(source, 24000)
        from pocket_tts import TTSModel

        donor = TTSModel.load_model(config=str(weights / "config.yaml"))
        donor = donor.to(args.device).eval()
        features = extract_pocket_mimi(raw, donor=donor)
    elif args.family == "minicpm47":
        if original.modality not in {"image", "video"}:
            raise ValueError("MiniCPM-V only accepts image or video observations")
        features = extract_minicpm_v47(
            raw, model_dir=str(weights), device=args.device, downsample_mode=args.downsample_mode
        )
    else:
        if original.modality not in {"image", "video"}:
            raise ValueError("SigLIP2 only accepts image/video observations")
        features = extract_siglip2(raw, model_dir=str(weights), device=args.device)
    if args.expected_width and features.shape[-1] != args.expected_width:
        raise ValueError("Extracted view width differs from declared model geometry")
    identity = (
        f"{args.family}:{digest}:preprocess:{args.preprocessing_id}:"
        f"downsample:{args.downsample_mode}"
    )
    cached = pinned_export(store, original, features, extractor_id=identity, feature_slot=args.slot)
    target = Path(args.output)
    if target.exists():
        raise FileExistsError("Pinned feature extraction never overwrites an existing record")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(cached.to_bytes())
    print(
        json.dumps(
            {
                "record": str(target),
                "view_slot": args.slot,
                "extractor_id": identity,
                "feature_shape": list(features.shape),
                "feature_sha256": (
                    cached.source if args.slot == "primary" else cached.views[-1]
                ).sha256,
                "paired_source_id": args.paired_source_id or None,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
