#!/usr/bin/env python3
"""Prepare an inspected AMI recording through the ordinary teacher/codec path."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from megatoken.data import DedupConfig, DedupStore
from megatoken.data.chunking import ChunkingConfig
from megatoken.data.media_prepare import MediaPreparer
from megatoken.data.meetings import ami_conversation
from megatoken.data.sources import TermsReview
from megatoken.data.storage import export_prepared
from megatoken.io import sha256_file, write_json
from megatoken.loading import load_codec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--signals", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--packs", required=True)
    parser.add_argument("--codec", required=True, help="Matching training codec; byte for a pilot")
    parser.add_argument("--terms-json", type=Path, required=True, help="Reviewed source terms")
    parser.add_argument("--meeting", default="ES2002a")
    parser.add_argument("--start", type=int, default=76)
    parser.add_argument("--chunks", type=int, default=4)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="train")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    terms = TermsReview(**json.loads(args.terms_json.read_text()))
    terms.require_approval()
    codec = load_codec(args.codec)
    if args.root.exists():
        raise FileExistsError("Prepared media is immutable; choose a fresh root")
    args.root.mkdir(parents=True)
    media = MediaPreparer(args.packs, args.root, device=args.device)
    episode = ami_conversation(
        media,
        args.signals,
        args.annotations,
        meeting=args.meeting,
        start_second=args.start,
        chunks=args.chunks,
        split=args.split,
    )
    with DedupStore(
        args.root / "spool.sqlite", DedupConfig(train_fraction=1, validation_fraction=0)
    ) as store:
        with store.transaction():
            store.add(episode, codec)
        store.freeze()
        report = export_prepared(
            store,
            args.root,
            codec=codec,
            source_receipts=[
                {
                    "source_id": "ami-meetings",
                    "revision": "manual-1.6.2",
                    "annotations_sha256": sha256_file(args.annotations),
                    "license": "CC-BY-4.0",
                    "license_url": "https://groups.inf.ed.ac.uk/ami/download/",
                    "origin": "natural_recorded_speaker_continuation",
                    "meeting": args.meeting,
                    "reader": "ami-causal-prefix-v1",
                    "word_timing": "publisher; not independently reviewed",
                    "speaker_channels": "original individual headset files",
                    "packs_sha256": sha256_file(args.packs),
                    "terms": asdict(terms),
                }
            ],
            pipeline_fingerprint="ami-causal-prefix-v1",
            chunking=ChunkingConfig().with_hard_limit(512, max_bytes=2048),
        )
    write_json(
        args.root / "preparation-receipt.json",
        {
            "fingerprint": report["fingerprint"],
            "coverage": report["behavior_coverage"],
            "generated_audio_chunks": args.chunks - 1,
            "sampled_model_quality_established": False,
        },
    )


if __name__ == "__main__":
    main()
