#!/usr/bin/env python3
"""Build the media-identity training set: autoencode each prepared media event.

Each record from scripts/prepare_media.py (with a realization target) becomes one
episode: a short instruction, then the assistant realizes that media. The event
observes nothing of the target; the meridian forward encodes its teacher features
through E and trains R to reproduce its codec latents (plus P's facet prediction).
Optional --text-episodes imports schema-v3 native text episodes into the "text"
stream, so one task recipe mixes text and identity media.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

from megatoken.data import DedupConfig, DedupStore, Episode
from megatoken.data.media_assets import MediaRecord, TensorAssetStore
from megatoken.data.open_media import identity_episode
from megatoken.data.storage import export_prepared
from megatoken.loading import load_codec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True, help="Root holding the assets")
    parser.add_argument("--codec", required=True)
    parser.add_argument("--text-episodes", type=Path)
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    args = parser.parse_args()
    if (args.root / "manifest.json").exists():
        parser.error("Prepared manifest already exists; prepared data is immutable")
    store = TensorAssetStore(args.root)
    codec = load_codec(args.codec)
    dedup = DedupStore(
        args.root / "preparation.sqlite",
        DedupConfig(
            train_fraction=1 - args.validation_fraction,
            validation_fraction=args.validation_fraction,
        ),
    )
    counts: dict[str, int] = {}
    try:
        with dedup.transaction():
            for line in args.records.read_text(encoding="utf-8").split("\n"):
                if not line.strip():
                    continue
                record = MediaRecord.from_bytes(json.loads(line)["record"].encode())
                for asset in (record.source, record.target, record.embedding, record.rendered):
                    if asset is not None:
                        store.load(asset)  # every referenced asset is present and intact
                episode = identity_episode(record)
                dedup.add(episode, codec)
                counts[episode.stream] = counts.get(episode.stream, 0) + 1
            if args.text_episodes is not None:
                for line in args.text_episodes.read_text(encoding="utf-8").split("\n"):
                    if not line.strip():
                        continue
                    episode = Episode.from_dict(json.loads(line))
                    if any(
                        event.modality != "text"
                        for meridian in episode.meridians
                        for event in meridian.events
                    ):
                        raise ValueError("Text episodes must contain only text events")
                    dedup.add(replace(episode, stream="text"), codec)
                    counts["text"] = counts.get("text", 0) + 1
        dedup.freeze()
        export_prepared(
            dedup,
            args.root,
            codec=codec,
            source_receipts=[
                {
                    "name": "media_identity",
                    "records_sha256": hashlib.sha256(args.records.read_bytes()).hexdigest(),
                    "counts": counts,
                }
            ],
            pipeline_fingerprint="media-identity-v1",
        )
    finally:
        dedup.close()
    print(json.dumps(counts, sort_keys=True))


if __name__ == "__main__":
    main()
