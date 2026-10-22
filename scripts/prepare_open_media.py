#!/usr/bin/env python3
"""Build the open-data media streams of one prepared root with the locked teacher and realizers.

Reads the pinned public datasets listed in data/open_media.py from ``--inputs`` (the layout the
cloud job ``prepare-open-mix`` downloads) and writes one prepared root holding

  audio_identity, transcription              LibriSpeech (CC-BY-4.0)
  speech_edit                                LibriSpeech + deterministic ffmpeg edits
  verbalization                              LibriTTS-R (CC-BY-4.0)
  speech_translation                         FLEURS source-language -> English (CC-BY-4.0)
  image_identity, image_description,
  image_creation                             DOCCI (CC-BY-4.0)
  image_edit                                 MagicBrush (CC-BY-4.0; COCO source images)
  video_identity, av_creation, av_edit       RekaDaily-10k-processed (Apache-2.0) + ffmpeg edits

Each count is the number of source clips drawn; 0 skips the dataset. Streams that no open
dataset supplies are listed in docs/data.md.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from megatoken.data import DedupConfig, DedupStore
from megatoken.data.media_prepare import MediaPreparer
from megatoken.data.open_media import (
    SOURCES,
    CachingPreparer,
    av_episodes,
    edit_episodes,
    image_episodes,
    read_docci,
    read_fleurs,
    read_librispeech,
    read_libritts,
    read_magicbrush,
    read_reka,
    speech_edit,
    speech_episodes,
    translation_episodes,
    verbalization_episodes,
    video_edit,
    voice_references,
)
from megatoken.data.storage import export_prepared
from megatoken.loading import load_codec

LANGUAGES = ("de_de", "fr_fr", "es_419", "it_it")


def episodes(args: argparse.Namespace, media: CachingPreparer):
    inputs, work = args.inputs, args.work
    if args.librispeech:
        clips = list(
            read_librispeech(
                sorted((inputs / "librispeech").glob("*.parquet")),
                work / "librispeech",
                args.librispeech,
            )
        )
        for clip in clips:
            yield from speech_episodes(media, clip)
        for clip in clips[: args.speech_edits]:
            yield from edit_episodes(
                media,
                SOURCES["librispeech_asr"],
                "speech",
                speech_edit(clip, work / "speech-edits"),
                provenance="executed",
                tier="procedural",
            )
    if args.libritts:
        clips = list(
            read_libritts(
                sorted((inputs / "libritts_r").glob("*.parquet")), work / "libritts", args.libritts
            )
        )
        references = voice_references(clips)
        for clip in clips:
            yield from verbalization_episodes(media, clip, references.get(clip["id"]))
    if args.fleurs:
        for pair in read_fleurs(inputs / "fleurs", LANGUAGES, work / "fleurs", args.fleurs):
            yield from translation_episodes(media, pair)
    if args.docci:
        for clip in read_docci(
            inputs / "docci/docci_descriptions.jsonlines",
            inputs / "docci/docci_images.tar.gz",
            work / "docci",
            args.docci,
        ):
            yield from image_episodes(media, clip)
    if args.magicbrush:
        for clip in read_magicbrush(
            sorted((inputs / "magicbrush").glob("*.parquet")), work / "magicbrush", args.magicbrush
        ):
            yield from edit_episodes(
                media,
                SOURCES["magicbrush"],
                "image",
                clip,
                provenance="natural",
                tier="publisher-annotated",
            )
    if args.reka:
        for clip in read_reka(inputs / "reka/shard-00000.tar", work / "reka", args.reka):
            yield from av_episodes(media, clip)
            yield from edit_episodes(
                media,
                SOURCES["rekadaily_processed"],
                "av",
                video_edit(clip, work / "video-edits"),
                provenance="executed",
                tier="procedural",
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packs", required=True, help="Locked pack manifest")
    parser.add_argument("--root", type=Path, required=True, help="Prepared dataset root (new)")
    parser.add_argument("--codec", required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True, help="Scratch for unpacked media")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--librispeech", type=int, default=2000)
    parser.add_argument("--speech-edits", type=int, default=600)
    parser.add_argument("--libritts", type=int, default=1500)
    parser.add_argument("--fleurs", type=int, default=400, help="Pairs per source language")
    parser.add_argument("--docci", type=int, default=2000)
    parser.add_argument("--magicbrush", type=int, default=1000)
    parser.add_argument("--reka", type=int, default=600)
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    args = parser.parse_args()
    if (args.root / "manifest.json").exists():
        parser.error("Prepared manifest already exists; prepared data is immutable")
    media = CachingPreparer(MediaPreparer(args.packs, args.root, device=args.device))
    codec = load_codec(args.codec)
    dedup = DedupStore(
        args.root / "preparation.sqlite",
        DedupConfig(
            train_fraction=1 - args.validation_fraction,
            validation_fraction=args.validation_fraction,
        ),
    )
    counts: Counter[str] = Counter()
    try:
        with dedup.transaction():
            for episode in episodes(args, media):
                dedup.add(episode, codec)
                counts[episode.stream] += 1
                if sum(counts.values()) % 100 == 0:
                    print(json.dumps(counts, sort_keys=True), flush=True)
        dedup.freeze()
        export_prepared(
            dedup,
            args.root,
            codec=codec,
            source_receipts=[
                {
                    "name": "open_media",
                    "sources": {key: asdict(value) for key, value in SOURCES.items()},
                    "counts": dict(counts),
                }
            ],
            pipeline_fingerprint="open-media-v1",
        )
    finally:
        dedup.close()
    print(json.dumps(counts, sort_keys=True))


if __name__ == "__main__":
    main()
