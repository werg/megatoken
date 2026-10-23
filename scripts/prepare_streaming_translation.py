#!/usr/bin/env python3
"""Preserve publisher audio and translations as sustained, causally safe sessions.

This conservative baseline waits for the announced end of each utterance before
emitting its publisher translation. It invents neither word alignments nor
assistant timing. Speaker-family splits are frozen before chunking.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from collections import Counter
from pathlib import Path

import soundfile as sf

from megatoken.data.open_media import (
    Shards,
    audio_blob,
    field_of,
    frozen_split,
    load_sources,
    parquet_records,
)
from megatoken.io import write_json


def acquire(source, root, *, train_families, validation_families, max_rows=10000):
    """Bound acquisition by original families; retain exact source payloads."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    source.terms.require_approval()
    shards = Shards(source)
    names = shards.names(["fr_en/train*.parquet"])
    if not names:
        raise ValueError("Pinned source has no French/English training shard")
    shards.files = names
    desired = {"train": train_families, "validation": validation_families}
    families, counts, rows, receipts = set(), Counter(), [], []
    iterator = parquet_records(source, shards)
    try:
        for index, record in enumerate(iterator):
            if index >= max_rows:
                break
            speaker = field_of(record, source.columns["family"])
            if not speaker:
                # Without a speaker identity we cannot establish a held-out
                # speaker family; do not substitute an utterance identifier.
                continue
            family = source.source_id + ":" + str(speaker)
            split = frozen_split(family)
            if family in families or counts[split] >= desired[split]:
                continue
            translation = field_of(record, source.columns["translation"])
            transcript = field_of(record, source.columns["text"])
            blob = audio_blob(field_of(record, source.columns["audio"]))
            if not blob or not translation or not transcript:
                continue
            waveform, rate = sf.read(io.BytesIO(blob), dtype="float32", always_2d=True)
            if waveform.shape[1] != 1 or not rate // 2 <= len(waveform) <= rate * 20:
                continue
            identity = hashlib.sha256(blob).hexdigest()
            original = root / (identity + ".source-audio")
            original.write_bytes(blob)
            audio = root / (identity + ".wav")
            sf.write(audio, waveform, rate, subtype="FLOAT")
            # Exact samples, no full-clip features cropped after extraction.
            boundaries = [0, len(waveform) // 3, 2 * len(waveform) // 3, len(waveform)]
            chunks = []
            for j, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:], strict=True)):
                chunk = {
                    "start_sample": start,
                    "end_sample": end,
                    "channel": 0,
                    "speaker": family,
                    "available_at_ns": end * 1_000_000_000 // rate,
                }
                if j == 2:
                    chunk.update(
                        translation=str(translation),
                        translation_span={"start_sample": 0, "end_sample": len(waveform)},
                    )
                else:
                    chunk["wait_for_more"] = True
                chunks.append(chunk)
            rows.append(
                {
                    "recording_id": source.source_id + ":" + identity,
                    "source_id": source.source_id,
                    "family_id": family,
                    "revision": source.revision,
                    "split": split,
                    "audio": audio.name,
                    "audio_sha256": hashlib.sha256(audio.read_bytes()).hexdigest(),
                    "sample_rate": rate,
                    "channels": 1,
                    "chunks": chunks,
                    "streaming_translation": {
                        "policy": "segment_end",
                        "language": {"source": "fr", "target": "en"},
                    },
                }
            )
            receipts.append(
                {
                    "repository": source.repository,
                    "revision": source.revision,
                    "publisher_row": record["__key__"],
                    "publisher_id": str(field_of(record, source.columns["id"])),
                    "family_id": family,
                    "split": split,
                    "original_path": str(original),
                    "original_sha256": identity,
                    "decoded_path": str(audio),
                    "sample_count": len(waveform),
                    "sample_rate": rate,
                    "publisher_transcript": str(transcript),
                    "publisher_translation": str(translation),
                    "transformation": "lossy source decoded once to lossless float WAV; no resampling",
                    "scheduling_provenance": "authored segment-end policy; not recorded assistant behavior",
                }
            )
            families.add(family)
            counts[split] += 1
            print(json.dumps({"acquired_families": dict(counts)}), flush=True)
            if all(counts[key] >= count for key, count in desired.items()):
                break
    finally:
        iterator.close()
    manifest = root / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    write_json(root / "acquisition.json", {"sources": receipts, "families": dict(counts)})
    if any(counts[key] < count for key, count in desired.items()):
        raise ValueError(
            f"Acquisition did not establish requested original families: {dict(counts)}"
        )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--packs", type=Path, required=True)
    parser.add_argument("--codec", required=True)
    parser.add_argument("--train-families", type=int, default=16)
    parser.add_argument("--validation-families", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--manifest", type=Path, help="Resume from an already preserved acquisition"
    )
    args = parser.parse_args()
    if min(args.train_families, args.validation_families) < 1:
        parser.error("Both training and held-out original families are required")
    from megatoken.hub import install

    install()
    source = load_sources()["covost2"]
    manifest = args.manifest or acquire(
        source,
        args.source_root,
        train_families=args.train_families,
        validation_families=args.validation_families,
    )
    from megatoken.data.media_prepare import MediaPreparer
    from megatoken.data.timed_media import prepare_rows
    from megatoken.loading import load_codec

    if args.root.exists():
        raise ValueError("Preparation needs a new immutable output root")
    media = MediaPreparer(args.packs, args.root, device=args.device)
    report = prepare_rows(
        manifest,
        media=media,
        output=args.root,
        source_terms=source.terms,
        source_id=source.source_id,
        codec=load_codec(args.codec),
    )
    print(json.dumps(report, default=str), flush=True)


if __name__ == "__main__":
    main()
