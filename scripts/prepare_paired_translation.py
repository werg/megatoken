#!/usr/bin/env python3
"""Prepare natural paired-language speech for joint and sequential translation."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import soundfile as sf

from megatoken.data.open_media import Shards, frozen_split, load_sources, read_fleurs
from megatoken.io import sha256_file, write_json


def acquire(source, root, *, train_families, validation_families):
    root.mkdir(parents=True, exist_ok=False)
    source.terms.require_approval()
    source = source.with_options(languages=["fr_fr"])
    desired = {"train": train_families, "validation": validation_families}
    counts, rows, receipts = Counter(), [], []
    pairs = read_fleurs(source, Shards(source), root / "originals", 128)
    try:
        for pair in pairs:
            split = frozen_split(pair["family"])
            if counts[split] >= desired[split]:
                continue
            incoming, outgoing = Path(pair["source_audio"]), Path(pair["target_audio"])
            si, ti = sf.info(incoming), sf.info(outgoing)
            if si.channels != 1 or ti.channels != 1 or si.frames < 3:
                continue
            boundaries = [0, si.frames // 3, 2 * si.frames // 3, si.frames]
            chunks = []
            for j, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:], strict=True)):
                chunk = {
                    "start_sample": start,
                    "end_sample": end,
                    "channel": 0,
                    "available_at_ns": end * 1_000_000_000 // si.samplerate,
                }
                if j < 2:
                    chunk["wait_for_more"] = True
                else:
                    chunk.update(
                        translation=pair["target_text"],
                        translation_span={"start_sample": 0, "end_sample": si.frames},
                        translated_speech={
                            "path": str(outgoing.relative_to(root)),
                            "start_sample": 0,
                            "end_sample": ti.frames,
                            "channel": 0,
                            "sample_rate": ti.samplerate,
                            "provenance": "natural",
                        },
                    )
                chunks.append(chunk)
            rows.append(
                {
                    "recording_id": f"fleurs:fr_fr:{pair['id']}",
                    "source_id": "fleurs",
                    "family_id": pair["family"],
                    "revision": source.revision,
                    "split": split,
                    "audio": str(incoming.relative_to(root)),
                    "audio_sha256": sha256_file(incoming),
                    "sample_rate": si.samplerate,
                    "channels": 1,
                    "chunks": chunks,
                    "streaming_translation": {
                        "policy": "segment_end",
                        "language": {"source": "fr", "target": "en"},
                        "output_modalities": ["text", "audio"],
                    },
                }
            )
            receipts.append(
                {
                    "publisher_id": pair["id"],
                    "family_id": pair["family"],
                    "split": split,
                    "source_audio": str(incoming),
                    "target_audio": str(outgoing),
                    "source_sha256": sha256_file(incoming),
                    "target_sha256": sha256_file(outgoing),
                    "publisher_transcript": pair["source_text"],
                    "publisher_translation": pair["target_text"],
                    "speaker_identity": "not supplied by publisher; no speaker-disjoint claim",
                    "repository": source.repository,
                    "revision": source.revision,
                }
            )
            counts[split] += 1
            print(json.dumps({"acquired_paired_families": dict(counts)}), flush=True)
            if all(counts[k] >= v for k, v in desired.items()):
                break
    finally:
        pairs.close()
    manifest = root / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    write_json(root / "acquisition.json", {"sources": receipts, "families": dict(counts)})
    if any(counts[k] < v for k, v in desired.items()):
        raise ValueError(f"Paired acquisition lacks requested original families: {dict(counts)}")
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--packs", type=Path, required=True)
    p.add_argument("--train-families", type=int, default=8)
    p.add_argument("--validation-families", type=int, default=2)
    p.add_argument("--device", default="cuda")
    p.add_argument("--manifest", type=Path)
    args = p.parse_args()
    if min(args.train_families, args.validation_families) < 1:
        p.error("Both training and held-out original families are required")
    from megatoken.hub import install

    install()
    source = load_sources()["fleurs"]
    manifest = args.manifest or acquire(
        source,
        args.source_root,
        train_families=args.train_families,
        validation_families=args.validation_families,
    )
    from megatoken.data.media_prepare import MediaPreparer
    from megatoken.data.timed_media import prepare_rows
    from megatoken.generation.codec import ByteCodec

    if args.root.exists():
        raise ValueError("Preparation needs a new immutable output root")
    report = prepare_rows(
        manifest,
        media=MediaPreparer(args.packs, args.root, device=args.device),
        output=args.root,
        source_terms=source.terms,
        source_id=source.source_id,
        codec=ByteCodec.byte_fallback(),
    )
    print(json.dumps(report, default=str), flush=True)


if __name__ == "__main__":
    main()
