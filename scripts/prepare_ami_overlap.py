#!/usr/bin/env python3
"""Prepare naturally overlapping AMI speakers with disjoint held-out meetings.

Word times and microphone/participant identities come from the publisher's
annotation archive. No assistant turn timing or audiovisual offsets are inferred.
"""

from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import numpy as np
import soundfile as sf

from megatoken.data.fetch import download
from megatoken.io import sha256_file, write_json


def utterances(archive, meeting, agent):
    root = ET.fromstring(archive.read(f"words/{meeting}.{agent}.words.xml"))
    groups = []
    current = []
    for word in root:
        if word.tag != "w" or not word.get("starttime") or not word.get("endtime"):
            continue
        start, end = float(word.get("starttime")), float(word.get("endtime"))
        text = "".join(word.itertext()).strip()
        if not text or end <= start:
            continue
        if current and (start - current[-1][1] > 0.3 or end - current[0][0] > 6):
            groups.append(current)
            current = []
        current.append((start, end, text))
    if current:
        groups.append(current)
    return [
        (g[0][0], g[-1][1], " ".join(w[2] for w in g))
        for g in groups
        if 0.5 <= g[-1][1] - g[0][0] <= 6
    ]


def acquire(
    source_root,
    annotations,
    *,
    pairs=4,
    selected_meetings=(("ES2002a", "train"), ("ES2003a", "validation")),
):
    if (source_root / "manifest.jsonl").exists():
        raise ValueError("Use --manifest to resume an already committed acquisition")
    source_root.mkdir(parents=True, exist_ok=True)
    names = [meeting for meeting, _ in selected_meetings]
    if len(names) != len(set(names)) or {split for _, split in selected_meetings} != {
        "train",
        "validation",
    }:
        raise ValueError("Declare distinct original meetings with both training and validation")
    rows, receipts, seen_speakers = [], [], set()
    with ZipFile(annotations) as archive, ThreadPoolExecutor(max_workers=4) as pool:
        meetings = ET.fromstring(archive.read("corpusResources/meetings.xml"))
        pending = {
            (meeting, channel): pool.submit(
                download,
                f"https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/{meeting}/audio/{meeting}.Headset-{channel}.wav",
                source_root / f"{meeting}.Headset-{channel}.wav",
            )
            for meeting, _ in selected_meetings
            for channel in ("1", "3")
        }
        for meeting, split in selected_meetings:
            meta = next(m for m in meetings if m.get("observation") == meeting)
            speakers = sorted(
                (s for s in meta if s.get("channel") in {"1", "3"}),
                key=lambda s: int(s.get("channel")),
            )
            if len(speakers) != 2 or seen_speakers & {s.get("global_name") for s in speakers}:
                raise ValueError(
                    "Held-out meetings must use disjoint publisher participant identities"
                )
            seen_speakers.update(s.get("global_name") for s in speakers)
            files = []
            for speaker in speakers:
                filename = f"{meeting}.Headset-{speaker.get('channel')}.wav"
                url = f"https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/{meeting}/audio/{filename}"
                path = pending[(meeting, speaker.get("channel"))].result()
                files.append(path)
                receipts.append(
                    {
                        "url": url,
                        "path": str(path),
                        "sha256": sha256_file(path),
                        "speaker": speaker.get("global_name"),
                        "meeting": meeting,
                    }
                )
                print(json.dumps({"downloaded": filename}), flush=True)
            infos = [sf.info(p) for p in files]
            if infos[0].samplerate != infos[1].samplerate or any(i.channels != 1 for i in infos):
                raise ValueError("Original microphone clocks or channel declarations differ")
            rate, frames = infos[0].samplerate, min(i.frames for i in infos)
            combined = source_root / f"{meeting}.channels-1-3.wav"
            with (
                sf.SoundFile(files[0]) as a,
                sf.SoundFile(files[1]) as b,
                sf.SoundFile(
                    combined, mode="w", samplerate=rate, channels=2, subtype="FLOAT"
                ) as output,
            ):
                for offset in range(0, frames, 8192):
                    count = min(8192, frames - offset)
                    output.write(
                        np.stack(
                            (a.read(count, dtype="float32"), b.read(count, dtype="float32")),
                            axis=-1,
                        )
                    )
            segments = [utterances(archive, meeting, s.get("nxt_agent")) for s in speakers]
            candidates = sorted(
                (
                    (min(a[1], b[1]) - max(a[0], b[0]), a, b)
                    for a in segments[0]
                    for b in segments[1]
                    if min(a[1], b[1]) - max(a[0], b[0]) >= 0.05
                ),
                reverse=True,
            )
            accepted = []
            for _overlap, left, right in candidates:
                if any(
                    max(a[0], b[0]) < min(a[1], b[1])
                    for pair in accepted
                    for a, b in zip((left, right), pair, strict=True)
                ):
                    continue
                if max(left[1], right[1]) * rate > frames:
                    continue
                accepted.append((left, right))
                if len(accepted) == pairs:
                    break
            if len(accepted) < pairs:
                raise ValueError(
                    f"{meeting} lacks the requested independently timed overlap intervals"
                )
            for index, pair in enumerate(accepted):
                chunks = []
                for channel, (segment, speaker) in enumerate(zip(pair, speakers, strict=True)):
                    start, end, text = segment
                    chunks.append(
                        {
                            "start_sample": math.floor(start * rate),
                            "end_sample": math.ceil(end * rate),
                            "speaker": speaker.get("global_name"),
                            "channel": channel,
                            "transcript": text,
                        }
                    )
                rows.append(
                    {
                        "recording_id": f"ami:{meeting}:annotated-overlap:{index}",
                        "source_id": "ami-original-signals",
                        "family_id": f"ami:{meeting[:-1]}",
                        "revision": "ami_public_manual_1.6.2:" + sha256_file(annotations),
                        "split": split,
                        "audio": str(combined),
                        "audio_sha256": sha256_file(combined),
                        "sample_rate": rate,
                        "channels": 2,
                        "chunks": sorted(chunks, key=lambda c: (c["start_sample"], c["channel"])),
                        "original_source_files": [
                            {
                                "path": str(p),
                                "sha256": sha256_file(p),
                                "role": "original_microphone",
                            }
                            for p in files
                        ],
                        "transformation": "original microphone channels copied without resampling into a lossless float WAV; common tail only",
                        "annotations_sha256": sha256_file(annotations),
                    }
                )
    manifest = source_root / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(r) + "\n" for r in rows))
    write_json(
        source_root / "acquisition.json",
        {
            "sources": receipts,
            "annotation_archive_sha256": sha256_file(annotations),
            "assistant_timing_inferred": False,
            "audiovisual_offset_inferred": False,
        },
    )
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--annotations", type=Path, required=True)
    p.add_argument("--terms", type=Path, required=True)
    p.add_argument("--packs", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--manifest", type=Path)
    p.add_argument(
        "--pairs",
        type=int,
        default=1,
        help="Disjoint annotated overlap windows per original meeting",
    )
    p.add_argument(
        "--meeting",
        action="append",
        metavar="MEETING:SPLIT",
        help="Publisher meeting and explicit train or validation split; repeat for each meeting",
    )
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    if args.pairs < 1:
        p.error("At least one genuine overlapping interval is required")
    from megatoken.data.sources import TermsReview

    terms = json.loads(args.terms.read_text())
    TermsReview(**terms).require_approval()
    selected_meetings = (("ES2002a", "train"), ("ES2003a", "validation"))
    if args.meeting:
        selected_meetings = tuple(tuple(value.rsplit(":", 1)) for value in args.meeting)
        if any(len(value) != 2 for value in selected_meetings):
            p.error("Each meeting requires MEETING:train or MEETING:validation")
    manifest = args.manifest or acquire(
        args.source_root, args.annotations, pairs=args.pairs, selected_meetings=selected_meetings
    )
    from megatoken.data.media_prepare import MediaPreparer
    from megatoken.data.timed_media import prepare_rows
    from megatoken.generation.codec import ByteCodec

    if args.root.exists():
        raise ValueError("Preparation needs a new immutable output root")
    print(
        json.dumps(
            prepare_rows(
                manifest,
                media=MediaPreparer(args.packs, args.root, device=args.device),
                output=args.root,
                source_terms=terms,
                source_id="ami-original-signals",
                codec=ByteCodec.byte_fallback(),
            ),
            default=str,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
