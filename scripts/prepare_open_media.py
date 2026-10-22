#!/usr/bin/env python3
"""Build the open-data media streams of one prepared root with the locked teacher and realizers.

Every source is a pinned public dataset listed, with its licence and the maintainer's recorded
approval, in ``configs/data/sources.media.json``; ``data/open_media.py`` reads it (streamed from
the pinned Hub revision, or from ``--inputs/<source_id>`` when that directory exists), turns its
labelled files into aligned bundles and compiles them into the stage plan's own streams:

  audio_identity, transcription            read and in-the-wild speech with transcripts
  verbalization                            text-to-speech corpora (written text, spoken form, voice sample)
  speech_translation                       speech with a text translation; FLEURS speech in two languages
  speech_edit, av_edit                     deterministic ffmpeg edits of the clips above, and real video edits
  image_identity, image_description,
  image_creation                           captioned photos and synthetic images
  image_edit                               instructed image edits
  video_identity, av_creation              captioned clips (with sound for av_creation)

``--count SOURCE=N`` draws N source clips from a source (FLEURS: N pairs across its languages);
``--procedural SOURCE=N`` adds a procedural edit for the first N of them; ``--option
SOURCE.KEY=JSON`` overrides a source option (for example the FLEURS languages); ``--scale``
multiplies every count and ``--shard-slice I/N`` reads only every N-th file from the I-th, so
N jobs can build disjoint parts of one source. Without ``--count`` the pilot sizes below are used.
Streams that no open dataset supplies are listed in docs/data.md.

Full volume (``megatoken.cloud_prepare`` with a media run file) gives each sandbox shards of
``data/media_sharding.py``. ``--shard-spec FILE`` names the sources, files and stripes of one
shard and reads all of them; with ``--ingest-only`` the run stops after deduplicating inside the
shard and leaves ``digest.sqlite`` (for ``megatoken reconcile``) beside the shard's
``preparation.sqlite`` and its media assets. The same call with ``--decisions FILE`` (that
shard's verdicts from ``reconcile``) and no reading applies the verdicts and writes the root.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from megatoken.data import DedupConfig, DedupStore
from megatoken.data.media_prepare import MediaPreparer
from megatoken.data.open_media import SOURCES, CachingPreparer, source_episodes
from megatoken.data.storage import export_prepared
from megatoken.loading import load_codec

PILOT_COUNTS = {
    "librispeech_asr": 2000,
    "libritts_r": 1500,
    "fleurs": 1600,
    "docci": 2000,
    "magicbrush": 1000,
    "rekadaily_processed": 600,
}
PILOT_PROCEDURAL = {"librispeech_asr": 600, "rekadaily_processed": 600}
PILOT_OPTIONS = {"fleurs": {"languages": ["de_de", "fr_fr", "es_419", "it_it"]}}
FULL = 1 << 62  # a shard reads every clip of its files


def _pairs(values: list[str], parser: argparse.ArgumentParser, flag: str) -> dict[str, str]:
    result = {}
    for value in values:
        key, separator, rest = value.partition("=")
        if not separator or not key:
            parser.error(f"{flag} needs NAME=VALUE, got {value!r}")
        result[key] = rest
    return result


def main() -> None:
    from megatoken.hub import install

    install()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packs", help="Locked pack manifest (not needed with --decisions)")
    parser.add_argument("--root", type=Path, required=True, help="Prepared dataset root (new)")
    parser.add_argument("--codec", required=True)
    parser.add_argument(
        "--work", type=Path, help="Scratch for the clips in flight (not needed with --decisions)"
    )
    parser.add_argument(
        "--inputs", type=Path, help="Local copies of sources: <inputs>/<source_id>/"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--count", action="append", default=[], metavar="SOURCE=N")
    parser.add_argument("--procedural", action="append", default=[], metavar="SOURCE=N")
    parser.add_argument("--option", action="append", default=[], metavar="SOURCE.KEY=JSON")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--shard-slice", default="0/1", metavar="I/N")
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--shard-spec", type=Path, help="JSON: this shard's sources and files")
    parser.add_argument("--ingest-only", action="store_true", help="stop at digest.sqlite")
    parser.add_argument("--decisions", type=Path, help="this shard's verdicts from 'reconcile'")
    args = parser.parse_args()
    if (args.root / "manifest.json").exists():
        parser.error("Prepared manifest already exists; prepared data is immutable")
    if (args.ingest_only or args.decisions) and not args.shard_spec:
        parser.error("--ingest-only and --decisions need --shard-spec")
    if args.ingest_only and args.decisions:
        parser.error("A shard is either ingested or finished with decisions, not both")
    if args.shard_spec and (args.count or args.procedural or args.shard_slice != "0/1"):
        parser.error("--shard-spec replaces --count, --procedural and --shard-slice")
    shard = json.loads(args.shard_spec.read_text()) if args.shard_spec else None
    if shard is not None:
        counts = {source_id: FULL for source_id in shard}
        procedural = {}
        options = {source_id: dict(entry.get("options", {})) for source_id, entry in shard.items()}
    elif args.count:
        counts = {key: int(value) for key, value in _pairs(args.count, parser, "--count").items()}
        procedural = {
            key: int(value)
            for key, value in _pairs(args.procedural, parser, "--procedural").items()
        }
        options: dict[str, dict] = {}
    else:
        counts, procedural = PILOT_COUNTS, PILOT_PROCEDURAL
        options = {source_id: dict(values) for source_id, values in PILOT_OPTIONS.items()}
    for name, value in _pairs(args.option, parser, "--option").items():
        source_id, _, key = name.partition(".")
        options.setdefault(source_id, {})[key] = json.loads(value)
    if shard is not None and not shard:
        parser.error("The shard spec names no sources")
    unknown = (set(counts) | set(procedural) | set(options)) - set(SOURCES)
    if unknown:
        parser.error(
            f"Unknown media sources {sorted(unknown)}; see configs/data/sources.media.json"
        )
    index, _, total = args.shard_slice.partition("/")
    shard_slice = (int(index), int(total))
    if not 0 <= shard_slice[0] < shard_slice[1]:
        parser.error("--shard-slice needs I/N with 0 <= I < N")
    sources = {
        source_id: SOURCES[source_id].with_options(**options.get(source_id, {}))
        for source_id in counts
    }
    for source in sources.values():
        source.terms.require_approval()
    if args.decisions is None and (args.packs is None or args.work is None):
        parser.error("--packs and --work are required to read the sources")
    if args.ingest_only:
        if (args.root / "digest.sqlite").exists():
            print(json.dumps({"digest": "already written"}))
            return
        # A retry starts the shard over: its assets are content addressed and kept.
        for suffix in ("", "-wal", "-shm", "-journal"):
            (args.root / f"preparation.sqlite{suffix}").unlink(missing_ok=True)
    codec = load_codec(args.codec)
    dedup = DedupStore(
        args.root / "preparation.sqlite",
        DedupConfig(
            train_fraction=1 - args.validation_fraction,
            validation_fraction=args.validation_fraction,
        ),
    )
    tally: Counter[str] = Counter()
    try:
        if args.decisions is None:
            media = CachingPreparer(MediaPreparer(args.packs, args.root, device=args.device))
            with dedup.transaction():
                for source_id, source in sources.items():
                    entry = shard[source_id] if shard is not None else {}
                    for episode in source_episodes(
                        media,
                        source,
                        count=round(counts[source_id] * args.scale),
                        procedural=round(procedural.get(source_id, 0) * args.scale),
                        procedural_every=entry.get("procedural_every", 0),
                        work=args.work,
                        inputs=args.inputs,
                        shard_slice=shard_slice,
                        files=entry.get("files"),
                        stripe=tuple(entry["stripe"]) if entry.get("stripe") else None,
                    ):
                        dedup.add(episode, codec)
                        tally[episode.stream] += 1
                        if sum(tally.values()) % 100 == 0:
                            print(json.dumps(tally, sort_keys=True), flush=True)
                    print(
                        json.dumps({"finished": source_id, "streams": tally}, sort_keys=True),
                        flush=True,
                    )
            if args.ingest_only:
                print(json.dumps({"digest": dedup.write_digest(args.root / "digest.sqlite")}))
                return
            dedup.freeze()
        else:
            dedup.apply_decisions(args.decisions)
        export_prepared(
            dedup,
            args.root,
            codec=codec,
            source_receipts=[
                {
                    "name": "open_media",
                    "sources": {key: asdict(value) for key, value in sources.items()},
                    "counts": dict(tally),
                }
            ],
            pipeline_fingerprint="open-media-v2",
        )
    finally:
        dedup.close()
    print(json.dumps(tally, sort_keys=True))


if __name__ == "__main__":
    main()
