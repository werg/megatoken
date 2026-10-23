#!/usr/bin/env python3
"""Prepare timed chunks from an explicitly supplied local JSONL manifest."""

import argparse
import json
from pathlib import Path

from megatoken.data.media_prepare import MediaPreparer
from megatoken.data.timed_media import prepare_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, required=True, help="Local JSONL annotations and media paths"
    )
    parser.add_argument(
        "--packs", type=Path, required=True, help="Locked teacher and realizer pack manifest"
    )
    parser.add_argument("--root", type=Path, required=True, help="New output and tensor asset root")
    parser.add_argument(
        "--codec", type=Path, required=True, help="Training text codec for deduplication and export"
    )
    terms = parser.add_mutually_exclusive_group(required=True)
    terms.add_argument("--source-id", help="Approved source in configs/data source catalog")
    terms.add_argument(
        "--terms-json", type=Path, help="Explicit reviewed TermsReview JSON for a local source"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--pilot", action="store_true", help="Prepare at most the first two recordings"
    )
    args = parser.parse_args()
    if (args.root / "episodes.jsonl").exists() or (args.root / "source-receipts.json").exists():
        parser.error("Output already contains a committed timed-media preparation")
    from megatoken.loading import load_codec

    codec = load_codec(args.codec)
    if args.source_id:
        from megatoken.data.open_media import load_sources

        catalog = load_sources()
        if args.source_id not in catalog:
            parser.error(f"Unknown approved source ID: {args.source_id}")
        source = catalog[args.source_id]
        source.terms.require_approval()
        source_terms = source.terms
    else:
        from megatoken.data.sources import TermsReview

        try:
            source_terms = TermsReview(**json.loads(args.terms_json.read_text()))
            source_terms.require_approval()
        except (TypeError, ValueError) as exc:
            parser.error(f"Invalid or unapproved source terms record: {exc}")
    media = MediaPreparer(args.packs, args.root, device=args.device)
    prepare_rows(
        args.manifest,
        media=media,
        output=args.root,
        source_terms=source_terms,
        source_id=args.source_id,
        max_recordings=2 if args.pilot else None,
        codec=codec,
    )


if __name__ == "__main__":
    main()
