#!/usr/bin/env python3
"""Write a separate, pinned Korean-field correction corpus and import receipt."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from megatoken.data.korean_correction import (
    EXPECTED_PINNED_ROWS,
    KOREAN_SOURCE_ID,
    build_correction_corpus,
)
from megatoken.data.sources import iter_source_rows, load_source_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=Path("configs/data/sources.agentic-chat.json"),
    )
    parser.add_argument("--source-id", default=KOREAN_SOURCE_ID)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/work/out/prepared/heegyu-glaive-ko-korean-correction-v1"),
    )
    parser.add_argument("--max-record-bytes", type=int, default=16_384)
    args = parser.parse_args()
    if args.max_record_bytes <= 0:
        parser.error("--max-record-bytes must be positive")
    if args.output.exists():
        parser.error("--output must be a new, separate correction directory")

    sources = load_source_manifest(args.source_manifest)
    matches = [source for source in sources if source.source_id == args.source_id]
    if len(matches) != 1:
        raise SystemExit(f"Expected one pinned source entry named {args.source_id!r}")
    source = matches[0]
    receipt = build_correction_corpus(
        iter_source_rows(source, start_row=0),
        original_source=source,
        output_dir=args.output,
        max_record_bytes=args.max_record_bytes,
        expected_input_rows=EXPECTED_PINNED_ROWS,
    )
    print(json.dumps(asdict(receipt), sort_keys=True))
    print(f"native source manifest: {args.output / 'source-manifest.json'}")
    print(f"correction receipt: {args.output / 'receipt.json'}")


if __name__ == "__main__":
    main()
