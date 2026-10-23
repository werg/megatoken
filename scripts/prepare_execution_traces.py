#!/usr/bin/env python3
"""Prepare reviewed, native full-duplex execution traces without fetching data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from megatoken.data.execution_traces import prepare_traces
from megatoken.data.sources import TermsReview
from megatoken.loading import load_codec


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", required=True, help="Native Episode JSONL")
    parser.add_argument(
        "--source-assets", required=True, help="Root holding referenced media/*.f32le assets"
    )
    parser.add_argument("--output", required=True, help="Prepared corpus output root")
    parser.add_argument("--terms", required=True, help="Reviewed source terms JSON object")
    parser.add_argument("--codec", required=True, help="Pinned tokenizer codec manifest")
    parser.add_argument("--source-id", help="Require this exact Episode lineage source ID")
    parser.add_argument("--stream", default="duplex_execution")
    parser.add_argument("--shard-bytes", type=int, default=128 << 20)
    args = parser.parse_args(argv)
    terms = TermsReview(**json.loads(Path(args.terms).read_text()))
    result = prepare_traces(
        args.episodes,
        source_assets=args.source_assets,
        output=args.output,
        source_terms=terms,
        codec=load_codec(args.codec),
        stream=args.stream,
        source_id=args.source_id,
        shard_bytes=args.shard_bytes,
    )
    print(json.dumps(result, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
