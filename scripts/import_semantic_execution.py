#!/usr/bin/env python3
"""Validate tracer semantic JSONL and emit native v3 Episode JSONL.

This is an explicit converter, not a corpus admission certifier. The private
tracer sidecar is never copied into a physical episode or a predictor control.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from megatoken.data.semantic_execution import SemanticExecutionError, adapt_semantic_execution
from megatoken.generation.codec import (
    LING_TOKENIZER_SHA256,
    ByteCodec,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Tracer diagnostic JSONL")
    parser.add_argument("--output", required=True, type=Path, help="Native v3 Episode JSONL")
    codec = parser.add_mutually_exclusive_group(required=True)
    codec.add_argument("--tokenizer-json", type=Path, help="Pinned production raw-byte codec")
    codec.add_argument(
        "--diagnostic-byte",
        action="store_true",
        help="Use the 1-byte debug codec (not valid for production admission)",
    )
    parser.add_argument("--allow-local", action="store_true", help="Allow unadmitted diagnostics")
    parser.add_argument("--max-positions", type=int, default=1024)
    parser.add_argument("--max-history", type=int, default=1024)
    args = parser.parse_args()
    selected_codec = (
        ByteCodec.byte_fallback()
        if args.diagnostic_byte
        else ByteCodec.from_tokenizer_json(
            args.tokenizer_json, expected_sha256=LING_TOKENIZER_SHA256
        )
    )
    if args.output == args.input:
        parser.error("Input and output must not be the same file")
    accepted = 0
    seen: set[str] = set()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with (
        args.input.open("r", encoding="utf-8") as source,
        args.output.open("w", encoding="utf-8") as sink,
    ):
        for index, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                document = json.loads(line)
                episode, receipt = adapt_semantic_execution(
                    document,
                    codec=selected_codec,
                    max_positions=args.max_positions,
                    max_history=args.max_history,
                    allow_local=args.allow_local,
                )
                if episode.episode_id in seen:
                    raise SemanticExecutionError("Duplicate normalized execution episode")
                if receipt.admission == "admitted":
                    # Signed independent admission receipts require an external
                    # certifier; the CLI has none and must fail closed.
                    raise SemanticExecutionError("CLI cannot certify admitted executions")
            except (SemanticExecutionError, ValueError, TypeError) as error:
                raise SystemExit(f"Rejected line {index}: {error}") from error
            seen.add(episode.episode_id)
            sink.write(json.dumps(episode.to_dict(), ensure_ascii=True) + "\n")
            accepted += 1
    print(
        json.dumps(
            {
                "format": "megatoken-episode-v3",
                "source": str(args.input),
                "output": str(args.output),
                "converted": accepted,
                "admitted": 0,
                "codec_fingerprint": selected_codec.fingerprint,
                "status": "diagnostic only; not independent admission",
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
