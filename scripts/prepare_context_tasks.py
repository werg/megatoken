#!/usr/bin/env python3
"""Prepare authored context handoff, reread, retention and clarification tasks."""

from __future__ import annotations

import argparse
from pathlib import Path

from megatoken.data import DedupConfig, DedupStore
from megatoken.data.context_tasks import SOURCE_VERSION, context_episode
from megatoken.data.storage import export_prepared
from megatoken.io import write_json
from megatoken.loading import load_codec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--codec", required=True)
    parser.add_argument("--families", type=int, default=32)
    parser.add_argument("--local-events", type=int, default=4)
    parser.add_argument("--cross-events", type=int, default=1)
    args = parser.parse_args()
    if args.root.exists():
        raise FileExistsError("Context preparation requires a fresh output root")
    if args.families < 4 or args.local_events < 2 or not 0 <= args.cross_events < args.local_events:
        parser.error("Use at least four task families and a longer local window")
    codec = load_codec(args.codec)
    with DedupStore(
        args.root / "preparation.sqlite", DedupConfig(min_near_words=1_000_000)
    ) as store:
        for i in range(args.families):
            split = "test" if i % 8 == 0 else "validation" if i % 8 == 1 else "train"
            for variant in ("fetch", "pin", "local", "unavailable"):
                store.add(
                    context_episode(
                        i,
                        variant=variant,
                        split=split,
                        local_events=args.local_events,
                        cross_events=args.cross_events,
                    ),
                    codec,
                )
        store.freeze()
        report = export_prepared(
            store,
            args.root,
            codec=codec,
            source_receipts=[
                {
                    "source_id": SOURCE_VERSION,
                    "origin": "authored_executed",
                    "task_template": "remember_exact_project_key",
                }
            ],
            pipeline_fingerprint=f"{SOURCE_VERSION}:{args.local_events}:{args.cross_events}",
        )
    write_json(args.root / "context-preparation.json", report)


if __name__ == "__main__":
    main()
