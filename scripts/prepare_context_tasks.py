#!/usr/bin/env python3
"""Prepare authored context handoff, reread, retention and clarification tasks."""

from __future__ import annotations

import argparse
from pathlib import Path

from megatoken.data import DedupConfig, DedupStore, PreparedDataset
from megatoken.data import context_curriculum as producer
from megatoken.data.context_curriculum import (
    SOURCE_VERSION,
    TEMPLATES,
    WINDOW_PROFILES,
    curriculum_episodes,
    curriculum_summary,
)
from megatoken.data.storage import export_prepared
from megatoken.io import content_hash, sha256_file, write_json
from megatoken.loading import load_codec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--codec", required=True)
    parser.add_argument(
        "--families",
        type=int,
        default=32,
        help="Original tasks per template; all variants keep the same split",
    )
    parser.add_argument(
        "--local-events", type=int, help="Use one custom profile instead of the curriculum"
    )
    parser.add_argument("--cross-events", type=int)
    args = parser.parse_args()
    if args.root.exists():
        raise FileExistsError("Context preparation requires a fresh output root")
    if args.families < 8:
        parser.error("Use at least eight tasks per template so every split contains every template")
    profiles = WINDOW_PROFILES
    if (args.local_events is None) != (args.cross_events is None):
        parser.error("Set both custom windows together")
    if args.local_events is not None:
        if args.local_events < 16 or not 8 <= args.cross_events < args.local_events:
            parser.error(
                "Use at least sixteen local events and at least eight, but fewer, foreign events"
            )
        profiles = ((args.local_events, args.cross_events),)
    codec = load_codec(args.codec)
    producer_sha256 = sha256_file(Path(producer.__file__))
    with DedupStore(
        args.root / "preparation.sqlite", DedupConfig(min_near_words=1_000_000)
    ) as store:
        episodes = list(curriculum_episodes(args.families, profiles=profiles))
        with store.transaction():
            for episode in episodes:
                store.add(episode, codec)
        store.freeze()
        report = export_prepared(
            store,
            args.root,
            codec=codec,
            source_receipts=[
                {
                    "source_id": SOURCE_VERSION,
                    "revision": SOURCE_VERSION,
                    "producer_sha256": producer_sha256,
                    "origin": "authored_executed",
                    "task_templates": list(TEMPLATES),
                    "window_profiles": [list(profile) for profile in profiles],
                    "split_unit": "original_task_before_strategy_and_window_variants",
                }
            ],
            pipeline_fingerprint=content_hash(
                {
                    "source": SOURCE_VERSION,
                    "producer_sha256": producer_sha256,
                    "profiles": profiles,
                    "families": args.families,
                }
            ),
        )
    with PreparedDataset(args.root, split=None) as admitted:
        report["context_curriculum"] = curriculum_summary(admitted.iter_episodes())
    report["context_curriculum"]["generated_examples"] = len(episodes)
    write_json(args.root / "context-preparation.json", report)


if __name__ == "__main__":
    main()
