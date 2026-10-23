#!/usr/bin/env python3
"""Measure matched executed serial/delegated tasks and prepare cost-aware choices.

Both strategies execute real calls against isolated immutable snapshots. The
latency distribution is deliberately controlled and declared in the task; this
is an authored scheduling experiment, not evidence of learned autonomy.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

from megatoken.data.dedup import DedupConfig, DedupStore
from megatoken.data.dynamic_execution import SOURCE_VERSION, execution_episode
from megatoken.data.open_media import frozen_split
from megatoken.data.schema import canonical_json
from megatoken.data.storage import export_prepared
from megatoken.generation.codec import ByteCodec
from megatoken.io import write_json


def compare(root, *, families=32):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    # Warm validation/import paths outside the measured matched comparisons.
    execution_episode(987654, delegated=False)
    profiles = {
        "independent_slow_reads": {
            "read_delay_seconds": 0.04,
            "delegation_overhead_seconds": 0.001,
        },
        "independent_cheap_reads": {"read_delay_seconds": 0.0, "delegation_overhead_seconds": 0.02},
        "dependent_reads": {
            "read_delay_seconds": 0.01,
            "delegation_overhead_seconds": 0.01,
            "dependent": True,
        },
    }
    rows, examples, summaries = [], [], defaultdict(list)
    for index in range(families):
        for name, arguments in profiles.items():
            pairs = {}
            # Alternate ordering to expose warm-up/load bias in the archive.
            for delegated in (False, True) if index % 2 == 0 else (True, False):
                started = time.perf_counter()
                episode = execution_episode(index, delegated=delegated, **arguments)
                wall_seconds = time.perf_counter() - started
                split = frozen_split(episode.lineage.family_id)
                episode = replace(episode, lineage=replace(episode.lineage, frozen_split=split))
                details = json.loads(episode.supervision.annotations_json)
                details = {**details, "wall_elapsed_seconds": wall_seconds}
                pairs[delegated] = (episode, details)
                with (root / "executions.jsonl").open("a") as handle:
                    handle.write(canonical_json(episode.to_dict()) + "\n")
            serial, parallel = pairs[False][1], pairs[True][1]
            improvement = serial["wall_elapsed_seconds"] - parallel["wall_elapsed_seconds"]
            # Curriculum choice follows the publicly declared work/cost model,
            # not noisy timing or withheld answer contents. Matched measurements
            # independently test whether that choice actually helped.
            expected_delegation = not arguments.get("dependent", False) and (
                arguments["read_delay_seconds"] > 2 * arguments["delegation_overhead_seconds"]
            )
            chosen = pairs[expected_delegation][0]
            examples.append(chosen)
            row = {
                "family_id": chosen.lineage.family_id,
                "split": chosen.lineage.frozen_split,
                "profile": name,
                "serial_seconds": serial["wall_elapsed_seconds"],
                "delegated_seconds": parallel["wall_elapsed_seconds"],
                "delegation_seconds_saved": improvement,
                "timing_clock": "wall_perf_counter",
                "serial_schedule_seconds": serial["elapsed_seconds"],
                "delegated_schedule_seconds": parallel["elapsed_seconds"],
                "schedule_clock": serial.get("clock", "unspecified"),
                "serial_calls": len(pairs[False][0].actions),
                "delegated_calls": len(pairs[True][0].actions),
                "both_task_successes_independently_verified": all(
                    pairs[x][0].supervision.verified_success for x in pairs
                ),
                "supervised_choice_delegated": expected_delegation,
            }
            rows.append(row)
            summaries[name].append(improvement)
    codec = ByteCodec.byte_fallback()
    with DedupStore(
        root / "dedup.sqlite", DedupConfig(train_fraction=1, validation_fraction=0)
    ) as store:
        for episode in examples:
            store.add(episode, codec)
        store.freeze()
        export_prepared(
            store,
            root / "prepared",
            codec=codec,
            pipeline_fingerprint=SOURCE_VERSION + "-cost-aware-v1",
            source_receipts=[
                {
                    "origin": "authored_executed_controlled_latency",
                    "natural_timing": False,
                    "profiles": profiles,
                }
            ],
        )
    report = {
        "format": "delegation-comparison/v1",
        "matched_comparisons": rows,
        "means_seconds_saved_by_delegation": {k: sum(v) / len(v) for k, v in summaries.items()},
        "definition": "serial wall-clock seconds minus delegated wall-clock seconds; positive favors delegation",
        "resource_bounds": {"read_threads": 2, "max_active_tracks": 8, "shared_model_copies": 0},
        "learned_policy_evaluated": False,
        "model_compute_latency_included": False,
        "timing_clock": "wall_perf_counter",
        "modelled_schedule_times_used_as_wall_time": False,
    }
    write_json(root / "comparison.json", report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--families", type=int, default=32)
    args = p.parse_args()
    if args.families < 1:
        p.error("families must be positive")
    print(json.dumps(compare(args.root, families=args.families)))


if __name__ == "__main__":
    main()
