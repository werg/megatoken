#!/usr/bin/env python3
"""Collect genuine frozen-student attempts and optimize their verified outcomes.

This bounded acceptance run tests the normal collector, native replay, optimizer,
restart and export. It does not establish task competence or generalization.
"""

from __future__ import annotations

import argparse
import gc
import json
from dataclasses import asdict, replace
from pathlib import Path

from megatoken.data import DedupConfig, DedupStore
from megatoken.data.chunking import ChunkingConfig
from megatoken.data.collection import CollectionConfig, CollectionLedger, Collector, TaskSeed
from megatoken.data.schema import canonical_json, sequential_meridians
from megatoken.data.storage import export_prepared
from megatoken.data.student import ModelTeacher
from megatoken.environments import FIXTURE_TOOLS, FixtureEnvironment
from megatoken.io import sha256_file, write_json
from megatoken.served import export_served
from megatoken.training.config import TrainConfig
from megatoken.training.engine import train


class _UnusedTeacher:
    model = "student-only-no-teacher"

    def respond(self, request):
        raise AssertionError("Student-only collection must never call a teacher")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tasks", type=int, default=8)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--requests", type=int, default=3)
    args = parser.parse_args()
    if min(args.tasks, args.steps, args.requests) < 1:
        raise ValueError("Task, update and request bounds must be positive")
    args.root.mkdir(parents=True, exist_ok=False)
    student = ModelTeacher(args.bundle, device=args.device, effort=1, seed=37)
    codec, snapshot = student.codec, student.snapshot
    tasks = [
        TaskSeed(
            f"outcome-{i}",
            f"outcome-family-{i}",
            "terminal",
            f"Set answer.txt to blue-{i}, then finish.".encode(),
            canonical_json(
                {
                    "initial_files": {"answer.txt": f"red-{i}"},
                    "expected_files": {"answer.txt": f"blue-{i}"},
                }
            ),
            canonical_json(FIXTURE_TOOLS),
            "executed-outcome-acceptance-v1",
        )
        for i in range(args.tasks)
    ]
    collection = CollectionConfig(
        distinct_tasks=args.tasks,
        attempts_per_task=1,
        max_requests_per_attempt=args.requests,
        max_output_per_request=128,
        seed_mix=(("terminal", 1.0),),
    )
    outcomes, decisions = [], 0
    try:
        with (
            DedupStore(
                args.root / "families.sqlite",
                DedupConfig(train_fraction=1, validation_fraction=0),
            ) as families,
            CollectionLedger(args.root / "collection.sqlite", collection) as ledger,
            DedupStore(
                args.root / "spool.sqlite",
                DedupConfig(train_fraction=1, validation_fraction=0),
            ) as prepared,
        ):
            with families.transaction():
                for task in tasks:
                    families.register_family((task.family_id,))
            families.freeze()
            collector = Collector(_UnusedTeacher(), FixtureEnvironment, codec, ledger, families)
            for task in tasks:
                episode = collector.collect(
                    task, student=student, student_snapshot=snapshot, student_only=True
                )
                if not episode.meridians:
                    episode = sequential_meridians(episode)
                entries = json.loads(episode.supervision.annotations_json)["native_action_tapes"]
                decisions += sum("outcome_credit" in entry for entry in entries)
                outcomes.append(episode.supervision.outcome)
                with prepared.transaction():
                    prepared.add(episode, codec)
            prepared.freeze()
            report = export_prepared(
                prepared,
                args.root / "data",
                codec=codec,
                chunking=ChunkingConfig().with_hard_limit(512, max_bytes=2048),
                pipeline_fingerprint="executed-outcome-acceptance-v1",
                source_receipts=[{"student_snapshot": snapshot, "origin": "executed_student"}],
            )
            ledger.export(args.root / "collection")
    finally:
        student.runtime.close()
    del student
    gc.collect()
    if not decisions:
        raise RuntimeError("No independently verified native policy decisions were collected")
    # Use the frozen actor's ordinary training recipe and checkpoint as the
    # starting point; a fresh initialization would make stale-policy ratios useless.
    actor_checkpoint = args.bundle.parent / "training/checkpoints/latest.json"
    actor_config = TrainConfig.load(args.bundle.parent / "train-config.json")
    config = replace(
        actor_config,
        phase="D",
        datasets=[str(args.root / "data")],
        output=str(args.root / "training"),
        device=args.device,
        max_steps=args.steps,
        eval_every=0,
        accumulate_events=1,
        use_stage_mixture=False,
    )
    write_json(args.root / "train-config.json", asdict(config))
    first = train(config, resume=str(actor_checkpoint), phase_transition=True)
    gc.collect()
    checkpoint = args.root / "training/checkpoints/latest.json"
    restarted = train(replace(config, max_steps=2), resume=str(checkpoint))
    exported = export_served(checkpoint, args.root / "served", codec_path=config.codec)
    write_json(
        args.root / "acceptance.json",
        {
            "script_sha256": sha256_file(__file__),
            "student_snapshot": snapshot,
            "tasks": args.tasks,
            "outcomes": outcomes,
            "verified_selected_decisions": decisions,
            "prepared": report,
            "training": first,
            "restart": restarted,
            "export": exported,
            "quality_established": False,
        },
    )


if __name__ == "__main__":
    main()
