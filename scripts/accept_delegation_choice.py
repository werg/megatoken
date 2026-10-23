#!/usr/bin/env python3
"""Learn whether a bounded, actually executed tool program should delegate reads.

The model chooses between two public procedures. Their inner read/write/lifecycle
programs are fixed and independently verified. This isolates scheduling learning
from general language/arithmetic competence; it is not free-form agent success.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from dataclasses import asdict, replace
from pathlib import Path

from megatoken.data import DedupConfig, DedupStore, Episode, Lineage, LogicalAction, PhysicalRecord
from megatoken.data.dynamic_execution import execution_episode, task_definition
from megatoken.data.masking import apply_tool_masks
from megatoken.data.open_media import frozen_split
from megatoken.data.schema import Supervision, canonical_json, digest_bytes, sequential_meridians
from megatoken.data.storage import PreparedDataset, export_prepared
from megatoken.data.student import ModelTeacher
from megatoken.environments import EnvironmentResult
from megatoken.generation.codec import ByteCodec
from megatoken.io import write_json
from megatoken.tool_schema import parse_tool_action

TOOLS = [
    {
        "name": name,
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    }
    for name in ("serial_reads", "delegated_reads")
]
PROFILES = {
    "independent_slow_reads": {"read_delay_seconds": 0.04, "delegation_overhead_seconds": 0.001},
    "independent_cheap_reads": {"read_delay_seconds": 0.0, "delegation_overhead_seconds": 0.02},
    "dependent_reads": {
        "read_delay_seconds": 0.01,
        "delegation_overhead_seconds": 0.01,
        "dependent": True,
    },
}


class StrategyEnvironment:
    def reset(self, task):
        self.index, self.profile = task["index"], task["profile"]
        self.state_id = uuid.uuid4().hex
        self.done = False
        self.receipt = None
        options = dict(PROFILES[self.profile])
        scale = 0.8 + (self.index % 17) / 40
        for key in ("read_delay_seconds", "delegation_overhead_seconds"):
            options[key] *= scale
        self.options = options
        self.baselines = {}
        self.schedule_baselines = {}
        for delegated in (False, True):
            started = time.perf_counter()
            episode = execution_episode(self.index, delegated=delegated, **options)
            self.baselines[delegated] = time.perf_counter() - started
            self.schedule_baselines[delegated] = json.loads(episode.supervision.annotations_json)[
                "elapsed_seconds"
            ]
        self.best = min(self.baselines, key=self.baselines.get)
        public_task = task_definition(self.index, dependent=options.get("dependent", False))
        prompt = canonical_json(
            {
                "program": public_task["prompt"],
                "initial_files": public_task["initial_files"],
                "costs": options,
                "procedures": "serial_reads: coordinator; delegated_reads: two bounded workers",
            }
        )
        return EnvironmentResult(prompt.encode(), self.state_id)

    def step(self, action, *, action_id, expected_state_id):
        if expected_state_id != self.state_id or self.done:
            raise ValueError("Strategy action requires its exact available state")
        name, arguments, _, _ = parse_tool_action(action)
        if arguments or name not in {t["name"] for t in TOOLS}:
            raise ValueError("Only the public, argument-free read procedures are supported")
        delegated = name == "delegated_reads"
        started = time.perf_counter()
        actual = execution_episode(self.index, delegated=delegated, **self.options)
        elapsed = time.perf_counter() - started
        self.receipt = {
            "procedure": name,
            "task_verified": actual.supervision.verified_success,
            "cost_choice_correct": delegated == self.best,
            "program_elapsed_seconds": elapsed,
            "timing_clock": "wall_perf_counter",
            "modelled_schedule_seconds": json.loads(actual.supervision.annotations_json)[
                "elapsed_seconds"
            ],
            "measured_reference_seconds": {str(k): v for k, v in self.baselines.items()},
            "executed_program": actual.to_dict(),
            "selected_payload_sha256": digest_bytes(action),
        }
        self.done = True
        self.execution_id = f"strategy:{self.state_id}:{action_id}"
        return EnvironmentResult(
            b'{"program_completed":true}', self.state_id, self.execution_id, done=True
        )

    def verify(self):
        success = (
            self.receipt is not None
            and self.receipt["task_verified"]
            and self.receipt["cost_choice_correct"]
        )
        return EnvironmentResult(
            b'{"verification_complete":true}',
            self.state_id,
            done=True,
            verified=True,
            reward=1.0 if success else -1.0,
        )


def prepare(root, *, families):
    root.mkdir(parents=True, exist_ok=False)
    codec = ByteCodec.byte_fallback()
    episode_log = []
    with DedupStore(
        root / "dedup.sqlite", DedupConfig(train_fraction=1, validation_fraction=0)
    ) as store:
        for index in range(families):
            family = f"delegation-choice:{index}"
            for profile in PROFILES:
                environment = StrategyEnvironment()
                observation = environment.reset({"index": index, "profile": profile})
                name = "delegated_reads" if environment.best else "serial_reads"
                payload = canonical_json({"name": name, "arguments": {}}).encode()
                result = environment.step(
                    payload, action_id="selected", expected_state_id=observation.state_id
                )
                verdict = environment.verify()
                if verdict.reward != 1:
                    raise ValueError("Authored measured scheduling choice did not verify")
                records = (
                    PhysicalRecord(
                        "request",
                        "user",
                        b"Choose the faster valid procedure from the public costs.",
                    ),
                    PhysicalRecord("costs", "environment", observation.observation),
                    PhysicalRecord(
                        "choice", "assistant", payload, policy_mask=True, action_id="selected"
                    ),
                    PhysicalRecord(
                        "result",
                        "tool",
                        result.observation,
                        result_of="selected",
                        execution_id=result.execution_id,
                    ),
                )
                episode = Episode(
                    f"choice:{index}:{profile}",
                    records,
                    Lineage(
                        "measured-delegation-choice",
                        "v1",
                        family,
                        frozen_split=frozen_split(family),
                    ),
                    actions=(
                        LogicalAction(
                            "selected",
                            "selected",
                            name,
                            "{}",
                            ("choice",),
                            digest_bytes(payload),
                            result.execution_id,
                        ),
                    ),
                    tools_json=canonical_json(TOOLS),
                    stream="dynamic_execution",
                    supervision=Supervision(
                        outcome="success",
                        verification="local_assertions",
                        reward=1.0,
                        annotations_json=canonical_json(
                            {
                                "scope": "cost choice over fixed executed programs",
                                "profile": profile,
                            }
                        ),
                    ),
                )
                episode = apply_tool_masks(sequential_meridians(episode))
                store.add(episode, codec)
                episode_log.append(
                    {
                        "family": family,
                        "split": episode.lineage.frozen_split,
                        "profile": profile,
                        "reference_seconds": environment.receipt["measured_reference_seconds"],
                        "chosen": name,
                        "program_verified": True,
                    }
                )
        store.freeze()
        export_prepared(
            store,
            root,
            codec=codec,
            source_receipts=[
                {"origin": "authored_measured_cost_choices", "natural_conversation": False}
            ],
            pipeline_fingerprint="measured-delegation-choice-v1",
        )
    write_json(root / "measurements.json", episode_log)


def evaluate(bundle_path, data_path, output):
    from megatoken.data.collection import CollectionConfig, CollectionLedger, Collector, TaskSeed

    class UnusedTeacher:
        model = "none"

        def respond(self, _):
            raise AssertionError("Evaluation must remain entirely student-only")

    student = ModelTeacher(bundle_path, device="cuda", effort=1, seed=39, allow_text=False)
    tasks = []
    with PreparedDataset(data_path, split="validation") as data:
        for episode in data.iter_episodes():
            _, index, profile = episode.episode_id.split(":", 2)
            tasks.append(
                TaskSeed(
                    episode.episode_id,
                    episode.lineage.family_id,
                    "terminal",
                    b"Choose the faster valid procedure from the public costs.",
                    canonical_json({"index": int(index), "profile": profile}),
                    canonical_json(TOOLS),
                    "measured-delegation-choice-v1",
                )
            )
    outcomes = []
    executions = []
    environments = []
    decision_times = []
    original_respond = student.respond

    def timed_respond(request):
        started = time.perf_counter()
        try:
            return original_respond(request)
        finally:
            decision_times.append(time.perf_counter() - started)

    student.respond = timed_respond

    def environment_factory():
        environment = StrategyEnvironment()
        environments.append(environment)
        return environment

    config = CollectionConfig(
        distinct_tasks=len(tasks),
        attempts_per_task=1,
        max_requests_per_attempt=1,
        max_output_per_request=128,
        chunking=student.runtime.chunking,
        max_record_tokens=student.runtime.chunking.hard_max_tokens,
        max_record_bytes=student.runtime.chunking.max_bytes,
        seed_mix=(("terminal", 1.0),),
    )
    try:
        with (
            DedupStore(
                output / "families.sqlite", DedupConfig(train_fraction=0, validation_fraction=1)
            ) as families,
            CollectionLedger(output / "collection.sqlite", config) as ledger,
        ):
            for task in tasks:
                families.register_family((task.family_id,))
            families.freeze()
            collector = Collector(
                UnusedTeacher(), environment_factory, student.codec, ledger, families
            )
            for task in tasks:
                episode = collector.collect(
                    task, student=student, student_snapshot=student.snapshot, student_only=True
                )
                outcomes.append(episode.supervision.outcome)
                environment = environments[-1]
                receipt = environment.receipt
                executions.append(
                    {
                        "task": task.seed_id,
                        "profile": environment.profile,
                        "outcome": episode.supervision.outcome,
                        "procedure": None if receipt is None else receipt["procedure"],
                        "cost_choice_correct": False
                        if receipt is None
                        else receipt["cost_choice_correct"],
                        "program_verified": False if receipt is None else receipt["task_verified"],
                        "program_elapsed_seconds": None
                        if receipt is None
                        else receipt["program_elapsed_seconds"],
                        "serial_reference_seconds": environment.baselines[False],
                        "delegated_reference_seconds": environment.baselines[True],
                        "model_decision_seconds": decision_times[-1],
                        "serial_is_best": not environment.best,
                    }
                )
            ledger.export(output / "collection")
    finally:
        student.runtime.close()
    success_rate = outcomes.count("success") / len(outcomes)
    per_profile = {
        profile: sum(
            x["cost_choice_correct"] and x["program_verified"]
            for x in executions
            if x["profile"] == profile
        )
        / sum(x["profile"] == profile for x in executions)
        for profile in PROFILES
    }
    serial_baseline = sum(x["serial_is_best"] for x in executions) / len(executions)
    delegate_baseline = 1 - serial_baseline
    return {
        "acceptance_thresholds": {
            "minimum_overall_verified_cost_choice_rate": 0.9,
            "minimum_per_profile_verified_cost_choice_rate": 0.75,
            "must_exceed_both_constant_baselines": True,
            "scope": "bounded procedure choice only",
        },
        "bounded_choice_pass": success_rate >= 0.9
        and min(per_profile.values()) >= 0.75
        and success_rate > max(serial_baseline, delegate_baseline),
        "per_profile_verified_cost_choice_rate": per_profile,
        "heldout_attempts": len(outcomes),
        "verified_cost_successes": outcomes.count("success"),
        "verified_cost_success_rate": outcomes.count("success") / len(outcomes),
        "teacher_substitutions": 0,
        "execution_timing_clock": "wall_perf_counter",
        "modelled_schedule_times_used_as_wall_time": False,
        "always_serial_cost_success_rate": sum(x["serial_is_best"] for x in executions)
        / len(executions),
        "always_delegate_cost_success_rate": sum(not x["serial_is_best"] for x in executions)
        / len(executions),
        "executions": executions,
        "reference_measurements_excluded_from_decision_time": True,
        "public_output_constraint": "one of the two procedure tools; ordinary text is disabled for this bounded application",
        "free_form_autonomous_scheduler_claimed": False,
        "scope": "neural choice between two fixed, executed procedures",
        "open_ended_controller_success_claimed": False,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--families", type=int, default=64)
    p.add_argument("--steps", type=int, default=64)
    p.add_argument("--foundation-steps", type=int, default=256)
    p.add_argument(
        "--_stage",
        choices=("prepare", "foundation", "train", "restart", "export", "evaluate"),
        default="prepare",
    )
    args = p.parse_args()
    if min(args.families, args.steps, args.foundation_steps) < 1:
        p.error("Family and update budgets must be positive")

    def next_stage(stage):
        os.execv(
            sys.executable,
            [
                sys.executable,
                __file__,
                "--root",
                str(args.root),
                "--families",
                str(args.families),
                "--steps",
                str(args.steps),
                "--foundation-steps",
                str(args.foundation_steps),
                "--_stage",
                stage,
            ],
        )

    checkpoint = args.root / "training/checkpoints/latest.json"
    if args._stage == "prepare":
        args.root.mkdir(parents=True, exist_ok=False)
        prepare(args.root / "data", families=args.families)
        with PreparedDataset(args.root / "data", split="validation") as heldout:
            if not len(heldout):
                raise ValueError("Prepare held-out original families before training")
        next_stage("foundation")
    from megatoken.training.config import TrainConfig
    from megatoken.training.engine import train

    config = TrainConfig(
        purpose="diagnostic",
        model="tiny",
        model_overrides={"max_capacity": 512},
        phase="P1",
        representation_mode="recurrent",
        budget=1000000,
        datasets=[str(args.root / "data")],
        output=str(args.root / "training"),
        codec="byte",
        allow_byte_codec=True,
        device="cuda",
        precision="fp32",
        max_steps=args.steps,
        microbatch_episodes=4,
        max_accumulation_windows=1,
        accumulate_scored=1,
        max_window_records=4,
        max_history_tokens=1024,
        max_record_bytes=8192,
        generated_capacity=512,
        effort=1,
        expert_backend="triton",
        gradient_checkpointing=False,
        use_stage_mixture=False,
        use_effort_mixture=False,
        checkpoint_every=16,
    )
    if args._stage == "foundation":
        foundation = replace(
            config,
            phase="B0",
            representation_mode="bootstrap",
            budget=2000000,
            max_steps=args.foundation_steps,
            output=str(args.root / "foundation"),
        )
        write_json(args.root / "foundation-config.json", asdict(foundation))
        result = train(foundation)
        write_json(args.root / "foundation.json", result)
        if result["termination"] == "requested":
            raise SystemExit(130)
        next_stage("train")
    if args._stage == "train":
        write_json(args.root / "train-config.json", asdict(config))
        result = train(
            config,
            resume=str(args.root / "foundation/checkpoints/latest.json"),
            phase_transition=True,
        )
        write_json(args.root / "training.json", result)
        if result["termination"] == "requested":
            raise SystemExit(130)
        next_stage("restart")
    config = TrainConfig.load(args.root / "train-config.json")
    if args._stage == "restart":
        result = train(replace(config, max_steps=2), resume=str(checkpoint))
        write_json(args.root / "restart.json", result)
        if result["termination"] == "requested":
            raise SystemExit(130)
        next_stage("export")
    if args._stage == "export":
        from megatoken.served import export_served

        write_json(
            args.root / "export.json",
            export_served(checkpoint, args.root / "served", codec_path="byte"),
        )
        next_stage("evaluate")
    write_json(
        args.root / "evaluation.json", evaluate(args.root / "served", args.root / "data", args.root)
    )


if __name__ == "__main__":
    main()
