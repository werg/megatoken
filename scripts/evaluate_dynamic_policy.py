#!/usr/bin/env python3
"""Matched autonomous policy executions through the exported serving work loop.

Only public task observations enter the model. Every subsequent tool observation
comes from the model's exact selected call. There are no prerecorded successful
receipts or teacher substitutions. Failed bounded attempts are retained.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import torch

from megatoken.contracts import ROLE_TO_ID
from megatoken.data.dynamic_execution import SOURCE_VERSION, task_definition
from megatoken.data.open_media import frozen_split
from megatoken.data.schema import stable_id
from megatoken.environments import FIXTURE_TOOLS, SnapshotEnvironment
from megatoken.execution import SessionExecutor
from megatoken.io import canonical_json, sha256_file, write_json
from megatoken.model.multitrack import MultiTrackGenerator, ObservedEvent
from megatoken.multitrack_runtime import ConcurrentToolRunner, MultiTrackRuntime
from megatoken.served import ServedBundle
from megatoken.tool_schema import parse_tool_action


def attempt(model, *, index, profile, serial, decisions, timeout, seed):
    dependent = profile == "dependent_reads"
    delay = 0.04 if profile == "independent_slow_reads" else 0.01 if dependent else 0.0
    overhead = 0.001 if profile == "independent_slow_reads" else 0.01 if dependent else 0.02
    environment = SnapshotEnvironment()
    initial = environment.reset(task_definition(index, dependent=dependent))
    public_tools = copy.deepcopy(FIXTURE_TOOLS)
    public_paths = json.loads(initial.observation)["paths"]
    for tool in public_tools:
        if tool["name"] == "read":
            tool["parameters"]["properties"]["path"]["enum"] = public_paths
        elif tool["name"] == "write":
            # The output filename is explicitly named in the public request.
            # Its expected contents remain private to the verifier.
            tool["parameters"]["properties"]["path"]["enum"] = ["answer.txt"]
    runtime = MultiTrackRuntime(MultiTrackGenerator(model), record_policy_context=True)
    session = runtime.open(effort=1, memory_writes=False)
    session.tracks.max_active = 1 if serial else model.config.max_tracks
    runtime.observe_text(session, "user", initial.observation, role="user", effort=1)
    runtime.observe(
        session,
        (
            ObservedEvent(
                "public-costs",
                "text",
                canonical_json(
                    {
                        "read_delay_seconds": delay,
                        "worker_creation_delay_seconds": overhead,
                        "dependent_reads": dependent,
                    }
                ),
                ROLE_TO_ID["system"],
                event_id="public-costs",
            ),
        ),
        effort=1,
    )
    calls, events = [], []
    known_state = initial.state_id
    selected_snapshots = {}
    observed_actions = set()
    selected_observations = {}

    original_collect = runtime.collect_tools

    def collect_tools(*args, **kwargs):
        nonlocal known_state
        ready, failed = original_collect(*args, **kwargs)
        for action_id, payload in ready.items():
            receipt = json.loads(payload)
            if "state_id" in receipt:
                known_state = receipt["state_id"]
                observed_actions.add(action_id)
        return ready, failed

    runtime.collect_tools = collect_tools
    original_start = runtime.start_generation

    def start_generation(*args, **kwargs):
        snapshot = known_state
        revision = original_start(*args, **kwargs)
        selected_snapshots[session.rendering[revision].selection.tape.meridian_id] = snapshot
        selected_observations[session.rendering[revision].selection.tape.meridian_id] = tuple(
            observed_actions
        )
        return revision

    runtime.start_generation = start_generation

    class ActualTool:
        def __init__(self, snapshot, prior_actions=()):
            self.snapshot = snapshot
            self.prior_actions = prior_actions

        def execute_selected(self, *, action_id, payload):
            name, _, _, _ = parse_tool_action(payload)
            snapshot = self.snapshot
            started = time.monotonic_ns()
            if name == "read":
                time.sleep(delay)
            result = environment.step(payload, action_id=action_id, expected_state_id=snapshot)
            finished = time.monotonic_ns()
            calls.append(
                {
                    "action_id": action_id,
                    "payload_hex": payload.hex(),
                    "input_snapshot": snapshot,
                    "observed_actions_before_selection": self.prior_actions,
                    "execution_id": result.execution_id,
                    "started_at_ns": started,
                    "completed_at_ns": finished,
                    "observation_hex": result.observation.hex(),
                }
            )
            return canonical_json(
                {
                    "observation": json.loads(result.observation),
                    "execution_id": result.execution_id,
                    "state_id": result.state_id,
                    "done": result.done,
                    "verified": result.verified,
                    "reward": result.reward,
                    "provenance": "executed",
                }
            )

    # Apply the same declared cost to real lifecycle creation, after model
    # selection. Setup of the one host-owned root is outside task timing.
    original_change = runtime.track_action

    def track_action(*args, **kwargs):
        operation = args[1] if len(args) > 1 else kwargs.get("operation")
        if operation == "create" and session.tracks.tracks:
            time.sleep(overhead)
        return original_change(*args, **kwargs)

    runtime.track_action = track_action
    adapter = ActualTool(initial.state_id)
    original_dispatch = runtime.dispatch_tools

    def dispatch_tools(session, group, runner, adapters, **kwargs):
        binding = ActualTool(
            selected_snapshots[group.tape.meridian_id],
            selected_observations[group.tape.meridian_id],
        )
        return original_dispatch(
            session, group, runner, {name: binding for name in adapters}, **kwargs
        )

    runtime.dispatch_tools = dispatch_tools
    executor = SessionExecutor(
        runtime,
        session,
        tools=tuple(public_tools),
        adapters={t["name"]: adapter for t in public_tools},
        max_selections=decisions,
        runner=ConcurrentToolRunner(max_workers=1 if serial else 2),
    )
    started = time.perf_counter()
    first_useful = None
    error = None
    try:
        while time.perf_counter() - started < timeout:
            step = executor.tick(effort=1, seed=seed)
            events.append(step)
            if calls and first_useful is None:
                first_useful = time.perf_counter() - started
            for item in step.get("outputs", ()):
                runtime.stop_delivery(session, item["selection_revision"])
                runtime.release_generation(session, item["selection_revision"])
            if environment.done:
                break
            unfinished = any(w.status == "rendering" for w in session.rendering.values())
            if (
                step["status"] == "budget_exhausted"
                and not unfinished
                and not executor.runner.pending
            ):
                break
            time.sleep(0.001)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        executor.close()
        runtime.close()
    elapsed = time.perf_counter() - started
    verdict = environment.verify()
    # Reaching the expected final files without submitting is not completion.
    read_calls = [
        (call, parse_tool_action(bytes.fromhex(call["payload_hex"]))[1])
        for call in calls
        if parse_tool_action(bytes.fromhex(call["payload_hex"]))[0] == "read"
    ]
    required = {"left.txt", f"right-{index}.txt" if dependent else "right.txt"}
    read_paths = {args["path"] for _, args in read_calls}
    dependency_respected = not dependent or any(
        first["action_id"] in second["observed_actions_before_selection"]
        for first, a in read_calls
        for second, b in read_calls
        if a["path"] == "left.txt" and b["path"] == f"right-{index}.txt"
    )
    success = (
        environment.done
        and verdict.verified
        and verdict.reward == 1
        and required <= read_paths
        and dependency_respected
    )
    return {
        "profile": profile,
        "original_family": stable_id(SOURCE_VERSION, str(index)),
        "serial_ablation": serial,
        "seed": seed,
        "task_success": success,
        "submitted": environment.done,
        "required_reads_executed": required <= read_paths,
        "dependency_respected": dependency_respected,
        "elapsed_seconds": elapsed,
        "first_tool_result_seconds": first_useful,
        "selected_groups": executor.selections,
        "external_calls": calls,
        "runtime_events": events,
        "error": error,
        "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated()
        if next(model.parameters()).is_cuda
        else None,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--packs", type=Path)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--families", type=int, default=2)
    p.add_argument("--decisions", type=int, default=8)
    p.add_argument("--timeout", type=float, default=120.0)
    args = p.parse_args()
    if min(args.families, args.decisions, args.timeout) <= 0:
        p.error("Positive family, decision and timeout bounds are required")
    args.root.mkdir(parents=True, exist_ok=False)
    bundle = ServedBundle(args.bundle, packs=args.packs, device=args.device)
    indices = []
    for index in range(100000):
        if frozen_split(stable_id(SOURCE_VERSION, str(index))) == "validation":
            indices.append(index)
        if len(indices) == args.families:
            break
    results = []
    for index in indices:
        for profile in ("independent_slow_reads", "independent_cheap_reads", "dependent_reads"):
            for serial in (True, False):
                row = attempt(
                    bundle.model,
                    index=index,
                    profile=profile,
                    serial=serial,
                    decisions=args.decisions,
                    timeout=args.timeout,
                    seed=17 + index,
                )
                write_json(args.root / f"attempt-{len(results)}.json", row)
                results.append(row)
                print(
                    json.dumps(
                        {
                            "attempt": len(results),
                            "success": row["task_success"],
                            "elapsed_seconds": row["elapsed_seconds"],
                        }
                    ),
                    flush=True,
                )
    write_json(
        args.root / "report.json",
        {
            "format": "autonomous-delegation-evaluation/v1",
            "served_manifest_sha256": sha256_file(args.bundle / "manifest.json"),
            "attempts": len(results),
            "successes": sum(r["task_success"] for r in results),
            "task_success_rate": sum(r["task_success"] for r in results) / len(results),
            "serial_successes": sum(r["task_success"] for r in results if r["serial_ablation"]),
            "dynamic_successes": sum(
                r["task_success"] for r in results if not r["serial_ablation"]
            ),
            "teacher_substitutions": 0,
            "prerecorded_tool_results": 0,
            "success_threshold": "every submitted held-out task must match the independent environment state verifier",
            "speedup_claim": False,
        },
    )


if __name__ == "__main__":
    main()
