"""Authored, actually executed lifecycle and concurrent-tool training tasks.

Every target call is executed against a fresh task snapshot. The source is
explicitly authored (not a natural conversation or a student rollout). Serial
variants remain serial; concurrent reads share the exact immutable snapshot.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict

from megatoken.environments import FIXTURE_TOOLS, SnapshotEnvironment
from megatoken.lifecycle import TrackRegistry, lifecycle_tools

from .schema import (
    Dependency,
    Episode,
    Lineage,
    LogicalAction,
    Meridian,
    PhysicalRecord,
    Supervision,
    TrackEvent,
    TrackRevision,
    TrackSpec,
    canonical_json,
    digest_bytes,
    stable_id,
)

SOURCE_VERSION = "dynamic-executions-v3"


def task_definition(index: int, *, dependent: bool = False) -> dict:
    """Private verifier data stays in the environment; reset returns public inputs."""
    a, b = index * 7 + 11, index * 13 + 5
    initial_files = {"left.txt": str(a), "right.txt": str(b)}
    prompt = "Read left.txt and right.txt, add their numbers, write the result to answer.txt, then finish."
    if dependent:
        initial_files = {"left.txt": f"right-{index}.txt", f"right-{index}.txt": str(b)}
        prompt = "Read left.txt to discover which file to read next. Read that file, double its number, write the result to answer.txt, then finish."
        a = b
    return {
        "prompt": prompt,
        "initial_files": initial_files,
        "expected_files": {**initial_files, "answer.txt": str(a + b)},
    }


def execution_episode(
    index: int,
    *,
    delegated: bool = True,
    split: str = "train",
    read_delay_seconds: float = 0.0,
    delegation_overhead_seconds: float = 0.0,
    dependent: bool = False,
) -> Episode:
    """A verified task with replayable public observations and exact effects."""
    if min(read_delay_seconds, delegation_overhead_seconds) < 0:
        raise ValueError("Controlled execution delays must be nonnegative")
    # A logical clock in nanoseconds, driven by the controlled delays, not by the machine:
    # the same index must give the same episode bytes, and the recorded intervals still show
    # exactly which reads overlapped. Real sleeps below only make the execution genuinely slow.
    clock = [1_000_000]
    read_delay_ns = round(read_delay_seconds * 1e9)
    overhead_ns = round(delegation_overhead_seconds * 1e9)
    family = stable_id(SOURCE_VERSION, str(index))
    variant = canonical_json(
        {
            "delegated": delegated,
            "dependent": dependent,
            "read_delay_seconds": read_delay_seconds,
            "delegation_overhead_seconds": delegation_overhead_seconds,
        }
    )
    task = task_definition(index, dependent=dependent)
    env = SnapshotEnvironment()
    initial = env.reset(task, session=stable_id("execution-session", family, variant)[:32])
    registry = TrackRegistry(max_active=8, address_prefix=f"task-{family[:12]}")
    root = registry.create(
        "Answer the user's request using actual tool results",
        completion="The answer file is verified",
        capabilities=("text", "read", "write", "finish"),
    )
    records, groups, actions, dependencies, execution_log = [], [], [], [], []
    execution_intervals = []
    tracks = {root.address: TrackSpec(root.address, "assistant", description=root.purpose)}
    available = []
    revisions = []
    descriptions = {root.address: (root.purpose, root.revision)}

    def group(items):
        number = len(groups)
        events = []
        for item in items:
            role, track, payload, scored, extra = item
            record_id = stable_id("execution-record", family, variant, str(len(records)))
            record = PhysicalRecord(
                record_id,
                role,
                payload,
                logical_id=record_id,
                turn_index=number,
                policy_mask=scored,
                provenance="executed",
                **extra,
            )
            records.append(record)
            tracks.setdefault(track, TrackSpec(track, role))
            events.append(TrackEvent(record_id, track, "text", number, role))
        groups.append(Meridian(number, tuple(events), selected_at_ns=clock[0]))
        clock[0] += 1_000_000
        return tuple(records[-len(items) :])

    group(
        [
            ("user", "user", json.loads(initial.observation)["task"].encode(), False, {}),
            (
                "system",
                root.address,
                canonical_json({"task": root.address, "purpose": root.purpose}).encode(),
                False,
                {},
            ),
        ]
    )
    available.append(records[0].record_id)
    group(
        [
            (
                "system",
                root.address,
                canonical_json(
                    {
                        "execution_costs": {
                            "read_delay_seconds": read_delay_seconds,
                            "worker_creation_delay_seconds": delegation_overhead_seconds,
                        },
                        "read_dependency": "second path is learned from first result"
                        if dependent
                        else "independent paths",
                    }
                ).encode(),
                False,
                {},
            )
        ]
    )

    def calls(values):
        registry.charge(tuple(value[2] for value in values))
        items, selected = [], []
        for name, arguments, track in values:
            payload = canonical_json({"name": name, "arguments": arguments}).encode()
            action_id = stable_id(
                "execution-action",
                family,
                variant,
                str(len(actions) + len(selected)),
                payload,
            )
            items.append(("assistant", track, payload, True, {"action_id": action_id}))
            selected.append((name, arguments, track, payload, action_id))
        physical = group(items)
        for record in physical:
            dependencies.extend(
                Dependency(writer, record.record_id, "available_result") for writer in available
            )
        return [(value, record) for value, record in zip(selected, physical, strict=True)]

    def result(selected, observation, execution_id, state_id):
        (name, arguments, track, payload, action_id), record = selected
        actions.append(
            LogicalAction(
                action_id,
                action_id,
                name,
                canonical_json(arguments),
                (record.record_id,),
                digest_bytes(payload),
                execution_id=execution_id,
            )
        )
        (observed,) = group(
            [
                (
                    "tool",
                    track,
                    observation,
                    False,
                    {"result_of": action_id, "execution_id": execution_id},
                )
            ]
        )
        available.append(observed.record_id)
        changed = json.loads(observation).get("track")
        if isinstance(changed, dict) and "purpose" in changed:
            identity = (changed["purpose"], changed["revision"])
            if descriptions.get(changed["address"]) != identity:
                revisions.append(TrackRevision(observed.record_id, changed["address"], *identity))
                descriptions[changed["address"]] = identity
        execution_log.append(
            {
                "action_id": action_id,
                "execution_id": execution_id,
                "input_state": state_id,
                "result_record": observed.record_id,
                "available_at_ns": clock[0],
            }
        )

    workers = []
    if delegated:
        selected = calls(
            [
                (
                    "track_create",
                    {
                        "purpose": (
                            "Read the path published by the first completed read and publish its number"
                            if dependent and path == "right.txt"
                            else f"Read {path} and publish its contents"
                        ),
                        "completion": "The number is available",
                        "parent": root.address,
                        "capabilities": ["text", "read"],
                        "budget": 8,
                    },
                    root.address,
                )
                for path in ("left.txt", "right.txt")
            ]
        )
        for ordinal, item in enumerate(selected):
            time.sleep(delegation_overhead_seconds)
            clock[0] += overhead_ns
            arguments = dict(item[0][1])
            purpose = arguments.pop("purpose")
            worker = registry.create(purpose, **arguments)
            workers.append(worker)
            tracks[worker.address] = TrackSpec(worker.address, "assistant")
            result(
                item,
                canonical_json({"track": asdict(worker)}).encode(),
                f"lifecycle:{family}:create:{ordinal}",
                "available-public-task",
            )
        if index % 3 == 1:
            worker = workers[0]
            (item,) = calls(
                [
                    (
                        "track_assign",
                        {
                            "address": worker.address,
                            "purpose": "Read left.txt and retain its exact numeric value",
                            "completion": "The exact value is published",
                        },
                        root.address,
                    )
                ]
            )
            changed = registry.change(
                worker.address,
                "assign",
                purpose=item[0][1]["purpose"],
                completion=item[0][1]["completion"],
            )
            result(
                item,
                canonical_json({"track": asdict(changed)}).encode(),
                f"lifecycle:{family}:assign",
                "available-public-task",
            )
        queued = calls(
            [
                (
                    "track_output",
                    {"address": worker.address, "request": f"Read {path}", "modality": "text"},
                    root.address,
                )
                for worker, path in zip(workers, ("left.txt", "right.txt"), strict=True)
            ]
        )
        for ordinal, item in enumerate(queued):
            if not registry.selectable(workers[ordinal].address, set()):
                raise RuntimeError("Authored output requested an ineligible worker")
            result(
                item,
                canonical_json({"event_id": f"read:{ordinal}", "status": "queued"}).encode(),
                f"lifecycle:{family}:queue:{ordinal}",
                "available-public-task",
            )
    else:
        workers = [root, root]
    snapshot = env.state_id

    def take_step(item, state, start):
        observed = env.step(item[0][3], action_id=item[0][4], expected_state_id=state)
        return observed, start, start + read_delay_ns

    def execute(item, state, start):
        time.sleep(read_delay_seconds)
        return take_step(item, state, start)

    def completed(item, value, state):
        observed, start, end = value
        clock[0] = max(clock[0], end)
        result(item, observed.observation, observed.execution_id, state)
        execution_intervals.append(
            {
                "action_id": item[0][4],
                "execution_id": observed.execution_id,
                "started_at_ns": start,
                "completed_at_ns": end,
            }
        )
        return observed

    if delegated and not dependent:
        selected = calls(
            [
                ("read", {"path": path}, worker.address)
                for worker, path in zip(workers, ("left.txt", "right.txt"), strict=True)
            ]
        )
        with ThreadPoolExecutor(max_workers=2) as executor:
            dispatched = clock[0]  # both reads start together: that is the overlap on record
            # The reads really run side by side (each waits out its delay in its own thread);
            # the environment records them below in selection order, so its execution
            # numbers never depend on which thread woke first.
            pending = {
                executor.submit(time.sleep, read_delay_seconds): item for item in selected
            }
            if index % 3 == 0:
                group(
                    [
                        (
                            "user",
                            "user",
                            b"Pause the second worker while its pending read finishes; retain the result under that original task.",
                            False,
                            {},
                        )
                    ]
                )
                (item,) = calls([("track_cancel", {"address": workers[1].address}, root.address)])
                changed = registry.change(workers[1].address, "cancel")
                result(
                    item,
                    canonical_json({"track": asdict(changed)}).encode(),
                    f"lifecycle:{family}:cancel",
                    "available-public-task",
                )
            # In the order the actions were selected, not the order threads happened to finish.
            for future, item in pending.items():
                future.result()
                completed(item, take_step(item, snapshot, dispatched), snapshot)
        if index % 3 == 0:
            group(
                [
                    (
                        "user",
                        "user",
                        b"Resume the second worker and finish the original request using the completed reads.",
                        False,
                        {},
                    )
                ]
            )
            (item,) = calls([("track_resume", {"address": workers[1].address}, root.address)])
            changed = registry.change(workers[1].address, "resume")
            result(
                item,
                canonical_json({"track": asdict(changed)}).encode(),
                f"lifecycle:{family}:resume",
                "available-public-task",
            )
    else:
        path = "left.txt"
        for ordinal, worker in enumerate(workers):
            if ordinal == 1 and not dependent:
                path = "right.txt"
            (item,) = calls([("read", {"path": path}, worker.address)])
            observed = completed(item, execute(item, env.state_id, clock[0]), snapshot)
            if dependent and ordinal == 0:
                path = json.loads(observed.observation)["content"]
    if delegated:
        for operation in ("progress", "complete"):
            selected = calls(
                [
                    (
                        "track_" + operation,
                        {
                            "address": worker.address,
                            **(
                                {"progress": "The requested number is published"}
                                if operation == "progress"
                                else {}
                            ),
                        },
                        worker.address,
                    )
                    for worker in workers
                ]
            )
            for ordinal, item in enumerate(selected):
                arguments = dict(item[0][1])
                address = arguments.pop("address")
                changed = registry.change(address, operation, **arguments)
                result(
                    item,
                    canonical_json({"track": asdict(changed)}).encode(),
                    f"lifecycle:{family}:{operation}:{ordinal}",
                    "available-public-task",
                )
    (item,) = calls(
        [
            (
                "write",
                {"path": "answer.txt", "content": task["expected_files"]["answer.txt"]},
                root.address,
            )
        ]
    )
    before = env.state_id
    observed = env.step(item[0][3], action_id=item[0][4], expected_state_id=before)
    result(item, observed.observation, observed.execution_id, before)
    (item,) = calls([("finish", {}, root.address)])
    before = env.state_id
    observed = env.step(item[0][3], action_id=item[0][4], expected_state_id=before)
    result(item, observed.observation, observed.execution_id, before)
    verdict = env.verify()
    if not verdict.verified or verdict.reward != 1:
        raise RuntimeError("Authored execution failed independent environment verification")
    return Episode(
        stable_id("execution-episode", family, variant),
        tuple(records),
        Lineage(SOURCE_VERSION, SOURCE_VERSION, family, frozen_split=split),
        actions=tuple(actions),
        dependencies=tuple(dependencies),
        tracks=tuple(tracks.values()),
        track_revisions=tuple(revisions),
        meridians=tuple(groups),
        stream="dynamic_execution",
        control_seed=b"Use only available observations. Create useful work when appropriate, preserve dependencies, and verify actual effects.",
        tools_json=canonical_json([*lifecycle_tools(), *FIXTURE_TOOLS]),
        supervision=Supervision(
            outcome="success",
            verification="local_assertions",
            reward=1.0,
            annotations_json=canonical_json(
                {
                    "origin": "authored_executed",
                    "execution_contract": "immutable_reads_per_resource_writes",
                    "delegated": delegated,
                    "dependent_reads": dependent,
                    "controlled_read_delay_seconds": read_delay_seconds,
                    "controlled_worker_creation_delay_seconds": delegation_overhead_seconds,
                    "elapsed_seconds": clock[0] / 1e9,
                    "clock": "logical",
                    "execution_intervals": execution_intervals,
                    "executions": execution_log,
                    "natural_timing": False,
                }
            ),
        ),
    )
