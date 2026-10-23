"""Authored, actually executed lifecycle and concurrent-tool training tasks.

Every target call is executed against a fresh task snapshot. The source is
explicitly authored (not a natural conversation or a student rollout). Serial
variants remain serial; concurrent reads share the exact immutable snapshot.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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

SOURCE_VERSION = "dynamic-executions-v2"


def execution_episode(index: int, *, delegated: bool = True, split: str = "train") -> Episode:
    """A verified task with replayable public observations and exact effects."""
    family = stable_id(SOURCE_VERSION, str(index))
    a, b = index * 7 + 11, index * 13 + 5
    env = SnapshotEnvironment()
    initial = env.reset(
        {
            "prompt": "Read left.txt and right.txt, add their numbers, write the result to answer.txt, then finish.",
            "initial_files": {"left.txt": str(a), "right.txt": str(b)},
            "expected_files": {"left.txt": str(a), "right.txt": str(b), "answer.txt": str(a + b)},
        }
    )
    registry = TrackRegistry(max_active=8, address_prefix=f"task-{family[:12]}")
    root = registry.create(
        "Answer the user's request using actual tool results",
        completion="The answer file is verified",
        capabilities=("text", "read", "write", "finish"),
    )
    records, groups, actions, dependencies, execution_log = [], [], [], [], []
    tracks = {root.address: TrackSpec(root.address, "assistant", description=root.purpose)}
    available = []
    revisions = []
    descriptions = {root.address: (root.purpose, root.revision)}

    def group(items):
        number = len(groups)
        events = []
        for item in items:
            role, track, payload, scored, extra = item
            record_id = stable_id("execution-record", family, str(delegated), str(len(records)))
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
        groups.append(Meridian(number, tuple(events), selected_at_ns=time.monotonic_ns()))
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

    def calls(values):
        items, selected = [], []
        for name, arguments, track in values:
            payload = canonical_json({"name": name, "arguments": arguments}).encode()
            action_id = stable_id(
                "execution-action",
                family,
                str(delegated),
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
                "available_at_ns": time.monotonic_ns(),
            }
        )

    workers = []
    if delegated:
        selected = calls(
            [
                (
                    "track_create",
                    {
                        "purpose": f"Read {path} and publish its number",
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
    if delegated:
        selected = calls(
            [
                ("read", {"path": path}, worker.address)
                for worker, path in zip(workers, ("left.txt", "right.txt"), strict=True)
            ]
        )
        with ThreadPoolExecutor(max_workers=2) as executor:
            pending = {
                executor.submit(
                    env.step, item[0][3], action_id=item[0][4], expected_state_id=snapshot
                ): item
                for item in selected
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
            for future in as_completed(pending):
                observed = future.result()
                result(pending[future], observed.observation, observed.execution_id, snapshot)
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
        for worker, path in zip(workers, ("left.txt", "right.txt"), strict=True):
            (item,) = calls([("read", {"path": path}, worker.address)])
            observed = env.step(item[0][3], action_id=item[0][4], expected_state_id=env.state_id)
            result(item, observed.observation, observed.execution_id, snapshot)
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
    (item,) = calls([("write", {"path": "answer.txt", "content": str(a + b)}, root.address)])
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
        stable_id("execution-episode", family, str(delegated)),
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
                    "executions": execution_log,
                    "natural_timing": False,
                }
            ),
        ),
    )
