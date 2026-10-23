"""Authored context-management demonstrations with actual bounded archive reads.

Variants share original task ancestry. Numeric variation is task-instance
variation; the explicit template label must be held out for template evaluations.
"""

from __future__ import annotations

from megatoken.context_archive import ContextArchive
from megatoken.lifecycle import lifecycle_tools
from megatoken.model.multitrack import ObservedEvent

from .schema import (
    Dependency,
    Episode,
    Lineage,
    LogicalAction,
    Meridian,
    PhysicalRecord,
    Supervision,
    TrackEvent,
    TrackSpec,
    canonical_json,
    digest_bytes,
    stable_id,
)

SOURCE_VERSION = "context-demonstrations-v1"


def context_episode(index, *, variant="fetch", split="train", local_events=4, cross_events=1):
    if (
        type(local_events) is not int
        or type(cross_events) is not int
        or not 0 <= cross_events < local_events
        or local_events < 2
    ):
        raise ValueError(
            "Context demonstrations need room for local evidence and its policy announcement"
        )
    if variant not in {"fetch", "pin", "local", "unavailable"}:
        raise ValueError("Unknown context demonstration variant")
    family = stable_id(SOURCE_VERSION, str(index))
    worker, main, busy = (f"{kind}:{family[:12]}" for kind in ("worker", "main", "busy"))
    records, groups, actions, dependencies = [], [], [], []
    key = f"key-{index * 17 + 31}"
    archive = ContextArchive(max_events=0 if variant == "unavailable" else 32)

    def add(name, track, role, payload, *, scored=False, **kwargs):
        event_id = f"{family[:12]}:{name}"
        record = PhysicalRecord(
            event_id,
            role,
            payload,
            turn_index=len(groups),
            policy_mask=scored,
            provenance="executed",
            **kwargs,
        )
        records.append(record)
        groups.append(
            Meridian(len(groups), (TrackEvent(event_id, track, "text", len(groups), role),))
        )
        return record

    add(
        "request",
        main,
        "user",
        b"Remember the exact project key. Report it after the unrelated work, retrieving the source if needed.",
    )
    source = add("evidence", worker if variant == "local" else main, "document", key.encode())
    archive.put(ObservedEvent(source.role, "text", source.payload, event_id=source.record_id))
    settings = {
        "local_events": local_events,
        "cross_events": cross_events,
        "pinned_events": [source.record_id] if variant == "pin" else [],
    }
    add("policy", worker, "system", canonical_json({"context_window": settings}).encode())
    if variant in {"fetch", "unavailable"}:
        add(
            "release",
            main,
            "system",
            canonical_json(
                {"context_window": {"local_events": 0, "cross_events": 0, "pinned_events": []}}
            ).encode(),
        )
    for i in range(max(cross_events + 2, 3)):
        add(
            f"unrelated-{i}",
            busy,
            "document",
            f"Unrelated result {i}: no project key here".encode(),
        )
    receipts = []
    if variant in {"fetch", "unavailable"}:
        arguments = {"address": worker, "source_event": source.record_id}
        payload = canonical_json({"name": "track_fetch", "arguments": arguments}).encode()
        action_id = f"fetch:{family}:{variant}"
        call = add("fetch-call", worker, "assistant", payload, scored=True, action_id=action_id)
        actual = archive.get(source.record_id)
        execution_id = stable_id("context-read-execution", family, variant, source.payload)
        actions.append(
            LogicalAction(
                action_id,
                action_id,
                "track_fetch",
                canonical_json(arguments),
                (call.record_id,),
                digest_bytes(payload),
                execution_id=execution_id,
            )
        )
        observation = (
            actual.payload
            if actual is not None
            else b"Context unavailable; request the original source or clarification."
        )
        result = add(
            "fetch-result",
            worker,
            "tool",
            observation,
            result_of=action_id,
            execution_id=execution_id,
        )
        dependencies.append(Dependency(source.record_id, call.record_id, "requested_source"))
        receipts.append(
            {
                "action_id": action_id,
                "source_event_id": source.record_id,
                "result_record_id": result.record_id,
                "available": actual is not None,
                "sha256": digest_bytes(observation),
            }
        )
        last_evidence = result
    else:
        last_evidence = source
    answer = add(
        "answer",
        worker,
        "assistant",
        key.encode()
        if variant != "unavailable"
        else b"Please provide the original project key again.",
        scored=True,
    )
    dependencies.append(Dependency(last_evidence.record_id, answer.record_id, "actual_evidence"))
    assert variant == "unavailable" or answer.payload == source.payload
    return Episode(
        stable_id(SOURCE_VERSION, family, variant),
        tuple(records),
        Lineage(SOURCE_VERSION, SOURCE_VERSION, family, frozen_split=split),
        stream="dynamic_execution",
        actions=tuple(actions),
        dependencies=tuple(dependencies),
        supervision=Supervision(
            outcome="incomplete" if variant == "unavailable" else "success",
            verification="local_assertions",
            annotations_json=canonical_json(
                {
                    "task_template": "remember_exact_project_key",
                    "origin": "authored_executed",
                    "context_variant": variant,
                    "context_read_receipts": receipts,
                }
            ),
        ),
        tracks=(
            TrackSpec(main, "assistant", description="Keep the user's task and source evidence"),
            TrackSpec(
                worker,
                "assistant",
                description="Return the exact available project key or request missing evidence",
            ),
            TrackSpec(busy, "document", description="Independent unrelated work"),
        ),
        meridians=tuple(groups),
        tools_json=canonical_json(lifecycle_tools()),
    )
