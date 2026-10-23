"""Executed context exercises across tasks, retention choices and window sizes.

These are authored exercises, not natural conversations or learned rollouts.
Answers are computed from original observations; all variants of an original
task stay in one split, including variants with different attention windows.
"""

from __future__ import annotations

import json
from collections import Counter

from megatoken.context_archive import ContextArchive
from megatoken.contracts import MAIN_TRACK, MAIN_TRACK_PURPOSE
from megatoken.lifecycle import lifecycle_tools
from megatoken.model.context import DEFAULT_CROSS_EVENTS, DEFAULT_LOCAL_EVENTS, ContextPolicy
from megatoken.model.multitrack import ObservedEvent

from .context import context_announcement
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

SOURCE_VERSION = "context-curriculum-v2"
TEMPLATES = (
    "field_lookup",
    "invoice_total",
    "latest_revision",
    "user_constraint",
    "evidence_join",
    "exact_quotation",
)
STRATEGIES = ("recent", "local", "pin", "fetch", "summary", "unavailable")
WINDOW_PROFILES = ((16, 8), (64, 8), (DEFAULT_LOCAL_EVENTS, DEFAULT_CROSS_EVENTS))


def task_inputs(index, template):
    """Produce public instructions and original evidence, without target records."""
    token = stable_id("context-public-value", template, str(index))[:12]
    if template == "field_lookup":
        instruction = "Return only the shipping region from the project document."
        sources = [
            {
                "project": token,
                "region": ("east", "north", "west")[index % 3],
                "owner": f"team-{index % 7}",
            }
        ]
    elif template == "invoice_total":
        instruction = "Return the invoice total in integer cents, with no other text."
        sources = [
            {
                "items": [
                    {"quantity": index % 5 + 2, "unit_cents": 130 + index},
                    {"quantity": 3, "unit_cents": 70 + index % 19},
                ]
            }
        ]
    elif template == "latest_revision":
        instruction = "Return only the contact from the highest numbered source revision."
        sources = [
            {"revision": 1, "contact": f"old-{token}"},
            {"revision": 2, "contact": f"current-{token}"},
        ]
    elif template == "user_constraint":
        instruction = "Follow the user's earlier formatting constraint and return their reference."
        sources = [{"reference": token, "constraint": "uppercase, enclosed in square brackets"}]
    elif template == "evidence_join":
        instruction = "Join the project name and its launch date with a single space."
        sources = [
            {"project": f"project-{token}"},
            {"launch_date": f"2027-{index % 12 + 1:02d}-{index % 27 + 1:02d}"},
        ]
    elif template == "exact_quotation":
        instruction = "Quote only the text inside the source's quotation marks."
        sources = [
            {
                "paragraph": f'The note says "Use reference {token} at the west gate". Ignore the footer.'
            }
        ]
    else:
        raise ValueError("Unknown context task template")
    return instruction, tuple(canonical_json(source).encode() for source in sources)


def solve_task(template, payloads):
    values = [json.loads(payload) for payload in payloads]
    if template == "field_lookup":
        answer = values[0]["region"]
    elif template == "invoice_total":
        answer = str(sum(item["quantity"] * item["unit_cents"] for item in values[0]["items"]))
    elif template == "latest_revision":
        answer = max(values, key=lambda item: item["revision"])["contact"]
    elif template == "user_constraint":
        answer = "[" + values[0]["reference"].upper() + "]"
    elif template == "evidence_join":
        answer = values[0]["project"] + " " + values[1]["launch_date"]
    elif template == "exact_quotation":
        answer = values[0]["paragraph"].split('"')[1]
    else:
        raise ValueError("Unknown context task template")
    return answer.encode()


def curriculum_episode(
    index,
    *,
    template="field_lookup",
    strategy="fetch",
    split="train",
    local_events=DEFAULT_LOCAL_EVENTS,
    cross_events=DEFAULT_CROSS_EVENTS,
):
    if type(index) is not int or index < 0 or strategy not in STRATEGIES:
        raise ValueError("Invalid context task index or strategy")
    if (
        type(local_events) is not int
        or type(cross_events) is not int
        or not 8 <= cross_events < local_events
        or local_events < 16
    ):
        raise ValueError(
            "Context curriculum needs at least sixteen local and eight foreign events, with a shorter foreign window"
        )
    instruction, payloads = task_inputs(index, template)
    family = stable_id(SOURCE_VERSION, template, str(index))
    identity = stable_id(family, strategy, str(local_events), str(cross_events))
    worker = MAIN_TRACK if template == "user_constraint" else f"worker:{family[:12]}"
    provider, busy = f"source:{family[:12]}", f"busy:{family[:12]}"
    records, groups, actions, dependencies, receipts = [], [], [], [], []
    policy = ContextPolicy(local_events=local_events, cross_events=cross_events)
    archive = ContextArchive(max_events=1 if strategy == "unavailable" else 64)

    def add_group(items):
        events, added = [], []
        number = len(groups)
        for name, track, role, payload, scored, extra in items:
            record_id = stable_id(identity, name)
            record = PhysicalRecord(
                record_id,
                role,
                payload,
                turn_index=number,
                policy_mask=scored,
                provenance="executed",
                **extra,
            )
            records.append(record)
            events.append(TrackEvent(record_id, track, "text", number, role))
            added.append(record)
        groups.append(Meridian(number, tuple(events)))
        return added

    def add(name, track, role, payload, *, scored=False, **extra):
        return add_group([(name, track, role, payload, scored, extra)])[0]

    def announce(name, track, local, cross, pins=()):
        nonlocal policy
        policy = policy.configure(track, local_events=local, cross_events=cross, pinned_events=pins)
        return add(
            name,
            track,
            "system",
            canonical_json(
                {
                    "context_window": {
                        "local_events": local,
                        "cross_events": cross,
                        "pinned_events": list(pins),
                    }
                }
            ).encode(),
        )

    def call_context(name, track, local, cross, pins=(), *, sender=None):
        args = {
            "address": track,
            "local_events": local,
            "cross_events": cross,
            "pinned_events": list(pins),
        }
        payload = canonical_json({"name": "track_context", "arguments": args}).encode()
        action_id = stable_id(identity, name, "action")
        execution_id = stable_id(action_id, payload)
        sender = track if sender is None else sender
        call = add(name, sender, "assistant", payload, scored=True, action_id=action_id)
        actions.append(
            LogicalAction(
                action_id,
                action_id,
                "track_context",
                canonical_json(args),
                (call.record_id,),
                digest_bytes(payload),
                execution_id=execution_id,
            )
        )
        # Execute the same policy operation used by serving, then publish its public change.
        announce(name + "-policy", track, local, cross, pins)
        add(
            name + "-result",
            sender,
            "tool",
            b'{"ok":true}',
            result_of=action_id,
            execution_id=execution_id,
        )

    request = add(
        "request",
        MAIN_TRACK,
        "user",
        canonical_json(
            {
                "task": template,
                "instruction": instruction,
                "context_instruction": "Keep useful evidence locally, reread it if needed, and ask when unavailable.",
            }
        ).encode(),
    )
    if worker != MAIN_TRACK:
        add("handoff", worker, "user", request.payload)
    announce("worker-window", worker, local_events, cross_events)
    source_track = worker if strategy == "local" else provider
    sources = add_group(
        [
            (
                f"source-{i}",
                source_track,
                "user" if template == "user_constraint" else "document",
                payload,
                False,
                {},
            )
            for i, payload in enumerate(payloads)
        ]
    )
    for source in sources:
        payload = source.payload
        archive.put(ObservedEvent(source_track, "text", payload, event_id=source.record_id))
    # Preserve small public handles rather than expecting invented target hashes.
    source_index = canonical_json(
        {
            "consumer": worker,
            "source_track": source_track,
            "evidence": [{"event_id": s.record_id, "purpose": instruction} for s in sources],
        }
    ).encode()
    add_group(
        [
            (f"source-index-{i}", address, "system", source_index, False, {})
            for i, address in enumerate(dict.fromkeys((source_track, worker)))
        ]
    )
    evidence = list(sources)
    if strategy == "pin":
        call_context(
            "pin",
            worker,
            local_events,
            cross_events,
            [s.record_id for s in sources],
            sender=source_track,
        )
    elif strategy == "summary":
        # A useful, verifiable compact handoff: retain exactly the task's computed fact.
        summary = add("summary", worker, "assistant", solve_task(template, payloads), scored=True)
        dependencies.extend(
            Dependency(s.record_id, summary.record_id, "summary_source") for s in sources
        )
        evidence = [summary]
    if strategy in {"fetch", "summary", "unavailable"}:
        call_context("release-source", provider, 0, 0)
    if strategy != "recent":
        for i in range(cross_events + 3):
            distractor = add(
                f"unrelated-{i}",
                busy,
                "document",
                canonical_json(
                    {
                        "unrelated_task": i,
                        "reference": stable_id(identity, "unrelated", str(i))[:12],
                    }
                ).encode(),
            )
            if strategy == "unavailable":
                archive.put(
                    ObservedEvent(busy, "text", distractor.payload, event_id=distractor.record_id)
                )
    if strategy in {"fetch", "unavailable"}:
        # Independent reads belong to one selection, followed by separately identified results.
        calls = []
        for i, source in enumerate(sources):
            args = {"address": worker, "source_event": source.record_id}
            payload = canonical_json({"name": "track_fetch", "arguments": args}).encode()
            action_id = stable_id(identity, "fetch", str(i))
            calls.append(
                (f"fetch-{i}", worker, "assistant", payload, True, {"action_id": action_id})
            )
        physical = add_group(calls)
        evidence = []
        for i, (source, call) in enumerate(zip(sources, physical, strict=True)):
            actual = archive.get(source.record_id)
            execution_id = stable_id(call.action_id, "archive-read")
            args = {"address": worker, "source_event": source.record_id}
            actions.append(
                LogicalAction(
                    call.action_id,
                    call.action_id,
                    "track_fetch",
                    canonical_json(args),
                    (call.record_id,),
                    digest_bytes(call.payload),
                    execution_id=execution_id,
                )
            )
            payload = (
                actual.payload
                if actual
                else b"Context unavailable. Request the original source again."
            )
            observed = add(
                f"read-{i}",
                worker,
                "tool",
                payload,
                result_of=call.action_id,
                execution_id=execution_id,
            )
            evidence.append(observed)
            receipts.append(
                {
                    "action_id": call.action_id,
                    "source_event_id": source.record_id,
                    "result_record_id": observed.record_id,
                    "available": actual is not None,
                    "sha256": digest_bytes(payload),
                }
            )
    answer_payload = (
        b"Please provide the original source again; I cannot verify the answer."
        if strategy == "unavailable"
        else solve_task(template, payloads)
    )
    answer = add("answer", worker, "assistant", answer_payload, scored=True)
    dependencies.extend(
        Dependency(e.record_id, answer.record_id, "actual_evidence") for e in evidence
    )
    # Releasing detail is itself a supervised choice, after the answer has been produced.
    call_context("release-worker", worker, 0, 0)
    episode = Episode(
        identity,
        tuple(records),
        Lineage(SOURCE_VERSION, SOURCE_VERSION, family, frozen_split=split),
        stream="context_management",
        actions=tuple(actions),
        dependencies=tuple(dependencies),
        supervision=Supervision(
            outcome="incomplete" if strategy == "unavailable" else "success",
            verification="local_assertions",
            annotations_json=canonical_json(
                {
                    "origin": "authored_executed",
                    "task_template": template,
                    "context_strategy": strategy,
                    "context_profile": [local_events, cross_events],
                    "context_request": request.record_id,
                    "context_sources": [s.record_id for s in sources],
                    "context_answer": answer.record_id,
                    "context_read_receipts": receipts,
                }
            ),
        ),
        tracks=tuple(
            TrackSpec(address, "assistant", description=purpose)
            for address, purpose in {
                MAIN_TRACK: MAIN_TRACK_PURPOSE,
                worker: instruction,
                provider: "Preserve original source evidence",
                busy: "Independent unrelated work",
            }.items()
        ),
        meridians=tuple(groups),
        tools_json=canonical_json(lifecycle_tools()),
    )
    validate_curriculum_episode(episode)
    return episode


def validate_curriculum_episode(episode):
    """Check computed targets and direct evidence visibility before every scored answer.

    This checks public-event visibility. The runtime may also commit retrieval
    events; the runtime regression checks separately exercise that path.
    """
    annotations = json.loads(episode.supervision.annotations_json)
    if episode.lineage.source_id != SOURCE_VERSION:
        return set()
    records = {r.record_id: r for r in episode.records}
    template, strategy = annotations["task_template"], annotations["context_strategy"]
    request = json.loads(records[annotations["context_request"]].payload)
    if request["task"] != template or strategy not in STRATEGIES:
        raise ValueError("Context task declaration differs from public request")
    sources = [records[address] for address in annotations["context_sources"]]
    answer = records[annotations["context_answer"]]
    expected = (
        b"Please provide the original source again; I cannot verify the answer."
        if strategy == "unavailable"
        else solve_task(template, [s.payload for s in sources])
    )
    if answer.payload != expected or not answer.policy_mask:
        raise ValueError("Context answer does not follow the actual source evidence")
    policy = ContextPolicy()
    owners, positions, addresses = [], [], []
    by_record = {e.record_id: e for group in episode.meridians for e in group.events}
    reader = by_record[answer.record_id].track_id
    initial_reader_window = None
    dependencies = {}
    for dependency in episode.dependencies:
        dependencies.setdefault(dependency.reader_id, []).append(dependency.writer_id)
    reached_answer = False
    answer_policy = None
    for group in episode.meridians:
        for event in group.events:
            record = records[event.record_id]
            window = context_announcement(record.payload, record.role)
            if window is not None:
                if event.track_id == reader and initial_reader_window is None:
                    initial_reader_window = (window["local_events"], window["cross_events"])
                if any(address not in addresses for address in window["pinned_events"]):
                    raise ValueError("Context pin refers to unavailable future evidence")
                policy = policy.configure(event.track_id, **window)
            if record.action_id:
                action = json.loads(record.payload)
                arguments = action["arguments"]
                visible_text = b"\n".join(
                    records[address].payload
                    for address, keep in zip(
                        addresses,
                        policy.visible(
                            event.track_id, owners, positions, addresses, len(addresses)
                        ),
                        strict=True,
                    )
                    if keep
                )
                handles = [arguments["address"]]
                if action["name"] == "track_fetch":
                    handles.append(arguments["source_event"])
                handles.extend(arguments.get("pinned_events", ()))
                if any(handle.encode() not in visible_text for handle in handles):
                    raise ValueError(
                        "Context action names an address absent from its visible inputs"
                    )
            if record.record_id == answer.record_id or any(
                d.kind == "summary_source" and d.reader_id == record.record_id
                for d in episode.dependencies
            ):
                visible = {
                    address
                    for address, keep in zip(
                        addresses,
                        policy.visible(
                            event.track_id, owners, positions, addresses, len(addresses)
                        ),
                        strict=True,
                    )
                    if keep
                }
                if not set(dependencies.get(record.record_id, ())) <= visible:
                    raise ValueError("Context target depends on evidence outside its direct window")
                reached_answer |= record.record_id == answer.record_id
                if record.record_id == answer.record_id:
                    answer_policy = policy
        for event in group.events:
            owners.append(event.track_id)
            positions.append(len(addresses))
            addresses.append(event.record_id)
    if not reached_answer or not all(source.record_id in by_record for source in sources):
        raise ValueError("Context task lacks original evidence or answer")
    if initial_reader_window != tuple(annotations["context_profile"]):
        raise ValueError("Context profile differs from its public window announcement")
    source_ids = {s.record_id for s in sources}
    evidence_ids = set(dependencies.get(answer.record_id, ()))
    if not evidence_ids:
        raise ValueError("Context answer lacks evidence dependencies")
    if strategy in {"local", "pin", "recent"} and evidence_ids != source_ids:
        raise ValueError("Direct context answer must use its original evidence")
    if strategy == "local" and any(by_record[s.record_id].track_id != reader for s in sources):
        raise ValueError("Local context evidence belongs to a different track")
    if strategy == "pin" and not all((reader, s) in answer_policy.pins for s in source_ids):
        raise ValueError("Pinned context answer lacks retained references")
    if strategy == "summary":
        if len(evidence_ids) != 1:
            raise ValueError("Context summary must be a single compact handoff")
        summary = records[next(iter(evidence_ids))]
        if (
            not summary.policy_mask
            or summary.role != "assistant"
            or summary.payload != expected
            or set(dependencies.get(summary.record_id, ())) != source_ids
        ):
            raise ValueError("Context summary does not follow its original evidence")
    if strategy in {"fetch", "unavailable"}:
        receipts = annotations["context_read_receipts"]
        if {r["source_event_id"] for r in receipts} != source_ids or len(receipts) != len(sources):
            raise ValueError("Context reads must cover exactly the requested original sources")
        actions = {a.action_id: a for a in episode.actions}
        order = {r.record_id: i for i, r in enumerate(episode.records)}
        for receipt in receipts:
            action = actions[receipt["action_id"]]
            observed = records[receipt["result_record_id"]]
            original = records[receipt["source_event_id"]]
            arguments = json.loads(action.arguments_json)
            available = strategy == "fetch"
            if (
                action.tool_name != "track_fetch"
                or not action.execution_id
                or observed.result_of != action.action_id
                or observed.execution_id != action.execution_id
                or receipt["available"] is not available
                or digest_bytes(observed.payload) != receipt["sha256"]
                or arguments != {"address": reader, "source_event": original.record_id}
                or (available and observed.payload != original.payload)
                or not order[original.record_id]
                < order[action.record_ids[0]]
                < order[observed.record_id]
                or order[observed.record_id] >= order[answer.record_id]
            ):
                raise ValueError("Context read receipt does not match executed evidence")
        if evidence_ids != {r["result_record_id"] for r in receipts}:
            raise ValueError("Context answer must use its actual read results")
    if strategy == "unavailable" and episode.supervision.verified_success:
        raise ValueError("Missing evidence cannot count as successful task completion")
    order = {record.record_id: i for i, record in enumerate(episode.records)}
    cleanup = [
        action
        for action in episode.actions
        if action.tool_name == "track_context"
        and action.execution_id
        and json.loads(action.arguments_json)
        == {"address": reader, "local_events": 0, "cross_events": 0, "pinned_events": []}
        and order[action.record_ids[0]] > order[answer.record_id]
    ]
    if not cleanup or policy.window(reader) != (0, 0):
        raise ValueError("Context task must release completed worker history after answering")
    result = {
        "context_task_verified",
        "context_released",
        "context_strategy_" + strategy,
        "context_template_" + template,
    }
    if tuple(annotations["context_profile"]) == (DEFAULT_LOCAL_EVENTS, DEFAULT_CROSS_EVENTS):
        result.add("context_default_windows")
    return result


def curriculum_episodes(families=32, *, profiles=WINDOW_PROFILES, templates=TEMPLATES):
    """Split by original task, never by retention strategy or window variant."""
    if type(families) is not int or families < 8:
        raise ValueError("Use at least eight original tasks per template for three splits")
    for index in range(families):
        split = "test" if index % 8 == 0 else "validation" if index % 8 == 1 else "train"
        for template in templates:
            for local, cross in profiles:
                for strategy in STRATEGIES:
                    yield curriculum_episode(
                        index,
                        template=template,
                        strategy=strategy,
                        split=split,
                        local_events=local,
                        cross_events=cross,
                    )


def curriculum_summary(episodes):
    counts = Counter()
    families = {}
    for episode in episodes:
        annotations = json.loads(episode.supervision.annotations_json)
        split = episode.lineage.frozen_split
        counts[f"{split}|{annotations['task_template']}|{annotations['context_strategy']}"] += 1
        families.setdefault(split, set()).add(episode.lineage.family_id)
    return {
        "examples": dict(sorted(counts.items())),
        "original_tasks": {split: len(ids) for split, ids in sorted(families.items())},
    }
