"""Collect coordinator teaching through the serving lifecycle and admitted children.

The coordinator is an explicit authored orchestration policy. Guest labels come
only from independently source/native admitted executions. Replaying those
labels does not call native adapters, sample a model, or optimize weights.
"""

from __future__ import annotations

import json
from dataclasses import asdict, replace

from megatoken.contracts import ROLE_TO_ID
from megatoken.lifecycle import ExecutionBinding, lifecycle_tools
from megatoken.model.multitrack import ObservedEvent

from .host_operations import CONTRACT_DIGEST
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
    encode_record,
    stable_id,
)
from .semantic_admission import native_admission, validate_native_visibility
from .semantic_contract import FIXTURE_DIGEST, SCHEMA_DIGEST
from .semantic_corpus import public_digest, verify_admission
from .semantic_harness_v4 import adapt_selected_harness_v4

ORCHESTRATION_POLICY = {
    "schema": "semantic-orchestration-policy/4",
    "name": "spawn-observed-child-join-publish",
    "authority": "coordinator",
    "decisions": [
        "Create one child from its already observed immutable execution proposal.",
        "Join using the runtime-issued completion dependency from creation acknowledgement.",
        "After delivered child completion, publish that exact return or throw as progress.",
        "Repeat for each observed child proposal, then complete the coordinator.",
    ],
}
POLICY_DIGEST = digest_bytes(canonical_json(ORCHESTRATION_POLICY).encode())


class _Recorder:
    def __init__(self, session):
        self.session = session
        self.seen = set()
        self.records, self.meridians, self.metadata = [], [], {}
        self.actions, self.dependencies = [], []
        self.roles = {number: name for name, number in ROLE_TO_ID.items()}

    def drain(self):
        batch, message = [], []
        for identity, (event, _) in self.session.context_archive.events.items():
            if identity in self.seen:
                continue
            message.append(event)
            if event.terminal:
                batch.append(message)
                message = []
        if message:
            raise ValueError("Collection cannot end inside a partially published native message")
        result = []
        for message in batch:
            logical = "collection:" + message[0].event_id
            raw = b"".join(event.payload for event in message)
            if self.roles[message[0].role_id] == "system":
                try:
                    if "context_window" in json.loads(raw):
                        logical = "semantic:context:" + logical
                except (ValueError, TypeError):
                    pass
            for index, event in enumerate(message):
                identity = event.event_id
                metadata = self.metadata.get(identity, {})
                row = PhysicalRecord(
                    identity,
                    self.roles[event.role_id],
                    event.payload,
                    turn_index=len(self.meridians),
                    logical_id=logical,
                    part_index=index,
                    part_count=len(message),
                    policy_mask=metadata.get("policy_mask", False),
                    provenance="teacher" if metadata.get("policy_mask") else "executed",
                    action_id=metadata.get("action_id"),
                    result_of=metadata.get("result_of"),
                    execution_id=metadata.get("execution_id"),
                )
                # Independently framed guest records retain their exact logical
                # boundaries even when this drain contains several messages.
                if "original" in metadata:
                    original = metadata["original"]
                    row = replace(
                        row,
                        logical_id=metadata["logical_id"],
                        part_index=original.part_index,
                        part_count=original.part_count,
                    )
                self.records.append(row)
                self.meridians.append(
                    Meridian(
                        len(self.meridians),
                        (
                            TrackEvent(
                                row.record_id, event.track, "text", len(self.meridians), row.role
                            ),
                        ),
                    )
                )
                self.seen.add(identity)
                result.append(row)
        return result


def collect_semantic_lifecycle(
    documents,
    *,
    runtime,
    codec,
    producer_revision,
    consumer_revision,
    source_verifier,
    max_history=1024,
    max_positions=1024,
    max_active=1,
):
    """Execute an authored coordinator against separately certified guest episodes.

    Children must be collected with distinct invocation IDs, rather than renamed
    after certification. One guest activation per child keeps ownership explicit;
    multi-event handler sessions remain their own certified corpus episodes.
    """
    if runtime.generator.generator.codec.fingerprint != codec.fingerprint:
        raise ValueError("Collector runtime and native admission must use the same tokenizer")
    if not callable(source_verifier) or not documents:
        raise ValueError("Lifecycle collection requires independently verified guest children")
    prepared, audits, identities = [], [], set()
    for document in documents:
        if document.get("sidecar", {}).get("representation") != "compact":
            raise ValueError("Lifecycle child differs from the declared certified representation")
        source = source_verifier(document)
        native = native_admission(
            document,
            codec=codec,
            consumer_revision=consumer_revision,
            max_positions=max_positions,
            max_history=max_history,
        )
        verify_admission(
            document,
            source,
            native,
            producer_revision=producer_revision,
            consumer_revision=consumer_revision,
            codec=codec,
        )
        episode, imported = adapt_selected_harness_v4(
            document,
            codec=codec,
            max_positions=max_positions,
            max_history=max_history,
            admission_verifier=lambda _: True,
        )
        invocations = [
            item for item in document["public"]["timeline"] if item["kind"] == "invocation"
        ]
        if len(invocations) != 1 or invocations[0]["parentInvocationId"] is not None:
            raise ValueError(
                "Coordinator children require one independently admitted root activation"
            )
        invocation = invocations[0]
        if invocation["invocationId"] in identities:
            raise ValueError("Children need unique invocation identities at source collection")
        identities.add(invocation["invocationId"])
        program = next(
            item
            for item in document["public"]["timeline"]
            if item["kind"] == "program" and item["programId"] == invocation["programId"]
        )
        terminal = next(
            (
                {key: completion[key] for key in ("kind", "value")}
                for completion in imported.completions
                if completion["invocationId"] == invocation["invocationId"]
            ),
            None,
        )
        if terminal is None:
            raise ValueError("Child lacks an admitted root completion")
        prepared.append((document, episode, program, invocation, terminal))
        audits.append(
            {"episodeDigest": public_digest(document), "source": source, "native": native}
        )
    splits = {episode.lineage.frozen_split for _, episode, _, _, _ in prepared}
    if len(splits) != 1 or not splits <= {"train", "validation", "test"}:
        raise ValueError("Lifecycle children must retain one declared whole-family split")
    frozen_split = next(iter(splits))
    session = runtime.open(
        metadata={"task": "semantic-orchestration/4", "modality": "text"},
        effort=1,
        memory_writes=False,
    )
    session.tracks.max_active = max_active
    recorder = _Recorder(session)
    grants = sorted(
        {grant for _, _, program, _, _ in prepared for grant in program["capabilities"]}
    )
    coordinator = runtime.track_action(
        session,
        "create",
        purpose="Execute the observed child tasks in order",
        completion="Every child result is delivered and published",
        capabilities=("text", *grants),
        budget=100000,
    )
    recorder.drain()
    proposals = []
    for _, _, program, invocation, _ in prepared:
        reference = runtime.register_execution_context(
            session, program=program, invocation=invocation
        )
        binding = ExecutionBinding(
            program_id=program["programId"],
            source_digest=program["digest"],
            entry=invocation["entry"],
            invocation_id=invocation["invocationId"],
            realm_id=invocation["realmId"],
            trigger_id=invocation["triggerId"],
            input_ref=reference,
            realm_policy=invocation["realmPolicy"],
            completion_mode=invocation["completionPolicy"],
            capability_grants=tuple(program["capabilities"]),
        )
        proposals.append(binding)
    runtime.observe_text(
        session,
        coordinator.address,
        canonical_json(
            {
                "schema": "semantic-orchestration-input/4",
                "policy": ORCHESTRATION_POLICY,
                "children": [
                    {"execution": asdict(binding), "program": program, "invocation": invocation}
                    for binding, (_, _, program, invocation, _) in zip(
                        proposals, prepared, strict=True
                    )
                ],
            }
        ).encode(),
        role="user",
        effort=1,
    )
    recorder.drain()
    decisions, checkpoints, child_receipts = 0, [], []

    def act(name, arguments):
        nonlocal decisions
        payload = canonical_json({"name": name, "arguments": arguments}).encode()
        group = runtime.commit_lifecycle_teacher(
            session,
            track=coordinator.address,
            payload=payload,
            policy_digest=POLICY_DIGEST,
            decision=decisions,
        )
        decisions += 1
        identity = group.tape.event_ids[0]
        action_id = f"{group.tape.meridian_id}:{identity}"
        for event_id in session.context_archive.events:
            if event_id.startswith(identity + ":part:"):
                recorder.metadata[event_id] = {"policy_mask": True, "action_id": action_id}
        selected = recorder.drain()
        owned = tuple(row.record_id for row in selected if row.action_id == action_id)
        if not owned:
            raise ValueError("Coordinator decision was not committed before its tool execution")
        available = [row.record_id for row in recorder.records if not row.policy_mask]
        recorder.dependencies.extend(
            Dependency(writer, owned[0], "available_observation") for writer in available[-1:]
        )
        outcomes = runtime.execute_lifecycle(session, group)
        if len(outcomes) != 1 or "error" in outcomes[0]:
            raise ValueError("Authored coordinator lifecycle action failed: " + str(outcomes))
        for event_id in session.context_archive.events:
            if event_id.startswith("result:" + action_id + ":part:"):
                recorder.metadata[event_id] = {"result_of": action_id, "execution_id": action_id}
        observed = recorder.drain()
        recorder.actions.append(
            LogicalAction(
                action_id,
                action_id,
                name,
                canonical_json(arguments),
                owned,
                digest_bytes(payload),
                execution_id=action_id,
                schema_id="semantic-orchestration/4",
            )
        )
        checkpoints.append(
            {
                "decision": decisions - 1,
                "action": action_id,
                "active": sum(track.status == "active" for track in session.tracks.tracks.values()),
                "lifecycle": session.tracks.checkpoint(),
            }
        )
        return outcomes[0], observed

    for ordinal, (binding, (document, episode, _, invocation, terminal)) in enumerate(
        zip(proposals, prepared, strict=True)
    ):
        created, bootstrap = act(
            "track_create",
            {
                "purpose": "Execute observed guest " + binding.invocation_id,
                "completion": "Publish its independently admitted return or throw",
                "budget": 4096,
                "capabilities": ["text", *binding.capability_grants],
                "execution": asdict(binding),
            },
        )
        child = created["track"]["address"]
        dependency = created["completion_dependency"]
        act("track_join", {"address": coordinator.address, "completion_dependency": dependency})
        if not session.tracks.selectable(child, session.completed_events):
            raise ValueError("Typed join did not admit ready child capacity")
        runtime.configure_context(
            session,
            track=child,
            local_events=max_history,
            cross_events=max_history,
            pinned_events=(),
        )
        recorder.drain()
        mapped = {}
        # Source/input were actually published by execute_lifecycle after the
        # creation acknowledgement. All later child bytes remain byte exact.
        for parts in episode.iter_logical_records():
            raw = b"".join(part.payload for part in parts)
            item = None if parts[0].policy_mask else json.loads(raw)
            if item is not None and item.get("kind") in {"program", "invocation"}:
                matches = [row.record_id for row in bootstrap if row.role == "user"]
                if not matches:
                    raise ValueError("Child bootstrap failed to publish source and invocation")
                for part in parts:
                    mapped[part.record_id] = matches[-1]
                continue
            for part in parts:
                identity = f"child:{ordinal}:" + part.record_id
                mapped[part.record_id] = identity
                prefix = f"child:{ordinal}:"
                recorder.metadata[identity] = {
                    "original": part,
                    "logical_id": prefix + part.logical_id,
                    "policy_mask": part.policy_mask,
                    "action_id": prefix + part.action_id if part.action_id else None,
                    "result_of": prefix + part.result_of if part.result_of else None,
                    "execution_id": prefix + part.execution_id if part.execution_id else None,
                }
                if part.logical_id.startswith("semantic:context:"):
                    recorder.metadata[identity]["logical_id"] = "semantic:context:" + identity
                runtime.observe(
                    session,
                    (
                        ObservedEvent(
                            child,
                            "text",
                            part.payload,
                            ROLE_TO_ID[part.role],
                            terminal=part.part_index + 1 == part.part_count,
                            event_id=identity,
                        ),
                    ),
                    effort=1,
                )
            recorder.drain()
        prefix = f"child:{ordinal}:"
        recorder.actions.extend(
            replace(
                action,
                action_id=prefix + action.action_id,
                call_id=prefix + action.call_id,
                record_ids=tuple(mapped[record] for record in action.record_ids),
                execution_id=prefix + action.execution_id if action.execution_id else None,
            )
            for action in episode.actions
        )
        recorder.dependencies.extend(
            Dependency(mapped[edge.writer_id], mapped[edge.reader_id], edge.kind)
            for edge in episode.dependencies
            if mapped[edge.writer_id] != mapped[edge.reader_id]
        )
        returned = runtime.complete_execution(session, child, **terminal)
        delivery = recorder.drain()
        if not session.tracks.selectable(coordinator.address, session.completed_events):
            raise ValueError("Delivered child completion did not wake its coordinator")
        completion_rows = [row for row in delivery if row.role == "environment"]
        if not completion_rows or returned["dependency"] != dependency:
            raise ValueError("Child completion was confused with creation acknowledgement")
        child_receipts.append(
            {
                "child": child,
                "invocationId": invocation["invocationId"],
                "episodeDigest": public_digest(document),
                "completion": returned,
                "deliveryRecords": [row.record_id for row in completion_rows],
            }
        )
        act(
            "track_progress", {"address": coordinator.address, "progress": canonical_json(terminal)}
        )
    act("track_complete", {"address": coordinator.address})
    if len(recorder.records) > max_history:
        raise ValueError("Lifecycle corpus exceeds the declared native history budget")
    family = stable_id(
        "semantic-orchestration-family/4",
        POLICY_DIGEST,
        *(episode.lineage.family_id for _, episode, _, _, _ in prepared),
    )
    episode_id = stable_id(
        "semantic-orchestration/4", family, *(audit["episodeDigest"] for audit in audits)
    )
    ancestors = tuple(
        sorted({key for _, episode, _, _, _ in prepared for key in episode.lineage.family_keys})
    )
    annotations = {
        "semantic_geometry": {
            "schema": "semantic-native-geometry/4",
            "maxPositions": max_positions,
            "maxHistory": max_history,
        },
        "semantic_execution": {
            "schema": "semantic-execution/4",
            "continuation": "teacher_conditioned",
            "representation": "compact",
            "orchestration": "runtime-certified",
        },
        "orchestration_policy": {"digest": POLICY_DIGEST, "definition": ORCHESTRATION_POLICY},
        "component_admission": audits,
        "child_completions": child_receipts,
        "lifecycle_replay": checkpoints,
        "optimizerSteps": 0,
    }
    episode = Episode(
        episode_id,
        tuple(recorder.records),
        Lineage(
            "semantic-orchestration/4",
            consumer_revision,
            family,
            ancestor_ids=ancestors,
            task_id=family,
            frozen_split=frozen_split,
        ),
        actions=tuple(recorder.actions),
        dependencies=tuple(recorder.dependencies),
        meridians=tuple(recorder.meridians),
        tracks=tuple(
            TrackSpec(track.address, "assistant", description=track.purpose)
            for track in session.tracks.tracks.values()
        ),
        stream="semantic_execution",
        tools_json=canonical_json(lifecycle_tools()),
        supervision=Supervision(
            outcome="success",
            verification="local_assertions",
            reward=1.0,
            annotations_json=canonical_json(annotations),
        ),
    )
    validate_native_visibility(episode)
    encoded = [encode_record(record, codec) for record in episode.records]
    if any(item.P for item in encoded if not item.record.policy_mask):
        raise ValueError("Lifecycle observation acquired generation loss")
    receipt = {
        "schema": "semantic-lifecycle-admission/4",
        "episodeId": episode_id,
        "episodeDigest": digest_bytes(canonical_json(episode.to_dict()).encode()),
        "policyDigest": POLICY_DIGEST,
        "representation": "compact",
        "producerRevision": producer_revision,
        "consumerRevision": consumer_revision,
        "catalogDigest": CONTRACT_DIGEST,
        "schemaDigest": SCHEMA_DIGEST,
        "fixtureDigest": FIXTURE_DIGEST,
        "tokenizerIdentity": codec.fingerprint,
        "split": frozen_split,
        "children": len(prepared),
        "lifetimeTracks": len(session.tracks.tracks),
        "maxActive": max_active,
        "observedPeakActive": max(point["active"] for point in checkpoints),
        "lifecycleActions": decisions,
        "guestActions": sum(len(item[1].actions) for item in prepared),
        "childAdmissionValidated": True,
        "lossMaskValidated": True,
        "visibilityValidated": True,
        "nativeEffectsDuringReplay": 0,
        "optimizerSteps": 0,
        "admitted": True,
    }
    return episode, receipt
