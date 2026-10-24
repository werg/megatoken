#!/usr/bin/env python3
"""Real-weight controlled speech/tool/source/interruption acceptance experiment.

The host fixes requests and arrival order. Recorded speaker audio and the actual
speech decoder are used, and selected calls execute against a real immutable
snapshot. This measures protocol correctness, not spontaneous task competence or
whether the generated speech is intelligible or appropriate to the recording.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path
from threading import Event

import torch

from megatoken.contracts import ROLE_TO_ID
from megatoken.controls import build_control_payload
from megatoken.data import DedupConfig, DedupStore
from megatoken.data.media_assets import MEDIA_RECORD_TYPE, MediaRecord
from megatoken.data.media_prepare import MediaPreparer
from megatoken.data.schema import (
    Episode,
    LogicalAction,
    Meridian,
    PhysicalRecord,
    Supervision,
    TrackEvent,
    TrackSpec,
    canonical_json,
    digest_bytes,
)
from megatoken.data.storage import PreparedDataset, export_prepared
from megatoken.environments import SnapshotEnvironment, SnapshotToolAdapter
from megatoken.generation.structure import ActionSpec
from megatoken.io import sha256_file, write_json
from megatoken.model.media import MediaOutputSpec
from megatoken.model.multitrack import MultiTrackGenerator, ObservedEvent
from megatoken.multitrack_runtime import ConcurrentToolRunner, MultiTrackRuntime
from megatoken.retrieval.collection import SourceCollection
from megatoken.served import ServedBundle
from megatoken.tool_schema import compile_tool_spec, parse_tool_action
from megatoken.training.history import NativeHistory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--packs", type=Path, required=True)
    parser.add_argument("--recording-data", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--split", choices=("train", "validation"), default="train")
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=False)
    bundle = ServedBundle(args.bundle, packs=args.packs, device=args.device)
    media = MediaPreparer(args.packs, args.root / "data", device=args.device)
    runtime = MultiTrackRuntime(
        MultiTrackGenerator(bundle.model),
        chunking=bundle.chunking,
        record_policy_context=True,
    )
    runner = ConcurrentToolRunner(3)
    journal, observed_media, tool_receipts = [], {}, {}
    release_voice, voice_entered, release_slow = Event(), Event(), Event()
    original_observe, original_render = runtime.observe, runtime.generator.render_selected

    def observe(session, events, *, effort=1):
        events = tuple(
            replace(event, event_id=event.event_id or f"observed:{session.revision}:{index}")
            for index, event in enumerate(events)
        )
        result = original_observe(session, events, effort=effort)
        journal.append(("observed", events, time.monotonic_ns()))
        return result

    def render(selected):
        if selected.tape.event_ids == ("initial-voice",):
            voice_entered.set()
            if not release_voice.wait(120):
                raise TimeoutError("Controlled voice schedule was not released")
        return original_render(selected)

    runtime.observe, runtime.generator.render_selected = observe, render
    session = runtime.open(effort=1)
    collection = SourceCollection.create(
        args.root / "source", name="live", teacher=bundle.model.config.media_extractor
    )
    try:
        with PreparedDataset(args.recording_data, split=args.split) as dataset:
            original = None
            original_receipts = dataset.manifest["sources"]
            for candidate in dataset.iter_episodes():
                available = [
                    e
                    for g in candidate.meridians
                    for e in g.events
                    if e.modality == "audio" and e.role == "user"
                ]
                if len(available) >= 2 and len({e.speaker for e in available if e.speaker}) >= 2:
                    original = candidate
                    break
            if original is None:
                raise ValueError("Provide a checked recording with two distinct speaker channels")
            by_id = {r.record_id: r for r in original.records}
            inputs = [
                e
                for g in original.meridians
                for e in g.events
                if e.modality == "audio" and e.role == "user"
            ][:2]
            if len(inputs) != 2 or inputs[0].speaker == inputs[1].speaker:
                raise ValueError("Provide a checked recording with two distinct speaker channels")
            for index, event in enumerate(inputs):
                descriptor = MediaRecord.from_bytes(by_id[event.record_id].payload)
                dataset.assets.copy_record(descriptor, media.store)
                observed_media[f"speaker-{index}"] = (descriptor, event)

        def incoming(index):
            descriptor, event = observed_media[f"speaker-{index}"]
            features = torch.from_numpy(media.store.load(descriptor.source)).to(args.device)[None]
            runtime.observe(
                session,
                (
                    ObservedEvent(
                        f"speaker-{index}",
                        "audio",
                        features,
                        ROLE_TO_ID["user"],
                        extractor=descriptor.source.extractor,
                        event_id=f"speaker-{index}",
                    ),
                ),
                effort=1,
            )

        incoming(0)
        speech_spec = MediaOutputSpec("audio", sample_count=24000, sample_rate=24000)

        def select(**kwargs):
            revision = runtime.start_generation(session, effort=1, **kwargs)
            journal.append(("selected", revision, time.monotonic_ns()))
            write_json(
                args.root / f"selection-{revision}.json",
                json.loads(session.rendering[revision].selection.tape.to_json()),
            )
            return revision

        voice = select(tracks=("assistant",), event_ids=("initial-voice",), specs=(speech_spec,))
        assert voice_entered.wait(10)
        incoming(1)
        assert not session.rendering[voice].future.done()

        environment = SnapshotEnvironment()
        initial = environment.reset({"initial_files": {"left.txt": "11", "right.txt": "19"}})
        adapter = SnapshotToolAdapter(environment, initial.state_id)
        schemas = [
            {
                "name": "read",
                "parameters": {
                    "type": "object",
                    "const": {"path": path},
                    "properties": {"path": {"type": "string", "enum": [path]}},
                    "required": ["path"],
                    "additionalProperties": False,
                },
            }
            for path in ("left.txt", "right.txt")
        ]
        tools_revision = select(
            tracks=("read-left", "read-right"),
            event_ids=("read-left", "read-right"),
            specs=tuple(
                compile_tool_spec(json.dumps([schema]), max_tokens=512) for schema in schemas
            ),
            controls=tuple(
                build_control_payload(tools_json=json.dumps([schema])) for schema in schemas
            ),
        )
        session.rendering[tools_revision].future.result(timeout=120)
        tools = runtime.poll_generation(session, tools_revision).result
        if tools is not None:
            write_json(args.root / "decoded-tools.json", json.loads(tools.tape.to_json()))
        if tools is None or not all(tools.tape.valid):
            raise RuntimeError("The actual constrained tool decoder did not complete both calls")
        intervals = []

        class ScheduledRead:
            def execute_selected(self, *, action_id, payload):
                started = time.monotonic_ns()
                if parse_tool_action(payload)[1]["path"] == "left.txt" and not release_slow.wait(
                    120
                ):
                    raise TimeoutError("Controlled tool schedule was not released")
                result = adapter.execute_selected(action_id=action_id, payload=payload)
                receipt = json.loads(result)
                intervals.append(
                    {
                        "action_id": action_id,
                        "execution_id": receipt["execution_id"],
                        "started_at_ns": started,
                        "completed_at_ns": time.monotonic_ns(),
                    }
                )
                return result

        runtime.dispatch_tools(session, tools, runner, {"read": ScheduledRead()})
        fast_id = f"{tools.tape.meridian_id}:read-right"
        runner.pending[fast_id].result(timeout=120)
        ready, failed = runtime.collect_tools(session, runner, effort=1)
        assert set(ready) == {fast_id} and not failed
        tool_receipts.update(ready)
        document = (
            b"The current evidence contains two independent numbers: 11 and 19. Their sum is 30."
        )

        def publish():
            _, embedding = media.text(document)
            row = collection.add_text(
                document, media.store.load(embedding), origin="controlled-evidence.txt"
            )
            collection.reindex(cells=1)
            return collection.publication(row)

        source_action = runtime.prepare_source(
            session, track="source-worker", event_id="document", runner=runner, prepare=publish
        )
        runner.pending[source_action].result(timeout=120)
        source_ready, failed = runtime.collect_tools(session, runner, effort=1)
        assert source_action in source_ready and not failed
        runtime.observe(
            session,
            (
                ObservedEvent(
                    "document",
                    "text",
                    document,
                    ROLE_TO_ID["document"],
                    event_id="admitted-document",
                ),
            ),
            effort=1,
        )
        runtime.observe(
            session,
            (
                ObservedEvent(
                    "user",
                    "text",
                    b"Stop speaking. Wait for the remaining read, then answer using the new document.",
                    event_id="interruption",
                ),
            ),
            effort=1,
        )
        runtime.stop_delivery(session, voice)
        release_voice.set()
        session.rendering[voice].future.result(timeout=120)
        assert runtime.poll_generation(session, voice).status == "cancelled"
        release_slow.set()
        for future in runner.pending.values():
            future.result(timeout=120)
        late, failed = runtime.collect_tools(session, runner, effort=1)
        assert not failed and len(late) == 1
        tool_receipts.update(late)
        assert {json.loads(raw)["observation"]["content"] for raw in tool_receipts.values()} == {
            "11",
            "19",
        }
        reference = "admitted-document"
        if reference not in session.state.history_event_ids:
            # Small diagnostic models have deliberately tiny retained history.
            # Re-read the still-published item instead of accepting an evicted
            # address or treating its spelling as a substitute for its evidence.
            reference = "readmitted-document"
            runtime.observe(
                session,
                (
                    ObservedEvent(
                        "document",
                        "text",
                        document,
                        ROLE_TO_ID["document"],
                        event_id=reference,
                    ),
                ),
                effort=1,
            )
        resumed = select(
            tracks=("assistant", "assistant"),
            event_ids=("resumed-voice", "caption"),
            specs=(speech_spec, ActionSpec(min_tokens=1, max_tokens=16)),
            controls=(build_control_payload(reference_event_ids=(reference,)),) * 2,
        )
        session.rendering[resumed].future.result(timeout=120)
        completed = runtime.poll_generation(session, resumed).result
        assert completed is not None and all(completed.tape.valid)
        runtime.delivered(session, resumed, ("resumed-voice",), extents={"resumed-voice": 12000})
        playback = runtime.stop_delivery(session, resumed)

        records, groups, tracks, actions, tapes = [], [], {}, [], []
        inverse_roles = {v: k for k, v in ROLE_TO_ID.items()}
        for kind, value, when in journal:
            entries = []
            cursor, number = len(records), len(groups)
            if kind == "selected":
                work = session.rendering[value]
                tape = work.selection.tape if value == voice else work.future.result().tape
                bindings = []
                for index, event_id in enumerate(tape.event_ids):
                    if value == voice or not tape.valid[index] or not tape.payloads[index]:
                        bindings.append(None)
                        continue
                    modality, payload = tape.modalities[index], tape.payloads[index]
                    media_type = "text/plain"
                    if modality != "text":
                        payload = media.rendered(
                            work.future.result().media_outputs[index], speech_spec
                        ).to_bytes()
                        media_type = MEDIA_RECORD_TYPE
                    action_id = None
                    if value == tools_revision:
                        action_id = f"{tape.meridian_id}:{event_id}"
                        name, arguments, _, _ = parse_tool_action(payload)
                        receipt = json.loads(tool_receipts[action_id])
                        actions.append(
                            LogicalAction(
                                action_id,
                                event_id,
                                name,
                                canonical_json(arguments),
                                (event_id,),
                                digest_bytes(payload),
                                execution_id=receipt["execution_id"],
                            )
                        )
                    records.append(
                        PhysicalRecord(
                            event_id,
                            "assistant",
                            payload,
                            turn_index=number,
                            policy_mask=value == tools_revision,
                            provenance="student",
                            media_type=media_type,
                            emitted_ids=tape.emitted_ids[index] if modality == "text" else None,
                            codec_fingerprint=bundle.codec.fingerprint
                            if modality == "text"
                            else None,
                            action_id=action_id,
                        )
                    )
                    entries.append(
                        TrackEvent(event_id, tape.tracks[index], modality, number, "assistant")
                    )
                    bindings.append(event_id)
                entry = {
                    "record_cursor": cursor,
                    "record_ids": bindings,
                    "tape": json.loads(tape.to_json()),
                }
                if value == tools_revision:
                    entry["outcome_credit"] = {
                        "reward": 1.0,
                        "baseline": 0.5,
                        "selected_events": 2,
                        "verification": "local_assertions",
                    }
                tapes.append(entry)
            else:
                for event in value:
                    role = inverse_roles[event.role_id]
                    payload, media_type, metadata = event.payload, "text/plain", {}
                    if event.event_id in observed_media:
                        descriptor, original_event = observed_media[event.event_id]
                        payload, media_type = descriptor.to_bytes(), MEDIA_RECORD_TYPE
                        metadata = {
                            "media_span": original_event.media_span,
                            "speaker": original_event.speaker,
                            "direction": "incoming",
                            "source_revision": original_event.source_revision
                            or descriptor.source.source_family,
                            "available_at_ns": when,
                        }
                    elif event.modality == "text":
                        try:
                            receipt = json.loads(payload).get("source_ready")
                        except (ValueError, AttributeError):
                            receipt = None
                        if receipt is not None:
                            metadata = {
                                "source_revision": f"{receipt['item_id']}:{receipt['revision']}",
                                "available_at_ns": when,
                            }
                    action_id = next(
                        (
                            identity
                            for identity in tool_receipts
                            if event.event_id.startswith(f"result:{identity}:part:")
                        ),
                        None,
                    )
                    records.append(
                        PhysicalRecord(
                            event.event_id,
                            role,
                            payload,
                            turn_index=number,
                            media_type=media_type,
                            provenance="natural"
                            if event.event_id in observed_media
                            else "executed",
                            result_of=action_id,
                            execution_id=None
                            if action_id is None
                            else json.loads(tool_receipts[action_id])["execution_id"],
                        )
                    )
                    entries.append(
                        TrackEvent(
                            event.event_id, event.track, event.modality, number, role, **metadata
                        )
                    )
            if entries:
                for event in entries:
                    tracks.setdefault(event.track_id, TrackSpec(event.track_id, event.role))
                groups.append(Meridian(number, tuple(entries), selected_at_ns=when))
        episode = Episode(
            "controlled-duplex-runtime",
            tuple(records),
            replace(
                original.lineage,
                source_id="controlled-duplex-execution",
                task_id="controlled-protocol-v1",
            ),
            actions=tuple(actions),
            tracks=tuple(tracks.values()),
            meridians=tuple(groups),
            stream="duplex_execution",
            supervision=Supervision(
                outcome="success",
                verification="local_assertions",
                reward=1.0,
                annotations_json=canonical_json(
                    {
                        "native_action_tapes": tapes,
                        "execution_intervals": intervals,
                        "host_controlled_schedule": True,
                        "quality_established": False,
                        "verification_scope": "immutable read effects, causal arrival, cancellation and partial delivery only",
                        "availability_clock": "session-monotonic; original wall-clock publication retained in receipt bytes",
                    }
                ),
            ),
        )
        NativeHistory.from_episode(episode, bundle.model)
        with DedupStore(
            args.root / "spool.sqlite", DedupConfig(train_fraction=1, validation_fraction=0)
        ) as store:
            store.add(episode, bundle.codec)
            store.freeze()
            exported = export_prepared(
                store,
                args.root / "data",
                codec=bundle.codec,
                source_receipts=[
                    {
                        "recording_data": str(args.recording_data),
                        "student_model_sha256": bundle.manifest["model_sha256"],
                        "origin": "controlled_executed_protocol_with_real_recorded_channels",
                        "recording_sources": original_receipts,
                        "recording_manifest_sha256": sha256_file(
                            args.recording_data / "manifest.json"
                        ),
                    }
                ],
                pipeline_fingerprint=sha256_file(__file__),
                chunking=bundle.chunking,
            )
        write_json(
            args.root / "acceptance.json",
            {
                "protocol_passed": True,
                "quality_established": False,
                "prepared_fingerprint": exported["fingerprint"],
                "coverage": exported["behavior_coverage"],
                "delivery_progress": playback.delivery_progress,
                "delivery_log": playback.delivery_log,
                "host_controlled_requests_and_schedule": True,
            },
        )
    finally:
        release_voice.set()
        release_slow.set()
        runner.close()
        runtime.close()
        collection.close()


if __name__ == "__main__":
    main()
