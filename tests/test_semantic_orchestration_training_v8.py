"""Source-qualified lifecycle rows match the model-visible V8 runtime route."""

from __future__ import annotations

import base64
import gzip
import json
from pathlib import Path

from megatoken.controls import (
    build_request_control,
    canonical_semantic_observation,
    resolve_event_control,
    semantic_orchestration_ack_observation_v8,
    semantic_orchestration_input_v8,
    semantic_orchestration_instruction_v8,
)
from megatoken.data.native_v8 import prepare_native_v8_training
from megatoken.data.semantic_orchestration_training_v8 import (
    validate_orchestration_records_v8,
    validate_orchestration_transcript_v8,
)
from megatoken.generation.codec import ByteCodec


def teacher_document():
    path = Path(__file__).parent / "fixtures/semantic-orchestration-teacher-v8.json.gz.b64"
    return json.loads(gzip.decompress(base64.b64decode(path.read_bytes())))


def logical_rows(episode):
    events = {event.record_id: event for meridian in episode.meridians for event in meridian.events}
    rows = []
    for parts in episode.iter_logical_records():
        rows.append(
            {
                "id": parts[0].logical_id or parts[0].record_id,
                "payload": b"".join(part.payload for part in parts),
                "parts": parts,
                "events": [events[part.record_id] for part in parts],
                "turn": parts[0].turn_index,
            }
        )
    return rows


def test_physical_orchestration_rows_match_runtime_ingress_bootstrap_ack_and_completion():
    document = teacher_document()
    validation = validate_orchestration_transcript_v8(document)
    codec = ByteCodec.byte_fallback()
    episode, training = prepare_native_v8_training(
        document, codec=codec, max_positions=256, max_history=8
    )
    assert training["orchestrationRecords"] == 4
    assert training["orchestrationTranscriptDigest"] == validation["transcriptDigest"]
    assert validate_orchestration_records_v8(episode, validation)
    rows = logical_rows(episode)
    by_id = {row["id"]: row for row in rows}
    transcript = validation["transcript"]
    public = document["public"]
    initial_public = {**public, "timeline": public["timeline"][:2]}
    initial_child = {
        "program": public["timeline"][0],
        "invocation": public["timeline"][1],
        "execution": transcript["ingress"]["execution"],
        "track": None,
        "completion": None,
    }
    assert initial_child == validation["initialChildren"][0]
    ingress = semantic_orchestration_input_v8(public=initial_public, children=[initial_child])
    assert ingress == {
        "schema": "semantic-orchestration-input/8",
        "task": (
            "Create each declared child; bootstrap is a separate host acknowledgement. Join its "
            "exact issued dependency, wait for its delivered completion, publish its exact "
            "kind/value as progress, then complete this coordinator."
        ),
        "sourceAdmissionClaim": False,
        "public": initial_public,
        "children": [initial_child],
    }
    ingress_bytes = canonical_semantic_observation(ingress)
    ingress_rows = [row for row in rows if row["payload"] == ingress_bytes]
    assert len(ingress_rows) == 1
    assert ingress_rows[0]["id"] == "semantic:v8:orchestration:ingress"
    assert ingress_rows[0]["events"][0].track_id == validation["coordinatorTrack"]
    assert ingress_rows[0]["events"][0].role == "user"

    # Runtime bootstrap republishes the exact source and input observations on
    # the newly reserved child track, in source-before-input order.
    source_bytes, invocation_bytes = (
        json.dumps(
            public["timeline"][index], sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        for index in (0, 1)
    )
    bootstrap_rows = []
    for expected in (source_bytes, invocation_bytes):
        matches = [row for row in rows if row["payload"] == expected]
        assert len(matches) == 1
        row = matches[0]
        assert all(event.track_id == validation["childTrack"] for event in row["events"])
        assert all(record.role == "user" for record in row["parts"])
        bootstrap_rows.append(row)
    bootstrap = transcript["events"][2]["receipt"]
    bootstrap_ack = canonical_semantic_observation(bootstrap)
    bootstrap_ack_row = next(row for row in rows if row["payload"] == bootstrap_ack)
    assert all(
        event.track_id == validation["coordinatorTrack"] for event in bootstrap_ack_row["events"]
    )
    assert bootstrap_rows[0]["turn"] < bootstrap_rows[1]["turn"] < bootstrap_ack_row["turn"]
    assert by_id[bootstrap_ack_row["id"]]["payload"] == bootstrap_ack

    # Model-visible commit observations use a stable projection with no
    # selection ID. The original teacher receipt and native API outcome remain
    # separately typed and retain their source-specific schemas.
    for event in transcript["events"]:
        if event["kind"] != "ack":
            continue
        ordinal, teacher_receipt = event["ordinal"], event["receipt"]
        dependency = teacher_receipt.get("completion_dependency")
        projected = semantic_orchestration_ack_observation_v8(
            track=teacher_receipt["track"], completion_dependency=dependency
        )
        assert projected["schema"] == "semantic-orchestration-ack-observation/8"
        assert projected["committed"] is True
        assert "selectionId" not in projected
        assert teacher_receipt["schema"] == "semantic-orchestration-teacher-commit/1"
        assert "selectionId" not in teacher_receipt
        assert projected["track"] == teacher_receipt["track"]
        assert projected.get("completion_dependency") == dependency
        ack_row = by_id[f"semantic:v8:orchestration:ack:{ordinal}"]
        assert ack_row["payload"] == canonical_semantic_observation(projected)
        assert all(event.track_id == validation["coordinatorTrack"] for event in ack_row["events"])

    completion_event = next(row for row in transcript["events"] if row["kind"] == "completion")
    delivered = canonical_semantic_observation(completion_event["delivered"])
    completed = canonical_semantic_observation(completion_event["receipt"])
    delivered_row = next(row for row in rows if row["payload"] == delivered)
    completed_row = next(row for row in rows if row["payload"] == completed)
    assert completion_event["delivered"]["schema"] == "semantic-completion/4"
    assert completion_event["receipt"]["schema"] == "semantic-child-completion/8"
    assert delivered_row["turn"] < completed_row["turn"]
    assert all(
        event.track_id == validation["coordinatorTrack"] for event in delivered_row["events"]
    )
    assert all(
        event.track_id == validation["coordinatorTrack"] for event in completed_row["events"]
    )

    # Source/input are first exposed to the coordinator in its one wrapper;
    # the duplicate rows above belong only to the child bootstrap history.
    for expected in (source_bytes, invocation_bytes):
        copies = [row for row in rows if row["payload"] == expected]
        assert len(copies) == 1
        assert copies[0]["events"][0].track_id == validation["childTrack"]
    for row in public["timeline"]:
        if row["mode"] == "generated":
            target = by_id[f"semantic:v8:target:{row['sequence']}"]
            assert all(event.track_id == validation["childTrack"] for event in target["events"])


def test_orchestration_target_controls_match_the_runtime_frame_including_continuations():
    document = teacher_document()
    validation = validate_orchestration_transcript_v8(document)
    episode, _training = prepare_native_v8_training(
        document, codec=ByteCodec.byte_fallback(), max_positions=256, max_history=8
    )
    expected_initial = build_request_control(instruction=semantic_orchestration_instruction_v8())
    expected_continuation = build_request_control(
        instruction=semantic_orchestration_instruction_v8(continuation=True)
    )
    targets = [
        record
        for record in episode.records
        if (record.logical_id or record.record_id).startswith("semantic:v8:orchestration:target:")
    ]
    assert len({record.logical_id for record in targets}) == 4
    assert any(record.part_count > 1 for record in targets), (
        "create target should exercise continuation"
    )
    events = {event.record_id: event for meridian in episode.meridians for event in meridian.events}
    for record in targets:
        event = events[record.record_id]
        assert event.track_id == validation["coordinatorTrack"]
        assert event.role == "assistant"
        expected = expected_continuation if record.part_index else expected_initial
        assert resolve_event_control(episode, event) == expected
        assert event.instruction == semantic_orchestration_instruction_v8(
            continuation=record.part_index > 0
        )
