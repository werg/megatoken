"""Regression checks for dynamic collection task and student event handling."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest
import torch
from native_support import tiny_bundle

from megatoken.data.cli import iter_task_seeds
from megatoken.data.collection import TeacherRequest
from megatoken.data.schema import PhysicalRecord
from megatoken.data.student import ModelTeacher, _group_event_ids
from megatoken.generation import ByteCodec
from megatoken.model.multitrack import GroupActionTape


def test_task_seed_reads_both_public_output_forms(tmp_path):
    path = tmp_path / "tasks.jsonl"
    path.write_text(
        json.dumps(
            {
                "seed_id": "s",
                "family_id": "f",
                "category": "code",
                "prompt": "p",
                "tools": [],
                "outputs": [{"track": "assistant", "event_id": "reply"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    seed = next(iter_task_seeds(path))
    assert json.loads(seed.outputs_json) == [{"track": "assistant", "event_id": "reply"}]


def test_task_seed_rejects_malformed_outputs_with_source_line(tmp_path):
    path = tmp_path / "tasks.jsonl"
    path.write_text(
        json.dumps(
            {
                "seed_id": "s",
                "family_id": "f",
                "category": "code",
                "prompt": "p",
                "tools": [],
                "outputs_json": "{bad",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="line 1.*outputs_json"):
        next(iter_task_seeds(path))


def test_group_event_addresses_are_unique_across_repeated_same_track_requests():
    def request(request_id):
        return TeacherRequest(
            request_id=request_id,
            messages_json="[]",
            tools_json="[]",
            max_output_tokens=8,
            codec_fingerprint="codec",
            session_id="session",
        )

    outputs = ({"track": "assistant", "event_id": "reply"},)
    first, second = (
        _group_event_ids(request("request-a"), outputs),
        _group_event_ids(request("request-b"), outputs),
    )
    assert first[0] != second[0]
    assert first[0].endswith(":reply") and second[0].endswith(":reply")


def test_student_can_reference_an_actual_observed_record():
    codec = ByteCodec.byte_fallback()
    student = ModelTeacher(tiny_bundle(codec), effort=1)
    try:
        response = student.respond(
            TeacherRequest(
                request_id="evidence-request",
                messages_json="[]",
                tools_json="[]",
                max_output_tokens=4,
                codec_fingerprint=codec.fingerprint,
                session_id="evidence-session",
                records=(PhysicalRecord("source-record", "user", b"The apple is red."),),
                outputs=(
                    {
                        "track": "assistant",
                        "max_tokens": 4,
                        "reference_event_ids": ["source-record"],
                    },
                ),
            )
        )
        assert response.native_tape_json
        assert "source-record" in student._session.memory.state.history_event_ids
    finally:
        student.runtime.close()


def test_synchronous_student_releases_archived_snapshots_for_long_sessions():
    codec = ByteCodec.byte_fallback()
    torch.manual_seed(9)
    student = ModelTeacher(tiny_bundle(codec), effort=1, seed=2)
    student.runtime.max_retained_outputs = 1
    original = student.runtime.generate

    def nonempty(session, **kwargs):
        kwargs["specs"] = tuple(replace(spec, min_tokens=1) for spec in kwargs["specs"])
        return original(session, **kwargs)

    student.runtime.generate = nonempty
    records = [PhysicalRecord("prompt", "user", b"choose")]
    for index in range(4):
        request = TeacherRequest(
            request_id=f"r{index}",
            messages_json="[]",
            tools_json="[]",
            max_output_tokens=8,
            codec_fingerprint=codec.fingerprint,
            session_id="long-session",
            records=tuple(records),
        )
        response = student.respond(request)
        tape = GroupActionTape.from_json(response.native_tape_json)
        assert not student._session.selected_groups
        if tape.valid[0] and tape.payloads[0]:
            records.append(
                PhysicalRecord(
                    f"answer-{index}",
                    "assistant",
                    tape.payloads[0],
                    turn_index=len(records),
                    emitted_ids=tape.emitted_ids[0],
                    codec_fingerprint=codec.fingerprint,
                )
            )
        records.append(PhysicalRecord(f"result-{index}", "tool", b"ok", turn_index=len(records)))


def test_grouped_student_reuses_track_labels_without_reusing_event_ids():
    codec = ByteCodec.byte_fallback()
    torch.manual_seed(19)
    student = ModelTeacher(tiny_bundle(codec), effort=1, seed=3)
    student.runtime.max_retained_outputs = 2
    original = student.runtime.start_generation

    def nonempty(session, **kwargs):
        kwargs["specs"] = tuple(replace(spec, min_tokens=1) for spec in kwargs["specs"])
        return original(session, **kwargs)

    student.runtime.start_generation = nonempty
    outputs = tuple(
        {"track": "assistant", "event_id": label, "modality": "text", "max_tokens": 4}
        for label in ("reply-a", "reply-b")
    )
    records = [PhysicalRecord("prompt", "user", b"choose")]
    event_ids = []
    for index in range(3):
        response = student.respond(
            TeacherRequest(
                request_id=f"group-{index}",
                messages_json="[]",
                tools_json="[]",
                max_output_tokens=8,
                codec_fingerprint=codec.fingerprint,
                session_id="grouped-session",
                records=tuple(records),
                outputs=outputs,
            )
        )
        tape = GroupActionTape.from_json(response.native_tape_json)
        assert tape.tracks == ("assistant", "assistant")
        assert tuple(value.rsplit(":", 1)[-1] for value in tape.event_ids) == ("reply-a", "reply-b")
        assert len(set(tape.event_ids)) == 2
        event_ids.append(tape.event_ids)
        assert len(student._session.rendering) <= 1
        assert len(student._session.selected_groups) <= 1
        for member, event in enumerate(response.events):
            if not tape.valid[member] or not event.payload:
                continue
            records.append(
                PhysicalRecord(
                    f"group-{index}-member-{member}",
                    event.track,
                    event.payload,
                    turn_index=len(records),
                    emitted_ids=event.emitted_ids,
                    codec_fingerprint=codec.fingerprint if event.emitted_ids is not None else None,
                )
            )
        records.append(PhysicalRecord(f"result-{index}", "tool", b"done", turn_index=len(records)))
    assert len({ids[0] for ids in event_ids}) == 3
