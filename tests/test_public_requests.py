"""Training and actual collected generation must receive identical public requests."""

import base64
import json
from dataclasses import asdict, replace

import pytest
from native_support import tiny_bundle

from megatoken.controls import build_control_payload, resolve_event_control
from megatoken.data.collection import TeacherRequest
from megatoken.data.schema import Episode, Lineage, PhysicalRecord, sequential_meridians
from megatoken.data.student import ModelTeacher
from megatoken.generation import ByteCodec
from megatoken.multitrack_service import _output
from megatoken.training.config import TrainConfig


def episode():
    return sequential_meridians(
        Episode(
            "public-request",
            (
                PhysicalRecord("question", "user", b"What is available?"),
                PhysicalRecord("answer", "assistant", b"A document.", policy_mask=True),
            ),
            Lineage("fixture", "v1", "public-request"),
            control_seed=b"Use available evidence.",
        )
    )


def test_default_training_request_is_byte_exact_with_normal_serving():
    sample = episode()
    event = sample.meridians[-1].events[0]
    trained = resolve_event_control(sample, event)
    _, served = _output({}, 128, [], control_seed=sample.control_seed)
    assert trained == served == build_control_payload(seed=sample.control_seed)
    assert b"meridian_id" not in trained and b"available_input_tracks" not in trained
    with pytest.raises(TypeError):
        resolve_event_control(sample, event, protocol_version=1)


def test_explicit_settings_instructions_and_evidence_match_actual_student_generation():
    sample = episode()
    event = replace(
        sample.meridians[-1].events[0],
        instruction="Answer using the question.",
        settings_json='{"language":"en"}',
        reference_event_ids=("question",),
    )
    trained = resolve_event_control(sample, event)
    student = ModelTeacher(tiny_bundle(ByteCodec.byte_fallback()), effort=1)
    try:
        response = student.respond(
            TeacherRequest(
                request_id="public-request",
                messages_json="[]",
                tools_json="[]",
                max_output_tokens=4,
                codec_fingerprint=student.codec.fingerprint,
                session_id="one-session",
                records=sample.records[:1],
                control_seed=sample.control_seed,
                outputs=(
                    {
                        "track": "assistant",
                        "max_tokens": 4,
                        "instruction": event.instruction,
                        "settings": {"language": "en"},
                        "reference_event_ids": ["question"],
                    },
                ),
            )
        )
        tape = json.loads(response.native_tape_json)
        assert base64.b64decode(tape["controls"][0]) == trained
    finally:
        student.runtime.close()


def test_public_tools_are_framed_once_when_grouped_collection_inherits_them():
    tools = json.dumps([{"name": "read", "parameters": {"type": "object"}}])
    _, control = _output(
        {"instruction": "Choose a procedure."},
        128,
        [],
        control_seed=b"policy",
        tools_json_default=tools,
    )
    assert control.count(b"\n<tools>\n") == 1
    assert control.startswith(b"policy")


def test_removed_request_modes_are_rejected():
    config = TrainConfig(purpose="diagnostic", max_steps=1)
    for name in ("request_supervision_version", "format_supervision_version"):
        with pytest.raises(ValueError, match="Unknown training settings"):
            TrainConfig.from_dict({**asdict(config), name: 1})
