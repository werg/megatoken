"""Message roles share a task's long window; delegated tool results keep ownership."""

from dataclasses import replace

import torch
from native_support import tiny_bundle

from megatoken.contracts import MAIN_TRACK
from megatoken.data.collection import TeacherRequest, _attach_selected_groups
from megatoken.data.schema import (
    Episode,
    Lineage,
    LogicalAction,
    PhysicalRecord,
    canonical_json,
    digest_bytes,
    sequential_meridians,
)
from megatoken.data.student import ModelTeacher
from megatoken.generation.codec import ByteCodec
from megatoken.loading import build_model
from megatoken.model import ModelConfig
from megatoken.model.multitrack import GroupActionTape, MultiTrackGenerator, ObservedEvent
from megatoken.multitrack_runtime import MultiTrackRuntime


def test_linear_preparation_preserves_roles_in_one_continuing_conversation():
    records = tuple(
        PhysicalRecord(str(i), role, b"input", turn_index=i)
        for i, role in enumerate(("system", "user", "assistant", "tool", "user"))
    )
    episode = sequential_meridians(
        Episode("conversation", records, Lineage("fixture", "v1", "family"))
    )
    assert len(episode.tracks) == 1
    assert {event.track_id for group in episode.meridians for event in group.events} == {MAIN_TRACK}
    assert [event.role for group in episode.meridians for event in group.events] == [
        r.role for r in records
    ]


def test_old_user_constraint_remains_local_despite_workers_and_role_changes():
    torch.set_num_threads(1)
    codec = ByteCodec.byte_fallback()
    model = build_model(
        ModelConfig.tiny(max_history_chunks=32, context_local_events=8, context_cross_events=1),
        codec,
    )
    runtime = MultiTrackRuntime(MultiTrackGenerator.of(model))
    try:
        session = runtime.open(effort=1)
        runtime.observe(
            session,
            (
                ObservedEvent(
                    MAIN_TRACK, "text", b"Keep the original units", 1, event_id="constraint"
                ),
            ),
            effort=1,
        )
        for i in range(6):
            runtime.observe(
                session,
                (ObservedEvent("worker", "text", b"Unrelated work", 4, event_id=f"work-{i}"),),
                effort=1,
            )
        runtime.observe(
            session,
            (ObservedEvent(MAIN_TRACK, "text", b"Continue the answer", 2, event_id="reply"),),
            effort=1,
        )
        assert "constraint" in runtime.context_status(session, track=MAIN_TRACK)["visible_events"]
        assert "constraint" not in runtime.context_status(session, track="worker")["visible_events"]
    finally:
        runtime.close()


def test_collected_worker_tool_results_replay_on_their_original_tracks():
    torch.set_num_threads(1)
    codec = ByteCodec.byte_fallback()
    teacher = ModelTeacher(tiny_bundle(codec), effort=1)
    tools = canonical_json(
        [
            {
                "name": "read",
                "parameters": {
                    "type": "object",
                    "const": {"path": "a"},
                    "properties": {"path": {"type": "string", "enum": ["a"]}},
                    "required": ["path"],
                    "additionalProperties": False,
                },
            }
        ]
    )
    request = TeacherRequest(
        request_id="first",
        messages_json="[]",
        tools_json="[]",
        max_output_tokens=128,
        codec_fingerprint=codec.fingerprint,
        session_id="worker-routing",
        records=(PhysicalRecord("prompt", "user", b"Read the file"),),
        outputs=tuple(
            {"track": track, "tools_json": tools, "max_tokens": 128}
            for track in ("reader-a", "reader-b")
        ),
    )
    try:
        response = teacher.respond(request)
        tape = GroupActionTape.from_json(response.native_tape_json)
        assert tape.valid == (True, True) and len(response.calls) == 2
        outputs = tuple(
            PhysicalRecord(
                f"action-{i}",
                "assistant",
                event.payload,
                turn_index=1,
                action_id=f"call-{i}",
                policy_mask=True,
                emitted_ids=event.emitted_ids,
                codec_fingerprint=codec.fingerprint,
            )
            for i, event in enumerate(response.events)
        )
        results = tuple(
            PhysicalRecord(
                f"result-{i}",
                "tool",
                f"Result {i}".encode(),
                turn_index=2 + i,
                result_of=f"call-{i}",
                execution_id=f"execution-{i}",
            )
            for i in range(2)
        )
        actual = (*request.records, *outputs, *results)
        teacher.respond(replace(request, request_id="next", records=actual, outputs=()))
        assert teacher._action_tracks == {}
        assert any(tracks == ("reader-a",) for _, tracks in teacher._session.events)
        assert any(tracks == ("reader-b",) for _, tracks in teacher._session.events)
        actions = tuple(
            LogicalAction(
                f"call-{i}",
                response.calls[i].call_id,
                "read",
                response.calls[i].arguments_json,
                (outputs[i].record_id,),
                digest_bytes(outputs[i].payload),
                execution_id=f"execution-{i}",
            )
            for i in range(2)
        )
        episode = Episode(
            "routing", actual, Lineage("fixture", "v1", "routing-family"), actions=actions
        )
        attached = _attach_selected_groups(
            episode, [(tuple(record.record_id for record in outputs), tape, (0, 1))]
        )
        routed = {
            event.record_id: event.track_id
            for group in attached.meridians
            for event in group.events
        }
        assert routed["prompt"] == MAIN_TRACK
        assert routed["result-0"] == "reader-a" and routed["result-1"] == "reader-b"
    finally:
        teacher.runtime.close()


def test_invalid_collected_peers_release_storage_without_delivery_claims():
    torch.set_num_threads(1)
    codec = ByteCodec.byte_fallback()
    teacher = ModelTeacher(tiny_bundle(codec), effort=1)
    teacher.runtime.max_retained_outputs = 1
    render = teacher.runtime.generator.render_selected

    def invalid(selection):
        group = render(selection)
        return replace(
            group,
            tape=replace(
                group.tape, valid=(False, False), payloads=(b"", b""), emitted_ids=((), ())
            ),
        )

    teacher.runtime.generator.render_selected = invalid
    records = []
    try:
        for i in range(3):
            records.append(PhysicalRecord(f"request-{i}", "user", b"Try again", turn_index=i))
            response = teacher.respond(
                TeacherRequest(
                    request_id=f"invalid-{i}",
                    messages_json="[]",
                    tools_json="[]",
                    max_output_tokens=4,
                    codec_fingerprint=codec.fingerprint,
                    session_id="invalid-collection",
                    records=tuple(records),
                    outputs=({"track": "a", "max_tokens": 4}, {"track": "b", "max_tokens": 4}),
                )
            )
            assert GroupActionTape.from_json(response.native_tape_json).valid == (False, False)
            assert not teacher._session.selected_groups and not teacher._session.rendering
            assert all(not event.payload for event in response.events)
    finally:
        teacher.runtime.close()
