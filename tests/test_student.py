"""The frozen student rolls out on the serving runtime, so its tapes are served and trained alike."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest
import torch
from native_support import as_group, tiny_bundle

from megatoken.contracts import ROLE_TO_ID
from megatoken.controls import build_control_payload
from megatoken.data.collection import TeacherRequest
from megatoken.data.schema import Episode, Lineage, PhysicalRecord, Supervision, canonical_json
from megatoken.data.student import ModelTeacher
from megatoken.generation import ActionSpec, ByteCodec
from megatoken.model.multitrack import GroupActionTape, MultiTrackGenerator, ObservedEvent
from megatoken.multitrack_runtime import MultiTrackRuntime
from megatoken.training.history import NativeHistory

CODEC = ByteCodec.byte_fallback()
PROMPT = PhysicalRecord("prompt", "user", b"choose an action")


@pytest.fixture
def bundle():
    torch.set_num_threads(1)
    torch.manual_seed(5)
    return tiny_bundle(CODEC)


def request_for(records, index=0, **options):
    return TeacherRequest(
        request_id=f"r{index}",
        messages_json="[]",
        tools_json="[]",
        max_output_tokens=8,
        codec_fingerprint=CODEC.fingerprint,
        session_id="session",
        records=tuple(records),
        control_seed=b"public policy",
        **options,
    )


def draw_seed(seed, request_id):
    digest = hashlib.sha256(f"{seed}:{request_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**63)


def test_a_student_rollout_is_exactly_a_served_session(bundle):
    student = ModelTeacher(bundle, effort=1, seed=2)
    response = student.respond(request_for([PROMPT]))
    # The same thing done by hand on the serving runtime: one user message, one draw.
    runtime = MultiTrackRuntime(
        MultiTrackGenerator(bundle.model), chunking=bundle.chunking, record_policy_context=True
    )
    session = runtime.open(effort=1)
    event = ObservedEvent(
        "assistant",
        "text",
        PROMPT.payload,
        ROLE_TO_ID["user"],
        terminal=True,
        event_id=PROMPT.record_id,
    )
    runtime.observe(session, (event,), effort=1)
    expected = runtime.generate(
        session,
        tracks=("assistant",),
        specs=(ActionSpec(max_tokens=8),),
        controls=(build_control_payload(seed=b"public policy", tools_json="[]"),),
        effort=1,
        generator=torch.Generator().manual_seed(draw_seed(2, "r0")),
    )
    assert response.native_tape_json == expected.tape.to_json()
    tape = GroupActionTape.from_json(response.native_tape_json)
    assert tape.tracks == ("assistant",) and tape.controls[0].startswith(b"public policy")
    # It behaved as served does: the lookups at the metadata and after the finished question.
    assert student._session.memory.agenda.lookups >= 2
    assert student._session.state.track_registry[-1] == "assistant"
    assert "assistant" in student._session.state.track_registry
    assert "user" not in student._session.state.track_registry


def test_each_record_is_read_once_on_the_main_track_and_the_draw_is_not_reread(bundle):
    student = ModelTeacher(bundle, effort=1, seed=2)
    first = student.respond(request_for([PROMPT]))
    answer = GroupActionTape.from_json(first.native_tape_json)
    assert answer.valid == (True,)
    records = [PROMPT]
    if answer.payloads[0]:
        records.append(
            PhysicalRecord(
                "answer",
                "assistant",
                answer.payloads[0],
                turn_index=1,
                emitted_ids=answer.emitted_ids[0],
                codec_fingerprint=CODEC.fingerprint,
                provenance="student",
            )
        )
    records.append(PhysicalRecord("result", "tool", b'{"ok":true}', turn_index=2))
    student.respond(request_for(records, 1))
    seen = [tracks for _, tracks in student._session.events]
    assert seen.count(("assistant",)) == 4  # user input, two draws, one tool result
    assert seen.count(("user",)) == seen.count(("tool",)) == 0


def _nonempty_rollout(bundle):
    # These integrity scenarios require a physical action. Apply the runtime's
    # public minimum-length constraint; keep the selected bytes and probability
    # genuine, instead of relying on a random initialization emitting content.
    student = ModelTeacher(bundle, effort=1, seed=2)
    original = student.runtime.generate

    def constrained(session, **kwargs):
        kwargs["specs"] = tuple(replace(spec, min_tokens=1) for spec in kwargs["specs"])
        return original(session, **kwargs)

    student.runtime.generate = constrained
    response = student.respond(request_for([PROMPT]))
    return student, response, GroupActionTape.from_json(response.native_tape_json)


def test_a_changed_or_missing_executed_action_is_refused(bundle):
    student, _, answer = _nonempty_rollout(bundle)
    assert answer.valid[0] and answer.payloads[0]
    executed = PhysicalRecord(
        "answer",
        "assistant",
        answer.payloads[0],
        turn_index=1,
        emitted_ids=answer.emitted_ids[0],
        codec_fingerprint=CODEC.fingerprint,
        provenance="student",
    )
    with pytest.raises(ValueError, match="missing from actual history"):
        student.respond(request_for([PROMPT], 1))
    with pytest.raises(ValueError, match="changed before execution"):
        student.respond(request_for([PROMPT, replace(executed, payload=b"other")], 1))


def test_a_student_tape_is_a_valid_native_annotation_of_its_episode(bundle):
    _, response, tape = _nonempty_rollout(bundle)
    assert tape.valid[0] and tape.payloads[0]
    answer = PhysicalRecord(
        "answer",
        "assistant",
        tape.payloads[0],
        turn_index=1,
        emitted_ids=tape.emitted_ids[0],
        codec_fingerprint=CODEC.fingerprint,
        provenance="student",
    )
    entry = {"record_cursor": 1, "tape": json.loads(response.native_tape_json)}
    episode = Episode(
        "student-episode",
        (PROMPT, answer),
        Lineage("fixture", "v1", "student"),
        supervision=Supervision(annotations_json=canonical_json({"native_action_tapes": [entry]})),
    )
    # The data schema accepts it, and training replays it against the model that drew it.
    history = NativeHistory.from_episode(episode, bundle.model)
    assert history.paired_tape(1) == tape and history.has_generated_record(1)
    assert history.events_for_record(1, include_record=False) == ()


def test_only_single_event_group_tapes_are_native_history(bundle):
    one = as_group(
        bundle.model.sample(
            bundle.model.core.initial_state(1),
            spec=ActionSpec(min_tokens=1, max_tokens=4),
            generator=torch.Generator().manual_seed(1),
        ).tape,
        CODEC,
    )
    two = replace(
        one,
        tracks=("assistant", "voice"),
        event_ids=("first", "second"),
        modality_masks=one.modality_masks * 2,
        modality_log_probs=one.modality_log_probs * 2,
        roles=one.roles * 2,
        controls=one.controls * 2,
        spec_fingerprints=one.spec_fingerprints * 2,
        facet_shape=(2, *one.facet_shape[1:]),
        raw_f32=one.raw_f32 * 2,
        behavior_log_prob=one.behavior_log_prob * 2,
        payloads=one.payloads * 2,
        emitted_ids=one.emitted_ids * 2,
        valid=one.valid * 2,
        terminal=one.terminal * 2,
        modalities=one.modalities * 2,
        event_log_probs=one.event_log_probs * 2,
        renderer_seeds=one.renderer_seeds * 2,
        pack_fingerprints=one.pack_fingerprints * 2,
    )
    prompt = PhysicalRecord("prompt", "user", b"x")
    episode = Episode(
        "joint",
        (prompt,),
        Lineage("fixture", "v1", "joint"),
        supervision=Supervision(
            annotations_json=canonical_json(
                {"native_action_tapes": [{"record_cursor": 1, "tape": json.loads(two.to_json())}]}
            )
        ),
    )
    with pytest.raises(ValueError, match="single-event"):
        NativeHistory.from_episode(episode, bundle.model)
