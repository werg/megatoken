"""Closed model sessions retain bounded recovery and attempt authority."""

import copy
from types import SimpleNamespace

import pytest

from megatoken.data.semantic_json import stable_algorithm_hash
from megatoken.semantic_checkpoint_v8 import capture_random_state
from megatoken.semantic_model_v8 import NativeSelectionV8Error, NativeSemanticInferenceV8
from tests.test_prediction_prefix_v8 import examples
from tests.test_semantic_model_v8 import FakeRuntime
from tests.test_semantic_v8_serving_restore import (
    actual_adapter,
    actual_attempt,
    commit,
    rehash,
    select,
    target,
)


def with_session(example, session):
    document = copy.deepcopy(example)
    for row in document["public"]["timeline"]:
        row["fields"]["session"] = session
    timeline = document["public"]["timeline"]
    for context in document["contexts"]:
        context["prefixDigest"] = stable_algorithm_hash(timeline[: context["visibleThrough"]])
    return document


class MultiSessionFakeRuntime(FakeRuntime):
    """Give each simultaneously open fake native session a distinct memory ID."""

    def __init__(self, fragments):
        super().__init__(fragments)
        self.next_memory_id = 0

    def open(self, **kwargs):
        previous = dict(self.memory.sessions)
        session = super().open(**kwargs)
        self.memory.sessions.clear()
        self.memory.sessions.update(previous)
        self.next_memory_id += 1
        session.memory.session_id = f"memory{self.next_memory_id}"
        self.memory.sessions[session.memory.session_id] = session.memory
        return session


class SessionAwareFakeRuntime(MultiSessionFakeRuntime):
    """Supplied outputs belong to a model session, independent of restore order."""

    def __init__(self, session_streams):
        super().__init__([])
        self.session_streams = [list(stream) for stream in session_streams]
        self.outputs_by_memory_id = {}

    def open(self, **kwargs):
        session = super().open(**kwargs)
        self.outputs_by_memory_id[session.memory.session_id] = self.session_streams.pop(0)
        return session

    def generate(self, session, *, controls, event_ids, **kwargs):
        payload, terminal = self.outputs_by_memory_id[session.memory.session_id].pop(0)
        self.controls.extend(controls)
        session.memory.state += 1
        session.revision += 1
        session.occupied_events.update(event_ids)
        return SimpleNamespace(
            tape=SimpleNamespace(
                payloads=(payload,),
                terminal=(terminal,),
                valid=(True,),
                modalities=("text",),
                tracks=("assistant",),
                emitted_ids=(tuple(self.codec.encode(payload)),),
                event_ids=event_ids,
                meridian_id=session.revision,
            )
        )


def adapter_for_sessions(fragments, *, max_attempts=10000, max_sessions=16):
    runtime = MultiSessionFakeRuntime([(fragment, True) for fragment in fragments])
    adapter = NativeSemanticInferenceV8(
        runtime,
        model_identity="a" * 64,
        max_sessions=max_sessions,
        max_attempts=max_attempts,
    )
    adapter.initialize(
        seed=17,
        checkpointIdentity="a" * 64,
        tokenizerIdentity=runtime.codec.fingerprint,
    )
    return adapter, runtime


def adapter_for_session_streams(streams, *, max_sessions=2):
    runtime = SessionAwareFakeRuntime(streams)
    adapter = NativeSemanticInferenceV8(
        runtime,
        model_identity="a" * 64,
        max_sessions=max_sessions,
    )
    adapter.initialize(
        seed=17,
        checkpointIdentity="a" * 64,
        tokenizerIdentity=runtime.codec.fingerprint,
    )
    return adapter, runtime


def test_cancelled_sibling_keeps_its_attempt_and_opening_while_other_session_resumes():
    base = examples("semantic-regions-v8.json.gz.b64")[0]
    active = with_session(base, "active-session")
    cancelled = with_session(base, "cancelled-session")
    original, _ = adapter_for_sessions([target(active, 0), target(cancelled, 0)])

    accepted = select(original, active, 0)
    commit(original, active, 0, accepted)
    select(original, cancelled, 0)
    original.cancel(session="cancelled-session", reason="host cancelled this episode")
    packet = original.checkpoint()

    restored, runtime = adapter_for_sessions([target(active, 0), target(active, 1)])
    receipt = restored.restore(checkpoint=packet)
    assert receipt["restoredSessions"] == 1
    assert receipt["reexecutedSelections"] == 1
    assert receipt["consumedAttempts"] == 2
    assert set(restored.sessions) == {"active-session"}
    assert set(restored.closed_sessions) == {"cancelled-session"}
    assert [attempt["status"] for attempt in restored.attempts] == ["committed", "cancelled"]
    assert [row["session"] for row in restored.openings] == [
        "active-session",
        "cancelled-session",
    ]
    assert restored.metrics()["generatedTokens"] == original.metrics()["generatedTokens"]
    assert runtime.fragments == [(target(active, 1), True)]

    resumed = select(restored, active, 1)
    assert resumed["selection"]["selectionId"] == "v8:2"
    assert commit(restored, active, 1, resumed)["committed"]
    assert restored.metrics()["committed"] == 2


def test_multiple_closed_sessions_restore_sequentially_at_one_session_capacity():
    base = examples("semantic-regions-v8.json.gz.b64")[0]
    first = with_session(base, "first-closed")
    second = with_session(base, "second-closed")
    active = with_session(base, "active-session")
    original, _ = adapter_for_sessions(
        [target(first, 0), target(second, 0), target(active, 0)], max_sessions=1
    )

    first_result = select(original, first, 0)
    commit(original, first, 0, first_result)
    original.cancel(session="first-closed", reason="first episode complete")
    select(original, second, 0)
    original.cancel(session="second-closed", reason="second episode cancelled")
    active_result = select(original, active, 0)
    commit(original, active, 0, active_result)
    packet = original.checkpoint()

    restored, runtime = adapter_for_sessions(
        [target(first, 0), target(active, 0), target(active, 1)], max_sessions=1
    )
    receipt = restored.restore(checkpoint=packet)
    assert receipt["restoredSessions"] == 1
    assert receipt["reexecutedSelections"] == 2
    assert receipt["consumedAttempts"] == 3
    assert set(restored.sessions) == {"active-session"}
    assert set(restored.closed_sessions) == {"first-closed", "second-closed"}
    assert [attempt["status"] for attempt in restored.attempts] == [
        "committed",
        "cancelled",
        "committed",
    ]
    assert [row["session"] for row in restored.openings] == [
        "first-closed",
        "second-closed",
        "active-session",
    ]
    resumed = select(restored, active, 1)
    assert resumed["selection"]["selectionId"] == "v8:3"
    assert commit(restored, active, 1, resumed)["committed"]
    assert restored.metrics()["committed"] == 3
    assert runtime.fragments == []


def test_interleaved_session_outputs_replay_by_owner_then_the_active_session_continues():
    a_base = examples("semantic-callback-regions-v8.json.gz.b64")[0]
    b_base = examples("semantic-regions-v8.json.gz.b64")[0]
    session_a = with_session(a_base, "interleaved-a")
    session_b = with_session(b_base, "interleaved-b")
    assert len(session_a["contexts"]) >= 3
    streams = [
        [(target(session_a, index), True) for index in range(3)],
        [(target(session_b, 0), True)],
    ]
    original, _ = adapter_for_session_streams(streams)

    first_a = select(original, session_a, 0)
    commit(original, session_a, 0, first_a)
    first_b = select(original, session_b, 0)
    commit(original, session_b, 0, first_b)
    second_a = select(original, session_a, 1)
    commit(original, session_a, 1, second_a)
    original.cancel(session="interleaved-b", reason="close B after its accepted record")
    packet = original.checkpoint()
    assert [attempt["session"] for attempt in packet["attempts"]] == [
        "interleaved-a",
        "interleaved-b",
        "interleaved-a",
    ]

    restored, runtime = adapter_for_session_streams(streams)
    receipt = restored.restore(checkpoint=packet)
    assert receipt["reexecutedSelections"] == 3
    assert receipt["restoredSessions"] == 1
    assert set(restored.sessions) == {"interleaved-a"}
    assert set(restored.closed_sessions) == {"interleaved-b"}
    assert [attempt["session"] for attempt in restored.attempts] == [
        "interleaved-a",
        "interleaved-b",
        "interleaved-a",
    ]
    resumed = select(restored, session_a, 2)
    assert resumed["selection"]["selectionId"] == "v8:3"
    assert commit(restored, session_a, 2, resumed)["committed"]
    assert runtime.outputs_by_memory_id["memory1"] == []
    assert runtime.outputs_by_memory_id["memory2"] == []


def test_cancelled_session_is_a_checkpointed_tombstone_and_repeat_close_is_idempotent():
    example = examples("semantic-regions-v8.json.gz.b64")[0]
    original, _ = adapter_for_sessions([target(example, 0)])
    selected = select(original, example, 0)
    commit(original, example, 0, selected)
    session = selected["selection"]["session"]
    first_close = original.cancel(session=session, reason="episode complete")
    packet = original.checkpoint()

    restored, _ = adapter_for_sessions([target(example, 0)])
    receipt = restored.restore(checkpoint=packet)
    assert receipt["restoredSessions"] == 0
    assert receipt["reexecutedSelections"] == 1
    assert restored.sessions == {}
    assert session in restored.closed_sessions
    before = copy.deepcopy(restored.closed_sessions)
    second_close = restored.cancel(session=session, reason="repeat close")
    assert second_close["closed"] and first_close["closed"]
    assert restored.closed_sessions == before
    with pytest.raises(NativeSelectionV8Error, match="[Cc]losed"):
        select(restored, example, 1)


def test_rehashed_closed_session_receipt_cannot_change_its_verified_model_cut():
    example = examples("semantic-regions-v8.json.gz.b64")[0]
    original, _ = adapter_for_sessions([target(example, 0)])
    selected = select(original, example, 0)
    commit(original, example, 0, selected)
    session = selected["selection"]["session"]
    original.cancel(session=session, reason="episode complete")
    packet = original.checkpoint()
    saved = copy.deepcopy(packet)
    saved["closedSessions"][session]["continuationDigest"] = "0" * 64
    rehash(saved)

    restored, runtime = adapter_for_sessions([target(example, 0)])
    random_state_before = capture_random_state()
    with pytest.raises(NativeSelectionV8Error, match="continuation or observation cursor differs"):
        restored.restore(checkpoint=saved)
    assert not restored.sessions and not restored.closed_sessions and not runtime.memory.sessions
    assert capture_random_state() == random_state_before


def test_cancel_before_opening_persists_a_tombstone_without_fabricated_costs():
    example = examples("semantic-regions-v8.json.gz.b64")[0]
    original, _ = adapter_for_sessions([])
    original.cancel(session="never-opened", reason="cancelled before model selection")
    packet = original.checkpoint()

    restored, runtime = adapter_for_sessions([])
    receipt = restored.restore(checkpoint=packet)
    assert receipt["restoredSessions"] == 0
    assert restored.closed_sessions == {"never-opened": None}
    assert restored.openings == []
    assert restored.attempts == []
    with pytest.raises(NativeSelectionV8Error, match="[Cc]losed"):
        select(restored, with_session(example, "never-opened"), 0)
    assert not runtime.memory.sessions


def test_cancel_snapshot_failure_leaves_pending_attempt_and_session_untouched(monkeypatch):
    example = examples("semantic-regions-v8.json.gz.b64")[0]
    adapter, runtime = adapter_for_sessions([target(example, 0)])
    selected = select(adapter, example, 0)
    session = selected["selection"]["session"]
    work = adapter.sessions[session]
    attempt = work.pending["attempt"]
    memory = work.runtime.memory

    def fail_snapshot(*_args):
        raise NativeSelectionV8Error("closed-cut snapshot failed")

    with monkeypatch.context() as patcher:
        patcher.setattr(adapter, "_session_checkpoint", fail_snapshot)
        with pytest.raises(NativeSelectionV8Error, match="snapshot failed"):
            adapter.cancel(session=session, reason="attempted close")

    assert adapter.sessions[session] is work
    assert work.pending["attempt"] is attempt
    assert attempt["status"] == "awaiting-commit"
    assert session not in adapter.closed_sessions
    assert runtime.memory.sessions[memory.session_id] is memory
    adapter.cancel(session=session, reason="cleanup")


def test_actual_cpu_two_session_restore_preserves_attempts_without_claiming_semantic_success():
    """Real tiny-model sampling tests session recovery, not useful output quality."""
    import torch

    torch.set_num_threads(1)
    base = examples("semantic-regions-v8.json.gz.b64")[0]
    active = with_session(base, "actual-active")
    cancelled = with_session(base, "actual-cancelled")
    original, original_runtime = actual_adapter()
    original.max_sessions = 1
    restored = restored_runtime = None
    try:
        actual_attempt(original, cancelled)
        original.cancel(session="actual-cancelled", reason="cancel the sibling session")
        actual_attempt(original, active)
        packet = original.checkpoint()
        assert all(attempt["status"] != "committed" for attempt in original.attempts)
        assert len(original.attempts) == 2

        active_work = original.sessions["actual-active"]
        if active_work.pending is not None:
            pending_id = active_work.pending["attempt"]["selectionId"]
            original.reject(
                session="actual-active",
                selectionId=pending_id,
                reason="resume comparison after discarding pending output",
            )
        expected = actual_attempt(original, active)

        restored, restored_runtime = actual_adapter()
        restored.max_sessions = 1
        receipt = restored.restore(checkpoint=packet)
        assert receipt["restoredSessions"] == 1
        assert receipt["reexecutedSelections"] == 0
        assert receipt["consumedAttempts"] == 2
        assert set(restored.sessions) == {"actual-active"}
        assert set(restored.closed_sessions) == {"actual-cancelled"}
        assert restored.attempts[0]["status"] in {"rejected", "cancelled"}
        assert restored.attempts[1]["status"] in {"rejected", "abandoned-on-restore"}
        actual = actual_attempt(restored, active)
        for field in ("selectionId", "fragments", "payloadHex"):
            assert actual[field] == expected[field]
        assert actual["selectionId"] == "v8:2"
    finally:
        for adapter in (original, restored):
            if adapter is not None:
                for session in list(adapter.sessions):
                    adapter.cancel(session=session, reason="test cleanup")
        for runtime in (original_runtime, restored_runtime):
            if runtime is not None:
                runtime.close()
