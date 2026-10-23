"""Behavioral regressions at the training and serving boundaries."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from test_multitrack_runtime import _runtime

from megatoken.controls import build_control_payload, separate_event_references
from megatoken.generation.structure import ActionSpec
from megatoken.model.multitrack import ObservedEvent
from megatoken.model.sources import source_event
from megatoken.multitrack_service import MultiTrackService
from megatoken.retrieval.collection import Item
from megatoken.retrieval.session import Retrieved


def test_reference_selects_evidence_without_learning_its_address():
    torch.set_num_threads(1)
    _, runtime = _runtime()
    multi = runtime.generator
    state = multi.tracks.initial_state(1)
    state = multi.tracks.observe_group(
        state,
        (
            ObservedEvent("source", "text", b"The apple is red.", event_id="a"),
            ObservedEvent("source", "text", b"The sky is blue.", event_id="b"),
        ),
    )
    a = build_control_payload(seed=b"Translate the selected passage", reference_event_ids=("a",))
    b = build_control_payload(seed=b"Translate the selected passage", reference_event_ids=("b",))
    assert separate_event_references(a)[0] == separate_event_references(b)[0]
    first = multi._controls((a,), state, ("reply",))
    second = multi._controls((b,), state, ("reply",))
    assert not torch.allclose(first, second)
    renamed = replace(state, history_event_ids=("renamed-a", "renamed-b"))
    other = build_control_payload(
        seed=b"Translate the selected passage", reference_event_ids=("renamed-a",)
    )
    torch.testing.assert_close(first, multi._controls((other,), renamed, ("reply",)))
    with pytest.raises(ValueError, match="retained earlier event"):
        multi._controls(
            (build_control_payload(reference_event_ids=("future",)),), state, ("reply",)
        )
    first.square().sum().backward()
    assert any(p.grad is not None for p in multi.generator.core.reference_projection.parameters())
    runtime.close()


def test_retrieved_source_address_selects_its_actual_retained_evidence():
    _, runtime = _runtime()
    try:
        first = Retrieved("documents", Item(0, "a", "text", "a.txt", b"Red apple", None), 1, 1)
        second = Retrieved("documents", Item(1, "b", "text", "b.txt", b"Blue sky", None), 1, 2)
        state = runtime.generator.tracks.initial_state(1)
        for entry in (first, second):
            event = source_event(SimpleNamespace(), entry, state.device)
            assert event.event_id == entry.event_id
            state = runtime.generator.tracks.observe_group(state, (event,))
        controls = [
            runtime.generator._controls(
                (build_control_payload(reference_event_ids=(entry.event_id,)),), state, ("answer",)
            )
            for entry in (first, second)
        ]
        assert not torch.allclose(*controls)
        from megatoken.multitrack_service import _retrieved

        answer, count = _retrieved(
            SimpleNamespace(memory=SimpleNamespace(sources=SimpleNamespace(log=[first, second]))), 0
        )
        assert count == 2
        assert [item["event_id"] for item in answer] == [first.event_id, second.event_id]
    finally:
        runtime.close()


def test_asynchronous_completion_runs_the_same_turn_lookup_once():
    torch.set_num_threads(1)
    _, runtime = _runtime()
    session = runtime.open(effort=1)
    before = session.memory.agenda.lookups
    revision = runtime.start_generation(
        session,
        tracks=("reply",),
        specs=(ActionSpec(min_tokens=1, max_tokens=1, require_terminal=True),),
        effort=1,
    )
    try:
        session.rendering[revision].future.result(timeout=10)
        work = runtime.poll_generation(session, revision)
        assert work.result.tape.terminal == (True,)
        assert session.memory.agenda.lookups == before + 1
        runtime.poll_generation(session, revision)
        assert session.memory.agenda.lookups == before + 1
    finally:
        runtime.close()


def test_observation_addresses_survive_service_and_reject_duplicate_delivery():
    torch.set_num_threads(1)
    _, runtime = _runtime()
    service = MultiTrackService(runtime)
    session_id = service.handle({"op": "open", "effort": 1})["session"]
    request = {
        "op": "observe",
        "session": session_id,
        "effort": 1,
        "events": [{"track": "input", "text": "Available evidence", "event_id": "source-1"}],
    }
    try:
        assert service.handle(request)["ok"]
        session = service.sessions[session_id]
        assert "source-1" in session.completed_events
        revision = session.revision
        assert not service.handle(request)["ok"]
        assert session.revision == revision
    finally:
        runtime.close()
