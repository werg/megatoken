"""Committed semantic request bridge: one selected action, no synthetic tool JSON."""

import json
from threading import RLock
from types import SimpleNamespace

import pytest

from megatoken.multitrack_runtime import MultiTrackRuntime


def witness(effect="fx1", payload='["counter"]'):
    fields = {
        "effect": effect,
        "operation": "demo.readAsync",
        "invocation": "i1",
        "site": {"start": 12, "end": 31},
        "arguments": [["str", "counter"]],
        "payload": payload,
    }
    return ("request " + json.dumps(fields, separators=(",", ":"))).encode()


def group_and_session(line=None):
    tape = SimpleNamespace(
        payloads=(witness() if line is None else line,),
        modalities=("text",),
        valid=(True,),
        tracks=("worker:17",),
        event_ids=("semantic-event1",),
        meridian_id=3,
    )
    group = SimpleNamespace(tape=tape)
    session = SimpleNamespace(
        lock=RLock(), selected_groups={1: group}, tool_routes={}
    )
    return group, session


class Runner:
    def __init__(self):
        self.submitted = {}

    def submit(self, identity, call):
        if identity in self.submitted:
            raise AssertionError("Repeated native submission")
        self.submitted[identity] = call


class SelectedAdapter:
    def __init__(self):
        self.calls = []

    def execute_selected(self, *, action_id, payload):
        self.calls.append((action_id, payload))
        return b'{"status":"fulfilled","value":7}'


def test_validated_selected_request_has_single_stable_native_action_and_route():
    group, session = group_and_session()
    runner = Runner()
    adapter = SelectedAdapter()
    runtime = object.__new__(MultiTrackRuntime)
    def verifier(track, fields):
        return (
            track == "worker:17"
            and fields["invocation"] == "i1"
            and fields["site"] == {"start": 12, "end": 31}
        )
    submitted = runtime.dispatch_semantic_requests(
        session, group, runner, {"demo.readAsync": adapter},
        verify_source=verifier,
    )
    assert submitted == ("3:semantic-event1",)
    assert session.tool_routes == {
        "3:semantic-event1": ("worker:17", "semantic-event1")
    }
    assert adapter.calls == []
    assert runner.submitted[submitted[0]]() == b'{"status":"fulfilled","value":7}'
    assert adapter.calls == [(submitted[0], witness())]
    assert runtime.dispatch_semantic_requests(
        session, group, runner, {"demo.readAsync": adapter},
        verify_source=verifier,
    ) == ()
    assert len(runner.submitted) == 1
    assert len(adapter.calls) == 1


@pytest.mark.parametrize("line", [
    witness(payload='["forged"]'),
    b"request {\"effect\":",
    witness(effect="fx1").replace(b'"invocation":"i1"', b'"invocation":"broken-name"'),
])
def test_malformed_or_ungrounded_request_cannot_submit(line):
    group, session = group_and_session(line)
    runner = Runner()
    runtime = object.__new__(MultiTrackRuntime)
    with pytest.raises(ValueError):
        runtime.dispatch_semantic_requests(
            session, group, runner, {"demo.readAsync": SelectedAdapter()},
            verify_source=lambda track, fields: True,
        )
    assert runner.submitted == {}
    assert session.tool_routes == {}


def test_untrusted_source_verification_and_missing_adapter_refuse_dispatch():
    group, session = group_and_session()
    runtime = object.__new__(MultiTrackRuntime)
    runner = Runner()
    with pytest.raises(ValueError, match="source"):
        runtime.dispatch_semantic_requests(
            session, group, runner, {"demo.readAsync": SelectedAdapter()},
            verify_source=lambda track, fields: False,
        )
    with pytest.raises(ValueError, match="adapter"):
        runtime.dispatch_semantic_requests(
            session, group, runner, {}, verify_source=lambda track, fields: True,
        )
    assert runner.submitted == {}
