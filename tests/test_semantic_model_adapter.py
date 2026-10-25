"""Selected Megatoken recurrence takes public code, selected bytes, observed results only."""

from __future__ import annotations

import copy
import hashlib
from types import SimpleNamespace

import pytest

from megatoken.lifecycle import TrackRegistry
from megatoken.semantic_model_adapter import SelectedSemanticInference, SemanticSelectionError


def fixture():
    source = "function run(){return 7;}"
    digest = hashlib.sha256(source.encode()).hexdigest()
    common = {"schema": "semantic-execution/4", "session": "session1"}
    published = {
        **common, "sequence": 1, "kind": "program", "mode": "observed",
        "track": "host", "programId": "p", "source": source,
        "digest": digest, "entry": "run", "capabilities": ["demo.readAsync"],
    }
    invocation = {
        **common, "sequence": 2, "kind": "invocation", "mode": "observed",
        "track": "worker1", "programId": "p", "invocationId": "i1",
        "realmId": "r1", "realmPolicy": "fresh", "triggerId": "outside1",
        "entry": "run", "sourceDigest": digest, "completionPolicy": "immediate-root",
        "inputGraph": {"roots": [], "objects": []},
    }
    prefix = {
        "schema": "selected-inference-prefix/1", "programId": "p",
        "source": source, "sourceDigest": digest, "entry": "run",
        "invocationId": "i1", "trackId": "worker1", "realmId": "r1",
        "completionPolicy": "immediate-root", "maxOutstanding": 64,
        "committed": [published, invocation],
    }
    return prefix


class FakeRuntime:
    """A real runtime-shaped controlled generator, never a teacher reference."""

    def __init__(self, selections=None):
        self.generator = SimpleNamespace(generator=SimpleNamespace(
            config=SimpleNamespace(max_capacity=2048),
            codec=SimpleNamespace(fingerprint="pinned-test-codec/1"),
        ))
        self.selections = list(selections or [
            b'request {"effect":"fx1","operation":"demo.readAsync","arguments":[["str","x"]],"invocation":"i1","site":{"start":0,"end":1},"payload":"[\\"x\\"]"}',
            b'return {"value":["num",7]}',
            b'end {"kind":"return"}',
        ])
        self.sessions = []
        self.native_calls = 0
        self.generated_specs = []

    def open(self, **kwargs):
        assert kwargs["memory_writes"] is False
        session = SimpleNamespace(
            tracks=TrackRegistry(max_active=8, max_waiting=16),
            completed_events=set(),
            observations=[],
        )
        self.sessions.append(session)
        return session

    def track_action(self, session, operation, **kwargs):
        assert operation == "create"
        return session.tracks.create(**kwargs)

    def observe_text(self, session, track, payload, **kwargs):
        session.observations.append((track, payload, kwargs["role"]))
        return 1

    def generate(self, session, *, tracks, specs, controls, effort):
        assert len(tracks) == len(specs) == len(controls) == 1
        assert session.tracks.selectable(tracks[0], session.completed_events)
        assert b"Never predict observations" in controls[0]
        self.generated_specs.append(specs[0])
        payload = self.selections.pop(0)
        return SimpleNamespace(tape=SimpleNamespace(
            valid=(True,), modalities=("text",), tracks=tracks,
            event_ids=(f"selected{len(self.generated_specs)}",),
            payloads=(payload,), meridian_id=len(self.generated_specs),
        ))


def add(prefix, kind, mode="generated", **fields):
    prefix["committed"].append({
        "schema": "semantic-execution/4", "session": "session1",
        "sequence": len(prefix["committed"]) + 1, "track": "worker1",
        "kind": kind, "mode": mode, "invocationId": "i1", **fields,
    })


def test_real_runtime_interface_commits_one_selection_before_any_future_observation():
    runtime = FakeRuntime()
    adapter = SelectedSemanticInference(runtime, max_tokens=512)
    prefix = fixture()
    first = adapter.select(prefix)
    assert first["schema"] == "selected-model-output/1"
    assert first["selection"]["modelSelected"] is True
    assert first["selection"]["ownerTrack"] == "worker1"
    assert first["selection"]["modelTrack"] == "track:0"
    assert first["selection"]["codecFingerprint"] == "pinned-test-codec/1"
    assert len(runtime.sessions[0].observations) == 2
    with pytest.raises(SemanticSelectionError, match="not committed"):
        adapter.select(prefix)
    add(prefix, "semantic", line=first["record"])
    add(prefix, "dispatch", mode="observed", effectId="fx1")
    add(prefix, "result", mode="observed", effectId="fx1",
        result=["num", 7], status="fulfilled",
        line='observe {"effect":"fx1","status":"fulfilled","result":["num",7]}')
    second = adapter.select(prefix)
    assert second["record"].startswith("return ")
    assert len(runtime.sessions[0].observations) == 4
    assert all(x[2] == "environment" for x in runtime.sessions[0].observations[2:])
    add(prefix, "semantic", line=second["record"])
    third = adapter.select(prefix)
    assert third["record"] == 'end {"kind":"return"}'
    add(prefix, "semantic", line=third["record"])
    adapter.close_invocation("session1", "i1", prefix)
    track = runtime.sessions[0].tracks.tracks["track:0"]
    assert track.status == "completed"
    assert runtime.native_calls == 0
    assert all(spec.require_terminal for spec in runtime.generated_specs)


def test_rewinding_or_replacing_earlier_observation_never_changes_model_history():
    adapter = SelectedSemanticInference(FakeRuntime())
    prefix = fixture()
    selected = adapter.select(prefix)
    add(prefix, "semantic", line=selected["record"])
    add(prefix, "result", mode="observed", effectId="fx1", result=["num", 7])
    adapter.select(prefix)
    previous = copy.deepcopy(prefix)
    previous["committed"][-1]["result"] = ["num", 999]
    with pytest.raises(SemanticSelectionError, match="rewritten"):
        adapter.select(previous)
    assert adapter.sessions["session1"].work["i1"].selections == 2


def test_invocation_bindings_prevent_source_reordering_or_impersonating_model_calls():
    adapter = SelectedSemanticInference(FakeRuntime())
    doc = fixture()
    for mutation in (
        lambda p: p["committed"].reverse(),
        lambda p: p["committed"][0].update(digest="bad"),
        lambda p: p["committed"][1].update(realmId="forged"),
        lambda p: p["committed"][1].update(mode="generated"),
        lambda p: p.update(entry="unpublished"),
    ):
        bad = copy.deepcopy(doc)
        mutation(bad)
        with pytest.raises(SemanticSelectionError):
            adapter.select(bad)
    assert not adapter.sessions


def test_more_than_eight_lifetime_invocations_reuse_one_model_session_without_aliasing():
    responses = [b'return {"value":["num",1]}', b'end {"kind":"return"}'] * 12
    runtime = FakeRuntime(selections=responses)
    adapter = SelectedSemanticInference(runtime)
    for number in range(12):
        prefix = fixture()
        prefix["invocationId"] = f"call{number}"
        prefix["realmId"] = f"r{number}"
        prefix["trackId"] = f"worker{number}"
        prefix["committed"][1].update(
            invocationId=prefix["invocationId"],
            realmId=prefix["realmId"],
            track=prefix["trackId"],
            triggerId=f"outside{number}",
        )
        one = adapter.select(prefix)
        add(prefix, "semantic", invocationId=prefix["invocationId"],
            track=prefix["trackId"], line=one["record"])
        two = adapter.select(prefix)
        add(prefix, "semantic", invocationId=prefix["invocationId"],
            track=prefix["trackId"], line=two["record"])
        adapter.close_invocation("session1", prefix["invocationId"], prefix)
    session = runtime.sessions[0]
    assert len(session.tracks.tracks) == 12
    assert all(track.status == "completed" for track in session.tracks.tracks.values())
    assert len(adapter.sessions) == 1
