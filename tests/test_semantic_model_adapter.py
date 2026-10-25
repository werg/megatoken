"""Selected Megatoken recurrence takes public code, selected bytes, observed results only."""

from __future__ import annotations

import copy
import hashlib
from threading import RLock
from types import SimpleNamespace

import pytest

from megatoken.lifecycle import TrackRegistry
from megatoken.semantic_model_adapter import SelectedSemanticInference, SemanticSelectionError


def fixture():
    source = "function run(){return 7;}"
    digest = hashlib.sha256(source.encode()).hexdigest()
    common = {"schema": "semantic-execution/4", "session": "session1", "ownerSequence": 1}
    published = {
        **common,
        "sequence": 1,
        "kind": "program",
        "mode": "observed",
        "track": "host",
        "programId": "p",
        "source": source,
        "digest": digest,
        "entry": "run",
        "capabilities": ["demo.readAsync"],
        "sites": [],
        "functions": [],
        "dependencies": [],
        "sourceMap": None,
        "sourceMetadata": None,
        "languageProfile": "owned-javascript/1",
        "module": False,
    }
    invocation = {
        **common,
        "sequence": 2,
        "kind": "invocation",
        "mode": "observed",
        "track": "worker1",
        "programId": "p",
        "invocationId": "i1",
        "realmId": "r1",
        "realmPolicy": "fresh",
        "triggerId": "outside1",
        "entry": "run",
        "sourceDigest": digest,
        "completionPolicy": "call-return",
        "inputGraph": {"roots": [], "objects": []},
        "receiver": ["undefined"],
        "closureId": None,
        "activation": "call",
        "parentInvocationId": None,
        "taskRevision": 1,
        "deadline": None,
        "bindings": {},
        "ingressKey": "outside1",
    }
    prefix = {
        "schema": "selected-inference-prefix/2",
        "programId": "p",
        "source": source,
        "sourceDigest": digest,
        "entry": "run",
        "invocationId": "i1",
        "trackId": "worker1",
        "realmId": "r1",
        "completionPolicy": "call-return",
        "maxOutstanding": 64,
        "committed": [published, invocation],
    }
    return prefix


class FakeRuntime:
    """A real runtime-shaped controlled generator, never a teacher reference."""

    def __init__(self, selections=None):
        self.generator = SimpleNamespace(
            generator=SimpleNamespace(
                config=SimpleNamespace(max_capacity=2048),
                codec=SimpleNamespace(fingerprint="pinned-test-codec/1"),
            )
        )
        self.selections = list(
            selections
            or [
                b'request {"effect":"fx1","operation":"demo.readAsync","arguments":[["str","x"]],"invocation":"i1","promise":"p1","site":{"start":0,"end":1},"payload":"[\\"x\\"]"}',
                b'return {"value":["num",7]}',
                b'end {"kind":"return"}',
            ]
        )
        self.sessions = []
        self.native_calls = 0
        self.generated_specs = []

    def open(self, **kwargs):
        assert kwargs["memory_writes"] is False
        session = SimpleNamespace(
            tracks=TrackRegistry(max_active=8, max_waiting=16),
            completed_events=set(),
            observations=[],
            selected_groups={},
            tool_routes={},
            lock=RLock(),
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
        terminal = True
        if isinstance(payload, tuple):
            payload, terminal = payload
        group = SimpleNamespace(
            tape=SimpleNamespace(
                valid=(True,),
                modalities=("text",),
                tracks=tracks,
                event_ids=(f"selected{len(self.generated_specs)}",),
                payloads=(payload,),
                meridian_id=len(self.generated_specs),
                terminal=(terminal,),
            )
        )
        session.selected_groups[len(self.generated_specs)] = group
        return group

    def dispatch_semantic_requests(self, *args, **kwargs):
        from megatoken.multitrack_runtime import MultiTrackRuntime

        return MultiTrackRuntime.dispatch_semantic_requests(self, *args, **kwargs)

    def release_generation(self, session, revision):
        session.selected_groups.pop(revision, None)


def add(prefix, kind, mode="generated", **fields):
    import json

    required = {}
    if kind == "semantic":
        required["traceFormat"] = "semantic-trace/4"
    elif kind == "dispatch":
        required.update(key="session1/i1/" + fields["effectId"], attempt=1)
    elif kind == "result":
        required.update(status="fulfilled", errorOrigin=None)
        required["line"] = "observe " + json.dumps(
            {
                "effect": fields["effectId"],
                "status": fields.get("status", "fulfilled"),
                "result": fields["result"],
            },
            separators=(",", ":"),
        )
    prefix["committed"].append(
        {
            "schema": "semantic-execution/4",
            "session": "session1",
            "sequence": len(prefix["committed"]) + 1,
            "ownerSequence": len(prefix["committed"]),
            "track": "worker1",
            "kind": kind,
            "mode": mode,
            "invocationId": "i1",
            **required,
            **fields,
        }
    )


def test_real_runtime_interface_commits_one_selection_before_any_future_observation():
    runtime = FakeRuntime()
    adapter = SelectedSemanticInference(runtime, max_tokens=512, verify_context=False)
    prefix = fixture()
    first = adapter.select(prefix)
    assert first["schema"] == "selected-model-output/2"
    assert first["selection"]["modelSelected"] is True
    assert first["selection"]["ownerTrack"] == "worker1"
    assert first["selection"]["modelTrack"] == "track:0"
    assert first["selection"]["codecFingerprint"] == "pinned-test-codec/1"
    assert len(runtime.sessions[0].observations) == 2
    with pytest.raises(SemanticSelectionError, match="not committed"):
        adapter.select(prefix)
    add(prefix, "semantic", line=first["record"])
    add(prefix, "dispatch", mode="observed", effectId="fx1")
    add(
        prefix,
        "result",
        mode="observed",
        effectId="fx1",
        result=["num", 7],
        status="fulfilled",
        line='observe {"effect":"fx1","status":"fulfilled","result":["num",7]}',
    )
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
    assert all(spec.require_terminal is None for spec in runtime.generated_specs)


def test_inference_initialization_attests_actual_checkpoint_and_seeds_rngs():
    import random

    import numpy as np
    import torch

    runtime = FakeRuntime()
    adapter = SelectedSemanticInference(runtime, model_identity="a" * 64, verify_context=False)
    with pytest.raises(SemanticSelectionError, match="loaded model"):
        adapter.initialize(seed=17, checkpointIdentity="b" * 64)
    with pytest.raises(SemanticSelectionError, match="declared seed"):
        adapter.initialize(seed=18, checkpointIdentity="a" * 64)
    receipt = adapter.initialize(seed=17, checkpointIdentity="a" * 64)
    values = random.random(), np.random.rand(), torch.rand(1).item()
    assert adapter.initialize(seed=17, checkpointIdentity="a" * 64) == receipt
    assert values == (random.random(), np.random.rand(), torch.rand(1).item())
    assert receipt["schema"] == "semantic-inference-identity/4"
    assert receipt["tokenizerIdentity"] == "pinned-test-codec/1"
    adapter.select(fixture())
    with pytest.raises(SemanticSelectionError, match="active session"):
        adapter.initialize(seed=29, checkpointIdentity="a" * 64)


def test_unsupported_semantic_representations_fail_before_guest_execution():
    runtime = FakeRuntime()
    adapter = SelectedSemanticInference(runtime, model_identity="a" * 64, verify_context=False)
    for obsolete in ("explicit", "outcome-only", "unknown"):
        with pytest.raises(SemanticSelectionError, match="canonical semantic witness"):
            adapter.select(fixture(), representation=obsolete)
    assert runtime.native_calls == 0
    assert len(runtime.generated_specs) == 0


def test_rewinding_or_replacing_earlier_observation_never_changes_model_history():
    adapter = SelectedSemanticInference(FakeRuntime(), verify_context=False)
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


def test_invocation_representation_cannot_change_before_or_after_checkpoint():
    adapter = SelectedSemanticInference(
        FakeRuntime(), verify_context=False, model_identity="a" * 64
    )
    prefix = fixture()
    selected = adapter.select(prefix, representation="compact")
    add(prefix, "semantic", line=selected["record"])
    with pytest.raises(SemanticSelectionError, match="canonical semantic witness"):
        adapter.select(prefix, representation="explicit")
    checkpoint = adapter.checkpoint("session1", [prefix])
    assert checkpoint["work"][0]["representation"] == "compact"
    restored = SelectedSemanticInference(
        FakeRuntime(), verify_context=False, model_identity="a" * 64
    )
    restored.restore(checkpoint)
    with pytest.raises(SemanticSelectionError, match="canonical semantic witness"):
        restored.select(prefix, representation="explicit")


def test_invocation_bindings_prevent_source_reordering_or_impersonating_model_calls():
    adapter = SelectedSemanticInference(FakeRuntime(), verify_context=False)
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
    adapter = SelectedSemanticInference(runtime, verify_context=False)
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
        add(
            prefix,
            "semantic",
            invocationId=prefix["invocationId"],
            track=prefix["trackId"],
            line=one["record"],
        )
        two = adapter.select(prefix)
        add(
            prefix,
            "semantic",
            invocationId=prefix["invocationId"],
            track=prefix["trackId"],
            line=two["record"],
        )
        adapter.close_invocation("session1", prefix["invocationId"], prefix)
    session = runtime.sessions[0]
    assert len(session.tracks.tracks) == 12
    assert all(track.status == "completed" for track in session.tracks.tracks.values())
    assert len(adapter.sessions) == 1


def test_model_restart_is_explicit_observed_bootstrap_and_never_reexecutes_tools():
    runtime = FakeRuntime(selections=[b'return {"value":["num",7]}'])
    adapter = SelectedSemanticInference(
        runtime, verify_context=False, model_identity="fixed-fixture-model"
    )
    prefix = fixture()
    selected = adapter.select(prefix)
    with pytest.raises(SemanticSelectionError, match="unacknowledged"):
        adapter.checkpoint("session1", [prefix])
    add(prefix, "semantic", line=selected["record"])
    checkpoint = adapter.checkpoint("session1", [prefix])
    restored_runtime = FakeRuntime(selections=[b'end {"kind":"return"}'])
    restored = SelectedSemanticInference(
        restored_runtime, verify_context=False, model_identity="fixed-fixture-model"
    )
    receipt = restored.restore(checkpoint)
    assert receipt["equivalence"] == "observed-checkpoint-bootstrap"
    assert receipt["nativeEffects"] == 0
    assert b"checkpoint-bootstrap" in restored_runtime.sessions[0].observations[-1][1]
    final = restored.select(prefix)
    add(prefix, "semantic", line=final["record"])
    restored.close_invocation("session1", "i1", prefix)
    assert restored_runtime.native_calls == 0
    tampered = copy.deepcopy(checkpoint)
    tampered["work"][0]["prefix"]["source"] += "changed"
    with pytest.raises(SemanticSelectionError, match="bytes changed"):
        SelectedSemanticInference(
            FakeRuntime(), verify_context=False, model_identity="fixed-fixture-model"
        ).restore(tampered)


def test_actual_initialized_model_inference_and_source_reread_change_no_weights():
    import torch

    from megatoken.generation.codec import ByteCodec
    from megatoken.generation.policy import FacetedGenerator
    from megatoken.model import FacetedCore, ModelConfig
    from megatoken.model.multitrack import MultiTrackGenerator
    from megatoken.multitrack_runtime import MultiTrackRuntime

    torch.set_num_threads(1)
    torch.manual_seed(27)
    model = FacetedGenerator(
        FacetedCore(ModelConfig.tiny(max_history_chunks=64, max_capacity=512)),
        ByteCodec.byte_fallback(),
    )
    runtime = MultiTrackRuntime(MultiTrackGenerator.of(model))
    model.eval()
    before = {name: value.clone() for name, value in model.state_dict().items()}
    adapter = SelectedSemanticInference(runtime, max_tokens=8, work_budget=4)
    prefix = fixture()
    try:
        with torch.inference_mode():
            try:
                selected = adapter.select(prefix)
                assert selected["selection"]["modelSelected"] is True
            except SemanticSelectionError as error:
                assert "complete valid text action" in str(error)
                assert error.evidence["payloadHex"]
            state = adapter.sessions["session1"]
            work = state.work["i1"]
            assert work.source_events
            runtime.configure_context(
                state.runtime_session, track=work.actual_track, local_events=0, cross_events=0
            )
            assert not set(work.source_events) <= set(
                runtime.context_status(state.runtime_session, track=work.actual_track)[
                    "visible_events"
                ]
            )
            receipt = adapter._ensure_source(state, work, prefix["committed"][0])
            assert receipt[0]["mode"] == "checked-reread"
            assert set(work.source_events) <= set(
                runtime.context_status(state.runtime_session, track=work.actual_track)[
                    "visible_events"
                ]
            )
            adapter.cancel_invocation("session1", "i1")
            assert not state.runtime_session.selected_groups
            assert not state.runtime_session.reserved_output_bytes
        assert all(torch.equal(value, model.state_dict()[name]) for name, value in before.items())
    finally:
        runtime.close()


def test_fragmented_model_request_reaches_real_native_commit_bridge_once():
    from megatoken.data.witness_v4 import parse_record

    original = FakeRuntime().selections[0]
    boundary = original.index(b'"payload"')
    runtime = FakeRuntime(selections=[(original[:boundary], False), (original[boundary:], True)])
    adapter = SelectedSemanticInference(runtime, verify_context=False)
    prefix = fixture()
    selected = adapter.select(prefix)
    assert selected["record"].encode() == original
    assert len(selected["selection"]["fragments"]) == 2
    assert not runtime.sessions[0].tool_routes
    _, fields = parse_record(selected["record"])
    validation = {
        "schema": "selected-source-validation/4",
        "sourceDigest": prefix["sourceDigest"],
        "invocationId": prefix["invocationId"],
        "effectId": fields["effect"],
        "payloadDigest": hashlib.sha256(original).hexdigest(),
        "heap": {},
    }
    descriptor = adapter.commit_request(
        "session1", "i1", selected["selection"], selected["record"], validation
    )
    assert descriptor["schema"] == "semantic-native-action/4"
    assert descriptor["actionKey"] == "2:selected2"
    assert descriptor["payload"] == '["x"]'
    assert len(runtime.sessions[0].tool_routes) == 1
    assert (
        adapter.commit_request(
            "session1", "i1", selected["selection"], selected["record"], validation
        )
        == descriptor
    )
    assert len(runtime.generated_specs) == 2
    altered = selected["record"].replace('"x"', '"y"')
    with pytest.raises(SemanticSelectionError, match="bytes"):
        adapter.commit_request("session1", "i1", selected["selection"], altered, validation)
    assert runtime.native_calls == 0  # JS broker owns execution; this commits its descriptor.


def test_native_request_authority_survives_later_prefix_acknowledgement():
    """Async dispatch may reach the bridge after the next selection consumed feedback."""
    from megatoken.data.witness_v4 import parse_record

    runtime = FakeRuntime()
    adapter = SelectedSemanticInference(runtime, verify_context=False)
    prefix = fixture()
    request = adapter.select(prefix)
    add(prefix, "semantic", line=request["record"])
    adapter.select(prefix)
    work = adapter.sessions["session1"].work["i1"]
    assert request["record"].encode() not in work.selected
    _, fields = parse_record(request["record"])
    receipt = {
        "schema": "selected-source-validation/4",
        "sourceDigest": prefix["sourceDigest"],
        "invocationId": "i1",
        "effectId": fields["effect"],
        "payloadDigest": hashlib.sha256(request["record"].encode()).hexdigest(),
        "heap": {},
    }
    committed = adapter.commit_request(
        "session1", "i1", request["selection"], request["record"], receipt
    )
    assert committed["authority"] == "actual-model-selected"
    assert committed["recordDigest"] == receipt["payloadDigest"]
    assert len(runtime.sessions[0].tool_routes) == 1
