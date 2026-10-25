"""Actual shared native coordinator/child execution, with explicit control authority."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

from megatoken.data.semantic_contract import CONTRACT
from megatoken.generation.codec import ByteCodec
from megatoken.generation.policy import FacetedGenerator
from megatoken.model import FacetedCore, ModelConfig
from megatoken.model.multitrack import MultiTrackGenerator
from megatoken.multitrack_runtime import MultiTrackRuntime
from megatoken.semantic_model_adapter import SelectedSemanticInference, SemanticSelectionError


def proposal():
    doc = copy.deepcopy(CONTRACT["fixtures"]["envelope"]["valid"][0])
    program, invocation = doc["public"]["timeline"][:2]
    return {
        "schema": "selected-inference-prefix/2",
        "programId": program["programId"],
        "source": program["source"],
        "sourceDigest": program["digest"],
        "entry": invocation["entry"],
        "invocationId": invocation["invocationId"],
        "trackId": invocation["track"],
        "realmId": invocation["realmId"],
        "completionPolicy": invocation["completionPolicy"],
        "committed": [program, invocation],
    }, doc["public"]["timeline"][2:]


@pytest.fixture
def native_runtime():
    torch.set_num_threads(1)
    torch.manual_seed(17)
    model = FacetedGenerator(
        FacetedCore(ModelConfig.tiny(max_history_chunks=128, max_capacity=512)),
        ByteCodec.byte_fallback(),
    ).eval()
    runtime = MultiTrackRuntime(MultiTrackGenerator.of(model))
    try:
        with torch.inference_mode():
            yield runtime
    finally:
        runtime.close()


def test_controlled_coordinator_and_child_share_actual_bootstrap_join_and_completion(
    native_runtime,
):
    prefix, selected = proposal()
    session = prefix["committed"][0]["session"]
    adapter = SelectedSemanticInference(
        native_runtime, model_identity="a" * 64, allow_controlled_lifecycle=True
    )
    opened = adapter.open_orchestration(
        session=session, proposals=[prefix], origin="controlled", representation="compact"
    )
    track = opened["track"]

    def act(name, arguments):
        return adapter.step_orchestration(
            session=session,
            track=track,
            controlledRecord=json.dumps({"name": name, "arguments": arguments}),
        )

    with pytest.raises(SemanticSelectionError, match="actual coordinator creation"):
        adapter.select(prefix, controlledRecord=selected[0]["line"])
    created = act(
        "track_create",
        {
            "purpose": "Run observed child",
            "completion": "Return exact value",
            "budget": 64,
            "capabilities": ["text"],
            "execution": opened["children"][0]["execution"],
        },
    )
    outcome = created["decision"]["outcomes"][0]
    child = outcome["track"]["address"]
    assert created["decision"]["selection"]["modelSelected"] is False
    assert created["decision"]["selection"]["origin"] == "controlled"
    waiting = act(
        "track_join", {"address": track, "completion_dependency": outcome["completion_dependency"]}
    )
    assert waiting["status"] == "waiting"
    assert next(item for item in waiting["tracks"] if item["address"] == child)["bootstrap_ready"]
    for event in selected:
        result = adapter.select(prefix, controlledRecord=event["line"])
        assert result["record"] == event["line"] and not result["selection"]["modelSelected"]
        prefix["committed"].append(event)
    prefix["selectedState"] = {
        "schema": "semantic-continuation/4",
        "completion": {"kind": "return", "value": ["num", 7]},
    }
    adapter.close_invocation(session, prefix["invocationId"], prefix)
    state = adapter.status_orchestration(session=session, track=track)
    assert state["status"] == "active"
    completion = next(
        receipt for receipt in state["receipts"] if receipt.get("schema") == "semantic-completion/4"
    )
    assert completion["kind"] == "return" and completion["value"] == ["num", 7]
    assert completion["dependency"] == outcome["completion_dependency"]
    packet = adapter.checkpoint(session, [prefix])
    restored = SelectedSemanticInference(
        native_runtime, model_identity="a" * 64, allow_controlled_lifecycle=True
    )
    restored.restore(packet)
    assert (
        restored.status_orchestration(session=session, track=track)["receipts"] == state["receipts"]
    )
    with pytest.raises(SemanticSelectionError, match="unauthorized restored"):
        SelectedSemanticInference(native_runtime, model_identity="a" * 64).restore(packet)
    act(
        "track_progress",
        {
            "address": track,
            "progress": json.dumps({"kind": completion["kind"], "value": completion["value"]}),
        },
    )
    finished = act("track_complete", {"address": track})
    assert finished["status"] == "completed"
    assert all(item["status"] == "completed" for item in finished["tracks"])
    assert not adapter.sessions[session].runtime_session.selected_groups
    assert not adapter.sessions[session].runtime_session.teacher_lifecycle_actions


def test_actual_coordinator_generation_retains_invalid_bytes_without_fixture_fallback(
    native_runtime,
):
    prefix, _ = proposal()
    session = prefix["committed"][0]["session"]
    adapter = SelectedSemanticInference(native_runtime, model_identity="a" * 64, max_tokens=8)
    with pytest.raises(SemanticSelectionError, match="authorized authority"):
        adapter.open_orchestration(session=session, proposals=[prefix], origin="controlled")
    opened = adapter.open_orchestration(session=session, proposals=[prefix], origin="model")
    with pytest.raises(SemanticSelectionError, match="cannot receive controlled"):
        adapter.step_orchestration(session=session, track=opened["track"], controlledRecord="{}")
    with pytest.raises(SemanticSelectionError) as rejected:
        adapter.step_orchestration(session=session, track=opened["track"])
    assert rejected.value.evidence["origin"] == "model"
    assert rejected.value.evidence["payloadHex"]
    assert adapter.sessions[session].runtime_session.selected_groups
    assert not adapter.sessions[session].runtime_session.controlled_fixture_actions
    cancelled = adapter.cancel_orchestration(
        session=session, track=opened["track"], reason="evaluation boundary"
    )
    assert all(item["status"] == "cancelled" for item in cancelled["tracks"])


def test_persistent_subscription_owns_new_callbacks_after_registration_call_completes(
    native_runtime,
):
    document = json.loads(
        (Path(__file__).parent / "fixtures/semantic-v4/retained-events.json").read_text()
    )
    timeline = document["public"]["timeline"]
    program = timeline[0]
    registration = next(row for row in timeline if row["kind"] == "invocation")
    adapter = SelectedSemanticInference(native_runtime, model_identity="a" * 64)
    state = adapter._opened(program["session"])
    native = state.runtime_session
    native.tracks.max_active = 1
    parent = adapter._bootstrap(state, program, registration)
    # These are lifecycle integration inputs, explicitly independent of model
    # witness correctness (checked by the source/selected-driver fixture gate).
    native_runtime.complete_execution(
        native, parent.actual_track, kind="return", value=["str", "callback1"]
    )
    parent.completed = True
    owner = None
    children = []
    for invocation in (
        row for row in timeline if row["kind"] == "invocation" and row["activation"] == "callback"
    ):
        realm = [row for row in timeline if row["sequence"] <= invocation["sequence"]]
        allocated = adapter._subscription_owner(state, invocation, realm)
        assert owner is None or allocated == owner
        owner = allocated
        child = adapter._bootstrap(state, program, invocation, native_parent=owner)
        children.append(child.actual_track)
        assert native.tracks.tracks[child.actual_track].parent == owner
        assert native.tracks.tracks[parent.actual_track].status == "completed"
        assert native.tracks.selectable(child.actual_track, native.completed_events)
        receipt = native_runtime.complete_execution(
            native, child.actual_track, kind="return", value=["num", len(children)]
        )
        assert receipt["target"] == owner
        child.completed = True
    assert len(set(children)) == 2
    assert native.tracks.tracks[owner].status == "waiting"
    closed = next(row for row in timeline if row["kind"] == "handler-close")
    state.subscriptions["callback1"]["closed"] = closed
    adapter._retire_subscriptions(state)
    assert native.tracks.tracks[owner].status == "completed"
    assert all(row.status == "completed" for row in native.tracks.tracks.values())
    callback = next(
        row for row in timeline if row["kind"] == "invocation" and row["activation"] == "callback"
    )
    with pytest.raises(SemanticSelectionError, match="outlived"):
        adapter._subscription_owner(state, callback, timeline)


def test_returned_owned_closure_uses_explicit_external_ingress_after_parent_completion(
    native_runtime,
):
    prefix, _ = proposal()
    program, invocation = prefix["committed"]
    adapter = SelectedSemanticInference(native_runtime, model_identity="a" * 64)
    state = adapter._opened(program["session"])
    parent = adapter._bootstrap(state, program, invocation)
    # Lifecycle transport receives an already validated controller completion.
    # It does not execute source or claim these controlled inputs are teacher data.
    snapshot = {
        "heap": [
            ["fn", {"kind": "closure", "code": "function:20", "properties": []}],
            ["unexported", {"kind": "closure", "code": "function:20", "properties": []}],
        ]
    }
    receipt = native_runtime.complete_execution(
        state.runtime_session, parent.actual_track, kind="return", value=["ref", "fn"]
    )
    parent.completed = True
    adapter._completion_escapes(parent, snapshot, receipt)
    callback = {
        **invocation,
        "invocationId": "callbackinvoke1",
        "parentInvocationId": invocation["invocationId"],
        "activation": "callback",
        "closureId": "fn",
        "realmPolicy": "retained",
        "entry": "function:20",
        "triggerId": "callbackinvoke1",
        "ingressKey": "callbackinvoke1",
    }
    ingress = adapter._external_closure_ingress(state, callback, {"selectedState": snapshot})
    assert ingress["escapeEvidence"]["dependency"] == receipt["dependency"]
    child = adapter._bootstrap(state, program, callback, external_ingress=ingress)
    assert state.runtime_session.tracks.tracks[child.actual_track].parent is None
    assert state.runtime_session.tracks.tracks[parent.actual_track].status == "completed"
    assert ingress["registrationInvocationId"] == invocation["invocationId"]
    with pytest.raises(SemanticSelectionError, match="committed escape receipt"):
        adapter._external_closure_ingress(
            state, {**callback, "closureId": "unexported"}, {"selectedState": snapshot}
        )
    with pytest.raises(SemanticSelectionError, match="committed escape receipt"):
        adapter._external_closure_ingress(
            state,
            callback,
            {
                "selectedState": snapshot,
                "realmCommitted": [{"kind": "cancel", "invocationId": parent.invocation_id}],
            },
        )
    native_runtime.complete_execution(
        state.runtime_session, child.actual_track, kind="return", value=["num", 1]
    )
    assert all(
        track.status == "completed" for track in state.runtime_session.tracks.tracks.values()
    )
