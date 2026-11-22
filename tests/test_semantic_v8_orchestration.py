"""Shared native coordinator/child recovery with explicit engineering decoding.

Torch and the actual recurrent runtime are required. Controlled lifecycle input
and supplied lexical child bytes are never labeled learned success or source
admission. One additional control uses the unchanged actual model decoder and
retains its invalid output without any fixture fallback.
"""

from __future__ import annotations

import base64
import copy
import gzip
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from megatoken.contracts import ROLE_TO_ID
from megatoken.controls import (
    canonical_semantic_observation,
    semantic_orchestration_ack_observation_v8,
    semantic_orchestration_input_v8,
)
from megatoken.data.semantic_json import stable_algorithm_hash
from megatoken.data.semantic_orchestration_training_v8 import (
    orchestration_transcript_rows_v8,
    validate_orchestration_transcript_v8,
)
from megatoken.generation.codec import ByteCodec
from megatoken.generation.policy import FacetedGenerator
from megatoken.model import FacetedCore, ModelConfig
from megatoken.model.multitrack import MultiTrackGenerator
from megatoken.multitrack_runtime import MultiTrackRuntime
from megatoken.semantic_checkpoint_v8 import capture_random_state, continuation_fingerprint
from megatoken.semantic_model_v8 import NativeSelectionV8Error, NativeSemanticInferenceV8
from megatoken.semantic_orchestration_v8 import _context as orchestration_context
from scripts.serve_semantic_v8 import dispatch
from tests.test_semantic_v8_orchestration_authority import example, ingress


@contextmanager
def real_runtime():
    torch.set_num_threads(1)
    torch.manual_seed(17)
    codec = ByteCodec.byte_fallback()
    model = FacetedGenerator(
        FacetedCore(ModelConfig.tiny(max_history_chunks=128, max_capacity=512)), codec
    ).eval()
    runtimes, adapters, decoder = [], [], {"records": []}
    original = MultiTrackGenerator._realize

    def realize(multi, raw, specs, control, **kwargs):
        if decoder["records"] is None:
            return original(multi, raw, specs, control, **kwargs)
        assert decoder["records"], "Unexpected native generation outside the supplied control"
        payload = decoder["records"].pop(0)
        return (payload,), (tuple(codec.encode(payload)),), (True,), (True,), (None,)

    def open_adapter(*, controlled=True, max_tokens=512):
        runtime = MultiTrackRuntime(MultiTrackGenerator.of(model))
        runtimes.append(runtime)
        adapter = NativeSemanticInferenceV8(
            runtime,
            model_identity="a" * 64,
            allow_controlled_lifecycle=controlled,
            max_tokens=max_tokens,
            max_fragments=16,
        )
        adapter.initialize(
            seed=17, checkpointIdentity="a" * 64, tokenizerIdentity=codec.fingerprint
        )
        adapters.append(adapter)
        return adapter

    try:
        with torch.inference_mode(), patch.object(MultiTrackGenerator, "_realize", realize):
            yield open_adapter, decoder
    finally:
        for adapter in adapters:
            for session in list(adapter.sessions):
                adapter.cancel(session=session, reason="Engineering control finished")
        for runtime in runtimes:
            runtime.close()


def request(adapter, name, **fields):
    return dispatch(adapter, {"id": "test-request", "op": name, **fields})


def open_coordinator(adapter):
    public = ingress()
    session = public["timeline"][0]["fields"]["session"]
    return request(
        adapter,
        "open-orchestration",
        session=session,
        public=public,
        origin="controlled",
        childBudget=64,
    )


def chosen(adapter, opened, name, arguments):
    return request(
        adapter,
        "select-orchestration",
        session=opened["session"],
        track=opened["track"],
        controlledRecord=json.dumps({"name": name, "arguments": arguments}),
    )


def acknowledge(adapter, opened, selected):
    return request(
        adapter,
        "commit-orchestration",
        session=opened["session"],
        track=opened["track"],
        record=selected["record"],
        selection=selected["selection"],
    )


def create_arguments(opened):
    return {
        "purpose": "Run the exact declared source",
        "completion": "Deliver its exact value",
        "budget": 64,
        "capabilities": ["text"],
        "execution": opened["children"][0]["execution"],
    }


def bootstrap(adapter, opened, child):
    return request(
        adapter,
        "bootstrap-orchestration-child",
        session=opened["session"],
        track=opened["track"],
        childTrack=child,
    )


def select_child(adapter, document, index):
    context = document["contexts"][index]
    public = {
        **document["public"],
        "timeline": document["public"]["timeline"][: context["visibleThrough"]],
    }
    return adapter.select(
        session=public["timeline"][0]["fields"]["session"],
        public=public,
        contexts=document["contexts"][:index],
        context=context,
    )


def commit_child(adapter, document, index, selected):
    context = document["contexts"][index]
    public = {
        **document["public"],
        "timeline": document["public"]["timeline"][: context["visibleThrough"] + 1],
    }
    return adapter.commit(
        session=public["timeline"][0]["fields"]["session"],
        public=public,
        contexts=document["contexts"][: index + 1],
        invocationId=context["invocationId"],
        record=selected["record"],
        selection=selected["selection"],
    )


def targets(document):
    return [
        document["public"]["timeline"][context["visibleThrough"]]["fields"]["line"].encode()
        for context in document["contexts"]
    ]


def teacher_document():
    path = Path(__file__).parent / "fixtures/semantic-orchestration-teacher-v8.json.gz.b64"
    return json.loads(gzip.decompress(base64.b64decode(path.read_bytes())))


def rehash(packet):
    packet["digest"] = stable_algorithm_hash(
        {key: value for key, value in packet.items() if key != "digest"}
    )
    return packet


def recover(open_adapter, adapter, session, packet):
    expected = continuation_fingerprint(adapter.sessions[session].runtime)
    adapter.cancel(session=session, reason="Checkpoint recovery boundary")
    restored = open_adapter()
    receipt = restored.restore(checkpoint=json.loads(json.dumps(packet)))
    assert receipt["hostDispatches"] == 0
    assert continuation_fingerprint(restored.sessions[session].runtime) == expected
    return restored, receipt


def run_complete_sequence():
    document = example()
    with real_runtime() as (open_adapter, decoder):
        adapter = open_adapter()
        opened = open_coordinator(adapter)
        session, track = opened["session"], opened["track"]
        before = continuation_fingerprint(adapter.sessions[session].runtime)
        selected = chosen(adapter, opened, "track_create", create_arguments(opened))
        assert selected["selection"]["origin"] == "controlled"
        assert not selected["selection"]["modelSelected"]
        assert continuation_fingerprint(adapter.sessions[session].runtime) == before
        assert len(adapter.sessions[session].runtime.tracks.tracks) == 1
        # Another session or a rejected attempt may consume randomness while
        # this candidate waits for acknowledgement. Host operations retain
        # their own RNG cuts, independently of candidate generation.
        torch.rand(3)
        created = acknowledge(adapter, opened, selected)
        child = created["track"]["address"]
        assert created["track"]["status"] == "waiting" and not created["track"]["bootstrap_ready"]
        assert created["track"]["parent"] == track
        assert created["track"]["execution"]["completion_target"] == track
        packet = adapter.checkpoint()
        adapter, receipt = recover(open_adapter, adapter, session, packet)
        assert receipt["consumedAttempts"] == 1
        with pytest.raises(ValueError, match="bootstrap"):
            select_child(adapter, document, 0)
        torch.rand(7)
        bootstrap(adapter, opened, child)
        work = adapter.sessions[session]
        assert work.runtime.execution_publication[child] == 2
        assert (
            work.runtime.execution_programs[opened["children"][0]["execution"]["program_id"]]
            == ingress()["timeline"][0]
        )
        before = continuation_fingerprint(work.runtime)
        assert bootstrap(adapter, opened, child)["bootstrapped"]
        assert continuation_fingerprint(work.runtime) == before
        acknowledge(
            adapter,
            opened,
            chosen(
                adapter,
                opened,
                "track_join",
                {"address": track, "completion_dependency": created["completion_dependency"]},
            ),
        )
        status = adapter.status_orchestration(session=session, track=track)
        assert status["status"] == "waiting"
        assert (
            next(item for item in status["tracks"] if item["address"] == child)["status"]
            == "active"
        )
        packet = adapter.checkpoint()
        adapter, _ = recover(open_adapter, adapter, session, packet)
        assert adapter.status_orchestration(session=session, track=track)["status"] == "waiting"
        assert adapter.sessions[session].runtime.execution_publication[child] == 2
        decoder["records"] = targets(document)
        for index in range(len(document["contexts"])):
            before = continuation_fingerprint(adapter.sessions[session].runtime)
            selected = select_child(adapter, document, index)
            assert all(
                fragment["modelTrack"] == child for fragment in selected["selection"]["fragments"]
            )
            assert continuation_fingerprint(adapter.sessions[session].runtime) == before
            assert not adapter.sessions[session].runtime.tool_routes
            commit_child(adapter, document, index, selected)
        assert not decoder["records"]
        invocation = document["contexts"][0]["invocationId"]
        complete = {
            "session": session,
            "public": document["public"],
            "contexts": document["contexts"],
            "invocationId": invocation,
        }
        before = continuation_fingerprint(adapter.sessions[session].runtime)
        with pytest.raises(ValueError, match="Caller completion"):
            adapter.complete(**complete, completion={"kind": "return", "value": ["num", 99]})
        assert continuation_fingerprint(adapter.sessions[session].runtime) == before
        torch.rand(5)
        delivered = adapter.complete(**complete)
        exact = delivered["childCompletion"]
        assert exact["completion"] == {"kind": "return", "value": ["ref", "o0"]}
        assert exact["heap"]
        before = continuation_fingerprint(adapter.sessions[session].runtime)
        assert adapter.complete(**complete) == delivered
        assert continuation_fingerprint(adapter.sessions[session].runtime) == before
        delivered_packet = adapter.checkpoint()
        state = adapter.status_orchestration(session=session, track=track)
        assert state["status"] == "active"
        assert (
            len(state["receipts"]) == 1
            and state["receipts"][0]["dependency"] == created["completion_dependency"]
        )
        acknowledge(
            adapter,
            opened,
            chosen(
                adapter,
                opened,
                "track_progress",
                {"address": track, "progress": json.dumps(exact["completion"])},
            ),
        )
        acknowledge(adapter, opened, chosen(adapter, opened, "track_complete", {"address": track}))
        finished = adapter.status_orchestration(session=session, track=track)
        assert finished["status"] == "completed"
        assert all(item["status"] == "completed" for item in finished["tracks"])
        before = continuation_fingerprint(adapter.sessions[session].runtime)
        assert adapter.complete(**complete) == delivered
        assert continuation_fingerprint(adapter.sessions[session].runtime) == before
        for changed in (
            {
                **complete,
                "public": {**complete["public"], "timeline": complete["public"]["timeline"][:-1]},
            },
            {**complete, "contexts": []},
            {**complete, "completion": {"kind": "throw", "value": ["num", 99]}},
        ):
            with pytest.raises(ValueError):
                adapter.complete(**changed)
            assert continuation_fingerprint(adapter.sessions[session].runtime) == before
        packet = adapter.checkpoint()
        decoder["records"] = targets(document)
        closed_adapter = adapter
        adapter, receipt = recover(open_adapter, adapter, session, packet)
        historical = closed_adapter.complete(**complete)
        assert historical["childCompletion"] == exact and historical["sessionRetained"] is False
        assert not closed_adapter.sessions and not closed_adapter.runtime.memory.sessions
        assert not decoder["records"] and receipt["consumedAttempts"] == 6
        assert adapter.status_orchestration(session=session, track=track) == finished
        assert not adapter.sessions[session].runtime.tool_routes
        assert not adapter.sessions[session].runtime.selected_groups
        with pytest.raises(ValueError, match="Terminal"):
            bootstrap(adapter, opened, child)
        decoder["records"] = targets(document)
        cancelled = open_adapter()
        cancelled.restore(checkpoint=delivered_packet)
        assert not decoder["records"]
        cancelled.cancel_orchestration(
            session=session, track=track, reason="Terminal receipt read control"
        )
        before = continuation_fingerprint(cancelled.sessions[session].runtime)
        assert cancelled.complete(**complete) == delivered
        assert continuation_fingerprint(cancelled.sessions[session].runtime) == before
        with pytest.raises(ValueError, match="Terminal"):
            bootstrap(cancelled, opened, child)
        return {
            "coordinatorActions": 4,
            "childRecords": len(document["contexts"]),
            "typedCompletionDeliveries": len(finished["receipts"]),
            "reservedAndWaitingRestored": True,
            "hostDispatches": receipt["hostDispatches"],
            "sourceAdmissionClaim": False,
        }


def _observed_messages(events):
    messages, pending = [], []
    for event in events:
        if pending:
            assert event.track == pending[0].track
            assert event.role_id == pending[0].role_id
        pending.append(event)
        if event.terminal:
            messages.append(
                (
                    pending[0].track,
                    b"".join(item.payload for item in pending),
                    pending[0].role_id,
                )
            )
            pending = []
    assert not pending
    return messages


def test_controlled_native_rows_match_source_qualified_training_projection_and_owners():
    document = teacher_document()
    validation = validate_orchestration_transcript_v8(document)
    transcript_events = validation["transcript"]["events"]
    materialized = orchestration_transcript_rows_v8(validation)
    raw_observations = []
    original_observe = MultiTrackRuntime.observe

    def capture_observations(runtime, session, events, *, effort=6):
        ready = tuple(events)
        raw_observations.extend(ready)
        return original_observe(runtime, session, ready, effort=effort)

    def exact_json(value):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        )

    def find_message(messages, payload, track):
        matches = [
            (index, role)
            for index, (owner, observed, role) in enumerate(messages)
            if owner == track and observed == payload
        ]
        assert len(matches) == 1, (track, payload[:80], len(matches))
        return matches[0]

    with patch.object(MultiTrackRuntime, "observe", capture_observations):
        with real_runtime() as (open_adapter, decoder):
            adapter = open_adapter()
            session = validation["session"]
            opened = adapter.open_orchestration(
                session=session,
                public=validation["initialPublic"],
                origin="controlled",
                childBudget=100000,
            )
            coordinator, child = opened["track"], validation["childTrack"]
            assert opened["children"] == validation["initialChildren"]
            status = adapter.status_orchestration(session=session, track=coordinator)
            assert (
                next(item for item in status["tracks"] if item["address"] == coordinator)["budget"]
                == 100000
            )
            ingress = canonical_semantic_observation(
                semantic_orchestration_input_v8(
                    public=validation["initialPublic"], children=validation["initialChildren"]
                )
            )
            messages = _observed_messages(raw_observations)
            ingress_index, ingress_role = find_message(messages, ingress, coordinator)
            assert ingress_role == ROLE_TO_ID["user"]
            assert materialized["buckets"][2][0]["payload"] == ingress

            def select_commit(ordinal):
                context_event = transcript_events[ordinal - 1]
                assert context_event["kind"] == "decision"
                assert (
                    orchestration_context(adapter, session=session, track=coordinator)
                    == context_event["context"]
                )
                name, arguments, target = validation["decisions"][ordinal]
                selected = request(
                    adapter,
                    "select-orchestration",
                    session=session,
                    track=coordinator,
                    controlledRecord=target,
                )
                assert selected["record"] == target
                target_spec = next(
                    spec
                    for specs in materialized["buckets"].values()
                    for spec in specs
                    if spec["recordId"] == f"semantic:v8:orchestration:target:{ordinal}"
                )
                assert target_spec["payload"] == target.encode("utf-8")
                assert selected["selection"]["contextDigest"] == stable_algorithm_hash(
                    context_event["context"]
                )
                assert selected["selection"]["origin"] == "controlled"
                assert selected["selection"]["modelSelected"] is False
                outcome = acknowledge(adapter, opened, selected)
                assert outcome["schema"] == "semantic-orchestration-commit/8"
                assert outcome["selectionId"] == selected["selection"]["selectionId"]
                assert outcome["committed"] is True
                acknowledgement = transcript_events[ordinal]
                assert acknowledgement["kind"] == "ack"
                teacher_receipt = acknowledgement["receipt"]
                assert teacher_receipt["schema"] == "semantic-orchestration-teacher-commit/1"
                assert outcome["track"] == teacher_receipt["track"]
                projection = semantic_orchestration_ack_observation_v8(
                    track=teacher_receipt["track"],
                    completion_dependency=teacher_receipt.get("completion_dependency"),
                )
                if "completion_dependency" in teacher_receipt:
                    assert (
                        outcome["completion_dependency"] == teacher_receipt["completion_dependency"]
                    )
                status = adapter.status_orchestration(session=session, track=coordinator)
                installed_track = next(
                    item
                    for item in status["tracks"]
                    if item["address"] == teacher_receipt["track"]["address"]
                )
                assert installed_track == teacher_receipt["track"]
                payload = canonical_semantic_observation(projection)
                messages = _observed_messages(raw_observations)
                ack_index, ack_role = find_message(messages, payload, coordinator)
                assert ack_role == ROLE_TO_ID["environment"]
                ack_spec = next(
                    spec
                    for specs in materialized["buckets"].values()
                    for spec in specs
                    if spec["recordId"]
                    == f"semantic:v8:orchestration:ack:{acknowledgement['ordinal']}"
                )
                assert ack_spec["payload"] == payload
                return outcome, ack_index

            created, create_ack_index = select_commit(1)
            assert created["track"]["address"] == child
            assert created["completion_dependency"] == validation["dependency"]
            bootstrap_event = transcript_events[2]
            bootstrap = adapter.bootstrap_orchestration_child(
                session=session, track=coordinator, childTrack=child
            )
            assert bootstrap == bootstrap_event["receipt"]
            messages = _observed_messages(raw_observations)
            program, invocation = validation["initialPublic"]["timeline"]
            program_index, program_role = find_message(messages, exact_json(program), child)
            invocation_index, invocation_role = find_message(
                messages, exact_json(invocation), child
            )
            bootstrap_index, bootstrap_role = find_message(
                messages, canonical_semantic_observation(bootstrap), coordinator
            )
            assert program_role == invocation_role == ROLE_TO_ID["user"]
            assert bootstrap_role == ROLE_TO_ID["environment"]
            assert create_ack_index < program_index < invocation_index < bootstrap_index

            joined, join_ack_index = select_commit(4)
            assert joined["track"]["status"] == "waiting"
            status = adapter.status_orchestration(session=session, track=coordinator)
            assert status["status"] == "waiting"
            assert (
                next(item for item in status["tracks"] if item["address"] == child)["status"]
                == "active"
            )
            decoder["records"] = targets(document)
            for index, context in enumerate(document["contexts"]):
                selected = select_child(adapter, document, index)
                assert all(
                    fragment["modelTrack"] == child
                    for fragment in selected["selection"]["fragments"]
                )
                assert commit_child(adapter, document, index, selected)
                context_payload = canonical_semantic_observation(context)
                messages = _observed_messages(raw_observations)
                context_index, context_role = find_message(messages, context_payload, child)
                assert context_role == ROLE_TO_ID["system"]
                assert join_ack_index < context_index
            assert not decoder["records"]

            invocation_id = validation["invocationId"]
            completion = adapter.complete(
                session=session,
                public=document["public"],
                contexts=document["contexts"],
                invocationId=invocation_id,
            )
            exact_completion = completion["childCompletion"]
            assert exact_completion == validation["transcript"]["child"]["completion"]
            delivered = validation["transcript"]["child"]["delivered"]
            messages = _observed_messages(raw_observations)
            delivered_index, delivered_role = find_message(
                messages, exact_json(delivered), coordinator
            )
            completion_index, completion_role = find_message(
                messages, canonical_semantic_observation(exact_completion), coordinator
            )
            delivered_row = next(
                spec
                for specs in materialized["buckets"].values()
                for spec in specs
                if spec["recordId"] == "semantic:v8:orchestration:delivery:6"
            )
            assert delivered_row["payload"] == exact_json(delivered)
            assert delivered["schema"] == "semantic-completion/4"
            assert exact_completion["schema"] == "semantic-child-completion/8"
            assert delivered_role == completion_role == ROLE_TO_ID["environment"]
            assert delivered_index < completion_index
            assert join_ack_index < delivered_index

            assert (
                orchestration_context(adapter, session=session, track=coordinator)
                == transcript_events[6]["context"]
            )
            progress, _ = select_commit(7)
            assert progress["track"]["progress"] == validation["decisions"][7][1]["progress"]
            assert (
                orchestration_context(adapter, session=session, track=coordinator)
                == transcript_events[8]["context"]
            )
            finished, _ = select_commit(9)
            status = adapter.status_orchestration(session=session, track=coordinator)
            tracks = {item["address"]: item for item in status["tracks"]}
            assert finished["track"]["status"] == "completed"
            assert status["status"] == "completed"
            assert tracks[coordinator]["budget"] == 99996
            assert tracks[child]["budget"] == 62
            assert ingress_index < create_ack_index < program_index < invocation_index


def test_real_shared_tracks_reserved_and_waiting_restore_then_exact_completion():
    assert run_complete_sequence() == {
        "coordinatorActions": 4,
        "childRecords": 2,
        "typedCompletionDeliveries": 1,
        "reservedAndWaitingRestored": True,
        "hostDispatches": 0,
        "sourceAdmissionClaim": False,
    }


def test_real_shared_sequence_runs_in_fresh_process_without_legacy_clients():
    code = r"""
import importlib.abc, json, sys
blocked={"algorithm_v7","detailed_v7","detailed_harness_v7","detailed_context_v7","semantic_harness_v6","semantic_contract","semantic_contract_v7","witness_v6","semantic_admission","semantic_corpus","semantic_budget"}
class NoLegacy(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.rsplit(".",1)[-1] in blocked: raise AssertionError("Legacy wire dependency: "+fullname)
sys.meta_path.insert(0,NoLegacy())
from tests.test_semantic_v8_orchestration import run_complete_sequence
receipt=run_complete_sequence()
assert not [name for name in sys.modules if name.rsplit(".",1)[-1] in blocked]
print(json.dumps(receipt))
"""
    process = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        env=dict(os.environ),
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
    )
    assert process.returncode == 0, process.stderr
    assert json.loads(process.stdout)["reservedAndWaitingRestored"]


def test_exact_acknowledgement_lost_ack_retry_and_cancellation_race():
    with real_runtime() as (open_adapter, _):
        adapter = open_adapter()
        opened = open_coordinator(adapter)
        session, track = opened["session"], opened["track"]
        selected = chosen(adapter, opened, "track_create", create_arguments(opened))
        before = continuation_fingerprint(adapter.sessions[session].runtime)
        for mutation in (
            lambda value: value.update(record=value["record"] + " "),
            lambda value: value["selection"].update(contextDigest="a" * 64),
        ):
            changed = copy.deepcopy(selected)
            mutation(changed)
            with pytest.raises(ValueError, match="exact pending"):
                acknowledge(adapter, opened, changed)
            assert continuation_fingerprint(adapter.sessions[session].runtime) == before
        created = acknowledge(adapter, opened, selected)
        before = continuation_fingerprint(adapter.sessions[session].runtime)
        assert acknowledge(adapter, opened, selected) == created
        assert continuation_fingerprint(adapter.sessions[session].runtime) == before
        child = created["track"]["address"]
        forged = copy.deepcopy(created["completion_dependency"])
        forged["invocation"] = "foreign"
        selected = chosen(
            adapter, opened, "track_join", {"address": track, "completion_dependency": forged}
        )
        with pytest.raises(ValueError, match="forged"):
            acknowledge(adapter, opened, selected)
        assert continuation_fingerprint(adapter.sessions[session].runtime) == before
        adapter.reject(
            session=session,
            selectionId=selected["selection"]["selectionId"],
            reason="Forged dependency control",
        )
        pending = chosen(
            adapter,
            opened,
            "track_join",
            {"address": track, "completion_dependency": created["completion_dependency"]},
        )
        adapter.cancel_orchestration(
            session=session, track=track, reason="Cancellation won before acknowledgement"
        )
        status = adapter.status_orchestration(session=session, track=track)
        assert all(item["status"] == "cancelled" for item in status["tracks"])
        with pytest.raises(ValueError, match="Terminal"):
            acknowledge(adapter, opened, pending)
        with pytest.raises(ValueError, match="Terminal"):
            bootstrap(adapter, opened, child)
        with pytest.raises(ValueError, match="Terminal"):
            select_child(adapter, example(), 0)
        assert adapter.sessions[session].public == ingress()
        assert not adapter.sessions[session].runtime.tool_routes
        packet = adapter.checkpoint()
        restored = open_adapter()
        restored.restore(checkpoint=packet)
        assert restored.status_orchestration(session=session, track=track) == status
        assert restored.attempts[-1]["status"] == "cancelled"


def test_reserved_pending_candidate_is_discarded_but_consumed_attempts_are_restored():
    with real_runtime() as (open_adapter, _):
        adapter = open_adapter()
        opened = open_coordinator(adapter)
        session, track = opened["session"], opened["track"]
        selected = chosen(adapter, opened, "track_create", create_arguments(opened))
        packet = adapter.checkpoint()
        before = continuation_fingerprint(adapter.sessions[session].runtime)
        restored = open_adapter()
        restored.restore(checkpoint=packet)
        assert continuation_fingerprint(restored.sessions[session].runtime) == before
        assert restored.sessions[session].pending is None
        assert len(restored.sessions[session].runtime.tracks.tracks) == 1
        assert restored.attempts[0]["status"] == "abandoned-on-restore"
        with pytest.raises(ValueError, match="exact pending"):
            acknowledge(restored, opened, selected)
        denied = open_adapter(controlled=False)
        with pytest.raises(ValueError, match="settings|authority"):
            denied.restore(checkpoint=packet)
        assert not denied.sessions and not denied.runtime.memory.sessions
        assert track == restored.status_orchestration(session=session, track=track)["track"]


def test_rehashed_changed_bindings_and_reordered_operations_refuse_atomically():
    with real_runtime() as (open_adapter, _):
        adapter = open_adapter()
        opened = open_coordinator(adapter)
        session, track = opened["session"], opened["track"]
        created = acknowledge(
            adapter, opened, chosen(adapter, opened, "track_create", create_arguments(opened))
        )
        bootstrap(adapter, opened, created["track"]["address"])
        acknowledge(
            adapter,
            opened,
            chosen(
                adapter,
                opened,
                "track_join",
                {"address": track, "completion_dependency": created["completion_dependency"]},
            ),
        )
        packet = adapter.checkpoint()
        mutations = [
            lambda saved: saved["orchestration"]["children"][0]["execution"].update(
                capability_grants=["host.write"]
            ),
            lambda saved: saved["orchestration"]["children"][0]["execution"].update(
                completion_target="foreign"
            ),
            lambda saved: saved["orchestration"]["operations"].reverse(),
            lambda saved: saved["orchestration"]["operations"].insert(
                2, copy.deepcopy(saved["orchestration"]["operations"][1])
            ),
            lambda saved: saved["orchestration"]["operations"].pop(0),
            lambda saved: saved["orchestration"]["decisions"][0]["outcome"][
                "completion_dependency"
            ].update(invocation="foreign"),
        ]
        for mutation in mutations:
            changed = copy.deepcopy(packet)
            mutation(changed["sessions"][session])
            rehash(changed)
            restored = open_adapter()
            before_random = capture_random_state()
            with pytest.raises(ValueError):
                restored.restore(checkpoint=changed)
            assert (
                not restored.sessions
                and not restored.attempts
                and not restored.runtime.memory.sessions
            )
            assert capture_random_state() == before_random


def test_actual_model_failure_preserves_original_bytes_without_controlled_fallback():
    with real_runtime() as (open_adapter, decoder):
        adapter = open_adapter(controlled=False, max_tokens=8)
        decoder["records"] = None
        public = ingress()
        session = public["timeline"][0]["fields"]["session"]
        with pytest.raises(ValueError, match="authority"):
            adapter.open_orchestration(session=session, public=public, origin="controlled")
        opened = adapter.open_orchestration(session=session, public=public)
        before = continuation_fingerprint(adapter.sessions[session].runtime)
        with pytest.raises(ValueError, match="cannot receive controlled"):
            adapter.select_orchestration(
                session=session, track=opened["track"], controlledRecord="{}"
            )
        try:
            selected = adapter.select_orchestration(session=session, track=opened["track"])
        except NativeSelectionV8Error as error:
            evidence = error.evidence
        else:
            evidence = selected["selection"]
            with pytest.raises(ValueError):
                acknowledge(adapter, opened, selected)
        assert evidence["modelSelected"] and evidence["origin"] == "actual-model-selected"
        assert evidence["fragments"] and any(
            fragment["payloadHex"] is not None for fragment in evidence["fragments"]
        )
        assert continuation_fingerprint(adapter.sessions[session].runtime) == before
        assert len(adapter.sessions[session].runtime.tracks.tracks) == 1
        assert not adapter.sessions[session].runtime.controlled_fixture_actions
        adapter.cancel_orchestration(
            session=session, track=opened["track"], reason="Actual invalid output retained"
        )


def test_child_budget_uses_parent_balance_after_selecting_its_creation():
    with real_runtime() as (open_adapter, _):
        adapter = open_adapter()
        opened = adapter.open_orchestration(
            session="teacher1", public=ingress(), childBudget=100000, origin="controlled"
        )
        arguments = create_arguments(opened)
        arguments["budget"] = 100000
        selected = chosen(adapter, opened, "track_create", arguments)
        work = adapter.sessions["teacher1"]
        assert work.runtime.tracks.tracks[opened["track"]].budget == 100000
        assert work.pending["branch"].tracks.tracks[opened["track"]].budget < 100000
        before = continuation_fingerprint(work.runtime)
        with pytest.raises(ValueError, match="budget"):
            acknowledge(adapter, opened, selected)
        assert continuation_fingerprint(work.runtime) == before
        assert len(work.runtime.tracks.tracks) == 1


def test_interleaved_sessions_preserve_global_attempts_and_operation_rng_cuts():
    first = example()
    second = copy.deepcopy(first)
    for row in second["public"]["timeline"]:
        row["fields"]["session"] = "second"
    for context in second["contexts"]:
        context["prefixDigest"] = stable_algorithm_hash(
            second["public"]["timeline"][: context["visibleThrough"]]
        )
    with real_runtime() as (open_adapter, decoder):
        adapter = open_adapter()
        opened = open_coordinator(adapter)
        session, track = opened["session"], opened["track"]
        decoder["records"] = [
            targets(second)[0],
            targets(second)[0],
            targets(first)[0],
            targets(second)[1],
            targets(first)[1],
        ]
        waiting_ack = chosen(adapter, opened, "track_create", create_arguments(opened))
        rejected = select_child(adapter, second, 0)
        adapter.reject(
            session="second",
            selectionId=rejected["selection"]["selectionId"],
            reason="Interleaved rejected native attempt",
        )
        created = acknowledge(adapter, opened, waiting_ack)
        commit_child(adapter, second, 0, select_child(adapter, second, 0))
        bootstrap(adapter, opened, created["track"]["address"])
        acknowledge(
            adapter,
            opened,
            chosen(
                adapter,
                opened,
                "track_join",
                {"address": track, "completion_dependency": created["completion_dependency"]},
            ),
        )
        commit_child(adapter, first, 0, select_child(adapter, first, 0))
        commit_child(adapter, second, 1, select_child(adapter, second, 1))
        commit_child(adapter, first, 1, select_child(adapter, first, 1))
        assert not decoder["records"]
        for identity, document in (("second", second), (session, first)):
            adapter.complete(
                session=identity,
                public=document["public"],
                contexts=document["contexts"],
                invocationId=document["contexts"][0]["invocationId"],
            )
        packet = adapter.checkpoint()
        expected = {
            identity: continuation_fingerprint(work.runtime)
            for identity, work in adapter.sessions.items()
        }
        assert len(packet["attempts"]) == 7
        assert packet["attempts"][1]["status"] == "semantic-rejected"
        # Restoration groups sessions by their original opening, while every
        # candidate and host operation restores its own original RNG boundary.
        decoder["records"] = targets(first) + targets(second)
        restored = open_adapter()
        result = restored.restore(checkpoint=json.loads(json.dumps(packet)))
        assert not decoder["records"]
        assert result["restoredSessions"] == 2
        assert result["consumedAttempts"] == 7 and result["reexecutedSelections"] == 6
        assert capture_random_state() == packet["randomState"]
        assert {
            identity: continuation_fingerprint(work.runtime)
            for identity, work in restored.sessions.items()
        } == expected
        assert restored.attempts == packet["attempts"]
        assert restored.sessions[session].orchestration == adapter.sessions[session].orchestration
        assert restored.sessions["second"].public == second["public"]
