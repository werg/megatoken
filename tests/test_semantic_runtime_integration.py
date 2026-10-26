"""Controlled lifecycle actions use the actual model/runtime; no optimizer or teacher repair."""

import hashlib
import json
from dataclasses import asdict

import torch

from megatoken.generation.codec import ByteCodec
from megatoken.generation.policy import FacetedGenerator
from megatoken.lifecycle import ExecutionBinding
from megatoken.model import FacetedCore, ModelConfig
from megatoken.model.multitrack import MultiTrackGenerator
from megatoken.multitrack_runtime import MultiTrackRuntime
from megatoken.semantic_model_adapter import SelectedSemanticInference


def test_actual_committed_child_creation_bootstrap_capacity_join_and_completion():
    torch.set_num_threads(1)
    torch.manual_seed(17)
    model = FacetedGenerator(
        FacetedCore(ModelConfig.tiny(max_history_chunks=128, max_capacity=512)),
        ByteCodec.byte_fallback(),
    )
    model.eval()
    weights = {key: value.clone() for key, value in model.state_dict().items()}
    runtime = MultiTrackRuntime(MultiTrackGenerator.of(model))
    adapter = SelectedSemanticInference(runtime, model_identity="a" * 64)
    state = adapter._opened("lifecycle")
    session = state.runtime_session
    session.tracks.max_active = 1
    source = "function run(){return 7;}"
    digest = hashlib.sha256(source.encode()).hexdigest()
    program = {
        "schema": "semantic-execution/5",
        "session": "lifecycle",
        "sequence": 1,
        "ownerSequence": 1,
        "track": "host",
        "kind": "program",
        "mode": "observed",
        "programId": "p",
        "source": source,
        "digest": digest,
        "entry": "run",
        "capabilities": ["demo.readAsync"],
        "functions": [],
        "sites": [],
        "dependencies": [],
        "sourceMap": None,
        "sourceMetadata": None,
        "module": False,
        "languageProfile": "owned-javascript/1",
    }

    def invocation(identity, parent=None):
        return {
            "schema": "semantic-execution/5",
            "session": "lifecycle",
            "sequence": 2,
            "ownerSequence": 1,
            "track": identity,
            "kind": "invocation",
            "mode": "observed",
            "programId": "p",
            "invocationId": identity,
            "realmId": identity,
            "realmPolicy": "fresh",
            "triggerId": "outside:" + identity,
            "ingressKey": "outside:" + identity,
            "entry": "run",
            "sourceDigest": digest,
            "completionPolicy": "call-return",
            "inputGraph": {"roots": [], "objects": []},
            "receiver": ["undefined"],
            "taskRevision": 1,
            "activation": "call",
            "closureId": None,
            "parentInvocationId": parent,
            "deadline": None,
            "bindings": {},
        }

    def committed_action(name, arguments, owner):
        # Controlled injection is explicitly limited to the test's selected
        # output boundary; all validation, routing and publication are real.
        return runtime.commit_fixture_action(
            session,
            track=owner,
            payload=json.dumps({"name": name, "arguments": arguments}).encode(),
            fixture_digest=hashlib.sha256(b"child-bootstrap-join-fixture").hexdigest(),
            decision=len(session.controlled_fixture_actions),
            family="lifecycle",
        )

    try:
        with torch.inference_mode():
            parent = adapter._bootstrap(state, program, invocation("parent"))
            child_input = invocation("child", "parent")
            input_ref = runtime.register_execution_context(
                session, program=program, invocation=child_input
            )
            binding = ExecutionBinding(
                program_id="p",
                source_digest=digest,
                entry="run",
                invocation_id="child",
                realm_id="child",
                trigger_id="outside:child",
                input_ref=input_ref,
                completion_mode="call-return",
                capability_grants=("demo.readAsync",),
            )
            create = committed_action(
                "track_create",
                {
                    "purpose": "Child execution",
                    "completion": "Return its invocation result",
                    "budget": 8,
                    "capabilities": ["text"],
                    "execution": asdict(binding),
                },
                parent.actual_track,
            )
            outcomes = runtime.execute_lifecycle(session, create)
            child = outcomes[0]["track"]["address"]
            dependency = outcomes[0]["completion_dependency"]
            assert session.tracks.tracks[child].bootstrap_ready
            assert session.tracks.tracks[child].status == "waiting"
            assert runtime.execute_lifecycle(session, create) == []
            assert len(session.tracks.tracks) == 2
            history = list(session.history)
            creation = next(
                i for i, value in enumerate(history) if b'"completion_dependency"' in value
            )
            child_source = next(
                i
                for i, value in enumerate(history[creation + 1 :], creation + 1)
                if b'"kind":"program"' in value
            )
            child_invocation = next(
                i
                for i, value in enumerate(history[child_source + 1 :], child_source + 1)
                if b'"kind":"invocation"' in value
            )
            assert creation < child_source < child_invocation
            prefix = {
                "schema": "selected-inference-prefix/2",
                "programId": "p",
                "source": source,
                "sourceDigest": digest,
                "entry": "run",
                "invocationId": "parent",
                "trackId": "parent",
                "realmId": "parent",
                "completionPolicy": "call-return",
                "maxOutstanding": 64,
                "committed": [program, invocation("parent")],
            }
            packet = adapter.checkpoint("lifecycle", [prefix])
            restored = SelectedSemanticInference(runtime, model_identity="a" * 64)
            recovery = restored.restore(packet)
            assert recovery["pendingExecutions"] == 1
            restored_session = restored.sessions["lifecycle"].runtime_session
            assert set(restored_session.tracks.tracks) == {parent.actual_track, child}
            assert restored_session.tracks.tracks[child].bootstrap_ready
            assert restored_session.tracks.tracks[child].status == "waiting"
            assert restored_session.execution_source_events[child]
            adopted = adapter._bootstrap(state, program, child_input)
            assert adopted.actual_track == child
            assert len(session.tracks.tracks) == 2
            join = committed_action(
                "track_join",
                {"address": parent.actual_track, "completion_dependency": dependency},
                parent.actual_track,
            )
            runtime.execute_lifecycle(session, join)
            assert session.tracks.selectable(child, session.completed_events)
            assert session.tracks.tracks[parent.actual_track].status == "waiting"
            assert dependency["id"] not in session.completed_events
            completion = runtime.complete_execution(session, child, kind="return", value=["num", 7])
            assert completion["dependency"] == dependency
            assert session.tracks.selectable(parent.actual_track, session.completed_events)
            assert session.tracks.tracks[child].status == "completed"
            assert (
                runtime.complete_execution(session, child, kind="return", value=["num", 7])
                == completion
            )
            assert (
                sum(
                    receipt.get("schema") == "semantic-completion/4"
                    for receipt in session.execution_receipts
                )
                == 1
            )
            # Failed bootstrap consumes its reservation and is separately visible.
            missing = asdict(binding)
            missing.update(
                invocation_id="missing", realm_id="missing", trigger_id="outside:missing"
            )
            failed = committed_action(
                "track_create",
                {
                    "purpose": "Missing input",
                    "completion": "Publish failure",
                    "budget": 1,
                    "execution": missing,
                },
                parent.actual_track,
            )
            receipt = runtime.execute_lifecycle(session, failed)[0]
            assert session.tracks.tracks[receipt["track"]["address"]].status == "cancelled"
            assert session.execution_receipts[-1]["schema"] == "semantic-bootstrap/4"
            assert runtime.execute_lifecycle(session, failed) == []
        assert all(torch.equal(value, model.state_dict()[key]) for key, value in weights.items())
    finally:
        runtime.close()
