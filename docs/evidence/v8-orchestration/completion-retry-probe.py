"""Model-free probe of immutable completion receipt retries.

Uses the actual public completion method, original checked V8 fixture, native
semantic replay, and typed lifecycle registry. The already-delivered model
state is assembled explicitly; this is not model execution evidence.
"""
import copy
import hashlib
import json
from pathlib import Path
from threading import RLock

from megatoken.data.prefix_v8 import validate_v8_committed_prefix
from megatoken.data.semantic_json import stable_algorithm_hash
from megatoken.semantic_model_v8 import NativeSemanticInferenceV8
from megatoken.semantic_orchestration_v8 import derive_completion
from tests.test_semantic_v8_orchestration_authority import example, reserve, work


ROOT = Path(__file__).resolve().parents[3]


def setup():
    document = example()
    state = work()
    child_track, dependency = reserve(state)
    registry = state.runtime.tracks
    binding = child_track.execution
    registry.activate_execution(
        child_track.address, source_digest=binding.source_digest, input_ref=binding.input_ref
    )
    parent = state.orchestration["track"]
    registry.wait_for_execution(parent, dependency, delivered=set())
    invocation = document["contexts"][0]["invocationId"]
    exact, heap = derive_completion(document["public"], document["contexts"], invocation)
    native_receipt = registry.complete_execution(child_track.address, **exact)
    state.runtime.execution_delivered.add(dependency["id"])
    state.runtime.completed_events.add(dependency["id"])
    registry.wake_ready_executions(delivered=state.runtime.completed_events)
    state.runtime.execution_receipts = [native_receipt]
    session = document["public"]["timeline"][0]["fields"]["session"]
    completion_receipt = {
        "schema": "semantic-child-completion/8",
        "session": session,
        "invocationId": invocation,
        "track": child_track.address,
        "completion": exact,
        "heap": heap,
        "publicDigest": stable_algorithm_hash(document["public"]),
    }
    # A future repair may bind contexts explicitly; only add the field when
    # the implementation has introduced that exact wire field.
    source = (ROOT / "src/megatoken/semantic_orchestration_v8.py").read_text()
    if '"contextsDigest"' in source:
        completion_receipt["contextsDigest"] = stable_algorithm_hash(document["contexts"])
    state.orchestration["children"][0]["completion"] = completion_receipt
    state.public = copy.deepcopy(document["public"])
    state.contexts = copy.deepcopy(document["contexts"])
    state.completed = {invocation}
    state.pending = None
    adapter = object.__new__(NativeSemanticInferenceV8)
    adapter.lock = RLock()
    adapter.sessions = {session: state}
    adapter.closed_sessions = {}
    request = {
        "session": session,
        "public": document["public"],
        "contexts": document["contexts"],
        "invocationId": invocation,
    }
    return adapter, state, request


def attempt(change):
    adapter, state, request = setup()
    original = adapter.complete(**request)
    change(state)
    before = copy.deepcopy((state.orchestration, state.runtime.tracks.checkpoint(), state.public))
    try:
        actual = adapter.complete(**request)
        result = {"accepted": True, "exactOriginalReceipt": actual == original}
    except Exception as error:
        result = {"accepted": False, "errorType": type(error).__name__, "error": str(error)}
    result["authorityUnchanged"] = before == (
        state.orchestration, state.runtime.tracks.checkpoint(), state.public
    )
    return result


def advance_observed_prefix(state):
    extra = copy.deepcopy(state.public["timeline"][0])
    extra["sequence"] = len(state.public["timeline"]) + 1
    extra["fields"]["programId"] = "program2"
    state.public["timeline"].append(extra)
    validate_v8_committed_prefix(state.public, state.contexts)


paths = [
    "src/megatoken/semantic_orchestration_v8.py",
    "src/megatoken/semantic_model_v8.py",
    "tests/fixtures/semantic-regions-v8.json.gz.b64",
]
print(json.dumps({
    "scope": "model-free historical completion retry; no actual-model or dispatch claim",
    "hashes": {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in paths},
    "cases": {
        "liveExactRetry": attempt(lambda state: None),
        "afterCoordinatorComplete": attempt(
            lambda state: state.runtime.tracks.change(state.orchestration["track"], "complete")
        ),
        "afterCoordinatorCancel": attempt(
            lambda state: state.runtime.tracks.change(state.orchestration["track"], "cancel")
        ),
        "afterNewObservedProgram": attempt(advance_observed_prefix),
        "duringLaterPendingSelection": attempt(lambda state: setattr(state, "pending", {})),
    },
}, indent=2, sort_keys=True))
