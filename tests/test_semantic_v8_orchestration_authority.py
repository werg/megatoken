"""Model-free authority controls for the bounded native V8 child bridge.

These exercise the actual neutral binding, lifecycle registry and independent
V8 replay. They do not qualify model continuation, source admission or training.
"""

import base64
import copy
import gzip
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from megatoken.execution_context import execution_binding, validate_execution_context
from megatoken.lifecycle import TrackRegistry
from megatoken.semantic_orchestration_v8 import (
    derive_completion,
    semantic_track,
    validate_action,
    validate_ingress,
)


def example():
    document = json.loads(
        gzip.decompress(
            base64.b64decode(
                (Path(__file__).parent / "fixtures/semantic-regions-v8.json.gz.b64").read_bytes()
            )
        )
    )
    row = document["examples"][0]
    return copy.deepcopy(row.get("artifact", row))


def ingress():
    document = example()
    public = copy.deepcopy(document["public"])
    public["timeline"] = public["timeline"][:2]
    return public


def work():
    public = ingress()
    children = validate_ingress("teacher1", public, max_active=1, child_budget=64)
    registry = TrackRegistry(max_active=1)
    coordinator = registry.create("Execute the declared V8 root", budget=100)
    child = children[0]
    child["execution"] = json.loads(
        json.dumps(asdict(execution_binding(**child, completion_target=coordinator.address)))
    )
    child.update(track=None, completion=None)
    return SimpleNamespace(
        runtime=SimpleNamespace(tracks=registry, completed_events=set(), execution_delivered=set()),
        orchestration={
            "track": coordinator.address,
            "initialPublic": public,
            "childBudget": 64,
            "children": children,
            "published": [],
        },
    )


def create(state):
    return {
        "name": "track_create",
        "arguments": {
            "purpose": "Run source-bound child",
            "completion": "Return its exact value",
            "budget": 64,
            "capabilities": ["text"],
            "execution": copy.deepcopy(state.orchestration["children"][0]["execution"]),
        },
    }


def reserve(state):
    action = create(state)
    validate_action(state, json.dumps(action))
    description = state.runtime.tracks.admit_execution(
        "selection1", parent=state.orchestration["track"], **action["arguments"]
    )
    state.orchestration["children"][0]["track"] = description.address
    return description, state.runtime.tracks.completion_dependency(description.address)


def test_ingress_binds_exact_source_input_parent_and_grants_without_legacy(tmp_path):
    public = ingress()
    source, invocation = public["timeline"]
    normalized_source, normalized_input, reference = validate_execution_context(source, invocation)
    binding = execution_binding(source, invocation, completion_target="track:0")
    assert normalized_source == source and normalized_input == invocation
    assert binding.source_digest == source["fields"]["digest"] and binding.input_ref == reference
    assert binding.invocation_id == invocation["fields"]["invocationId"]
    assert binding.completion_target == "track:0" and binding.capability_grants == ()
    normalized_source["fields"]["source"] = "changed"
    assert normalized_source != source
    path = tmp_path / "ingress.json"
    path.write_text(json.dumps(example()))
    code = r"""
import importlib.abc, json, sys
blocked = {"algorithm_v7", "detailed_v7", "detailed_harness_v7", "detailed_context_v7", "semantic_harness_v6", "semantic_contract", "semantic_contract_v7", "witness_v6", "semantic_admission", "semantic_corpus", "semantic_budget"}
class NoLegacy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.rsplit(".",1)[-1] in blocked: raise AssertionError("legacy: "+fullname)
sys.meta_path.insert(0,NoLegacy())
from megatoken.semantic_orchestration_v8 import validate_ingress, derive_completion
from megatoken.execution_context import execution_binding
x=json.load(open(sys.argv[1])); public={**x["public"],"timeline":x["public"]["timeline"][:2]}
children=validate_ingress("teacher1",public,max_active=1,child_budget=64)
binding=execution_binding(**children[0],completion_target="track:0")
completion,heap=derive_completion(x["public"],x["contexts"],binding.invocation_id)
print(json.dumps({"children":len(children),"completion":completion,"heap":heap,"legacy":sorted(name for name in sys.modules if name.rsplit(".",1)[-1] in blocked)}))
"""
    process = subprocess.run(
        [sys.executable, "-c", code, str(path)],
        env=dict(os.environ),
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert process.returncode == 0, process.stderr
    output = json.loads(process.stdout)
    assert output["children"] == 1 and output["completion"] == {
        "kind": "return",
        "value": ["ref", "o0"],
    }
    assert output["heap"] and not output["legacy"]


@pytest.mark.parametrize(
    "change",
    [
        lambda public: public["timeline"][0]["fields"].update(source="function run(){return 99}"),
        lambda public: public["timeline"][1]["fields"].update(sourceDigest="a" * 64),
        lambda public: public["timeline"][1]["fields"].update(session="different"),
        lambda public: public["timeline"][1]["fields"].update(parentInvocationId="foreign"),
        lambda public: public["timeline"][1]["fields"].update(activation="callback"),
        lambda public: public["timeline"][1]["fields"].update(taskRevision=2),
        lambda public: public["timeline"].append(copy.deepcopy(public["timeline"][1])),
    ],
)
def test_changed_or_dynamic_source_ingress_is_refused(change):
    public = ingress()
    change(public)
    with pytest.raises((ValueError, KeyError)):
        validate_ingress("teacher1", public, max_active=1, child_budget=64)
    with pytest.raises(ValueError):
        validate_ingress("teacher1", example()["public"], max_active=1, child_budget=64)


@pytest.mark.parametrize(
    "change",
    [
        lambda action: action["arguments"]["execution"].update(source_digest="a" * 64),
        lambda action: action["arguments"]["execution"].update(input_ref="a" * 64),
        lambda action: action["arguments"]["execution"].update(completion_target="foreign"),
        lambda action: action["arguments"]["execution"].update(capability_grants=["host.write"]),
        lambda action: action["arguments"].update(capabilities=["text", "host.write"]),
        lambda action: action["arguments"].update(budget=65),
        lambda action: action["arguments"].update(budget=True),
        lambda action: action["arguments"].update(parent="foreign"),
    ],
)
def test_child_creation_rejects_changed_scope_without_mutating_registry(change):
    state = work()
    action = create(state)
    before = copy.deepcopy(state.runtime.tracks.checkpoint())
    change(action)
    with pytest.raises(ValueError):
        validate_action(state, json.dumps(action))
    assert state.runtime.tracks.checkpoint() == before
    validate_action(state, json.dumps(create(state)))


def test_reserved_bootstrap_and_join_require_original_dependency_before_selection():
    state = work()
    invocation = state.orchestration["children"][0]["execution"]["invocation_id"]
    with pytest.raises(ValueError, match="reservation"):
        semantic_track(state, invocation, ingress())
    child, dependency = reserve(state)
    with pytest.raises(ValueError, match="bootstrap"):
        semantic_track(state, invocation, ingress())
    join = {
        "name": "track_join",
        "arguments": {"address": state.orchestration["track"], "completion_dependency": dependency},
    }
    for key, value in (("invocation", "other"), ("revision", 1), ("track", "foreign")):
        forged = copy.deepcopy(join)
        forged["arguments"]["completion_dependency"][key] = value
        with pytest.raises(ValueError, match="forged"):
            validate_action(state, json.dumps(forged))
    validate_action(state, json.dumps(join))
    state.runtime.tracks.activate_execution(
        child.address,
        source_digest=child.execution.source_digest,
        input_ref=child.execution.input_ref,
    )
    with pytest.raises(ValueError, match="readiness"):
        semantic_track(state, invocation, ingress())
    state.runtime.tracks.wait_for_execution(state.orchestration["track"], dependency)
    assert semantic_track(state, invocation, ingress()) == child.address
    with pytest.raises(ValueError, match="reused"):
        validate_action(state, json.dumps(create(state)))
    state.runtime.tracks.change(state.orchestration["track"], "cancel")
    with pytest.raises(ValueError, match="Terminal"):
        semantic_track(state, invocation, ingress())


def test_exact_completion_and_progress_never_accept_caller_supplied_result():
    document = example()
    invocation = document["contexts"][0]["invocationId"]
    completion, heap = derive_completion(document["public"], document["contexts"], invocation)
    assert completion == {"kind": "return", "value": ["ref", "o0"]} and heap
    state = work()
    reserve(state)
    child = state.orchestration["children"][0]
    progress = {
        "name": "track_progress",
        "arguments": {"address": state.orchestration["track"], "progress": json.dumps(completion)},
    }
    with pytest.raises(ValueError, match="completion"):
        validate_action(state, json.dumps(progress))
    child["completion"] = {"completion": completion}
    validate_action(state, json.dumps(progress))
    changed = copy.deepcopy(progress)
    changed["arguments"]["progress"] = json.dumps({"kind": "throw", "value": ["num", 99]})
    with pytest.raises(ValueError, match="completion"):
        validate_action(state, json.dumps(changed))
    finished = {"name": "track_complete", "arguments": {"address": state.orchestration["track"]}}
    with pytest.raises(ValueError, match="every declared child"):
        validate_action(state, json.dumps(finished))
    state.orchestration["published"].append(invocation)
    validate_action(state, json.dumps(finished))
    with pytest.raises(ValueError, match="completion"):
        validate_action(state, json.dumps(progress))
    with pytest.raises(ValueError, match="duplicate"):
        validate_action(
            state,
            '{"name":"track_complete","name":"track_cancel","arguments":{"address":"track:0"}}',
        )


def test_existing_v6_context_keeps_json_normalization_and_input_identity():
    import hashlib

    from megatoken.data.semantic_contract import CONTRACT

    class InputMapping(dict):
        pass

    source, invocation = copy.deepcopy(
        CONTRACT["fixtures"]["envelope"]["valid"][0]["public"]["timeline"][:2]
    )
    source = InputMapping(source)
    source["capabilities"] = tuple(source["capabilities"])
    normalized_source, normalized_input, reference = validate_execution_context(source, invocation)
    assert type(normalized_source) is dict
    assert type(normalized_source["capabilities"]) is list
    assert normalized_input == invocation
    assert (
        reference
        == hashlib.sha256(
            json.dumps(
                invocation["inputGraph"], sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
        ).hexdigest()
    )
