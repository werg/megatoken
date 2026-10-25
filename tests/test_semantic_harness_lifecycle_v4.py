"""Typed execution-track creation is not runnable before observed code and input."""

import json

import pytest

from megatoken.lifecycle import ExecutionBinding, TrackRegistry, lifecycle_tools


def binding(invocation="i1", grants=("demo.readAsync",)):
    return ExecutionBinding(
        program_id="counter",
        source_digest="a" * 64,
        entry="run",
        invocation_id=invocation,
        realm_id="r1",
        trigger_id="outside1",
        input_ref="input:sha256:deadbeef",
        realm_policy="fresh",
        completion_mode="await-returned-promise",
        capability_grants=grants,
    )


def test_executable_track_must_observe_code_and_input_before_activation():
    registry = TrackRegistry(max_active=1, max_waiting=4)
    parent = registry.create("Coordinate", budget=30)
    child = registry.create("Run a verified program", parent=parent.address,
                            execution=binding())
    assert child.status == "waiting"
    assert child.bootstrap_ready is False
    assert not registry.selectable(child.address, ())
    with pytest.raises(ValueError, match="bootstrap"):
        registry.change(child.address, "continue")
    with pytest.raises(ValueError, match="bootstrap"):
        registry.activate_execution(child.address, source_digest="wrong",
                                    input_ref=child.execution.input_ref)
    with pytest.raises(ValueError, match="capacity"):
        registry.activate_execution(child.address, source_digest="a" * 64,
                                    input_ref=child.execution.input_ref)
    registry.change(parent.address, "wait", prerequisites=("child:complete",))
    live = registry.activate_execution(child.address, source_digest="a" * 64,
                                       input_ref=child.execution.input_ref)
    assert live.bootstrap_ready
    assert live.status == "active"
    assert registry.selectable(live.address, ())
    assert registry.activate_execution(
        live.address, source_digest="a" * 64, input_ref=live.execution.input_ref
    ) is live
    assert len([name for name, _ in registry.transitions if name == "bootstrap"]) == 1


def test_no_duplicate_invocations_and_child_cannot_expand_grants():
    registry = TrackRegistry(max_active=1, max_waiting=5)
    parent = registry.create("Parent invocation", execution=binding("parent"))
    child = registry.create("Child", parent=parent.address,
                            execution=binding("child", ("demo.readAsync",)))
    assert child.status == "waiting"
    with pytest.raises(ValueError, match="expand"):
        registry.create("Unauthorized child", parent=parent.address,
                        execution=binding("bad", ("demo.store",)))
    with pytest.raises(ValueError, match="second track"):
        registry.create("Duplicate invocation", execution=binding("child"))
    assert registry.change(child.address, "cancel").status == "cancelled"
    with pytest.raises(ValueError, match="Closed|mutated"):
        registry.activate_execution(child.address, source_digest="a" * 64,
                                    input_ref=child.execution.input_ref)


def test_lifecycle_schema_exposes_versioned_typed_execution_binding():
    create = next(t for t in lifecycle_tools() if t["name"] == "track_create")
    field = create["parameters"]["properties"]["execution"]
    assert field["additionalProperties"] is False
    assert field["properties"]["schema"]["const"] == "semantic-execution-binding/4"
    assert {"program_id", "invocation_id", "realm_id", "source_digest", "input_ref"} <= set(
        field["required"]
    )
    assert json.dumps(field)


def test_binding_never_grants_itself_new_capabilities_or_uses_invalid_source():
    with pytest.raises(ValueError, match="source"):
        ExecutionBinding(
            program_id="p", source_digest="not pinned", entry="run",
            invocation_id="i", realm_id="r", trigger_id="t", input_ref="input",
        )
    with pytest.raises(ValueError, match="distinct"):
        binding(grants=("demo.readAsync", "demo.readAsync"))
