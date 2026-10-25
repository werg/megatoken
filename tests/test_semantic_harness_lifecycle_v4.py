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
    parent = registry.create("Coordinate", budget=30, capabilities=("text", "demo.readAsync"))
    child = registry.create("Run a verified program", parent=parent.address, execution=binding())
    assert child.status == "waiting"
    assert child.bootstrap_ready is False
    assert not registry.selectable(child.address, set())
    with pytest.raises(ValueError, match="bootstrap"):
        registry.change(child.address, "continue")
    with pytest.raises(ValueError, match="bootstrap"):
        registry.activate_execution(
            child.address, source_digest="wrong", input_ref=child.execution.input_ref
        )
    queued = registry.activate_execution(
        child.address, source_digest="a" * 64, input_ref=child.execution.input_ref
    )
    assert queued.bootstrap_ready and queued.status == "waiting"
    registry.change(parent.address, "wait", prerequisites=("child:complete",))
    live = registry.wake_execution(child.address, delivered=set())
    assert live.bootstrap_ready
    assert live.status == "active"
    assert registry.selectable(live.address, set())
    assert (
        registry.activate_execution(
            live.address, source_digest="a" * 64, input_ref=live.execution.input_ref
        )
        is live
    )
    assert len([name for name, _ in registry.transitions if name == "bootstrap"]) == 1


def test_no_duplicate_invocations_and_child_cannot_expand_grants():
    registry = TrackRegistry(max_active=1, max_waiting=5)
    parent = registry.create("Parent invocation", execution=binding("parent"))
    child = registry.create(
        "Child", parent=parent.address, execution=binding("child", ("demo.readAsync",))
    )
    assert child.status == "waiting"
    with pytest.raises(ValueError, match="expand"):
        registry.create(
            "Unauthorized child", parent=parent.address, execution=binding("bad", ("demo.store",))
        )
    with pytest.raises(ValueError, match="second track"):
        registry.create("Duplicate invocation", execution=binding("child"))
    assert registry.change(child.address, "cancel").status == "cancelled"
    with pytest.raises(ValueError, match="Closed|mutated"):
        registry.activate_execution(
            child.address, source_digest="a" * 64, input_ref=child.execution.input_ref
        )


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
            program_id="p",
            source_digest="not pinned",
            entry="run",
            invocation_id="i",
            realm_id="r",
            trigger_id="t",
            input_ref="input",
        )
    with pytest.raises(ValueError, match="distinct"):
        binding(grants=("demo.readAsync", "demo.readAsync"))


def test_idempotent_ingress_checkpoint_and_independent_child_completion():
    registry = TrackRegistry(max_active=2, max_waiting=8)
    parent = registry.admit_execution(
        "outside:parent",
        purpose="Parent",
        execution=binding("parent"),
        capabilities=("text", "demo.readAsync"),
    )
    registry.activate_execution(
        parent.address, source_digest="a" * 64, input_ref=parent.execution.input_ref
    )
    child = registry.admit_execution(
        "selected:child", purpose="Child", parent=parent.address, execution=binding("child")
    )
    assert (
        registry.admit_execution(
            "selected:child", purpose="Child", parent=parent.address, execution=binding("child")
        ).address
        == child.address
    )
    with pytest.raises(ValueError, match="retry changed"):
        registry.admit_execution(
            "selected:child", purpose="Changed", parent=parent.address, execution=binding("child")
        )
    dependency = registry.completion_dependency(child.address)
    registry.wait_for_execution(parent.address, dependency)
    registry.activate_execution(
        child.address, source_digest="a" * 64, input_ref=child.execution.input_ref
    )
    restored = TrackRegistry.restore(json.loads(json.dumps(registry.checkpoint())))
    with pytest.raises(ValueError, match="delivered"):
        restored.wake_execution(parent.address, delivered=set())
    with pytest.raises(ValueError, match="issued"):
        restored.wait_for_execution(parent.address, {**dependency, "invocation": "forged"})
    receipt = restored.complete_execution(child.address, kind="return", value=["num", 9])
    assert receipt["dependency"] == dependency
    assert receipt["target"] == parent.address
    assert restored.tracks[parent.address].status == "waiting"
    restored.wake_execution(parent.address, delivered={dependency["id"]})
    assert restored.selectable(parent.address, {dependency["id"]})
    assert restored.complete_execution(child.address, kind="return", value=["num", 9]) == receipt
    with pytest.raises(ValueError, match="cannot change"):
        restored.complete_execution(child.address, kind="return", value=["num", 10])


def test_cancel_propagates_to_descendants_and_no_closed_parent_can_spawn():
    registry = TrackRegistry(max_active=3, max_waiting=8)
    parent = registry.create("Parent", execution=binding("parent"))
    child = registry.create("Child", parent=parent.address, execution=binding("child"))
    grandchild = registry.create(
        "Grandchild", parent=child.address, execution=binding("grandchild")
    )
    registry.change(parent.address, "cancel")
    assert all(
        registry.tracks[item.address].status == "cancelled" for item in (parent, child, grandchild)
    )
    with pytest.raises(ValueError, match="Closed parent"):
        registry.create("Late child", parent=parent.address, execution=binding("late"))
    with pytest.raises(ValueError, match="Cancelled"):
        registry.complete_execution(child.address, kind="return", value=["num", 1])


def test_assign_binding_advances_task_revision_and_revalidates_observed_context():
    registry = TrackRegistry(max_active=1, max_waiting=4)
    worker = registry.create("Idle worker", capabilities=("text", "demo.readAsync"))
    registry.change(worker.address, "wait")
    assigned = registry.change(
        worker.address, "assign", purpose="New invocation", execution=binding("new")
    )
    assert assigned.revision == 1 and not assigned.bootstrap_ready
    with pytest.raises(ValueError, match="bootstrap"):
        registry.change(worker.address, "continue")
    registry.activate_execution(
        worker.address, source_digest="a" * 64, input_ref=binding("new").input_ref
    )
    assert registry.selectable(worker.address, set())
    assign = next(t for t in lifecycle_tools() if t["name"] == "track_assign")
    assert (
        assign["parameters"]["properties"]["execution"]["properties"]["schema"]["const"]
        == "semantic-execution-binding/4"
    )
