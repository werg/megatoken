"""Async calls detach from their caller and resume under independent FIFO state."""

import copy

import pytest

from megatoken.data.native_v8 import _apply_detail
from megatoken.data.promise_scheduler_v8 import create_promise, settle_promise
from megatoken.data.protocol_v8 import (
    ProtocolV8Error,
    _external_machine,
    _internal_machine,
    apply_protocol_record,
    create_protocol_state,
    validate_protocol_state,
)


def fixture():
    machine = {
        "heap": {},
        "locals": {"live": ["num", 1]},
        "captureAliases": {},
        "receiver": ["undefined"],
        "stack": ["root"],
        "frameFunctions": {"root": "run"},
        "pending": None,
        "lastReturn": None,
        "completion": None,
        "protocol": create_protocol_state(),
        "jobs": [],
        "continuations": [],
        "ready": [],
        "activeJob": None,
        "scheduler": {"registration": 0, "ready": 0},
    }
    apply(
        machine,
        {
            "kind": "allocate",
            "name": "helper",
            "objectKind": "closure",
            "length": None,
            "properties": [],
            "code": "helper",
            "captures": [{"binding": "n", "cell": "live"}],
            "functionKind": "ordinary",
            "lexicalReceiver": None,
        },
    )
    apply(machine, {"kind": "async.function", "object": "helper"})
    create_promise(machine, "input")
    settle_promise(machine, "input", "fulfilled", ["num", 0])
    return machine


def apply(machine, record):
    apply_protocol_record(machine, record)
    external = _external_machine(machine)
    assert validate_protocol_state(external["protocol"], external["heap"], external)


def call(machine, name="first"):
    create_promise(machine, name)
    apply(
        machine,
        {
            "kind": "async.call",
            "promise": name,
            "child": name + "-body",
            "callee": ["ref", "helper"],
            "receiver": ["num", 6],
            "arguments": [["num", 2]],
        },
    )


def suspend(machine, name="first", continuation=None):
    apply(
        machine,
        {
            "kind": "async.await",
            "promise": name,
            "invocation": name + "-body",
            "awaited": "input",
            "continuation": continuation or name + "-await",
        },
    )


def resume(name="first", continuation=None):
    return {
        "kind": "async.resume",
        "promise": name,
        "invocation": name + "-body",
        "continuation": continuation or name + "-await",
        "status": "fulfilled",
        "value": ["num", 0],
    }


def finish(name="first", kind="return", value=3):
    return {
        "kind": "async.complete",
        "promise": name,
        "invocation": name + "-body",
        "completion": {"kind": kind, "value": ["num", value]},
    }


def root_return(machine):
    apply(machine, {"kind": "return", "invocation": "root", "value": ["num", 99]})


def detail(machine, kind, **fields):
    external = _external_machine(machine)
    _apply_detail(external, kind, fields, {}, root="root")
    machine.clear()
    machine.update(_internal_machine(external))
    assert validate_protocol_state(external["protocol"], external["heap"], external)


def test_first_await_returns_execution_promise_and_restores_caller_scope():
    machine = fixture()
    call(machine)
    assert machine["receiver"] == ["num", 6]
    assert machine["captureAliases"] == {"n": "live"}
    suspend(machine)
    assert machine["stack"] == ["root"]
    assert machine["protocol"]["callFrames"] == []
    assert machine["receiver"] == ["undefined"]
    assert machine["captureAliases"] == {}
    assert machine["lastReturn"] == {"kind": "return", "value": ["ref", "first"]}
    assert machine["ready"] == ["first-await"]


def test_async_returned_promise_remains_pending_until_fifo_forwarding_job():
    machine = fixture()
    create_promise(machine, "source")
    settle_promise(machine, "source", "fulfilled", ["num", 23])
    call(machine)
    apply(
        machine,
        {
            "kind": "async.complete",
            "promise": "first",
            "invocation": "first-body",
            "completion": {"kind": "return", "value": ["ref", "source"]},
        },
    )
    assert machine["heap"]["first"]["state"] == "pending"
    apply(machine, {"kind": "promise.adopt.enqueue", "promise": "first", "target": "source", "job": "adopt"})
    machine["stack"] = []
    apply(machine, {"kind": "promise.adopt.enter", "promise": "first", "target": "source", "job": "adopt", "forwarding": "forward"})
    assert machine["ready"] == ["forward"]
    apply(
        machine,
        {"kind": "promise.adopt.forward", "promise": "first", "target": "source", "job": "forward", "status": "fulfilled", "value": ["num", 23]},
    )
    assert machine["heap"]["first"]["state"] == "fulfilled"
    assert machine["heap"]["first"]["result"] == ["num", 23]


def test_detached_resume_keeps_live_cells_and_completion_does_not_replace_root():
    machine = fixture()
    call(machine)
    suspend(machine)
    apply(machine, {"kind": "bind", "name": "live", "value": ["num", 7]})
    root_return(machine)
    apply(machine, resume())
    assert machine["locals"][machine["captureAliases"]["n"]] == ["num", 7]
    apply(machine, finish(value=7))
    assert machine["pending"] == {"kind": "return", "value": ["num", 99]}
    assert machine["heap"]["first"]["result"] == ["num", 7]
    assert machine["stack"] == [] and machine["captureAliases"] == {}
    assert machine["completion"] is None


@pytest.mark.parametrize("kind", ["return", "throw"])
def test_immediate_async_completion_returns_its_own_settled_promise(kind):
    machine = fixture()
    call(machine)
    apply(machine, finish(kind=kind, value=4))
    assert machine["heap"]["first"]["state"] == ("fulfilled" if kind == "return" else "rejected")
    assert machine["stack"] == ["root"]
    assert machine["lastReturn"] == {"kind": "return", "value": ["ref", "first"]}
    assert machine["pending"] is None


def test_siblings_and_repeated_awaits_use_one_fifo_without_duplicate_execution():
    machine = fixture()
    for name in ("first", "second"):
        call(machine, name)
        suspend(machine, name)
    root_return(machine)
    before = copy.deepcopy(machine)
    with pytest.raises(ProtocolV8Error, match="FIFO"):
        apply_protocol_record(machine, resume("second"))
    assert machine == before
    apply(machine, resume())
    suspend(machine, continuation="first-again")
    assert machine["ready"] == ["second-await", "first-again"]
    apply(machine, resume("second"))
    apply(machine, finish("second", kind="throw", value=8))
    apply(machine, resume(continuation="first-again"))
    apply(machine, finish())
    assert machine["heap"]["first"]["state"] == "fulfilled"
    assert machine["heap"]["second"]["state"] == "rejected"


def test_forged_resume_and_generic_completion_cannot_mutate_async_state():
    machine = fixture()
    call(machine)
    before = copy.deepcopy(machine)
    for record in [
        {"kind": "return", "invocation": "first-body", "value": ["num", 1]},
        {
            "kind": "async.call",
            "promise": "first",
            "child": "other",
            "callee": ["ref", "helper"],
            "receiver": ["undefined"],
            "arguments": [],
        },
    ]:
        with pytest.raises(ProtocolV8Error):
            apply_protocol_record(machine, record)
        assert machine == before
    suspend(machine)
    before = copy.deepcopy(machine)
    with pytest.raises(ProtocolV8Error, match="FIFO"):
        apply_protocol_record(machine, resume())
    assert machine == before
    root_return(machine)
    before = copy.deepcopy(machine)
    for field, value in [
        ("promise", "input"),
        ("invocation", "root"),
        ("value", ["num", 99]),
        ("status", "rejected"),
    ]:
        with pytest.raises(ProtocolV8Error):
            apply_protocol_record(machine, {**resume(), field: value})
        assert machine == before


def test_restore_refuses_changed_receiver_captures_or_execution_settlement():
    machine = fixture()
    call(machine)
    suspend(machine)
    snapshot = _external_machine(machine)
    for field, value in [("receiver", ["num", 1]), ("captureAliases", [])]:
        forged = copy.deepcopy(snapshot)
        forged["continuations"][0][1][field] = value
        with pytest.raises(ProtocolV8Error, match="receiver or captured"):
            validate_protocol_state(forged["protocol"], forged["heap"], forged)
    forged = copy.deepcopy(snapshot)
    promise = dict(forged["heap"])["first"]
    promise["state"], promise["result"] = "fulfilled", ["num", 3]
    with pytest.raises(ProtocolV8Error, match="unfinished async"):
        validate_protocol_state(forged["protocol"], forged["heap"], forged)


def test_async_throw_keeps_an_owned_promise_as_opaque_rejection_after_restore():
    machine = fixture()
    call(machine)
    apply(
        machine,
        {
            "kind": "async.complete",
            "promise": "first",
            "invocation": "first-body",
            "completion": {"kind": "throw", "value": ["ref", "input"]},
        },
    )
    machine = _internal_machine(copy.deepcopy(_external_machine(machine)))
    external = _external_machine(machine)
    assert validate_protocol_state(external["protocol"], external["heap"], external)
    assert machine["heap"]["first"]["state"] == "rejected"
    assert machine["heap"]["first"]["result"] == ["ref", "input"]
    assert machine["heap"]["first"].get("adoption") is None
    assert machine["jobs"] == []
    before = copy.deepcopy(machine)
    with pytest.raises(ProtocolV8Error):
        apply_protocol_record(
            machine,
            {"kind": "promise.adopt.enqueue", "promise": "first", "target": "input", "job": "invented-adoption"},
        )
    assert machine == before
    assert machine["heap"]["input"]["result"] == ["num", 0]


def test_async_forwarding_identity_cannot_collide_with_jobs_or_other_adopters():
    machine = fixture()
    for name in ("first", "second"):
        call(machine, name)
        apply(
            machine,
            {
                "kind": "async.complete",
                "promise": name,
                "invocation": name + "-body",
                "completion": {"kind": "return", "value": ["ref", "input"]},
            },
        )
        apply(machine, {"kind": "promise.adopt.enqueue", "promise": name, "target": "input", "job": name + "-adopt"})
        if name == "first":
            detail(machine, "job", action="enqueue", job="ordinary", owner="root")
    root_return(machine)
    assert machine["ready"] == ["first-adopt", "ordinary", "second-adopt"]
    for forwarding in ("ordinary", "second-adopt", "first-adopt"):
        before = copy.deepcopy(machine)
        with pytest.raises(ProtocolV8Error):
            apply_protocol_record(
                machine,
                {"kind": "promise.adopt.enter", "promise": "first", "target": "input", "job": "first-adopt", "forwarding": forwarding},
            )
        assert machine == before
    apply(machine, {"kind": "promise.adopt.enter", "promise": "first", "target": "input", "job": "first-adopt", "forwarding": "first-forward"})
    machine = _internal_machine(copy.deepcopy(_external_machine(machine)))
    detail(machine, "job", action="dispatch", job="ordinary", owner="root")
    detail(machine, "job", action="complete", job="ordinary", owner="root")
    before = copy.deepcopy(machine)
    with pytest.raises(ProtocolV8Error):
        apply_protocol_record(
            machine,
            {"kind": "promise.adopt.enter", "promise": "second", "target": "input", "job": "second-adopt", "forwarding": "first-forward"},
        )
    assert machine == before
    apply(machine, {"kind": "promise.adopt.enter", "promise": "second", "target": "input", "job": "second-adopt", "forwarding": "second-forward"})
    machine = _internal_machine(copy.deepcopy(_external_machine(machine)))
    external = _external_machine(machine)
    assert validate_protocol_state(external["protocol"], external["heap"], external)
    assert machine["ready"] == ["first-forward", "second-forward"]
    assert machine["heap"]["first"]["adoption"]["forwarding"] == "first-forward"
    assert machine["heap"]["second"]["adoption"]["forwarding"] == "second-forward"
