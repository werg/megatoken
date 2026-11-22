"""Async calls detach from their caller and resume under independent FIFO state."""

import copy

import pytest

from megatoken.data.native_v8 import NativeV8Error, _apply_detail
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


def detail(machine, kind, fields):
    external = _external_machine(machine)
    _apply_detail(external, kind, fields, {}, root="root")
    machine.clear()
    machine.update(_internal_machine(external))
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


def promise_return_fixture(*, settled=True):
    machine = fixture()
    create_promise(machine, "target")
    if settled:
        settle_promise(machine, "target", "fulfilled", ["num", 1])
    call(machine)
    apply(
        machine,
        {
            "kind": "async.complete",
            "promise": "first",
            "invocation": "first-body",
            "completion": {"kind": "return", "value": ["ref", "target"]},
        },
    )
    return machine


def enqueue_resolution(machine, job="j1", settlement="j2", target="target"):
    apply(
        machine,
        {
            "kind": "async.resolve.enqueue",
            "promise": "first",
            "invocation": "first-body",
            "target": target,
            "job": job,
            "settlementJob": settlement,
        },
    )


def subscribe_resolution(machine, job="j1", settlement="j2"):
    apply(
        machine,
        {
            "kind": "async.resolve.subscribe",
            "promise": "first",
            "invocation": "first-body",
            "target": "target",
            "job": job,
            "settlementJob": settlement,
        },
    )


def settle_resolution(machine, job="j2", status="fulfilled", value=None):
    apply(
        machine,
        {
            "kind": "async.resolve.settle",
            "promise": "first",
            "invocation": "first-body",
            "target": "target",
            "job": job,
            "status": status,
            "value": value or ["num", 1],
        },
    )


def dispatch_complete(machine, job):
    detail(machine, "job", {"action": "dispatch", "job": job, "owner": "root"})
    detail(machine, "job", {"action": "complete", "job": job, "owner": "root"})


def test_owned_async_promise_return_requires_three_exact_fifo_turns():
    machine = promise_return_fixture()
    assert machine["heap"]["first"]["state"] == "pending"
    enqueue_resolution(machine)
    detail(
        machine, "promise", {"action": "react", "promise": "first", "job": "j3", "reaction": "then"}
    )
    detail(
        machine,
        "promise",
        {"action": "react", "promise": "target", "job": "j4", "reaction": "then"},
    )
    detail(machine, "job", {"action": "enqueue", "job": "j5", "owner": "root"})
    detail(machine, "await", {"invocation": "root", "promise": "first", "continuation": "c1"})
    assert machine["ready"] == ["j1", "j4", "j5"]
    subscribe_resolution(machine)
    assert machine["heap"]["first"]["state"] == "pending"
    assert machine["ready"] == ["j4", "j5", "j2"]
    dispatch_complete(machine, "j4")
    dispatch_complete(machine, "j5")
    assert machine["ready"] == ["j2"]
    settle_resolution(machine)
    assert machine["heap"]["first"]["state"] == "fulfilled"
    assert machine["heap"]["first"]["result"] == ["num", 1]
    assert machine["ready"] == ["j3", "c1"]
    validate_protocol_state(machine["protocol"], machine["heap"], _external_machine(machine))


def test_pending_target_observer_and_settlement_preserve_registration_fifo():
    machine = promise_return_fixture(settled=False)
    enqueue_resolution(machine)
    detail(
        machine,
        "promise",
        {"action": "react", "promise": "target", "job": "j3", "reaction": "then"},
    )
    detail(
        machine, "promise", {"action": "react", "promise": "first", "job": "j4", "reaction": "then"}
    )
    detail(machine, "await", {"invocation": "root", "promise": "first", "continuation": "c1"})
    subscribe_resolution(machine)
    assert dict(machine["jobs"])["j2"]["status"] == "waiting"
    assert machine["ready"] == []
    detail(machine, "promise", {"action": "resolve", "promise": "target", "value": ["num", 7]})
    assert machine["ready"] == ["j3", "j2"]
    dispatch_complete(machine, "j3")
    settle_resolution(machine, value=["num", 7])
    assert machine["heap"]["first"]["result"] == ["num", 7]
    assert machine["ready"] == ["j4", "c1"]


def test_resolution_jobs_reject_retargeting_skips_generic_dispatch_and_self_resolution():
    machine = fixture()
    call(machine)
    before = copy.deepcopy(machine)
    with pytest.raises(ProtocolV8Error, match="distinct owned target|cycle"):
        apply_protocol_record(
            machine,
            {
                "kind": "async.complete",
                "promise": "first",
                "invocation": "first-body",
                "completion": {"kind": "return", "value": ["ref", "first"]},
            },
        )
    assert machine == before

    machine = promise_return_fixture()
    before = copy.deepcopy(machine)
    with pytest.raises(ProtocolV8Error, match="exact completed async return"):
        enqueue_resolution(machine, "bad1", "bad2", "input")
    assert machine == before
    enqueue_resolution(machine)
    before = copy.deepcopy(machine)
    with pytest.raises(ProtocolV8Error, match="subscribed FIFO observer"):
        settle_resolution(machine)
    assert machine == before
    detail(machine, "await", {"invocation": "root", "promise": "first", "continuation": "c1"})
    before = copy.deepcopy(machine)
    with pytest.raises(NativeV8Error, match="explicit FIFO protocol"):
        detail(machine, "job", {"action": "dispatch", "job": "j1", "owner": "root"})
    assert machine == before
    subscribe_resolution(machine)
    before = copy.deepcopy(machine)
    for record in [
        {
            "kind": "async.resolve.settle",
            "promise": "first",
            "invocation": "first-body",
            "target": "input",
            "job": "j2",
            "status": "fulfilled",
            "value": ["num", 1],
        },
        {
            "kind": "async.resolve.settle",
            "promise": "first",
            "invocation": "first-body",
            "target": "target",
            "job": "j2",
            "status": "rejected",
            "value": ["num", 1],
        },
        {
            "kind": "async.resolve.settle",
            "promise": "first",
            "invocation": "wrong-body",
            "target": "target",
            "job": "j2",
            "status": "fulfilled",
            "value": ["num", 99],
        },
    ]:
        with pytest.raises(ProtocolV8Error):
            apply_protocol_record(machine, record)
        assert machine == before
