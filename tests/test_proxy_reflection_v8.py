"""Independent reflection replay, mutation, derived faults and recovery checks."""

import copy
import json
from pathlib import Path

import pytest

from megatoken.data.protocol_v8 import (
    ProtocolV8Error,
    apply_protocol_to_shared_state,
    create_protocol_state,
    encode_protocol_record,
    validate_protocol_state,
)


def ref(name):
    return ["ref", name]


def base(operation=None):
    state = copy.deepcopy(
        json.loads((Path(__file__).parent / "fixtures/semantic-protocol-v8.json").read_text())[
            "cases"
        ][0]["initialState"]
    )
    state["protocol"] = create_protocol_state()
    step(state, {"kind": "promise.intrinsics"})
    for name in ("target", "handler", "other", "trap"):
        closure = name == "trap"
        step(
            state,
            {
                "kind": "allocate",
                "name": name,
                "objectKind": "closure" if closure else "object",
                "length": None,
                "properties": [],
                "code": name if closure else None,
                "captures": [],
                "functionKind": "ordinary" if closure else None,
                "lexicalReceiver": None,
            },
        )
    step(
        state,
        {
            "kind": "proxy.create",
            "name": "proxy",
            "target": ref("target"),
            "handler": ref("handler"),
        },
    )
    if operation:
        step(
            state,
            {
                "kind": "property.set",
                "object": "handler",
                "key": ["str", operation],
                "receiver": ref("handler"),
                "value": ref("trap"),
                "success": True,
            },
        )
    return state


def step(state, record):
    result = apply_protocol_to_shared_state(state, encode_protocol_record(record))
    state.clear()
    state.update(result)


def call(state, operation, caller=None, arguments=()):
    step(
        state,
        {
            "kind": "proxy.reflect.call",
            "id": "reflection",
            "object": "proxy",
            "operation": operation,
            "caller": caller or "Reflect." + operation,
            "arguments": list(arguments),
            "child": "callback",
            "route": [
                {
                    "object": "proxy",
                    "target": ref("target"),
                    "handler": ref("handler"),
                    "method": ref("trap"),
                }
            ],
        },
    )
    validate_protocol_state(state["protocol"], state["heap"], state)


def returned(state, value, kind="return"):
    step(state, {"kind": kind, "invocation": "callback", "value": value})


def finish(state, value, kind="normal"):
    step(
        state,
        {
            "kind": "proxy.reflect.finish",
            "id": "reflection",
            "completion": {"kind": kind, "value": value},
        },
    )


def test_live_extensibility_after_retained_trap_selection():
    state = base("isExtensible")
    call(state, "isExtensible")
    step(
        state,
        {
            "kind": "property.set",
            "object": "handler",
            "key": ["str", "isExtensible"],
            "receiver": ref("handler"),
            "value": ["undefined"],
            "success": True,
        },
    )
    step(state, {"kind": "property.preventExtensions", "object": "target"})
    validate_protocol_state(state["protocol"], state["heap"], state)
    returned(state, ["num", 0])
    finish(state, ["bool", False])


def test_result_type_fault_and_atomic_false_completion():
    state = base("getPrototypeOf")
    call(state, "getPrototypeOf")
    returned(state, ["num", 7])
    before = copy.deepcopy(state)
    with pytest.raises(ProtocolV8Error):
        finish(state, ["num", 7])
    assert state == before
    step(state, {"kind": "proxy.reflect.fault", "id": "reflection", "error": "error"})
    node = dict(state["heap"])["error"]
    assert dict(node["properties"])["message"] == [
        "str",
        "'getPrototypeOf' on proxy: trap returned neither object nor null",
    ]


def test_prototype_invariant_uses_callback_modified_target():
    state = base("getPrototypeOf")
    call(state, "getPrototypeOf")
    step(
        state,
        {
            "kind": "property.prototype",
            "object": "target",
            "prototype": ref("other"),
            "success": True,
        },
    )
    step(state, {"kind": "property.preventExtensions", "object": "target"})
    returned(state, ref("other"))
    finish(state, ref("other"))


def test_object_vs_reflect_false_mutator():
    for caller in ("Object.setPrototypeOf", "Reflect.setPrototypeOf"):
        state = base("setPrototypeOf")
        call(state, "setPrototypeOf", caller, [["null"]])
        returned(state, ["bool", False])
        if caller.startswith("Object."):
            step(state, {"kind": "proxy.reflect.fault", "id": "reflection", "error": "error"})
        else:
            finish(state, ["bool", False])


def test_prevent_extensions_requires_real_target_mutation():
    for mutate in (False, True):
        state = base("preventExtensions")
        call(state, "preventExtensions", "Object.preventExtensions")
        if mutate:
            step(state, {"kind": "property.preventExtensions", "object": "target"})
        returned(state, ref("target"))
        if mutate:
            finish(state, ref("proxy"))
        else:
            step(state, {"kind": "proxy.reflect.fault", "id": "reflection", "error": "error"})


def test_recovery_rejects_changed_retained_argument_and_skipped_finish():
    state = base("setPrototypeOf")
    call(state, "setPrototypeOf", arguments=[ref("other")])
    forged = copy.deepcopy(state)
    forged["protocol"]["proxyReflection"]["pending"][0]["arguments"] = [["null"]]
    with pytest.raises(ProtocolV8Error):
        validate_protocol_state(forged["protocol"], forged["heap"], forged)
    returned(state, ["bool", True])
    with pytest.raises(ProtocolV8Error):
        step(state, {"kind": "bind", "name": "skip", "value": ["bool", True]})
    finish(state, ["bool", True])


def test_thrown_identity_suppresses_normal_invariant():
    state = base("getPrototypeOf")
    call(state, "getPrototypeOf")
    returned(state, ref("other"), "throw")
    finish(state, ref("other"), "throw")
