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
            "descriptorObject": "descriptor" if operation == "defineProperty" else None,
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


def allocate_data(state, name, fields, array=False):
    step(
        state,
        {
            "kind": "allocate",
            "name": name,
            "objectKind": "array" if array else "object",
            "length": len(fields) if array else None,
            "properties": [{"key": ["str", str(key)], "value": value} for key, value in fields],
            "code": None,
            "captures": [],
            "functionKind": None,
            "lexicalReceiver": None,
        },
    )


def test_descriptor_result_is_fresh_and_completed_with_defaults():
    state = base("getOwnPropertyDescriptor")
    call(state, "getOwnPropertyDescriptor", arguments=[["str", "x"]])
    allocate_data(state, "raw", [("value", ["num", 7]), ("configurable", ["bool", True])])
    returned(state, ref("raw"))
    before = copy.deepcopy(state)
    with pytest.raises(ProtocolV8Error):
        finish(state, ref("raw"))
    assert state == before
    finish(state, ref("output"))
    properties = dict(dict(state["heap"])["output"]["properties"])
    assert properties == {
        "value": ["num", 7],
        "writable": ["bool", False],
        "enumerable": ["bool", False],
        "configurable": ["bool", True],
    }


def test_ownkeys_preserves_trap_order_and_requires_fixed_keys():
    state = base("ownKeys")
    step(
        state,
        {
            "kind": "property.define",
            "object": "target",
            "key": ["str", "x"],
            "descriptor": {
                "kind": "data",
                "value": ["num", 1],
                "writable": False,
                "enumerable": False,
                "configurable": False,
            },
            "success": True,
        },
    )
    call(state, "ownKeys")
    allocate_data(state, "raw", [(0, ["str", "extra"]), (1, ["str", "x"])], array=True)
    returned(state, ref("raw"))
    finish(state, ref("output"))
    assert dict(dict(state["heap"])["output"]["properties"]) == {
        "length": ["num", 2],
        "0": ["str", "extra"],
        "1": ["str", "x"],
    }


@pytest.mark.parametrize(
    "fields,message",
    [
        (
            [(0, ["str", "x"]), (1, ["str", "x"])],
            "'ownKeys' on proxy: trap returned duplicate entries",
        ),
        ([(0, ["num", 7])], "7 is not a valid property name"),
    ],
)
def test_ownkeys_derived_list_faults(fields, message):
    state = base("ownKeys")
    call(state, "ownKeys")
    allocate_data(state, "raw", fields, array=True)
    returned(state, ref("raw"))
    step(state, {"kind": "proxy.reflect.fault", "id": "reflection", "error": "error"})
    assert dict(dict(state["heap"])["error"]["properties"])["message"] == ["str", message]


def test_descriptor_cannot_hide_nonconfigurable_target():
    state = base("getOwnPropertyDescriptor")
    step(
        state,
        {
            "kind": "property.define",
            "object": "target",
            "key": ["str", "x"],
            "descriptor": {
                "kind": "data",
                "value": ["num", 1],
                "writable": False,
                "enumerable": False,
                "configurable": False,
            },
            "success": True,
        },
    )
    call(state, "getOwnPropertyDescriptor", arguments=[["str", "x"]])
    returned(state, ["undefined"])
    step(state, {"kind": "proxy.reflect.fault", "id": "reflection", "error": "error"})
    assert "non-configurable" in dict(dict(state["heap"])["error"]["properties"])["message"][1]


def test_define_uses_snapshot_and_fresh_mutable_callback_descriptor():
    state = base("defineProperty")
    step(
        state,
        {
            "kind": "property.define",
            "object": "target",
            "key": ["str", "x"],
            "descriptor": {
                "kind": "data",
                "value": ["num", 1],
                "writable": True,
                "enumerable": False,
                "configurable": False,
            },
            "success": True,
        },
    )
    allocate_data(state, "input", [("writable", ["bool", False])])
    call(state, "defineProperty", arguments=[["str", "x"], ref("input")])
    assert dict(dict(state["heap"])["descriptor"]["properties"]) == {"writable": ["bool", False]}
    step(
        state,
        {
            "kind": "property.set",
            "object": "descriptor",
            "key": ["str", "writable"],
            "receiver": ref("descriptor"),
            "value": ["bool", True],
            "success": True,
        },
    )
    returned(state, ["bool", True])
    step(state, {"kind": "proxy.reflect.fault", "id": "reflection", "error": "error"})
    assert (
        "cannot be non-writable" in dict(dict(state["heap"])["error"]["properties"])["message"][1]
    )


def test_define_generic_partial_fields_preserve_existing_value():
    state = base()
    step(
        state,
        {
            "kind": "property.set",
            "object": "target",
            "key": ["str", "x"],
            "receiver": ref("target"),
            "value": ["num", 7],
            "success": True,
        },
    )
    allocate_data(state, "input", [("enumerable", ["bool", False])])
    step(
        state,
        {
            "kind": "proxy.reflect.direct",
            "object": "proxy",
            "operation": "defineProperty",
            "caller": "Reflect.defineProperty",
            "arguments": [["str", "x"], ref("input")],
            "result": ["bool", True],
            "route": [
                {
                    "object": "proxy",
                    "target": ref("target"),
                    "handler": ref("handler"),
                    "method": ["undefined"],
                }
            ],
        },
    )
    target = dict(state["heap"])["target"]
    assert dict(target["properties"])["x"] == ["num", 7]
    assert dict(target["descriptors"])["x"] == {
        "kind": "data",
        "writable": True,
        "enumerable": False,
        "configurable": True,
    }


def test_define_false_reflect_result_does_not_change_target():
    state = base("defineProperty")
    allocate_data(state, "input", [("value", ["num", 7])])
    call(state, "defineProperty", arguments=[["str", "x"], ref("input")])
    returned(state, ["bool", False])
    finish(state, ["bool", False])
    assert "x" not in dict(dict(state["heap"])["target"]["properties"])


def test_define_snapshot_is_bound_to_active_and_returned_callback_scope():
    state = base("defineProperty")
    allocate_data(state, "input", [("writable", ["bool", False])])
    call(state, "defineProperty", arguments=[["str", "x"], ref("input")])
    retained = {"kind": "proxy.defineProperty", "definition": {"writable": False}}
    assert state["protocol"]["callFrames"][-1]["proxyReflectionScope"] == retained

    forged_row = copy.deepcopy(state)
    forged_row["protocol"]["proxyReflection"]["pending"][-1]["definition"]["writable"] = True
    with pytest.raises(ProtocolV8Error, match="active callback scope"):
        validate_protocol_state(forged_row["protocol"], forged_row["heap"], forged_row)

    forged_scope = copy.deepcopy(state)
    forged_scope["protocol"]["callFrames"][-1]["proxyReflectionScope"]["definition"]["writable"] = (
        True
    )
    with pytest.raises(ProtocolV8Error, match="active callback scope"):
        validate_protocol_state(forged_scope["protocol"], forged_scope["heap"], forged_scope)

    returned(state, ["bool", True])
    validate_protocol_state(state["protocol"], state["heap"], state)
    assert state["lastReturn"]["proxyReflectionScope"] == retained

    forged_return = copy.deepcopy(state)
    forged_return["lastReturn"]["proxyReflectionScope"]["definition"]["writable"] = True
    with pytest.raises(ProtocolV8Error, match="returned callback scope"):
        validate_protocol_state(forged_return["protocol"], forged_return["heap"], forged_return)

    finish(state, ["bool", True])
    assert "proxyReflectionScope" not in state["lastReturn"]


def test_define_fault_consumes_snapshot_receipt_after_mutable_copy():
    state = base("defineProperty")
    step(
        state,
        {
            "kind": "property.define",
            "object": "target",
            "key": ["str", "x"],
            "descriptor": {
                "kind": "data",
                "value": ["num", 1],
                "writable": True,
                "enumerable": False,
                "configurable": False,
            },
            "success": True,
        },
    )
    allocate_data(state, "input", [("writable", ["bool", False])])
    call(state, "defineProperty", arguments=[["str", "x"], ref("input")])
    step(
        state,
        {
            "kind": "property.set",
            "object": "descriptor",
            "key": ["str", "writable"],
            "receiver": ref("descriptor"),
            "value": ["bool", True],
            "success": True,
        },
    )
    returned(state, ["bool", True])
    validate_protocol_state(state["protocol"], state["heap"], state)
    assert state["lastReturn"]["proxyReflectionScope"]["definition"] == {"writable": False}
    step(state, {"kind": "proxy.reflect.fault", "id": "reflection", "error": "error"})
    assert "proxyReflectionScope" not in state["lastReturn"]
