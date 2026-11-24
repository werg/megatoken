"""Independent Proxy reflection, retained callback lifecycle and live invariants."""

from __future__ import annotations

import copy

from .proxy_reflection_descriptors_v8 import (
    _compatible,
    allocate_partial_descriptor,
    complete_descriptor,
    definition_fault,
    describe_result,
    materialize_result,
    read_partial_descriptor,
)
from .proxy_v8 import _derive_route, _eq, _id, _retained_route, _truthy

SCHEMA = "semantic-proxy-reflection/1"
OPERATIONS = (
    "getPrototypeOf",
    "setPrototypeOf",
    "isExtensible",
    "preventExtensions",
    "getOwnPropertyDescriptor",
    "ownKeys",
    "defineProperty",
)
FIELDS = {
    "proxy.reflect.call": {
        "kind",
        "id",
        "object",
        "operation",
        "caller",
        "arguments",
        "child",
        "route",
        "descriptorObject",
    },
    "proxy.reflect.direct": {
        "kind",
        "object",
        "operation",
        "caller",
        "arguments",
        "result",
        "route",
    },
    "proxy.reflect.finish": {"kind", "id", "completion"},
    "proxy.reflect.fault": {"kind", "id", "error"},
}
CALL_FIELDS = {
    "id",
    "object",
    "operation",
    "caller",
    "arguments",
    "parent",
    "child",
    "owner",
    "target",
    "handler",
    "callee",
    "route",
    "definition",
    "descriptorObject",
}


def _p():
    from . import protocol_v8

    return protocol_v8


def _require(condition, message):
    if not condition:
        _p()._fail("Proxy reflection: " + message)


def validate_record(record):
    if type(record) is not dict or record.get("kind") not in FIELDS:
        return False
    p = _p()
    p._exact(record, FIELDS[record["kind"]], "Proxy reflection record")
    if record["kind"] == "proxy.reflect.finish":
        _id(record["id"])
        p._exact(record["completion"], {"kind", "value"}, "Proxy reflection completion")
        _require(record["completion"]["kind"] in ("normal", "throw"), "completion kind")
        p.validate_protocol_value(record["completion"]["value"])
        return True
    if record["kind"] == "proxy.reflect.fault":
        _id(record["id"])
        _id(record["error"])
        return True
    _id(record["object"])
    operation = record["operation"]
    _require(
        operation in OPERATIONS
        and record["caller"]
        in (
            ("Reflect.ownKeys",)
            if operation == "ownKeys"
            else ("Object." + operation, "Reflect." + operation)
        ),
        "operation/caller",
    )
    _require(
        type(record["arguments"]) is list
        and len(record["arguments"])
        == (
            2
            if operation == "defineProperty"
            else 1
            if operation in ("setPrototypeOf", "getOwnPropertyDescriptor")
            else 0
        ),
        "argument count",
    )
    for value in record["arguments"]:
        p.validate_protocol_value(value)
    _require(type(record["route"]) is list, "route array")
    if record["kind"] == "proxy.reflect.call":
        _id(record["id"])
        _id(record["child"])
        if operation == "defineProperty":
            _id(record["descriptorObject"])
        else:
            _require(record["descriptorObject"] is None, "only definition has descriptor object")
    else:
        p.validate_protocol_value(record["result"])
    return True


def _state(machine):
    if machine["protocol"].get("proxyReflection") is None:
        machine["protocol"]["proxyReflection"] = {"schema": SCHEMA, "pending": []}
    return machine["protocol"]["proxyReflection"]


def _ordinary(machine, reference):
    _p()._known(machine, reference)
    node = machine["heap"].get(reference[1]) if reference[0] == "ref" else None
    _require(node and node["kind"] == "object", "target requires ordinary internal methods")
    return node


def _prototype(machine, node):
    from .promise_object_v8 import intrinsic_id

    prototype = node.get("prototype", "ordinary")
    if prototype == "ordinary":
        value = ["ref", intrinsic_id("Object.prototype")]
    elif prototype is None:
        value = ["null"]
    else:
        value = prototype
    _require(type(value) is list, "prototype representation")
    _p()._known(machine, value)
    return value


def _arguments(machine, row):
    for value in row["arguments"]:
        _p()._known(machine, value)
    if row["operation"] in ("getOwnPropertyDescriptor", "defineProperty"):
        _p().validate_protocol_key(row["arguments"][0])
    if row["operation"] == "setPrototypeOf":
        _require(row["arguments"][0][0] in ("ref", "null"), "prototype must be object or null")


def _fault(machine, row, value):
    if row["operation"] == "defineProperty":
        return definition_fault(machine, row, value)
    if row["operation"] in ("getOwnPropertyDescriptor", "ownKeys"):
        return describe_result(machine, row, value).get("fault")
    target = _ordinary(machine, row["target"])
    extensible, truth = target.get("extensible", True), _truthy(value)
    operation = row["operation"]
    if operation == "getPrototypeOf":
        if value[0] not in ("ref", "null"):
            return "'getPrototypeOf' on proxy: trap returned neither object nor null"
        if not extensible and not _eq(value, _prototype(machine, target)):
            return "'getPrototypeOf' on proxy: proxy target is non-extensible but the trap did not return its actual prototype"
    elif operation == "setPrototypeOf":
        if truth and not extensible and not _eq(row["arguments"][0], _prototype(machine, target)):
            return "'setPrototypeOf' on proxy: trap returned truish for setting a new prototype on the non-extensible proxy target"
        if not truth and row["caller"] == "Object.setPrototypeOf":
            return "'setPrototypeOf' on proxy: trap returned falsish for property 'undefined'"
    elif operation == "isExtensible":
        if truth != extensible:
            return (
                "'isExtensible' on proxy: trap result does not reflect extensibility of proxy target (which is '"
                + str(extensible).lower()
                + "')"
            )
    elif truth and extensible:
        return (
            "'preventExtensions' on proxy: trap returned truish but the proxy target is extensible"
        )
    elif not truth and row["caller"] == "Object.preventExtensions":
        return "'preventExtensions' on proxy: trap returned falsish"
    return None


def _normal(row, value):
    if row["operation"] == "getPrototypeOf":
        return copy.deepcopy(value)
    if row["operation"] == "isExtensible" or row["caller"].startswith("Reflect."):
        return ["bool", _truthy(value)]
    return ["ref", row["object"]]


def _direct(machine, row, target):
    if row["operation"] == "defineProperty":
        key = _p()._key(row["arguments"][0])
        current = _p()._descriptor(target, key)
        completed = complete_descriptor(row["definition"], current)
        success = _compatible(current, completed, target.get("extensible", True))
        _require(
            success or row["caller"] == "Reflect.defineProperty",
            "Object.defineProperty failure needs ordinary fault",
        )
        if success:
            _p()._install(target, key, completed)
        return (
            ["bool", success]
            if row["caller"] == "Reflect.defineProperty"
            else ["ref", row["object"]]
        )
    if row["operation"] in ("getOwnPropertyDescriptor", "ownKeys"):
        return materialize_result(machine, row, None, row["result"], True)
    operation = row["operation"]
    if operation == "getPrototypeOf":
        return _prototype(machine, target)
    if operation == "isExtensible":
        return ["bool", target.get("extensible", True)]
    if operation == "preventExtensions":
        target["extensible"] = False
        return ["bool", True] if row["caller"].startswith("Reflect.") else ["ref", row["object"]]
    prototype = row["arguments"][0]
    success = _eq(_prototype(machine, target), prototype) or target.get("extensible", True)
    if success and prototype[0] == "ref":
        current, seen = prototype, set()
        while current[0] == "ref":
            if current[1] == row["target"][1]:
                success = False
                break
            _require(current[1] not in seen, "existing prototype cycle")
            seen.add(current[1])
            current = _prototype(machine, _ordinary(machine, current))
    _require(
        success or row["caller"] == "Reflect.setPrototypeOf",
        "Object.setPrototypeOf false result requires ordinary fault boundary",
    )
    if success:
        target["prototype"] = None if prototype[0] == "null" else copy.deepcopy(prototype)
    return (
        ["bool", success] if row["caller"] == "Reflect.setPrototypeOf" else ["ref", row["object"]]
    )


def assert_next(machine, record):
    state = machine.get("protocol", {}).get("proxyReflection")
    row = state["pending"][-1] if state and state["pending"] else None
    if row and row["child"] not in machine["stack"]:
        _require(
            record.get("kind") in ("proxy.reflect.finish", "proxy.reflect.fault")
            and record.get("id") == row["id"],
            "pending reflection requires completion",
        )
    return True


def apply_record(machine, record):
    kind = record.get("kind")
    if kind not in FIELDS:
        return False
    p = _p()
    if kind in ("proxy.reflect.finish", "proxy.reflect.fault"):
        pending = _state(machine)["pending"]
        row = pending[-1] if pending else None
        completion = machine.get("lastReturn")
        _require(
            row
            and row["id"] == record["id"]
            and machine["stack"][-1] == row["parent"]
            and row["child"] not in machine["stack"]
            and completion
            and completion["kind"] in ("return", "throw"),
            "callback completion",
        )
        _retained_route(machine, row)
        _check_callback_scope(completion.get("proxyReflectionScope"), row, "completed callback scope")
        fault = (
            _fault(machine, row, completion["value"]) if completion["kind"] == "return" else None
        )
        if kind == "proxy.reflect.fault":
            _require(completion["kind"] == "return" and fault is not None, "fault is not derived")
            from .intrinsic_error_v8 import allocate_intrinsic_error_v8

            value = allocate_intrinsic_error_v8(machine, record["error"], "TypeError", fault)
            machine["lastReturn"] = {"kind": "throw", "value": value}
        else:
            _require(fault is None, "derived fault required")
            expected = {
                "kind": "throw" if completion["kind"] == "throw" else "normal",
                "value": completion["value"]
                if completion["kind"] == "throw"
                else materialize_result(
                    machine, row, completion["value"], record["completion"]["value"]
                )
                if row["operation"] in ("getOwnPropertyDescriptor", "ownKeys")
                else _normal(row, completion["value"]),
            }
            _require(_eq(record["completion"], expected), "false completion")
            completion.pop("proxyReflectionScope", None)
        pending.pop()
        return True
    _require(machine["heap"].get(record["object"], {}).get("kind") == "proxy", "object identity")
    _arguments(machine, record)
    definition = (
        read_partial_descriptor(machine, record["arguments"][1])
        if record["operation"] == "defineProperty"
        else None
    )
    route = _derive_route(machine, record["object"], record["operation"])
    _require(_eq(record["route"], route["route"]), "route differs from live lookup")
    target = _ordinary(machine, route["target"])
    if kind == "proxy.reflect.direct":
        _require(route["selected"][0] in ("undefined", "null"), "explicit trap call required")
        value = _direct(
            machine, {**record, "target": route["target"], "definition": definition}, target
        )
        _require(_eq(record["result"], value), "false direct result")
        return True
    pending = _state(machine)["pending"]
    _require(
        len(pending) < 64 and all(row["id"] != record["id"] for row in pending),
        "call identity/bound",
    )
    method = route["selected"]
    _require(
        method[0] == "ref" and p._node(machine, method[1])["kind"] == "closure",
        "selected trap must be callable",
    )
    row = {
        key: copy.deepcopy(record[key])
        for key in ("id", "object", "operation", "caller", "arguments", "child", "route")
    }
    row.update(
        parent=machine["stack"][-1],
        owner=route["owner"],
        target=copy.deepcopy(route["target"]),
        handler=copy.deepcopy(route["route"][-1]["handler"]),
        callee=copy.deepcopy(method),
        definition=copy.deepcopy(definition),
        descriptorObject=record["descriptorObject"],
    )
    from .construction_v8 import _call

    if row["operation"] == "defineProperty":
        allocate_partial_descriptor(machine, row["descriptorObject"], definition)
    _call(machine, row["child"], row["callee"], row["handler"], _callback_arguments(row))
    if row["operation"] == "defineProperty":
        machine["protocol"]["callFrames"][-1]["proxyReflectionScope"] = _callback_scope(row)
    pending.append(row)
    return True


def _callback_arguments(row):
    return (
        [row["target"], row["arguments"][0], ["ref", row["descriptorObject"]]]
        if row["operation"] == "defineProperty"
        else [row["target"], *row["arguments"]]
    )


def _callback_scope(row):
    return (
        {"kind": "proxy.defineProperty", "definition": copy.deepcopy(row["definition"])}
        if row["operation"] == "defineProperty"
        else None
    )


def _check_callback_scope(receipt, row, label):
    expected = _callback_scope(row)
    if expected is None:
        _require(receipt is None, label + " on non-definition callback")
    else:
        _require(receipt == expected, label + " differs from retained ToPropertyDescriptor")


def validate_state(machine):
    snapshot = machine.get("protocol", {}).get("proxyReflection")
    if snapshot is None:
        _require(
            not any("proxyReflectionScope" in frame for frame in machine["protocol"].get("callFrames", []))
            and "proxyReflectionScope" not in (machine.get("lastReturn") or {}),
            "orphaned Proxy reflection callback scope",
        )
        return True
    _p()._exact(snapshot, {"schema", "pending"}, "Proxy reflection state")
    _require(
        snapshot["schema"] == SCHEMA
        and type(snapshot["pending"]) is list
        and len(snapshot["pending"]) <= 64,
        "schema/bound",
    )
    seen, prior_depth = set(), -1
    for row in snapshot["pending"]:
        _p()._exact(row, CALL_FIELDS, "Proxy reflection retained call")
        validate_record(
            {
                "kind": "proxy.reflect.call",
                **{
                    key: row[key]
                    for key in (
                        "id",
                        "object",
                        "operation",
                        "caller",
                        "arguments",
                        "child",
                        "route",
                        "descriptorObject",
                    )
                },
            }
        )
        _id(row["parent"])
        _id(row["owner"])
        _require(row["id"] not in seen, "duplicate call")
        seen.add(row["id"])
        _arguments(machine, row)
        _retained_route(machine, row)
        _require(
            _eq(row["target"], row["route"][-1]["target"])
            and _eq(row["handler"], row["route"][-1]["handler"])
            and _eq(row["callee"], row["route"][-1]["method"]),
            "retained identities",
        )
        if row["operation"] == "defineProperty":
            fields = row["definition"]
            _require(
                type(fields) is dict
                and set(fields)
                <= {"value", "writable", "get", "set", "enumerable", "configurable"},
                "retained descriptor fields",
            )
            for key, value in fields.items():
                if key in ("writable", "enumerable", "configurable"):
                    _require(type(value) is bool, "retained descriptor flag")
                else:
                    _p()._known(machine, value)
            _require(
                not ({"get", "set"} & set(fields) and {"value", "writable"} & set(fields)),
                "retained descriptor conflict",
            )
            _p()._known(machine, ["ref", row["descriptorObject"]])
        else:
            _require(
                row["definition"] is None and row["descriptorObject"] is None,
                "unexpected retained descriptor",
            )
        callee = _p()._node(machine, row["callee"][1])
        _require(
            callee["kind"] == "closure"
            and machine["frameFunctions"].get(row["child"]) == callee["code"]
            and row["parent"] in machine["frameFunctions"],
            "callback source identity",
        )
        from .construction_v8 import _lexical_receiver

        receiver = _lexical_receiver(machine, callee, row["handler"])
        _require(machine["locals"].get(row["child"] + ":this") == receiver, "callback receiver")
        if row["child"] in machine["stack"]:
            depth = machine["stack"].index(row["child"])
            _require(
                depth == machine["stack"].index(row["parent"]) + 1 and depth > prior_depth,
                "callback stack order",
            )
            prior_depth = depth
            frame = next(
                (
                    frame
                    for frame in machine["protocol"]["callFrames"]
                    if frame["child"] == row["child"]
                ),
                None,
            )
            _require(
                frame
                and _eq(frame["callee"], row["callee"])
                and _eq(frame["arguments"], _callback_arguments(row)),
                "callback arguments",
            )
            _check_callback_scope(frame.get("proxyReflectionScope"), row, "active callback scope")
        else:
            _require(
                row is snapshot["pending"][-1]
                and machine["stack"][-1] == row["parent"]
                and (machine.get("lastReturn") or {}).get("kind") in ("return", "throw"),
                "returned callback owner",
            )
            _check_callback_scope(
                (machine.get("lastReturn") or {}).get("proxyReflectionScope"),
                row,
                "returned callback scope",
            )
    scoped_frames = [
        frame for frame in machine["protocol"]["callFrames"] if "proxyReflectionScope" in frame
    ]
    _require(
        all(
            any(row["child"] == frame["child"] and row["operation"] == "defineProperty"
                for row in snapshot["pending"])
            for frame in scoped_frames
        ),
        "orphaned active Proxy reflection scope",
    )
    if "proxyReflectionScope" in (machine.get("lastReturn") or {}):
        _require(
            snapshot["pending"][-1]["operation"] == "defineProperty"
            and snapshot["pending"][-1]["child"] not in machine["stack"],
            "orphaned returned Proxy reflection scope",
        )
    return True
