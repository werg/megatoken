"""Independent bounded replay for ordinary synchronous Proxy get/set calls."""

from __future__ import annotations

import copy
import math

from . import algorithm_v7 as scalar

PROFILE = "ordinary-data-proxy/1"
SCHEMA = "semantic-proxy-call-state/1"
BOUND = 64
TRAPS = (
    "getPrototypeOf",
    "setPrototypeOf",
    "isExtensible",
    "preventExtensions",
    "getOwnPropertyDescriptor",
    "defineProperty",
    "has",
    "get",
    "set",
    "deleteProperty",
    "ownKeys",
    "apply",
    "construct",
)
FIELDS = {
    "proxy.create": {"kind", "name", "target", "handler"},
    "proxy.get": {"kind", "object", "key", "result"},
    "proxy.set": {"kind", "object", "key", "value", "success"},
    "proxy.get.call": {"kind", "id", "object", "key", "child"},
    "proxy.set.call": {"kind", "id", "object", "key", "value", "child"},
    "proxy.finish": {"kind", "id", "completion"},
}
HEAP_FIELDS = {"kind", "target", "handler", "properties"}
CALL_FIELDS = {
    "id",
    "operation",
    "parent",
    "child",
    "object",
    "target",
    "handler",
    "key",
    "value",
    "callee",
}


def _protocol():
    from . import protocol_v8

    return protocol_v8


def _fail(message):
    _protocol()._fail("Proxy: " + message)


def _id(value):
    if not isinstance(value, str) or not value or len(value) > 256:
        _fail("identity")


def validate_record(record):
    kind = record.get("kind") if type(record) is dict else None
    if kind not in FIELDS:
        return False
    p = _protocol()
    p._exact(record, FIELDS[kind], "Proxy record")
    if kind == "proxy.create":
        _id(record["name"])
        p.validate_protocol_value(record["target"])
        p.validate_protocol_value(record["handler"])
    elif kind == "proxy.finish":
        _id(record["id"])
        completion = record["completion"]
        p._exact(completion, {"kind", "value"}, "Proxy completion")
        if completion["kind"] not in ("normal", "throw"):
            _fail("completion kind")
        p.validate_protocol_value(completion["value"])
    else:
        _id(record["object"])
        p.validate_protocol_key(record["key"])
        if kind.endswith(".call"):
            _id(record["id"])
            _id(record["child"])
        if kind in ("proxy.get", "proxy.set"):
            p.validate_protocol_value(record["result"] if kind == "proxy.get" else record["value"])
        elif kind == "proxy.set.call":
            p.validate_protocol_value(record["value"])
        if kind == "proxy.set" and type(record["success"]) is not bool:
            _fail("set result")
    return True


def _ordinary(machine, reference, label):
    p = _protocol()
    p._known(machine, reference)
    if reference[0] != "ref":
        _fail(label + " identity")
    node = p._node(machine, reference[1])
    if node["kind"] != "object" or node.get("prototype", "ordinary") not in (None, "ordinary"):
        _fail(label + " must have an ordinary object prototype")
    for key in set(node.get("properties", {})) | set(node.get("descriptors", {})):
        descriptor = p._descriptor(node, key)
        if descriptor and descriptor["kind"] != "data":
            _fail(label + " accessors require their own protocol")
    return node


def _route(machine, node):
    _ordinary(machine, node["target"], "Proxy target")
    handler = _ordinary(machine, node["handler"], "Proxy handler")
    p = _protocol()
    for trap in TRAPS:
        descriptor = p._descriptor(handler, trap)
        if descriptor and descriptor["kind"] != "data":
            _fail("Proxy trap accessors require their own protocol")
        if descriptor and descriptor["value"][0] not in ("undefined", "null"):
            value = descriptor["value"]
            if (
                trap not in ("get", "set")
                or value[0] != "ref"
                or p._node(machine, value[1])["kind"] != "closure"
            ):
                _fail("Proxy trap " + trap + " requires its explicit call boundary")
    return handler


def _state(machine):
    state = machine["protocol"].get("proxyCalls")
    if state is None:
        state = {"schema": SCHEMA, "pending": []}
        machine["protocol"]["proxyCalls"] = state
    return state


def _eq(left, right):
    if left == right:
        # Tagged numeric -0 and +0 compare equal in Python but not SameValue.
        if isinstance(left, list) and len(left) == 2 and left[0] == "num":
            try:
                return math.copysign(1.0, float(left[1])) == math.copysign(1.0, float(right[1]))
            except (TypeError, ValueError, OverflowError):
                return True
        return True
    if (
        isinstance(left, list)
        and isinstance(right, list)
        and len(left) == len(right) == 2
        and left[0] == right[0] == "num"
    ):
        try:
            a, b = float(left[1]), float(right[1])
            return math.isnan(a) and math.isnan(b)
        except (TypeError, ValueError, OverflowError):
            return False
    return False


def _args(row):
    args = [row["target"], row["key"]]
    if row["operation"] == "set":
        args.append(row["value"])
    args.append(["ref", row["object"]])
    return args


def assert_next(machine, record):
    pending = (machine.get("protocol", {}).get("proxyCalls") or {}).get("pending", [])
    if pending and pending[-1]["child"] not in machine["stack"]:
        _require(
            record["kind"] == "proxy.finish" and record.get("id") == pending[-1]["id"],
            "pending callback requires its finish",
        )
    return True


def _require(condition, message):
    if not condition:
        _fail(message)


def apply_record(machine, record):
    if not validate_record(record):
        return False
    p = _protocol()
    kind = record["kind"]
    if kind == "proxy.create":
        _id(record["name"])
        reserved = {row["id"] for row in machine["protocol"].get("symbols", [])}
        reserved.update(
            row["id"]
            for row in (machine["protocol"].get("construction") or {}).get("privateNames", [])
        )
        _require(
            record["name"] not in machine["heap"] and record["name"] not in reserved,
            "fresh identity",
        )
        node = {
            "kind": "proxy",
            "target": copy.deepcopy(record["target"]),
            "handler": copy.deepcopy(record["handler"]),
            "properties": {},
        }
        _route(machine, node)
        machine["heap"][record["name"]] = node
        _state(machine)
        return True
    if kind == "proxy.finish":
        pending = _state(machine)["pending"]
        returned = machine.get("lastReturn")
        _require(
            bool(pending)
            and pending[-1]["id"] == record["id"]
            and machine["stack"][-1] == pending[-1]["parent"]
            and pending[-1]["child"] not in machine["stack"]
            and returned
            and returned["kind"] in ("return", "throw"),
            "callback completion identity",
        )
        row = pending[-1]
        node = machine["heap"].get(row["object"])
        _require(node and node["kind"] == "proxy", "known Proxy identity required")
        _route(machine, node)
        expected = {"kind": "throw", "value": copy.deepcopy(returned["value"])}
        if returned["kind"] == "return":
            if row["operation"] == "set":
                value = returned["value"]
                truthy = (
                    value[0] == "symbol"
                    or value[0] == "bigint"
                    and value[1] != "0"
                    or value[0] != "bigint"
                    and scalar._truthy(value)
                )
                _require(truthy, "false Proxy set trap needs abrupt completion")
            target = p._node(machine, node["target"][1])
            descriptor = p._descriptor(target, p._key(row["key"]))
            if descriptor and not descriptor["configurable"] and not descriptor["writable"]:
                trapped = returned["value"] if row["operation"] == "get" else row["value"]
                _require(
                    _eq(descriptor["value"], trapped),
                    "Proxy trap violates frozen target data invariant",
                )
            expected = {
                "kind": "normal",
                "value": copy.deepcopy(
                    returned["value"] if row["operation"] == "get" else row["value"]
                ),
            }
        _require(record["completion"] == expected, "false Proxy callback completion")
        pending.pop()
        return True
    node = machine["heap"].get(record["object"])
    _require(node and node["kind"] == "proxy", "known Proxy identity required")
    handler = _route(machine, node)
    p._known(machine, ["symbol", record["key"][1]]) if record["key"][0] == "symbol" else None
    operation = "get" if kind.startswith("proxy.get") else "set"
    method = (p._descriptor(handler, operation) or {}).get("value", ["undefined"])
    if kind.endswith(".call"):
        pending = _state(machine)["pending"]
        _require(
            len(pending) < BOUND and all(row["id"] != record["id"] for row in pending),
            "pending call identity or bound",
        )
        _require(
            method[0] == "ref" and p._node(machine, method[1])["kind"] == "closure",
            "call requires a live owned trap",
        )
        value = copy.deepcopy(record["value"]) if operation == "set" else None
        if value is not None:
            p._known(machine, value)
        row = {
            "id": record["id"],
            "operation": operation,
            "parent": machine["stack"][-1],
            "child": record["child"],
            "object": record["object"],
            "target": copy.deepcopy(node["target"]),
            "handler": copy.deepcopy(node["handler"]),
            "key": copy.deepcopy(record["key"]),
            "value": value,
            "callee": copy.deepcopy(method),
        }
        from .construction_v8 import _call

        _call(machine, row["child"], row["callee"], row["handler"], _args(row))
        pending.append(row)
        return True
    _require(method[0] in ("undefined", "null"), "trap requires its explicit call boundary")
    if kind == "proxy.get":
        _, descriptor = p._lookup_descriptor(machine, node["target"][1], p._key(record["key"]))
        _require(not descriptor or descriptor["kind"] == "data", "accessors require call boundary")
        expected = descriptor["value"] if descriptor else ["undefined"]
        _require(_eq(expected, record["result"]), "false Proxy data read")
    else:
        # Ordinary target [[Set]] forwards to the target as the receiver.
        target = p._node(machine, node["target"][1])
        key = p._key(record["key"])
        _, descriptor = p._lookup_descriptor(machine, node["target"][1], key)
        p._known(machine, record["value"])
        success = descriptor is None or descriptor["kind"] == "data" and descriptor["writable"]
        if success:
            own = p._descriptor(target, key)
            if own is not None and (own["kind"] != "data" or not own["writable"]):
                success = False
            else:
                current = own or {
                    "kind": "data",
                    "writable": True,
                    "enumerable": True,
                    "configurable": True,
                }
                success = p._define_own_property(
                    target, key, {**current, "value": copy.deepcopy(record["value"])}
                )
        _require(success == record["success"], "false Proxy assignment result")
    return True


def validate_state(machine):
    state = machine.get("protocol", {}).get("proxyCalls")
    proxies = 0
    for identity, node in machine["heap"].items():
        if node.get("kind") != "proxy":
            continue
        proxies += 1
        _id(identity)
        _protocol()._exact(node, HEAP_FIELDS, "Proxy heap node")
        _require(
            type(node["properties"]) is dict and not node["properties"],
            "Proxy has invented own storage",
        )
        _route(machine, node)
    if state is None:
        _require(proxies == 0, "Proxy heap lacks call state")
        return True
    _protocol()._exact(state, {"schema", "pending"}, "Proxy call state")
    _require(
        state["schema"] == SCHEMA
        and type(state["pending"]) is list
        and len(state["pending"]) <= BOUND,
        "state schema or bound",
    )
    seen = set()
    prior_depth = -1
    for row in state["pending"]:
        _protocol()._exact(row, CALL_FIELDS, "Proxy pending call")
        for name in ("id", "parent", "child", "object"):
            _id(row[name])
        _require(row["id"] not in seen and row["operation"] in ("get", "set"), "call identity")
        seen.add(row["id"])
        node = machine["heap"].get(row["object"])
        _require(
            node
            and node["kind"] == "proxy"
            and node["target"] == row["target"]
            and node["handler"] == row["handler"],
            "target/handler identity",
        )
        _protocol().validate_protocol_key(row["key"])
        if row["key"][0] == "symbol":
            _protocol()._symbol(machine, row["key"][1])
        _protocol()._known(machine, row["callee"])
        callee = _protocol()._node(machine, row["callee"][1])
        _require(
            row["callee"][0] == "ref"
            and callee["kind"] == "closure"
            and machine["frameFunctions"].get(row["child"]) == callee["code"]
            and row["parent"] in machine["frameFunctions"],
            "callback source identity",
        )
        if row["operation"] == "get":
            _require(row["value"] is None, "get has no assignment value")
        else:
            _protocol()._known(machine, row["value"])
        receiver = (
            callee.get("lexicalReceiver")
            if callee.get("functionKind") == "arrow"
            else row["handler"]
        )
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
                frame is not None and frame["arguments"] == _args(row),
                "callback retained arguments",
            )
        else:
            _require(
                row is state["pending"][-1]
                and machine["stack"][-1] == row["parent"]
                and machine.get("lastReturn", {}).get("kind") in ("return", "throw"),
                "finished callback owner",
            )
    return True
