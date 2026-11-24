"""Independent replay for retained ordinary Proxy property operations."""

from __future__ import annotations

import copy
import math

from . import semantic_primitives as scalar
from .semantic_json import _js_string_length

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
OPERATION_TRAPS = {"get": "get", "set": "set", "has": "has", "delete": "deleteProperty"}
FIELDS = {
    "proxy.create": {"kind", "name", "target", "handler"},
    "proxy.get": {"kind", "object", "key", "result"},
    "proxy.set": {"kind", "object", "key", "value", "success"},
    "proxy.has": {"kind", "object", "key", "result"},
    "proxy.delete": {"kind", "object", "key", "success"},
    "proxy.delete.fault": {"kind", "object", "key", "error"},
    "proxy.get.call": {"kind", "id", "object", "key", "child"},
    "proxy.set.call": {"kind", "id", "object", "key", "value", "child"},
    "proxy.has.call": {"kind", "id", "object", "key", "child"},
    "proxy.delete.call": {"kind", "id", "object", "key", "child", "strict"},
    "proxy.finish": {"kind", "id", "completion"},
    "proxy.fault": {"kind", "id", "error"},
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
    "strict",
}


def _protocol():
    from . import protocol_v8

    return protocol_v8


def _fail(message):
    _protocol()._fail("Proxy: " + message)


def _id(value):
    if not isinstance(value, str) or not value or _js_string_length(value) > 256:
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
    elif kind == "proxy.fault":
        _id(record["id"])
        _id(record["error"])
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
        if kind == "proxy.delete.fault":
            _id(record["error"])
        if kind.endswith(".call"):
            _id(record["id"])
            _id(record["child"])
        if kind in ("proxy.get", "proxy.has", "proxy.set"):
            p.validate_protocol_value(record["value"] if kind == "proxy.set" else record["result"])
        if kind == "proxy.has" and record["result"][0] != "bool":
            _fail("has result must be Boolean")
        elif kind == "proxy.set.call":
            p.validate_protocol_value(record["value"])
        if kind == "proxy.delete.call" and type(record["strict"]) is not bool:
            _fail("strict deletion flag")
        if kind in ("proxy.set", "proxy.delete") and type(record["success"]) is not bool:
            _fail("Boolean operation result")
    return True


def _ordinary(machine, reference, label):
    p = _protocol()
    p._known(machine, reference)
    if reference[0] != "ref":
        _fail(label + " identity")
    node = p._node(machine, reference[1])
    if node["kind"] != "object" or node.get("prototype", "ordinary") not in (None, "ordinary"):
        _fail(label + " must have an ordinary object prototype")
    return node


def _route(machine, node):
    # State validation authenticates the retained identities and shapes, not
    # the current value of a trap that guest code may already have replaced.
    _ordinary(machine, node["target"], "Proxy target")
    return _ordinary(machine, node["handler"], "Proxy handler")


def _method(machine, handler, operation):
    _, descriptor = _protocol()._lookup_descriptor(
        machine, handler, OPERATION_TRAPS[operation], for_has=True
    )
    _require(
        descriptor is None or descriptor["kind"] == "data",
        "trap accessor requires its explicit getter boundary",
    )
    return descriptor["value"] if descriptor else ["undefined"]


def _truthy(value):
    if value[0] == "symbol":
        return True
    if value[0] == "bigint":
        return value[1] != "0"
    return scalar._truthy(value)


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
    if row["operation"] in ("get", "set"):
        args.append(["ref", row["object"]])
    return args


def assert_next(machine, record):
    pending = (machine.get("protocol", {}).get("proxyCalls") or {}).get("pending", [])
    if pending and pending[-1]["child"] not in machine["stack"]:
        _require(
            record["kind"] in ("proxy.finish", "proxy.fault")
            and record.get("id") == pending[-1]["id"],
            "pending callback requires its finish",
        )
    return True


def _require(condition, message):
    if not condition:
        _fail(message)


def _key_display(machine, key):
    if key[0] == "str":
        return key[1]
    symbol = _protocol()._symbol(machine, key[1])
    description = (
        ["str", "Symbol." + symbol["name"]]
        if symbol["kind"] == "well-known"
        else symbol["description"]
    )
    return "Symbol(" + (description[1] if description[0] == "str" else "") + ")"


def _invariant_fault(machine, row, returned):
    if returned["kind"] != "return":
        return None
    p = _protocol()
    operation, value = row["operation"], returned["value"]
    truth = _truthy(value)
    if operation == "delete" and row["strict"] and not truth:
        return (
            "'deleteProperty' on proxy: trap returned falsish for property '"
            + _key_display(machine, row["key"])
            + "'"
        )
    target = p._node(machine, row["target"][1])
    descriptor = p._descriptor(target, p._key(row["key"]))
    if not descriptor:
        return None
    restricted = not descriptor["configurable"] or not target.get("extensible", True)
    if operation == "has" and not truth and restricted:
        return (
            "'has' on proxy: trap returned falsish for property '"
            + _key_display(machine, row["key"])
            + (
                "' which exists in the proxy target as non-configurable"
                if not descriptor["configurable"]
                else "' but the proxy target is not extensible"
            )
        )
    if operation == "delete" and truth and restricted:
        return (
            "'deleteProperty' on proxy: trap returned truish for property '"
            + _key_display(machine, row["key"])
            + (
                "' which is non-configurable in the proxy target"
                if not descriptor["configurable"]
                else "' but the proxy target is non-extensible"
            )
        )
    if operation in ("get", "set") and not descriptor["configurable"]:
        if operation == "set" and not truth:
            return None  # Caller strictness is a separate operation.
        if descriptor["kind"] == "data" and not descriptor["writable"]:
            if not _eq(descriptor["value"], value if operation == "get" else row["value"]):
                return (
                    "Proxy "
                    + operation
                    + " trap contradicted a non-configurable, non-writable target property"
                )
        elif descriptor["kind"] == "accessor" and descriptor[operation] == ["undefined"]:
            if operation == "set" or value != ["undefined"]:
                return (
                    "Proxy " + operation + " trap contradicted a non-configurable target accessor"
                )
    return None


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
    if kind in ("proxy.finish", "proxy.fault"):
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
        fault = _invariant_fault(machine, row, returned)
        if kind == "proxy.fault":
            _require(fault is not None, "Proxy fault requires a derived target invariant violation")
            from .intrinsic_error_v8 import allocate_intrinsic_error_v8

            value = allocate_intrinsic_error_v8(machine, record["error"], "TypeError", fault)
            pending.pop()
            machine["lastReturn"] = {"kind": "throw", "value": copy.deepcopy(value)}
            return True
        _require(fault is None, "Proxy trap violates target invariant; fault required")
        expected = {"kind": "throw", "value": copy.deepcopy(returned["value"])}
        if returned["kind"] == "return":
            operation = row["operation"]
            truthy = _truthy(returned["value"])
            if operation == "set":
                _require(truthy, "false Proxy set trap needs abrupt completion")
            result = (
                ["bool", truthy]
                if operation in ("has", "delete")
                else returned["value"]
                if operation == "get"
                else row["value"]
            )
            expected = {"kind": "normal", "value": copy.deepcopy(result)}
        _require(record["completion"] == expected, "false Proxy callback completion")
        pending.pop()
        return True
    node = machine["heap"].get(record["object"])
    _require(node and node["kind"] == "proxy", "known Proxy identity required")
    _route(machine, node)
    p._known(machine, ["symbol", record["key"][1]]) if record["key"][0] == "symbol" else None
    operation = kind.split(".")[1]
    method = _method(machine, node["handler"][1], operation)
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
            "strict": record["strict"] if operation == "delete" else False,
        }
        from .construction_v8 import _call

        _call(machine, row["child"], row["callee"], row["handler"], _args(row))
        pending.append(row)
        return True
    _require(method[0] in ("undefined", "null"), "trap requires its explicit call boundary")
    if kind == "proxy.has":
        _, descriptor = p._lookup_descriptor(
            machine, node["target"][1], p._key(record["key"]), for_has=True
        )
        _require(
            record["result"] == ["bool", descriptor is not None], "false Proxy existence result"
        )
    elif kind in ("proxy.delete", "proxy.delete.fault"):
        target = p._node(machine, node["target"][1])
        key = p._key(record["key"])
        descriptor = p._descriptor(target, key)
        success = descriptor is None or descriptor["configurable"]
        if kind == "proxy.delete.fault":
            _require(not success, "strict deletion fault requires a nonconfigurable own property")
            from .intrinsic_error_v8 import allocate_intrinsic_error_v8

            message = (
                "Cannot delete property '" + _key_display(machine, record["key"]) + "' of #<Object>"
            )
            value = allocate_intrinsic_error_v8(machine, record["error"], "TypeError", message)
            machine["lastReturn"] = {"kind": "throw", "value": copy.deepcopy(value)}
        else:
            _require(record["success"] == success, "false Proxy deletion result")
            if success:
                p._remove(target, key)
    elif kind == "proxy.get":
        _, descriptor = p._lookup_descriptor(machine, node["target"][1], p._key(record["key"]))
        _require(not descriptor or descriptor["kind"] == "data", "accessors require call boundary")
        expected = descriptor["value"] if descriptor else ["undefined"]
        _require(_eq(expected, record["result"]), "false Proxy data read")
    else:
        # The original receiver is the Proxy. Forwarding its own-descriptor
        # and definition requests directly is valid only when both traps are absent.
        for trap in ("getOwnPropertyDescriptor", "defineProperty"):
            _, selected = p._lookup_descriptor(machine, node["handler"][1], trap, for_has=True)
            _require(
                selected is None
                or selected["kind"] == "data"
                and selected["value"][0] in ("undefined", "null"),
                "transparent set requires explicit receiver descriptor/definition traps",
            )
        target = p._node(machine, node["target"][1])
        key = p._key(record["key"])
        _, descriptor = p._lookup_descriptor(machine, node["target"][1], key, for_set=True)
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
        _require(row["id"] not in seen and row["operation"] in OPERATION_TRAPS, "call identity")
        seen.add(row["id"])
        _require(
            type(row["strict"]) is bool and (row["operation"] == "delete" or not row["strict"]),
            "retained strict deletion flag",
        )
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
        if row["operation"] != "set":
            _require(row["value"] is None, "non-set operation has no assignment value")
        else:
            _protocol()._known(machine, row["value"])
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
                frame is not None
                and frame["callee"] == row["callee"]
                and frame["arguments"] == _args(row),
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
