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
    "proxy.revocable": {"kind", "proxy", "revoke", "result", "target", "handler"},
    "proxy.revoke": {"kind", "object"},
    "proxy.getMethod.begin": {
        "kind",
        "id",
        "operation",
        "object",
        "target",
        "handler",
        "receiver",
        "key",
        "child",
        "getter",
    },
    "proxy.getMethod.finish": {"kind", "id", "completion"},
    "proxy.revoked.fault": {"kind", "object", "operation", "error"},
    "proxy.get": {"kind", "object", "key", "result", "route"},
    "proxy.set": {"kind", "object", "key", "value", "success", "route"},
    "proxy.has": {"kind", "object", "key", "result", "route"},
    "proxy.delete": {"kind", "object", "key", "success", "route"},
    "proxy.delete.fault": {"kind", "object", "key", "error", "route"},
    "proxy.get.call": {"kind", "id", "object", "owner", "receiver", "key", "child", "route"},
    "proxy.set.call": {
        "kind",
        "id",
        "object",
        "owner",
        "receiver",
        "key",
        "value",
        "child",
        "strict",
        "route",
    },
    "proxy.has.call": {"kind", "id", "object", "owner", "receiver", "key", "child", "route"},
    "proxy.delete.call": {
        "kind",
        "id",
        "object",
        "owner",
        "receiver",
        "key",
        "child",
        "strict",
        "route",
    },
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
    "owner",
    "receiver",
    "target",
    "handler",
    "route",
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
    fields = set(FIELDS[kind])
    if kind == "proxy.has" and "id" in record:
        fields.add("id")
    p._exact(record, fields, "Proxy record")
    if kind == "proxy.create":
        _id(record["name"])
        p.validate_protocol_value(record["target"])
        p.validate_protocol_value(record["handler"])
    elif kind == "proxy.revocable":
        _id(record["proxy"])
        _id(record["revoke"])
        _id(record["result"])
        if len({record["proxy"], record["revoke"], record["result"]}) != 3:
            _fail("revocable Proxy identities")
        p.validate_protocol_value(record["target"])
        p.validate_protocol_value(record["handler"])
    elif kind == "proxy.revoke":
        _id(record["object"])
    elif kind == "proxy.getMethod.begin":
        _id(record["id"])
        _id(record["child"])
        _id(record["object"])
        for name in ("target", "handler", "receiver", "key", "getter"):
            p.validate_protocol_value(record[name]) if name != "key" else p.validate_protocol_key(
                record[name]
            )
        if (
            record["operation"] != "has"
            or record["target"][0] != "ref"
            or record["handler"][0] != "ref"
            or record["receiver"][0] != "ref"
            or record["getter"][0] != "ref"
            or record["key"][0] not in ("str", "symbol")
        ):
            _fail("GetMethod getter selection")
    elif kind == "proxy.getMethod.finish":
        _id(record["id"])
        completion = record["completion"]
        p._exact(completion, {"kind", "value"}, "Proxy GetMethod completion")
        if completion["kind"] not in ("normal", "throw"):
            _fail("GetMethod completion kind")
        p.validate_protocol_value(completion["value"])
    elif kind == "proxy.revoked.fault":
        _id(record["object"])
        _id(record["error"])
        if record["operation"] not in ("get", "set", "has", "deleteProperty"):
            _fail("revoked operation name")
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
        _validate_route(record["route"])
        if kind == "proxy.delete.fault":
            _id(record["error"])
        if kind.endswith(".call"):
            _id(record["id"])
            _id(record["child"])
            _id(record["owner"])
            p.validate_protocol_value(record["receiver"])
            if record["receiver"][0] != "ref":
                _fail("operation receiver identity")
        if kind in ("proxy.get", "proxy.has", "proxy.set"):
            p.validate_protocol_value(record["value"] if kind == "proxy.set" else record["result"])
        if kind == "proxy.has" and record["result"][0] != "bool":
            _fail("has result must be Boolean")
        if kind == "proxy.has" and "id" in record:
            _id(record["id"])
        elif kind == "proxy.set.call":
            p.validate_protocol_value(record["value"])
        if kind in ("proxy.delete.call", "proxy.set.call") and type(record["strict"]) is not bool:
            _fail("caller strictness flag")
        if kind in ("proxy.set", "proxy.delete") and type(record["success"]) is not bool:
            _fail("Boolean operation result")
    return True


def _validate_route(route):
    from . import protocol_v8 as p

    if type(route) is not list or not 1 <= len(route) <= BOUND:
        _fail("retained route bound")
    for row in route:
        p._exact(row, {"object", "target", "handler", "method"}, "retained route row")
        _id(row["object"])
        p.validate_protocol_value(row["target"])
        p.validate_protocol_value(row["handler"])
        p.validate_protocol_value(row["method"])
        if row["target"][0] != "ref" or row["handler"][0] != "ref":
            _fail("route identity references")


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
    # Authenticate immutable target/handler identities. A target can itself be
    # a Proxy; derive its internal-method dispatch separately at entry.
    p = _protocol()
    p._known(machine, node["target"])
    if node["target"] == ["null"] or node["handler"] == ["null"]:
        _require(
            node["target"] == ["null"] and node["handler"] == ["null"], "partial Proxy revocation"
        )
        return None
    _require(node["target"][0] == "ref", "target identity")
    target = machine["heap"].get(node["target"][1])
    _require(target and target["kind"] in ("object", "array", "proxy"), "target object identity")
    return _ordinary(machine, node["handler"], "Proxy handler")


def _method(machine, node, operation):
    _route(machine, node)
    _, descriptor = _protocol()._lookup_descriptor(
        machine, node["handler"][1], OPERATION_TRAPS.get(operation, operation), for_has=True
    )
    _require(
        descriptor is None or descriptor["kind"] == "data",
        "trap accessor requires its explicit getter boundary",
    )
    return descriptor["value"] if descriptor else ["undefined"]


def _derive_route(machine, object_id, operation):
    route, visited = [], set()
    node_id = object_id
    node = machine["heap"].get(node_id)
    target_ref = None
    owner = None
    selected = ["undefined"]
    while node and node["kind"] == "proxy":
        _require(node_id not in visited and len(route) < BOUND, "target route cycle or bound")
        visited.add(node_id)
        _require(
            node.get("target") is not None and node.get("handler") is not None,
            "revoked Proxy route",
        )
        _require(
            node["target"] != ["null"] and node["handler"] != ["null"],
            "revoked Proxy route requires its abrupt record",
        )
        selected = _method(machine, node, operation)
        entry = {
            "object": node_id,
            "target": copy.deepcopy(node["target"]),
            "handler": copy.deepcopy(node["handler"]),
            "method": copy.deepcopy(selected),
        }
        route.append(entry)
        owner = node_id
        target_ref = copy.deepcopy(node["target"])
        if selected[0] not in ("undefined", "null"):
            break
        node_id = target_ref[1]
        node = machine["heap"].get(node_id)
    _require(
        node and node["kind"] in ("object", "array", "proxy"), "target internal-method boundary"
    )
    return {
        "route": route,
        "owner": owner,
        "target": target_ref,
        "selected": selected,
        "terminal": node,
    }


def _retained_route(machine, row, *, selected=True):
    _validate_route(row["route"])
    route = row["route"]
    _require(
        route[0]["object"] == row["object"] and route[-1]["object"] == row["owner"],
        "route endpoints",
    )
    for index, entry in enumerate(route):
        node = machine["heap"].get(entry["object"])
        _require(
            node
            and node["kind"] == "proxy"
            and (
                (node.get("target") == ["null"] and node.get("handler") == ["null"])
                or _eq(node.get("target"), entry["target"])
                and _eq(node.get("handler"), entry["handler"])
            ),
            "retained route target/handler identity",
        )
        _protocol()._known(machine, entry["method"])
        if index < len(route) - 1:
            _require(
                entry["method"][0] in ("undefined", "null")
                and entry["target"][1] == route[index + 1]["object"],
                "retained transparent route edge",
            )
        elif selected:
            _require(
                entry["method"][0] not in ("undefined", "null"),
                "retained route lacks selected trap",
            )
    return machine["heap"].get(route[-1]["target"][1])


def _truthy(value):
    if value[0] == "symbol":
        return True
    if value[0] == "bigint":
        return value[1] != "0"
    return scalar._truthy(value)


def _state(machine):
    state = machine["protocol"].get("proxyCalls")
    if state is None:
        state = {"schema": SCHEMA, "pending": [], "getMethods": []}
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
        args.append(row["receiver"])
    return args


def assert_next(machine, record):
    state = machine.get("protocol", {}).get("proxyCalls") or {}
    lookup = (state.get("getMethods") or [None])[-1]
    if lookup:
        property_call = next(
            (
                row
                for row in machine["protocol"].get("pendingCalls", [])
                if row["id"] == lookup["id"]
            ),
            None,
        )
        if lookup["phase"] == "selected":
            if lookup["child"] not in machine["frameFunctions"]:
                _require(
                    record.get("kind") == "property.get.call"
                    and record.get("id") == lookup["id"]
                    and record.get("child") == lookup["child"]
                    and record.get("object") == lookup["handler"][1]
                    and record.get("key") == ["str", "has"]
                    and record.get("receiver") == lookup["receiver"],
                    "GetMethod selection requires its exact handler getter call",
                )
            elif property_call and lookup["child"] not in machine["stack"]:
                _require(
                    record.get("kind") == "property.finish" and record.get("id") == lookup["id"],
                    "GetMethod getter completion requires property.finish",
                )
            elif not property_call and lookup["child"] not in machine["stack"]:
                _require(
                    record.get("kind") == "proxy.getMethod.finish"
                    and record.get("id") == lookup["id"],
                    "GetMethod property completion requires its retained finish",
                )
        else:
            expected = "proxy.has" if lookup["method"][0] == "undefined" else "proxy.has.call"
            _require(
                record.get("kind") == expected and record.get("id") == lookup["id"],
                "GetMethod result requires its matching has operation",
            )
    pending = state.get("pending", [])
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
    if operation in ("delete", "set") and row["strict"] and not truth:
        trap_name = "deleteProperty" if operation == "delete" else "set"
        return (
            "'"
            + trap_name
            + "' on proxy: trap returned falsish for property '"
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
    if kind == "proxy.getMethod.begin":
        proxy_state = _state(machine)
        node = machine["heap"].get(record["object"])
        _require(
            len(proxy_state["getMethods"]) < BOUND
            and all(row["id"] != record["id"] for row in proxy_state["getMethods"])
            and all(row["id"] != record["id"] for row in proxy_state["pending"])
            and all(
                row["id"] != record["id"] for row in machine["protocol"].get("pendingCalls", [])
            ),
            "GetMethod identity or bound",
        )
        _require(
            node
            and node["kind"] == "proxy"
            and _eq(node["target"], record["target"])
            and _eq(node["handler"], record["handler"]),
            "GetMethod requires live selected target and handler",
        )
        target = _ordinary(machine, record["target"], "GetMethod target")
        handler = _ordinary(machine, record["handler"], "GetMethod handler")
        p.validate_protocol_key(record["key"])
        if record["key"][0] == "symbol":
            p._symbol(machine, record["key"][1])
        p._known(machine, record["receiver"])
        p._known(machine, record["getter"])
        descriptor = p._descriptor(handler, p._key(["str", "has"]))
        _require(
            record["receiver"] == record["handler"]
            and descriptor is not None
            and descriptor["kind"] == "accessor"
            and _eq(descriptor["get"], record["getter"])
            and record["getter"][0] == "ref"
            and p._node(machine, record["getter"][1])["kind"] == "closure",
            "GetMethod must retain its exact own getter and handler receiver",
        )
        _require(
            record["child"] not in machine["frameFunctions"] and bool(machine["stack"]),
            "GetMethod getter child is not fresh",
        )
        proxy_state["getMethods"].append(
            {
                "id": record["id"],
                "phase": "selected",
                "operation": record["operation"],
                "parent": machine["stack"][-1],
                "child": record["child"],
                "object": record["object"],
                "target": copy.deepcopy(record["target"]),
                "handler": copy.deepcopy(record["handler"]),
                "receiver": copy.deepcopy(record["receiver"]),
                "key": copy.deepcopy(record["key"]),
                "getter": copy.deepcopy(record["getter"]),
                "rawReturn": None,
                "method": None,
            }
        )
        return True
    if kind == "proxy.getMethod.finish":
        proxy_state = _state(machine)
        lookup = proxy_state["getMethods"][-1] if proxy_state["getMethods"] else None
        returned = machine.get("lastReturn")
        _require(
            lookup
            and lookup["id"] == record["id"]
            and lookup["phase"] == "selected"
            and machine["stack"][-1] == lookup["parent"]
            and lookup["child"] not in machine["stack"]
            and lookup["child"] in machine["frameFunctions"]
            and all(
                row["id"] != lookup["id"] for row in machine["protocol"].get("pendingCalls", [])
            )
            and returned
            and returned["kind"] in ("return", "throw"),
            "GetMethod finish lacks its exact returned getter call",
        )
        expected = {
            "kind": "throw" if returned["kind"] == "throw" else "normal",
            "value": copy.deepcopy(returned["value"]),
        }
        _require(record["completion"] == expected, "false GetMethod completion")
        if returned["kind"] == "throw":
            proxy_state["getMethods"].pop()
            return True
        raw = returned["value"]
        if raw[0] in ("undefined", "null"):
            method = ["undefined"]
        else:
            _require(
                raw[0] == "ref" and p._node(machine, raw[1])["kind"] == "closure",
                "non-callable GetMethod result outside bounded profile",
            )
            method = copy.deepcopy(raw)
        lookup["phase"] = "returned"
        lookup["rawReturn"] = copy.deepcopy(raw)
        lookup["method"] = method
        return True
    if kind == "proxy.revoke":
        node = machine["heap"].get(record["object"])
        _require(node and node["kind"] == "proxy", "known Proxy identity required")
        node["target"] = ["null"]
        node["handler"] = ["null"]
        _state(machine)
        return True
    if kind == "proxy.revoked.fault":
        node = machine["heap"].get(record["object"])
        _require(
            node
            and node["kind"] == "proxy"
            and node["target"] == ["null"]
            and node["handler"] == ["null"],
            "revoked Proxy fault requires cleared internal slots",
        )
        from .intrinsic_error_v8 import allocate_intrinsic_error_v8

        message = "Cannot perform '" + record["operation"] + "' on a proxy that has been revoked"
        value = allocate_intrinsic_error_v8(machine, record["error"], "TypeError", message)
        machine["lastReturn"] = {"kind": "throw", "value": copy.deepcopy(value)}
        return True
    if kind == "proxy.revocable":
        reserved = {row["id"] for row in machine["protocol"].get("symbols", [])}
        identities = (record["proxy"], record["revoke"], record["result"])
        _require(
            all(
                identity not in machine["heap"] and identity not in reserved
                for identity in identities
            )
            and len(set(identities)) == 3,
            "fresh revocable Proxy identities",
        )
        _ordinary(machine, record["handler"], "Proxy handler")
        _require(
            record["target"][0] == "ref"
            and machine["heap"].get(record["target"][1], {}).get("kind")
            in ("object", "array", "proxy"),
            "Proxy target object identity",
        )
        proxy_node = {
            "kind": "proxy",
            "target": copy.deepcopy(record["target"]),
            "handler": copy.deepcopy(record["handler"]),
            "properties": {},
        }
        _route(machine, proxy_node)
        revoke_node = {
            "kind": "closure",
            "code": "Proxy.revoke:" + record["proxy"],
            "captures": [],
            "functionKind": "ordinary",
            "lexicalReceiver": None,
            "prototype": "ordinary",
            "properties": {"length": ["num", 0], "name": ["str", ""]},
            "descriptors": {
                "length": {
                    "kind": "data",
                    "writable": False,
                    "enumerable": False,
                    "configurable": True,
                },
                "name": {
                    "kind": "data",
                    "writable": False,
                    "enumerable": False,
                    "configurable": True,
                },
            },
            "keyOrder": ["length", "name"],
        }
        result_node = {
            "kind": "object",
            "code": None,
            "captures": [],
            "prototype": "ordinary",
            "properties": {"proxy": ["ref", record["proxy"]], "revoke": ["ref", record["revoke"]]},
        }
        machine["heap"][record["proxy"]] = proxy_node
        machine["heap"][record["revoke"]] = revoke_node
        machine["heap"][record["result"]] = result_node
        _state(machine)
        return True
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
        target = _retained_route(machine, row)
        owner_node = machine["heap"].get(row["owner"], {})
        owner_revoked = owner_node.get("target") == ["null"] and owner_node.get("handler") == [
            "null"
        ]
        _require(
            _eq(target and ["ref", row["route"][-1]["target"][1]], row["target"])
            and (owner_revoked or _eq(owner_node.get("handler"), row["handler"])),
            "pending owner identities",
        )
        _protocol()._known(machine, row["receiver"])
        _require(_eq(row["receiver"], ["ref", row["object"]]), "original receiver differs")
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
                _require(
                    truthy or not row["strict"],
                    "strict false Proxy set trap needs its caller fault",
                )
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
    p._known(machine, ["symbol", record["key"][1]]) if record["key"][0] == "symbol" else None
    operation = kind.split(".")[1]
    proxy_state = _state(machine)
    lookup = next(
        (
            row
            for row in proxy_state["getMethods"]
            if operation == "has"
            and isinstance(record.get("id"), str)
            and row["id"] == record["id"]
        ),
        None,
    )
    if lookup:
        _require(
            lookup["phase"] == "returned"
            and lookup["operation"] == "has"
            and lookup["object"] == record["object"]
            and _eq(lookup["key"], record["key"])
            and machine.get("lastReturn", {}).get("kind") == "return"
            and _eq(machine["lastReturn"]["value"], lookup["rawReturn"]),
            "Proxy has differs from retained GetMethod result",
        )
        proxy_node = machine["heap"].get(lookup["object"])
        _require(
            proxy_node
            and proxy_node["kind"] == "proxy"
            and (
                proxy_node["target"] == ["null"]
                and proxy_node["handler"] == ["null"]
                or _eq(proxy_node["target"], lookup["target"])
                and _eq(proxy_node["handler"], lookup["handler"])
            ),
            "GetMethod lost retained target or handler",
        )
        route = [
            {
                "object": lookup["object"],
                "target": copy.deepcopy(lookup["target"]),
                "handler": copy.deepcopy(lookup["handler"]),
                "method": copy.deepcopy(lookup["method"]),
            }
        ]
        derived = {
            "route": route,
            "owner": lookup["object"],
            "target": copy.deepcopy(lookup["target"]),
            "selected": copy.deepcopy(lookup["method"]),
            "terminal": machine["heap"].get(lookup["target"][1]),
        }
    else:
        if kind == "proxy.has" and "id" in record:
            _require(False, "GetMethod identity is not retained")
        derived = _derive_route(machine, record["object"], operation)
    _require(
        _eq(record["route"], derived["route"]), "operation route differs from live entry lookup"
    )
    method = derived["selected"]
    target_ref = derived["target"]
    handler_ref = derived.get("handler", derived["route"][-1]["handler"])
    if kind.endswith(".call"):
        pending = _state(machine)["pending"]
        _require(
            len(pending) < BOUND and all(row["id"] != record["id"] for row in pending),
            "pending call identity or bound",
        )
        _require(
            record["owner"] == derived["owner"]
            and _eq(record["receiver"], ["ref", record["object"]]),
            "original receiver or trap owner differs",
        )
        _require(
            method[0] == "ref" and p._node(machine, method[1])["kind"] == "closure",
            "call requires a live owned trap",
        )
        _ordinary(machine, target_ref, "Proxy trap target invariant boundary")
        value = copy.deepcopy(record["value"]) if operation == "set" else None
        if value is not None:
            p._known(machine, value)
        row = {
            "id": record["id"],
            "operation": operation,
            "parent": machine["stack"][-1],
            "child": record["child"],
            "object": record["object"],
            "owner": derived["owner"],
            "receiver": copy.deepcopy(record["receiver"]),
            "target": copy.deepcopy(target_ref),
            "handler": copy.deepcopy(handler_ref),
            "route": copy.deepcopy(derived["route"]),
            "key": copy.deepcopy(record["key"]),
            "value": value,
            "callee": copy.deepcopy(method),
            "strict": record["strict"] if operation in ("delete", "set") else False,
        }
        from .construction_v8 import _call

        _call(machine, row["child"], row["callee"], row["handler"], _args(row))
        pending.append(row)
        if lookup:
            proxy_state["getMethods"].pop()
        return True
    _require(method[0] in ("undefined", "null"), "trap requires its explicit call boundary")
    _ordinary(machine, target_ref, "Proxy target")
    if kind == "proxy.has":
        _, descriptor = p._lookup_descriptor(
            machine, target_ref[1], p._key(record["key"]), for_has=True
        )
        _require(
            record["result"] == ["bool", descriptor is not None], "false Proxy existence result"
        )
        if lookup:
            proxy_state["getMethods"].pop()
    elif kind in ("proxy.delete", "proxy.delete.fault"):
        target = p._node(machine, target_ref[1])
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
        _, descriptor = p._lookup_descriptor(machine, target_ref[1], p._key(record["key"]))
        _require(not descriptor or descriptor["kind"] == "data", "accessors require call boundary")
        expected = descriptor["value"] if descriptor else ["undefined"]
        _require(_eq(expected, record["result"]), "false Proxy data read")
    else:
        # The original receiver is the Proxy. Forwarding its own-descriptor
        # and definition requests directly is valid only when both traps are absent.
        for entry in derived["route"]:
            for trap in ("getOwnPropertyDescriptor", "defineProperty"):
                _, selected = p._lookup_descriptor(machine, entry["handler"][1], trap, for_has=True)
                _require(
                    selected is None
                    or selected["kind"] == "data"
                    and selected["value"][0] in ("undefined", "null"),
                    "transparent set requires explicit receiver descriptor/definition traps",
                )
        _require(
            len(derived["route"]) == 1,
            "nested transparent set receiver methods need their continuation",
        )
        target = p._node(machine, target_ref[1])
        key = p._key(record["key"])
        _, descriptor = p._lookup_descriptor(machine, target_ref[1], key, for_set=True)
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
    for identity, node in machine["heap"].items():
        if node.get("kind") == "proxy" and node["target"] != ["null"]:
            visited, target_id = {identity}, node["target"][1]
            target = machine["heap"].get(target_id)
            while target and target.get("kind") == "proxy" and target["target"] != ["null"]:
                _require(target_id not in visited, "target graph cycle")
                visited.add(target_id)
                target_id = target["target"][1]
                target = machine["heap"].get(target_id)
    if state is None:
        _require(proxies == 0, "Proxy heap lacks call state")
        return True
    _protocol()._exact(state, {"schema", "pending", "getMethods"}, "Proxy call state")
    _require(
        state["schema"] == SCHEMA
        and type(state["pending"]) is list
        and len(state["pending"]) <= BOUND
        and type(state["getMethods"]) is list
        and len(state["getMethods"]) <= BOUND,
        "state schema or bound",
    )
    seen = set()
    prior_depth = -1
    for row in state["pending"]:
        _protocol()._exact(row, CALL_FIELDS, "Proxy pending call")
        for name in ("id", "parent", "child", "object", "owner"):
            _id(row[name])
        _require(row["id"] not in seen and row["operation"] in OPERATION_TRAPS, "call identity")
        seen.add(row["id"])
        _require(
            type(row["strict"]) is bool
            and (row["operation"] in ("delete", "set") or not row["strict"]),
            "retained strict deletion flag",
        )
        node = machine["heap"].get(row["owner"])
        _require(
            node
            and node["kind"] == "proxy"
            and (
                (node["target"] == ["null"] and node["handler"] == ["null"])
                or node["target"] == row["target"]
                and node["handler"] == row["handler"]
            ),
            "target/handler identity",
        )
        _retained_route(machine, row)
        _require(row["route"][-1]["target"] == row["target"], "retained selected target")
        _require(row["route"][-1]["handler"] == row["handler"], "retained selected handler")
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
    for lookup in state["getMethods"]:
        _protocol()._exact(
            lookup,
            {
                "id",
                "phase",
                "operation",
                "parent",
                "child",
                "object",
                "target",
                "handler",
                "receiver",
                "key",
                "getter",
                "rawReturn",
                "method",
            },
            "Proxy GetMethod call",
        )
        for name in ("id", "parent", "child", "object"):
            _id(lookup[name])
        _require(
            lookup["id"] not in seen
            and lookup["operation"] == "has"
            and lookup["phase"] in ("selected", "returned"),
            "GetMethod identity or phase",
        )
        seen.add(lookup["id"])
        proxy_node = machine["heap"].get(lookup["object"])
        _require(
            proxy_node
            and proxy_node["kind"] == "proxy"
            and (
                proxy_node["target"] == ["null"]
                and proxy_node["handler"] == ["null"]
                or _eq(proxy_node["target"], lookup["target"])
                and _eq(proxy_node["handler"], lookup["handler"])
            ),
            "GetMethod retained Proxy target/handler",
        )
        _ordinary(machine, lookup["target"], "GetMethod target")
        handler = _ordinary(machine, lookup["handler"], "GetMethod handler")
        _protocol()._known(machine, lookup["receiver"])
        _require(lookup["receiver"] == lookup["handler"], "GetMethod handler receiver")
        _protocol().validate_protocol_key(lookup["key"])
        if lookup["key"][0] == "symbol":
            _protocol()._symbol(machine, lookup["key"][1])
        _protocol()._known(machine, lookup["getter"])
        _require(lookup["getter"][0] == "ref", "GetMethod getter identity")
        getter = _protocol()._node(machine, lookup["getter"][1])
        _require(
            getter["kind"] == "closure" and lookup["parent"] in machine["frameFunctions"],
            "GetMethod getter source identity",
        )
        property_call = next(
            (
                row
                for row in machine["protocol"].get("pendingCalls", [])
                if row["id"] == lookup["id"]
            ),
            None,
        )
        child_active = lookup["child"] in machine["frameFunctions"]
        if not child_active:
            _require(
                _eq(proxy_node["target"], lookup["target"])
                and _eq(proxy_node["handler"], lookup["handler"]),
                "GetMethod begin requires live target/handler slots",
            )
            _require(
                lookup["phase"] == "selected"
                and property_call is None
                and lookup["child"] not in machine["stack"]
                and machine["stack"][-1] == lookup["parent"]
                and lookup["rawReturn"] is None
                and lookup["method"] is None,
                "GetMethod is not awaiting its exact getter call",
            )
            descriptor = _protocol()._descriptor(handler, _protocol()._key(["str", "has"]))
            _require(
                descriptor
                and descriptor["kind"] == "accessor"
                and _eq(descriptor["get"], lookup["getter"]),
                "GetMethod selected accessor differs from live handler getter",
            )
            continue

        _require(
            machine["frameFunctions"].get(lookup["child"]) == getter["code"],
            "GetMethod getter invocation source differs",
        )
        from .construction_v8 import _lexical_receiver

        receiver = _lexical_receiver(machine, getter, lookup["receiver"])
        _require(
            machine["locals"].get(lookup["child"] + ":this") == receiver,
            "GetMethod getter activation receiver",
        )
        if property_call:
            _require(
                lookup["phase"] == "selected"
                and lookup["rawReturn"] is None
                and lookup["method"] is None
                and property_call["operation"] == "get"
                and property_call["child"] == lookup["child"]
                and property_call["object"] == lookup["handler"][1]
                and property_call["key"] == ["str", "has"]
                and _eq(property_call["receiver"], lookup["receiver"])
                and _eq(property_call["callee"], lookup["getter"])
                and property_call["parent"] == lookup["parent"]
                and property_call["value"] is None,
                "GetMethod differs from its shared property getter call",
            )
            frame = next(
                (
                    frame
                    for frame in machine["protocol"].get("callFrames", [])
                    if frame["child"] == lookup["child"]
                ),
                None,
            )
            if frame:
                _require(
                    frame["callee"] == lookup["getter"]
                    and frame["parent"] == lookup["parent"]
                    and frame["arguments"] == [],
                    "GetMethod active getter call frame differs",
                )
            if lookup["child"] in machine["stack"]:
                _require(
                    machine["stack"].index(lookup["child"])
                    > machine["stack"].index(lookup["parent"]),
                    "GetMethod getter stack order",
                )
            else:
                _require(
                    machine["stack"][-1] == lookup["parent"]
                    and machine.get("lastReturn", {}).get("kind") in ("return", "throw"),
                    "GetMethod getter has no returned completion",
                )
        elif lookup["phase"] == "selected":
            _require(
                lookup["child"] not in machine["stack"]
                and machine["stack"][-1] == lookup["parent"]
                and machine.get("lastReturn", {}).get("kind") in ("return", "throw")
                and lookup["rawReturn"] is None
                and lookup["method"] is None,
                "GetMethod property completion is not retained",
            )
        else:
            _require(
                lookup["child"] not in machine["stack"]
                and machine["stack"][-1] == lookup["parent"]
                and machine.get("lastReturn", {}).get("kind") == "return"
                and _eq(machine["lastReturn"]["value"], lookup["rawReturn"]),
                "GetMethod returned value differs from retained completion",
            )
            expected_method = (
                ["undefined"]
                if lookup["rawReturn"] and lookup["rawReturn"][0] in ("undefined", "null")
                else lookup["rawReturn"]
            )
            _require(
                lookup["rawReturn"] is not None
                and _eq(lookup["method"], expected_method)
                and (
                    expected_method[0] == "undefined"
                    or expected_method[0] == "ref"
                    and _protocol()._node(machine, expected_method[1])["kind"] == "closure"
                ),
                "GetMethod method identity differs from its raw return",
            )
    return True
