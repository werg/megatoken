"""Intrinsic Promise objects and reflection records for the shared v8 heap."""

from __future__ import annotations

import copy

FIELDS = {
    "promise.intrinsics": {"kind"},
    "promise.intrinsic.get": {"kind", "base", "key", "result"},
    "property.getPrototypeOf": {"kind", "object", "result"},
    "property.isExtensible": {"kind", "object", "result"},
    "property.descriptor": {"kind", "object", "key", "descriptor", "name"},
}
STATE_SCHEMA = "semantic-promise-object-state/1"
OBJECT_METHODS = (
    "__defineGetter__",
    "__defineSetter__",
    "hasOwnProperty",
    "__lookupGetter__",
    "__lookupSetter__",
    "isPrototypeOf",
    "propertyIsEnumerable",
    "toString",
    "valueOf",
    "toLocaleString",
)
NAMES = (
    "Promise",
    "Promise.prototype",
    "Promise.prototype.then",
    "Promise.prototype.catch",
    "Promise.prototype.finally",
    "Object",
    "Object.prototype",
    *("Object.prototype." + name for name in OBJECT_METHODS),
    "Object.prototype.__proto__.get",
    "Object.prototype.__proto__.set",
)


def intrinsic_id(name):
    return "intrinsic" + "".join(ch.lower() for ch in name if ch.isalnum())


IDS = tuple(intrinsic_id(name) for name in NAMES)


def _fail(message):
    from .protocol_v8 import ProtocolV8Error

    raise ProtocolV8Error("Invalid Promise object protocol: " + message)


def _row(kind, prototype="ordinary", intrinsic=None, callable_kind=None):
    row = {"kind": kind, "properties": {}}
    if intrinsic is not None:
        row.update(intrinsicName=intrinsic, callableKind=callable_kind)
    return row


def _data(value, *, writable=True, enumerable=False, configurable=True):
    return {
        "kind": "data",
        "value": copy.deepcopy(value),
        "writable": writable,
        "enumerable": enumerable,
        "configurable": configurable,
    }


def initialize(machine):
    """Install the fixed intrinsic group, refusing partial or colliding rows."""
    from .protocol_v8 import _assert_fresh_heap_identity

    state = machine["protocol"].get("promiseObjects")
    if state is not None:
        validate_state(machine)
        _fail("intrinsics already initialized")
    for identity in IDS:
        _assert_fresh_heap_identity(machine, identity)

    for node in machine["heap"].values():
        if node.get("kind") == "promise" and node.get("prototype", "ordinary") == "ordinary":
            node["prototype"] = ["ref", intrinsic_id("Promise.prototype")]

    heap = machine["heap"]
    callables = [name for name in NAMES if name not in ("Promise.prototype", "Object.prototype")]
    for name in callables:
        identity = intrinsic_id(name)
        heap[identity] = _row(
            "intrinsic",
            intrinsic=name,
            callable_kind="constructor" if name in ("Promise", "Object") else "function",
        )
    object_proto = _row("object", None)
    object_proto.update(
        code=None, captures=[], prototype=None, descriptors={}, keyOrder=[], extensible=True
    )
    heap[intrinsic_id("Object.prototype")] = object_proto
    promise_proto = _row("object", ["ref", intrinsic_id("Object.prototype")])
    promise_proto.update(
        code=None,
        captures=[],
        prototype=["ref", intrinsic_id("Object.prototype")],
        descriptors={},
        keyOrder=[],
        extensible=True,
    )
    heap[intrinsic_id("Promise.prototype")] = promise_proto
    _install_value(promise_proto, "constructor", ["ref", intrinsic_id("Promise")])
    _install_value(object_proto, "constructor", ["ref", intrinsic_id("Object")])
    for name in OBJECT_METHODS:
        if name == "__proto__":
            continue
        if name == "toLocaleString":
            continue
        _install_value(object_proto, name, ["ref", intrinsic_id("Object.prototype." + name)])
    object_proto["descriptors"]["__proto__"] = {
        "kind": "accessor",
        "get": ["ref", intrinsic_id("Object.prototype.__proto__.get")],
        "set": ["ref", intrinsic_id("Object.prototype.__proto__.set")],
        "enumerable": False,
        "configurable": True,
    }
    object_proto["keyOrder"].append("__proto__")
    _install_value(
        object_proto, "toLocaleString", ["ref", intrinsic_id("Object.prototype.toLocaleString")]
    )
    for name in ("then", "catch", "finally"):
        identity = intrinsic_id("Promise.prototype." + name)
        _install_value(promise_proto, name, ["ref", identity])
    symbol_tag = ("symbol", "well-known:toStringTag")
    _install_value(promise_proto, symbol_tag, ["str", "Promise"], writable=False)
    machine["protocol"]["promiseObjects"] = {
        "schema": STATE_SCHEMA,
        "initialized": True,
    }


def _install_value(node, key, value, *, writable=True, order_front=False):
    node["properties"][key] = copy.deepcopy(value)
    node["descriptors"][key] = {
        "kind": "data",
        "writable": writable,
        "enumerable": False,
        "configurable": True,
    }
    if order_front:
        node["keyOrder"].insert(0, key)
    else:
        node["keyOrder"].append(key)


def _descriptor_object(machine, identity, descriptor, name):
    if name is None:
        if descriptor is not None:
            _fail("descriptor result requires a returned object identity")
        return
    from .protocol_v8 import _assert_fresh_heap_identity

    if descriptor is None:
        _fail("missing property descriptor cannot allocate a result")
    _assert_fresh_heap_identity(machine, name)
    field_order = (
        ("value", "writable", "enumerable", "configurable")
        if descriptor["kind"] == "data"
        else ("get", "set", "enumerable", "configurable")
    )
    fields = {
        key: ["bool", descriptor[key]]
        if type(descriptor[key]) is bool
        else copy.deepcopy(descriptor[key])
        for key in field_order
    }
    props = {key: value for key, value in fields.items()}
    machine["heap"][name] = {
        "kind": "object",
        "code": None,
        "captures": [],
        "properties": props,
        "prototype": "ordinary",
    }


def validate_record(record):
    from .protocol_v8 import _exact, _identity, validate_protocol_key, validate_protocol_value

    if type(record) is not dict or record.get("kind") not in FIELDS:
        return False
    _exact(record, FIELDS[record["kind"]], "Promise object record")
    kind = record["kind"]
    if kind == "promise.intrinsics":
        return True
    if kind == "promise.intrinsic.get":
        if record["base"] not in ("Promise", "Object") or record["key"] != ["str", "prototype"]:
            _fail("intrinsic prototype read shape")
        validate_protocol_value(record["result"])
        if record["result"] != ["ref", intrinsic_id(record["base"] + ".prototype")]:
            _fail("intrinsic prototype read result")
    elif kind == "property.getPrototypeOf":
        _identity(record["object"])
        result = record["result"]
        validate_protocol_value(result)
        if result[0] not in ("ref", "null"):
            _fail("prototype result must be a reference or null")
    elif kind == "property.isExtensible":
        _identity(record["object"])
        validate_protocol_value(record["result"])
        if record["result"][0] != "bool":
            _fail("extensibility result must be boolean")
    elif kind == "property.descriptor":
        _identity(record["object"])
        if record["descriptor"] is not None:
            _identity(record["name"])
        elif record["name"] is not None:
            _fail("absent descriptor allocation")
        validate_protocol_key(record["key"])
        if record["descriptor"] is not None:
            from .protocol_v8 import _validate_descriptor

            _validate_descriptor(record["descriptor"])
    return True


def apply_record(machine, record):
    from .protocol_v8 import _descriptor, _key, _node

    kind = record["kind"]
    if kind == "promise.intrinsics":
        initialize(machine)
        return True
    if machine["protocol"].get("promiseObjects") is None:
        _fail("intrinsics are not initialized")
    if kind == "promise.intrinsic.get":
        intrinsic = machine["heap"].get(record["result"][1])
        expected_name = record["base"] + ".prototype"
        if (
            intrinsic is None
            or intrinsic.get("kind") != "object"
            or intrinsic_id(expected_name) != record["result"][1]
        ):
            _fail("intrinsic prototype identity is unavailable")
        return True
    node = _node(machine, record["object"])
    if kind == "property.getPrototypeOf":
        expected = node.get("prototype", "ordinary")
        if expected == "ordinary":
            if node["kind"] != "object":
                _fail("prototype query requires a modeled prototype")
            expected = ["ref", intrinsic_id("Object.prototype")]
        elif expected is None:
            expected = ["null"]
        if expected != record["result"]:
            _fail("false prototype query")
    elif kind == "property.isExtensible":
        if ["bool", node.get("extensible", True)] != record["result"]:
            _fail("false extensibility query")
    elif kind == "property.descriptor":
        descriptor = _descriptor(node, _key(record["key"]))
        if descriptor != record["descriptor"]:
            _fail("false descriptor query")
        name = record["name"]
        if descriptor is None:
            if name is not None:
                _fail("missing descriptor allocated a result")
        else:
            _descriptor_object(machine, record["object"], descriptor, name)
    return True


def validate_state(machine):

    state = machine["protocol"].get("promiseObjects")
    if state is None:
        if any(identity in machine["heap"] for identity in IDS):
            _fail("uninitialized Promise intrinsic rows")
        return True
    if state != {"schema": STATE_SCHEMA, "initialized": True}:
        _fail("malformed intrinsic initialization state")
    heap = machine["heap"]
    if any(identity not in heap for identity in IDS):
        _fail("intrinsic rows are incomplete")
    for name in NAMES:
        identity = intrinsic_id(name)
        row = heap[identity]
        expected_kind = (
            "object" if name in ("Promise.prototype", "Object.prototype") else "intrinsic"
        )
        if row.get("kind") != expected_kind:
            _fail("intrinsic row kind changed")
        if expected_kind == "intrinsic":
            expected_callable_kind = "constructor" if name in ("Promise", "Object") else "function"
            if (
                row.get("intrinsicName") != name
                or row.get("callableKind") != expected_callable_kind
                or row.get("properties") != {}
                or set(row) != {"kind", "intrinsicName", "callableKind", "properties"}
            ):
                _fail("intrinsic callable identity changed")
    for identity, row in heap.items():
        if row.get("kind") == "intrinsic" and identity not in IDS:
            _fail("undeclared intrinsic identity")
    return True
