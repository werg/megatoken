"""Independent Set transitions for the bounded issue51 collection profile."""

from __future__ import annotations

import copy
import math

from . import semantic_primitives as primitives

FIELDS = {
    "set.lookup": {"kind", "object", "key", "method"},
    "set.create": {"kind", "name", "iterable"},
    "set.add": {"kind", "object", "value", "result"},
    "set.has": {"kind", "object", "value", "result"},
}
STATE_SCHEMA = "semantic-collections-state/1"
BOUND = 10000


def _fail(message):
    from .protocol_v8 import ProtocolV8Error

    raise ProtocolV8Error("Invalid Set protocol: " + message)


def _value(value):
    from .protocol_v8 import validate_protocol_value

    return validate_protocol_value(value)


def _identity(value):
    if not isinstance(value, str) or not value or len(value) > 256:
        _fail("identity")


def _validate_record(record):
    kind = record.get("kind") if type(record) is dict else None
    fields = FIELDS.get(kind)
    if fields is None:
        return False
    if set(record) != fields:
        _fail("record fields")
    for field in ("object", "name"):
        if field in record:
            _identity(record[field])
    if kind == "set.lookup":
        key = record["key"]
        if (
            not isinstance(key, list)
            or len(key) != 2
            or key[0] != "str"
            or key[1] not in ("add", "has")
            or record["method"] != "Set.prototype." + key[1]
        ):
            _fail("Set method identity")
    elif kind == "set.create":
        _identity(record["name"])
        _value(record["iterable"])
        if record["iterable"][0] not in ("ref", "null", "undefined"):
            _fail("constructor iterable tag")
    else:
        _value(record["value"])
        _value(record["result"])
        if kind == "set.add" and record["result"][0] != "ref":
            _fail("Set.add receiver identity")
    return True


def _known(machine, value):
    from .protocol_v8 import _known

    return _known(machine, value)


def _same_value_zero(left, right):
    if left[0] != right[0]:
        return False
    if left[0] in ("ref", "symbol"):
        return left[1] == right[1]
    if left[0] == "num":
        first = primitives._decode_scalar(left)
        second = primitives._decode_scalar(right)
        return first == second or math.isnan(first) and math.isnan(second)
    return left == right


def _normal(value):
    if value[0] == "num":
        number = primitives._decode_scalar(value)
        if number == 0:
            return ["num", 0]
    return copy.deepcopy(value)


def _array_values(machine, value):
    _known(machine, value)
    if value[0] in ("undefined", "null"):
        return []
    if value[0] != "ref":
        _fail("constructor requires a dense ordinary array")
    array = machine["heap"].get(value[1])
    if (
        not array
        or array.get("kind") != "array"
        or array.get("prototype") not in (None, "ordinary", "Array.prototype")
        or any(metadata.get("kind") != "data" for metadata in array.get("descriptors", {}).values())
        or ("symbol", "well-known:iterator") in array.get("properties", {})
    ):
        _fail("constructor iterator/accessor profile")
    length = array.get("properties", {}).get("length")
    if not isinstance(length, list) or length[0] != "num":
        _fail("constructor array length")
    size = primitives._decode_scalar(length)
    if not isinstance(size, (int, float)) or not math.isfinite(size) or not float(size).is_integer() or not 0 <= size <= BOUND:
        _fail("constructor array length bound")
    size = int(size)
    entries = []
    for index in range(size):
        item = array.get("properties", {}).get(str(index))
        if item is None:
            _fail("sparse Set constructor iterable is unsupported")
        _known(machine, item)
        if not any(_same_value_zero(item, previous) for previous in entries):
            entries.append(_normal(item))
    return entries


def _registry(machine):
    state = machine["protocol"]
    if state.get("collections") is None:
        state["collections"] = {"schema": STATE_SCHEMA, "collections": []}
    registry = state["collections"]
    if type(registry) is not dict or registry.get("schema") != STATE_SCHEMA:
        _fail("collection registry")
    return registry["collections"]


def validate_record(record):
    return _validate_record(record)


def apply_record(machine, record):
    if not _validate_record(record):
        return False
    kind = record["kind"]
    if kind == "set.lookup":
        node = machine["heap"].get(record["object"])
        if not node or node.get("kind") != "set":
            _fail("Set method receiver")
        if record["method"] != "Set.prototype." + record["key"][1]:
            _fail("false Set method selection")
    elif kind == "set.create":
        identity = record["name"]
        if identity in machine["heap"] or any(
            row["id"] == identity for row in machine["protocol"]["symbols"]
        ):
            _fail("Set identity collision")
        entries = _array_values(machine, record["iterable"])
        machine["heap"][identity] = {"kind": "set", "entries": entries, "properties": {}}
        rows = _registry(machine)
        if len(rows) >= 4096 or any(row.get("id") == identity for row in rows):
            _fail("Set registry bound or duplicate")
        rows.append({"id": identity, "kind": "set"})
    else:
        node = machine["heap"].get(record["object"])
        if not node or node.get("kind") != "set":
            _fail("Set receiver")
        _known(machine, record["value"])
        index = next(
            (i for i, value in enumerate(node["entries"]) if _same_value_zero(value, record["value"])),
            None,
        )
        if kind == "set.has":
            _known(machine, record["result"])
            if record["result"] != ["bool", index is not None]:
                _fail("false Set.has result")
        else:
            _known(machine, record["result"])
            if record["result"] != ["ref", record["object"]]:
                _fail("Set.add receiver result")
            if index is None:
                if len(node["entries"]) >= BOUND:
                    _fail("Set size bound")
                node["entries"].append(_normal(record["value"]))
    validate_state(machine, machine["protocol"].get("collections"))
    return True


def validate_state(machine, snapshot=None):
    if snapshot is None:
        snapshot = machine.get("protocol", {}).get("collections")
    nodes = {identity: node for identity, node in machine["heap"].items() if node.get("kind") == "set"}
    if snapshot is None:
        if nodes:
            _fail("Set nodes lack a retained allocation registry")
        return {"valid": True, "collections": 0}
    if (
        type(snapshot) is not dict
        or set(snapshot) != {"schema", "collections"}
        or snapshot["schema"] != STATE_SCHEMA
        or not isinstance(snapshot["collections"], list)
        or len(snapshot["collections"]) > 4096
    ):
        _fail("collection registry shape")
    rows = {}
    for row in snapshot["collections"]:
        if type(row) is not dict or set(row) != {"id", "kind"} or row["kind"] != "set":
            _fail("Set registry row shape")
        _identity(row["id"])
        if row["id"] in rows:
            _fail("duplicate Set registry row")
        rows[row["id"]] = row["kind"]
    if len(rows) != len(nodes):
        _fail("Set registry/node count")
    for identity, node in nodes.items():
        entries = node.get("entries")
        if (
            rows.get(identity) != "set"
            or set(node) != {"kind", "entries", "properties"}
            or node["properties"] != {}
            or not isinstance(entries, list)
            or len(entries) > BOUND
        ):
            _fail("Set heap shape or allocation registry")
        seen = []
        for value in entries:
            _known(machine, value)
            if any(_same_value_zero(value, previous) for previous in seen):
                _fail("duplicate Set entry")
            seen.append(value)
    return {"valid": True, "collections": len(nodes)}
