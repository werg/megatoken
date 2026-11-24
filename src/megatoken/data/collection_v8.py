"""Independent bounded Set/Map transitions for issue51 collection semantics."""

from __future__ import annotations

import copy
import math

from . import semantic_primitives as primitives

FIELDS = {
    "set.lookup": {"kind", "object", "key", "method"},
    "set.create": {"kind", "name", "iterable"},
    "set.size": {"kind", "object", "result"},
    "set.add": {"kind", "object", "value", "result"},
    "set.has": {"kind", "object", "value", "result"},
    "set.delete": {"kind", "object", "value", "result"},
    "set.clear": {"kind", "object", "result"},
    "map.lookup": {"kind", "object", "key", "method"},
    "map.create": {"kind", "name", "iterable"},
    "map.size": {"kind", "object", "result"},
    "map.get": {"kind", "object", "key", "result"},
    "map.has": {"kind", "object", "key", "result"},
    "map.set": {"kind", "object", "key", "value", "result"},
    "map.delete": {"kind", "object", "key", "result"},
    "map.clear": {"kind", "object", "result"},
}
STATE_SCHEMA = "semantic-collections-state/1"
BOUND = 10000


def _fail(message):
    from .protocol_v8 import ProtocolV8Error

    raise ProtocolV8Error("Invalid collection protocol: " + message)


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
    if kind in ("set.lookup", "map.lookup"):
        key = record["key"]
        method_names = ("add", "has", "delete", "clear") if kind == "set.lookup" else (
            "get",
            "set",
            "has",
            "delete",
            "clear",
        )
        prefix = "Set" if kind == "set.lookup" else "Map"
        if (
            not isinstance(key, list)
            or len(key) != 2
            or key[0] != "str"
            or key[1] not in method_names
            or record["method"] != prefix + ".prototype." + key[1]
        ):
            _fail("collection method identity")
    elif kind in ("set.create", "map.create"):
        _identity(record["name"])
        _value(record["iterable"])
        if record["iterable"][0] not in ("ref", "null", "undefined"):
            _fail("constructor iterable tag")
    elif kind in ("set.size", "map.size"):
        _value(record["result"])
        if record["result"][0] != "num":
            _fail("size result")
        size = primitives._decode_scalar(record["result"])
        if not math.isfinite(size) or not float(size).is_integer() or size < 0:
            _fail("size result")
    else:
        if kind in ("set.add", "set.has", "set.delete"):
            _value(record["value"])
        if kind in ("map.get", "map.has", "map.set", "map.delete"):
            _value(record["key"])
        if kind == "map.set":
            _value(record["value"])
        _value(record["result"])
        if kind in ("set.add", "map.set") and record["result"][0] != "ref":
            _fail("collection mutator receiver identity")
        if kind in ("set.has", "set.delete", "map.has", "map.delete") and record["result"][0] != "bool":
            _fail("collection boolean result")
        if kind in ("set.clear", "map.clear") and record["result"] != ["undefined"]:
            _fail("collection clear result")
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
    if value[0] == "num" and primitives._decode_scalar(value) == 0:
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
    if (
        not isinstance(size, (int, float))
        or not math.isfinite(size)
        or not float(size).is_integer()
        or not 0 <= size <= BOUND
    ):
        _fail("constructor array length bound")
    size = int(size)
    entries = []
    for index in range(size):
        item = array.get("properties", {}).get(str(index))
        if item is None:
            _fail("sparse collection constructor iterable is unsupported")
        _known(machine, item)
        entries.append(copy.deepcopy(item))
    return entries


def _set_entries(machine, value):
    result = []
    for item in _array_values(machine, value):
        if not any(_same_value_zero(item, previous) for previous in result):
            result.append(_normal(item))
    return result


def _map_entries(machine, value):
    result = []
    for pair_tag in _array_values(machine, value):
        pair = _array_values(machine, pair_tag)
        if len(pair) != 2:
            _fail("Map constructor entry must contain exactly two dense values")
        key, item = pair
        index = next((i for i, row in enumerate(result) if _same_value_zero(row[0], key)), None)
        if index is None:
            result.append([_normal(key), copy.deepcopy(item)])
        else:
            result[index][1] = copy.deepcopy(item)
    return result


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


def _node(machine, identity, kind):
    node = machine["heap"].get(identity)
    if not node or node.get("kind") != kind or not isinstance(node.get("entries"), list):
        _fail(("Set" if kind == "set" else "Map") + " receiver")
    return node


def apply_record(machine, record):
    if not _validate_record(record):
        return False
    kind_name = record["kind"].split(".", 1)[0]
    kind = "set" if kind_name == "set" else "map"
    action = record["kind"].split(".", 1)[1]
    if action == "lookup":
        _node(machine, record["object"], kind)
        if record["method"] != ("Set" if kind == "set" else "Map") + ".prototype." + record["key"][1]:
            _fail("false collection method selection")
    elif action == "create":
        identity = record["name"]
        if identity in machine["heap"] or any(
            row["id"] == identity for row in machine["protocol"]["symbols"]
        ):
            _fail("collection identity collision")
        entries = _set_entries(machine, record["iterable"]) if kind == "set" else _map_entries(machine, record["iterable"])
        machine["heap"][identity] = {"kind": kind, "entries": entries, "properties": {}}
        rows = _registry(machine)
        if len(rows) >= 4096 or any(row.get("id") == identity for row in rows):
            _fail("collection registry bound or duplicate")
        rows.append({"id": identity, "kind": kind})
    else:
        node = _node(machine, record["object"], kind)
        live = [entry for entry in node["entries"] if entry is not None]
        size = len(live)
        if action == "size":
            _known(machine, record["result"])
            if _same_value_zero(record["result"], ["num", size]) is False:
                _fail("false collection size")
        elif action == "clear":
            _known(machine, record["result"])
            if record["result"] != ["undefined"]:
                _fail("false collection clear result")
            node["entries"] = [None for _ in node["entries"]]
        else:
            lookup = record["value"] if kind == "set" else record["key"]
            _known(machine, lookup)
            index = next(
                (
                    i
                    for i, entry in enumerate(node["entries"])
                    if entry is not None
                    and _same_value_zero(entry if kind == "set" else entry[0], lookup)
                ),
                None,
            )
            if action in ("has", "delete"):
                _known(machine, record["result"])
                if record["result"] != ["bool", index is not None]:
                    _fail("false collection membership/deletion result")
                if action == "delete" and index is not None:
                    node["entries"][index] = None
            elif kind == "set" and action == "add":
                _known(machine, record["result"])
                if record["result"] != ["ref", record["object"]]:
                    _fail("Set.add receiver result")
                if index is None:
                    if size >= BOUND or len(node["entries"]) >= BOUND:
                        _fail("Set size/entry bound")
                    node["entries"].append(_normal(record["value"]))
            elif kind == "map" and action == "get":
                _known(machine, record["result"])
                expected = ["undefined"] if index is None else node["entries"][index][1]
                if record["result"] != expected:
                    _fail("false Map.get result")
            elif kind == "map" and action == "set":
                _known(machine, record["value"])
                _known(machine, record["result"])
                if record["result"] != ["ref", record["object"]]:
                    _fail("Map.set receiver result")
                if index is None:
                    if size >= BOUND or len(node["entries"]) >= BOUND:
                        _fail("Map size/entry bound")
                    node["entries"].append([_normal(record["key"]), copy.deepcopy(record["value"])])
                else:
                    node["entries"][index][1] = copy.deepcopy(record["value"])
            else:
                _fail("unknown collection operation")
    validate_state(machine, machine["protocol"].get("collections"))
    return True


def validate_state(machine, snapshot=None):
    if snapshot is None:
        snapshot = machine.get("protocol", {}).get("collections")
    nodes = {
        identity: node
        for identity, node in machine["heap"].items()
        if node.get("kind") in ("set", "map")
    }
    if snapshot is None:
        if nodes:
            _fail("collection nodes lack a retained allocation registry")
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
        if type(row) is not dict or set(row) != {"id", "kind"} or row["kind"] not in ("set", "map"):
            _fail("collection registry row shape")
        _identity(row["id"])
        if row["id"] in rows:
            _fail("duplicate collection registry row")
        rows[row["id"]] = row["kind"]
    if len(rows) != len(nodes):
        _fail("collection registry/node count")
    for identity, node in nodes.items():
        entries = node.get("entries")
        kind = node["kind"]
        if (
            rows.get(identity) != kind
            or set(node) != {"kind", "entries", "properties"}
            or node["properties"] != {}
            or not isinstance(entries, list)
            or len(entries) > BOUND
        ):
            _fail("collection heap shape or allocation registry")
        seen = []
        for entry in entries:
            if entry is None:
                continue
            if kind == "set":
                _known(machine, entry)
                key = entry
            else:
                if not isinstance(entry, list) or len(entry) != 2:
                    _fail("Map entry shape")
                _known(machine, entry[0])
                _known(machine, entry[1])
                key = entry[0]
            if any(_same_value_zero(key, previous) for previous in seen):
                _fail("duplicate collection key")
            seen.append(key)
    return {"valid": True, "collections": len(nodes)}
