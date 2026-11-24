"""Independent native replay for the bounded builtin iterator profile."""

from __future__ import annotations

import copy

FIELDS = {
    "iterator.lookup": {"kind", "receiver", "key", "method"},
    "iterator.create": {"kind", "name", "iterated", "iterationKind"},
    "iterator.next": {"kind", "object", "name", "entry", "result", "done"},
    "iterator.identity": {"kind", "object", "result"},
    "iterator.read": {"kind", "object", "key", "result"},
    "iterator.arrayGuard": {"kind", "iterated", "result"},
    "iterator.spread": {"kind", "iterated", "values"},
}
ARRAY_KINDS = {"values", "keys", "entries"}
COLLECTION_KINDS = {"set-values", "set-keys", "set-entries", "map-keys", "map-values", "map-entries"}
KINDS = ARRAY_KINDS | COLLECTION_KINDS | {"string"}
METHODS = {
    "Array.prototype.values",
    "Array.prototype.keys",
    "Array.prototype.entries",
    "String.prototype.iterator",
    "Iterator.next",
    "Iterator.identity",
    "Set.prototype.values",
    "Set.prototype.keys",
    "Set.prototype.entries",
    "Map.prototype.keys",
    "Map.prototype.values",
    "Map.prototype.entries",
}
BOUND = 10000


def _fail(message):
    from .protocol_v8 import ProtocolV8Error

    raise ProtocolV8Error("Invalid builtin iterator protocol: " + message)


def _validate_value(value):
    from .protocol_v8 import validate_protocol_value

    return validate_protocol_value(value)


def _identity(value, allow_none=False):
    if allow_none and value is None:
        return
    if not isinstance(value, str) or not value or len(value) > 256:
        _fail("identity")


def _key(value):
    if (
        not isinstance(value, list)
        or len(value) != 2
        or value[0] not in ("str", "symbol")
        or not isinstance(value[1], str)
    ):
        _fail("iterator property key")
    if value[0] == "symbol":
        _identity(value[1])


def validate_record(record):
    kind = record.get("kind") if type(record) is dict else None
    fields = FIELDS.get(kind)
    if fields is None:
        return False
    if set(record) != fields:
        _fail("record fields")
    for field in ("object", "name", "entry"):
        if field in record:
            _identity(record[field], field == "entry")
    if "key" in record:
        _key(record["key"])
    if kind == "iterator.lookup":
        _validate_value(record["receiver"])
        if record["method"] not in METHODS:
            _fail("iterator builtin")
    elif kind == "iterator.create":
        _identity(record["name"])
        _validate_value(record["iterated"])
        if record["iterationKind"] not in KINDS:
            _fail("iteration kind")
    elif kind == "iterator.arrayGuard":
        _validate_value(record["iterated"])
        if type(record["result"]) is not bool:
            _fail("array guard boolean")
    elif kind == "iterator.spread":
        _validate_value(record["iterated"])
        values = record["values"]
        if not isinstance(values, list) or len(values) > BOUND:
            _fail("spread values bound")
        for item in values:
            _validate_value(item)
    else:
        _validate_value(record["result"])
        if kind == "iterator.next" and type(record["done"]) is not bool:
            _fail("done boolean")
    return True


def _tagged_key(key):
    return key[1] if key[0] == "str" else ("symbol", key[1])


def _descriptor(node, key):
    from .protocol_v8 import _descriptor as get

    return get(node, _tagged_key(key))


def _known(machine, value):
    from .protocol_v8 import _known as known

    return known(machine, value)


def _standard_array(machine, value):
    _known(machine, value)
    node = machine["heap"].get(value[1]) if value[0] == "ref" else None
    if not node or node.get("kind") != "array" or node.get("prototype", "ordinary") != "ordinary":
        return False
    return _descriptor(node, ["symbol", "well-known:iterator"]) is None


def _selected_method(machine, receiver, key):
    _known(machine, receiver)
    if key[0] == "symbol" and not any(
        row["id"] == key[1] for row in machine["protocol"]["symbols"]
    ):
        _fail("unknown iterator Symbol key")
    if receiver[0] == "str":
        if key != ["symbol", "well-known:iterator"]:
            _fail("string iterator lookup key")
        return "String.prototype.iterator"
    if receiver[0] != "ref":
        _fail("iterator lookup receiver")
    node = machine["heap"].get(receiver[1])
    if node and node.get("kind") == "array":
        if node.get("prototype", "ordinary") != "ordinary" or _descriptor(node, key) is not None:
            _fail("array iterator is overridden")
        if key == ["symbol", "well-known:iterator"]:
            return "Array.prototype.values"
        if key[0] == "str" and key[1] in ("values", "keys", "entries"):
            return "Array.prototype." + key[1]
        _fail("unsupported array iterator lookup")
    node = machine["heap"].get(receiver[1])
    if node and node.get("kind") in ("set", "map"):
        kind = node["kind"]
        allowed = ("values", "keys", "entries") if kind == "set" else ("keys", "values", "entries")
        if key == ["symbol", "well-known:iterator"]:
            return "Set.prototype.values" if kind == "set" else "Map.prototype.entries"
        if key[0] == "str" and key[1] in allowed:
            if kind == "set" and key[1] == "keys":
                return "Set.prototype.values"
            return ("Set" if kind == "set" else "Map") + ".prototype." + key[1]
        _fail(kind + " iterator lookup key")
    if node and node.get("kind") == "iterator":
        if key == ["str", "next"]:
            return "Iterator.next"
        if key == ["symbol", "well-known:iterator"]:
            return "Iterator.identity"
    _fail("iterator lookup receiver kind")


def _array_length(machine, identity):
    node = machine["heap"].get(identity)
    if not node or node.get("kind") != "array":
        _fail("iterator source array required")
    descriptor = _descriptor(node, ["str", "length"])
    value = descriptor.get("value") if descriptor and descriptor.get("kind") == "data" else None
    if (
        not isinstance(value, list)
        or value[0] != "num"
        or type(value[1]) is not int
        or not 0 <= value[1] <= BOUND
    ):
        _fail("bounded array length")
    return value[1]


def _read_index(machine, identity, index):
    from .protocol_v8 import _lookup_descriptor

    _, descriptor = _lookup_descriptor(machine, identity, str(index))
    if descriptor is None:
        return ["undefined"]
    if descriptor["kind"] != "data":
        if descriptor["get"][0] != "undefined":
            _fail("iterator index getter requires scoped call record")
        return ["undefined"]
    return copy.deepcopy(descriptor["value"])


def _spread_values(machine, iterable):
    _known(machine, iterable)
    if iterable[0] == "str":
        units = _utf16_units(iterable[1])
        result = []
        index = 0
        while index < len(units):
            first = int.from_bytes(units[index], "little")
            width = (
                2
                if 0xD800 <= first <= 0xDBFF
                and index + 1 < len(units)
                and 0xDC00 <= int.from_bytes(units[index + 1], "little") <= 0xDFFF
                else 1
            )
            result.append(["str", _utf16_slice(units, index, width)])
            index += width
        if len(result) > BOUND:
            _fail("spread element bound")
        return result
    if not _standard_array(machine, iterable):
        _fail("spread requires a bounded ordinary array or primitive string")
    size = _array_length(machine, iterable[1])
    array = machine["heap"].get(iterable[1])
    result = []
    for index in range(size):
        descriptor = _descriptor(array, ["str", str(index)])
        if descriptor is None or descriptor.get("kind") != "data":
            _fail("sparse/accessor spread requires its own iterator profile")
        _known(machine, descriptor["value"])
        result.append(copy.deepcopy(descriptor["value"]))
    return result


def _utf16_units(text):
    raw = text.encode("utf-16-le", "surrogatepass")
    return [raw[i : i + 2] for i in range(0, len(raw), 2)]


def _utf16_slice(units, start, width):
    raw = b"".join(units[start : start + width])
    return raw.decode("utf-16-le", "surrogatepass")


def _allocate(machine, record):
    from .protocol_v8 import _apply

    _apply(
        machine,
        {
            "kind": "allocate",
            "name": record["name"],
            "objectKind": record["objectKind"],
            "length": record["length"],
            "properties": record["properties"],
            "code": None,
            "captures": [],
            "functionKind": None,
            "lexicalReceiver": None,
        },
    )


def apply_record(machine, record):
    if not validate_record(record):
        return False
    kind = record["kind"]
    if kind == "iterator.lookup":
        if _selected_method(machine, record["receiver"], record["key"]) != record["method"]:
            _fail("false builtin iterator lookup")
    elif kind == "iterator.arrayGuard":
        if _standard_array(machine, record["iterated"]) != record["result"]:
            _fail("false array iterator guard")
    elif kind == "iterator.spread":
        if _spread_values(machine, record["iterated"]) != record["values"]:
            _fail("spread snapshot differs from its live iterable")
    elif kind == "iterator.create":
        _known(machine, record["iterated"])
        if record["name"] in machine["heap"] or any(
            row["id"] == record["name"] for row in machine["protocol"]["symbols"]
        ):
            _fail("iterator identity collision")
        if record["iterationKind"] == "string":
            if record["iterated"][0] != "str" or len(_utf16_units(record["iterated"][1])) > BOUND:
                _fail("bounded primitive string iterator")
        elif record["iterationKind"] in ARRAY_KINDS:
            if record["iterated"][0] != "ref":
                _fail("array iterator source")
            _array_length(machine, record["iterated"][1])
        else:
            if record["iterated"][0] != "ref":
                _fail("collection iterator source")
            collection = machine["heap"].get(record["iterated"][1])
            expected_kind = record["iterationKind"].split("-", 1)[0]
            if (
                expected_kind not in ("set", "map")
                or not collection
                or collection.get("kind") != expected_kind
                or not isinstance(collection.get("entries"), list)
                or len(collection["entries"]) > BOUND
            ):
                _fail("collection iterator kind/registry")
        machine["heap"][record["name"]] = {
            "kind": "iterator",
            "iterated": copy.deepcopy(record["iterated"]),
            "iterationKind": record["iterationKind"],
            "nextIndex": 0,
            "done": False,
            "properties": {},
        }
    else:
        node = machine["heap"].get(record["object"])
        if not node or node.get("kind") != "iterator":
            _fail("builtin iterator object")
        if kind == "iterator.identity":
            if record["result"] != ["ref", record["object"]]:
                _fail("iterator identity changed")
        elif kind == "iterator.read":
            _key(record["key"])
            expected = ["undefined"]
            if record["key"] == ["symbol", "well-known:toStringTag"]:
                iterator_kind = node["iterationKind"]
                expected = [
                    "str",
                    "String Iterator" if iterator_kind == "string" else
                    "Set Iterator" if iterator_kind.startswith("set-") else
                    "Map Iterator" if iterator_kind.startswith("map-") else "Array Iterator",
                ]
            elif (
                record["key"][0] == "str"
                and record["key"][1] in ("next", "constructor", "__proto__", "toString", "valueOf")
                or record["key"] == ["symbol", "well-known:iterator"]
            ):
                _fail("iterator inherited property needs explicit operation")
            if record["result"] != expected:
                _fail("false iterator property read")
        else:
            index = node["nextIndex"]
            if type(index) is not int or not 0 <= index <= BOUND:
                _fail("iterator cursor bound")
            value = ["undefined"]
            if not node["done"]:
                if node["iterationKind"] == "string":
                    units = _utf16_units(node["iterated"][1])
                    if index >= len(units):
                        node["done"] = True
                    else:
                        first = int.from_bytes(units[index], "little")
                        width = (
                            2
                            if 0xD800 <= first <= 0xDBFF
                            and index + 1 < len(units)
                            and 0xDC00 <= int.from_bytes(units[index + 1], "little") <= 0xDFFF
                            else 1
                        )
                        value = ["str", _utf16_slice(units, index, width)]
                        node["nextIndex"] += width
                elif node["iterationKind"] in ARRAY_KINDS:
                    length = _array_length(machine, node["iterated"][1])
                    if index >= length:
                        node["done"] = True
                    else:
                        node["nextIndex"] += 1
                        value = (
                            ["num", index]
                            if node["iterationKind"] == "keys"
                            else _read_index(machine, node["iterated"][1], index)
                        )
                        if node["iterationKind"] == "entries":
                            if record["entry"] is None:
                                _fail("entry iterator requires pair allocation")
                            _allocate(
                                machine,
                                {
                                    "name": record["entry"],
                                    "objectKind": "array",
                                    "length": 2,
                                    "properties": [
                                        {"key": ["str", "0"], "value": ["num", index]},
                                        {"key": ["str", "1"], "value": value},
                                    ],
                                },
                            )
                            value = ["ref", record["entry"]]
                else:
                    collection = machine["heap"].get(node["iterated"][1])
                    entries = collection.get("entries") if collection else None
                    if not isinstance(entries, list):
                        _fail("collection iterator source")
                    position = index
                    while position < len(entries) and entries[position] is None:
                        position += 1
                    if position >= len(entries):
                        node["nextIndex"] = position
                        node["done"] = True
                    else:
                        raw = entries[position]
                        node["nextIndex"] = position + 1
                        if node["iterationKind"] in ("set-values", "set-keys"):
                            value = copy.deepcopy(raw)
                        elif node["iterationKind"] == "map-keys":
                            value = copy.deepcopy(raw[0])
                        elif node["iterationKind"] == "map-values":
                            value = copy.deepcopy(raw[1])
                        else:
                            pair_values = [copy.deepcopy(raw), copy.deepcopy(raw)] if node["iterationKind"] == "set-entries" else [copy.deepcopy(raw[0]), copy.deepcopy(raw[1])]
                            if record["entry"] is None:
                                _fail("collection entries iterator requires pair allocation")
                            _allocate(
                                machine,
                                {
                                    "name": record["entry"],
                                    "objectKind": "array",
                                    "length": 2,
                                    "properties": [
                                        {"key": ["str", "0"], "value": pair_values[0]},
                                        {"key": ["str", "1"], "value": pair_values[1]},
                                    ],
                                },
                            )
                            value = ["ref", record["entry"]]
            if record["entry"] is not None and (
                node["done"] or node["iterationKind"] not in COLLECTION_KINDS | {"entries"}
            ):
                _fail("unexpected iterator entry identity")
            if record["done"] != node["done"] or record["result"] != value:
                _fail("false iterator next result")
            _allocate(
                machine,
                {
                    "name": record["name"],
                    "objectKind": "object",
                    "length": None,
                    "properties": [
                        {"key": ["str", "value"], "value": value},
                        {"key": ["str", "done"], "value": ["bool", node["done"]]},
                    ],
                },
            )
    validate_state(machine)
    return True


def validate_state(machine):
    count = 0
    for identity, node in machine["heap"].items():
        if node.get("kind") != "iterator":
            continue
        count += 1
        if (
            set(node) != {"kind", "iterated", "iterationKind", "nextIndex", "done", "properties"}
            or node["iterationKind"] not in KINDS
            or type(node["nextIndex"]) is not int
            or not 0 <= node["nextIndex"] <= BOUND
            or type(node["done"]) is not bool
            or node["properties"] != {}
        ):
            _fail("iterator state exact fields")
        _known(machine, node["iterated"])
        if node["iterationKind"] == "string":
            if (
                node["iterated"][0] != "str"
                or len(_utf16_units(node["iterated"][1])) > BOUND
                or node["nextIndex"] > len(_utf16_units(node["iterated"][1]))
            ):
                _fail("string iterator cursor")
            units = _utf16_units(node["iterated"][1])
            index = node["nextIndex"]
            if (
                0 < index < len(units)
                and 0xD800 <= int.from_bytes(units[index - 1], "little") <= 0xDBFF
                and 0xDC00 <= int.from_bytes(units[index], "little") <= 0xDFFF
            ):
                _fail("surrogate-splitting iterator cursor")
            if node["done"] and index != len(units):
                _fail("completed string cursor")
        elif node["iterationKind"] in ARRAY_KINDS:
            if node["iterated"][0] != "ref" or node["iterated"][1] == identity:
                _fail("array iterator source identity")
            size = _array_length(machine, node["iterated"][1])
            if node["nextIndex"] > size:
                _fail("array iterator cursor")
        else:
            if node["iterated"][0] != "ref" or node["iterated"][1] == identity:
                _fail("collection iterator source identity")
            collection = machine["heap"].get(node["iterated"][1])
            expected_kind = node["iterationKind"].split("-", 1)[0]
            if (
                expected_kind not in ("set", "map")
                or not collection
                or collection.get("kind") != expected_kind
                or not isinstance(collection.get("entries"), list)
                or node["nextIndex"] > len(collection["entries"])
            ):
                _fail("collection iterator cursor")
    return {"valid": True, "iterators": count}
