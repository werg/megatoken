"""Independent Number conversion with explicit method and continuation checks."""

from __future__ import annotations

import copy

from . import algorithm_v7 as scalar

SCHEMA = "semantic-number-conversion-state/1"
FIELDS = {
    "number.begin": {"kind", "id", "value"},
    "number.select": {"kind", "id", "key", "method", "builtin"},
    "number.call": {"kind", "id", "child"},
    "number.builtin": {"kind", "id", "result"},
    "number.return": {"kind", "id", "value"},
    "number.finish": {"kind", "id", "result"},
}
OPERATION_FIELDS = {
    "id",
    "parent",
    "input",
    "index",
    "method",
    "builtin",
    "child",
    "phase",
    "primitive",
    "result",
}
PRIMITIVES = {"undefined", "null", "bool", "str", "num", "bigint", "symbol"}
NUMERIC = PRIMITIVES - {"symbol"}
BUILTINS = {None, "Object.valueOf", "Object.toString"}
PHASES = {"select", "call", "builtin", "waiting", "convert", "complete"}


def _protocol():
    from . import protocol_v8

    return protocol_v8


def _require(condition, message):
    if not condition:
        _protocol()._fail("Number conversion: " + message)


def _primitive(value):
    return isinstance(value, list) and bool(value) and value[0] in PRIMITIVES


def _numeric(value):
    return isinstance(value, list) and bool(value) and value[0] in NUMERIC


def scalar_number(value):
    """Number accepts BigInt explicitly; ordinary numeric coercion does not."""
    _require(_numeric(value), "scalar requires a separately modeled abrupt conversion")
    if value[0] in ("str", "bigint"):
        _require(len(value[1]) <= 10000, "scalar work bound")
    if value[0] == "bigint":
        number = 0.0 if value[1] in ("0", "-0") else float(value[1])
    else:
        number = scalar._to_number(value)
    return _protocol()._normalize_number(scalar._encode_scalar(float(number)))


def validate_record(record):
    kind = record.get("kind") if type(record) is dict else None
    if kind not in FIELDS:
        return False
    p = _protocol()
    p._exact(record, FIELDS[kind], "Number record")
    p._identity(record["id"])
    if kind == "number.call":
        p._identity(record["child"])
    elif kind == "number.select":
        p.validate_protocol_key(record["key"])
        p.validate_protocol_value(record["method"])
        _require(record["builtin"] in BUILTINS, "builtin identity")
    else:
        p.validate_protocol_value(
            record["value"] if kind in ("number.begin", "number.return") else record["result"]
        )
    return True


def _state(machine):
    state = machine["protocol"].get("numberConversions")
    if state is None:
        state = {"schema": SCHEMA, "operations": []}
        machine["protocol"]["numberConversions"] = state
    return state


def _lookup(machine, value, key):
    p = _protocol()
    current = value[1]
    seen = set()
    while True:
        _require(current not in seen, "cyclic prototype")
        seen.add(current)
        node = p._node(machine, current)
        descriptor = p._descriptor(node, p._key(key))
        if descriptor is not None:
            _require(descriptor["kind"] == "data", "accessor requires its getter continuation")
            return descriptor["value"], None
        prototype = node.get("prototype", "ordinary")
        if prototype is None:
            return ["undefined"], None
        if prototype == "ordinary":
            if key == ["str", "valueOf"]:
                return ["undefined"], "Object.valueOf"
            if key == ["str", "toString"]:
                _require(node["kind"] != "array", "array stringification requires its own protocol")
                return ["undefined"], "Object.toString"
            return ["undefined"], None
        p._known(machine, prototype)
        _require(prototype[0] == "ref", "prototype identity")
        current = prototype[1]


def _accept(row, value):
    if _primitive(value):
        row["primitive"] = copy.deepcopy(value)
        row["phase"] = "convert"
    else:
        _require(row["index"] == 0, "object result requires an abrupt conversion record")
        row.update(index=1, method=None, builtin=None, child=None, phase="select")


def assert_next(machine, record):
    operations = (machine.get("protocol", {}).get("numberConversions") or {}).get("operations", [])
    row = next((entry for entry in reversed(operations) if entry["phase"] != "complete"), None)
    if row is None:
        return True
    if row["phase"] == "waiting" and row["child"] in machine["stack"]:
        return True
    _require(bool(machine["stack"]) and machine["stack"][-1] == row["parent"], "continuation owner")
    expected = {
        "select": "number.select",
        "call": "number.call",
        "builtin": "number.builtin",
        "waiting": "number.return",
        "convert": "number.finish",
    }[row["phase"]]
    _require(
        record["kind"] == expected and record.get("id") == row["id"],
        "pending continuation requires " + expected,
    )
    return True


def apply_record(machine, record):
    if not validate_record(record):
        return False
    p = _protocol()
    if record["kind"] == "number.begin":
        state = _state(machine)
        value = p._known(machine, record["value"])
        _require(
            len(state["operations"]) < 4096
            and all(row["id"] != record["id"] for row in state["operations"]),
            "identity or work bound",
        )
        _require(bool(machine["stack"]), "active invocation required")
        primitive = _primitive(value)
        if not primitive:
            _require(
                value[0] == "ref" and p._node(machine, value[1])["kind"] in ("object", "array"),
                "ordinary input object required",
            )
        state["operations"].append(
            {
                "id": record["id"],
                "parent": machine["stack"][-1],
                "input": copy.deepcopy(value),
                "index": -1,
                "method": None,
                "builtin": None,
                "child": None,
                "phase": "convert" if primitive else "select",
                "primitive": copy.deepcopy(value) if primitive else None,
                "result": None,
            }
        )
        return True
    row = next(
        (entry for entry in _state(machine)["operations"] if entry["id"] == record["id"]), None
    )
    _require(row is not None, "unknown operation")
    _require(
        bool(machine["stack"]) and machine["stack"][-1] == row["parent"], "record outside caller"
    )
    kind = record["kind"]
    if kind == "number.select":
        key = (
            ["symbol", "well-known:toPrimitive"]
            if row["index"] == -1
            else ["str", "valueOf" if row["index"] == 0 else "toString"]
        )
        _require(row["phase"] == "select" and record["key"] == key, "method order")
        if key[0] == "symbol":
            p._symbol(machine, key[1])
        p._known(machine, record["method"])
        selected = _lookup(machine, row["input"], key)
        _require(selected == (record["method"], record["builtin"]), "false method selection")
        row["method"], row["builtin"] = copy.deepcopy(record["method"]), record["builtin"]
        if row["builtin"] is not None:
            row["phase"] = "builtin"
        elif row["index"] == -1 and row["method"][0] in ("undefined", "null"):
            row.update(index=0, method=None, phase="select")
        elif row["method"][0] == "ref" and p._node(machine, row["method"][1])["kind"] == "closure":
            row["phase"] = "call"
        else:
            _require(row["index"] == 0, "noncallable method requires an abrupt conversion record")
            row.update(index=1, method=None, phase="select")
    elif kind == "number.call":
        _require(row["phase"] == "call", "captured callable required")
        from .construction_v8 import _call

        _call(
            machine,
            record["child"],
            row["method"],
            row["input"],
            [["str", "number"]] if row["index"] == -1 else [],
        )
        row["child"], row["phase"] = record["child"], "waiting"
    elif kind == "number.builtin":
        _require(row["phase"] == "builtin", "captured intrinsic required")
        expected = row["input"]
        if row["builtin"] == "Object.toString":
            tag, _ = _lookup(machine, row["input"], ["symbol", "well-known:toStringTag"])
            node = p._node(machine, row["input"][1])
            name = tag[1] if tag[0] == "str" else "Array" if node["kind"] == "array" else "Object"
            expected = ["str", "[object " + name + "]"]
        _require(record["result"] == expected, "false builtin result")
        _accept(row, expected)
    elif kind == "number.return":
        returned = machine.get("lastReturn")
        _require(
            row["phase"] == "waiting"
            and row["child"] not in machine["stack"]
            and returned is not None
            and returned["kind"] == "return"
            and returned["value"] == record["value"],
            "callback completion differs",
        )
        p._known(machine, record["value"])
        _accept(row, record["value"])
    else:
        _require(row["phase"] == "convert", "primitive conversion phase required")
        expected = scalar_number(row["primitive"])
        p._known(machine, record["result"])
        _require(record["result"] == expected, "false scalar result")
        row["result"], row["phase"] = copy.deepcopy(expected), "complete"
        machine["lastReturn"] = {"kind": "return", "value": copy.deepcopy(expected)}
    return True


def validate_state(machine):
    state = machine.get("protocol", {}).get("numberConversions")
    if state is None:
        return True
    p = _protocol()
    p._exact(state, {"schema", "operations"}, "Number state")
    _require(
        state["schema"] == SCHEMA
        and isinstance(state["operations"], list)
        and len(state["operations"]) <= 4096,
        "state schema or bound",
    )
    seen = set()
    for row in state["operations"]:
        p._exact(row, OPERATION_FIELDS, "Number continuation")
        p._identity(row["id"])
        p._identity(row["parent"])
        _require(row["id"] not in seen, "duplicate operation")
        seen.add(row["id"])
        _require(
            row["parent"] in machine["frameFunctions"]
            and type(row["index"]) is int
            and row["index"] in (-1, 0, 1)
            and row["phase"] in PHASES,
            "phase or owner",
        )
        p._known(machine, row["input"])
        for value in (row["method"], row["primitive"], row["result"]):
            if value is not None:
                p._known(machine, value)
        if row["primitive"] and row["primitive"][0] in ("str", "bigint"):
            _require(len(row["primitive"][1]) <= 10000, "retained scalar bound")
        _require(row["builtin"] in BUILTINS, "retained builtin")
        if row["child"] is not None:
            p._identity(row["child"])
            _require(row["child"] in machine["frameFunctions"], "callback identity")
        if _primitive(row["input"]):
            _require(
                row["index"] == -1
                and row["method"] is None
                and row["builtin"] is None
                and row["child"] is None
                and row["primitive"] == row["input"]
                and row["phase"] in ("convert", "complete"),
                "primitive entry state",
            )
        if row["phase"] == "select":
            _require(
                row["method"] is None and row["builtin"] is None and row["child"] is None,
                "pending selection state",
            )
        if row["phase"] in ("call", "builtin"):
            _require(row["child"] is None, "unentered callback identity")
        if row["phase"] in ("select", "call", "builtin", "waiting"):
            _require(
                row["input"][0] == "ref" and row["primitive"] is None and row["result"] is None,
                "object phase values",
            )
        if row["phase"] in ("call", "waiting"):
            _require(
                row["method"] is not None
                and row["method"][0] == "ref"
                and p._node(machine, row["method"][1])["kind"] == "closure"
                and row["builtin"] is None,
                "captured callable",
            )
        if row["phase"] == "waiting":
            _require(
                row["child"] is not None
                and machine["frameFunctions"][row["child"]]
                == p._node(machine, row["method"][1])["code"],
                "callback source identity",
            )
            if row["child"] in machine["stack"]:
                _require(
                    row["parent"] in machine["stack"]
                    and machine["stack"].index(row["child"])
                    == machine["stack"].index(row["parent"]) + 1,
                    "callback stack",
                )
        if row["phase"] == "builtin":
            _require(
                row["builtin"] is not None and row["method"] == ["undefined"], "intrinsic state"
            )
        if row["phase"] in ("convert", "complete"):
            _require(_primitive(row["primitive"]), "primitive result absent")
        if row["phase"] == "complete":
            _require(
                _numeric(row["primitive"]) and row["result"] == scalar_number(row["primitive"]),
                "false completed result",
            )
        else:
            _require(row["result"] is None, "unfinished result")
    return True
