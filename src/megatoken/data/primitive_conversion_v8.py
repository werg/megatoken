"""Independent shared Number/PropertyKey conversion and continuation checks.

Lookups are derived from the native heap. Getters and methods execute only as
checked guest calls; scalar leaves do not execute JavaScript or trust the producer.
"""

from __future__ import annotations

import copy

from . import semantic_primitives as scalar
from .number_v8 import scalar_number
from .semantic_json import _js_string_length, stable_algorithm_json

SCHEMA = "semantic-primitive-conversion-state/1"
FIELDS = {
    "conversion.begin": {"kind", "id", "goal", "value"},
    "conversion.lookup": {"kind", "id", "key", "descriptor", "value", "builtin"},
    "conversion.getter.call": {"kind", "id", "child"},
    "conversion.getter.return": {"kind", "id", "value"},
    "conversion.call": {"kind", "id", "child"},
    "conversion.return": {"kind", "id", "value"},
    "conversion.builtin": {"kind", "id", "result"},
    "conversion.throw": {"kind", "id", "value"},
    "conversion.fault": {"kind", "id", "error"},
    "conversion.finish": {"kind", "id", "result"},
}
OPERATION_FIELDS = {
    "id",
    "parent",
    "goal",
    "input",
    "index",
    "phase",
    "lookupKind",
    "method",
    "builtin",
    "child",
    "candidate",
    "result",
    "fault",
}
PHASES = {
    "lookup",
    "getter-call",
    "getter-waiting",
    "call",
    "waiting",
    "builtin",
    "finish",
    "fault",
    "complete",
    "abrupt",
}
FAULTS = {
    "symbol-to-number": "Cannot convert a Symbol value to a number",
    "exotic-not-callable": "Symbol.toPrimitive is not callable",
    "exotic-nonprimitive": "Symbol.toPrimitive must return a primitive",
    "ordinary-exhausted": "Cannot convert object to primitive value",
}
BUILTINS = {"Object.valueOf", "Object.toString", "Array.toString"}
GOALS = {"Number", "PropertyKey"}
DESCRIPTORS = {"missing", "data", "accessor", "builtin"}
PRIMITIVES = {"undefined", "null", "bool", "str", "num", "bigint", "symbol"}


def _protocol():
    from . import protocol_v8

    return protocol_v8


def _require(condition, message):
    if not condition:
        _protocol()._fail("Primitive conversion: " + message)


def _eq(left, right):
    return stable_algorithm_json(left) == stable_algorithm_json(right)


def _identity(value):
    _protocol()._identity(value)
    _require(_js_string_length(value) <= 256, "conversion identity bound")


def _primitive(value):
    return isinstance(value, list) and bool(value) and value[0] in PRIMITIVES


def _scalar_bound(value):
    if value is not None and value[0] in ("str", "bigint"):
        _require(_js_string_length(value[1]) <= 10000, "primitive scalar conversion bound")


def _terminal(row):
    return row["phase"] in ("complete", "abrupt")


def _callable(machine, value):
    return (
        isinstance(value, list)
        and len(value) == 2
        and value[0] == "ref"
        and machine["heap"].get(value[1], {}).get("kind") == "closure"
    )


def _assert_supported_conversion_method(machine, value):
    if value is None or value[0] != "ref":
        return
    selected = machine["heap"].get(value[1])
    _require(
        not selected
        or selected.get("kind") != "proxy"
        and not (
            selected.get("kind") == "intrinsic"
            and selected.get("callableKind") in ("function", "constructor")
        ),
        "intrinsic or Proxy conversion callable requires its own protocol",
    )


def _hint(row):
    return "number" if row["goal"] == "Number" else "string"


def _key_for(row):
    if row["index"] == -1:
        return ["symbol", "well-known:toPrimitive"]
    keys = ("valueOf", "toString") if row["goal"] == "Number" else ("toString", "valueOf")
    return ["str", keys[int(row["index"])]]


def _lookup(machine, value, key):
    p = _protocol()
    promise_objects = (machine.get("protocol") or {}).get("promiseObjects") is not None
    current, seen = value[1], set()
    while True:
        _require(current not in seen, "cyclic primitive conversion prototype")
        seen.add(current)
        node = p._node(machine, current)
        _require(node["kind"] in ("object", "array", "closure"), "ordinary conversion prototype")
        descriptor = p._descriptor(node, p._key(key))
        if descriptor is not None:
            if descriptor["kind"] == "data" and descriptor["value"][0] == "ref":
                method = machine["heap"].get(descriptor["value"][1])
                if (
                    method
                    and method.get("kind") == "intrinsic"
                    and method.get("callableKind") == "function"
                ):
                    if (
                        key == ["str", "valueOf"]
                        and method.get("intrinsicName") == "Object.prototype.valueOf"
                    ):
                        return {
                            "descriptor": "builtin",
                            "value": ["undefined"],
                            "builtin": "Object.valueOf",
                        }
                    if (
                        key == ["str", "toString"]
                        and method.get("intrinsicName") == "Object.prototype.toString"
                    ):
                        return {
                            "descriptor": "builtin",
                            "value": ["undefined"],
                            "builtin": "Object.toString",
                        }
            return {
                "descriptor": descriptor["kind"],
                "value": descriptor["get"]
                if descriptor["kind"] == "accessor"
                else descriptor["value"],
                "builtin": None,
            }
        prototype = node.get("prototype", "ordinary")
        if prototype is None:
            return {"descriptor": "missing", "value": ["undefined"], "builtin": None}
        if prototype == "ordinary":
            if promise_objects:
                if node["kind"] == "array" and key in (
                    ["str", "constructor"],
                    ["str", "toLocaleString"],
                ):
                    _require(False, "array intrinsic conversion property requires its own profile")
                if node["kind"] == "closure" and key in (
                    ["str", "constructor"],
                    ["str", "toString"],
                    ["symbol", "well-known:toStringTag"],
                ):
                    _require(
                        False, "function intrinsic prototype requires its own conversion profile"
                    )
                if node["kind"] == "array" and key == ["str", "toString"]:
                    return {
                        "descriptor": "builtin",
                        "value": ["undefined"],
                        "builtin": "Array.toString",
                    }
                if node["kind"] == "array" and key == ["str", "join"]:
                    return {
                        "descriptor": "builtin",
                        "value": ["undefined"],
                        "builtin": "Array.prototype.join",
                    }
                from .promise_object_v8 import intrinsic_id

                owner = p._node(machine, intrinsic_id("Object.prototype"))
                inherited = p._descriptor(owner, p._key(key))
                if inherited is None:
                    return {"descriptor": "missing", "value": ["undefined"], "builtin": None}
                expected_builtin = {
                    ("str", "valueOf"): "Object.valueOf",
                    ("str", "toString"): "Object.toString",
                }.get(tuple(key))
                if (
                    expected_builtin is not None
                    and inherited["kind"] == "data"
                    and inherited["value"] == ["ref", intrinsic_id("Object.prototype." + key[1])]
                ):
                    return {
                        "descriptor": "builtin",
                        "value": ["undefined"],
                        "builtin": expected_builtin,
                    }
                return {
                    "descriptor": inherited["kind"],
                    "value": inherited["get"]
                    if inherited["kind"] == "accessor"
                    else inherited["value"],
                    "builtin": None,
                }
            _require(
                node["kind"] != "closure"
                or key not in (["str", "toString"], ["symbol", "well-known:toStringTag"]),
                "function intrinsic prototype requires its own conversion profile",
            )
            builtin = None
            if key == ["str", "valueOf"]:
                builtin = "Object.valueOf"
            elif key == ["str", "toString"]:
                builtin = "Array.toString" if node["kind"] == "array" else "Object.toString"
            elif key == ["str", "join"] and node["kind"] == "array":
                builtin = "Array.prototype.join"
            return {
                "descriptor": "builtin" if builtin else "missing",
                "value": ["undefined"],
                "builtin": builtin,
            }
        p._known(machine, prototype)
        _require(prototype[0] == "ref", "conversion prototype identity")
        current = prototype[1]


def scalar_result(row):
    value = row["candidate"]
    _require(_primitive(value), "conversion scalar candidate")
    _scalar_bound(value)
    if row["goal"] == "PropertyKey":
        return copy.deepcopy(value) if value[0] == "symbol" else ["str", scalar._to_string(value)]
    _require(value[0] != "symbol", "Symbol Number conversion requires its derived fault")
    return scalar_number(value)


def _next_lookup(row, index):
    row.update(
        index=index,
        phase="lookup",
        lookupKind=None,
        method=None,
        builtin=None,
        child=None,
        candidate=None,
        fault=None,
    )


def _fault(row, cause, candidate):
    row.update(phase="fault", child=None, candidate=copy.deepcopy(candidate), fault=cause)


def _accept_candidate(row, value):
    row["child"] = None
    if _primitive(value):
        _scalar_bound(value)
        row["candidate"] = copy.deepcopy(value)
        if row["goal"] == "Number" and value[0] == "symbol":
            _fault(row, "symbol-to-number", value)
        else:
            row["phase"] = "finish"
    elif row["index"] == -1:
        _fault(row, "exotic-nonprimitive", value)
    elif row["index"] == 0:
        _next_lookup(row, 1)
    else:
        _fault(row, "ordinary-exhausted", value)


def _select_method(machine, row, value):
    row.update(method=copy.deepcopy(value), builtin=None, child=None)
    _assert_supported_conversion_method(machine, value)
    if row["index"] == -1 and value[0] in ("undefined", "null"):
        _next_lookup(row, 0)
    elif _callable(machine, value):
        row["phase"] = "call"
    elif row["index"] == -1:
        _fault(row, "exotic-not-callable", value)
    elif row["index"] == 0:
        _next_lookup(row, 1)
    else:
        _fault(row, "ordinary-exhausted", value)


def _builtin_result(machine, row):
    p = _protocol()
    if row["builtin"] == "Object.valueOf":
        return copy.deepcopy(row["input"])
    node = p._node(machine, row["input"][1])
    if row["builtin"] == "Object.toString":
        found = _lookup(machine, row["input"], ["symbol", "well-known:toStringTag"])
        _require(found["descriptor"] != "accessor", "toStringTag getter needs its own continuation")
        tag = found["value"]
        name = tag[1] if tag[0] == "str" else "Array" if node["kind"] == "array" else "Object"
        return ["str", "[object " + name + "]"]
    _require(row["builtin"] == "Array.toString", "unknown conversion builtin")
    _require(node["kind"] == "array", "Array.toString requires an ordinary array")
    _require(
        _lookup(machine, row["input"], ["str", "join"])["builtin"] == "Array.prototype.join",
        "Array.toString join override requires its own continuation",
    )
    length = p._descriptor(node, "length")
    _require(
        length is not None and length["kind"] == "data" and length["value"][0] == "num",
        "Array.toString length bound",
    )
    count = length["value"][1]
    _require(
        type(count) in (int, float) and 0 <= count <= 10000 and int(count) == count,
        "Array.toString length bound",
    )
    values = []
    for index in range(int(count)):
        descriptor = p._descriptor(node, str(index))
        value = descriptor.get("value") if descriptor is not None else None
        _require(
            descriptor is not None
            and descriptor["kind"] == "data"
            and _primitive(value)
            and value[0] != "symbol",
            "Array.toString requires dense own scalar elements",
        )
        _scalar_bound(value)
        values.append("" if value[0] in ("null", "undefined") else scalar._to_string(value))
    result = ["str", ",".join(values)]
    _scalar_bound(result)
    return result


def _assert_fault(machine, row):
    cause, candidate, method = row["fault"], row["candidate"], row["method"]
    _assert_supported_conversion_method(machine, method)
    _require(isinstance(cause, str) and cause in FAULTS, "unknown primitive conversion fault")
    if cause == "symbol-to-number":
        _require(
            row["goal"] == "Number"
            and candidate is not None
            and candidate[0] == "symbol"
            and (_primitive(row["input"]) or _callable(machine, method)),
            "false Symbol Number fault",
        )
    elif cause == "exotic-not-callable":
        _require(
            row["index"] == -1
            and row["builtin"] is None
            and method is not None
            and method[0] not in ("undefined", "null")
            and not _callable(machine, method)
            and _eq(candidate, method),
            "false noncallable exotic fault",
        )
    elif cause == "exotic-nonprimitive":
        _require(
            row["index"] == -1
            and row["builtin"] is None
            and _callable(machine, method)
            and candidate is not None
            and not _primitive(candidate),
            "false exotic nonprimitive fault",
        )
    else:
        derived = (
            row["builtin"] == "Object.valueOf" and _eq(candidate, row["input"])
            if row["builtin"] is not None
            else not _primitive(candidate)
            if _callable(machine, method)
            else _eq(candidate, method)
        )
        _require(
            row["index"] == 1 and candidate is not None and derived,
            "false ordinary exhaustion fault",
        )


def validate_record(record):
    kind = record.get("kind") if type(record) is dict else None
    if not isinstance(kind, str) or kind not in FIELDS:
        return False
    p = _protocol()
    p._exact(record, FIELDS[kind], "primitive conversion record")
    _identity(record["id"])
    if kind == "conversion.begin":
        _require(isinstance(record["goal"], str) and record["goal"] in GOALS, "conversion goal")
        p.validate_protocol_value(record["value"])
    elif kind in ("conversion.call", "conversion.getter.call"):
        _identity(record["child"])
    elif kind == "conversion.fault":
        _identity(record["error"])
        p.validate_protocol_value(["ref", record["error"]])
    elif kind == "conversion.lookup":
        p.validate_protocol_key(record["key"])
        p.validate_protocol_value(record["value"])
        _require(
            isinstance(record["descriptor"], str) and record["descriptor"] in DESCRIPTORS,
            "conversion descriptor",
        )
        _require(
            record["builtin"] is None
            or isinstance(record["builtin"], str)
            and record["builtin"] in BUILTINS,
            "conversion builtin",
        )
    else:
        p.validate_protocol_value(
            record["result"]
            if kind in ("conversion.builtin", "conversion.finish")
            else record["value"]
        )
    return True


def assert_next(machine, record):
    rows = (machine.get("protocol", {}).get("primitiveConversions") or {}).get("operations", [])
    row = next((item for item in reversed(rows) if not _terminal(item)), None)
    if row is None:
        return True
    fields = record.get("fields") or {}
    if (
        row["phase"] in ("lookup", "getter-call", "call", "builtin", "finish", "fault")
        and record.get("kind") == "transfer"
        and fields.get("kind") == "cleanup"
        and _eq(fields.get("value"), ["undefined"])
        and (fields.get("handler") or {}).get("schema") == "semantic-handler-scopes/1"
    ):
        return True
    if row["phase"] in ("waiting", "getter-waiting") and row["child"] in machine["stack"]:
        return True
    _require(
        bool(machine["stack"]) and machine["stack"][-1] == row["parent"],
        "conversion continuation owner",
    )
    threw = (machine.get("lastReturn") or {}).get("kind") == "throw"
    expected = {
        "lookup": "conversion.lookup",
        "getter-call": "conversion.getter.call",
        "call": "conversion.call",
        "builtin": "conversion.builtin",
        "finish": "conversion.finish",
        "fault": "conversion.fault",
        "waiting": "conversion.throw" if threw else "conversion.return",
        "getter-waiting": "conversion.throw" if threw else "conversion.getter.return",
    }[row["phase"]]
    _require(
        record.get("kind") == expected and record.get("id") == row["id"],
        "pending primitive conversion requires " + expected,
    )
    return True


def apply_record(machine, record):
    if not validate_record(record):
        return False
    p = _protocol()
    kind = record["kind"]
    if kind == "conversion.begin":
        value = p._known(machine, record["value"])
        state = machine["protocol"].get("primitiveConversions")
        rows = state["operations"] if state is not None else []
        _require(
            len(rows) < 4096 and all(row["id"] != record["id"] for row in rows),
            "conversion identity/bound",
        )
        _require(bool(machine["stack"]), "conversion requires an active invocation")
        if _primitive(value):
            _scalar_bound(value)
        else:
            _require(
                value[0] == "ref" and p._node(machine, value[1])["kind"] in ("object", "array"),
                "conversion requires ordinary object state",
            )
        row = {
            "id": record["id"],
            "parent": machine["stack"][-1],
            "goal": record["goal"],
            "input": copy.deepcopy(value),
            "index": -1,
            "phase": "lookup",
            "lookupKind": None,
            "method": None,
            "builtin": None,
            "child": None,
            "candidate": None,
            "result": None,
            "fault": None,
        }
        if _primitive(value):
            _accept_candidate(row, value)
        if state is None:
            state = {"schema": SCHEMA, "operations": []}
            machine["protocol"]["primitiveConversions"] = state
        state["operations"].append(row)
        return True
    state = machine["protocol"].get("primitiveConversions") or {"operations": []}
    row = next((item for item in state["operations"] if item["id"] == record["id"]), None)
    _require(row is not None, "unknown primitive conversion")
    _require(
        bool(machine["stack"]) and machine["stack"][-1] == row["parent"],
        "conversion record is outside its caller",
    )
    if kind == "conversion.lookup":
        _require(
            row["phase"] == "lookup" and _eq(record["key"], _key_for(row)),
            "conversion method order",
        )
        if record["key"][0] == "symbol":
            p._symbol(machine, record["key"][1])
        p._known(machine, record["value"])
        found = _lookup(machine, row["input"], record["key"])
        _require(
            _eq(found, {key: record[key] for key in ("descriptor", "value", "builtin")}),
            "false primitive conversion lookup",
        )
        row.update(
            lookupKind=found["descriptor"],
            method=copy.deepcopy(found["value"]),
            builtin=found["builtin"],
        )
        if found["descriptor"] == "accessor" and found["value"][0] != "undefined":
            _require(
                _callable(machine, found["value"]), "conversion getter requires an owned closure"
            )
            row["phase"] = "getter-call"
        elif found["builtin"] is not None:
            row["phase"] = "builtin"
        else:
            _select_method(machine, row, found["value"])
    elif kind in ("conversion.call", "conversion.getter.call"):
        getter = kind == "conversion.getter.call"
        _require(
            row["phase"] == ("getter-call" if getter else "call"),
            "conversion call requires its captured callable",
        )
        from .construction_v8 import _call

        _call(
            machine,
            record["child"],
            row["method"],
            row["input"],
            [["str", _hint(row)]] if not getter and row["index"] == -1 else [],
        )
        row.update(child=record["child"], phase="getter-waiting" if getter else "waiting")
    elif kind == "conversion.builtin":
        _require(row["phase"] == "builtin", "conversion builtin requires its captured intrinsic")
        expected = _builtin_result(machine, row)
        _require(_eq(record["result"], expected), "false primitive conversion builtin result")
        _accept_candidate(row, expected)
    elif kind in ("conversion.return", "conversion.getter.return"):
        getter = kind == "conversion.getter.return"
        returned = machine.get("lastReturn") or {}
        _require(
            row["phase"] == ("getter-waiting" if getter else "waiting")
            and row["child"] not in machine["stack"]
            and returned.get("kind") == "return"
            and _eq(returned.get("value"), record["value"]),
            "conversion callback completion differs",
        )
        p._known(machine, record["value"])
        if getter:
            _select_method(machine, row, record["value"])
        else:
            _accept_candidate(row, record["value"])
    elif kind == "conversion.throw":
        returned = machine.get("lastReturn") or {}
        _require(
            row["phase"] in ("waiting", "getter-waiting")
            and row["child"] not in machine["stack"]
            and returned.get("kind") == "throw"
            and _eq(returned.get("value"), record["value"]),
            "conversion callback abrupt completion differs",
        )
        p._known(machine, record["value"])
        row.update(candidate=None, result=copy.deepcopy(record["value"]), phase="abrupt")
    elif kind == "conversion.fault":
        _require(row["phase"] == "fault", "conversion fault requires a derived cause")
        _assert_fault(machine, row)
        from .intrinsic_error_v8 import allocate_intrinsic_error_v8

        allocate_intrinsic_error_v8(machine, record["error"], "TypeError", FAULTS[row["fault"]])
        row.update(result=["ref", record["error"]], phase="abrupt")
        machine["lastReturn"] = {"kind": "throw", "value": copy.deepcopy(row["result"])}
    else:
        _require(row["phase"] == "finish", "conversion finish requires a primitive candidate")
        expected = scalar_result(row)
        p._known(machine, record["result"])
        _require(_eq(record["result"], expected), "false primitive scalar conversion")
        row.update(result=copy.deepcopy(expected), phase="complete")
        machine["lastReturn"] = {"kind": "return", "value": copy.deepcopy(expected)}
    return True


def validate_state(machine):
    state = machine.get("protocol", {}).get("primitiveConversions")
    if state is None:
        return True
    p = _protocol()
    p._exact(state, {"schema", "operations"}, "primitive conversion state")
    _require(
        state["schema"] == SCHEMA
        and type(state["operations"]) is list
        and len(state["operations"]) <= 4096,
        "conversion state schema/bound",
    )
    seen, children, active_depth = set(), set(), -1
    for row in state["operations"]:
        p._exact(row, OPERATION_FIELDS, "primitive conversion continuation")
        _identity(row["id"])
        _identity(row["parent"])
        _require(row["id"] not in seen, "duplicate primitive conversion")
        seen.add(row["id"])
        _require(
            row["parent"] in machine["frameFunctions"]
            and isinstance(row["goal"], str)
            and row["goal"] in GOALS
            and type(row["index"]) in (int, float)
            and row["index"] in (-1, 0, 1)
            and isinstance(row["phase"], str)
            and row["phase"] in PHASES,
            "conversion phase/goal/owner",
        )
        if not _terminal(row):
            depth = (
                machine["stack"].index(row["parent"]) if row["parent"] in machine["stack"] else -1
            )
            _require(
                depth > active_depth, "unfinished conversion requires a distinct live nested parent"
            )
            active_depth = depth
        p._known(machine, row["input"])
        _scalar_bound(row["input"])
        for value in (row["method"], row["candidate"], row["result"]):
            if value is not None:
                p._known(machine, value)
                _scalar_bound(value)
        _assert_supported_conversion_method(machine, row["method"])
        _require(
            row["lookupKind"] is None
            or isinstance(row["lookupKind"], str)
            and row["lookupKind"] in DESCRIPTORS,
            "conversion lookup kind",
        )
        _require(
            row["builtin"] is None
            or isinstance(row["builtin"], str)
            and row["builtin"] in BUILTINS,
            "conversion builtin identity",
        )
        _require(
            row["fault"] is None or isinstance(row["fault"], str) and row["fault"] in FAULTS,
            "conversion fault identity",
        )
        if row["child"] is not None:
            _identity(row["child"])
            _require(
                row["child"] not in children and row["child"] in machine["frameFunctions"],
                "conversion callback identity",
            )
            children.add(row["child"])
        if _primitive(row["input"]):
            _require(
                row["index"] == -1
                and all(row[key] is None for key in ("lookupKind", "method", "builtin", "child"))
                and _eq(row["candidate"], row["input"])
                and (
                    row["phase"] in ("finish", "complete")
                    or row["phase"] in ("fault", "abrupt")
                    and row["fault"] == "symbol-to-number"
                ),
                "primitive conversion entry state",
            )
        else:
            _require(
                row["input"][0] == "ref"
                and p._node(machine, row["input"][1])["kind"] in ("object", "array")
                and (
                    row["phase"] == "lookup"
                    or row["lookupKind"] is not None
                    and row["method"] is not None
                ),
                "conversion object input/selection",
            )
        _require(
            (row["lookupKind"] == "builtin") == (row["builtin"] is not None),
            "conversion builtin descriptor binding",
        )
        if row["builtin"] is not None:
            _require(
                row["method"] == ["undefined"]
                and row["index"] != -1
                and (
                    row["builtin"] == "Object.valueOf"
                    if _key_for(row)[1] == "valueOf"
                    else row["builtin"] in ("Object.toString", "Array.toString")
                ),
                "conversion builtin method binding",
            )
        if row["phase"] == "lookup":
            _require(
                all(
                    row[key] is None
                    for key in (
                        "lookupKind",
                        "method",
                        "builtin",
                        "child",
                        "candidate",
                        "result",
                        "fault",
                    )
                ),
                "pending conversion lookup state",
            )
        if row["phase"] in ("getter-call", "getter-waiting", "call", "waiting", "builtin"):
            _require(
                row["input"][0] == "ref"
                and all(row[key] is None for key in ("candidate", "result", "fault")),
                "conversion callable phase values",
            )
        if row["phase"] in ("getter-call", "getter-waiting"):
            _require(row["lookupKind"] == "accessor", "getter conversion lookup kind")
        if row["phase"] in ("getter-call", "getter-waiting", "call", "waiting"):
            _require(
                _callable(machine, row["method"])
                and row["lookupKind"] in ("data", "accessor")
                and row["builtin"] is None,
                "captured conversion callable",
            )
        if row["phase"] in ("getter-call", "call", "builtin"):
            _require(row["child"] is None, "unentered conversion callback identity")
        if row["phase"] == "builtin":
            _require(
                row["lookupKind"] == "builtin" and row["builtin"] is not None,
                "conversion intrinsic state",
            )
        if (
            row["phase"] in ("waiting", "getter-waiting")
            or row["phase"] == "abrupt"
            and row["fault"] is None
        ):
            _require(
                row["child"] is not None
                and _callable(machine, row["method"])
                and row["builtin"] is None
                and machine["frameFunctions"][row["child"]]
                == p._node(machine, row["method"][1])["code"],
                "conversion callback source identity",
            )
            if row["child"] in machine["stack"]:
                _require(
                    row["parent"] in machine["stack"]
                    and machine["stack"].index(row["child"])
                    == machine["stack"].index(row["parent"]) + 1,
                    "conversion callback stack",
                )
                frame = next(
                    (
                        item
                        for item in machine["protocol"]["callFrames"]
                        if item["child"] == row["child"]
                    ),
                    None,
                )
                getter, node = row["phase"] == "getter-waiting", p._node(machine, row["method"][1])
                _require(
                    frame is not None
                    and frame["parent"] == row["parent"]
                    and _eq(
                        frame["arguments"],
                        [["str", _hint(row)]] if not getter and row["index"] == -1 else [],
                    )
                    and _eq(
                        machine["locals"].get(row["child"] + ":this"),
                        node.get("lexicalReceiver")
                        if node.get("functionKind") == "arrow"
                        else row["input"],
                    ),
                    "conversion callback receiver or hint",
                )
            elif row["phase"] != "abrupt":
                _require(
                    bool(machine["stack"])
                    and machine["stack"][-1] == row["parent"]
                    and (machine.get("lastReturn") or {}).get("kind") in ("return", "throw"),
                    "completed conversion child requires its immediately pending result",
                )
                p._known(machine, machine["lastReturn"]["value"])
            if row["phase"] == "abrupt":
                _require(
                    row["child"] not in machine["stack"]
                    and row["candidate"] is None
                    and row["result"] is not None,
                    "abrupt conversion callback state",
                )
        if row["phase"] in ("finish", "complete"):
            _require(
                row["child"] is None
                and row["fault"] is None
                and _primitive(row["candidate"])
                and (row["goal"] != "Number" or row["candidate"][0] != "symbol")
                and (
                    _primitive(row["input"])
                    or _callable(machine, row["method"])
                    or row["builtin"] is not None
                ),
                "conversion scalar state",
            )
        if row["phase"] == "complete":
            _require(_eq(row["result"], scalar_result(row)), "false completed primitive conversion")
        if row["phase"] == "fault" or row["phase"] == "abrupt" and row["fault"] is not None:
            _require(row["child"] is None, "derived conversion fault cannot retain a callback")
            _assert_fault(machine, row)
            if row["phase"] == "abrupt":
                _require(
                    row["result"] is not None
                    and row["result"][0] == "ref"
                    and p._node(machine, row["result"][1])["kind"] == "object",
                    "conversion fault error identity",
                )
        if not _terminal(row):
            _require(row["result"] is None, "unfinished conversion has a result")
        if row["phase"] not in ("fault", "abrupt"):
            _require(row["fault"] is None, "conversion phase cannot have a fault")
        if row["phase"] == "abrupt":
            _require(row["result"] is not None, "abrupt conversion lacks its original value")
    return True
