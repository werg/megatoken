"""Fresh intrinsic errors allocated only after a machine-derived fault."""

from __future__ import annotations

import copy

from .semantic_json import _js_string_length

ERROR_TYPES = frozenset(
    ("Error", "EvalError", "RangeError", "ReferenceError", "SyntaxError", "TypeError", "URIError")
)
FIELDS = {"property.fault": {"kind", "operation", "target", "key", "error"}}
STATE_SCHEMA = "semantic-intrinsic-error-state/1"
ERROR_BOUND = 4096


def _protocol():
    from . import protocol_v8

    return protocol_v8


def allocate_intrinsic_error_v8(machine, identity, error_type, message):
    """Private allocator; callers must derive their operation's failure first."""
    p = _protocol()
    if not isinstance(identity, str) or not identity or _js_string_length(identity) > 256:
        p._fail("intrinsic error identity")
    if (
        not isinstance(error_type, str)
        or error_type not in ERROR_TYPES
        or not isinstance(message, str)
        or _js_string_length(message) > 4096
    ):
        p._fail("intrinsic error type or message")
    validate_state(machine)
    registry = machine["protocol"].get("intrinsicErrors")
    if registry is None:
        registry = {"schema": STATE_SCHEMA, "errors": []}
    if len(registry["errors"]) >= ERROR_BOUND:
        p._fail("intrinsic error allocation bound")
    from .promise_object_v8 import initialize, intrinsic_id

    if machine["protocol"].get("promiseObjects") is None:
        initialize(machine)
    record = {
        "kind": "allocate",
        "name": identity,
        "objectKind": "object",
        "length": None,
        "properties": [
            {"key": ["str", "message"], "value": ["str", message]},
        ],
        "code": None,
        "captures": [],
        "functionKind": None,
        "lexicalReceiver": None,
    }
    p.validate_protocol_record(record)
    prior = machine.get("_internalProtocolTransition")
    machine["_internalProtocolTransition"] = True
    try:
        p._apply(machine, record)
    finally:
        if prior is None:
            machine.pop("_internalProtocolTransition", None)
        else:
            machine["_internalProtocolTransition"] = prior
    node = machine["heap"][identity]
    node["declaredError"] = error_type
    node["prototype"] = ["ref", intrinsic_id(error_type + ".prototype")]
    p._install(
        node,
        "message",
        {
            "kind": "data",
            "value": ["str", message],
            "writable": True,
            "enumerable": False,
            "configurable": True,
        },
    )
    registry["errors"].append({"id": identity, "errorType": error_type})
    machine["protocol"]["intrinsicErrors"] = registry
    return ["ref", identity]


def validate_record(record):
    if type(record) is not dict or record.get("kind") not in FIELDS:
        return False
    p = _protocol()
    p._exact(record, FIELDS[record["kind"]], "property fault record")
    if record["operation"] != "has":
        p._fail("unsupported property fault operation")
    for name in ("target", "key"):
        p.validate_protocol_value(record[name])
    if (
        not isinstance(record["error"], str)
        or not record["error"]
        or _js_string_length(record["error"]) > 256
    ):
        p._fail("property fault identity")
    return True


def invalid_has_message(machine, key, target):
    from .semantic_primitives import _to_string

    p = _protocol()

    def display(value):
        if value[0] == "symbol":
            identity = value[1]
            if identity.startswith("well-known:"):
                description = ["str", "Symbol." + identity[11:]]
            else:
                row = next(
                    (row for row in machine["protocol"]["symbols"] if row["id"] == identity), None
                )
                description = row.get("description") if row else None
            if not description or description[0] not in ("str", "undefined"):
                p._fail("invalid-in diagnostic requires a declared Symbol")
            return "Symbol(" + (description[1] if description[0] == "str" else "") + ")"
        if value[0] == "ref":
            node = machine["heap"].get(value[1], {})
            if node.get("kind") == "array":
                return "[object Array]"
            if (
                node.get("kind") == "object"
                and node.get("prototype", "ordinary") in ("ordinary", "Object.prototype", None)
                and "declaredError" not in node
            ):
                return (
                    "[object Object]" if node.get("prototype", "ordinary") is None else "#<Object>"
                )
            p._fail("invalid-in diagnostic requires its owned object display profile")
        return _to_string(value)

    return "Cannot use 'in' operator to search for '" + display(key) + "' in " + display(target)


def apply_record(machine, record):
    if not validate_record(record):
        return False
    p = _protocol()
    # An invalid RHS throws before ToPropertyKey, including when the key is an
    # object with a callable conversion hook. No lookup or callback runs here.
    p._known(machine, record["target"])
    p._known(machine, record["key"])
    if record["target"][0] not in ("undefined", "null", "bool", "num", "str", "bigint", "symbol"):
        p._fail("property fault requires a non-object has target")
    value = allocate_intrinsic_error_v8(
        machine,
        record["error"],
        "TypeError",
        invalid_has_message(machine, record["key"], record["target"]),
    )
    machine["lastReturn"] = {"kind": "throw", "value": copy.deepcopy(value)}
    return True


def validate_error_node(node):
    if "declaredError" in node and (
        node.get("kind") != "object"
        or not isinstance(node["declaredError"], str)
        or node["declaredError"] not in ERROR_TYPES
    ):
        _protocol()._fail("intrinsic error brand")


def validate_state(machine):
    p = _protocol()
    branded = {
        identity: node for identity, node in machine["heap"].items() if "declaredError" in node
    }
    for node in branded.values():
        validate_error_node(node)
    registry = machine["protocol"].get("intrinsicErrors")
    if registry is None:
        if branded:
            p._fail("intrinsic error brand lacks its allocation registry")
        return True
    p._exact(registry, {"schema", "errors"}, "intrinsic error registry")
    if (
        registry["schema"] != STATE_SCHEMA
        or type(registry["errors"]) is not list
        or len(registry["errors"]) > ERROR_BOUND
    ):
        p._fail("intrinsic error registry schema or bound")
    seen = set()
    for row in registry["errors"]:
        p._exact(row, {"id", "errorType"}, "intrinsic error allocation")
        identity, error_type = row["id"], row["errorType"]
        if (
            not isinstance(identity, str)
            or not identity
            or _js_string_length(identity) > 256
            or identity in seen
        ):
            p._fail("intrinsic error allocation identity")
        if not isinstance(error_type, str) or error_type not in ERROR_TYPES:
            p._fail("intrinsic error allocation type")
        node = branded.get(identity)
        if node is None or node["declaredError"] != error_type:
            p._fail("intrinsic error allocation identity or type differs from heap")
        seen.add(identity)
    if seen != set(branded):
        p._fail("intrinsic error brand lacks its allocation registry entry")
    return True
