"""Fresh intrinsic errors allocated only after a machine-derived fault."""

from __future__ import annotations

import copy

from .semantic_json import _js_string_length

ERROR_TYPES = frozenset(
    ("Error", "EvalError", "RangeError", "ReferenceError", "SyntaxError", "TypeError", "URIError")
)
FIELDS = {"property.fault": {"kind", "operation", "target", "key", "error"}}
INVALID_HAS_TARGET = "Right-hand side of in is not an object"


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
    record = {
        "kind": "allocate",
        "name": identity,
        "objectKind": "object",
        "length": None,
        "properties": [
            {"key": ["str", "name"], "value": ["str", error_type]},
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
    machine["heap"][identity]["declaredError"] = error_type
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
    value = allocate_intrinsic_error_v8(machine, record["error"], "TypeError", INVALID_HAS_TARGET)
    machine["lastReturn"] = {"kind": "throw", "value": copy.deepcopy(value)}
    return True


def validate_error_node(node):
    if "declaredError" in node and (
        node.get("kind") != "object"
        or not isinstance(node["declaredError"], str)
        or node["declaredError"] not in ERROR_TYPES
    ):
        _protocol()._fail("intrinsic error brand")
