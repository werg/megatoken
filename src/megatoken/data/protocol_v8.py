"""Independent decoder and validator for semantic-protocol-record/1.

This protocol extends only the v8 shared machine. The older detailed and pure
algorithm languages keep their narrower tagged-value domains.
"""

from __future__ import annotations

import copy
import json
import math

from . import algorithm_v7 as v7
from .algorithm_v7 import stable_algorithm_json
from .construction_v8 import FIELDS as CONSTRUCTION_FIELDS
from .construction_v8 import apply_record as apply_construction_record
from .construction_v8 import assert_next as assert_construction_next
from .construction_v8 import validate_record as validate_construction_record
from .construction_v8 import validate_state as validate_construction_record_state
from .generator_v8 import FIELDS as GENERATOR_FIELDS
from .generator_v8 import apply_record as apply_generator_record
from .generator_v8 import assert_next as assert_generator_next
from .generator_v8 import validate_record as validate_generator_record
from .generator_v8 import validate_state as validate_generator_state
from .iterator_v8 import FIELDS as ITERATOR_FIELDS
from .iterator_v8 import apply_record as apply_iterator_record
from .iterator_v8 import validate_record as validate_iterator_record
from .iterator_v8 import validate_state as validate_iterator_state
from .memory_v8 import FIELDS as MEMORY_FIELDS
from .memory_v8 import apply_record as apply_memory_record
from .memory_v8 import validate_record as validate_memory_record
from .memory_v8 import validate_state as validate_memory_state
from .number_v8 import FIELDS as NUMBER_FIELDS
from .number_v8 import apply_record as apply_number_record
from .number_v8 import assert_next as assert_number_next
from .number_v8 import validate_record as validate_number_record
from .number_v8 import validate_state as validate_number_state
from .promise_join_v8 import FIELDS as PROMISE_JOIN_FIELDS
from .promise_join_v8 import apply_record as apply_promise_join_record
from .promise_join_v8 import validate_record as validate_promise_join_record
from .proxy_v8 import FIELDS as PROXY_FIELDS
from .proxy_v8 import apply_record as apply_proxy_record
from .proxy_v8 import assert_next as assert_proxy_next
from .proxy_v8 import validate_record as validate_proxy_record
from .proxy_v8 import validate_state as validate_proxy_state

PREFIX = "protocol "
SCHEMA = "semantic-protocol-record/1"
STATE_SCHEMA = "semantic-protocol-state/1"
WELL_KNOWN = (
    "iterator",
    "asyncIterator",
    "toPrimitive",
    "hasInstance",
    "species",
    "isConcatSpreadable",
    "match",
    "matchAll",
    "replace",
    "search",
    "split",
    "toStringTag",
    "unscopables",
)
FIELDS = {
    "symbol.create": {"kind", "id", "description"},
    "symbol.for": {"kind", "id", "key"},
    "symbol.read": {"kind", "symbol", "operation", "result", "bind"},
    "allocate": {
        "kind",
        "name",
        "objectKind",
        "length",
        "properties",
        "code",
        "captures",
        "functionKind",
        "lexicalReceiver",
    },
    "bind": {"kind", "name", "value"},
    "compute": {"kind", "expr", "result", "bind"},
    "choose": {"kind", "expr", "decision", "path"},
    "call": {"kind", "child", "callee", "receiver", "arguments"},
    "return": {"kind", "value", "invocation"},
    "throw": {"kind", "value", "invocation"},
    "end": {"kind", "completion"},
    "property.define": {"kind", "object", "key", "descriptor", "success"},
    "property.get": {"kind", "object", "key", "receiver", "result"},
    "property.set": {"kind", "object", "key", "receiver", "value", "success"},
    "property.delete": {"kind", "object", "key", "success"},
    "property.keys": {"kind", "object", "mode", "name", "keys"},
    "property.prototype": {"kind", "object", "prototype", "success"},
    "property.preventExtensions": {"kind", "object"},
    "property.get.call": {"kind", "id", "child", "object", "key", "receiver"},
    "property.set.call": {"kind", "id", "child", "object", "key", "receiver", "value"},
    "property.finish": {"kind", "id", "completion"},
}


class ProtocolV8Error(ValueError):
    pass


FIELDS.update(CONSTRUCTION_FIELDS)
FIELDS.update(MEMORY_FIELDS)
FIELDS.update(ITERATOR_FIELDS)
FIELDS.update(PROMISE_JOIN_FIELDS)
FIELDS.update(NUMBER_FIELDS)
FIELDS.update(GENERATOR_FIELDS)
FIELDS.update(PROXY_FIELDS)


def _fail(message):
    raise ProtocolV8Error(f"Invalid {SCHEMA}: {message}")


def _exact(value, fields, label):
    if type(value) is not dict or set(value) != set(fields):
        _fail(label + " fields")


def _identity(value):
    if not isinstance(value, str) or not value or len(value) > 256:
        _fail("identity")


def validate_protocol_value(value):
    if isinstance(value, list) and value and value[0] == "symbol":
        if len(value) != 2:
            _fail("symbol value")
        _identity(value[1])
        return True
    v7._validate_value(value)
    return True


def validate_protocol_key(key):
    if (
        not isinstance(key, list)
        or len(key) != 2
        or key[0] not in ("str", "symbol")
        or not isinstance(key[1], str)
    ):
        _fail("tagged property key")
    if key[0] == "symbol":
        _identity(key[1])
    return True


def _validate_descriptor(value):
    if not isinstance(value, dict):
        _fail("descriptor")
    if value.get("kind") == "data":
        _exact(
            value, {"kind", "value", "writable", "enumerable", "configurable"}, "data descriptor"
        )
        validate_protocol_value(value["value"])
        if type(value["writable"]) is not bool:
            _fail("descriptor boolean")
    elif value.get("kind") == "accessor":
        _exact(value, {"kind", "get", "set", "enumerable", "configurable"}, "accessor descriptor")
        for field in ("get", "set"):
            validate_protocol_value(value[field])
            if value[field][0] not in ("ref", "undefined"):
                _fail("accessor closure or undefined")
    else:
        _fail("descriptor kind")
    if type(value["enumerable"]) is not bool or type(value["configurable"]) is not bool:
        _fail("descriptor boolean")


def _expression(term, depth=0, budget=None):
    if budget is None:
        budget = [128]
    budget[0] -= 1
    if budget[0] < 0 or depth > 16:
        _fail("expression bound")
    op = term.get("op") if type(term) is dict else None
    if op == "literal":
        _exact(term, {"op", "value"}, "literal")
        validate_protocol_value(term["value"])
    elif op == "local":
        _exact(term, {"op", "name"}, "local")
        _identity(term["name"])
    elif op == "unary":
        _exact(term, {"op", "operator", "arg"}, "unary")
        if term["operator"] not in ("!", "+", "-", "~", "typeof", "void"):
            _fail("unary operator")
        _expression(term["arg"], depth + 1, budget)
    elif op == "binary":
        _exact(term, {"op", "operator", "left", "right"}, "binary")
        if term["operator"] not in (
            "+",
            "-",
            "*",
            "/",
            "%",
            "**",
            "<",
            "<=",
            ">",
            ">=",
            "===",
            "!==",
            "==",
            "!=",
            "&",
            "|",
            "^",
            "<<",
            ">>",
            ">>>",
        ):
            _fail("binary operator")
        _expression(term["left"], depth + 1, budget)
        _expression(term["right"], depth + 1, budget)
    else:
        _fail("expression requires explicit boundary")


def validate_protocol_record(record):
    if type(record) is not dict or not isinstance(record.get("kind"), str):
        _fail("record kind must be a string")
    if type(record) is dict and record.get("kind") in PROXY_FIELDS:
        return validate_proxy_record(record)
    if type(record) is dict and record.get("kind") in NUMBER_FIELDS:
        return validate_number_record(record)
    if type(record) is dict and record.get("kind") in GENERATOR_FIELDS:
        return validate_generator_record(record)
    if type(record) is dict and record.get("kind") in PROMISE_JOIN_FIELDS:
        try:
            validate_promise_join_record(record)
        except ProtocolV8Error:
            raise
        except (TypeError, KeyError, IndexError) as error:
            raise ProtocolV8Error("Invalid Promise join record") from error
        return True
    if type(record) is dict and record.get("kind") in CONSTRUCTION_FIELDS:
        try:
            validate_construction_record(record)
        except ProtocolV8Error:
            raise
        except (TypeError, KeyError, IndexError) as error:
            raise ProtocolV8Error("Invalid construction record") from error
        return True
    if type(record) is dict and record.get("kind") in MEMORY_FIELDS:
        try:
            validate_memory_record(record)
        except ProtocolV8Error:
            raise
        except (TypeError, KeyError, IndexError) as error:
            raise ProtocolV8Error("Invalid binary-memory record") from error
        return True
    if type(record) is dict and record.get("kind") in ITERATOR_FIELDS:
        try:
            validate_iterator_record(record)
        except ProtocolV8Error:
            raise
        except (TypeError, KeyError, IndexError) as error:
            raise ProtocolV8Error("Invalid iterator record") from error
        return True
    if type(record) is not dict or record.get("kind") not in FIELDS:
        _fail("record kind")
    _exact(record, FIELDS[record["kind"]], "record")
    for key in ("id", "name", "object", "child", "invocation"):
        if key in record:
            _identity(record[key])
    if "bind" in record and record["bind"] is not None:
        _identity(record["bind"])
    for key in ("value", "result", "decision", "receiver", "callee", "symbol"):
        if key in record:
            validate_protocol_value(record[key])
    if "success" in record and type(record["success"]) is not bool:
        _fail("success boolean")
    if "expr" in record:
        _expression(record["expr"])
    if record["kind"].startswith("property.") and "key" in record:
        validate_protocol_key(record["key"])
    kind = record["kind"]
    if kind == "symbol.create":
        validate_protocol_value(record["description"])
        if record["description"][0] not in ("str", "undefined"):
            _fail("symbol description")
    elif kind == "symbol.for":
        if not isinstance(record["key"], str):
            _fail("registry key")
    elif kind == "symbol.read":
        if record["symbol"][0] != "symbol" or record["operation"] not in (
            "description",
            "keyFor",
            "toString",
        ):
            _fail("symbol read operation")
    elif kind == "allocate":
        if record["objectKind"] not in ("object", "array", "closure"):
            _fail("allocation kind")
        if record["objectKind"] == "array":
            if type(record["length"]) is not int or not 0 <= record["length"] <= 4096:
                _fail("array length")
        elif record["length"] is not None:
            _fail("non-array length")
        if (
            not isinstance(record["properties"], list)
            or len(record["properties"]) + (record["objectKind"] == "array") > 4096
        ):
            _fail("allocation properties")
        for item in record["properties"]:
            _exact(item, {"key", "value"}, "property")
            validate_protocol_key(item["key"])
            validate_protocol_value(item["value"])
        if not isinstance(record["captures"], list) or len(record["captures"]) > 4096:
            _fail("captures")
        for capture in record["captures"]:
            _exact(capture, {"binding", "cell"}, "capture")
            _identity(capture["binding"])
            _identity(capture["cell"])
        if record["objectKind"] == "closure":
            _identity(record["code"])
            if record["functionKind"] not in ("ordinary", "arrow"):
                _fail("closure function kind")
            if record["functionKind"] == "arrow":
                validate_protocol_value(record["lexicalReceiver"])
            elif record["lexicalReceiver"] is not None:
                _fail("ordinary closure lexical receiver")
        elif (
            record["code"] is not None
            or record["captures"]
            or record["functionKind"] is not None
            or record["lexicalReceiver"] is not None
        ):
            _fail("non-closure code or captures")
    elif kind == "call":
        if (
            record["callee"][0] != "ref"
            or not isinstance(record["arguments"], list)
            or len(record["arguments"]) > 4096
        ):
            _fail("call shape")
        for arg in record["arguments"]:
            validate_protocol_value(arg)
    elif kind == "choose" and record["path"] not in ("then", "else", "evaluate-right", "use-left"):
        _fail("choice path")
    elif kind == "property.define":
        _validate_descriptor(record["descriptor"])
    elif kind == "property.keys":
        if record["mode"] not in (
            "own",
            "strings",
            "symbols",
            "enumerable-strings",
        ) or not isinstance(record["keys"], list):
            _fail("key enumeration")
        for key in record["keys"]:
            validate_protocol_key(key)
    elif kind == "property.prototype":
        proto = record["prototype"]
        if proto is not None and proto != "ordinary":
            validate_protocol_value(proto)
            if proto[0] != "ref":
                _fail("prototype")
    elif kind == "property.finish":
        completion = record["completion"]
        _exact(completion, {"kind", "value"}, "completion")
        if completion["kind"] not in ("normal", "throw"):
            _fail("completion kind")
        validate_protocol_value(completion["value"])
    elif kind == "end" and record["completion"] not in ("return", "throw"):
        _fail("terminal completion")
    return True


def encode_protocol_record(record):
    validate_protocol_record(record)
    return PREFIX + stable_algorithm_json(record)


def decode_protocol_record(line):
    if (
        not isinstance(line, str)
        or not line.startswith(PREFIX)
        or len(line.encode("utf-8")) > 65536
    ):
        _fail("canonical line")
    try:
        record = json.loads(line[len(PREFIX) :])
    except (TypeError, json.JSONDecodeError) as error:
        raise ProtocolV8Error("Invalid protocol record JSON") from error
    validate_protocol_record(record)
    if encode_protocol_record(record) != line:
        _fail("noncanonical bytes")
    return record


def create_protocol_state():
    return {
        "schema": STATE_SCHEMA,
        "symbols": [
            {"id": "well-known:" + name, "kind": "well-known", "name": name} for name in WELL_KNOWN
        ],
        "pendingCalls": [],
        "callFrames": [],
        "sequence": 0,
        "construction": None,
        "numberConversions": None,
        "proxyCalls": None,
    }


def _key(tag):
    return tag[1] if tag[0] == "str" else ("symbol", tag[1])


def _public(key):
    return ["symbol", key[1]] if isinstance(key, tuple) else ["str", key]


def _symbol(machine, identity):
    row = next((row for row in machine["protocol"]["symbols"] if row["id"] == identity), None)
    if row is None:
        _fail("undeclared Symbol identity")
    return row


def _known(machine, value, self_id=None):
    validate_protocol_value(value)
    if value[0] == "symbol":
        _symbol(machine, value[1])
    if value[0] == "ref" and value[1] != self_id and value[1] not in machine["heap"]:
        _fail("unallocated reference")
    return value


def _assert_fresh_heap_identity(machine, identity):
    _identity(identity)
    protocol = machine.get("protocol") or {}
    heap = machine["heap"]
    heap_ids = set(heap) if isinstance(heap, dict) else {row[0] for row in heap}
    reserved = {row["id"] for row in protocol.get("symbols", [])}
    reserved.update(
        row["id"] for row in (protocol.get("construction") or {}).get("privateNames", [])
    )
    if identity in heap_ids or identity in reserved:
        _fail("duplicate or reserved heap identity")


def _node(machine, identity):
    node = machine["heap"].get(identity)
    if not isinstance(node, dict) or node.get("kind") not in ("object", "array", "closure"):
        _fail("ordinary object required")
    node.setdefault("properties", {})
    node.setdefault("descriptors", {})
    node.setdefault("prototype", "ordinary")
    node.setdefault("extensible", True)
    return node


def _descriptor(node, key):
    metadata = node.get("descriptors", {}).get(key)
    if metadata and metadata.get("kind") == "accessor":
        return copy.deepcopy(metadata)
    if key not in node["properties"]:
        return None
    default = {"kind": "data", "writable": True, "enumerable": True, "configurable": True}
    if node["kind"] == "array" and key == "length":
        default = {"kind": "data", "writable": True, "enumerable": False, "configurable": False}
    return {
        **default,
        **copy.deepcopy(metadata or {}),
        "value": copy.deepcopy(node["properties"][key]),
    }


def _lookup_descriptor(machine, identity, key):
    """Find an own or inherited descriptor and its owning object."""
    seen = set()
    while identity is not None:
        if identity in seen:
            _fail("prototype cycle")
        seen.add(identity)
        node = _node(machine, identity)
        descriptor = _descriptor(node, key)
        if descriptor is not None:
            return node, descriptor
        prototype = node.get("prototype", "ordinary")
        identity = prototype[1] if isinstance(prototype, list) and prototype[0] == "ref" else None
    return None, None


def _install(node, key, d):
    exists = _descriptor(node, key) is not None
    if not exists:
        if len(set(node["properties"]) | set(node.get("descriptors", {}))) >= 4096:
            _fail("own property bound")
        if "keyOrder" not in node:
            node["keyOrder"] = _ordered(node)
        node["keyOrder"].append(key)
    else:
        # The producer serializes its Map-backed properties and descriptors in
        # insertion order. Its descriptor installation deletes and reinserts
        # an existing key while preserving the separate public keyOrder list.
        node["properties"].pop(key, None)
        node["descriptors"].pop(key, None)
    if d["kind"] == "data":
        node["properties"][key] = copy.deepcopy(d["value"])
        node["descriptors"][key] = {k: copy.deepcopy(v) for k, v in d.items() if k != "value"}
    else:
        node["properties"].pop(key, None)
        node["descriptors"][key] = copy.deepcopy(d)


def _remove(node, key):
    node["properties"].pop(key, None)
    node["descriptors"].pop(key, None)
    if "keyOrder" in node:
        node["keyOrder"] = [item for item in node["keyOrder"] if item != key]


def _array_index(key):
    """Return an array index for a canonical array-index string, else None."""
    if (
        not isinstance(key, str)
        or len(key) > 10
        or not key.isascii()
        or not key.isdecimal()
        or key.startswith("0")
        and key != "0"
    ):
        return None
    index = int(key)
    return index if index < 4294967295 else None


def _array_length(value):
    # ArraySetLength applies ToNumber/ToUint32 before comparing the values.
    # Abrupt conversions need an explicit throw record, not a false success flag.
    try:
        number = v7._to_number(value)
    except TypeError:
        _fail("array length coercion requires an explicit abrupt boundary")
    if not math.isfinite(number) or not 0 <= number <= 4294967295 or int(number) != number:
        _fail("invalid array length")
    return int(number)


def _set_array_length(node, new_length, writable=None):
    """Apply array length shrink/growth, including partial failure at a fixed index."""
    old = _descriptor(node, "length")
    current = old["value"][1]
    if new_length > current:
        node["properties"]["length"] = ["num", new_length]
    elif new_length < current:
        blockers = sorted(
            (
                i
                for key in set(node["properties"]) | set(node["descriptors"])
                if (i := _array_index(key)) is not None and i >= new_length
            ),
            reverse=True,
        )
        for index in blockers:
            key = str(index)
            descriptor = _descriptor(node, key)
            if descriptor is None:
                continue
            if not descriptor["configurable"]:
                node["properties"]["length"] = ["num", index + 1]
                if writable is False:
                    node["descriptors"]["length"] = {
                        "kind": "data",
                        "writable": False,
                        "enumerable": False,
                        "configurable": False,
                    }
                return False
            _remove(node, key)
        node["properties"]["length"] = ["num", new_length]
    if writable is not None:
        node["descriptors"]["length"] = {
            "kind": "data",
            "writable": writable,
            "enumerable": False,
            "configurable": False,
        }
    return True


def _ordered(node):
    keys = []
    descriptors = node.get("descriptors", {})
    for key in [*node.get("keyOrder", []), *node["properties"], *descriptors]:
        if key not in keys and (key in node["properties"] or key in descriptors):
            keys.append(key)
    return (
        sorted((key for key in keys if _array_index(key) is not None), key=int)
        + [key for key in keys if isinstance(key, str) and _array_index(key) is None]
        + [key for key in keys if isinstance(key, tuple)]
    )


def _compatible(node, key, d):
    old = _descriptor(node, key)
    if old is None:
        return node["extensible"]
    if old["configurable"]:
        return True
    if d["configurable"] or d["enumerable"] != old["enumerable"] or d["kind"] != old["kind"]:
        return False
    if old["kind"] == "accessor":
        return old["get"] == d["get"] and old["set"] == d["set"]
    return old["writable"] or not d["writable"] and old["value"] == d["value"]


def _define_own_property(node, key, descriptor):
    """Apply the same descriptor/array rules to definitions and receiver writes."""
    index = None
    if node["kind"] == "array":
        length = _descriptor(node, "length")
        if key == "length":
            if descriptor["kind"] != "data":
                return False
            new_length = _array_length(descriptor["value"])
            descriptor = {**descriptor, "value": ["num", new_length]}
            if not _compatible(node, key, descriptor):
                return False
            return _set_array_length(node, new_length, descriptor["writable"])
        index = _array_index(key)
        if index is not None and index >= length["value"][1] and not length["writable"]:
            return False
    if not _compatible(node, key, descriptor):
        return False
    _install(node, key, descriptor)
    if index is not None and index >= length["value"][1]:
        node["properties"]["length"] = ["num", index + 1]
    return True


def _local(machine, name):
    cell = machine.get("captureAliases", {}).get(name, name)
    if cell not in machine["locals"]:
        _fail("unknown protocol local")
    return machine["locals"][cell]


def _eval(machine, expr):
    if expr["op"] == "literal":
        return _known(machine, expr["value"])
    if expr["op"] == "local":
        return _known(machine, _local(machine, expr["name"]))
    op = expr["op"]
    if op == "unary":
        a = _eval(machine, expr["arg"])
        if a[0] in ("ref", "symbol"):
            if expr["operator"] == "!":
                return ["bool", False]
            if expr["operator"] == "void":
                return ["undefined"]
            if expr["operator"] == "typeof":
                return [
                    "str",
                    "symbol"
                    if a[0] == "symbol"
                    else (
                        "function"
                        if machine["heap"].get(a[1], {}).get("kind") == "closure"
                        else "object"
                    ),
                ]
            _fail("behavioral value coercion needs an abrupt boundary")
        opmap = {
            "!": "not",
            "+": "positive",
            "-": "negate",
            "~": "bit_not",
            "typeof": "type_of",
            "void": "void",
        }
        return _normalize_number(v7._operation(opmap[expr["operator"]], [a]))
    left = _eval(machine, expr["left"])
    right = _eval(machine, expr["right"])
    operator = expr["operator"]
    if "symbol" in (left[0], right[0]):
        if (
            operator not in ("===", "!==", "==", "!=")
            or "ref" in (left[0], right[0])
            and operator in ("==", "!=")
        ):
            _fail("Symbol conversion needs an abrupt boundary")
        same = left[0] == right[0] == "symbol" and left[1] == right[1]
        return ["bool", same if operator in ("===", "==") else not same]
    opmap = {
        "+": "add",
        "-": "subtract",
        "*": "multiply",
        "/": "divide",
        "%": "remainder",
        "**": "power",
        "===": "equal",
        "!==": "not_equal",
        "==": "loose_equal",
        "!=": "loose_not_equal",
        "<": "less",
        "<=": "less_equal",
        ">": "greater",
        ">=": "greater_equal",
        "&": "bit_and",
        "|": "bit_or",
        "^": "bit_xor",
        "<<": "shift_left",
        ">>": "shift_right",
        ">>>": "unsigned_shift_right",
    }
    return _normalize_number(v7._operation(opmap[operator], [left, right]))


def _normalize_number(value):
    if (
        isinstance(value, list)
        and len(value) == 2
        and value[0] == "num"
        and type(value[1]) is float
        and value[1].is_integer()
    ):
        value[1] = int(value[1])
    return value


def _apply(machine, record):
    p = machine["protocol"]
    pending = p["pendingCalls"][-1] if p["pendingCalls"] else None
    if pending and pending["child"] not in machine["stack"] and record["kind"] != "property.finish":
        _fail("property finish must consume child completion")
    if machine.get("completion") is not None:
        _fail("record after completion")
    if not machine["stack"] and record["kind"] not in ("end", "promise.join.step"):
        _fail("active invocation required")
    kind = record["kind"]
    if not machine.get("_internalProtocolTransition"):
        assert_number_next(machine, record)
        assert_generator_next(machine, record)
        assert_proxy_next(machine, record)
    if p.get("construction") is not None:
        assert_construction_next(machine, record)
    if kind in MEMORY_FIELDS:
        if not apply_memory_record(machine, record):
            _fail("unsupported binary-memory transition")
        return
    if kind in ITERATOR_FIELDS:
        if not apply_iterator_record(machine, record):
            _fail("unsupported iterator transition")
        return
    if kind in PROMISE_JOIN_FIELDS:
        if not apply_promise_join_record(machine, record):
            _fail("unsupported Promise join transition")
        return
    if kind in NUMBER_FIELDS:
        if not apply_number_record(machine, record):
            _fail("unsupported Number transition")
        return
    if kind in GENERATOR_FIELDS:
        if not apply_generator_record(machine, record):
            _fail("unsupported generator transition")
        return
    if kind in PROXY_FIELDS:
        if not apply_proxy_record(machine, record):
            _fail("unsupported Proxy transition")
        return
    if kind in CONSTRUCTION_FIELDS:
        if not apply_construction_record(machine, record):
            _fail("unsupported construction transition")
        return
    for field in ("value", "receiver", "result", "decision", "callee"):
        if field in record:
            _known(machine, record[field])
    if "arguments" in record:
        for value in record["arguments"]:
            _known(machine, value)
    if kind.startswith("property.") and "key" in record and record["key"][0] == "symbol":
        _symbol(machine, record["key"][1])
    if kind == "property.prototype" and record["prototype"] is not None:
        _known(machine, record["prototype"])
    if kind in ("symbol.create", "symbol.for"):
        if kind == "symbol.for":
            existing = next(
                (
                    row
                    for row in p["symbols"]
                    if row["kind"] == "registry" and row["key"] == record["key"]
                ),
                None,
            )
            if existing:
                if record["id"] != existing["id"]:
                    _fail("registry identity changed")
                return
        reserved = {row["id"] for row in (p.get("construction") or {}).get("privateNames", [])}
        if (
            any(row["id"] == record["id"] for row in p["symbols"])
            or record["id"] in machine["heap"]
            or record["id"] in reserved
            or not __import__("re").fullmatch(r"o[1-9][0-9]*", record["id"])
        ):
            _fail("Symbol identity collision or format")
        p["symbols"].append(
            {
                "id": record["id"],
                "kind": "unique",
                "description": copy.deepcopy(record["description"]),
            }
            if kind == "symbol.create"
            else {
                "id": record["id"],
                "kind": "registry",
                "description": ["str", record["key"]],
                "key": record["key"],
            }
        )
    elif kind == "symbol.read":
        row = _symbol(machine, record["symbol"][1])
        d = (
            ["str", "Symbol." + row["name"]]
            if row["kind"] == "well-known"
            else row.get("description", ["undefined"])
        )
        expected = (
            d
            if record["operation"] == "description"
            else (["str", row["key"]] if row["kind"] == "registry" else ["undefined"])
            if record["operation"] == "keyFor"
            else ["str", "Symbol(" + (d[1] if d[0] == "str" else "") + ")"]
        )
        if record["result"] != expected:
            _fail("false Symbol conversion claim")
        if record["bind"] is not None:
            machine["locals"][
                machine.get("captureAliases", {}).get(record["bind"], record["bind"])
            ] = record["result"]
    elif kind == "allocate":
        _assert_fresh_heap_identity(machine, record["name"])
        captures = [
            {
                **copy.deepcopy(capture),
                "cell": machine.get("captureAliases", {}).get(capture["cell"], capture["cell"]),
            }
            for capture in record["captures"]
        ]
        if record["objectKind"] == "closure" and record["functionKind"] == "arrow":
            _known(machine, record["lexicalReceiver"], record["name"])
        node = {
            "kind": record["objectKind"],
            "code": record["code"],
            "captures": captures,
            "prototype": "ordinary",
            "properties": {},
            "descriptors": {},
            "extensible": True,
        }
        if record["objectKind"] == "array":
            node["properties"]["length"] = ["num", record["length"]]
        for prop in record["properties"]:
            key = _key(prop["key"])
            _known(machine, prop["value"], record["name"])
            if prop["key"][0] == "symbol":
                _symbol(machine, prop["key"][1])
            index = _array_index(key)
            if (
                key in node["properties"]
                or record["objectKind"] == "array"
                and (key == "length" or index is not None and index >= record["length"])
            ):
                _fail("duplicate or out-of-bounds allocation key")
            node["properties"][key] = copy.deepcopy(prop["value"])
        if record["objectKind"] == "closure":
            node.update(
                functionKind=record["functionKind"], lexicalReceiver=record["lexicalReceiver"]
            )
        machine["heap"][record["name"]] = node
    elif kind == "bind":
        machine["locals"][machine.get("captureAliases", {}).get(record["name"], record["name"])] = (
            _known(machine, record["value"])
        )
    elif kind in ("compute", "choose"):
        result = _eval(machine, record["expr"])
        claimed = record["result"] if kind == "compute" else record["decision"]
        if result != claimed:
            _fail("false protocol computation")
        if kind == "choose":
            if record["path"] not in ("then", "else", "use-left", "evaluate-right"):
                _fail("unsupported short-circuit path")
            truthy = result[0] == "symbol" or v7._truthy(result)
            if record["path"] == "then" and not truthy or record["path"] == "else" and truthy:
                _fail("false branch path")
            if (
                record["path"] == "use-left"
                and not truthy
                or record["path"] == "evaluate-right"
                and truthy
            ):
                _fail("false short-circuit path")
        if kind == "compute" and record["bind"] is not None:
            machine["locals"][
                machine.get("captureAliases", {}).get(record["bind"], record["bind"])
            ] = copy.deepcopy(result)
    elif kind == "call":
        callee = _known(machine, record["callee"])
        node = _node(machine, callee[1]) if callee[0] == "ref" else None
        if (
            node is None
            or node["kind"] != "closure"
            or record["child"] in machine["frameFunctions"]
        ):
            _fail("owned callable required")
        if node.get("generatorFunction") is True and not machine.get("_internalProtocolTransition"):
            _fail("generator call requires its declared creation/resume protocol")
        if node.get("callable", {}).get("classConstructor"):
            _fail("class constructor cannot be called without construction")
        if any(capture["cell"] not in machine["locals"] for capture in node["captures"]):
            _fail("closure capture is not initialized")
        if len(p["callFrames"]) >= 64:
            _fail("active call frame bound")
        p["callFrames"].append(
            {
                "child": record["child"],
                "parent": machine["stack"][-1],
                "priorReceiver": copy.deepcopy(machine.get("receiver", ["undefined"])),
                "priorAliases": [
                    [key, value] for key, value in machine.get("captureAliases", {}).items()
                ],
                "arguments": copy.deepcopy(record["arguments"]),
            }
        )
        machine["receiver"] = copy.deepcopy(
            node.get("lexicalReceiver")
            if node.get("functionKind") == "arrow"
            else record["receiver"]
        )
        machine["captureAliases"] = {row["binding"]: row["cell"] for row in node["captures"]}
        machine["locals"][record["child"] + ":this"] = copy.deepcopy(machine["receiver"])
        for index, value in enumerate(record["arguments"]):
            machine["locals"][record["child"] + ":argument:" + str(index)] = copy.deepcopy(value)
        machine["frameFunctions"][record["child"]] = node["code"]
        machine["stack"].append(record["child"])
    elif kind in ("return", "throw"):
        if machine["stack"][-1] != record["invocation"]:
            _fail("completion invocation")
        if len(machine["stack"]) > 1:
            if not p["callFrames"]:
                _fail("completion requires a protocol call frame")
            frame = p["callFrames"].pop()
            if frame["child"] != record["invocation"]:
                _fail("call frame completion")
            machine["stack"].pop()
            machine["receiver"] = frame["priorReceiver"]
            machine["captureAliases"] = dict(frame["priorAliases"])
            machine["lastReturn"] = {"kind": kind, "value": copy.deepcopy(record["value"])}
        else:
            machine["stack"].pop()
            machine["pending"] = {"kind": kind, "value": copy.deepcopy(record["value"])}
    elif kind == "end":
        if (
            machine["stack"]
            or p["pendingCalls"]
            or p["callFrames"]
            or (machine.get("pending") or {}).get("kind") != record["completion"]
        ):
            _fail("terminal completion or outstanding protocol calls")
        machine["completion"] = copy.deepcopy(machine["pending"])
    elif kind == "property.finish":
        if (
            not pending
            or pending["id"] != record["id"]
            or machine["stack"][-1] != pending["parent"]
        ):
            _fail("property call completion identity")
        returned = machine.get("lastReturn")
        if not returned or returned["kind"] not in ("return", "throw"):
            _fail("property call has no returned completion")
        expected = {
            "kind": "throw" if returned["kind"] == "throw" else "normal",
            "value": copy.deepcopy(
                returned["value"]
                if returned["kind"] == "throw" or pending["operation"] == "get"
                else pending["value"]
            ),
        }
        if record["completion"] != expected:
            _fail("false property completion")
        p["pendingCalls"].pop()
    else:
        node = _node(machine, record["object"])
        key = _key(record["key"]) if "key" in record else None
        if kind == "property.define":
            descriptor = copy.deepcopy(record["descriptor"])
            for field in ("value",) if descriptor["kind"] == "data" else ("get", "set"):
                value = _known(machine, descriptor[field])
                if (
                    field in ("get", "set")
                    and value[0] == "ref"
                    and _node(machine, value[1])["kind"] != "closure"
                ):
                    _fail("property accessor must be callable")
            success = _define_own_property(node, key, descriptor)
            if success != record["success"]:
                _fail("false descriptor compatibility")
            if success and "keyOrder" not in node:
                node["keyOrder"] = _ordered(node)
        elif kind == "property.get":
            _, descriptor = _lookup_descriptor(machine, record["object"], key)
            if (
                descriptor
                and descriptor["kind"] == "accessor"
                and descriptor["get"] != ["undefined"]
            ):
                _fail("getter needs call boundary")
            expected = (
                descriptor["value"]
                if descriptor and descriptor["kind"] == "data"
                else ["undefined"]
            )
            if expected != record["result"]:
                _fail("false property read")
        elif kind == "property.set":
            _known(machine, record["value"])
            _, descriptor = _lookup_descriptor(machine, record["object"], key)
            if (
                descriptor
                and descriptor["kind"] == "accessor"
                and descriptor["set"] != ["undefined"]
            ):
                _fail("setter needs call boundary")
            receiver = record["receiver"]
            receiver_node = _node(machine, receiver[1]) if receiver[0] == "ref" else None
            success = (
                descriptor is None or descriptor["kind"] == "data" and descriptor["writable"]
            ) and receiver_node is not None
            if success:
                receiver_own = _descriptor(receiver_node, key)
                if receiver_own is not None and (
                    receiver_own["kind"] != "data" or not receiver_own["writable"]
                ):
                    success = False
                else:
                    current = receiver_own or {
                        "kind": "data",
                        "writable": True,
                        "enumerable": True,
                        "configurable": True,
                    }
                    success = _define_own_property(
                        receiver_node, key, {**current, "value": copy.deepcopy(record["value"])}
                    )
            if success != record["success"]:
                _fail("false property write")
        elif kind == "property.delete":
            descriptor = _descriptor(node, key)
            success = descriptor is None or descriptor["configurable"]
            if success:
                _remove(node, key)
            if success != record["success"]:
                _fail("false property delete")
        elif kind == "property.keys":
            keys = [_public(item) for item in _ordered(node)]
            if record["mode"] in ("strings", "enumerable-strings"):
                keys = [k for k in keys if k[0] == "str"]
            if record["mode"] == "symbols":
                keys = [k for k in keys if k[0] == "symbol"]
            if record["mode"] == "enumerable-strings":
                keys = [k for k in keys if _descriptor(node, _key(k))["enumerable"]]
            if keys != record["keys"]:
                _fail("false property key order")
            array_record = {
                "kind": "allocate",
                "name": record["name"],
                "objectKind": "array",
                "length": len(keys),
                "properties": [
                    {"key": ["str", str(i)], "value": key} for i, key in enumerate(keys)
                ],
                "code": None,
                "captures": [],
                "functionKind": None,
                "lexicalReceiver": None,
            }
            _apply(machine, array_record)
        elif kind == "property.preventExtensions":
            node["extensible"] = False
        elif kind == "property.prototype":
            if isinstance(record["prototype"], list):
                _known(machine, record["prototype"])
            before = node.get("prototype", "ordinary")
            success = before == record["prototype"] or node["extensible"]
            if success and isinstance(record["prototype"], list):
                target = record["prototype"][1]
                seen = set()
                while True:
                    if target == record["object"] or target in seen:
                        success = False
                        break
                    seen.add(target)
                    ancestor = _node(machine, target)
                    parent = ancestor.get("prototype", "ordinary")
                    if not isinstance(parent, list) or parent[0] != "ref":
                        break
                    target = parent[1]
            if success:
                node["prototype"] = copy.deepcopy(record["prototype"])
            if success != record["success"]:
                _fail("false prototype transition")
        elif kind in ("property.get.call", "property.set.call"):
            _, descriptor = _lookup_descriptor(machine, record["object"], key)
            operation = "get" if kind.endswith("get.call") else "set"
            callee = (
                descriptor.get(operation)
                if descriptor and descriptor["kind"] == "accessor"
                else None
            )
            if not callee or callee[0] != "ref":
                _fail("property accessor callable required")
            parent = machine["stack"][-1]
            call_record = {
                "kind": "call",
                "child": record["child"],
                "callee": callee,
                "receiver": record["receiver"],
                "arguments": [] if operation == "get" else [record["value"]],
            }
            _apply(machine, call_record)
            p["pendingCalls"].append(
                {
                    "id": record["id"],
                    "operation": operation,
                    "child": record["child"],
                    "parent": parent,
                    "object": record["object"],
                    "key": copy.deepcopy(record["key"]),
                    "receiver": copy.deepcopy(record["receiver"]),
                    "callee": copy.deepcopy(callee),
                    "value": None if operation == "get" else copy.deepcopy(record["value"]),
                }
            )
        else:
            _fail("unsupported transition")


def apply_protocol_record(machine, record):
    """Apply one checked row transactionally to a native JSON-safe machine."""
    validate_protocol_record(record)
    staged = copy.deepcopy(machine)
    if "protocol" not in staged:
        staged["protocol"] = create_protocol_state()
    if staged["protocol"].get("schema") != STATE_SCHEMA:
        _fail("protocol state version")
    if staged["protocol"]["sequence"] >= 32768:
        _fail("protocol step budget")
    _apply(staged, record)
    staged["protocol"]["sequence"] += 1
    machine.clear()
    machine.update(staged)
    return {
        "status": "committed",
        "kind": record["kind"],
        "sequence": machine["protocol"]["sequence"],
        "completion": copy.deepcopy(machine.get("completion")),
    }


def _internal_machine(initial):
    machine = copy.deepcopy(initial)
    machine["heap"] = {}
    for identity, node in initial["heap"]:
        n = copy.deepcopy(node)
        n["properties"] = {
            _key(key) if isinstance(key, list) else key: value
            for key, value in n.get("properties", [])
        }
        if n.get("kind") not in (
            "array-buffer",
            "shared-int32-buffer",
            "typed-array",
            "shared-int32-view",
            "data-view",
            "shared-data-view",
            "iterator",
        ):
            if "descriptors" in n:
                n["descriptors"] = {
                    _key(key) if isinstance(key, list) else key: value
                    for key, value in n.get("descriptors", [])
                }
            if "keyOrder" in n:
                n["keyOrder"] = [
                    _key(key) if isinstance(key, list) else key for key in n.get("keyOrder", [])
                ]
        if isinstance(n.get("privateElements"), list):
            n["privateElements"] = {
                key: copy.deepcopy(value) for key, value in n["privateElements"]
            }
        machine["heap"][identity] = n
    for field in ("locals", "captureAliases", "frameFunctions"):
        machine[field] = (
            dict(initial[field])
            if isinstance(initial[field], list)
            else copy.deepcopy(initial[field])
        )
    return machine


def _external_machine(machine):
    result = copy.deepcopy(machine)
    result["heap"] = []
    for identity, node in machine["heap"].items():
        n = copy.deepcopy(node)
        n["properties"] = [
            [key if isinstance(key, str) else ["symbol", key[1]], value]
            for key, value in n.get("properties", {}).items()
        ]
        if n.get("descriptors"):
            n["descriptors"] = [
                [key if isinstance(key, str) else ["symbol", key[1]], value]
                for key, value in n["descriptors"].items()
            ]
        else:
            n.pop("descriptors", None)
        if n.get("keyOrder"):
            n["keyOrder"] = [
                key if isinstance(key, str) else ["symbol", key[1]] for key in n.get("keyOrder", [])
            ]
        else:
            n.pop("keyOrder", None)
        if n.get("extensible", True) is True and not n.get("descriptors"):
            n.pop("extensible", None)
        if isinstance(n.get("privateElements"), dict):
            n["privateElements"] = [
                [key, copy.deepcopy(value)] for key, value in n["privateElements"].items()
            ]
        result["heap"].append([identity, n])
    for field in ("locals", "captureAliases", "frameFunctions"):
        result[field] = [[key, value] for key, value in machine[field].items()]
    return result


def replay_protocol_case(case):
    if not isinstance(case, dict) or not isinstance(case.get("lines"), list):
        _fail("protocol case shape")
    machine = _internal_machine(case["initialState"])
    for number, line in enumerate(case["lines"]):
        try:
            record = decode_protocol_record(line)
        except ProtocolV8Error as error:
            raise ProtocolV8Error(f"protocol line {number}: {error}") from error
        apply_protocol_record(machine, record)
        if number + 1 in case.get("checkpoints", []):
            serialized = json.loads(
                json.dumps(_external_machine(machine), separators=(",", ":"), ensure_ascii=False)
            )
            machine = _internal_machine(serialized)
            if machine["protocol"]["sequence"] != number + 1:
                _fail("checkpoint sequence")
            validate_protocol_state(
                machine["protocol"], machine["heap"], _external_machine(machine)
            )
    return _external_machine(machine)


def validate_protocol_state(snapshot, heap_ids=(), shared_state=None):
    base_fields = {
        "schema",
        "symbols",
        "pendingCalls",
        "callFrames",
        "sequence",
        "numberConversions",
        "proxyCalls",
    }
    if type(snapshot) is dict and "construction" in snapshot:
        _exact(snapshot, base_fields | {"construction"}, "shared protocol state")
    else:
        _exact(snapshot, base_fields, "shared protocol state")
    heap_nodes = None
    if (
        isinstance(heap_ids, dict)
        or isinstance(heap_ids, list)
        and all(isinstance(row, list) and len(row) == 2 for row in heap_ids)
    ):
        heap_nodes = {}
        for identity, node in heap_ids.items() if isinstance(heap_ids, dict) else heap_ids:
            if identity in heap_nodes:
                _fail("duplicate heap identity")
            converted = copy.deepcopy(node)
            for field in ("properties", "descriptors"):
                if isinstance(converted.get(field), list):
                    converted[field] = {
                        _key(key) if isinstance(key, list) else key: value
                        for key, value in converted[field]
                    }
            if isinstance(converted.get("keyOrder"), list):
                converted["keyOrder"] = [
                    _key(key) if isinstance(key, list) else key for key in converted["keyOrder"]
                ]
            if isinstance(converted.get("privateElements"), list):
                converted["privateElements"] = {
                    key: value for key, value in converted["privateElements"]
                }
            heap_nodes[identity] = converted
    if (
        snapshot["schema"] != STATE_SCHEMA
        or type(snapshot["sequence"]) is not int
        or not 0 <= snapshot["sequence"] <= 32768
    ):
        _fail("protocol state schema or sequence")
    if not isinstance(snapshot["symbols"], list) or len(snapshot["symbols"]) > 4096:
        _fail("symbol table bound")
    identities = set()
    registry = set()
    for row in snapshot["symbols"]:
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("id"), str)
            or row["id"] in identities
        ):
            _fail("duplicate symbol identity")
        identities.add(row["id"])
        if row["kind"] == "well-known":
            _exact(row, {"id", "kind", "name"}, "well-known symbol")
            if row["name"] not in WELL_KNOWN or row["id"] != "well-known:" + row["name"]:
                _fail("well-known symbol table")
        elif row["kind"] == "unique":
            _exact(row, {"id", "kind", "description"}, "unique symbol")
            if not __import__("re").fullmatch(r"o[1-9][0-9]*", row["id"]):
                _fail("unique symbol identity")
            validate_protocol_value(row["description"])
            if row["description"][0] not in ("str", "undefined"):
                _fail("symbol description")
        elif row["kind"] == "registry":
            _exact(row, {"id", "kind", "description", "key"}, "registry symbol")
            if (
                not __import__("re").fullmatch(r"o[1-9][0-9]*", row["id"])
                or not isinstance(row["key"], str)
                or row["key"] in registry
                or row["description"] != ["str", row["key"]]
            ):
                _fail("registry symbol")
            registry.add(row["key"])
        else:
            _fail("symbol kind")
        if row["id"] in (heap_nodes if heap_nodes is not None else heap_ids):
            _fail("symbol/heap identity collision")
    if not all(
        any(
            row["id"] == "well-known:" + name and row["kind"] == "well-known"
            for row in snapshot["symbols"]
        )
        for name in WELL_KNOWN
    ):
        _fail("well-known table incomplete")
    if (
        not isinstance(snapshot["pendingCalls"], list)
        or len(snapshot["pendingCalls"]) > 64
        or not isinstance(snapshot["callFrames"], list)
        or len(snapshot["callFrames"]) > 64
    ):
        _fail("pending calls/frame bound")
    pending_ids = set()
    for pending in snapshot["pendingCalls"]:
        _exact(
            pending,
            {"id", "operation", "child", "parent", "object", "key", "receiver", "callee", "value"},
            "pending property call",
        )
        if pending["id"] in pending_ids or pending["operation"] not in ("get", "set"):
            _fail("pending call identity or kind")
        pending_ids.add(pending["id"])
        _identity(pending["child"])
        _identity(pending["parent"])
        _identity(pending["object"])
        validate_protocol_key(pending["key"])
        validate_protocol_value(pending["receiver"])
        validate_protocol_value(pending["callee"])
        if pending["callee"][0] != "ref":
            _fail("pending accessor closure")
        if pending["operation"] == "get" and pending["value"] is not None:
            _fail("getter pending value")
        if pending["operation"] == "set":
            validate_protocol_value(pending["value"])
    for frame in snapshot["callFrames"]:
        _exact(
            frame,
            {"child", "parent", "priorReceiver", "priorAliases", "arguments"},
            "protocol call frame",
        )
        _identity(frame["child"])
        _identity(frame["parent"])
        validate_protocol_value(frame["priorReceiver"])
        if not isinstance(frame["priorAliases"], list) or not isinstance(frame["arguments"], list):
            _fail("call frame arrays")
        seen = set()
        for pair in frame["priorAliases"]:
            if (
                not isinstance(pair, list)
                or len(pair) != 2
                or any(not isinstance(x, str) for x in pair)
                or pair[0] in seen
            ):
                _fail("call frame aliases")
            seen.add(pair[0])
        for value in frame["arguments"]:
            validate_protocol_value(value)
    if snapshot.get("construction") is not None:
        if heap_nodes is None:
            _fail("construction validation requires full shared heap")
        construction_machine = (
            _internal_machine(shared_state)
            if shared_state is not None
            else {"heap": heap_nodes, "protocol": snapshot}
        )
        validate_construction_record_state(snapshot["construction"], construction_machine)
    if shared_state is not None:
        active_machine = _internal_machine(shared_state)
        for binding, cell in active_machine.get("captureAliases", {}).items():
            if (
                not isinstance(binding, str)
                or not isinstance(cell, str)
                or cell not in active_machine["locals"]
            ):
                _fail("active closure capture is not initialized")
            _known(active_machine, active_machine["locals"][cell])
    if heap_nodes is not None:
        known_machine = {"heap": heap_nodes, "protocol": snapshot}
        for identity, node in heap_nodes.items():
            if not isinstance(node, dict):
                continue
            if len(set(node.get("properties", {})) | set(node.get("descriptors", {}))) > 4096:
                _fail("own property bound")
            for key, value in node.get("properties", {}).items():
                if isinstance(key, tuple):
                    _symbol(known_machine, key[1])
                _known(known_machine, value, identity)
            for key, descriptor in node.get("descriptors", {}).items():
                if isinstance(key, tuple):
                    _symbol(known_machine, key[1])
                if descriptor.get("kind") == "data":
                    _exact(
                        descriptor,
                        {"kind", "writable", "enumerable", "configurable"},
                        "stored data descriptor",
                    )
                    if key not in node.get("properties", {}):
                        _fail("data descriptor lacks stored value")
                    _validate_descriptor({**descriptor, "value": node["properties"][key]})
                else:
                    _validate_descriptor(descriptor)
                    for field in ("get", "set"):
                        value = _known(known_machine, descriptor[field], identity)
                        if (
                            value[0] == "ref"
                            and _node(known_machine, value[1])["kind"] != "closure"
                        ):
                            _fail("property accessor must be callable")
            prototype = node.get("prototype", "ordinary")
            if isinstance(prototype, list):
                _known(known_machine, prototype, identity)
            if node.get("functionKind") == "arrow":
                _known(known_machine, node.get("lexicalReceiver"), identity)
        memory_machine = (
            _internal_machine(shared_state) if shared_state is not None else {"heap": heap_nodes}
        )
        validate_memory_state(memory_machine)
        validate_iterator_state(memory_machine)
        from .promise_join_v8 import validate_state as validate_promise_join_state

        validate_promise_join_state(memory_machine)
        validate_number_state(memory_machine)
        validate_generator_state(memory_machine)
        validate_proxy_state(memory_machine)
    return True


def apply_protocol_to_shared_state(shared_state, line):
    """Apply a protocol/1 line to the v8 shared heap/locals/call stack."""
    record = decode_protocol_record(line)
    machine = _internal_machine(shared_state)
    if machine.get("protocol") is None:
        machine["protocol"] = create_protocol_state()
    external = _external_machine(machine)
    validate_protocol_state(external["protocol"], external["heap"], external)
    apply_protocol_record(machine, record)
    return _external_machine(machine)


def assert_detailed_record_compatible(shared_state, kind, fields):
    """Reject legacy detail that would bypass rich protocol property state."""
    protocol = shared_state.get("protocol")
    if protocol is None:
        return True
    machine = _internal_machine(shared_state)
    assert_construction_next(machine, {"kind": kind, **fields})
    assert_number_next(machine, {"kind": kind, **fields})
    assert_generator_next(machine, {"kind": kind, **fields})
    assert_proxy_next(machine, {"kind": kind, **fields})

    def rich(node):
        return bool(
            node.get("descriptors")
            or node.get("extensible") is False
            or node.get("prototype") not in (None, "ordinary")
            or any(isinstance(key, tuple) for key in node.get("properties", {}))
            or node.get("kind") == "generator"
            or node.get("generatorFunction") is True
            or node.get("kind") == "proxy"
        )

    def inspect(value):
        if not isinstance(value, (dict, list)):
            return
        if isinstance(value, list):
            if value and value[0] == "symbol":
                _fail("Symbol data must use protocol records")
            for child in value:
                inspect(child)
        else:
            if value.get("op") == "read":
                node = machine["heap"].get(value.get("object"))
                if node and rich(node):
                    _fail("Detailed read bypasses descriptor/prototype protocol")
            for child in value.values():
                inspect(child)

    inspect(fields)
    if kind in ("read", "write"):
        node = machine["heap"].get(fields.get("object"))
        if node and rich(node):
            _fail("Detailed property transition bypasses rich protocol state")
    if kind in ("call", "construct"):
        callee = fields.get("callee")
        identity = (
            callee[1]
            if isinstance(callee, list) and len(callee) == 2 and callee[0] == "ref"
            else callee
        )
        target = machine["heap"].get(identity)
        if target and target.get("generatorFunction") is True:
            _fail("Detailed call bypasses generator creation/resume protocol")
    if shared_state["protocol"]["pendingCalls"] and kind == "end":
        _fail("Unfinished accessor call")
    if kind in ("return", "throw") and machine["stack"]:
        child = fields.get("invocation", machine["stack"][-1])
        if any(frame["child"] == child for frame in protocol["callFrames"]):
            _fail("Protocol-owned invocation requires protocol completion")
    return True
