"""Strict, independent semantic-trace/3 record validation.

Unlike the archival version-two event projection this grammar describes bounded
public computations. All source/journal/certificate evidence stays outside
model-visible physical records. This module intentionally imports no tracer JS.
"""
from __future__ import annotations

import json
import math
import re
from typing import Any


class WitnessV3Error(ValueError):
    pass

FIELDS = {
    "code": {"entry", "source", "sites", "randomSeed"},
    "input": {"graph"},
    "compute": {"expr", "result", "bind", "invocation", "site"},
    "choose": {"expr", "decision", "path", "invocation", "site"},
    "allocate": {"name", "kind", "length", "properties", "code", "captures", "invocation"},
    "read": {"object", "key", "value", "present", "invocation"},
    "write": {"object", "key", "value", "present", "invocation"},
    "bind": {"name", "value", "from", "invocation"},
    "call": {"child", "callee", "receiver", "arguments", "invocation"},
    "transfer": {"kind", "value", "handler", "invocation"},
    "return": {"value", "invocation"},
    "throw": {"value", "invocation"},
    "request": {"effect", "operation", "arguments", "invocation"},
    "observe": {"effect", "result", "status"},
    "job": {"action", "job", "owner", "result"},
    "end": {"kind"},
    "value-open": {"id", "type"},
    "value-part": {"id", "data"},
    "value-close": {"id"},
}
REQUIRED = {
    "code": {"entry", "source", "sites"},
    "input": {"graph"},
    "compute": {"expr", "result"},
    "choose": {"expr", "decision", "path"},
    "allocate": {"name", "kind", "properties"},
    "read": {"object", "key", "value", "present"},
    "write": {"object", "key", "value", "present"},
    "bind": {"name", "value"},
    "call": {"child", "callee", "receiver", "arguments"},
    "transfer": {"kind", "value", "handler"},
    "return": {"value"},
    "throw": {"value"},
    "request": {"effect", "operation", "arguments"},
    "observe": {"effect", "result", "status"},
    "job": {"action", "job", "owner"},
    "end": {"kind"},
    "value-open": {"id", "type"},
    "value-part": {"id", "data"},
    "value-close": {"id"},
}
IDENTITY = re.compile(r"[a-z][a-z0-9]*\Z")
BIGINT = re.compile(r"(?:0|-?[1-9][0-9]*)\Z")
SPECIAL = {"-0", "NaN", "+Infinity", "-Infinity"}
OPERATORS = {"+", "-", "*", "/", "%", "**", "<", "<=", ">", ">=",
             "===", "!==", "==", "!=", "&", "|", "^", "<<", ">>", ">>>"}
UNARY = {"!", "+", "-", "~", "typeof", "void"}

def _fail(reason: str) -> None:
    raise WitnessV3Error("Invalid semantic-trace/3: " + reason)

def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail("Duplicate JSON field")
        result[key] = value
    return result

def value(tag: Any) -> None:
    if not isinstance(tag, list) or not tag:
        _fail("Untagged public value")
    name = tag[0]
    if name in ("null", "undefined") and len(tag) == 1:
        return
    if len(tag) != 2:
        _fail("Incorrect scalar arity")
    arg = tag[1]
    if name == "bool" and type(arg) is bool:
        return
    if name == "str" and isinstance(arg, str):
        return
    if name == "bigint" and isinstance(arg, str) and BIGINT.fullmatch(arg):
        return
    if name == "num" and (
        (type(arg) in (float, int) and math.isfinite(arg) and
         not (arg == 0 and math.copysign(1, arg) < 0))
        or (isinstance(arg, str) and arg in SPECIAL)
    ):
        return
    if name == "ref" and isinstance(arg, str) and IDENTITY.fullmatch(arg):
        return
    _fail("Unsupported or noncanonical scalar")

def expression(expr: Any) -> dict[str, int]:
    nodes = operators = operands = 0
    def visit(node: Any) -> None:
        nonlocal nodes, operators, operands
        if not isinstance(node, dict):
            _fail("Malformed expression")
        nodes += 1
        if nodes > 32:
            _fail("Expression node limit")
        op = node.get("op")
        if op == "random" and set(node) == {"op"}:
            operators += 1
        elif op == "literal" and set(node) == {"op", "value"}:
            value(node["value"])
            operands += 1
        elif op == "local" and set(node) == {"op", "name"} and isinstance(node["name"], str) and IDENTITY.fullmatch(node["name"]):
            operands += 1
        elif op == "read" and set(node) == {"op", "object", "key"} and isinstance(node["object"], str) and IDENTITY.fullmatch(node["object"]) and isinstance(node["key"], str):
            operands += 1
        elif op == "unary" and set(node) == {"op", "operator", "arg"} and node["operator"] in UNARY:
            operators += 1
            visit(node["arg"])
        elif op == "binary" and set(node) == {"op", "operator", "left", "right"} and node["operator"] in OPERATORS:
            operators += 1
            visit(node["left"])
            visit(node["right"])
        else:
            _fail("Unregistered primitive expression")
    visit(expr)
    if operators > 8:
        _fail("Expression operator limit")
    return {"operatorCount": operators, "operandCount": operands, "nodeCount": nodes}

def graph(data: Any) -> None:
    if not isinstance(data, dict) or set(data) != {"roots", "objects"}:
        _fail("Input graph envelope")
    roots, objects = data["roots"], data["objects"]
    if not isinstance(roots, list) or not isinstance(objects, list) or len(objects) > 256:
        _fail("Input graph budget")
    ids: set[str] = set()
    for node in objects:
        if not isinstance(node, dict) or set(node) not in (
            {"id", "kind", "properties"},
            {"id", "kind", "prototype", "properties"},
        ):
            _fail("Input graph node fields")
        name = node["id"]
        if (not isinstance(name, str) or not IDENTITY.fullmatch(name) or
            name in ids or node["kind"] not in ("object", "array")):
            _fail("Input graph identity")
        if "prototype" in node and (node["kind"] != "object" or node["prototype"] is not None):
            _fail("Input graph prototype")
        ids.add(name)
        properties = node["properties"]
        if not isinstance(properties, list) or len(properties) > 4096:
            _fail("Graph property budget")
        keys: set[str] = set()
        for property in properties:
            if not isinstance(property, dict) or set(property) != {"key", "value"}:
                _fail("Graph property fields")
            key = property["key"]
            if not isinstance(key, str) or key == "__proto__" or key in keys:
                _fail("Graph property key")
            keys.add(key)
            value(property["value"])
        if node["kind"] == "array":
            length = next((p["value"] for p in properties if p["key"] == "length"), None)
            if not isinstance(length, list) or len(length) != 2 or length[0] != "num" or type(length[1]) is not int or not 0 <= length[1] <= 4096:
                _fail("Array length")
            for key in keys:
                if key.isascii() and key.isdecimal() and str(int(key)) == key and int(key) >= length[1]:
                    _fail("Array index past length")
    for item in [*roots, *(p["value"] for n in objects for p in n["properties"])]:
        value(item)
        if item[0] == "ref" and item[1] not in ids:
            _fail("Dangling input reference")

def parse_record(line: Any) -> tuple[str, dict[str, Any]]:
    if not isinstance(line, str) or "\n" in line or "\r" in line:
        _fail("Trace record is not one line")
    kind, separator, body = line.partition(" ")
    if not separator or kind not in FIELDS:
        _fail("Unknown version-three operation")
    try:
        fields = json.loads(body, object_pairs_hook=_pairs,
                            parse_constant=lambda value: _fail("Nonfinite JSON literal"))
    except (ValueError, TypeError) as error:
        raise WitnessV3Error("Malformed semantic-trace/3 JSON") from error
    if not isinstance(fields, dict) or not REQUIRED[kind] <= set(fields) <= FIELDS[kind]:
        _fail("Unknown or missing typed operation fields")
    if kind == "code":
        if (not isinstance(fields["entry"], str) or not fields["entry"].replace("$", "_").replace("\u200c", "_").replace("\u200d", "_").isidentifier()
            or not isinstance(fields["source"], str) or not fields["source"]
            or not isinstance(fields["sites"], list)):
            _fail("Missing static code bundle")
        seed = fields.get("randomSeed")
        if seed is not None and (type(seed) is not int or not 0 <= seed <= 0xFFFFFFFF):
            _fail("Invalid published deterministic source seed")
    elif kind == "input":
        graph(fields["graph"])
    elif kind in ("compute", "choose"):
        expression(fields["expr"])
        value(fields["result"] if kind == "compute" else fields["decision"])
        if kind == "choose" and fields["path"] not in ("then", "else", "use-left", "evaluate-right"):
            _fail("Unregistered branch path")
    elif kind in ("read", "write", "bind"):
        value(fields["value"])
    elif kind in ("return", "throw", "transfer"):
        value(fields["value"])
    elif kind == "allocate":
        if fields["kind"] not in ("array", "object", "closure", "set", "regexp"):
            _fail("Unregistered allocation kind")
        if "captures" in fields:
            if fields["kind"] != "closure" or not isinstance(fields["captures"], list):
                _fail("Invalid closure capture owner")
            used: set[str] = set()
            for relation in fields["captures"]:
                if (not isinstance(relation, dict) or set(relation) != {"binding", "cell"}
                    or not isinstance(relation["binding"], str)
                    or relation["binding"] in used
                    or not isinstance(relation["cell"], str)
                    or not IDENTITY.fullmatch(relation["cell"])):
                    _fail("Malformed shared cell capture relation")
                used.add(relation["binding"])
    elif kind == "observe":
        value(fields["result"])
        if fields["status"] not in ("fulfilled", "rejected"):
            _fail("Invalid observation status")
    elif kind == "end":
        if fields["kind"] not in ("return", "throw"):
            _fail("Invalid terminal completion")
    if kind == "job" and fields["action"] not in ("enqueue", "dispatch", "complete"):
        _fail("Invalid owned job transition")
    if kind in ("read", "write") and (type(fields["present"]) is not bool or not isinstance(fields["key"], str)):
        _fail("Property presence contract")
    for key in ("name", "bind", "from", "object", "invocation", "child"):
        if key in fields and (not isinstance(fields[key], str) or not IDENTITY.fullmatch(fields[key])):
            _fail("Invalid witness identity")
    return kind, fields
