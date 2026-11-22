"""Independent codec and semantic core for semantic algorithm v7.

This module accepts only the v7 typed record language and v7 core shape.  It
does not execute JavaScript or authorize source/effect boundaries.  References
are opaque identities resolved only through the caller supplied data heap.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any

from .semantic_json import (
    SemanticJSONError as AlgorithmV7Error,
)
from .semantic_json import (
    _array_index as _array_index,
)
from .semantic_json import (
    _as_js_number,
    _js_string_length,
    _js_utf8,
    _number_text,
    _quote_string,
    stable_algorithm_hash,
    stable_algorithm_json,
)
from .semantic_json import (
    _stable_json as _stable_json,
)
from .semantic_primitives import (
    _ID as _ID,
)
from .semantic_primitives import (
    _MISSING as _MISSING,
)
from .semantic_primitives import (
    _SPECIAL_NUMBERS as _SPECIAL_NUMBERS,
)
from .semantic_primitives import (
    OPERATIONS as OPERATIONS,
)
from .semantic_primitives import (
    UNDEFINED as UNDEFINED,
)
from .semantic_primitives import SemanticRangeError
from .semantic_primitives import (
    _bigint_string as _bigint_string,
)
from .semantic_primitives import (
    _data_list as _data_list,
)
from .semantic_primitives import (
    _decode_scalar as _decode_scalar,
)
from .semantic_primitives import (
    _dense_items as _dense_items,
)
from .semantic_primitives import (
    _encode_scalar as _encode_scalar,
)
from .semantic_primitives import (
    _fields as _fields,
)
from .semantic_primitives import (
    _heap_map as _heap_map,
)
from .semantic_primitives import (
    _heap_node as _heap_node,
)
from .semantic_primitives import (
    _identity as _identity,
)
from .semantic_primitives import (
    _int32 as _int32,
)
from .semantic_primitives import (
    _js_divide as _js_divide,
)
from .semantic_primitives import (
    _js_power as _js_power,
)
from .semantic_primitives import (
    _js_remainder as _js_remainder,
)
from .semantic_primitives import (
    _lookup as _lookup,
)
from .semantic_primitives import (
    _loose_equal as _loose_equal,
)
from .semantic_primitives import (
    _number_from_string as _number_from_string,
)
from .semantic_primitives import (
    _operation as _operation,
)
from .semantic_primitives import (
    _plain_dict as _plain_dict,
)
from .semantic_primitives import (
    _primitive_pair as _primitive_pair,
)
from .semantic_primitives import (
    _property_rows as _property_rows,
)
from .semantic_primitives import (
    _relational as _relational,
)
from .semantic_primitives import (
    _snapshot_heap as _snapshot_heap,
)
from .semantic_primitives import (
    _strict_equal as _strict_equal,
)
from .semantic_primitives import (
    _tag as _tag,
)
from .semantic_primitives import (
    _to_number as _to_number,
)
from .semantic_primitives import (
    _to_numeric as _to_numeric,
)
from .semantic_primitives import (
    _to_string as _to_string,
)
from .semantic_primitives import (
    _truthy as _truthy,
)
from .semantic_primitives import (
    _uint32 as _uint32,
)
from .semantic_primitives import (
    _Undefined as _Undefined,
)
from .semantic_primitives import (
    _validate_value as _validate_value,
)
from .semantic_primitives import (
    algorithm_heap_from_graph as algorithm_heap_from_graph,
)
from .semantic_primitives import (
    canonical_algorithm_graph as canonical_algorithm_graph,
)

AlgorithmV7RangeError = SemanticRangeError

ALGORITHM_VERSION = "semantic-algorithm/1"
ALGORITHM_PROFILE = "ecmascript-primitive/1"
TRACE_VERSION = "semantic-trace/7"
INPUT_TYPES = ("scalar", "number", "dense-number-array", "data")
LIMITS = {
    "steps": 256,
    "term_nodes": 512,
    "depth": 64,
    "fold_items": 128,
    "record_bytes": 65536,
    "arguments": 128,
}
_PLAIN_NAME = re.compile(r"^[A-Za-z_$][A-Za-z0-9_$]*$")
_PLAIN_SYMBOL = re.compile(r"^[A-Za-z_$][A-Za-z0-9_$]*(?:\.[A-Za-z_$][A-Za-z0-9_$]*)*$")
_RESERVED_NAMES = {"request", "observe", "write", "return", "throw", "suspend"}
_NUMBER_TOKEN = re.compile(r"^-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?")
_BIGINT_TOKEN = re.compile(r"^-?(?:0|[1-9][0-9]*)n")
_IDENTITY_TOKEN = re.compile(r"^[A-Za-z_$][A-Za-z0-9_$]*")
_SYMBOL_TOKEN = re.compile(r"^[A-Za-z_$][A-Za-z0-9_$]*(?:\.[A-Za-z_$][A-Za-z0-9_$]*)*")


def _error(message: str) -> None:
    raise AlgorithmV7Error(f"Invalid {ALGORITHM_VERSION}: {message}")


def algorithm_digest(core: Any) -> str:
    validate_algorithm(core)
    return hashlib.sha256(stable_algorithm_json(core).encode("utf-8")).hexdigest()


def _term(
    term: Any,
    *,
    inputs: set[str] | None = None,
    values: set[str] | None = None,
    max_nodes: int = LIMITS["term_nodes"],
) -> dict:
    nodes = 0
    dependencies: list[str] = []
    input_dependencies: list[str] = []
    dependency_set: set[str] = set()
    input_set: set[str] = set()
    active: set[int] = set()

    def visit(node: Any, depth: int = 0) -> None:
        nonlocal nodes
        if not _plain_dict(node):
            _error("term shape or budget")
        identity = id(node)
        if identity in active or depth > LIMITS["depth"] or nodes + 1 > max_nodes:
            _error("term shape or budget")
        nodes += 1
        active.add(identity)
        op = node.get("op")
        if op == "literal":
            _fields(node, ("op", "value"), "literal")
            _validate_value(node["value"])
        elif op in ("input", "ref"):
            _fields(node, ("op", "id"), "dependency")
            dep = _identity(node["id"])
            scope = inputs if op == "input" else values
            if scope is not None and dep not in scope:
                _error(f"unbound {op} {dep}")
            selected = input_set if op == "input" else dependency_set
            ordered = input_dependencies if op == "input" else dependencies
            if dep not in selected:
                selected.add(dep)
                ordered.append(dep)
        elif op == "project":
            _fields(node, ("op", "object", "key"), "projection")
            if type(node["key"]) is not str or node["key"] == "__proto__":
                _error("data property key")
            visit(node["object"], depth + 1)
        elif op == "fold_left":
            _fields(node, ("op", "input", "initial", "step", "order", "profile"), "fold")
            if (node["step"], node["order"], node["profile"]) != (
                "add",
                "ascending",
                "dense-number-array/1",
            ):
                _error("ordered fold profile")
            visit(node["input"], depth + 1)
            visit(node["initial"], depth + 1)
        else:
            if type(op) is not str or op not in OPERATIONS:
                _error(f"unknown operation {op!s}")
            _fields(node, ("op", "args"), "operation")
            args = _data_list(node["args"], "operation arguments")
            if len(args) != OPERATIONS[op][0]:
                _error("operation arity")
            for arg in args:
                visit(arg, depth + 1)
        active.remove(identity)

    visit(term)
    return {"nodes": nodes, "dependencies": dependencies, "inputs": input_dependencies}


def validate_term(
    term: Any,
    *,
    inputs: set[str] | None = None,
    values: set[str] | None = None,
    max_nodes: int = LIMITS["term_nodes"],
) -> dict:
    return _term(term, inputs=inputs, values=values, max_nodes=max_nodes)


def validate_algorithm(core: Any) -> dict:
    _fields(core, ("schema", "profile", "inputs", "steps"), "core")
    if core["schema"] != ALGORITHM_VERSION or core["profile"] != ALGORITHM_PROFILE:
        _error("version or semantic profile")
    inputs_list = _data_list(core["inputs"], "inputs")
    steps = _data_list(core["steps"], "steps")
    if len(inputs_list) > 128 or not steps or len(steps) > LIMITS["steps"]:
        _error("core budget")
    inputs: set[str] = set()
    values: set[str] = set()
    requests: set[str] = set()
    observed: set[str] = set()
    all_ids: set[str] = set()

    def declare(identity: Any, scope: set[str]) -> str:
        selected = _identity(identity)
        if selected in all_ids:
            _error(f"duplicate identity {selected}")
        all_ids.add(selected)
        scope.add(selected)
        return selected

    for input_row in inputs_list:
        _fields(input_row, ("id", "type"), "input")
        declare(input_row["id"], inputs)
        if input_row["type"] not in INPUT_TYPES:
            _error("input type")

    def check_term(node: Any) -> None:
        _term(node, inputs=inputs, values=values)

    terminated = False
    for step in steps:
        if terminated:
            _error("work after completion")
        if not _plain_dict(step):
            _error("step")
        kind = step.get("kind")
        if kind == "compute":
            _fields(step, ("kind", "id", "term"), "compute")
            check_term(step["term"])
            declare(step["id"], values)
        elif kind == "request":
            _fields(step, ("kind", "id", "boundary", "operation", "arguments"), "request")
            _identity(step["boundary"], "effect boundary")
            if (
                type(step["operation"]) is not str
                or not step["operation"]
                or _js_string_length(step["operation"]) > 256
            ):
                _error("effect operation or arguments")
            args = _data_list(step["arguments"], "effect arguments")
            if len(args) > LIMITS["arguments"]:
                _error("effect operation or arguments")
            for node in args:
                check_term(node)
            declare(step["id"], requests)
        elif kind == "observe":
            _fields(step, ("kind", "id", "request"), "observation")
            if step["request"] not in requests or step["request"] in observed:
                _error("observation correlation")
            observed.add(step["request"])
            declare(step["id"], values)
        elif kind == "write":
            _fields(step, ("kind", "object", "key", "value", "present"), "visible write")
            if (
                type(step["key"]) is not str
                or step["key"] == "__proto__"
                or type(step["present"]) is not bool
            ):
                _error("visible property change")
            check_term(step["object"])
            check_term(step["value"])
        elif kind in ("return", "throw"):
            _fields(step, ("kind", "value"), "completion")
            check_term(step["value"])
            terminated = True
        elif kind == "suspend":
            _fields(step, ("kind", "requests", "live"), "suspension")
            pending = _data_list(step["requests"], "pending requests")
            live = _data_list(step["live"], "live values")
            if (
                not pending
                or len(pending) > LIMITS["arguments"]
                or len(live) > LIMITS["arguments"]
                or len(set(pending)) != len(pending)
                or len(set(live)) != len(live)
                or any(req not in requests or req in observed for req in pending)
                or any(value not in values for value in live)
            ):
                _error("live suspension interface")
            for identity in pending + live:
                _identity(identity)
            terminated = True
        else:
            _error(f"unknown step {kind!s}")
    if not terminated:
        _error("missing completion or suspension")
    return {
        "inputs": len(inputs),
        "values": len(values),
        "requests": len(requests),
        "steps": len(steps),
    }


def _encode_value(value: Any) -> str:
    _validate_value(value)
    tag = value[0]
    if tag in ("undefined", "null"):
        return tag
    payload = value[1]
    if tag in ("bool", "str"):
        return ("true" if payload else "false") if tag == "bool" else _quote_string(payload)
    if tag == "num":
        if type(payload) is str:
            return "Infinity" if payload == "+Infinity" else payload
        # The record language uses JSON.stringify for finite JS Number values;
        # unlike stable JSON, the tagged scalar spelling accepts -0 specially.
        if _as_js_number(payload) == 0 and math.copysign(1.0, _as_js_number(payload)) < 0:
            return "-0"
        return _number_text(payload)
    if tag == "bigint":
        return payload + "n"
    if tag == "ref":
        return "@" + _encode_name(payload)
    _error("unknown tagged value")
    raise AssertionError("unreachable")


def _encode_name(value: str) -> str:
    return (
        value
        if _PLAIN_NAME.fullmatch(value) and value not in _RESERVED_NAMES
        else _quote_string(value)
    )


def _encode_symbol(value: str) -> str:
    return value if _PLAIN_SYMBOL.fullmatch(value) else _quote_string(value)


def _term_text(term: dict) -> str:
    if term["op"] == "literal":
        return _encode_value(term["value"])
    if term["op"] == "input":
        return "$" + _encode_name(term["id"])
    if term["op"] == "ref":
        return "%" + _encode_name(term["id"])
    if term["op"] == "project":
        return "project(" + _term_text(term["object"]) + ", " + _quote_string(term["key"]) + ")"
    if term["op"] == "fold_left":
        return (
            "fold_left(" + _term_text(term["input"]) + ", " + _term_text(term["initial"]) + ", add)"
        )
    return term["op"] + "(" + ", ".join(_term_text(arg) for arg in term["args"]) + ")"


def encode_algorithm_term(term: Any) -> str:
    validate_term(term)
    return _term_text(term)


def validate_algorithm_record(record: Any, role: str = "any") -> dict:
    if role not in ("any", "generated", "observed"):
        _error("record role")
    if not _plain_dict(record) or record.get("kind") not in {
        "compute",
        "request",
        "observe",
        "write",
        "return",
        "throw",
        "suspend",
    }:
        _error("record kind")
    kind = record["kind"]
    if role == "generated" and kind == "observe":
        _error("an external observation cannot be generated")
    if role == "observed" and kind != "observe":
        _error("observed input requires an observation record")
    if kind == "compute":
        _fields(record, ("kind", "id", "term", "result"), "compute claim")
        _identity(record["id"])
        validate_term(record["term"])
        _validate_value(record["result"])
    elif kind == "request":
        _fields(record, ("kind", "id", "boundary", "operation", "arguments"), "request claim")
        _identity(record["id"])
        _identity(record["boundary"])
        if (
            type(record["operation"]) is not str
            or not record["operation"]
            or _js_string_length(record["operation"]) > 256
        ):
            _error("operation or argument budget")
        args = _data_list(record["arguments"], "request arguments")
        if len(args) > LIMITS["arguments"]:
            _error("operation or argument budget")
        for value in args:
            _validate_value(value)
    elif kind == "observe":
        status = record.get("status")
        if status not in ("fulfilled", "rejected", "completed"):
            _error("observation status")
        expected = (
            ("kind", "request", "id", "status")
            if status == "completed"
            else ("kind", "request", "id", "status", "value")
        )
        _fields(record, expected, "observation")
        _identity(record["request"])
        _identity(record["id"])
        if status != "completed":
            _validate_value(record["value"])
    elif kind == "write":
        _fields(record, ("kind", "object", "key", "value", "present"), "visible write claim")
        _validate_value(record["object"])
        _validate_value(record["value"])
        if (
            record["object"][0] != "ref"
            or type(record["key"]) is not str
            or record["key"] == "__proto__"
            or type(record["present"]) is not bool
        ):
            _error("visible write fields")
    elif kind in ("return", "throw"):
        _fields(record, ("kind", "value"), "completion claim")
        _validate_value(record["value"])
    elif kind == "suspend":
        _fields(record, ("kind", "requests", "live"), "suspension claim")
        for key in ("requests", "live"):
            values = _data_list(record[key], "suspension " + key)
            if len(values) > LIMITS["arguments"] or len(set(values)) != len(values):
                _error("suspension " + key)
            for value in values:
                _identity(value)
        if not record["requests"]:
            _error("suspension requires a pending request")
    return record


def encode_algorithm_record(record: Any, role: str = "any") -> str:
    validate_algorithm_record(record, role)
    kind = record["kind"]
    if kind == "compute":
        line = (
            _encode_name(record["id"])
            + " = "
            + _term_text(record["term"])
            + " => "
            + _encode_value(record["result"])
        )
    elif kind == "request":
        line = (
            "request "
            + _encode_name(record["id"])
            + " "
            + _encode_name(record["boundary"])
            + ": "
            + _encode_symbol(record["operation"])
            + "("
            + ", ".join(_encode_value(value) for value in record["arguments"])
            + ")"
        )
    elif kind == "observe":
        line = (
            "observe "
            + _encode_name(record["request"])
            + " as "
            + _encode_name(record["id"])
            + ": "
            + record["status"]
            + ("" if record["status"] == "completed" else " " + _encode_value(record["value"]))
        )
    elif kind == "write":
        line = (
            "write "
            + _encode_value(record["object"])
            + "["
            + _quote_string(record["key"])
            + "] = "
            + _encode_value(record["value"])
            + ("" if record["present"] else " absent")
        )
    elif kind in ("return", "throw"):
        line = kind + " " + _encode_value(record["value"])
    else:
        line = (
            "suspend ["
            + ", ".join(_encode_name(value) for value in record["requests"])
            + "] live ["
            + ", ".join(_encode_name(value) for value in record["live"])
            + "]"
        )
    if len(_js_utf8(line)) > LIMITS["record_bytes"]:
        _error("record byte budget")
    return line


class _Reader:
    def __init__(self, text: str):
        self.text = text
        self.offset = 0
        self.nodes = 0

    def fail(self, message: str) -> None:
        _error(f"at {self.offset}: {message}")

    def peek(self, token: str) -> bool:
        return self.text.startswith(token, self.offset)

    def maybe(self, token: str) -> bool:
        if not self.peek(token):
            return False
        self.offset += len(token)
        return True

    def take(self, token: str) -> None:
        if not self.maybe(token):
            self.fail("expected " + repr(token))

    def match(self, pattern: re.Pattern, label: str) -> str:
        found = pattern.match(self.text[self.offset :])
        if not found:
            self.fail("expected " + label)
        self.offset += len(found.group(0))
        return found.group(0)

    def string(self) -> str:
        start = self.offset
        self.take('"')
        escaped = False
        while self.offset < len(self.text):
            char = self.text[self.offset]
            self.offset += 1
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                try:
                    value = json.loads(self.text[start : self.offset])
                except (json.JSONDecodeError, ValueError):
                    self.fail("JSON string")
                if type(value) is not str:
                    self.fail("JSON string")
                return value
        self.fail("unterminated string")
        raise AssertionError("unreachable")

    def name(self) -> str:
        return self.string() if self.peek('"') else self.match(_IDENTITY_TOKEN, "identity")

    def symbol(self) -> str:
        return self.string() if self.peek('"') else self.match(_SYMBOL_TOKEN, "operation")

    def value(self) -> list:
        if self.peek('"'):
            return ["str", self.string()]
        if self.maybe("@"):
            return ["ref", self.name()]
        for token, value in (
            ("undefined", ["undefined"]),
            ("null", ["null"]),
            ("true", ["bool", True]),
            ("false", ["bool", False]),
            ("-Infinity", ["num", "-Infinity"]),
            ("Infinity", ["num", "+Infinity"]),
            ("NaN", ["num", "NaN"]),
        ):
            if self.maybe(token):
                return value
        found = _BIGINT_TOKEN.match(self.text[self.offset :])
        if found:
            self.offset += len(found.group(0))
            return ["bigint", found.group(0)[:-1]]
        raw = self.match(_NUMBER_TOKEN, "scalar value")
        try:
            number = float(raw)
        except ValueError:
            self.fail("scalar value")
        if number == 0 and math.copysign(1.0, number) < 0:
            return ["num", "-0"]
        return ["num", number]

    def list(self, read_item, end: str) -> list:
        values = []
        if self.maybe(end):
            return values
        while True:
            if len(values) >= LIMITS["arguments"]:
                self.fail("list budget")
            values.append(read_item())
            if self.maybe(", "):
                continue
            self.take(end)
            return values

    def term(self, depth: int = 0) -> dict:
        self.nodes += 1
        if self.nodes > LIMITS["term_nodes"] or depth > LIMITS["depth"]:
            self.fail("term budget")
        if self.maybe("$"):
            return {"op": "input", "id": self.name()}
        if self.maybe("%"):
            return {"op": "ref", "id": self.name()}
        found = re.match(r"^([a-z][a-z_]*)\(", self.text[self.offset :])
        if found:
            op = found.group(1)
            self.offset += len(found.group(0))
            if op == "project":
                obj = self.term(depth + 1)
                self.take(", ")
                key = self.string()
                self.take(")")
                return {"op": op, "object": obj, "key": key}
            if op == "fold_left":
                input_term = self.term(depth + 1)
                self.take(", ")
                initial = self.term(depth + 1)
                self.take(", add)")
                return {
                    "op": op,
                    "input": input_term,
                    "initial": initial,
                    "step": "add",
                    "order": "ascending",
                    "profile": "dense-number-array/1",
                }
            if op not in OPERATIONS:
                self.fail("unknown algorithm operation")
            args = self.list(lambda: self.term(depth + 1), ")")
            return {"op": op, "args": args}
        return {"op": "literal", "value": self.value()}

    def record(self) -> dict:
        if self.maybe("request "):
            identity = self.name()
            self.take(" ")
            boundary = self.name()
            self.take(": ")
            operation = self.symbol()
            self.take("(")
            args = self.list(self.value, ")")
            return {
                "kind": "request",
                "id": identity,
                "boundary": boundary,
                "operation": operation,
                "arguments": args,
            }
        if self.maybe("observe "):
            request = self.name()
            self.take(" as ")
            identity = self.name()
            self.take(": ")
            status = self.match(
                re.compile(r"^(?:fulfilled|rejected|completed)"), "observation status"
            )
            if status == "completed":
                return {"kind": "observe", "request": request, "id": identity, "status": status}
            self.take(" ")
            return {
                "kind": "observe",
                "request": request,
                "id": identity,
                "status": status,
                "value": self.value(),
            }
        if self.maybe("write "):
            obj = self.value()
            self.take("[")
            key = self.string()
            self.take("] = ")
            value = self.value()
            present = not self.maybe(" absent")
            return {"kind": "write", "object": obj, "key": key, "value": value, "present": present}
        for kind in ("return", "throw"):
            if self.maybe(kind + " "):
                return {"kind": kind, "value": self.value()}
        if self.maybe("suspend ["):
            requests = self.list(self.name, "]")
            self.take(" live [")
            return {"kind": "suspend", "requests": requests, "live": self.list(self.name, "]")}
        identity = self.name()
        self.take(" = ")
        term = self.term()
        self.take(" => ")
        return {"kind": "compute", "id": identity, "term": term, "result": self.value()}


def parse_algorithm_record(line: Any, role: str = "any") -> dict:
    if (
        type(line) is not str
        or not line
        or "\n" in line
        or "\r" in line
        or len(_js_utf8(line)) > LIMITS["record_bytes"]
    ):
        _error("single record or byte budget")
    reader = _Reader(line)
    record = reader.record()
    if reader.offset != len(line):
        reader.fail("trailing bytes")
    validate_algorithm_record(record, role)
    if encode_algorithm_record(record, role) != line:
        _error("noncanonical spelling")
    return record


# --------------------------- Primitive semantics ---------------------------


def validate_algorithm_inputs(core: Any, arguments: Any, *, heap: Any = None) -> dict[str, list]:
    validate_algorithm(core)
    args = _data_list(arguments, "input values")
    if len(args) != len(core["inputs"]):
        raise AlgorithmV7Error("Exact algorithm input arity required")
    actual_heap = _snapshot_heap(heap)
    inputs: dict[str, list] = {}
    for value, input_row in zip(args, core["inputs"], strict=True):
        _validate_value(value)
        kind = input_row["type"]
        if kind == "scalar" and value[0] == "ref":
            raise AlgorithmV7Error("Algorithm scalar input required")
        if kind == "number" and value[0] != "num":
            raise AlgorithmV7Error("Algorithm binary64 number input required")
        if kind == "dense-number-array":
            _dense_items(value, actual_heap)
        if kind == "data" and value[0] == "ref":
            _heap_node(actual_heap, value[1])
        inputs[input_row["id"]] = value.copy()
    return inputs


def evaluate_term(term: Any, environment: dict | None = None) -> list:
    validate_term(term)
    if environment is None:
        environment = {}
    if not _plain_dict(environment):
        raise AlgorithmV7Error("Algorithm environment must be an ordinary data record")
    inputs = environment.get("inputs", {})
    values = environment.get("values", {})
    heap = _snapshot_heap(environment.get("heap"))

    def committed(value: list) -> list:
        _validate_value(value)
        if value[0] == "ref":
            _heap_node(heap, value[1])
        return value.copy()

    def visit(node: dict) -> list:
        op = node["op"]
        if op == "literal":
            return committed(node["value"])
        if op in ("input", "ref"):
            mapping = inputs if op == "input" else values
            value = _lookup(mapping, node["id"])
            if value is _MISSING:
                raise AlgorithmV7Error("Uncommitted algorithm dependency " + node["id"])
            return committed(value)
        if op == "project":
            obj = visit(node["object"])
            if obj[0] != "ref":
                raise AlgorithmV7Error("Data projection requires an object identity")
            _node, properties = _heap_node(heap, obj[1])
            return committed(properties.get(node["key"], ["undefined"]))
        if op == "fold_left":
            collection = visit(node["input"])
            initial = visit(node["initial"])
            items = _dense_items(collection, heap)
            if initial[0] != "num":
                raise AlgorithmV7Error("Ordered fold initial value must be binary64")
            total = _decode_scalar(initial)
            for item in items:
                # Python float addition is IEEE-754 binary64, with the same
                # left-to-right rounding and signed-zero behavior as JS Number.
                total = total + _decode_scalar(item)
            return _encode_scalar(total)
        if op == "select":
            condition = visit(node["args"][0])
            return visit(node["args"][1 if _truthy(condition) else 2])
        if op in ("and", "or", "coalesce"):
            left = visit(node["args"][0])
            use_right = (
                _truthy(left)
                if op == "and"
                else not _truthy(left)
                if op == "or"
                else left[0] in ("null", "undefined")
            )
            return visit(node["args"][1]) if use_right else left
        return _operation(op, [visit(argument) for argument in node["args"]])

    return visit(term)


def evaluate_algorithm(
    core: Any, arguments: Any, *, heap: Any = None, observations: Any = None
) -> dict:
    validate_algorithm(core)
    observations = [] if observations is None else _data_list(observations, "observations")
    public_heap = _snapshot_heap(heap)
    inputs = validate_algorithm_inputs(core, arguments, heap=public_heap)
    values: dict[str, list] = {}
    requests: list[dict] = []
    delivered: list[dict] = []
    claims: list[dict] = []
    environment = {"inputs": inputs, "values": values, "heap": public_heap}
    completion = None
    cursor = 0
    for index, step in enumerate(core["steps"]):
        kind = step["kind"]
        if kind == "compute":
            result = evaluate_term(step["term"], environment)
            values[step["id"]] = result.copy()
            claims.append(
                {"step": index, "kind": "compute", "id": step["id"], "result": result.copy()}
            )
        elif kind == "request":
            request = {
                "id": step["id"],
                "boundary": step["boundary"],
                "operation": step["operation"],
                "arguments": [evaluate_term(term, environment) for term in step["arguments"]],
            }
            requests.append(request)
            claims.append({"step": index, "kind": "request", **request.copy()})
        elif kind == "observe":
            if cursor >= len(observations):
                completion = {
                    "kind": "suspend",
                    "requests": [step["request"]],
                    "live": list(values),
                    "step": index,
                }
                break
            observation = observations[cursor]
            if not _plain_dict(observation) or any(
                key not in ("request", "status", "result", "mode") for key in observation
            ):
                raise AlgorithmV7Error("Algorithm observation authority or delivery order")
            if (
                observation.get("request") != step["request"]
                or observation.get("status") not in ("fulfilled", "rejected", "completed")
                or observation.get("mode") not in (None, "observed")
                or ((observation["status"] == "completed") == ("result" in observation))
            ):
                raise AlgorithmV7Error("Algorithm observation authority or delivery order")
            value = ["undefined"] if observation["status"] == "completed" else observation["result"]
            _validate_value(value)
            cursor += 1
            delivered.append(
                {
                    key: (row.copy() if type(row) is list else row)
                    for key, row in observation.items()
                }
            )
            if observation["status"] == "rejected":
                completion = {"kind": "throw", "value": value.copy()}
                break
            values[step["id"]] = value.copy()
        elif kind == "write":
            obj = evaluate_term(step["object"], environment)
            value = evaluate_term(step["value"], environment)
            if obj[0] != "ref":
                raise AlgorithmV7Error("Visible write requires an identity")
            node, properties = _heap_node(public_heap, obj[1])
            if node["kind"] == "array":
                raise AlgorithmV7Error("Array writes require a separate shape profile")
            if step["present"]:
                properties[step["key"]] = value.copy()
            else:
                properties.pop(step["key"], None)
            node["properties"] = properties
            claims.append(
                {
                    "step": index,
                    "kind": "write",
                    "object": obj.copy(),
                    "key": step["key"],
                    "value": value.copy(),
                    "present": step["present"],
                }
            )
        elif kind in ("return", "throw"):
            value = evaluate_term(step["value"], environment)
            completion = {"kind": kind, "value": value.copy()}
            claims.append({"step": index, "kind": kind, "value": value.copy()})
        elif kind == "suspend":
            completion = {
                "kind": "suspend",
                "requests": step["requests"].copy(),
                "live": step["live"].copy(),
                "step": index,
            }
            claims.append(
                {
                    "step": index,
                    "kind": "suspend",
                    "requests": step["requests"].copy(),
                    "live": step["live"].copy(),
                }
            )
    if cursor != len(observations):
        raise AlgorithmV7Error("Unconsumed or future algorithm observation")
    return {
        "completion": completion,
        "claims": claims,
        "values": values,
        "requests": requests,
        "observations": delivered,
        "heap": public_heap,
    }


# Public spellings mirror the requested API and the JavaScript reference names.
stable_algorithm_json = stable_algorithm_json
stable_algorithm_hash = stable_algorithm_hash
hash_algorithm_data = stable_algorithm_hash
encode_algorithm_value = _encode_value
