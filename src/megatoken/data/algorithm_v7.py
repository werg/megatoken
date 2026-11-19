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
OPERATIONS = {
    "add": (2, "+"),
    "subtract": (2, "-"),
    "multiply": (2, "*"),
    "divide": (2, "/"),
    "remainder": (2, "%"),
    "power": (2, "**"),
    "less": (2, "<"),
    "less_equal": (2, "<="),
    "greater": (2, ">"),
    "greater_equal": (2, ">="),
    "equal": (2, "==="),
    "not_equal": (2, "!=="),
    "loose_equal": (2, "=="),
    "loose_not_equal": (2, "!="),
    "bit_and": (2, "&"),
    "bit_or": (2, "|"),
    "bit_xor": (2, "^"),
    "shift_left": (2, "<<"),
    "shift_right": (2, ">>"),
    "unsigned_shift_right": (2, ">>>"),
    "not": (1, "!"),
    "positive": (1, "+"),
    "negate": (1, "-"),
    "bit_not": (1, "~"),
    "type_of": (1, "typeof"),
    "void": (1, "void"),
    "select": (3, "?:"),
    "and": (2, "&&"),
    "or": (2, "||"),
    "coalesce": (2, "??"),
}
_ID = re.compile(r"^[A-Za-z_$][A-Za-z0-9_$:-]*$")
_PLAIN_NAME = re.compile(r"^[A-Za-z_$][A-Za-z0-9_$]*$")
_PLAIN_SYMBOL = re.compile(r"^[A-Za-z_$][A-Za-z0-9_$]*(?:\.[A-Za-z_$][A-Za-z0-9_$]*)*$")
_RESERVED_NAMES = {"request", "observe", "write", "return", "throw", "suspend"}
_SPECIAL_NUMBERS = {"-0", "NaN", "+Infinity", "-Infinity"}
_NUMBER_TOKEN = re.compile(r"^-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?")
_BIGINT_TOKEN = re.compile(r"^-?(?:0|[1-9][0-9]*)n")
_IDENTITY_TOKEN = re.compile(r"^[A-Za-z_$][A-Za-z0-9_$]*")
_SYMBOL_TOKEN = re.compile(r"^[A-Za-z_$][A-Za-z0-9_$]*(?:\.[A-Za-z_$][A-Za-z0-9_$]*)*")


class AlgorithmV7Error(ValueError):
    """Invalid v7 data, syntax, bindings, or profile shape."""


class AlgorithmV7RangeError(ArithmeticError):
    """Guest ECMAScript RangeError raised by a valid v7 operation."""

    name = "RangeError"


def _error(message: str) -> None:
    raise AlgorithmV7Error(f"Invalid {ALGORITHM_VERSION}: {message}")


def _plain_dict(value: Any) -> bool:
    return type(value) is dict


def _data_list(value: Any, label: str = "array") -> list:
    if type(value) is not list:
        _error(f"{label} must be an ordinary data array")
    return value


def _fields(value: Any, fields: tuple[str, ...] | list[str], label: str) -> None:
    if (
        not _plain_dict(value)
        or len(value) != len(fields)
        or any(key not in value for key in fields)
    ):
        _error(f"{label} fields")


def _identity(value: Any, label: str = "identity") -> str:
    if type(value) is not str or not _ID.fullmatch(value):
        _error(label)
    return value


def _js_string_length(value: str) -> int:
    return len(value.encode("utf-16-le", "surrogatepass")) // 2


def _js_utf8(value: str) -> bytes:
    """Encode a JS string as UTF-8 (lone surrogate code units become U+FFFD)."""
    normalized: list[str] = []
    index = 0
    while index < len(value):
        code = ord(value[index])
        if 0xD800 <= code <= 0xDBFF and index + 1 < len(value):
            low = ord(value[index + 1])
            if 0xDC00 <= low <= 0xDFFF:
                normalized.append(chr(0x10000 + ((code - 0xD800) << 10) + low - 0xDC00))
                index += 2
                continue
        if 0xD800 <= code <= 0xDFFF:
            normalized.append("\ufffd")
        else:
            normalized.append(value[index])
        index += 1
    return "".join(normalized).encode("utf-8")


def _js_string_units(value: str) -> tuple[int, ...]:
    raw = value.encode("utf-16-be", "surrogatepass")
    return tuple((raw[i] << 8) | raw[i + 1] for i in range(0, len(raw), 2))


def _array_index(key: str) -> int | None:
    """Return the ECMAScript array-index value for an enumerable string key."""
    if not key or not key.isascii() or not key.isdecimal():
        return None
    if key != "0" and key.startswith("0"):
        return None
    number = int(key)
    return number if number < 0xFFFFFFFF else None


def _quote_string(value: str) -> str:
    if type(value) is not str:
        _error("JSON string required")
    # Normalize explicit surrogate pairs to the same Python character as a
    # direct astral code point; keep isolated code units escaped as JS does.
    pieces = ['"']
    index = 0
    short = {8: "\\b", 9: "\\t", 10: "\\n", 12: "\\f", 13: "\\r", 34: '\\"', 92: "\\\\"}
    while index < len(value):
        code = ord(value[index])
        if code in short:
            pieces.append(short[code])
        elif code < 0x20:
            pieces.append(f"\\u{code:04x}")
        elif (
            0xD800 <= code <= 0xDBFF
            and index + 1 < len(value)
            and 0xDC00 <= ord(value[index + 1]) <= 0xDFFF
        ):
            low = ord(value[index + 1])
            pieces.append(chr(0x10000 + ((code - 0xD800) << 10) + low - 0xDC00))
            index += 1
        elif 0xD800 <= code <= 0xDFFF:
            pieces.append(f"\\u{code:04x}")
        else:
            pieces.append(value[index])
        index += 1
    pieces.append('"')
    return "".join(pieces)


def _as_js_number(value: Any) -> float:
    if type(value) not in (int, float):
        _error("binary64 number required")
    try:
        number = float(value)
    except OverflowError:
        number = -math.inf if value < 0 else math.inf
    return number


def _number_text(value: Any) -> str:
    number = _as_js_number(value)
    if not math.isfinite(number) or (number == 0 and math.copysign(1.0, number) < 0):
        _error("noncanonical JSON number")
    if number == 0:
        return "0"
    sign = "-" if number < 0 else ""
    raw = repr(abs(number)).lower()
    if "e" in raw:
        mantissa, exponent_text = raw.split("e", 1)
        exponent = int(exponent_text)
    else:
        mantissa, exponent = raw, 0
    if "." in mantissa:
        before, after = mantissa.split(".", 1)
    else:
        before, after = mantissa, ""
    digits = before + after
    decimal_position = len(before) + exponent
    leading = len(digits) - len(digits.lstrip("0"))
    digits = digits[leading:]
    decimal_position -= leading
    digits = digits.rstrip("0") or "0"
    # ECMAScript Number::toString uses decimal notation for 1e-6 through
    # values below 1e21, and exponent notation outside that interval.
    if decimal_position > 0 and decimal_position <= 21:
        if decimal_position >= len(digits):
            rendered = digits + "0" * (decimal_position - len(digits))
        else:
            rendered = digits[:decimal_position] + "." + digits[decimal_position:]
    elif decimal_position <= 0 and decimal_position > -6:
        rendered = "0." + "0" * (-decimal_position) + digits
    else:
        exponent = decimal_position - 1
        significand = digits[0] + ("." + digits[1:] if len(digits) > 1 else "")
        rendered = significand + "e" + ("+" if exponent >= 0 else "") + str(exponent)
    return sign + rendered


def _stable_json(value: Any, active: set[int]) -> str:
    if value is None:
        return "null"
    if type(value) is bool:
        return "true" if value else "false"
    if type(value) is str:
        return _quote_string(value)
    if type(value) in (int, float):
        return _number_text(value)
    if type(value) in (list, dict):
        identity = id(value)
        if identity in active:
            _error("cyclic JSON data")
        active.add(identity)
        try:
            if type(value) is list:
                return "[" + ",".join(_stable_json(item, active) for item in value) + "]"
            if any(type(key) is not str for key in value):
                _error("JSON object keys must be strings")
            # `stableAlgorithmJSON` sorts keys before constructing an ordinary
            # JS object, then JSON.stringify applies ECMAScript's own property
            # enumeration order. Canonical array-index keys therefore move to
            # the front in numeric order, even when lexical sorting put them
            # among ordinary keys (for example "1" sorts after "01" but is
            # serialized before it).
            indexed = []
            ordinary = []
            for key in value:
                index = _array_index(key)
                if index is None:
                    ordinary.append(key)
                else:
                    indexed.append((index, key))
            keys = [key for _, key in sorted(indexed)] + sorted(ordinary, key=_js_string_units)
            return (
                "{"
                + ",".join(
                    _quote_string(key) + ":" + _stable_json(value[key], active) for key in keys
                )
                + "}"
            )
        finally:
            active.remove(identity)
    _error("noncanonical JSON data")
    raise AssertionError("unreachable")


def stable_algorithm_json(value: Any) -> str:
    """Return byte-compatible stable JSON for ordinary JSON data."""
    return _stable_json(value, set())


def stable_algorithm_hash(value: Any) -> str:
    """Match JS ``hashAlgorithmData``: raw UTF-8 for strings, stable JSON otherwise."""
    text = value if type(value) is str else stable_algorithm_json(value)
    return hashlib.sha256(_js_utf8(text)).hexdigest()


def algorithm_digest(core: Any) -> str:
    validate_algorithm(core)
    return hashlib.sha256(stable_algorithm_json(core).encode("utf-8")).hexdigest()


def _validate_value(value: Any, *, allow_reference: bool = True) -> list:
    _data_list(value, "tagged value")
    if not value or type(value[0]) is not str:
        _error("value tag")
    tag = value[0]
    if tag in ("undefined", "null") and len(value) == 1:
        return value
    if len(value) != 2:
        _error("value arity")
    payload = value[1]
    if tag == "bool" and type(payload) is bool:
        return value
    if tag == "str" and type(payload) is str:
        return value
    if tag == "bigint" and type(payload) is str and re.fullmatch(r"(?:0|-?[1-9][0-9]*)", payload):
        return value
    if tag == "num":
        if type(payload) in (int, float):
            number = _as_js_number(payload)
            if math.isfinite(number) and not (number == 0 and math.copysign(1.0, number) < 0):
                return value
        elif type(payload) is str and payload in _SPECIAL_NUMBERS:
            return value
    if tag == "ref" and allow_reference:
        _identity(payload, "value reference")
        return value
    _error("noncanonical scalar or reference")
    raise AssertionError("unreachable")


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


class _Undefined:
    __slots__ = ()

    def __repr__(self) -> str:
        return "undefined"


UNDEFINED = _Undefined()


def _decode_scalar(value: list) -> Any:
    _validate_value(value)
    tag = value[0]
    if tag == "undefined":
        return UNDEFINED
    if tag == "null":
        return None
    if tag == "bool" or tag == "str":
        return value[1]
    if tag == "bigint":
        return int(value[1])
    if tag == "num":
        if type(value[1]) is str:
            return {"-0": -0.0, "NaN": math.nan, "+Infinity": math.inf, "-Infinity": -math.inf}[
                value[1]
            ]
        return _as_js_number(value[1])
    if tag == "ref":
        return ("__algorithm_ref__", value[1])
    raise AssertionError("validated value tag")


def _encode_scalar(value: Any) -> list:
    if value is UNDEFINED:
        return ["undefined"]
    if value is None:
        return ["null"]
    if type(value) is bool:
        return ["bool", value]
    if type(value) is str:
        return ["str", value]
    if type(value) is int:
        return ["bigint", str(value)]
    if type(value) is float:
        if math.isnan(value):
            return ["num", "NaN"]
        if math.isinf(value):
            return ["num", "+Infinity" if value > 0 else "-Infinity"]
        if value == 0 and math.copysign(1.0, value) < 0:
            return ["num", "-0"]
        return ["num", value]
    raise TypeError("Algorithm values require an explicit data identity")


def _tag(value: list) -> str:
    return value[0]


def _truthy(value: list) -> bool:
    tag = _tag(value)
    if tag == "ref":
        return True
    if tag in ("undefined", "null"):
        return False
    payload = _decode_scalar(value)
    if tag == "num":
        return payload != 0 and not math.isnan(payload)
    if tag == "bigint":
        return payload != 0
    if tag == "str":
        return len(payload) > 0
    return bool(payload)


def _to_string(value: list) -> str:
    tag = _tag(value)
    if tag == "undefined":
        return "undefined"
    if tag == "null":
        return "null"
    if tag == "bool":
        return "true" if value[1] else "false"
    if tag == "str":
        return value[1]
    if tag == "bigint":
        return value[1]
    if tag == "ref":
        raise TypeError("Opaque reference coercion is outside the algorithm profile")
    if type(value[1]) is str:
        return {"NaN": "NaN", "+Infinity": "Infinity", "-Infinity": "-Infinity", "-0": "0"}[
            value[1]
        ]
    number = _as_js_number(value[1])
    if number == 0:
        return "0"
    if math.isnan(number):
        return "NaN"
    if math.isinf(number):
        return "Infinity" if number > 0 else "-Infinity"
    return _number_text(number)


def _number_from_string(text: str) -> float:
    # StringNumericLiteral uses ECMAScript WhiteSpace and LineTerminator, not
    # Python's wider Unicode whitespace classification (for example U+0085).
    stripped = text.strip(
        "\u0009\u000b\u000c\u0020\u00a0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u202f\u205f\u3000\ufeff\u000a\u000d\u2028\u2029"
    )
    if not stripped:
        return 0.0
    if stripped in ("Infinity", "+Infinity"):
        return math.inf
    if stripped == "-Infinity":
        return -math.inf
    try:
        if re.fullmatch(r"0[xX][0-9a-fA-F]+", stripped):
            return float(int(stripped[2:], 16))
        if re.fullmatch(r"0[bB][01]+", stripped):
            return float(int(stripped[2:], 2))
        if re.fullmatch(r"0[oO][0-7]+", stripped):
            return float(int(stripped[2:], 8))
        if not re.fullmatch(
            r"[+-]?(?:(?:[0-9]+(?:\.[0-9]*)?)|(?:\.[0-9]+))(?:[eE][+-]?[0-9]+)?", stripped
        ):
            return math.nan
        return float(stripped)
    except (ValueError, OverflowError):
        return math.inf if not stripped.startswith("-") else -math.inf


def _to_number(value: list) -> float:
    tag = _tag(value)
    if tag == "undefined":
        return math.nan
    if tag == "null":
        return 0.0
    if tag == "bool":
        return 1.0 if value[1] else 0.0
    if tag == "str":
        return _number_from_string(value[1])
    if tag == "num":
        return _decode_scalar(value)
    if tag == "bigint":
        raise TypeError("Cannot convert a BigInt value to a number")
    raise TypeError("Opaque reference coercion is outside the algorithm profile")


def _to_numeric(value: list) -> tuple[str, Any]:
    if _tag(value) == "bigint":
        return "bigint", int(value[1])
    return "number", _to_number(value)


def _primitive_pair(values: list[list]) -> tuple[tuple[str, Any], tuple[str, Any]]:
    left, right = values
    left_kind, left_value = _to_numeric(left)
    right_kind, right_value = _to_numeric(right)
    if left_kind != right_kind:
        raise TypeError("Cannot mix BigInt and other types")
    return (left_kind, left_value), (right_kind, right_value)


def _int32(number: float) -> int:
    if not math.isfinite(number) or number == 0:
        return 0
    truncated = math.trunc(number)
    unsigned = truncated % (1 << 32)
    return unsigned - (1 << 32) if unsigned >= (1 << 31) else unsigned


def _uint32(number: float) -> int:
    if not math.isfinite(number) or number == 0:
        return 0
    return math.trunc(number) % (1 << 32)


def _js_divide(left: float, right: float) -> float:
    if math.isnan(left) or math.isnan(right):
        return math.nan
    if math.isinf(left) and math.isinf(right):
        return math.nan
    if right == 0:
        if left == 0:
            return math.nan
        sign = math.copysign(1.0, left) * math.copysign(1.0, right)
        return math.copysign(math.inf, sign)
    if math.isinf(left):
        return math.copysign(math.inf, left * right)
    if math.isinf(right):
        sign = math.copysign(1.0, left) * math.copysign(1.0, right)
        return math.copysign(0.0, sign)
    return left / right


def _js_remainder(left: float, right: float) -> float:
    if math.isnan(left) or math.isnan(right) or math.isinf(left) or right == 0:
        return math.nan
    if math.isinf(right):
        return left
    try:
        return math.fmod(left, right)
    except ValueError:
        return math.nan


def _js_power(left: float, right: float) -> float:
    # ECMAScript special cases differ from math.pow for zero, infinities and
    # negative nonintegral powers; handle them before delegating ordinary cases.
    if right == 0:
        return 1.0
    if math.isnan(right) or math.isnan(left):
        return math.nan
    abs_left = abs(left)
    if math.isinf(right):
        if abs_left == 1:
            return math.nan
        if (abs_left > 1) == (right > 0):
            return math.inf
        return 0.0
    if math.isinf(left):
        negative = left < 0 and math.isfinite(right) and right.is_integer() and int(right) % 2 != 0
        if right > 0:
            return -math.inf if negative else math.inf
        return -0.0 if negative else 0.0
    if left == 0:
        negative = math.copysign(1.0, left) < 0 and right.is_integer() and int(right) % 2 != 0
        if right > 0:
            return -0.0 if negative else 0.0
        return -math.inf if negative else math.inf
    if left < 0 and not right.is_integer():
        return math.nan
    try:
        return math.pow(left, right)
    except OverflowError:
        if left < 0 and right.is_integer() and int(right) % 2:
            return -math.inf
        return math.inf
    except (ValueError, ZeroDivisionError):
        return math.nan


def _bigint_string(text: str) -> int | None:
    stripped = text.strip()
    if not stripped:
        return 0
    try:
        if re.fullmatch(r"[+-]?[0-9]+", stripped):
            return int(stripped, 10)
        if re.fullmatch(r"0[xX][0-9a-fA-F]+", stripped):
            return int(stripped, 16)
        if re.fullmatch(r"0[bB][01]+", stripped):
            return int(stripped, 2)
        if re.fullmatch(r"0[oO][0-7]+", stripped):
            return int(stripped, 8)
    except ValueError:
        pass
    return None


def _strict_equal(left: list, right: list) -> bool:
    if left[0] != right[0]:
        return False
    tag = left[0]
    if tag in ("undefined", "null"):
        return True
    if tag == "ref":
        return left[1] == right[1]
    if tag == "num":
        a, b = _decode_scalar(left), _decode_scalar(right)
        return not (math.isnan(a) or math.isnan(b)) and a == b
    return left[1] == right[1]


def _loose_equal(left: list, right: list) -> bool:
    a, b = left[0], right[0]
    if a == b:
        return _strict_equal(left, right)
    if a in ("null", "undefined") and b in ("null", "undefined"):
        return True
    if "ref" in (a, b):
        return False
    if a == "bool":
        return _loose_equal(["num", 1 if left[1] else 0], right)
    if b == "bool":
        return _loose_equal(left, ["num", 1 if right[1] else 0])
    if a == "str" and b == "num":
        return _to_number(left) == _to_number(right) and not math.isnan(_to_number(right))
    if a == "num" and b == "str":
        return _loose_equal(right, left)
    if a == "bigint" and b == "str":
        parsed = _bigint_string(right[1])
        return parsed is not None and int(left[1]) == parsed
    if a == "str" and b == "bigint":
        return _loose_equal(right, left)
    if a == "bigint" and b == "num":
        number = _decode_scalar(right)
        return math.isfinite(number) and number.is_integer() and int(number) == int(left[1])
    if a == "num" and b == "bigint":
        return _loose_equal(right, left)
    return False


def _relational(left: list, right: list, op: str) -> bool:
    a, b = left[0], right[0]
    if a == "str" and b == "str":
        av, bv = _js_string_units(left[1]), _js_string_units(right[1])
        return {
            "less": av < bv,
            "less_equal": av <= bv,
            "greater": av > bv,
            "greater_equal": av >= bv,
        }[op]
    if a == "bigint" and b == "str":
        parsed = _bigint_string(right[1])
        # Abstract relational comparison uses StringToBigInt here. A decimal
        # fraction such as "1.5" is invalid for that conversion, even though
        # Number("1.5") would succeed.
        if parsed is None:
            return False
        bv = parsed
        av = int(left[1])
    elif a == "str" and b == "bigint":
        return _relational(
            right,
            left,
            {
                "less": "greater",
                "less_equal": "greater_equal",
                "greater": "less",
                "greater_equal": "less_equal",
            }[op],
        )
    else:
        ka, av = _to_numeric(left)
        kb, bv = _to_numeric(right)
        if ka == "bigint" and kb == "bigint":
            pass
        elif ka == "bigint":
            bv = _to_number(right)
        elif kb == "bigint":
            av = _to_number(left)
        if (type(av) is float and math.isnan(av)) or (type(bv) is float and math.isnan(bv)):
            return False
    return {"less": av < bv, "less_equal": av <= bv, "greater": av > bv, "greater_equal": av >= bv}[
        op
    ]


def _operation(name: str, values: list[list]) -> list:
    if name == "void":
        return ["undefined"]
    if name == "not":
        return ["bool", not _truthy(values[0])]
    if any(value[0] == "ref" for value in values):
        if name == "type_of":
            return ["str", "object"]
        left = values[0]
        right = values[1] if len(values) > 1 else None
        both_refs = left[0] == "ref" and right is not None and right[0] == "ref"
        allowed = {"equal", "not_equal", "loose_equal", "loose_not_equal"}
        null_loose = name.startswith("loose_") and (
            left[0] in ("null", "undefined")
            or (right is not None and right[0] in ("null", "undefined"))
        )
        if name not in allowed or (not both_refs and not null_loose):
            raise TypeError("Opaque reference coercion is outside the algorithm profile")
        equal = both_refs and left[1] == right[1]
        return ["bool", equal if name in ("equal", "loose_equal") else not equal]
    if name == "add":
        left, right = values
        if left[0] == "str" or right[0] == "str":
            return ["str", _to_string(left) + _to_string(right)]
        (kind_a, a), (kind_b, b) = _primitive_pair(values)
        return _encode_scalar(a + b)
    if name in ("less", "less_equal", "greater", "greater_equal"):
        return ["bool", _relational(values[0], values[1], name)]
    if name in ("equal", "not_equal"):
        equal = _strict_equal(values[0], values[1])
        return ["bool", equal if name == "equal" else not equal]
    if name in ("loose_equal", "loose_not_equal"):
        equal = _loose_equal(values[0], values[1])
        return ["bool", equal if name == "loose_equal" else not equal]
    if name in ("positive", "negate", "bit_not"):
        kind, value = _to_numeric(values[0])
        if name == "positive":
            if kind == "bigint":
                raise TypeError("Cannot convert a BigInt value to a number")
            return _encode_scalar(value)
        if name == "negate":
            return _encode_scalar(-value)
        if kind == "bigint":
            return _encode_scalar(~value)
        return _encode_scalar(~_int32(value))
    if name in (
        "bit_and",
        "bit_or",
        "bit_xor",
        "shift_left",
        "shift_right",
        "unsigned_shift_right",
    ):
        (kind_a, a), (kind_b, b) = _primitive_pair(values)
        if name == "unsigned_shift_right" and kind_a == "bigint":
            raise TypeError("BigInts have no unsigned right shift")
        if kind_a == "bigint":
            if name in ("shift_left", "shift_right"):
                if abs(b) > 1_000_000:
                    raise OverflowError("BigInt shift exceeds the bounded evaluator profile")
                if b < 0:
                    return _encode_scalar(a >> -b if name == "shift_left" else a << -b)
                return _encode_scalar(a << b if name == "shift_left" else a >> b)
            return _encode_scalar(
                {"bit_and": lambda: a & b, "bit_or": lambda: a | b, "bit_xor": lambda: a ^ b}[
                    name
                ]()
            )
        ia, ib = _int32(a), _int32(b)
        if name == "bit_and":
            result = ia & ib
        elif name == "bit_or":
            result = ia | ib
        elif name == "bit_xor":
            result = ia ^ ib
        elif name == "shift_left":
            result = _int32(ia << (_uint32(b) & 0x1F))
        elif name == "shift_right":
            result = ia >> (_uint32(b) & 0x1F)
        else:
            result = _uint32(a) >> (_uint32(b) & 0x1F)
        return ["num", result]
    if name == "type_of":
        tag = values[0][0]
        return [
            "str",
            {
                "undefined": "undefined",
                "null": "object",
                "bool": "boolean",
                "str": "string",
                "num": "number",
                "bigint": "bigint",
            }[tag],
        ]
    (kind_a, a), (kind_b, b) = _primitive_pair(values)
    if name == "subtract":
        result = a - b
    elif name == "multiply":
        result = a * b
    elif name == "divide":
        if kind_a == "bigint":
            if b == 0:
                raise AlgorithmV7RangeError("BigInt division by zero")
            result = abs(a) // abs(b) * (-1 if (a < 0) != (b < 0) else 1)
        else:
            result = _js_divide(a, b)
    elif name == "remainder":
        if kind_a == "bigint":
            if b == 0:
                raise AlgorithmV7RangeError("BigInt remainder by zero")
            quotient = abs(a) // abs(b) * (-1 if (a < 0) != (b < 0) else 1)
            result = a - quotient * b
        else:
            result = _js_remainder(a, b)
    elif name == "power":
        if kind_a == "bigint":
            if b < 0:
                raise AlgorithmV7RangeError("BigInt exponent must be nonnegative")
            if b > 1_000_000:
                raise OverflowError("BigInt exponent exceeds the bounded evaluator profile")
            result = pow(a, b)
        else:
            result = _js_power(a, b)
    else:
        raise TypeError("Unimplemented algorithm operation " + name)
    return _encode_scalar(result)


def _lookup(mapping: Any, identity: str) -> Any:
    if type(mapping) is dict:
        return mapping.get(identity, _MISSING)
    if type(mapping) is list:
        for row in mapping:
            if type(row) is list and len(row) == 2 and row[0] == identity:
                return row[1]
        return _MISSING
    raise AlgorithmV7Error("Algorithm state must be a data record or list of pairs")


_MISSING = object()


def _property_rows(properties: Any) -> list[tuple[str, Any]]:
    rows: list[tuple[str, Any]] = []
    if type(properties) is list:
        for prop in properties:
            if type(prop) is list:
                _data_list(prop, "data property")
                if len(prop) != 2:
                    raise AlgorithmV7Error("Algorithm data property pair")
                key, value = prop
            elif _plain_dict(prop) and "key" in prop and "value" in prop:
                key, value = prop["key"], prop["value"]
            else:
                raise AlgorithmV7Error("Algorithm data property descriptor")
            rows.append((key, value))
    elif type(properties) is dict:
        for key, raw in properties.items():
            if type(raw) is list:
                value = raw
            elif _plain_dict(raw) and "value" in raw:
                value = raw["value"]
            else:
                value = ["undefined"]
            rows.append((key, value))
    else:
        raise AlgorithmV7Error("Algorithm needs committed data properties")
    result: dict[str, Any] = {}
    for key, value in rows:
        if type(key) is not str or key == "__proto__":
            raise AlgorithmV7Error("Algorithm data property key")
        if key in result:
            raise AlgorithmV7Error("Duplicate algorithm data property")
        _validate_value(value)
        result[key] = value
    return list(result.items())


def _heap_map(heap: Any) -> dict[str, dict]:
    if heap is None:
        return {}
    if not _plain_dict(heap):
        raise AlgorithmV7Error("Algorithm heap must be an ordinary data record")
    if "objects" in heap and "roots" in heap:
        for root in _data_list(heap["roots"], "heap graph roots"):
            _validate_value(root)
        objects = _data_list(heap["objects"], "heap graph objects")
        result: dict[str, dict] = {}
        for source in objects:
            if (
                not _plain_dict(source)
                or not {"id", "kind", "properties"}.issubset(source)
                or set(source) - {"id", "kind", "prototype", "properties", "shapeVersion"}
            ):
                raise AlgorithmV7Error("Algorithm graph node schema")
            identity = _identity(source["id"], "heap identity")
            if identity in result:
                raise AlgorithmV7Error("Duplicate algorithm heap identity")
            result[identity] = {
                "kind": source["kind"],
                **({"prototype": source["prototype"]} if "prototype" in source else {}),
                "properties": source["properties"],
            }
        return result
    return heap


def _heap_node(heap: dict[str, dict], identity: str) -> tuple[dict, dict[str, list]]:
    node = heap.get(identity, _MISSING)
    if not _plain_dict(node) or node.get("kind") not in ("array", "object"):
        raise AlgorithmV7Error("Algorithm needs an ordinary data object")
    if "accessors" in node and (not _plain_dict(node["accessors"]) or len(node["accessors"]) != 0):
        raise AlgorithmV7Error("Algorithm needs an ordinary data object")
    if "proxy" in node or "iterator" in node:
        raise AlgorithmV7Error("Algorithm needs an ordinary data object")
    prototype = node.get("prototype", _MISSING)
    if (
        prototype is not _MISSING
        and prototype is not None
        and prototype != ("Array.prototype" if node["kind"] == "array" else "Object.prototype")
    ):
        raise AlgorithmV7Error("Algorithm needs an ordinary data object")
    if node["kind"] == "array" and prototype is None:
        raise AlgorithmV7Error("Array prototype is outside the dense profile")
    properties: dict[str, list] = {}
    for key, value in _property_rows(node.get("properties")):
        properties[key] = value
    return node, properties


def _snapshot_heap(heap: Any) -> dict[str, dict]:
    raw = _heap_map(heap)
    if len(raw) > 256:
        raise AlgorithmV7Error("Algorithm heap shape or node budget")
    result: dict[str, dict] = {}
    for identity in raw:
        _identity(identity, "heap identity")
        node, properties = _heap_node(raw, identity)
        result[identity] = {
            "kind": node["kind"],
            **({"prototype": None} if node.get("prototype", _MISSING) is None else {}),
            "properties": properties,
        }
    for node in result.values():
        for value in node["properties"].values():
            if value[0] == "ref" and value[1] not in result:
                raise AlgorithmV7Error("Dangling algorithm heap reference " + value[1])
    return result


def _dense_items(value: list, heap: dict[str, dict]) -> list[list]:
    _validate_value(value)
    if value[0] != "ref":
        raise AlgorithmV7Error("Ordered fold requires a dense numeric array identity")
    node, props = _heap_node(heap, value[1])
    length = props.get("length")
    if (
        node["kind"] != "array"
        or length is None
        or length[0] != "num"
        or type(length[1]) not in (int, float)
    ):
        raise AlgorithmV7Error("Ordered fold array bound")
    count_float = _as_js_number(length[1])
    if (
        not math.isfinite(count_float)
        or not count_float.is_integer()
        or count_float < 0
        or count_float > LIMITS["fold_items"]
    ):
        raise AlgorithmV7Error("Ordered fold array bound")
    count = int(count_float)
    if len(props) != count + 1:
        raise AlgorithmV7Error("Ordered fold refuses holes or extra properties")
    items = []
    for index in range(count):
        item = props.get(str(index))
        if (
            item is None
            or item[0] != "num"
            or type(item[1]) not in (int, float)
            or not math.isfinite(_as_js_number(item[1]))
        ):
            raise AlgorithmV7Error("Ordered fold requires dense finite binary64 elements")
        items.append(item)
    return items


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


def canonical_algorithm_graph(graph: Any, *, input_ids: Any = None) -> dict:
    """Project a rooted input/observable graph while preserving aliases and order.

    Unreachable private objects disappear.  Input identities are anchored as
    ``i0``, ``i1``, ...; newly reached objects receive ``g0``, ``g1``, ... .
    """
    if (
        not _plain_dict(graph)
        or set(graph) - {"roots", "objects"}
        or "roots" not in graph
        or "objects" not in graph
    ):
        raise AlgorithmV7Error("Observable graph schema")
    roots = _data_list(graph["roots"], "observable roots")
    source_nodes = _data_list(graph["objects"], "observable objects")
    source: dict[str, dict] = {}
    for node in source_nodes:
        if (
            not _plain_dict(node)
            or node.get("kind") not in ("object", "array")
            or not {"id", "kind", "properties"}.issubset(node)
            or set(node) - {"id", "kind", "prototype", "properties", "shapeVersion"}
        ):
            raise AlgorithmV7Error("Observable object schema")
        identity = _identity(node["id"])
        if identity in source:
            raise AlgorithmV7Error("Duplicate observable identity")
        prototype = node.get("prototype", _MISSING)
        if prototype is not _MISSING and prototype is not None:
            raise AlgorithmV7Error("Observable prototype profile")
        if node["kind"] == "array" and prototype is None:
            raise AlgorithmV7Error("Observable array prototype")
        source[identity] = node
        _data_list(node["properties"], "observable properties")
    input_ids = [] if input_ids is None else _data_list(input_ids, "input identity anchors")
    if len(set(input_ids)) != len(input_ids):
        raise AlgorithmV7Error("Observable input identity anchors")
    ids: dict[str, str] = {}
    for index, identity in enumerate(input_ids):
        _identity(identity)
        if identity not in source:
            raise AlgorithmV7Error("Missing input identity anchor")
        ids[identity] = "i" + str(index)
    objects: list[dict] = []
    active: set[str] = set()

    def rewrite(value: Any) -> list:
        _validate_value(value)
        if value[0] != "ref":
            return value.copy()
        identity = value[1]
        node = source.get(identity)
        if node is None:
            raise AlgorithmV7Error("Dangling observable identity " + identity)
        if identity not in ids:
            ids[identity] = "g" + str(sum(1 for item in ids.values() if item.startswith("g")))
        canonical_id = ids[identity]
        if identity not in active and not any(row["id"] == canonical_id for row in objects):
            active.add(identity)
            row = {
                "id": canonical_id,
                "kind": node["kind"],
                **({"prototype": None} if node.get("prototype", _MISSING) is None else {}),
                "properties": [],
            }
            objects.append(row)
            seen_keys: set[str] = set()
            for prop in node["properties"]:
                if type(prop) is list and len(prop) == 2:
                    _data_list(prop, "observable property")
                    key, child = prop
                elif (
                    _plain_dict(prop)
                    and {"key", "value"}.issubset(prop)
                    and not (set(prop) - {"key", "value", "version"})
                ):
                    key, child = prop["key"], prop["value"]
                else:
                    raise AlgorithmV7Error("Observable data property")
                if type(key) is not str or key == "__proto__" or key in seen_keys:
                    raise AlgorithmV7Error("Observable property identity")
                seen_keys.add(key)
                row["properties"].append([key, rewrite(child)])
            if node["kind"] == "array":
                length = next((v for k, v in row["properties"] if k == "length"), None)
                if (
                    length is None
                    or length[0] != "num"
                    or type(length[1]) not in (int, float)
                    or not math.isfinite(_as_js_number(length[1]))
                    or not _as_js_number(length[1]).is_integer()
                    or _as_js_number(length[1]) < 0
                    or _as_js_number(length[1]) > 9007199254740991
                ):
                    raise AlgorithmV7Error("Observable array length")
            active.remove(identity)
        return ["ref", canonical_id]

    # Input anchors are always retained, even if no root directly mentions them.
    for identity in input_ids:
        rewrite(["ref", identity])
    canonical_roots = [rewrite(value) for value in roots]
    return {"roots": canonical_roots, "objects": objects}


def algorithm_heap_from_graph(graph: Any) -> dict[str, dict]:
    """Convert the v7 input graph envelope into the heap accepted by the core."""
    if not _plain_dict(graph) or "objects" not in graph or "roots" not in graph:
        raise AlgorithmV7Error("Algorithm input graph schema")
    _data_list(graph["roots"], "input graph roots")
    result = {}
    for node in _data_list(graph["objects"], "input graph objects"):
        if not _plain_dict(node) or not {"id", "kind", "properties"}.issubset(node):
            raise AlgorithmV7Error("Algorithm graph node schema")
        identity = _identity(node["id"])
        if identity in result:
            raise AlgorithmV7Error("Duplicate algorithm heap identity")
        props = []
        for prop in _data_list(node["properties"], "input graph properties"):
            if _plain_dict(prop) and {"key", "value"}.issubset(prop):
                props.append([prop["key"], prop["value"]])
            elif type(prop) is list and len(prop) == 2:
                props.append(prop)
            else:
                raise AlgorithmV7Error("Algorithm graph property")
        result[identity] = {
            "kind": node["kind"],
            **({"prototype": node["prototype"]} if "prototype" in node else {}),
            "properties": props,
        }
    return result


# Public spellings mirror the requested API and the JavaScript reference names.
stable_algorithm_json = stable_algorithm_json
stable_algorithm_hash = stable_algorithm_hash
hash_algorithm_data = stable_algorithm_hash
encode_algorithm_value = _encode_value
