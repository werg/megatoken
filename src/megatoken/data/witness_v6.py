"""Strict, independent semantic-trace/6 record validation.

This module reads the concise production wire grammar and maps it to the same
typed internal claims consumed by the native harness. It imports no tracer JS;
source proof, journals, and certificates remain external to model-visible rows.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from typing import Any


class WitnessV6Error(ValueError):
    pass


FIELDS = {
    "code": {"entry", "source", "sites", "randomSeed"},
    "input": {"graph", "retained"},
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
    "request": {"effect", "operation", "arguments", "invocation", "site", "payload", "promise"},
    "promise": {
        "action",
        "promise",
        "value",
        "target",
        "job",
        "reaction",
        "closure",
        "rejectionClosure",
        "invocation",
    },
    "await": {"promise", "continuation", "invocation", "returnPromise"},
    "resume": {"continuation", "invocation", "result", "status"},
    "observe": {"effect", "result", "status", "graph"},
    "job": {"action", "job", "owner", "result", "closure"},
    "end": {"kind"},
    "summarize": {
        "entry",
        "arguments",
        "result",
        "profile",
        "decisions",
        "iterations",
        "accumulator",
        "invocation",
    },
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
    "request": {"effect", "operation", "arguments", "invocation", "site", "payload"},
    "promise": {"action", "promise"},
    "await": {"promise", "continuation", "invocation"},
    "resume": {"continuation", "invocation", "result", "status"},
    "observe": {"effect", "result", "status"},
    "job": {"action", "job", "owner"},
    "end": {"kind"},
    "summarize": {"entry", "arguments", "result", "profile"},
    "value-open": {"id", "type"},
    "value-part": {"id", "data"},
    "value-close": {"id"},
}
IDENTITY = re.compile(r"[a-z][a-z0-9]*\Z")
INVOCATION_IDENTITY = re.compile(r"[a-z][a-z0-9:_-]*\Z")
BIGINT = re.compile(r"(?:0|-?[1-9][0-9]*)\Z")
SPECIAL = {"-0", "NaN", "+Infinity", "-Infinity"}
OPERATORS = {
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
}
UNARY = {"!", "+", "-", "~", "typeof", "void"}


def _fail(reason: str) -> None:
    raise WitnessV6Error("Invalid semantic-trace/6: " + reason)


def _source_binding(name: Any) -> bool:
    if not isinstance(name, str):
        return False
    try:
        return _scan_identifier(name, 0)[1] == len(name)
    except WitnessV6Error:
        return False


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail("Duplicate JSON field")
        result[key] = value
    return result


def _trace_value(tag: Any) -> str:
    if not isinstance(tag, list) or not tag:
        _fail("Untagged public value")
    name = tag[0]
    if name in {"undefined", "null"} and len(tag) == 1:
        return name
    if len(tag) != 2:
        _fail("Incorrect scalar arity")
    item = tag[1]
    if name in {"str", "bool"}:
        return _json_text(item)
    if name == "num":
        if item == "+Infinity":
            return "Infinity"
        if isinstance(item, str):
            return item
        if type(item) is int:
            return _json_text(item)
        if type(item) is float:
            return _ecma_number(item)
    if name == "bigint" and isinstance(item, str):
        return item + "n"
    if name == "ref" and isinstance(item, str):
        return "@" + item
    _fail("Unsupported or noncanonical scalar")


def _trace_expression(expr: Any) -> str:
    if not isinstance(expr, dict):
        _fail("Malformed expression")
    operation = expr.get("op")
    if operation == "literal":
        return _trace_value(expr["value"])
    if operation == "local":
        name = expr["name"]
        return "local(" + _json_text(name) + ")" if name in _RESERVED else name
    if operation == "read":
        return "@" + expr["object"] + "[" + _json_text(expr["key"]) + "]"
    if operation == "random":
        return "random()"
    if operation == "unary":
        return "(" + expr["operator"] + " " + _trace_expression(expr["arg"]) + ")"
    if operation == "binary":
        return (
            "("
            + _trace_expression(expr["left"])
            + " "
            + expr["operator"]
            + " "
            + _trace_expression(expr["right"])
            + ")"
        )
    _fail("Unregistered primitive expression")


def _trace_symbol(name: str) -> str:
    return name if _is_symbol(name) else _json_text(name)


def encode_record(kind: str, fields: dict[str, Any]) -> str:
    """Encode a validated internal claim using the canonical v6 wire spelling."""
    text: str
    if kind == "code":
        text = (
            "source "
            + _json_text(fields["entry"])
            + " = "
            + _json_text(fields["source"])
            + " sites "
            + _json_text(fields["sites"])
            + (" seed " + str(fields["randomSeed"]) if "randomSeed" in fields else "")
        )
    elif kind == "input":
        text = "input " + _json_text(fields["graph"])
        if "retained" in fields:
            text += " retained " + _json_text(fields["retained"])
    elif kind == "compute":
        text = (
            "compute " + _trace_expression(fields["expr"]) + " => " + _trace_value(fields["result"])
        )
        if "bind" in fields:
            text += " as " + fields["bind"]
    elif kind == "choose":
        text = (
            "choose "
            + _trace_expression(fields["expr"])
            + " => "
            + _trace_value(fields["decision"])
            + " -> "
            + fields["path"]
        )
    elif kind == "bind":
        text = fields["name"] + " = " + _trace_value(fields["value"])
        if "from" in fields:
            text += " from " + fields["from"]
    elif kind == "allocate":
        properties = (
            "{"
            + ", ".join(
                _json_text(item["key"]) + ": " + _trace_value(item["value"])
                for item in fields["properties"]
            )
            + "}"
        )
        text = "new @" + fields["name"] + " " + fields["kind"] + " " + properties
        if "length" in fields:
            text += " length " + str(fields["length"])
        if "code" in fields:
            text += " code " + _json_text(fields["code"])
        if "captures" in fields:
            text += (
                " captures {"
                + ", ".join(item["binding"] + ": " + item["cell"] for item in fields["captures"])
                + "}"
            )
    elif kind in {"read", "write"}:
        text = kind + " @" + fields["object"] + "[" + _json_text(fields["key"]) + "]"
        text += " => " if kind == "read" else " = "
        text += _trace_value(fields["value"])
        if not fields["present"]:
            text += " absent"
    elif kind == "call":
        text = (
            "call "
            + _trace_symbol(fields["callee"])
            + "("
            + ", ".join(map(_trace_value, fields["arguments"]))
            + ") this "
            + _trace_value(fields["receiver"])
            + " -> @"
            + fields["child"]
        )
    elif kind == "transfer":
        text = "transfer " + fields["kind"] + " " + _trace_value(fields["value"])
        text += " via " + _json_text(fields["handler"])
    elif kind in {"return", "throw"}:
        text = kind + " " + _trace_value(fields["value"])
    elif kind == "request":
        text = (
            "request @"
            + fields["effect"]
            + " "
            + _trace_symbol(fields["operation"])
            + "("
            + ", ".join(map(_trace_value, fields["arguments"]))
            + ") payload "
            + fields["payload"]
        )
        if "promise" in fields:
            text += " promise @" + fields["promise"]
    elif kind == "observe":
        text = (
            "observe @"
            + fields["effect"]
            + " "
            + fields["status"]
            + " "
            + _trace_value(fields["result"])
        )
        if "graph" in fields:
            text += " graph " + _json_text(fields["graph"])
    elif kind == "job":
        text = "job " + fields["action"] + " @" + fields["job"] + " for @" + fields["owner"]
        if "result" in fields:
            text += " result " + _json_text(fields["result"])
        if "closure" in fields:
            text += " closure @" + fields["closure"]
    elif kind == "promise":
        text = "promise " + fields["action"] + " @" + fields["promise"]
        if fields["action"] in {"resolve", "reject"}:
            text += " = " + _trace_value(fields["value"])
        elif fields["action"] == "adopt":
            text += " -> @" + fields["target"]
        elif fields["action"] == "react":
            text += " " + fields["reaction"] + " as @" + fields["job"]
            if "closure" in fields:
                text += " closure @" + fields["closure"]
            if "rejectionClosure" in fields:
                text += " reject @" + fields["rejectionClosure"]
            if "target" in fields:
                text += " target @" + fields["target"]
    elif kind == "await":
        text = "await @" + fields["promise"] + " as @" + fields["continuation"]
        if "returnPromise" in fields:
            text += " returns @" + fields["returnPromise"]
    elif kind == "resume":
        text = (
            "resume @"
            + fields["continuation"]
            + " "
            + fields["status"]
            + " "
            + _trace_value(fields["result"])
        )
    elif kind == "end":
        text = "end " + fields["kind"]
    elif kind == "summarize":
        text = (
            "evaluate "
            + _trace_symbol(fields["entry"])
            + "("
            + ", ".join(map(_trace_value, fields["arguments"]))
            + ") => "
            + _trace_value(fields["result"])
        )
        if fields["profile"] == "pure-control-helper/1":
            text += " choosing " + ", ".join(
                f"{item['site']['start']}..{item['site']['end']}:{item['selected']}"
                for item in fields["decisions"]
            )
        elif fields["profile"] == "dense-array-sum/1":
            text += f" over {fields['iterations']} items into {fields['accumulator']}"
    elif kind == "value-open":
        text = "record @" + fields["id"] + " type " + _json_text(fields["type"])
    elif kind == "value-part":
        text = "part @" + fields["id"] + " " + _json_text(fields["data"])
    elif kind == "value-close":
        text = "close @" + fields["id"]
    else:
        _fail("Unknown trace text kind")
    if "invocation" in fields:
        text += " in @" + fields["invocation"]
    if "site" in fields:
        text += f" at {fields['site']['start']}..{fields['site']['end']}"
    return text


def _json_quote(value: str) -> str:
    output = ['"']
    escapes = {"\b": "\\b", "\t": "\\t", "\n": "\\n", "\f": "\\f", "\r": "\\r"}
    for char in value:
        codepoint = ord(char)
        if char == '"':
            output.append('\\"')
        elif char == "\\":
            output.append("\\\\")
        elif char in escapes:
            output.append(escapes[char])
        elif codepoint < 0x20 or 0xD800 <= codepoint <= 0xDFFF:
            output.append(f"\\u{codepoint:04x}")
        else:
            output.append(char)
    output.append('"')
    return "".join(output)


def _ecma_number(value: float) -> str:
    """The shortest decimal JSON.stringify uses for one finite binary64."""
    if not math.isfinite(value):
        return "null"
    if value == 0:
        return "0"
    negative = value < 0
    absolute = abs(value)
    text = repr(absolute).lower()
    mantissa, marker, exponent_text = text.partition("e")
    exponent = int(exponent_text) if marker else 0
    integer, dot, fraction = mantissa.partition(".")
    digits = integer + (fraction if dot else "")
    decimal_position = len(integer) + exponent
    while len(digits) > 1 and digits[0] == "0":
        digits = digits[1:]
        decimal_position -= 1
    while len(digits) > 1 and digits[-1] == "0":
        digits = digits[:-1]
    if 1e-6 <= absolute < 1e21:
        if decimal_position <= 0:
            rendered = "0." + "0" * (-decimal_position) + digits
        elif decimal_position >= len(digits):
            rendered = digits + "0" * (decimal_position - len(digits))
        else:
            rendered = digits[:decimal_position] + "." + digits[decimal_position:]
    else:
        power = decimal_position - 1
        significand = digits[0] + ("." + digits[1:] if len(digits) > 1 else "")
        rendered = significand + "e" + ("+" if power >= 0 else "") + str(power)
    return ("-" if negative else "") + rendered


def _json_text(value: Any) -> str:
    """Compact JSON with ECMAScript number and string spellings."""
    if value is None:
        return "null"
    if type(value) is bool:
        return "true" if value else "false"
    if isinstance(value, str):
        return _json_quote(value)
    if type(value) is int:
        if abs(value) <= 9007199254740991:
            return str(value)
        try:
            return _ecma_number(float(value))
        except OverflowError:
            return "null"
    if type(value) is float:
        return _ecma_number(value)
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_json_text(item) for item in value) + "]"
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            _fail("JSON object keys must be strings")
        indexed = []
        ordinary = []
        for key in value:
            index = _array_index(key)
            if index is None:
                ordinary.append(key)
            else:
                indexed.append((index, key))
        ordered = [key for _, key in sorted(indexed)] + ordinary
        return (
            "{" + ",".join(_json_quote(key) + ":" + _json_text(value[key]) for key in ordered) + "}"
        )
    _fail("Unsupported JSON data")


def _array_index(key: str) -> int | None:
    if not key or not key.isascii() or not key.isdecimal():
        return None
    if key != "0" and key.startswith("0"):
        return None
    number = int(key)
    return number if number < 0xFFFFFFFF else None


_OTHER_ID_START = {0x1885, 0x1886, 0x2118, 0x212E, 0x309B, 0x309C}
_OTHER_ID_CONTINUE = {0x00B7, 0x0387, *range(0x1369, 0x1372), 0x19DA}


def _identifier_start(char: str) -> bool:
    return (
        char in "$_"
        or ord(char) in _OTHER_ID_START
        or unicodedata.category(char)
        in {
            "Lu",
            "Ll",
            "Lt",
            "Lm",
            "Lo",
            "Nl",
        }
    )


def _identifier_continue(char: str) -> bool:
    return (
        _identifier_start(char)
        or char in "\u200c\u200d"
        or ord(char) in _OTHER_ID_CONTINUE
        or unicodedata.category(char) in {"Mn", "Mc", "Nd", "Pc"}
    )


def _scan_identifier(text: str, offset: int) -> tuple[str, int]:
    if offset >= len(text) or not _identifier_start(text[offset]):
        raise WitnessV6Error(
            "Invalid semantic-trace/6 text at " + str(offset) + ": expected source identifier"
        )
    end = offset + 1
    while end < len(text) and _identifier_continue(text[end]):
        end += 1
    return text[offset:end], end


def _is_symbol(value: str) -> bool:
    parts = value.split(".")
    if not parts:
        return False
    try:
        return all(_scan_identifier(part, 0)[1] == len(part) for part in parts)
    except WitnessV6Error:
        return False


_RESERVED = {"undefined", "null", "true", "false", "NaN", "Infinity", "typeof", "void"}
_NUMBER_TEXT = re.compile(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?")
_BIGINT_TEXT = re.compile(r"-?(?:0|[1-9][0-9]*)n")
_BINARY_TEXT = re.compile(r"(?:===|!==|>>>|\*\*|<=|>=|==|!=|<<|>>|[+*/%<>&|^\-])")


class _Reader:
    """Bounded tokenizer for the strict model-facing semantic-trace/6 text."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.offset = 0
        self.nodes = 0

    def fail(self, reason: str) -> None:
        raise WitnessV6Error(f"Invalid semantic-trace/6 text at {self.offset}: {reason}")

    def peek(self, value: str) -> bool:
        return self.text.startswith(value, self.offset)

    def take(self, value: str) -> None:
        if not self.peek(value):
            self.fail(f"expected {_json_text(value)}")
        self.offset += len(value)

    def maybe(self, value: str) -> bool:
        if not self.peek(value):
            return False
        self.offset += len(value)
        return True

    def match(self, pattern: re.Pattern[str], label: str) -> str:
        match = pattern.match(self.text, self.offset)
        if match is None:
            self.fail("expected " + label)
        self.offset = match.end()
        return match.group(0)

    def word(self) -> str:
        return self.match(re.compile(r"[a-z][a-z-]*"), "keyword")

    def identifier(self) -> str:
        value, self.offset = _scan_identifier(self.text, self.offset)
        return value

    def identity(self) -> str:
        self.take("@")
        return self.match(re.compile(r"[a-z][a-z0-9:_-]*"), "identity")

    def symbol(self) -> str:
        if self.peek('"'):
            return self.string()
        parts = [self.identifier()]
        while self.maybe("."):
            parts.append(self.identifier())
        return ".".join(parts)

    def string(self) -> str:
        result = self.json()
        if not isinstance(result, str):
            self.fail("expected string")
        return result

    def integer(self) -> int:
        text = self.match(re.compile(r"(?:0|[1-9][0-9]*)"), "nonnegative integer")
        number = int(text)
        if number > 9007199254740991:
            self.fail("integer range")
        return number

    def site(self) -> dict[str, int]:
        start = self.integer()
        self.take("..")
        return {"start": start, "end": self.integer()}

    def json(self) -> Any:
        begin = self.offset
        if self.offset < len(self.text) and self.text[self.offset] in "[{":
            stack: list[str] = []
            quoted = escaped = False
            while True:
                if self.offset >= len(self.text):
                    self.fail("unterminated JSON data")
                char = self.text[self.offset]
                self.offset += 1
                if quoted:
                    if escaped:
                        escaped = False
                    elif char == "\\":
                        escaped = True
                    elif char == '"':
                        quoted = False
                elif char == '"':
                    quoted = True
                elif char in "[{":
                    stack.append(char)
                    if len(stack) > 256:
                        self.fail("JSON nesting budget")
                elif char in "]}":
                    if not stack or stack.pop() != ("[" if char == "]" else "{"):
                        self.fail("JSON delimiter")
                    if not stack:
                        break
            self.offset = begin
        try:
            parser = json.JSONDecoder(
                object_pairs_hook=_pairs,
                parse_constant=lambda _: _fail("Nonfinite JSON literal"),
            )
            result, end = parser.raw_decode(self.text, self.offset)
        except (ValueError, TypeError) as error:
            raise WitnessV6Error(
                "Malformed semantic-trace/6 JSON at " + str(self.offset)
            ) from error
        original = self.text[begin:end]
        if _json_text(result) != original:
            self.fail("noncanonical JSON")
        self.offset = end
        return result

    def value(self) -> list[Any]:
        if self.peek('"'):
            return ["str", self.string()]
        if self.peek("@"):
            return ["ref", self.identity()]
        for text, tagged in (
            ("undefined", ["undefined"]),
            ("null", ["null"]),
            ("true", ["bool", True]),
            ("false", ["bool", False]),
            ("-Infinity", ["num", "-Infinity"]),
            ("Infinity", ["num", "+Infinity"]),
            ("NaN", ["num", "NaN"]),
        ):
            if self.maybe(text):
                return tagged
        bigint = _BIGINT_TEXT.match(self.text, self.offset)
        if bigint is not None:
            self.offset = bigint.end()
            return ["bigint", bigint.group(0)[:-1]]
        number = self.match(_NUMBER_TEXT, "value")
        try:
            numeric: int | float = (
                -0.0
                if number == "-0"
                else int(number)
                if not any(mark in number for mark in ".eE")
                else float(number)
            )
        except (ValueError, OverflowError) as error:
            raise WitnessV6Error("Invalid semantic-trace/6 number") from error
        if type(numeric) is float and numeric == 0 and math.copysign(1, numeric) < 0:
            return ["num", "-0"]
        return ["num", numeric]

    def expression(self) -> dict[str, Any]:
        self.nodes += 1
        if self.nodes > 32:
            self.fail("expression node budget")
        if self.maybe("("):
            for operator in ("typeof", "void", "!", "+", "-", "~"):
                if self.maybe(operator + " "):
                    arg = self.expression()
                    self.take(")")
                    return {"op": "unary", "operator": operator, "arg": arg}
            left = self.expression()
            self.take(" ")
            operator = self.match(_BINARY_TEXT, "binary operator")
            self.take(" ")
            right = self.expression()
            self.take(")")
            return {"op": "binary", "operator": operator, "left": left, "right": right}
        if self.maybe("local("):
            name = self.string()
            self.take(")")
            return {"op": "local", "name": name}
        if self.maybe("random()"):
            return {"op": "random"}
        if self.peek("@"):
            object_id = self.identity()
            if self.maybe("["):
                key = self.string()
                self.take("]")
                return {"op": "read", "object": object_id, "key": key}
            return {"op": "literal", "value": ["ref", object_id]}
        if self.offset < len(self.text) and _identifier_start(self.text[self.offset]):
            word = self.identifier()
            if word not in _RESERVED:
                return {"op": "local", "name": word}
            self.offset -= len(word)
        return {"op": "literal", "value": self.value()}

    def arguments(self) -> list[list[Any]]:
        self.take("(")
        result = []
        if self.maybe(")"):
            return result
        result.append(self.value())
        while self.maybe(", "):
            result.append(self.value())
        self.take(")")
        return result

    def properties(self) -> list[dict[str, Any]]:
        self.take("{")
        result = []
        if self.maybe("}"):
            return result
        result.append({"key": self.string(), "value": self._property_value_after_key()})
        while self.maybe(", "):
            result.append({"key": self.string(), "value": self._property_value_after_key()})
        self.take("}")
        return result

    def _property_value_after_key(self) -> list[Any]:
        self.take(": ")
        return self.value()

    def captures(self) -> list[dict[str, str]]:
        self.take("{")
        result = []
        if self.maybe("}"):
            return result
        result.append(self._capture())
        while self.maybe(", "):
            result.append(self._capture())
        self.take("}")
        return result

    def _capture(self) -> dict[str, str]:
        binding = self.identifier()
        self.take(": ")
        return {"binding": binding, "cell": self.identifier()}

    def read_record(self) -> tuple[str, dict[str, Any]]:
        kind: str
        fields: dict[str, Any] = {}
        start = self.offset
        try:
            self.identifier()
            is_binding = self.peek(" = ")
        except WitnessV6Error:
            is_binding = False
        self.offset = start
        if is_binding:
            kind = "bind"
            fields["name"] = self.identifier()
            self.take(" = ")
            fields["value"] = self.value()
            if self.maybe(" from "):
                fields["from"] = self.identifier()
        else:
            command = self.word()
            self.take(" ")
            kind = command
            if command == "source":
                kind = "code"
                fields["entry"] = self.string()
                self.take(" = ")
                fields["source"] = self.string()
                self.take(" sites ")
                fields["sites"] = self.json()
                if self.maybe(" seed "):
                    fields["randomSeed"] = self.integer()
            elif command == "input":
                fields["graph"] = self.json()
                if self.maybe(" retained "):
                    fields["retained"] = self.json()
            elif command == "compute":
                fields["expr"] = self.expression()
                self.take(" => ")
                fields["result"] = self.value()
                if self.maybe(" as "):
                    fields["bind"] = self.identifier()
            elif command == "choose":
                fields["expr"] = self.expression()
                self.take(" => ")
                fields["decision"] = self.value()
                self.take(" -> ")
                fields["path"] = self.word()
            elif command == "new":
                kind = "allocate"
                fields["name"] = self.identity()
                self.take(" ")
                fields["kind"] = self.word()
                self.take(" ")
                fields["properties"] = self.properties()
                if self.maybe(" length "):
                    fields["length"] = self.integer()
                if self.maybe(" code "):
                    fields["code"] = self.string()
                if self.maybe(" captures "):
                    fields["captures"] = self.captures()
            elif command in ("read", "write"):
                fields["object"] = self.identity()
                self.take("[")
                fields["key"] = self.string()
                self.take("]")
                self.take(" => " if command == "read" else " = ")
                fields["value"] = self.value()
                fields["present"] = not self.maybe(" absent")
            elif command == "call":
                fields["callee"] = self.symbol()
                fields["arguments"] = self.arguments()
                self.take(" this ")
                fields["receiver"] = self.value()
                self.take(" -> ")
                fields["child"] = self.identity()
            elif command == "transfer":
                fields["kind"] = self.word()
                self.take(" ")
                fields["value"] = self.value()
                self.take(" via ")
                fields["handler"] = self.json()
            elif command in ("return", "throw"):
                fields["value"] = self.value()
            elif command == "request":
                fields["effect"] = self.identity()
                self.take(" ")
                fields["operation"] = self.symbol()
                fields["arguments"] = self.arguments()
                self.take(" payload ")
                fields["payload"] = _json_text(self.json())
                if self.maybe(" promise "):
                    fields["promise"] = self.identity()
            elif command == "observe":
                fields["effect"] = self.identity()
                self.take(" ")
                fields["status"] = self.word()
                self.take(" ")
                fields["result"] = self.value()
                if self.maybe(" graph "):
                    fields["graph"] = self.json()
            elif command == "job":
                fields["action"] = self.word()
                self.take(" ")
                fields["job"] = self.identity()
                self.take(" for ")
                fields["owner"] = self.identity()
                if self.maybe(" result "):
                    fields["result"] = self.json()
                if self.maybe(" closure "):
                    fields["closure"] = self.identity()
            elif command == "promise":
                fields["action"] = self.word()
                self.take(" ")
                fields["promise"] = self.identity()
                if fields["action"] in {"resolve", "reject"}:
                    self.take(" = ")
                    fields["value"] = self.value()
                elif fields["action"] == "adopt":
                    self.take(" -> ")
                    fields["target"] = self.identity()
                elif fields["action"] == "react":
                    self.take(" ")
                    fields["reaction"] = self.word()
                    self.take(" as ")
                    fields["job"] = self.identity()
                    if self.maybe(" closure "):
                        fields["closure"] = self.identity()
                    if self.maybe(" reject "):
                        fields["rejectionClosure"] = self.identity()
                    if self.maybe(" target "):
                        fields["target"] = self.identity()
            elif command == "await":
                fields["promise"] = self.identity()
                self.take(" as ")
                fields["continuation"] = self.identity()
                if self.maybe(" returns "):
                    fields["returnPromise"] = self.identity()
            elif command == "resume":
                fields["continuation"] = self.identity()
                self.take(" ")
                fields["status"] = self.word()
                self.take(" ")
                fields["result"] = self.value()
            elif command == "end":
                fields["kind"] = self.word()
            elif command == "evaluate":
                kind = "summarize"
                fields["entry"] = self.symbol()
                fields["arguments"] = self.arguments()
                self.take(" => ")
                fields["result"] = self.value()
                fields["profile"] = "pure-helper/1"
                if self.maybe(" choosing "):
                    fields["profile"] = "pure-control-helper/1"
                    decisions = []
                    decisions.append(self._decision())
                    while self.maybe(", "):
                        decisions.append(self._decision())
                    fields["decisions"] = decisions
                elif self.maybe(" over "):
                    fields["profile"] = "dense-array-sum/1"
                    fields["iterations"] = self.integer()
                    self.take(" items into ")
                    fields["accumulator"] = self.identifier()
            elif command == "record":
                kind = "value-open"
                fields["id"] = self.identity()
                self.take(" type ")
                fields["type"] = self.string()
            elif command == "part":
                kind = "value-part"
                fields["id"] = self.identity()
                self.take(" ")
                fields["data"] = self.string()
            elif command == "close":
                kind = "value-close"
                fields["id"] = self.identity()
            else:
                self.fail("unknown record command")
        if self.maybe(" in "):
            fields["invocation"] = self.identity()
        if self.maybe(" at "):
            fields["site"] = self.site()
        if self.offset != len(self.text):
            self.fail("trailing or noncanonical bytes")
        return kind, fields

    def _decision(self) -> dict[str, Any]:
        site = self.site()
        self.take(":")
        return {"site": site, "selected": self.word()}


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
        (
            type(arg) in (float, int)
            and _finite_number(arg)
            and not (arg == 0 and math.copysign(1, arg) < 0)
        )
        or (isinstance(arg, str) and arg in SPECIAL)
    ):
        return
    if name == "ref" and isinstance(arg, str) and IDENTITY.fullmatch(arg):
        return
    _fail("Unsupported or noncanonical scalar")


def _finite_number(number: int | float) -> bool:
    try:
        return math.isfinite(number)
    except OverflowError:
        return False


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
            if node is not expr:
                _fail("Random transition requires a standalone computation")
            operators += 1
        elif op == "literal" and set(node) == {"op", "value"}:
            value(node["value"])
            operands += 1
        elif op == "local" and set(node) == {"op", "name"} and _source_binding(node["name"]):
            operands += 1
        elif (
            op == "read"
            and set(node) == {"op", "object", "key"}
            and isinstance(node["object"], str)
            and IDENTITY.fullmatch(node["object"])
            and isinstance(node["key"], str)
        ):
            operands += 1
        elif op == "unary" and set(node) == {"op", "operator", "arg"} and node["operator"] in UNARY:
            operators += 1
            visit(node["arg"])
        elif (
            op == "binary"
            and set(node) == {"op", "operator", "left", "right"}
            and node["operator"] in OPERATORS
        ):
            operators += 1
            visit(node["left"])
            visit(node["right"])
        else:
            _fail("Unregistered primitive expression")

    visit(expr)
    if operators > 8:
        _fail("Expression operator limit")
    return {"operatorCount": operators, "operandCount": operands, "nodeCount": nodes}


def graph(data: Any, *, retained: set[str] | None = None) -> None:
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
        if (
            not isinstance(name, str)
            or not IDENTITY.fullmatch(name)
            or name in ids
            or node["kind"] not in ("object", "array")
        ):
            _fail("Input graph identity")
        if "prototype" in node and (node["kind"] != "object" or node["prototype"] is not None):
            _fail("Input graph prototype")
        ids.add(name)
        properties = node["properties"]
        if not isinstance(properties, list):
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
            if (
                not isinstance(length, list)
                or len(length) != 2
                or length[0] != "num"
                or type(length[1]) is not int
                or not 0 <= length[1] <= 4096
            ):
                _fail("Array length")
            for key in keys:
                if (
                    key.isascii()
                    and key.isdecimal()
                    and str(int(key)) == key
                    and int(key) >= length[1]
                ):
                    _fail("Array index past length")
    for item in [*roots, *(p["value"] for n in objects for p in n["properties"])]:
        value(item)
        if item[0] == "ref" and item[1] not in ids | (retained or set()):
            _fail("Dangling input reference")


def parse_record(line: Any) -> tuple[str, dict[str, Any]]:
    if not isinstance(line, str) or "\n" in line or "\r" in line:
        _fail("Trace record is not one line")
    kind, fields = _Reader(line).read_record()
    if kind not in FIELDS or not REQUIRED[kind] <= set(fields) <= FIELDS[kind]:
        _fail("Unknown or missing typed operation fields")
    if kind == "code":
        if (
            not isinstance(fields["entry"], str)
            or not (
                _source_binding(fields["entry"])
                or re.fullmatch(r"function:[0-9]+", fields["entry"])
            )
            or not isinstance(fields["source"], str)
            or not fields["source"]
            or not isinstance(fields["sites"], list)
        ):
            _fail("Missing static code bundle")
        seed = fields.get("randomSeed")
        if "randomSeed" in fields and (type(seed) is not int or not 0 <= seed <= 0xFFFFFFFF):
            _fail("Invalid published deterministic source seed")
    elif kind == "input":
        retained = fields.get("retained", [])
        if (
            not isinstance(retained, list)
            or len(retained) != len(set(retained))
            or any(not isinstance(item, str) or not IDENTITY.fullmatch(item) for item in retained)
        ):
            _fail("Malformed retained reference grant")
        graph(fields["graph"], retained=set(retained))
    elif kind in ("compute", "choose"):
        expression(fields["expr"])
        value(fields["result"] if kind == "compute" else fields["decision"])
        if kind == "choose" and fields["path"] not in (
            "then",
            "else",
            "use-left",
            "evaluate-right",
        ):
            _fail("Unregistered branch path")
    elif kind in ("read", "write", "bind"):
        value(fields["value"])
    elif kind in ("return", "throw", "transfer"):
        value(fields["value"])
        if kind == "transfer":
            if fields["kind"] not in {"return", "throw", "cleanup"}:
                _fail("Unregistered completion transfer")
            _json_text(fields["handler"])
    elif kind == "allocate":
        if not IDENTITY.fullmatch(fields["name"]) or fields["kind"] not in (
            "array",
            "object",
            "closure",
            "set",
            "regexp",
        ):
            _fail("Unregistered allocation kind")
        if not isinstance(fields["properties"], list):
            _fail("Allocation properties must be a list")
        property_names: set[str] = set()
        for property in fields["properties"]:
            if (
                not isinstance(property, dict)
                or set(property) != {"key", "value"}
                or not isinstance(property["key"], str)
                or property["key"] in property_names
            ):
                _fail("Malformed or duplicate allocation property")
            property_names.add(property["key"])
            value(property["value"])
        if "length" in fields and (
            fields["kind"] != "array"
            or type(fields["length"]) is not int
            or not 0 <= fields["length"] <= 4096
        ):
            _fail("Invalid array allocation length")
        if "code" in fields and (
            fields["kind"] != "closure" or not isinstance(fields["code"], str) or not fields["code"]
        ):
            _fail("Invalid closure source code")
        if "captures" in fields and fields["kind"] != "closure":
            _fail("Captures require an owned closure")
        if "captures" in fields:
            if fields["kind"] != "closure" or not isinstance(fields["captures"], list):
                _fail("Invalid closure capture owner")
            used: set[str] = set()
            for relation in fields["captures"]:
                if (
                    not isinstance(relation, dict)
                    or set(relation) != {"binding", "cell"}
                    or not _source_binding(relation["binding"])
                    or relation["binding"] in used
                    # Both capture aliases and captured cells are source binding
                    # names, not heap reference identities.
                    or not _source_binding(relation["cell"])
                ):
                    _fail("Malformed shared cell capture relation")
                used.add(relation["binding"])
    elif kind == "request":
        site = fields["site"]
        if (
            not isinstance(fields["operation"], str)
            or not fields["operation"]
            or not isinstance(fields["effect"], str)
            or not IDENTITY.fullmatch(fields["effect"])
            or not isinstance(fields["arguments"], list)
            or len(fields["arguments"]) > 32
            or not isinstance(site, dict)
            or set(site) != {"start", "end"}
            or type(site["start"]) is not int
            or type(site["end"]) is not int
            or site["start"] < 0
            or site["end"] <= site["start"]
            or not isinstance(fields["payload"], str)
            or len(fields["payload"].encode("utf-8")) > 65536
        ):
            _fail("Invalid grounded external invocation")
        for argument in fields["arguments"]:
            value(argument)
        try:
            payload = json.loads(
                fields["payload"],
                object_pairs_hook=_pairs,
                parse_constant=lambda _: _fail("Nonfinite external JSON"),
            )
        except (TypeError, ValueError) as error:
            raise WitnessV6Error("Malformed external request payload") from error
        if (
            not isinstance(payload, list)
            and not (isinstance(payload, dict) and payload.get("schema") == "owned-graph/1")
        ) or _json_text(payload) != fields["payload"]:
            _fail("External request payload must be canonical declared JSON or owned graph data")
    elif kind == "observe":
        value(fields["result"])
        if fields["status"] not in ("fulfilled", "rejected"):
            _fail("Invalid observation status")
        if "graph" in fields:
            graph(fields["graph"])
            if fields["graph"]["roots"] != [fields["result"]]:
                _fail("Observed graph root differs from its supplied result")
    elif kind == "promise":
        action = fields["action"]
        extras = set(fields) - {"action", "promise", "invocation"}
        if action == "create" and extras:
            _fail("Promise creation cannot invent a settlement")
        elif action in {"resolve", "reject"}:
            if extras != {"value"}:
                _fail("Promise settlement needs its exact owned value")
            value(fields["value"])
        elif action == "adopt" and extras != {"target"}:
            _fail("Promise adoption needs a target promise")
        elif action == "react":
            if not {"job", "reaction"} <= extras <= {
                "job",
                "reaction",
                "closure",
                "rejectionClosure",
                "target",
            } or fields["reaction"] not in {"then", "catch", "finally"}:
                _fail("Promise reaction needs an owned job and declared reaction")
        elif action not in {"create", "resolve", "reject", "adopt", "react"}:
            _fail("Unknown owned Promise transition")
    elif kind == "resume":
        value(fields["result"])
        if fields["status"] not in {"fulfilled", "rejected"}:
            _fail("Resume requires an observed Promise settlement")
    elif kind == "end":
        if fields["kind"] not in ("return", "throw"):
            _fail("Invalid terminal completion")
    elif kind == "call":
        if not isinstance(fields["callee"], str) or not isinstance(fields["arguments"], list):
            _fail("Malformed call claim")
        value(fields["receiver"])
        for argument in fields["arguments"]:
            value(argument)
    elif kind == "value-open":
        if not isinstance(fields["type"], str) or not fields["type"]:
            _fail("Fragment type must be nonempty")
    elif kind == "value-part":
        if not isinstance(fields["data"], str):
            _fail("Fragment data must be text")
    elif kind == "summarize":
        if (
            not isinstance(fields["entry"], str)
            or not _source_binding(fields["entry"])
            or not isinstance(fields["arguments"], list)
            or len(fields["arguments"]) > 32
        ):
            _fail("Summary requires a public entry and bounded argument list")
        for argument in fields["arguments"]:
            value(argument)
        value(fields["result"])
        profile = fields["profile"]
        if profile == "pure-helper/1":
            if {"decisions", "iterations", "accumulator"} & set(fields):
                _fail("Pure helper summary has unrelated profile fields")
        elif profile == "pure-control-helper/1":
            decisions = fields.get("decisions")
            if (
                not isinstance(decisions, list)
                or not decisions
                or len(decisions) > 32
                or {"iterations", "accumulator"} & set(fields)
            ):
                _fail("Control summary needs ordered branch decisions only")
            for decision in decisions:
                if (
                    not isinstance(decision, dict)
                    or set(decision) != {"site", "selected"}
                    or not isinstance(decision["site"], dict)
                    or set(decision["site"]) != {"start", "end"}
                    or type(decision["site"]["start"]) is not int
                    or type(decision["site"]["end"]) is not int
                    or decision["site"]["start"] < 0
                    or decision["site"]["end"] <= decision["site"]["start"]
                    or decision["selected"] not in {"consequent", "alternate", "left", "right"}
                ):
                    _fail("Malformed source-bound control decision")
        elif profile == "dense-array-sum/1":
            if (
                "decisions" in fields
                or type(fields.get("iterations")) is not int
                or not 0 <= fields["iterations"] <= 1024
                or not _source_binding(fields.get("accumulator"))
            ):
                _fail("Array summary requires a bounded iteration count and source accumulator")
        else:
            _fail("Unknown summary profile")
    if kind == "job" and fields["action"] not in ("enqueue", "dispatch", "complete"):
        _fail("Invalid owned job transition")
    if kind == "job" and "closure" in fields and fields["action"] != "enqueue":
        _fail("Owned job closures belong only to registration")
    if kind == "promise" and "rejectionClosure" in fields and fields.get("reaction") != "then":
        _fail("Rejection closure requires a then reaction")
    if kind in ("read", "write") and (
        type(fields["present"]) is not bool or not isinstance(fields["key"], str)
    ):
        _fail("Property presence contract")
    for key in (
        "object",
        "effect",
        "promise",
        "continuation",
        "target",
        "job",
        "closure",
        "rejectionClosure",
        "returnPromise",
        "id",
    ):
        if key in fields and (
            not isinstance(fields[key], str) or not IDENTITY.fullmatch(fields[key])
        ):
            _fail("Invalid witness identity")
    for key in ("invocation", "child"):
        if key in fields and (
            not isinstance(fields[key], str) or not INVOCATION_IDENTITY.fullmatch(fields[key])
        ):
            _fail("Invalid dynamic invocation identity")
    if "name" in fields and not (
        IDENTITY.fullmatch(fields["name"])
        if kind == "allocate"
        else _source_binding(fields["name"])
    ):
        _fail("Invalid source binding name")
    for key in ("bind", "from"):
        if key in fields and not _source_binding(fields[key]):
            _fail("Invalid source binding name")
    if "from" in fields and fields["from"] == fields.get("name"):
        _fail("A binding cannot claim itself as its source cell")
    if "site" in fields and (
        not isinstance(fields["site"], dict)
        or set(fields["site"]) != {"start", "end"}
        or type(fields["site"]["start"]) is not int
        or type(fields["site"]["end"]) is not int
        or fields["site"]["start"] < 0
        or fields["site"]["end"] <= fields["site"]["start"]
    ):
        _fail("Invalid source span")
    if encode_record(kind, fields) != line:
        _fail("Noncanonical semantic-trace/6 bytes")
    return kind, fields
