"""Wire-neutral typed semantic value, expression, graph and operation validation."""

from __future__ import annotations

import json
import math
import re
import unicodedata
from typing import Any


class SemanticRecordError(ValueError):
    pass


def _fail(reason: str) -> None:
    raise SemanticRecordError("Invalid typed semantic record: " + reason)


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

# ECMAScript IdentifierStart/IdentifierPart exceptions used in addition to
# Unicode general categories. Kept local so the typed grammar is independent of
# historical witness parsers.
_OTHER_ID_START = {0x1885, 0x1886, 0x2118, 0x212E, 0x309B, 0x309C}
_OTHER_ID_CONTINUE = {0x00B7, 0x0387, *range(0x1369, 0x1372), 0x19DA}

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


def _source_binding(name: Any) -> bool:
    if not isinstance(name, str):
        return False
    try:
        return _scan_identifier(name, 0)[1] == len(name)
    except SemanticRecordError:
        return False


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail("Duplicate JSON field")
        result[key] = value
    return result


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
        raise SemanticRecordError(
            "Invalid semantic-trace/6 text at " + str(offset) + ": expected source identifier"
        )
    end = offset + 1
    while end < len(text) and _identifier_continue(text[end]):
        end += 1
    return text[offset:end], end


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


def validate_detailed_record(kind: str, fields: dict) -> bool:
    if not isinstance(kind, str) or type(fields) is not dict:
        _fail("Record kind and fields must be plain declared data")
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
            raise SemanticRecordError("Malformed external request payload") from error
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
    return True


GENERATED_KINDS = {
    "compute",
    "choose",
    "allocate",
    "read",
    "write",
    "bind",
    "call",
    "transfer",
    "return",
    "throw",
    "request",
    "job",
    "promise",
    "await",
    "resume",
    "end",
    "summarize",
}
OBSERVED_KINDS = {"observe"}
LOGICAL_FRAGMENTS = {"value-open", "value-part", "value-close"}
MAX_RECORD_BYTES = 65536


def validate_record_role(kind: str, role: str) -> None:
    if role not in {"any", "generated", "observed"}:
        _fail("Unknown authority role")
    if kind in LOGICAL_FRAGMENTS:
        _fail("Logical fragments are forbidden; native physical fragmentation is separate")
    if role == "generated" and kind not in GENERATED_KINDS:
        _fail("Only declared generated detailed records are selectable")
    if role == "observed" and kind not in OBSERVED_KINDS:
        _fail("Observed detail must be an environment observation")
