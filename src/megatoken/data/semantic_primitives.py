"""Independent ECMAScript primitives and ordinary data-heap operations.

This module has no trace-format client, record parser, source execution, or
legacy wire dependency. Typed callers own their grammar and resource policy.
References are opaque identities resolved only through the supplied data heap.
"""

from __future__ import annotations

import math
import re
from typing import Any

from .semantic_json import (
    SemanticJSONError as SemanticValueError,
)
from .semantic_json import (
    _array_index as _array_index,
)
from .semantic_json import (
    _as_js_number,
    _js_string_units,
    _number_text,
)

MAX_DENSE_ITEMS = 128


def _error(message: str) -> None:
    raise SemanticValueError(f"Invalid semantic data: {message}")


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

_SPECIAL_NUMBERS = {"-0", "NaN", "+Infinity", "-Infinity"}


class SemanticRangeError(ArithmeticError):
    """Guest ECMAScript RangeError raised by a valid primitive operation."""

    name = "RangeError"


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
        if name not in allowed or (name.startswith("loose_") and not both_refs and not null_loose):
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
                raise SemanticRangeError("BigInt division by zero")
            result = abs(a) // abs(b) * (-1 if (a < 0) != (b < 0) else 1)
        else:
            result = _js_divide(a, b)
    elif name == "remainder":
        if kind_a == "bigint":
            if b == 0:
                raise SemanticRangeError("BigInt remainder by zero")
            quotient = abs(a) // abs(b) * (-1 if (a < 0) != (b < 0) else 1)
            result = a - quotient * b
        else:
            result = _js_remainder(a, b)
    elif name == "power":
        if kind_a == "bigint":
            if b < 0:
                raise SemanticRangeError("BigInt exponent must be nonnegative")
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
    raise SemanticValueError("Algorithm state must be a data record or list of pairs")


_MISSING = object()


def _property_rows(properties: Any) -> list[tuple[str, Any]]:
    rows: list[tuple[str, Any]] = []
    if type(properties) is list:
        for prop in properties:
            if type(prop) is list:
                _data_list(prop, "data property")
                if len(prop) != 2:
                    raise SemanticValueError("Algorithm data property pair")
                key, value = prop
            elif _plain_dict(prop) and "key" in prop and "value" in prop:
                key, value = prop["key"], prop["value"]
            else:
                raise SemanticValueError("Algorithm data property descriptor")
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
        raise SemanticValueError("Algorithm needs committed data properties")
    result: dict[str, Any] = {}
    for key, value in rows:
        if type(key) is not str or key == "__proto__":
            raise SemanticValueError("Algorithm data property key")
        if key in result:
            raise SemanticValueError("Duplicate algorithm data property")
        _validate_value(value)
        result[key] = value
    return list(result.items())


def _heap_map(heap: Any) -> dict[str, dict]:
    if heap is None:
        return {}
    if not _plain_dict(heap):
        raise SemanticValueError("Algorithm heap must be an ordinary data record")
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
                raise SemanticValueError("Algorithm graph node schema")
            identity = _identity(source["id"], "heap identity")
            if identity in result:
                raise SemanticValueError("Duplicate algorithm heap identity")
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
        raise SemanticValueError("Algorithm needs an ordinary data object")
    if "accessors" in node and (not _plain_dict(node["accessors"]) or len(node["accessors"]) != 0):
        raise SemanticValueError("Algorithm needs an ordinary data object")
    if "proxy" in node or "iterator" in node:
        raise SemanticValueError("Algorithm needs an ordinary data object")
    prototype = node.get("prototype", _MISSING)
    if (
        prototype is not _MISSING
        and prototype is not None
        and prototype != ("Array.prototype" if node["kind"] == "array" else "Object.prototype")
    ):
        raise SemanticValueError("Algorithm needs an ordinary data object")
    if node["kind"] == "array" and prototype is None:
        raise SemanticValueError("Array prototype is outside the dense profile")
    properties: dict[str, list] = {}
    for key, value in _property_rows(node.get("properties")):
        properties[key] = value
    return node, properties


def _snapshot_heap(heap: Any) -> dict[str, dict]:
    raw = _heap_map(heap)
    if len(raw) > 256:
        raise SemanticValueError("Algorithm heap shape or node budget")
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
                raise SemanticValueError("Dangling algorithm heap reference " + value[1])
    return result


def _dense_items(value: list, heap: dict[str, dict]) -> list[list]:
    _validate_value(value)
    if value[0] != "ref":
        raise SemanticValueError("Ordered fold requires a dense numeric array identity")
    node, props = _heap_node(heap, value[1])
    length = props.get("length")
    if (
        node["kind"] != "array"
        or length is None
        or length[0] != "num"
        or type(length[1]) not in (int, float)
    ):
        raise SemanticValueError("Ordered fold array bound")
    count_float = _as_js_number(length[1])
    if (
        not math.isfinite(count_float)
        or not count_float.is_integer()
        or count_float < 0
        or count_float > MAX_DENSE_ITEMS
    ):
        raise SemanticValueError("Ordered fold array bound")
    count = int(count_float)
    if len(props) != count + 1:
        raise SemanticValueError("Ordered fold refuses holes or extra properties")
    items = []
    for index in range(count):
        item = props.get(str(index))
        if item is None or item[0] != "num" or not math.isfinite(_decode_scalar(item)):
            raise SemanticValueError("Ordered fold requires dense finite binary64 elements")
        items.append(item)
    return items


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
        raise SemanticValueError("Observable graph schema")
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
            raise SemanticValueError("Observable object schema")
        identity = _identity(node["id"])
        if identity in source:
            raise SemanticValueError("Duplicate observable identity")
        prototype = node.get("prototype", _MISSING)
        if prototype is not _MISSING and prototype is not None:
            raise SemanticValueError("Observable prototype profile")
        if node["kind"] == "array" and prototype is None:
            raise SemanticValueError("Observable array prototype")
        source[identity] = node
        _data_list(node["properties"], "observable properties")
    input_ids = [] if input_ids is None else _data_list(input_ids, "input identity anchors")
    if len(set(input_ids)) != len(input_ids):
        raise SemanticValueError("Observable input identity anchors")
    ids: dict[str, str] = {}
    for index, identity in enumerate(input_ids):
        _identity(identity)
        if identity not in source:
            raise SemanticValueError("Missing input identity anchor")
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
            raise SemanticValueError("Dangling observable identity " + identity)
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
                    raise SemanticValueError("Observable data property")
                if type(key) is not str or key == "__proto__" or key in seen_keys:
                    raise SemanticValueError("Observable property identity")
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
                    raise SemanticValueError("Observable array length")
            active.remove(identity)
        return ["ref", canonical_id]

    # Input anchors are always retained, even if no root directly mentions them.
    for identity in input_ids:
        rewrite(["ref", identity])
    canonical_roots = [rewrite(value) for value in roots]
    return {"roots": canonical_roots, "objects": objects}


def algorithm_heap_from_graph(graph: Any) -> dict[str, dict]:
    """Convert a typed input graph envelope into the heap accepted by the core."""
    if not _plain_dict(graph) or "objects" not in graph or "roots" not in graph:
        raise SemanticValueError("Algorithm input graph schema")
    _data_list(graph["roots"], "input graph roots")
    result = {}
    for node in _data_list(graph["objects"], "input graph objects"):
        if not _plain_dict(node) or not {"id", "kind", "properties"}.issubset(node):
            raise SemanticValueError("Algorithm graph node schema")
        identity = _identity(node["id"])
        if identity in result:
            raise SemanticValueError("Duplicate algorithm heap identity")
        props = []
        for prop in _data_list(node["properties"], "input graph properties"):
            if _plain_dict(prop) and {"key", "value"}.issubset(prop):
                props.append([prop["key"], prop["value"]])
            elif type(prop) is list and len(prop) == 2:
                props.append(prop)
            else:
                raise SemanticValueError("Algorithm graph property")
        result[identity] = {
            "kind": node["kind"],
            **({"prototype": node["prototype"]} if "prototype" in node else {}),
            "properties": props,
        }
    return result
