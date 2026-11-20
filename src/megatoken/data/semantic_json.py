"""Canonical ECMAScript-compatible JSON and hashes, independent of semantic wire versions."""

from __future__ import annotations

import hashlib
import math
from typing import Any


class SemanticJSONError(ValueError):
    pass


def _error(message: str) -> None:
    raise SemanticJSONError("Invalid semantic data: " + message)


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
