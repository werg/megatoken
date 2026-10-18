"""A conservative, executable wire-size witness for public typed schemas.

A schema is admitted only when at least one deterministic, schema-valid minimal
example serializes within the requested total token budget. This is a feasibility
gate, not a proof that every later lexical choice fits: final serialization still
checks the exact chosen bytes and BPE boundaries.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from jsonschema.exceptions import ValidationError

from .structure import FieldSpec, SchemaSpec


def _example_json(schema: Mapping[str, Any], depth: int = 0) -> Any:
    if depth > 8:
        raise ValueError("JSON schema example recursion limit")
    if "const" in schema:
        return schema["const"]
    if "enum" in schema and schema["enum"]:
        return min(schema["enum"], key=lambda v: len(json.dumps(v, separators=(",", ":"))))
    if "$ref" in schema or "not" in schema or "allOf" in schema:
        raise ValueError("Cannot construct a safe example for a recursive/compound schema")
    for alternative in ("anyOf", "oneOf"):
        if alternative in schema:
            for choice in schema[alternative]:
                try:
                    return _example_json(choice, depth + 1)
                except (KeyError, TypeError, ValueError):
                    continue
            raise ValueError("No supported example for union schema")
    kind = schema.get("type")
    if isinstance(kind, list):
        kind = next((k for k in kind if k != "null"), kind[0])
    if kind == "object":
        properties = schema.get("properties", {})
        required = list(schema.get("required", []))
        # Common optional properties do not need to be manufactured.
        if len(required) < schema.get("minProperties", 0):
            required.extend(k for k in properties if k not in required)
            required = required[: schema["minProperties"]]
        if any(k not in properties for k in required):
            raise ValueError("Unknown required property")
        return {name: _example_json(properties[name], depth + 1) for name in required}
    if kind == "array":
        count = schema.get("minItems", 0)
        if count > 32:
            raise ValueError("Excessive schema minimum")
        prefix = schema.get("prefixItems", ())
        rest = schema.get("items", {})
        return [
            _example_json(prefix[i] if i < len(prefix) else rest, depth + 1) for i in range(count)
        ]
    if kind == "string":
        if any(key in schema for key in ("pattern", "format")):
            raise ValueError("Unresolved string constraint")
        return "a" * int(schema.get("minLength", 0))
    if kind in ("integer", "number"):
        minimum = schema.get("minimum", 0)
        exclusive = schema.get("exclusiveMinimum")
        candidate = max(minimum, 0)
        if exclusive is not None and candidate <= exclusive:
            candidate = exclusive + 1
        multiple = schema.get("multipleOf")
        if multiple:
            import math

            candidate = math.ceil(candidate / multiple) * multiple
        return int(candidate) if kind == "integer" else candidate
    if kind == "boolean":
        return False
    if kind == "null":
        return None
    if kind is None:
        # A valid one-byte JSON number is a smaller witness than "null".
        return 0
    raise ValueError(f"Unsupported field type {kind!r}")


def _field_witness(field: FieldSpec) -> bytes:
    if field.fixed_json is not None:
        return field.fixed_json.encode("utf-8")
    if field.kind == "json":
        schema = json.loads(field.json_schema) if field.json_schema else {}
        return json.dumps(_example_json(schema), ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
    return ("a" * field.min_tokens).encode("utf-8")


def wire_witness_tokens(schema: SchemaSpec, codec: Any) -> int | None:
    """Smallest found *valid* serialized witness among public layouts.

    None means no reliable witness could be constructed, so the decoder must
    fall back to normal schema validation rather than silently assume validity.
    """
    values = {}
    identifiers = {}
    try:
        for field in schema.fields:
            if field.bound_to:
                value = identifiers[field.bound_to]
            else:
                value = _field_witness(field)
            if field.identifier:
                identifiers[field.identifier] = value
            values[field.name] = value
        sizes = []
        for layout in range(len(schema.orders)):
            serialized = schema.serialize(values, layout)
            sizes.append(len(codec.encode(serialized)))
        return min(sizes) if sizes else None
    except (KeyError, TypeError, ValueError, ValidationError, UnicodeError, RecursionError):
        return None


def wire_overhead_tokens(schema: SchemaSpec, codec: Any) -> int:
    """Reserve the fixed JSON wire, independently of modelled field-value tokens.

    Tokenize literal syntax fragments separately. For byte-fallback this is
    exact; for BPE this is a conservative reservation that does not rely on
    merges across the boundary between a field value and its envelope.
    Fixed JSON field values are *not* included: planned_fields already charges
    their tokenized lengths. Additional escaping of sampled string contents
    remains subject to the final exact serialized-size check.
    """
    costs = []
    fields = {field.name: field for field in schema.fields}
    for order in schema.orders:
        parts = [b"{", b"}"]
        for index, name in enumerate(order):
            if index:
                parts.append(b",")
            parts.append(json.dumps(name, ensure_ascii=False).encode("utf-8") + b":")
            field = fields[name]
            if field.fixed_json is None and field.kind != "json":
                parts.extend((b'"', b'"'))
        costs.append(sum(len(codec.encode(part)) for part in parts))
    return max(costs)
