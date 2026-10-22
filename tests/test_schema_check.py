"""The quick check of plain tool schemas never accepts a schema the metaschema refuses."""

from __future__ import annotations

import json
import random

import jsonschema
import pytest

from megatoken.generation.structure import _is_plain_schema, _validator

VALUES = [
    None,
    True,
    False,
    0,
    1,
    -1,
    2.5,
    1.0,
    "",
    "string",
    "object",
    "integer",
    "bogus",
    [],
    ["a"],
    ["a", "a"],
    ["string", "null"],
    ["string", "string"],
    [1],
    {},
    {"type": "string"},
    {"type": "bogus"},
    [{"type": "string"}],
    [{"type": "string"}, True],
    [{"type": 5}],
]
KEYWORDS = [
    "type",
    "properties",
    "required",
    "enum",
    "items",
    "additionalProperties",
    "anyOf",
    "allOf",
    "oneOf",
    "not",
    "prefixItems",
    "minimum",
    "maximum",
    "minLength",
    "maxItems",
    "title",
    "description",
    "default",
    "const",
    "deprecated",
    "examples",
    "pattern",
    "format",
    "$ref",
    "$defs",
    "$schema",
    "multipleOf",
    "unknownKeyword",
    "contentMediaType",
    "contentEncoding",
    "contentSchema",
    "example",
    "dependencies",
]


def random_schema(rng: random.Random, depth: int = 0):
    if depth > 3 or rng.random() < 0.15:
        return rng.choice(VALUES)
    schema = {}
    for _ in range(rng.randint(0, 4)):
        keyword = rng.choice(KEYWORDS)
        roll = rng.random()
        if roll < 0.45:
            schema[keyword] = random_schema(rng, depth + 1)
        elif roll < 0.6 and keyword in {"properties", "$defs"}:
            schema[keyword] = {
                rng.choice(["a", "b", "type"]): random_schema(rng, depth + 1)
                for _ in range(rng.randint(0, 3))
            }
        elif roll < 0.7 and keyword in {"anyOf", "allOf", "oneOf", "prefixItems"}:
            schema[keyword] = [random_schema(rng, depth + 1) for _ in range(rng.randint(0, 3))]
        else:
            schema[keyword] = rng.choice(VALUES)
    return schema


def refused(schema) -> bool:
    try:
        jsonschema.Draft202012Validator.check_schema(schema)
    except jsonschema.SchemaError:
        return True
    return False


def test_quick_acceptance_is_sound_against_the_metaschema():
    rng = random.Random(20261022)
    accepted = 0
    for _ in range(6000):
        schema = random_schema(rng)
        if _is_plain_schema(schema):
            accepted += 1
            assert not refused(schema), json.dumps(schema)
    assert accepted > 300  # the fuzz reaches the quick path often enough to mean something


@pytest.mark.parametrize(
    "schema",
    [
        {"type": "object", "properties": {"city": {"type": "string", "description": "Where"}}},
        {"type": ["string", "null"], "enum": ["a", None]},
        {"type": "array", "items": {"type": "integer", "minimum": 0}, "minItems": 1},
        {"anyOf": [{"type": "string"}, {"type": "number"}], "default": 3},
        True,
    ],
)
def test_common_tool_schemas_take_the_quick_path(schema):
    assert _is_plain_schema(schema)
    assert not refused(schema)


@pytest.mark.parametrize(
    "schema",
    [
        {"type": "bogus"},
        {"type": []},
        {"type": ["string", "string"]},
        {"required": ["a", "a"]},
        {"properties": {"a": 5}},
        {"minLength": -1},
        {"minItems": True},
        {"pattern": "("},
        {"$ref": "#/$defs/x"},
        {"items": [{"type": "string"}]},
    ],
)
def test_anything_else_goes_to_the_full_check(schema):
    assert not _is_plain_schema(schema)


def test_an_invalid_schema_is_still_refused_when_building_the_validator():
    _validator.cache_clear()
    with pytest.raises(jsonschema.SchemaError):
        _validator(json.dumps({"type": "bogus"}))
    assert _validator(json.dumps({"type": "string"})).is_valid("x")
