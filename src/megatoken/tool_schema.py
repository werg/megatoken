"""Compile public tool definitions into the shared structured action policy."""

from __future__ import annotations

import json
from dataclasses import replace
from functools import lru_cache
from typing import Any

from megatoken.generation.structure import (
    ActionSpec,
    CopySource,
    FieldSpec,
    SchemaSpec,
    strict_json_loads,
)


@lru_cache(maxsize=1024)
def compile_tool_spec(
    tools_json: str,
    max_tokens: int = 1024,
    wire_formats: tuple[str, ...] = ("canonical",),
) -> ActionSpec:
    """Use only schemas available before the action, never infer one from its target.

    The standard execution envelope is {name, arguments}. Nested arguments retain
    their real JSON types and are validated against the supplied JSON Schema.
    Fixed names, braces, keys, and escaping are deterministic serialization.
    """
    value = strict_json_loads(tools_json)
    if not isinstance(value, list) or not value:
        raise ValueError("Public tool schemas must be a nonempty JSON list")
    if not wire_formats or any(
        wire not in {"canonical", "openai", "anthropic"} for wire in wire_formats
    ):
        raise ValueError("Unknown tool wire format")
    schemas = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("Every public tool definition must be a JSON object")
        function = item.get("function", item)
        if not isinstance(function, dict) or "name" not in function:
            raise ValueError("Every public function definition must be an object with a name")
        name = function["name"]
        parameters = function.get("parameters", function.get("input_schema", {"type": "object"}))
        if not isinstance(name, str) or not name or not isinstance(parameters, dict):
            raise ValueError("Invalid tool name or parameters schema")
        argument_field = FieldSpec(
            "arguments",
            kind="json",
            json_schema=json.dumps(parameters, sort_keys=True),
            min_tokens=1,
            max_tokens=max_tokens,
        )
        fixed = parameters.get("const")
        if (
            parameters.get("type") == "object"
            and parameters.get("properties") == {}
            and parameters.get("additionalProperties") is False
        ):
            fixed = {}
        if fixed is not None:
            # A public singleton value contains no policy choice. Deterministic
            # serialization avoids wasting lexical capacity on known syntax.
            argument_field = replace(
                argument_field,
                min_tokens=0,
                fixed_json=json.dumps(fixed, separators=(",", ":"), ensure_ascii=False),
            )
        name_field = FieldSpec(
            "name", kind="json", fixed_json=json.dumps(name), max_tokens=max_tokens
        )
        if "canonical" in wire_formats:
            schemas.append(
                SchemaSpec(
                    name + ":canonical",
                    (name_field, argument_field),
                    layouts=(("arguments", "name"), ("name", "arguments")),
                )
            )
        if "openai" in wire_formats:
            function_schema = {
                "type": "object",
                "properties": {
                    "name": {"const": name},
                    "arguments": {
                        "type": "string",
                        "contentMediaType": "application/json",
                        "contentSchema": parameters,
                    },
                },
                "required": ["name", "arguments"],
                "additionalProperties": False,
            }
            schemas.append(
                SchemaSpec(
                    name + ":openai",
                    (
                        FieldSpec(
                            "function",
                            kind="json",
                            json_schema=json.dumps(function_schema),
                            min_tokens=1,
                            max_tokens=max_tokens,
                        ),
                        FieldSpec(
                            "id",
                            kind="identifier",
                            identifier="call_id",
                            min_tokens=1,
                            max_tokens=min(64, max_tokens),
                        ),
                        FieldSpec(
                            "type", kind="json", fixed_json='"function"', max_tokens=max_tokens
                        ),
                    ),
                    layouts=(("function", "id", "type"), ("id", "type", "function")),
                )
            )
        if "anthropic" in wire_formats:
            schemas.append(
                SchemaSpec(
                    name + ":anthropic",
                    (
                        FieldSpec(
                            "id",
                            kind="identifier",
                            identifier="call_id",
                            min_tokens=1,
                            max_tokens=min(64, max_tokens),
                        ),
                        name_field,
                        replace(argument_field, name="input"),
                        FieldSpec(
                            "type", kind="json", fixed_json='"tool_use"', max_tokens=max_tokens
                        ),
                    ),
                    layouts=(("id", "input", "name", "type"), ("type", "id", "name", "input")),
                )
            )
    return ActionSpec(
        allow_text=False,
        schemas=tuple(schemas),
        max_tokens=max_tokens,
        require_terminal=False,
    )


def matching_schema(spec: ActionSpec, payload: bytes) -> int | None:
    """Select a supervised structural label; it is not a model conditioning input."""
    try:
        value: Any = json.loads(payload)
    except (ValueError, UnicodeError):
        return None
    if not isinstance(value, dict):
        return None
    try:
        name, _, _, wire = parse_tool_action(payload)
    except ValueError:
        return None
    return next(
        (index for index, schema in enumerate(spec.schemas) if schema.name == name + ":" + wire),
        None,
    )


def parse_tool_action(payload: bytes) -> tuple[str, dict[str, Any], str | None, str]:
    """Normalize complete wire objects without reparsing or executing partial records."""
    try:
        value = strict_json_loads(payload)
        if not isinstance(value, dict):
            raise ValueError("Tool action must be a JSON object")
        keys = set(value)
        if keys == {"name", "arguments"}:
            name, arguments, call_id, wire = value["name"], value["arguments"], None, "canonical"
        elif keys == {"function", "id", "type"} and value["type"] == "function":
            function = value["function"]
            name, arguments = function["name"], function["arguments"]
            arguments = strict_json_loads(arguments) if isinstance(arguments, str) else arguments
            call_id, wire = value["id"], "openai"
        elif keys == {"id", "input", "name", "type"} and value["type"] == "tool_use":
            name, arguments, call_id, wire = value["name"], value["input"], value["id"], "anthropic"
        else:
            raise ValueError("Unsupported complete tool action envelope")
        if not isinstance(name, str) or not name or not isinstance(arguments, dict):
            raise ValueError("Tool action needs a name and a JSON object of arguments")
        if call_id is not None and (not isinstance(call_id, str) or not call_id):
            raise ValueError("Tool call ID must be a nonempty string")
        return name, arguments, call_id, wire
    except (KeyError, TypeError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid complete tool action") from exc


def recent_copy_sources(
    records: list[Any], *, limit: int = 16, max_bytes: int = 256
) -> tuple[CopySource, ...]:
    """A bounded literal index of actual earlier records, independent of the target."""
    candidates: list[CopySource] = []
    seen: set[bytes] = set()

    def add(name: str, payload: bytes) -> None:
        if payload and len(payload) <= max_bytes and payload not in seen:
            seen.add(payload)
            candidates.append(CopySource(name, payload))

    def visit(name: str, value: Any) -> None:
        if isinstance(value, str):
            add(name, value.encode())
        elif isinstance(value, dict):
            for key, child in value.items():
                visit(name + "/" + str(key), child)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(name + f"/{index}", child)

    for index, record in enumerate(records[-8:]):
        payload = record if isinstance(record, bytes) else record.payload
        add(f"recent-{index}", payload)
        try:
            visit(f"recent-{index}", json.loads(payload))
        except (ValueError, UnicodeError):
            pass
    return tuple(candidates[-limit:])


def supervised_bindings(spec: ActionSpec, schema_index: int, payload: bytes) -> dict[str, int]:
    """Derive only target labels for available copy sources; no new sources are invented."""
    parsed = json.loads(payload)
    result = {}
    for field in spec.schemas[schema_index].fields:
        if field.fixed_json is not None or field.bound_to or not field.allow_copy:
            continue
        value = parsed[field.name]
        raw = (
            json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode()
            if field.kind == "json"
            else value.encode()
        )
        choices = [
            (len(source.payload), index + 1)
            for index, source in enumerate(spec.copy_sources)
            if source.payload in raw and (not field.identifier or source.payload == raw)
        ]
        if choices:
            result[field.name] = max(choices)[1]
    return result
