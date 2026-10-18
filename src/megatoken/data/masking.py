"""Causal auxiliary text masks from already supplied public action schemas."""

from __future__ import annotations

import json
from dataclasses import replace

from .schema import ByteSpan, Episode


def _value_spans(payload: bytes) -> dict[str, tuple[int, int]]:
    """Find top-level JSON values while retaining their exact encoded byte spans."""
    text = payload.decode("utf-8")
    decoder = json.JSONDecoder()
    offsets = [0]
    for character in text:
        offsets.append(offsets[-1] + len(character.encode("utf-8")))
    position = 0

    def whitespace() -> None:
        nonlocal position
        while position < len(text) and text[position] in " \t\r\n":
            position += 1

    whitespace()
    if position >= len(text) or text[position] != "{":
        raise ValueError("A structured action must be a complete JSON object")
    position += 1
    spans = {}
    while True:
        whitespace()
        if position < len(text) and text[position] == "}":
            position += 1
            break
        key, position = decoder.raw_decode(text, position)
        if not isinstance(key, str) or key in spans:
            raise ValueError("Action keys must be unique JSON strings")
        whitespace()
        if position >= len(text) or text[position] != ":":
            raise ValueError("Invalid action object separator")
        position += 1
        whitespace()
        start = position
        _, position = decoder.raw_decode(text, position)
        spans[key] = (offsets[start], offsets[position])
        whitespace()
        if position < len(text) and text[position] == ",":
            position += 1
            continue
        if position < len(text) and text[position] == "}":
            position += 1
            break
        raise ValueError("Invalid action object delimiter")
    whitespace()
    if position != len(text):
        raise ValueError("Unexpected bytes after a complete action")
    return spans


def tool_free_spans(
    payload: bytes, tools_json: str, *, max_tokens: int = 4096
) -> tuple[ByteSpan, ...] | None:
    """Mask known deterministic syntax for the auxiliary raw reconstruction loss.

    The matched schema is a target label selected from already public tools. No
    schema is invented from target bytes. Actual ActionTape likelihood remains
    the native policy's responsibility. Unknown/legacy wire formats retain their
    explicitly raw-text objective. Native emitted IDs are never retokenized here.
    """
    if not json.loads(tools_json):
        return None
    from megatoken.tool_schema import compile_tool_spec, matching_schema

    spec = compile_tool_spec(
        tools_json, max_tokens=max_tokens, wire_formats=("canonical", "openai", "anthropic")
    )
    schema_index = matching_schema(spec, payload)
    if schema_index is None:
        return None
    values = _value_spans(payload)
    spans = []
    for field in spec.schemas[schema_index].fields:
        if field.fixed_json is not None or field.bound_to:
            continue
        start, end = values[field.name]
        if field.kind == "json":
            # Nested JSON punctuation is sampled as part of this free field by
            # the current realizer, so it belongs inside the lexical mask.
            if end > start:
                spans.append(ByteSpan(start, end))
            continue
        if payload[start : start + 1] != b'"' or payload[end - 1 : end] != b'"':
            raise ValueError("A text/identifier action field must be a JSON string")
        start, end = start + 1, end - 1
        position = run_start = start
        while position < end:
            if payload[position] == 92:
                if position > run_start:
                    spans.append(ByteSpan(run_start, position))
                # JSON escaping is deterministic framing for non-JSON fields.
                position += 6 if payload[position + 1 : position + 2] == b"u" else 2
                run_start = position
            else:
                position += 1
        if end > run_start:
            spans.append(ByteSpan(run_start, end))
    return tuple(sorted(spans))


def apply_tool_masks(episode: Episode, *, max_tokens: int = 4096) -> Episode:
    """Project public schema masks onto immutable native physical boundaries."""
    if not episode.actions or not json.loads(episode.tools_json):
        return episode
    records = []
    for group in episode.iter_logical_records():
        spans = (
            tool_free_spans(
                b"".join(record.payload for record in group),
                episode.tools_json,
                max_tokens=max_tokens,
            )
            if group[0].action_id
            else None
        )
        offset = 0
        for record in group:
            if spans is not None:
                local = tuple(
                    ByteSpan(
                        max(0, span.start - offset), min(len(record.payload), span.end - offset)
                    )
                    for span in spans
                    if span.start < offset + len(record.payload) and span.end > offset
                )
                if record.free_spans is not None:
                    local = tuple(
                        ByteSpan(max(left.start, right.start), min(left.end, right.end))
                        for left in local
                        for right in record.free_spans
                        if max(left.start, right.start) < min(left.end, right.end)
                    )
                record = replace(record, free_spans=tuple(sorted(local)))
            records.append(record)
            offset += len(record.payload)
    return replace(episode, records=tuple(records))
