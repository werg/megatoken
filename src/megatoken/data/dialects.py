"""Text dialects for tool dialogues, normalized to ordered content blocks.

Published tool-use sets spell calls and results in many ways: Hermes tags, Python
call syntax, role names, flat strings. Every converter here takes a source row and
returns ``(messages, tools)`` in one shape: an assistant call is a ``tool_use``
block and a result is a ``tool_result`` block naming the call it answers, so the one
chat parser builds actions and results the same way for all of them. There is one
output message per input message, so a source's original turn masks stay aligned.

Calls and results are paired strictly in order within one dialogue. A count that
does not match, or text that looks like a call but does not parse, is a
``SchemaError``; nothing is guessed. The importer never executes any call.
"""

from __future__ import annotations

import ast
import json
import re
from collections import deque
from collections.abc import Callable, Mapping
from typing import Any

from .adapters import SchemaError

Messages = list[dict[str, Any]]
Dialect = Callable[[Mapping[str, Any], Any], tuple[Messages, Any]]

ROLES = {
    "human": "user",
    "gpt": "assistant",
    "ai": "assistant",
    "model": "assistant",
    "function": "tool",
    "tool_response": "tool",
    "observation": "tool",
    "ipython": "tool",
    "function_response": "tool",
    "developer": "system",
}


def role_of(message: Mapping[str, Any]) -> str:
    role = message.get("role", message.get("from"))
    if not isinstance(role, str):
        raise SchemaError("Every message needs a string role")
    # Publishers capitalise role names ("User", "Assistant"); the meaning is the same.
    lowered = role.casefold()
    return ROLES.get(lowered, lowered)


def _text(message: Mapping[str, Any]) -> str:
    content = message.get("content", message.get("value", message.get("text")))
    if content is None:
        return ""
    if not isinstance(content, str):
        raise SchemaError("A dialect message needs string content")
    return content


def _literal(text: str, what: str) -> Any:
    """JSON, or the Python-literal spelling some sets use for the same object."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError, MemoryError, RecursionError) as exc:
        raise SchemaError(f"{what} is neither JSON nor a Python literal") from exc


def _use(call_id: str, name: Any, arguments: Any) -> dict[str, Any]:
    if isinstance(arguments, str):
        arguments = _literal(arguments, "Tool arguments")
    if not isinstance(name, str) or not name or not isinstance(arguments, dict):
        raise SchemaError("A tool call needs a name and a complete argument object")
    return {"type": "tool_use", "id": call_id, "name": name, "input": arguments}


def _result(call_id: str, content: str) -> dict[str, Any]:
    return {"type": "tool_result", "tool_use_id": call_id, "content": content}


def _blocks(text: str, calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return ([{"type": "text", "text": text}] if text.strip() else []) + calls


class _Ledger:
    """Call ids issued so far and not yet answered, oldest first."""

    def __init__(self) -> None:
        self.pending: deque[str] = deque()

    def issue(self, turn: int, count: int) -> list[str]:
        ids = [f"call-{turn}-{index}" for index in range(count)]
        self.pending.extend(ids)
        return ids

    def answer(self, count: int) -> list[str]:
        if count != len(self.pending):
            raise SchemaError(
                f"{count} tool result(s) for {len(self.pending)} pending call(s); results are "
                "paired with calls only when their counts match"
            )
        ids = list(self.pending)
        self.pending.clear()
        return ids


def _messages(value: Any) -> list[Mapping[str, Any]]:
    if isinstance(value, str):
        value = _literal(value, "messages")
    if not isinstance(value, list) or not value or any(not isinstance(m, Mapping) for m in value):
        raise SchemaError("messages must be a nonempty ordered list of objects")
    return value


_TAGGED = {
    "call": re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL),
    "response": re.compile(r"<tool_response>\s*(.*?)\s*</tool_response>", re.DOTALL),
}


def hermes(row: Mapping[str, Any], value: Any) -> tuple[Messages, Any]:
    """``<tool_call>{json}</tool_call>`` in assistant text, ``<tool_response>`` results."""
    ledger, out = _Ledger(), []
    for turn, message in enumerate(_messages(value)):
        role, text = role_of(message), _text(message)
        if role == "assistant" and "<tool_call>" in text:
            # With one capture group the split alternates prose, call, prose, call, ...
            pieces = _TAGGED["call"].split(text)
            ids = iter(ledger.issue(turn, len(pieces) // 2))
            blocks: list[dict[str, Any]] = []
            for index, piece in enumerate(pieces):
                if index % 2:
                    item = _literal(piece, "A <tool_call> body")
                    if not isinstance(item, dict):
                        raise SchemaError("A <tool_call> body must be an object")
                    blocks.append(_use(next(ids), item.get("name"), item.get("arguments", {})))
                else:
                    blocks += _blocks(piece, [])
            out.append({**message, "role": "assistant", "content": blocks})
        elif role == "tool" and "<tool_response>" in text:
            bodies = _TAGGED["response"].findall(text)
            ids = ledger.answer(len(bodies))
            results = [_result(i, body) for i, body in zip(ids, bodies, strict=True)]
            out.append({**message, "role": "tool", "content": results})
        else:
            out.append({**message, "role": role, "content": text})
    return out, row.get("tools")


def _split_top(text: str, separators: str) -> list[str]:
    """Split at separators outside quotes and brackets."""
    parts, depth, quote, start, index = [], 0, "", 0, 0
    while index < len(text):
        char = text[index]
        if quote:
            if char == "\\":
                index += 1
            elif char == quote:
                quote = ""
        elif char in "\"'":
            quote = char
        elif char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif depth == 0 and char in separators:
            parts.append(text[start:index])
            start = index + 1
        index += 1
    parts.append(text[start:])
    return [part.strip() for part in parts if part.strip()]


_CALL = re.compile(r"^([^()=\[\]]+?)\s*\((.*)\)$", re.DOTALL)
_LOOKS_LIKE_CALLS = re.compile(r"^\[\s*[\w .\-/:]+\(")


def _python_call(segment: str) -> tuple[str, dict[str, Any]]:
    match = _CALL.match(segment.strip())
    if match is None:
        raise SchemaError("A call must look like name(argument=value, ...)")
    name, source = match.group(1).strip(), match.group(2)
    try:
        tree = ast.parse(f"f({source})", mode="eval").body
        if not isinstance(tree, ast.Call) or tree.args or any(k.arg is None for k in tree.keywords):
            raise SchemaError("Call arguments must all be keyword=value")
        return name, {k.arg: ast.literal_eval(k.value) for k in tree.keywords}
    except (SyntaxError, ValueError, MemoryError, RecursionError) as exc:
        raise SchemaError(f"Call arguments are not literals: {name}(...)") from exc


def _bracketed_calls(text: str) -> list[tuple[str, dict[str, Any]]] | None:
    """``[f(a=1), g(b=2)]`` as the whole text; None when the text is ordinary prose."""
    stripped = text.strip()
    if not (stripped.endswith("]") and _LOOKS_LIKE_CALLS.match(stripped)):
        return None
    return [_python_call(segment) for segment in _split_top(stripped[1:-1], ",")]


def _line_calls(text: str) -> list[tuple[str, dict[str, Any]]]:
    return [_python_call(segment) for segment in _split_top(text, "\n")]


def _answers(content: Any, count: int) -> list[str]:
    """One result per pending call: a JSON list, one JSON value per line, or a single value."""
    text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
    if count == 1:
        return [text]
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = None
    if isinstance(value, list) and len(value) == count:
        return [json.dumps(item, ensure_ascii=False) for item in value]
    lines = _split_top(text, "\n")
    if len(lines) == count:
        return lines
    raise SchemaError(f"Cannot split a tool result into {count} results")


def python_calls(row: Mapping[str, Any], value: Any) -> tuple[Messages, Any]:
    """Calls as Python syntax: ``[f(a=1), g(b=2)]`` as the text, or one call per line in a
    ``function_calls`` field. A ``functions`` field on the system message lists the tools.

    One result message answers all of the preceding assistant turn's calls.
    """
    ledger, out, tools = _Ledger(), [], None
    system = row.get("system")
    if isinstance(system, str) and system:
        out.append({"role": "system", "content": system})
    for turn, message in enumerate(_messages(value)):
        role, text = role_of(message), _text(message)
        if role == "system" and message.get("functions"):
            tools = message["functions"]
        calls, prose = None, text
        if role == "assistant":
            field = message.get("function_calls")
            if isinstance(field, str) and field.strip():
                calls = _line_calls(field)
            elif (calls := _bracketed_calls(text)) is not None:
                prose = ""  # the content is the call syntax itself
        if calls:
            ids = ledger.issue(turn, len(calls))
            uses = [_use(i, name, args) for i, (name, args) in zip(ids, calls, strict=True)]
            out.append({**message, "role": "assistant", "content": _blocks(prose, uses)})
        elif role in {"tool", "environment"} and ledger.pending:
            answers = _answers(message.get("content", message.get("value")), len(ledger.pending))
            ids = ledger.answer(len(answers))
            results = [_result(i, answer) for i, answer in zip(ids, answers, strict=True)]
            out.append({**message, "role": "tool", "content": results})
        else:
            out.append({**message, "role": role, "content": text})
    return out, tools


def role_calls(row: Mapping[str, Any], value: Any) -> tuple[Messages, Any]:
    """One message per call and per result: ``tool_call``/``function_call`` and ``tool_response``.

    A call's content is ``{"name", "arguments"}`` (an object or its JSON text). Parallel
    calls are consecutive messages answered, in order, by the results that follow.
    """
    ledger, out = _Ledger(), []
    system = row.get("system")
    if isinstance(system, str) and system:
        out.append({"role": "system", "content": system})
    for turn, message in enumerate(_messages(value)):
        raw = message.get("role", message.get("from"))
        if raw in {"tool_call", "function_call"}:
            body = message.get("content", message.get("value"))
            if isinstance(body, str):
                body = _literal(body, "A tool call body")
            if not isinstance(body, Mapping):
                raise SchemaError("A tool call body must be an object")
            (call_id,) = ledger.issue(turn, 1)
            out.append(
                {
                    **message,
                    "role": "assistant",
                    "content": [_use(call_id, body.get("name"), body.get("arguments", {}))],
                }
            )
        elif raw in {"tool_response", "observation", "tool"} and ledger.pending:
            call_id = ledger.pending.popleft()
            out.append(
                {**message, "role": "tool", "content": [_result(call_id, _text_or_json(message))]}
            )
        else:
            out.append({**message, "role": role_of(message), "content": _text_or_json(message)})
    return out, row.get("tools")


def _text_or_json(message: Mapping[str, Any]) -> str:
    content = message.get("content", message.get("value"))
    if content is None:
        return ""
    return content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)


_GLAIVE_TURN = re.compile(r"(?m)^(USER|ASSISTANT|FUNCTION RESPONSE):[ \t]?")
_GLAIVE_CALL = re.compile(r"^<functioncall>\s*(\{.*\})\s*(?:<\|endoftext\|>)?$", re.DOTALL)
_GLAIVE_ARGS = re.compile(r"\"arguments\":\s*'(.*)'\s*\}$", re.DOTALL)


def glaive(row: Mapping[str, Any], value: Any) -> tuple[Messages, Any]:
    """``SYSTEM:``/``USER:``/``ASSISTANT:``/``FUNCTION RESPONSE:`` dumps (Glaive).

    ``system`` lists the functions as JSON objects. A call is ``<functioncall> {"name": ...,
    "arguments": '<json>'}``; its response is the next ``FUNCTION RESPONSE`` turn.
    """
    system, chat = row.get("system"), value
    if system is None and isinstance(chat, str):
        # The first release keeps system prompt and chat in one string.
        system, marker, rest = chat.partition("\nUSER:")
        chat = "USER:" + rest if marker else chat
    if not isinstance(chat, str) or not isinstance(system, str):
        raise SchemaError("Glaive rows need string system and chat fields")
    decoder, tools, position = json.JSONDecoder(), [], system.find("{")
    while position != -1:
        try:
            item, end = decoder.raw_decode(system, position)
        except json.JSONDecodeError:
            position = system.find("{", position + 1)
            continue
        if isinstance(item, dict) and isinstance(item.get("name"), str):
            tools.append(item)
        position = system.find("{", end)
    out: Messages = [{"role": "system", "content": system.removeprefix("SYSTEM:").strip()}]
    ledger, pieces = _Ledger(), _GLAIVE_TURN.split(chat)
    if pieces[0].strip():
        raise SchemaError("Glaive chat has text before its first turn marker")
    for turn, (marker, body) in enumerate(zip(pieces[1::2], pieces[2::2], strict=True), start=1):
        body = body.replace("<|endoftext|>", "").strip()
        if marker == "USER":
            out.append({"role": "user", "content": body})
        elif marker == "FUNCTION RESPONSE":
            (call_id,) = ledger.answer(1)
            out.append({"role": "tool", "content": [_result(call_id, body)]})
        elif body.startswith("<functioncall>"):
            match = _GLAIVE_CALL.match(body)
            if match is None:
                raise SchemaError("Unparseable <functioncall>")
            text = match.group(1)
            try:
                item = json.loads(text)
            except json.JSONDecodeError:
                quoted = _GLAIVE_ARGS.search(text)
                name = re.search(r"\"name\":\s*\"([^\"]+)\"", text)
                if quoted is None or name is None:
                    raise SchemaError("Unparseable <functioncall> arguments") from None
                item = {"name": name.group(1), "arguments": quoted.group(1)}
            (call_id,) = ledger.issue(turn, 1)
            out.append(
                {
                    "role": "assistant",
                    "content": [_use(call_id, item.get("name"), item.get("arguments", {}))],
                }
            )
        else:
            out.append({"role": "assistant", "content": body})
    return out, tools


_XLAM_TYPES = {
    "str": "string",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
    "list": "array",
    "dict": "object",
    "tuple": "array",
    "set": "array",
}


def _xlam_tool(tool: Mapping[str, Any]) -> dict[str, Any]:
    """xLAM lists parameters as ``{name: {type: "str", description, default}}``; make it JSON Schema.

    Re-releases wrap a tool as ``{"type": "function", "function": {...}}`` with JSON Schema
    parameters already; those pass through.
    """
    if isinstance(tool.get("function"), Mapping):
        return {**tool["function"]}
    properties, required = {}, []
    for name, spec in (tool.get("parameters") or {}).items():
        kind = str(spec.get("type", "")).split(",")[0].split("[")[0].strip().lower()
        properties[name] = {key: spec[key] for key in ("description",) if key in spec}
        if kind in _XLAM_TYPES:
            properties[name]["type"] = _XLAM_TYPES[kind]
        if "default" not in spec and "optional" not in str(spec.get("type", "")).lower():
            required.append(name)
    return {
        "name": tool["name"],
        "description": tool.get("description", ""),
        "parameters": {"type": "object", "properties": properties, "required": required},
    }


def xlam(row: Mapping[str, Any], value: Any) -> tuple[Messages, Any]:
    """A query and the (parallel) calls that answer it, with no executed result (xLAM)."""
    query, answers, tools = row.get("query"), row.get("answers"), row.get("tools")
    if not isinstance(query, str):
        raise SchemaError("xLAM rows need a string query")
    tools = _literal(tools, "tools") if isinstance(tools, str) else tools
    if isinstance(answers, str):
        try:
            answers = _literal(answers, "answers")
        except SchemaError:
            # A query the tools cannot answer is declined in plain words.
            if not isinstance(tools, list) or not answers.strip():
                raise
            return (
                [{"role": "user", "content": query}, {"role": "assistant", "content": answers}],
                [_xlam_tool(tool) for tool in tools],
            )
    if not isinstance(answers, list) or not answers or not isinstance(tools, list):
        raise SchemaError("xLAM rows need answer and tool lists")
    ledger = _Ledger()
    ids = ledger.issue(0, len(answers))
    uses = [
        _use(
            i,
            a.get("name") if isinstance(a, dict) else None,
            a.get("arguments", {}) if isinstance(a, dict) else None,
        )
        for i, a in zip(ids, answers, strict=True)
    ]
    return (
        [{"role": "user", "content": query}, {"role": "assistant", "content": uses}],
        [_xlam_tool(tool) for tool in tools],
    )


def turn_pairs(row: Mapping[str, Any], value: Any) -> tuple[Messages, Any]:
    """``[{"input": user, "output": assistant}, ...]`` exchanges (Capybara)."""
    if not isinstance(value, list) or not value:
        raise SchemaError("turn pairs must be a nonempty list")
    out: Messages = []
    for pair in value:
        if not isinstance(pair, Mapping) or not all(
            isinstance(pair.get(k), str) for k in ("input", "output")
        ):
            raise SchemaError("Every exchange needs string input and output")
        out += [
            {"role": "user", "content": pair["input"]},
            {"role": "assistant", "content": pair["output"]},
        ]
    return out, None


_HUMAN_ASSISTANT = re.compile(r"(?:^|\n\n)(Human|Assistant): ?")


def human_assistant(row: Mapping[str, Any], value: Any) -> tuple[Messages, Any]:
    """``\\n\\nHuman: ...\\n\\nAssistant: ...`` text dumps (Anthropic HH-RLHF)."""
    if not isinstance(value, str):
        raise SchemaError("A Human/Assistant dump must be a string")
    pieces = _HUMAN_ASSISTANT.split(value)
    if pieces[0].strip() or len(pieces) < 3:
        raise SchemaError("Text before the first Human/Assistant marker, or no turn at all")
    out: Messages = []
    for marker, body in zip(pieces[1::2], pieces[2::2], strict=True):
        if not body.strip():
            raise SchemaError("A Human/Assistant turn is empty")
        out.append({"role": "user" if marker == "Human" else "assistant", "content": body.strip()})
    return out, None


def alternating_turns(row: Mapping[str, Any], value: Any) -> tuple[Messages, Any]:
    """A list of bare strings, user first, alternating with the assistant (UltraChat)."""
    if not isinstance(value, list) or not value or not all(isinstance(t, str) for t in value):
        raise SchemaError("alternating turns must be a nonempty list of strings")
    return [
        {"role": "assistant" if index % 2 else "user", "content": text}
        for index, text in enumerate(value)
    ], None


def prompt_and_reply(row: Mapping[str, Any], value: Any) -> tuple[Messages, Any]:
    """The prompt turns in one field and the answer to them in another.

    The answer is ``generated_message`` (a message object that may carry
    ``reasoning_content``), ``teacher_response`` (a string), or an ``answer`` string with
    its ``reasoning`` beside it.
    """
    prompt = _messages(value)
    reply = row.get("generated_message", row.get("teacher_response", row.get("answer")))
    if isinstance(reply, str):
        reply = {
            "role": "assistant",
            "content": reply,
            "reasoning_content": row.get("reasoning"),
        }
    if not isinstance(reply, Mapping) or reply.get("tool_calls") or not reply.get("content"):
        raise SchemaError("The reply must be a message with text and no tool calls")
    return [
        *prompt,
        {
            "role": "assistant",
            "content": reply["content"],
            "reasoning_content": reply.get("reasoning_content"),
        },
    ], row.get("tools")


def prompt_completion(row: Mapping[str, Any], value: Any) -> tuple[Messages, Any]:
    """``prompt`` and ``completion`` message lists that together are one conversation."""
    completion = _messages(row.get("completion"))
    return [*_messages(value), *completion], row.get("tools")


DIALECTS: dict[str, Dialect] = {
    "hermes": hermes,
    "python_calls": python_calls,
    "role_calls": role_calls,
    "glaive": glaive,
    "xlam": xlam,
    "turn_pairs": turn_pairs,
    "human_assistant": human_assistant,
    "alternating_turns": alternating_turns,
    "prompt_and_reply": prompt_and_reply,
    "prompt_completion": prompt_completion,
}
