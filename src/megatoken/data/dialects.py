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
import keyword
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
    "function_call": "assistant",
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
        self.unfinished = False

    def issue(self, turn: int, count: int) -> list[str]:
        ids = [f"call-{turn}-{index}" for index in range(count)]
        self.pending.extend(ids)
        return ids

    def answer(self, count: int, *, in_parts: bool = False) -> list[str]:
        """Pair ``count`` results with calls. Only a source that sends one tool message per
        result may answer fewer calls than are pending (``in_parts``); the rest must follow."""
        if in_parts and 0 < count < len(self.pending):
            self.unfinished = True
            return [self.pending.popleft() for _ in range(count)]
        self.unfinished = False
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


_XML_CALL = re.compile(r"\A<function=([^>\s]+)>(.*)</function>\Z", re.DOTALL)
_XML_PARAMETER = re.compile(r"<parameter=([^>\s]+)>\n?(.*?)\n?</parameter>", re.DOTALL)


def _xml_value(text: str) -> Any:
    """A parameter written as XML text: containers, numbers and booleans keep their type."""
    stripped = text.strip()
    if stripped[:1] in "[{" or stripped in {"true", "false", "null"}:
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return text
    try:
        number = float(stripped)
    except ValueError:
        return text
    return int(number) if stripped.lstrip("-").isdigit() else number


def _call_body(piece: str) -> tuple[Any, Any]:
    """Name and arguments of one ``<tool_call>`` body: JSON, or Qwen3-Coder's
    ``<function=name><parameter=key>value</parameter></function>``."""
    xml = _XML_CALL.match(piece.strip())
    if xml is not None:
        return xml.group(1), {
            key: _xml_value(value) for key, value in _XML_PARAMETER.findall(xml.group(2))
        }
    item = _literal(piece, "A <tool_call> body")
    if not isinstance(item, dict):
        raise SchemaError("A <tool_call> body must be an object")
    return item.get("name"), item.get("arguments", {})


def _packed_results(body: str, count: int) -> list[str] | None:
    """One response holding a JSON list of ``{name, arguments, results}``, one entry per pending call."""
    try:
        entries = json.loads(body)
    except json.JSONDecodeError:
        return None
    if (
        isinstance(entries, list)
        and len(entries) == count
        and all(isinstance(entry, Mapping) and "results" in entry for entry in entries)
    ):
        return [json.dumps(entry["results"], ensure_ascii=False) for entry in entries]
    return None


def hermes(row: Mapping[str, Any], value: Any) -> tuple[Messages, Any]:
    """``<tool_call>{json}</tool_call>`` (or the XML ``<function=...>`` form) in assistant text,
    ``<tool_response>`` results."""
    ledger, out = _Ledger(), []
    messages = _messages(value)
    system = row.get("system_prompt", row.get("system"))
    if isinstance(system, str) and system.strip() and role_of(messages[0]) != "system":
        out.append({"role": "system", "content": system})  # a prompt kept outside the turns
    for turn, message in enumerate(messages):
        role, text = role_of(message), _text(message)
        # Qwen-style templates deliver results as a user turn made of response blocks only.
        blocks_only = (
            role == "user"
            and bool(ledger.pending)
            and text.lstrip().startswith("<tool_response>")
            and not _TAGGED["response"].sub("", text).strip()
        )
        is_result = blocks_only or (
            role == "tool" and (bool(ledger.pending) or "<tool_response>" in text)
        )
        if ledger.unfinished and ledger.pending and not is_result:
            raise SchemaError(
                f"The dialogue moves on with {len(ledger.pending)} call(s) only partly answered"
            )
        if role == "assistant" and "<tool_call>" in text:
            # With one capture group the split alternates prose, call, prose, call, ...
            pieces = _TAGGED["call"].split(text)
            ids = iter(ledger.issue(turn, len(pieces) // 2))
            blocks: list[dict[str, Any]] = []
            for index, piece in enumerate(pieces):
                if index % 2:
                    name, arguments = _call_body(piece)
                    blocks.append(_use(next(ids), name, arguments))
                else:
                    blocks += _blocks(piece, [])
            out.append({**message, "role": "assistant", "content": blocks})
        elif is_result:
            # A tool message without response tags is one result in the publisher's own words.
            bodies = _TAGGED["response"].findall(text) if "<tool_response>" in text else [text]
            if len(bodies) == 1 and len(ledger.pending) > 1:
                bodies = _packed_results(bodies[0], len(ledger.pending)) or bodies
            ids = ledger.answer(len(bodies), in_parts=True)
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


_LOOKS_LIKE_CALLS = re.compile(r"^\[\s*[\w .\-/:]+\(")
# Tool names with other characters (``Q&A Notifications``): a call only if every call parses.
_RELAXED_CALLS = re.compile(r"^\[\s*[^\[\]=()]+\(")


_JSON_NAMES = {"true": True, "false": False, "null": None}
_KEYWORD_PREFIX = "kw__"
_KEYWORD_ARG = re.compile(r"(?<![\w\"'])(" + "|".join(keyword.kwlist) + r")\s*=(?!=)")


class _JsonNames(ast.NodeTransformer):
    """Some sets write JSON's ``true``/``false``/``null`` inside Python call syntax, and values
    such as ``datetime(2023, 5, 12)``, which stay in the argument as their source text."""

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id in _JSON_NAMES:
            return ast.copy_location(ast.Constant(_JSON_NAMES[node.id]), node)
        return node

    def visit_Call(self, node: ast.Call) -> ast.AST:
        return ast.copy_location(ast.Constant(ast.unparse(node)), node)

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        if node.value is Ellipsis or isinstance(node.value, (bytes, complex)):
            return ast.copy_location(ast.Constant(ast.unparse(node)), node)  # not JSON values
        return node


def _python_call(segment: str) -> tuple[str, dict[str, Any]]:
    """``name(a=1, b="x")``; a tool name may itself contain parentheses (``Feed (Video) V2(a=1)``),
    so each ``(`` is tried as the start of the arguments, leftmost first."""
    text = segment.strip()
    failed = None
    if text.endswith(")"):
        for start, char in enumerate(text):
            name = text[:start].strip()
            if char != "(" or not name or any(c in name for c in "=[]"):
                continue
            try:
                source = text[start + 1 : -1]
                try:
                    tree = ast.parse(f"f({source})", mode="eval").body
                except SyntaxError:
                    # Parameters named after Python keywords (``from=``, ``in=``) are legal here.
                    renamed = _KEYWORD_ARG.sub(lambda m: f"{_KEYWORD_PREFIX}{m.group(1)}=", source)
                    tree = ast.parse(f"f({renamed})", mode="eval").body
                if tree.args or any(k.arg is None for k in tree.keywords):
                    raise ValueError("arguments must all be keyword=value")
                return name, {
                    k.arg.removeprefix(_KEYWORD_PREFIX): ast.literal_eval(
                        _JsonNames().visit(k.value)
                    )
                    for k in tree.keywords
                }
            except (SyntaxError, ValueError, MemoryError, RecursionError):
                failed = failed or name
    if failed is None:
        raise SchemaError("A call must look like name(argument=value, ...)")
    raise SchemaError(f"Call arguments are not literals: {failed}(...)")


def _bracketed_calls(text: str) -> list[tuple[str, dict[str, Any]]] | None:
    """``[f(a=1), g(b=2)]`` as the whole text; None when the text is ordinary prose."""
    stripped = text.strip()
    if not (stripped.endswith("]") and stripped.startswith("[")):
        return None
    strict = _LOOKS_LIKE_CALLS.match(stripped) is not None
    if not strict and _RELAXED_CALLS.match(stripped) is None:
        return None
    try:
        return [_python_call(segment) for segment in _split_top(stripped[1:-1], ",")]
    except SchemaError:
        if strict:
            raise
        return None  # bracketed prose that merely looks like a call


def _line_calls(text: str) -> list[tuple[str, dict[str, Any]]]:
    return [_python_call(segment) for segment in _split_top(text, "\n")]


_CODE_FENCE = re.compile(r"\A```(?:python|py)?[ \t]*\n(.*?)\n?```\Z", re.DOTALL)


def _fenced_calls(text: str) -> list[tuple[str, dict[str, Any]]] | None:
    """A reply that is one code block of calls (``f(a=1)`` per line, or a bracketed list)."""
    fenced = _CODE_FENCE.match(text.strip())
    if fenced is None:
        return None
    body = fenced.group(1).strip()
    try:
        return _bracketed_calls(body) if body.startswith("[") else _line_calls(body)
    except SchemaError:
        return None  # ordinary code, not a call


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
    """Calls as Python syntax: ``[f(a=1), g(b=2)]`` as the text (or one fenced code block of
    calls), or one call per line in a ``function_calls`` field. A ``functions`` field on the system message lists the tools.

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
            if isinstance(field, str) and field.strip() not in {"", "null", "None"}:
                calls = _line_calls(field)
            elif (calls := _bracketed_calls(text) or _fenced_calls(text)) is not None:
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


def _tagged_calls(text: str) -> tuple[str, list[Any]]:
    """The prose and the call objects of a message written ``<think>..</think><tool_call>{..}</tool_call>``;
    one block may also hold a JSON list of calls."""
    pieces = _TAGGED["call"].split(text)
    calls: list[Any] = []
    for piece in pieces[1::2]:
        body = _literal(piece, "A <tool_call> body")
        calls += body if isinstance(body, list) else [body]
    return "".join(pieces[0::2]).strip(), calls


def role_calls(row: Mapping[str, Any], value: Any) -> tuple[Messages, Any]:
    """One message per call and per result: ``tool_call``/``function_call`` and ``tool_response``.

    A call's content is ``{"name", "arguments"}`` (an object, a list of them, or JSON text), or
    LlamaFactory's ``<think>..</think><tool_call>{json}</tool_call>``, whose prose is kept. Parallel
    calls are consecutive messages (or one message) answered, in order, by the results that follow;
    a result may be wrapped in ``<tool_response>`` tags.
    """
    ledger, out = _Ledger(), []
    system = row.get("system")
    if isinstance(system, str) and system:
        out.append({"role": "system", "content": system})
    for turn, message in enumerate(_messages(value)):
        raw = message.get("role", message.get("from"))
        if isinstance(raw, str):
            raw = raw.replace(" ", "_")  # "tool call" is how one publisher spells tool_call
        if raw in {"tool_call", "function_call"}:
            body, prose = message.get("content", message.get("value")), ""
            if isinstance(body, str) and "<tool_call>" in body:
                prose, body = _tagged_calls(body)
            elif isinstance(body, str):
                body = _literal(body, "A tool call body")
            bodies = (
                body if isinstance(body, list) and body else [body]
            )  # several calls in one message
            if any(not isinstance(item, Mapping) for item in bodies):
                raise SchemaError("A tool call body must be an object")
            ids = ledger.issue(turn, len(bodies))
            uses = [
                _use(call_id, item.get("name"), item.get("arguments", {}))
                for call_id, item in zip(ids, bodies, strict=True)
            ]
            out.append({**message, "role": "assistant", "content": _blocks(prose, uses)})
        elif raw in {"tool_response", "observation", "tool"} and ledger.pending:
            text = _text_or_json(message)
            results = _TAGGED["response"].findall(text) if "<tool_response>" in text else [text]
            if len(results) > len(ledger.pending):
                raise SchemaError(
                    f"{len(results)} tool results for {len(ledger.pending)} pending call(s)"
                )
            blocks = [_result(ledger.pending.popleft(), result) for result in results]
            out.append({**message, "role": "tool", "content": blocks})
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
    """Exchanges as ``{"input": user, "output": assistant}`` (Capybara) or ``{"human", "assistant"}`` (MOSS)."""
    if not isinstance(value, list) or not value:
        raise SchemaError("turn pairs must be a nonempty list")
    out: Messages = []
    for pair in value:
        if not isinstance(pair, Mapping):
            raise SchemaError("Every exchange must be an object")
        user, reply = (
            (pair.get("input"), pair.get("output"))
            if "input" in pair
            else (pair.get("human"), pair.get("assistant"))
        )
        if not isinstance(user, str) or not isinstance(reply, str):
            raise SchemaError("Every exchange needs string input and output")
        out += [{"role": "user", "content": user}, {"role": "assistant", "content": reply}]
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
    ``reasoning_content``), ``teacher_response`` or ``output`` (strings), or an ``answer``
    string with its ``reasoning`` text beside it.
    """
    prompt = _messages(value)
    reply, reasoning = None, None
    for key in ("generated_message", "teacher_response", "answer", "output"):
        if row.get(key) is not None:
            reply = row[key]
            reasoning = row.get("reasoning") if key == "answer" else None
            break
    if isinstance(reply, str):
        reply = {"role": "assistant", "content": reply, "reasoning_content": reasoning}
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


def _harmony_text(message: Mapping[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "".join(block.get("text") or "" for block in content if isinstance(block, Mapping))


def _harmony_tools(messages: list[Mapping[str, Any]]) -> list[dict[str, Any]] | None:
    """Function schemas declared by the system message's tool namespaces, named ``namespace.tool``."""
    tools = []
    for message in messages:
        if message.get("role") != "system" or not isinstance(message.get("content"), list):
            continue
        for block in message["content"]:
            namespaces = block.get("tools") if isinstance(block, Mapping) else None
            for space, spec in (namespaces or {}).items():
                for tool in (spec or {}).get("tools") or ():
                    tools.append(
                        {
                            "name": f"{space}.{tool['name']}",
                            "description": tool.get("description") or "",
                            "parameters": tool.get("parameters") or {"type": "object"},
                        }
                    )
    return tools or None


def harmony(row: Mapping[str, Any], value: Any) -> tuple[Messages, Any]:
    """OpenAI harmony conversations (gpt-oss traces): messages with ``channel``, ``recipient`` and
    ``content`` as a list of text blocks.

    An assistant message addressed to a tool (``recipient`` ``browser.search``) is a call whose
    text is the JSON arguments; the next ``tool`` message is its result. Analysis and final
    channels are ordinary assistant text; a system message that only configures channels and
    tool namespaces carries no text and is dropped (its tools are returned).
    """
    messages = _messages(value)
    ledger, out = _Ledger(), []
    for turn, message in enumerate(messages):
        role, text = role_of(message), _harmony_text(message)
        recipient = message.get("recipient")
        if (
            role == "assistant"
            and isinstance(recipient, str)
            and recipient not in {"", "all", "assistant"}
        ):
            try:
                arguments = json.loads(text)
            except json.JSONDecodeError:
                arguments = None
            if not isinstance(arguments, dict):
                arguments = {"input": text}
            (call_id,) = ledger.issue(turn, 1)
            out.append({"role": "assistant", "content": [_use(call_id, recipient, arguments)]})
        elif role == "tool":
            if not ledger.pending:
                raise SchemaError("A tool message answers no pending call")
            # Parallel calls come one message each; their results follow in the same order.
            out.append({"role": "tool", "content": [_result(ledger.pending.popleft(), text)]})
        elif text.strip():
            out.append({"role": role, "content": text})
    return out, _harmony_tools(messages)


def tool_dict(row: Mapping[str, Any], value: Any) -> tuple[Messages, Any]:
    """Browsing-agent logs: ``{"role": "assistant", "content", "tool_call": {"tool", "args"}}`` and
    ``{"role": "tool", "tool", "content"}``, with the question in a ``question`` column.

    Each tool message answers the oldest call without a result; the last assistant message,
    which has no ``tool_call``, is the answer.
    """
    ledger, out = _Ledger(), []
    question = row.get("question", row.get("prompt"))
    if isinstance(question, str) and question.strip():
        out.append({"role": "user", "content": question})
    for turn, message in enumerate(_messages(value)):
        role, text = role_of(message), _text_or_json(message)
        call = message.get("tool_call")
        if role == "assistant" and isinstance(call, Mapping):
            (call_id,) = ledger.issue(turn, 1)
            args = call.get("args") if call.get("args") is not None else {}
            out.append(
                {
                    "role": "assistant",
                    "content": _blocks(text, [_use(call_id, call.get("tool"), args)]),
                }
            )
        elif role == "tool":
            if not ledger.pending:
                raise SchemaError("A tool message answers no pending call")
            out.append({"role": "tool", "content": [_result(ledger.pending.popleft(), text)]})
        else:
            out.append({"role": role, "content": text})
    return out, row.get("tools")


_CALL_LIST = re.compile(r"<call>\s*(.*?)\s*</call>", re.DOTALL)
_TOOL_LIST = re.compile(r"<tool>\s*(.*?)\s*</tool>", re.DOTALL)


def _call_list_tools(system: str) -> list[dict[str, Any]] | None:
    """Tools listed in the system prompt as ``<tool>[{name, description, parameters, required}]``
    where ``parameters`` maps each argument straight to its schema."""
    found = _TOOL_LIST.search(system)
    if found is None:
        return None
    try:
        listed = json.loads(found.group(1))
    except json.JSONDecodeError:
        return None
    tools = []
    for tool in listed if isinstance(listed, list) else ():
        if not isinstance(tool, Mapping) or not isinstance(tool.get("name"), str):
            continue
        parameters = tool.get("parameters")
        if isinstance(parameters, Mapping) and parameters.get("type") != "object":
            parameters = {
                "type": "object",
                "properties": parameters,
                "required": tool.get("required") or [],
            }
        tools.append(
            {
                "name": tool["name"],
                "description": tool.get("description") or "",
                "parameters": parameters or {"type": "object"},
            }
        )
    return tools or None


def call_list(row: Mapping[str, Any], value: Any) -> tuple[Messages, Any]:
    """BUTTON-style dialogues: ``<call>[{"name", "arguments"}, ...]</call>`` closes an assistant turn
    and one tool message holds a JSON list with a ``results`` entry per call, in call order;
    ``<final>`` marks the answer. Tools are listed in the system prompt between ``<tool>`` tags."""
    ledger, out, tools = _Ledger(), [], None
    for turn, message in enumerate(_messages(value)):
        role, text = role_of(message), _text(message)
        if role == "system" and tools is None:
            tools = _call_list_tools(text)
        if role == "assistant" and "<call>" in text:
            pieces = _CALL_LIST.split(text)
            blocks = _blocks(pieces[0], [])
            for body in pieces[1::2]:
                calls = _literal(body, "A <call> body")
                if not isinstance(calls, list) or not calls:
                    raise SchemaError("A <call> body must be a nonempty list")
                ids = ledger.issue(turn, len(calls))
                for call_id, call in zip(ids, calls, strict=True):
                    if not isinstance(call, Mapping):
                        raise SchemaError("Every call in a <call> list must be an object")
                    blocks.append(_use(call_id, call.get("name"), call.get("arguments", {})))
            blocks += _blocks("".join(pieces[2::2]), [])
            out.append({"role": "assistant", "content": blocks})
        elif role == "tool":
            bodies = _packed_results(text, len(ledger.pending)) or [text]
            ids = ledger.answer(len(bodies))
            out.append(
                {
                    "role": "tool",
                    "content": [_result(i, body) for i, body in zip(ids, bodies, strict=True)],
                }
            )
        else:
            out.append({"role": role, "content": text})
    return out, tools


def device_calls(row: Mapping[str, Any], value: Any) -> tuple[Messages, Any]:
    """tau2 customer-service logs: ``tool_calls`` of ``{id, name, arguments, requestor}`` and tool
    messages without ids, answered in call order.

    The agent's calls are calls. The simulated customer also operates a phone with tools of its
    own (``requestor`` ``user``); that is part of the customer's side of the conversation, so the
    customer's action and what it returned are spelled as customer text. Training rows that store
    the next step apart from the dialogue so far keep it in ``answer``; it closes the dialogue, with
    its ``thinking`` spelled as ``<think>``, and is the only turn trained.
    """
    pending: deque[tuple[str, str, Mapping[str, Any]]] = deque()
    out: Messages = []
    messages = list(_messages(value))
    last = isinstance(row.get("answer"), Mapping)
    if last:
        messages.append(row["answer"])
    first_new = 0
    for turn, message in enumerate(messages):
        if last and turn == len(messages) - 1:
            first_new = len(out)
        role, text = role_of(message), _text(message)
        thinking = message.get("thinking")
        if role == "assistant" and isinstance(thinking, str) and thinking.strip():
            text = f"<think>\n{thinking.strip()}\n</think>\n\n{text}"
        calls = message.get("tool_calls")
        if isinstance(calls, str):  # a JSON string, empty when the message has no calls
            calls = _literal(calls, "tool_calls") if calls.strip() else None
        if calls and not isinstance(calls, list):
            raise SchemaError("tool_calls must be a list")
        if role in {"assistant", "user"} and calls:
            blocks = []
            for index, raw in enumerate(calls):
                if not isinstance(raw, Mapping):
                    raise SchemaError("Every tool call must be an object")
                call = raw["function"] if isinstance(raw.get("function"), Mapping) else raw
                call_id = str(raw.get("id") or f"call-{turn}-{index}")
                pending.append((role, call_id, call))
                if role == "assistant":
                    blocks.append(_use(call_id, call.get("name"), call.get("arguments", {})))
            if role == "assistant":
                out.append({"role": "assistant", "content": _blocks(text, blocks)})
            else:
                out.append(
                    {"role": "user", "content": text if text.strip() else "(acts on the device)"}
                )
        elif role == "tool":
            if not pending:
                raise SchemaError("A tool message answers no pending call")
            owner, call_id, call = pending.popleft()
            if owner == "assistant":
                out.append({"role": "tool", "content": [_result(call_id, text)]})
            else:
                arguments = json.dumps(call.get("arguments") or {})
                out.append(
                    {
                        "role": "user",
                        "content": f"[Device action {call.get('name')}({arguments}) returned]\n{text}",
                    }
                )
        else:
            out.append({"role": role, "content": text})
    for message in out[:first_new]:
        # Every prefix of a dialogue is a row of its own; each row trains only its new step.
        message["train"] = False
    return out, row.get("tools")


def environment_roles(row: Mapping[str, Any], value: Any) -> tuple[Messages, Any]:
    """ReAct logs whose environment replies carry a ``function`` or ``tool`` role (ToolLLaMA):
    the reply is spelled as the environment's turn, which the ReAct reader links to the action."""
    out: Messages = []
    for message in _messages(value):
        role = role_of(message)
        out.append(
            {
                **message,
                "role": "user" if role == "tool" else role,
                "content": _text_or_json(message),
            }
        )
    return out, row.get("tools")


def _response_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "".join(
        block if isinstance(block, str) else str(block.get("text") or "")
        for block in content
        if isinstance(block, (str, Mapping))
    )


def responses(row: Mapping[str, Any], value: Any) -> tuple[Messages, Any]:
    """OpenAI Responses-API items: ``{role, content}`` messages, ``reasoning`` summaries,
    ``function_call`` (``call_id``, ``name``, JSON ``arguments``) and ``function_call_output``.

    Reasoning summaries, assistant text and calls that follow one another are one assistant turn
    (the summary is spelled as ``<think>``). The row's ``expected_action`` is the next step of the
    trajectory (RL data stores one row per step) and becomes the last assistant turn; its tools
    come from ``responses_create_params``.
    """
    params = row.get("responses_create_params")
    params = params if isinstance(params, Mapping) else {}
    items = list(_messages(value))
    expected = row.get("expected_action")
    if isinstance(expected, Mapping):
        items.append(expected)
    out: Messages = []
    thoughts: list[str] = []
    text: list[str] = []
    calls: list[dict[str, Any]] = []

    def flush() -> None:
        body = "\n\n".join(
            part for part in (*(f"<think>\n{t}\n</think>" for t in thoughts), *text) if part
        )
        if body.strip() or calls:
            out.append({"role": "assistant", "content": _blocks(body, calls)})
        thoughts.clear(), text.clear(), calls.clear()

    for item in items:
        kind = item.get("type")
        if kind == "reasoning":
            summary = "\n".join(
                str(part.get("text") or "")
                for part in item.get("summary") or ()
                if isinstance(part, Mapping)
            ).strip()
            if summary:
                thoughts.append(summary)
        elif kind == "function_call":
            calls.append(
                _use(
                    str(item.get("call_id") or item.get("id")),
                    item.get("name"),
                    item.get("arguments"),
                )
            )
        elif kind == "function_call_output":
            flush()
            output = item.get("output")
            if not isinstance(output, str):
                output = json.dumps(output)
            out.append({"role": "tool", "content": [_result(str(item.get("call_id")), output)]})
        else:
            # The step to come is a bare ``{"type": "message"}``: the assistant speaks.
            role = role_of(item) if "role" in item or "from" in item else "assistant"
            if role == "assistant":
                if calls:  # text written after a call starts a new turn
                    flush()
                text.append(_response_text(item.get("content")))
            else:
                flush()
                out.append({"role": role, "content": _response_text(item.get("content"))})
    flush()
    return [m for m in out if m["content"]], params.get("tools")


DIALECTS: dict[str, Dialect] = {
    "responses": responses,
    "environment_roles": environment_roles,
    "call_list": call_list,
    "device_calls": device_calls,
    "tool_dict": tool_dict,
    "harmony": harmony,
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
