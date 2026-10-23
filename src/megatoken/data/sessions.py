"""Coding-agent session logs, written as the content-block dialogues the chat reader takes.

Agent harnesses keep their own logs: Claude Code writes one JSON event per line, pi and the
Codex CLI write their own event streams, OpenCode exports a session document. Each converter
here takes the events (or document) of one session and returns ``(messages, None)`` in the
shape used by ``dialects.py``: an assistant call is a ``tool_use`` block, a result is a
``tool_result`` block in the next user message, and reasoning is written inline as
``<think>`` text, as every other source spells it. Nothing is executed or guessed:

- noise events (titles, queue operations, progress, snapshots, token counts) are dropped;
- a call repeated by a resumed session is kept once;
- a result whose call is not in the log (the log was compacted) stays as plain user text,
  because a result without its call cannot be linked.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from typing import Any

from .adapters import SchemaError

Messages = list[dict[str, Any]]
Dialect = Callable[[Mapping[str, Any], Any], tuple[Messages, Any]]

EMPTY_RESULT = "(empty result)"


def _list(value: Any, what: str) -> list[Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise SchemaError(f"{what} is not JSON") from exc
    if isinstance(value, Mapping):
        for key in ("entries", "events", "messages", "history", "trajectory"):
            if isinstance(value.get(key), list):
                value = value[key]
                break
    if not isinstance(value, list):
        raise SchemaError(f"{what} must be a list of events")
    return value


def _flatten(content: Any) -> str:
    """A tool result's content as text: a string, or the text blocks of a list."""
    if isinstance(content, str):
        return content or EMPTY_RESULT
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, Mapping):
                text = block.get("text")
                parts.append(text if isinstance(text, str) else f"[{block.get('type', 'block')}]")
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts) or EMPTY_RESULT
    if content is None:
        return EMPTY_RESULT
    return json.dumps(content, ensure_ascii=False)


def _thinking(text: Any) -> list[dict[str, Any]]:
    if isinstance(text, str) and text.strip():
        return [{"type": "text", "text": f"<think>\n{text.strip()}\n</think>"}]
    return []


class _Dialogue:
    """Messages being built; keeps every call issued and the ones already answered."""

    def __init__(self) -> None:
        self.messages: Messages = []
        self.calls: set[str] = set()
        self.answered: set[str] = set()

    def assistant(self, blocks: list[dict[str, Any]], *, merge: bool = False) -> None:
        kept = []
        for block in blocks:
            if block["type"] == "tool_use":
                if block["id"] in self.calls:
                    continue  # replayed by a resumed session
                self.calls.add(block["id"])
            kept.append(block)
        if not kept:
            return
        if merge and self.messages and self.messages[-1]["role"] == "assistant":
            self.messages[-1]["content"].extend(kept)
        else:
            self.messages.append({"role": "assistant", "content": kept})

    def user(self, text: str | None = None, results: list[tuple[str, str]] = ()) -> None:
        blocks: list[dict[str, Any]] = []
        for call_id, content in results:
            if call_id in self.answered:
                continue
            if call_id in self.calls:
                self.answered.add(call_id)
                blocks.append({"type": "tool_result", "tool_use_id": call_id, "content": content})
            else:
                # The call left the log (compaction): the output survives as text.
                blocks.append({"type": "text", "text": content})
        if text is not None and text.strip():
            blocks.append({"type": "text", "text": text})
        if blocks:
            self.messages.append({"role": "user", "content": blocks})

    def system(self, text: str) -> None:
        if text.strip():
            self.messages.append({"role": "system", "content": text})

    def finish(self) -> tuple[Messages, None]:
        if not self.messages:
            raise SchemaError("The session holds no conversation events")
        return self.messages, None


def _use(call_id: Any, name: Any, arguments: Any) -> list[dict[str, Any]]:
    """The call as a one-block list, or nothing for a partial call that cannot be linked."""
    if not isinstance(call_id, str) or not call_id:
        return []  # an in-progress call that never got an id has no result to pair with
    if not isinstance(name, str) or not name:
        name = "tool"
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            parsed = None
        arguments = parsed if isinstance(parsed, dict) else {"input": arguments}
    if not isinstance(arguments, Mapping):
        arguments = {"input": arguments}
    return [{"type": "tool_use", "id": call_id, "name": name, "input": dict(arguments)}]


# ---- Claude Code -----------------------------------------------------------------------------


def claude_code(row: Mapping[str, Any], value: Any) -> tuple[Messages, None]:
    """Claude Code log events (one JSON object per line, or a stored list of them).

    One model reply is spread over several lines that share ``message.id`` (one per block);
    they are one assistant message here. Subagent branches (``isSidechain``) are dropped, so
    the dialogue is the main conversation.
    """
    dialogue = _Dialogue()
    previous: str | None = None
    for event in _list(value, "Claude Code events"):
        if not isinstance(event, Mapping) or event.get("isSidechain"):
            continue
        kind, message = event.get("type"), event.get("message")
        if kind not in {"user", "assistant"} or not isinstance(message, Mapping):
            continue
        content = message.get("content")
        if kind == "assistant":
            blocks: list[dict[str, Any]] = []
            for block in (
                content if isinstance(content, list) else [{"type": "text", "text": content}]
            ):
                if not isinstance(block, Mapping):
                    continue
                if block.get("type") == "thinking":
                    blocks += _thinking(block.get("thinking"))
                elif block.get("type") == "text" and isinstance(block.get("text"), str):
                    if block["text"].strip():
                        blocks.append({"type": "text", "text": block["text"]})
                elif block.get("type") == "tool_use":
                    blocks += _use(block.get("id"), block.get("name"), block.get("input", {}))
            identifier = message.get("id")
            dialogue.assistant(blocks, merge=identifier is not None and identifier == previous)
            previous = identifier
            continue
        previous = None
        if isinstance(content, str):
            dialogue.user(content)
            continue
        results, texts = [], []
        for block in content if isinstance(content, list) else []:
            if not isinstance(block, Mapping):
                continue
            if block.get("type") == "tool_result" and isinstance(block.get("tool_use_id"), str):
                results.append((block["tool_use_id"], _flatten(block.get("content"))))
            elif block.get("type") == "text" and isinstance(block.get("text"), str):
                texts.append(block["text"])
        dialogue.user("\n".join(texts) or None, results)
    return dialogue.finish()


# ---- pi ---------------------------------------------------------------------------------------


def pi_session(row: Mapping[str, Any], value: Any) -> tuple[Messages, None]:
    """pi coding-agent events: ``message`` events with user, assistant and toolResult roles."""
    dialogue = _Dialogue()
    for event in _list(value, "pi events"):
        message = event.get("message") if isinstance(event, Mapping) else None
        if not isinstance(message, Mapping) or event.get("type") != "message":
            continue
        role, content = message.get("role"), message.get("content")
        blocks = content if isinstance(content, list) else [{"type": "text", "text": content}]
        text = "\n".join(
            b["text"]
            for b in blocks
            if isinstance(b, Mapping) and b.get("type") == "text" and isinstance(b.get("text"), str)
        )
        if role == "developer" or role == "system":
            dialogue.system(text)
        elif role == "user":
            dialogue.user(text)
        elif role == "toolResult":
            call_id = message.get("toolCallId")
            if isinstance(call_id, str):
                dialogue.user(None, [(call_id, _flatten(blocks))])
        elif role == "assistant":
            out: list[dict[str, Any]] = []
            for block in blocks:
                if not isinstance(block, Mapping):
                    continue
                if block.get("type") == "thinking":
                    out += _thinking(block.get("thinking"))
                elif block.get("type") == "text" and isinstance(block.get("text"), str):
                    if block["text"].strip():
                        out.append({"type": "text", "text": block["text"]})
                elif block.get("type") == "toolCall":
                    out += _use(block.get("id"), block.get("name"), block.get("arguments", {}))
            dialogue.assistant(out)
    return dialogue.finish()


# ---- Codex CLI --------------------------------------------------------------------------------


def codex_rollout(row: Mapping[str, Any], value: Any) -> tuple[Messages, None]:
    """Codex CLI rollout events: ``response_item`` payloads in order; ``event_msg`` repeats them.

    Function calls carry JSON text arguments; ``apply_patch`` is a custom tool whose input is
    the raw patch, kept whole as ``{"input": patch}``.
    """
    dialogue = _Dialogue()
    pending: list[dict[str, Any]] = []  # the assistant message being collected
    results: list[tuple[str, str]] = []

    def flush() -> None:
        nonlocal pending
        if pending:
            dialogue.assistant(pending)
            pending = []

    def flush_results() -> None:
        nonlocal results
        if results:
            dialogue.user(None, results)
            results = []

    for event in _list(value, "Codex events"):
        if not isinstance(event, Mapping):
            continue  # a stray non-object line is not an event
        payload = event.get("payload")
        if event.get("type") != "response_item" or not isinstance(payload, Mapping):
            continue
        kind = payload.get("type")
        if kind == "message":
            texts = [
                b["text"]
                for b in (payload.get("content") or [])
                if isinstance(b, Mapping) and isinstance(b.get("text"), str)
            ]
            text = "\n".join(texts)
            role = payload.get("role")
            if role == "assistant":
                flush_results()
                if text.strip():
                    pending.append({"type": "text", "text": text})
                continue
            flush()
            flush_results()
            if role == "developer":
                dialogue.system(text)
            elif role == "user":
                dialogue.user(text)
        elif kind == "reasoning":
            flush_results()
            summary = " ".join(
                b["text"]
                for b in (payload.get("summary") or [])
                if isinstance(b, Mapping) and isinstance(b.get("text"), str)
            )
            pending += _thinking(summary)
        elif kind in {"function_call", "custom_tool_call"}:
            flush_results()
            arguments = (
                payload.get("arguments") if kind == "function_call" else payload.get("input")
            )
            if kind == "custom_tool_call" or not isinstance(arguments, (str, Mapping)):
                arguments = {"input": arguments}
            pending += _use(payload.get("call_id"), payload.get("name"), arguments)
        elif kind in {"function_call_output", "custom_tool_call_output"}:
            flush()
            call_id = payload.get("call_id")
            if isinstance(call_id, str):
                output = payload.get("output")
                if isinstance(output, Mapping):
                    output = output.get("content", output)
                results.append((call_id, _flatten(output)))
    flush()
    flush_results()
    return dialogue.finish()


# ---- OpenCode ---------------------------------------------------------------------------------


def opencode_session(row: Mapping[str, Any], value: Any) -> tuple[Messages, None]:
    """An OpenCode session export: tool parts hold both a call and its result."""
    if isinstance(value, str):
        value = json.loads(value)
    messages = value.get("messages") if isinstance(value, Mapping) else None
    if not isinstance(messages, list):
        raise SchemaError("An OpenCode session needs a messages list")
    dialogue = _Dialogue()
    for message in messages:
        info = message.get("info", {}) if isinstance(message, Mapping) else {}
        parts = message.get("parts", []) if isinstance(message, Mapping) else []
        if info.get("role") == "user":
            dialogue.user(
                "\n".join(
                    p["text"]
                    for p in parts
                    if isinstance(p, Mapping)
                    and p.get("type") == "text"
                    and isinstance(p.get("text"), str)
                )
            )
            continue
        blocks: list[dict[str, Any]] = []
        results: list[tuple[str, str]] = []

        def flush() -> None:
            nonlocal blocks, results
            dialogue.assistant(blocks)
            dialogue.user(None, results)
            blocks, results = [], []

        for part in parts:
            if not isinstance(part, Mapping):
                continue
            kind = part.get("type")
            if kind == "reasoning":
                if results:
                    flush()
                blocks += _thinking(part.get("text"))
            elif kind == "text" and isinstance(part.get("text"), str) and part["text"].strip():
                if results:
                    flush()
                blocks.append({"type": "text", "text": part["text"]})
            elif kind == "tool" and isinstance(part.get("callID"), str):
                state = part.get("state") if isinstance(part.get("state"), Mapping) else {}
                blocks += _use(part["callID"], part.get("tool"), state.get("input", {}))
                output = state.get("output", state.get("error"))
                if output is not None:
                    results.append((part["callID"], _flatten(output)))
        flush()
    return dialogue.finish()


# ---- other published trace layouts -----------------------------------------------------------


def longest_root(row: Mapping[str, Any], value: Any) -> tuple[Messages, Any]:
    """Rollouts stored as several conversation roots: the one with the most messages.

    A harness may log an auxiliary conversation (a title request) or growing prefixes of the
    main one; the longest root is the whole agent run. Messages stay in the OpenAI shape.
    """
    roots = _list(value, "conversation roots")
    best = max(
        (
            r["messages"]
            for r in roots
            if isinstance(r, Mapping) and isinstance(r.get("messages"), list)
        ),
        key=len,
        default=None,
    )
    if not best:
        raise SchemaError("No conversation root holds messages")
    return best, None


def responses_items(row: Mapping[str, Any], value: Any) -> tuple[Messages, None]:
    """OpenAI Responses-API items (mini-swe-agent 1.1 logs): messages, response objects, outputs."""
    dialogue = _Dialogue()
    for item in _list(value, "Responses items"):
        if not isinstance(item, Mapping):
            continue
        if item.get("type") == "function_call_output" and isinstance(item.get("call_id"), str):
            dialogue.user(None, [(item["call_id"], _flatten(item.get("output")))])
        elif item.get("object") == "response":
            blocks: list[dict[str, Any]] = []
            for part in item.get("output") or []:
                if not isinstance(part, Mapping):
                    continue
                if part.get("type") == "reasoning":
                    blocks += _thinking(
                        " ".join(
                            b["text"]
                            for b in part.get("summary") or []
                            if isinstance(b, Mapping) and isinstance(b.get("text"), str)
                        )
                    )
                elif part.get("type") == "message":
                    for b in part.get("content") or []:
                        if (
                            isinstance(b, Mapping)
                            and isinstance(b.get("text"), str)
                            and b["text"].strip()
                        ):
                            blocks.append({"type": "text", "text": b["text"]})
                elif part.get("type") == "function_call":
                    blocks += _use(part.get("call_id"), part.get("name"), part.get("arguments", {}))
            dialogue.assistant(blocks)
        elif item.get("role") in {"system", "developer"} and isinstance(item.get("content"), str):
            dialogue.system(item["content"])
        elif item.get("role") == "user" and isinstance(item.get("content"), str):
            dialogue.user(item["content"])
    return dialogue.finish()


def step_records(row: Mapping[str, Any], value: Any) -> tuple[Messages, None]:
    """Runs stored as steps: ``{action: {content, tool_calls, reasoning_text}, observations}``.

    The row's ``initial_messages`` (system and task) come first; each step's observations
    answer its tool calls in order.
    """
    dialogue = _Dialogue()
    initial = row.get("initial_messages")
    if isinstance(initial, str):
        initial = json.loads(initial)
    for message in initial or []:
        if isinstance(message, Mapping) and isinstance(message.get("content"), str):
            if message.get("role") in {"system", "developer"}:
                dialogue.system(message["content"])
            elif message.get("role") == "user":
                dialogue.user(message["content"])
    for step in _list(value, "steps"):
        action = step.get("action") if isinstance(step, Mapping) else None
        if not isinstance(action, Mapping):
            continue
        blocks = _thinking(action.get("reasoning_text"))
        if isinstance(action.get("content"), str) and action["content"].strip():
            blocks.append({"type": "text", "text": action["content"]})
        ids = []
        for call in action.get("tool_calls") or []:
            function = call.get("function", {}) if isinstance(call, Mapping) else {}
            used = _use(
                call.get("id") if isinstance(call, Mapping) else None,
                function.get("name"),
                function.get("arguments", {}),
            )
            blocks += used
            ids += [b["id"] for b in used]
        dialogue.assistant(blocks)
        observations = step.get("observations") or []
        dialogue.user(None, [(i, _flatten(o)) for i, o in zip(ids, observations, strict=False)])
    return dialogue.finish()


def normalized_events(row: Mapping[str, Any], value: Any) -> tuple[Messages, None]:
    """Normalised agent events: prompt, assistant_message, tool_call, tool_result, final_message."""
    dialogue = _Dialogue()
    prompts = row.get("prompts")
    if isinstance(prompts, str):
        prompts = json.loads(prompts)
    for prompt in prompts or []:
        if isinstance(prompt, Mapping) and isinstance(prompt.get("content"), str):
            if prompt.get("role") == "system":
                dialogue.system(prompt["content"])
    for event in _list(value, "events"):
        if not isinstance(event, Mapping):
            continue
        kind = event.get("type")
        text = event.get("content") if isinstance(event.get("content"), str) else None
        if kind == "prompt":
            dialogue.user(text)
        elif kind in {"assistant_message", "final_message"} and text and text.strip():
            dialogue.assistant([{"type": "text", "text": text}], merge=True)
        elif kind == "tool_call":
            dialogue.assistant(
                _use(event.get("tool_call_id"), event.get("tool"), event.get("input", {})),
                merge=True,
            )
        elif kind == "tool_result" and isinstance(event.get("tool_call_id"), str):
            dialogue.user(None, [(event["tool_call_id"], _flatten(event.get("content")))])
    return dialogue.finish()


def parts_calls(row: Mapping[str, Any], value: Any) -> tuple[Messages, None]:
    """Assistant turns whose call sits in ``parts`` (a ``python`` part holding JSON), the next
    user message being its result: the layout of the nanoswe rollouts.

    ``content`` is empty; ``parts`` is ``[{type: text, text}, {type: python, text:
    '{"name": ..., "arguments": {...}}'}]``. The first user message is the task.
    """
    messages = _list(value, "messages")
    dialogue = _Dialogue()
    last_call: str | None = None
    for turn, message in enumerate(messages):
        if not isinstance(message, Mapping):
            continue
        content = message.get("content") if isinstance(message.get("content"), str) else ""
        if message.get("role") == "system":
            dialogue.system(content)
        elif message.get("role") == "user":
            if last_call is not None:
                dialogue.user(None, [(last_call, content or EMPTY_RESULT)])
                last_call = None
            else:
                dialogue.user(content)
        elif message.get("role") == "assistant":
            blocks = [{"type": "text", "text": content}] if content.strip() else []
            for part in message.get("parts") or []:
                if not isinstance(part, Mapping) or not isinstance(part.get("text"), str):
                    continue
                if part.get("type") == "text":
                    if part["text"].strip():
                        blocks.append({"type": "text", "text": part["text"]})
                elif part.get("type") == "python":
                    try:
                        call = json.loads(part["text"])
                    except json.JSONDecodeError:
                        continue
                    if isinstance(call, Mapping) and isinstance(call.get("name"), str):
                        used = _use(f"call-{turn}", call["name"], call.get("arguments", {}))
                        blocks += used
                        last_call = used[0]["id"] if used else last_call
            dialogue.assistant(blocks)
    return dialogue.finish()


_OBSERVATION = re.compile(r"<observation>(.*?)</observation>", re.DOTALL)


def observation_tags(row: Mapping[str, Any], value: Any) -> tuple[Messages, None]:
    """One output string with ``<observation>`` blocks: replies alternate with environment turns.

    The row's ``instruction`` (and ``input``) is the task; the text between observations is the
    agent's reply, each observation the interpreter's answer to the code before it.
    """
    if not isinstance(value, str) or not value.strip():
        raise SchemaError("The trajectory text is empty")
    task = "\n\n".join(
        part
        for part in (row.get("instruction"), row.get("input"))
        if isinstance(part, str) and part.strip()
    )
    if not task:
        raise SchemaError("The row has no task")
    messages: Messages = [{"role": "user", "content": task}]
    cursor = 0
    for match in _OBSERVATION.finditer(value):
        reply = value[cursor : match.start()].strip()
        if reply:
            messages.append({"role": "assistant", "content": reply})
        messages.append({"role": "user", "content": match.group(1).strip() or EMPTY_RESULT})
        cursor = match.end()
    tail = value[cursor:].strip()
    if tail:
        messages.append({"role": "assistant", "content": tail})
    return messages, None


SESSION_DIALECTS: dict[str, Dialect] = {
    "claude_code": claude_code,
    "pi_session": pi_session,
    "codex_rollout": codex_rollout,
    "opencode_session": opencode_session,
    "longest_root": longest_root,
    "responses_items": responses_items,
    "step_records": step_records,
    "normalized_events": normalized_events,
    "observation_tags": observation_tags,
    "parts_calls": parts_calls,
}
