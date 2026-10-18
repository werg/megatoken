"""Source profiles over one strict causal conversation/document parser.

Adapters describe published schemas; they never assume a dataset's rows are
verified successes. Field maps can override dotted paths for pinned variants.
Malformed or unavailable fields are explicit rejects, not guessed targets.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, replace
from typing import TYPE_CHECKING, Any

from .chunking import ChunkingConfig
from .schema import (
    Codec,
    Dependency,
    Episode,
    Lineage,
    LogicalAction,
    PhysicalRecord,
    Supervision,
    apply_recovery_policy,
    canonical_json,
    digest_bytes,
    make_records,
    stable_id,
)

if TYPE_CHECKING:
    from .parsing.chat import ChatParse


class SchemaError(ValueError):
    """A source row does not satisfy its selected, versioned adapter contract."""


class ChatFramingError(SchemaError):
    """Captured framing evidence needs parser review before assigning losses."""

    def __init__(self, message: str, diagnostics: tuple[str, ...]):
        super().__init__(message)
        self.diagnostics = diagnostics


@dataclass(frozen=True, slots=True)
class AdapterProfile:
    kind: str = "chat"
    fields: tuple[tuple[str, tuple[str, ...]], ...] = ()
    harness: str = "openai"
    observation_provenance: str = "unknown"
    emission_provenance: str = "teacher"
    outcome_verification: str = "publisher_outcome"
    require_turn_mask: bool = False
    require_parent: bool = False
    trajectory: bool = False
    unknown_imitation: bool = False
    content_kind: str = "auto"


DEFAULT_FIELDS = {
    "messages": ("messages", "conversations"),
    "id": ("trajectory_id", "uuid", "id"),
    "text": ("text", "content", "code"),
    "language": ("language", "metadata.language", "metadata.lang"),
    "filename": ("filename", "path", "file_path", "metadata.filename", "metadata.path"),
    "mime_type": ("mime_type", "mime", "metadata.mime_type", "metadata.mime"),
    "locale": ("locale", "metadata.locale"),
    "content_kind": ("content_kind", "metadata.content_kind"),
    "chat_template": ("chat_template", "metadata.chat_template"),
    "family": (
        "family_id",
        "parent_id",
        "metadata.parent_id",
        "metadata.warc_id",
        "url",
        "metadata.url",
    ),
    "ancestors": ("ancestor_ids", "parent_ids", "metadata.ancestor_ids"),
    "repository": ("repo", "repository", "metadata.repo"),
    "task": ("instance_id", "task_id", "metadata.instance_id", "metadata.task_id"),
    "created_at": ("created_at", "timestamp", "metadata.created_at"),
    "tools": ("tools", "tool_schemas"),
    "mask": ("metadata.train_turns_mask", "train_turns_mask"),
    "recovery": ("recovery_turns", "metadata.recovery_turns"),
    "outcome": ("resolved", "success", "metadata.verify_status", "outcome"),
    "reward": ("reward",),
    "prompt": ("prompt", "instruction", "issue", "problem_statement"),
    "target": ("target", "completion", "response"),
    "before": ("before", "pre_change_state"),
    "patch": ("patch", "target_patch"),
    "benchmark": ("benchmark_tags", "metadata.benchmark_tags"),
}


def _fields(**values: str | tuple[str, ...]) -> tuple[tuple[str, tuple[str, ...]], ...]:
    return tuple(
        (key, (value,) if isinstance(value, str) else value) for key, value in values.items()
    )


PROFILES: dict[str, AdapterProfile] = {
    "native": AdapterProfile(kind="native"),
    "document": AdapterProfile(kind="document", emission_provenance="natural"),
    "chat": AdapterProfile(),
    "serialized_chat": AdapterProfile(kind="serialized_chat"),
    "artifact": AdapterProfile(kind="artifact"),
    "repository": AdapterProfile(
        kind="repository", emission_provenance="natural", require_parent=True
    ),
    "fim": AdapterProfile(kind="fim", emission_provenance="natural", require_parent=True),
    "nemotron_cc": AdapterProfile(
        kind="document", fields=_fields(text="text"), require_parent=True
    ),
    "nemotron_code": AdapterProfile(
        kind="document", fields=_fields(text="content"), require_parent=True, content_kind="code"
    ),
    "nemotron_stem": AdapterProfile(
        kind="document", fields=_fields(text="text"), require_parent=True
    ),
    "smoltalk2": AdapterProfile(fields=_fields(messages="messages")),
    "nemotron_chat_v4": AdapterProfile(require_turn_mask=True),
    "nemotron_agentic_v4": AdapterProfile(observation_provenance="simulated"),
    "open_swe": AdapterProfile(
        trajectory=True,
        observation_provenance="executed",
        outcome_verification="publisher_tests",
        require_parent=True,
    ),
    "orchard_swe": AdapterProfile(
        trajectory=True,
        observation_provenance="executed",
        outcome_verification="publisher_tests",
        require_parent=True,
        fields=_fields(outcome="metadata.verify_status"),
    ),
    "agent_trove": AdapterProfile(
        trajectory=True,
        observation_provenance="executed",
        harness="terminus2",
        require_parent=True,
        fields=_fields(outcome="reward"),
    ),
    "nemotron_terminal": AdapterProfile(
        trajectory=True, observation_provenance="executed", harness="terminus2", require_parent=True
    ),
    # An exit_status of Submitted is explicitly NOT test verification.
    "swe_zero": AdapterProfile(
        trajectory=True,
        observation_provenance="executed",
        harness="mini_swe",
        require_parent=True,
        fields=_fields(outcome="__no_verified_outcome__"),
    ),
    "tool_use_seeds": AdapterProfile(kind="seed"),
}


@dataclass(frozen=True, slots=True)
class AdapterContext:
    source_id: str
    revision: str
    uri: str
    stream: str
    row_index: int = 0
    max_record_bytes: int = 16384
    field_map: Mapping[str, str | list[str]] = field(default_factory=dict)
    options: Mapping[str, Any] = field(default_factory=dict)
    codec: Codec | None = None
    max_record_tokens: int | None = None
    chunking: ChunkingConfig | None = None


def _object(value: Any, label: str) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError as exc:
            raise SchemaError(f"{label} must contain valid JSON") from exc
    return value


def _path(row: Mapping[str, Any], path: str) -> Any:
    current: Any = row
    for key in path.split("."):
        if isinstance(current, str):
            try:
                current = json.loads(current)
            except json.JSONDecodeError:
                return None
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


class _Reader:
    def __init__(self, row: Mapping[str, Any], profile: AdapterProfile, context: AdapterContext):
        self.row, self.profile, self.context = row, profile, context
        self.fields = {**DEFAULT_FIELDS, **dict(profile.fields)}
        for key, value in context.field_map.items():
            self.fields[key] = (value,) if isinstance(value, str) else tuple(value)

    def get(self, key: str, default: Any = None, *, required: bool = False) -> Any:
        for path in self.fields.get(key, (key,)):
            value = _path(self.row, path)
            if value is not None:
                return value
        if required:
            raise SchemaError(
                f"Missing {key!r}; configure a field_map from {self.fields.get(key, (key,))}"
            )
        return default

    def origin(self, key: str) -> str:
        return next(
            (path for path in self.fields.get(key, (key,)) if _path(self.row, path) is not None),
            key,
        )


def _bytes(value: Any, field_name: str) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, str):
        return value.encode("utf-8")
    raise SchemaError(
        f"{field_name} must be a string or explicitly decoded bytes, got {type(value).__name__}"
    )


def _outcome(value: Any) -> str:
    if value is True or (isinstance(value, (int, float)) and value == 1):
        return "success"
    if value is False or (isinstance(value, (int, float)) and value == 0):
        return "failure"
    normalized = str(value).casefold()
    if normalized in {"resolved", "success", "passed", "pass"}:
        return "success"
    if normalized in {"unresolved", "failure", "failed", "fail"}:
        return "failure"
    if normalized in {"incomplete", "timeout", "interrupted"}:
        return "incomplete"
    return "unknown"


def _supervision(reader: _Reader) -> Supervision:
    outcome = _outcome(reader.get("outcome"))
    verification = reader.context.options.get(
        "outcome_verification", reader.profile.outcome_verification
    )
    benchmark = reader.get("benchmark", ())
    if isinstance(benchmark, str):
        benchmark = (benchmark,)
    annotations = {
        key: reader.row[key]
        for key in ("metadata", "exit_status", "duration_sec", "license", "used_in")
        if key in reader.row
    }
    return Supervision(
        outcome=outcome,
        verification=verification if outcome != "unknown" else "none",
        reward=reader.get("reward"),
        recovery_turns=tuple(reader.get("recovery", ())),
        benchmark_tags=tuple(benchmark),
        annotations_json=canonical_json(annotations),
    )


def _lineage(reader: _Reader, raw_sha: str, prompt_override: bytes | None = None) -> Lineage:
    repository = reader.get("repository")
    task = reader.get("task")
    family = reader.get("family")
    ancestors = reader.get("ancestors", ())
    if isinstance(ancestors, str):
        ancestors = (ancestors,)
    if not family:
        if repository:
            family = f"repository:{str(repository).casefold()}"
        elif task:
            family = f"task:{task}"
        elif reader.context.options.get("require_parent", reader.profile.require_parent):
            raise SchemaError(
                "Missing original parent/task lineage; supply field_map.family or a joined parent manifest"
            )
        else:
            # Prompt hashing groups repeated completions of the same prompt.
            prompt = prompt_override if prompt_override is not None else reader.get("prompt")
            messages = reader.get("messages")
            if prompt is None and isinstance(messages, list):
                prompt = next(
                    (
                        m.get("content", m.get("value", ""))
                        for m in messages
                        if m.get("role", m.get("from")) in {"user", "human"}
                    ),
                    None,
                )
            family = "content:" + digest_bytes(
                _bytes(prompt, "prompt") if isinstance(prompt, (bytes, str)) else raw_sha.encode()
            )
    return Lineage(
        source_id=reader.context.source_id,
        revision=reader.context.revision,
        family_id=str(family),
        ancestor_ids=tuple(str(value) for value in ancestors),
        repository=repository,
        task_id=task,
        source_uri=reader.context.uri,
        source_row=reader.context.row_index,
        created_at=reader.get("created_at"),
        raw_sha256=raw_sha,
    )


def _source_chat(reader: _Reader, payload: bytes) -> ChatParse:
    """Apply source framing once, before any role-owned logical fields exist."""
    from .parsing.chat import parse_chat

    policy = reader.context.chunking or ChunkingConfig()
    template = reader.context.options.get("chat_template", reader.get("chat_template"))
    if template is None:
        template = policy.chat_template
    if not isinstance(template, str):
        raise SchemaError("Parser hint 'chat_template' must be a string or null")
    content_kind = reader.context.options.get(
        "content_kind", reader.get("content_kind", reader.profile.content_kind)
    )
    if reader.profile.kind == "document" and content_kind == "binary":
        template = "none"
    return parse_chat(payload, template, max_frames=policy.limits.max_regions)


def source_row_lineage(row: Mapping[str, Any], adapter: str, context: AdapterContext) -> Lineage:
    """Resolve declared ancestry even when message framing needs later repair."""
    if adapter not in PROFILES or PROFILES[adapter].kind == "native":
        raise SchemaError("Unparsed source lineage requires a non-native adapter")
    reader = _Reader(row, PROFILES[adapter], context)
    prompt = None
    if reader.profile.kind in {"serialized_chat", "document"}:
        payload = reader.get("text")
        if isinstance(payload, (str, bytes)):
            payload = _bytes(payload, "text")
            parsed = _source_chat(reader, payload)
            prompt = next(
                (
                    payload[frame.header_end : frame.body_end]
                    for frame in parsed.frames
                    if frame.role == "user"
                ),
                None,
            )
    return _lineage(reader, digest_bytes(canonical_json(row).encode("utf-8")), prompt)


def source_row_supervision(
    row: Mapping[str, Any], adapter: str, context: AdapterContext
) -> Supervision:
    """Keep benchmark/recovery annotations available for quarantined source rows."""
    if adapter not in PROFILES or PROFILES[adapter].kind == "native":
        raise SchemaError("Unparsed source supervision requires a non-native adapter")
    return _supervision(_Reader(row, PROFILES[adapter], context))


def _tools(value: Any) -> str:
    schemas = _object(value, "tools") if value is not None else []
    if not isinstance(schemas, list):
        raise SchemaError("tools must be a list (possibly JSON encoded)")
    schemas = [_object(item, "tool schema") for item in schemas]
    if any(not isinstance(item, dict) for item in schemas):
        raise SchemaError("Every tool schema must be an object")
    return canonical_json(schemas)


class _EpisodeBuilder:
    def __init__(self, episode_id: str, reader: _Reader):
        self.episode_id, self.reader = episode_id, reader
        self.tools_json = _tools(reader.get("tools"))
        self.records: list[PhysicalRecord] = []
        self.actions: list[LogicalAction] = []
        self.pending: dict[str, LogicalAction] = {}
        self.completed: set[str] = set()
        self.parser_annotations: dict[str, Any] = {}
        self.family_prompt: bytes | None = None

    def content_hints(self, *objects: Mapping[str, Any]) -> dict[str, Any]:
        """Specific block/message hints override source options and row defaults.

        Only an allowlist of parser hints leaves this method. Outcome labels and
        arbitrary metadata are never rendered or treated as content hints.
        """
        keys = ("content_kind", "language", "filename", "mime_type", "locale", "chat_template")
        hints = {key: self.reader.context.options.get(key, self.reader.get(key)) for key in keys}
        if hints["content_kind"] is None:
            hints["content_kind"] = (
                "code"
                if self.reader.profile.kind in {"repository", "fim"}
                else self.reader.profile.content_kind
            )
        for obj in objects:
            metadata = obj.get("metadata", {})
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except json.JSONDecodeError:
                    metadata = {}
            for source in (metadata, obj):
                if isinstance(source, Mapping):
                    for key in keys:
                        if source.get(key) is not None:
                            hints[key] = source[key]
        for key, value in hints.items():
            if value is not None and not isinstance(value, str):
                raise SchemaError(f"Parser hint {key!r} must be a string or null")
        # Source adapters frame any serialized conversation before append.
        # Every appended field already owns its role and canonical token stream.
        hints["chat_template"] = "none"
        return hints

    def append(
        self,
        payload: bytes,
        *,
        role: str,
        turn: int,
        score: bool = False,
        source_field: str = "",
        transform: str = "identity",
        action_id: str | None = None,
        result: LogicalAction | None = None,
        provenance: str | None = None,
        parse_hints: Mapping[str, Any] | None = None,
        source_offset: int = 0,
        chunk_diagnostics: tuple[str, ...] = (),
    ) -> tuple[PhysicalRecord, ...]:
        from .masking import tool_free_spans

        logical_id = stable_id("logical-v1", self.episode_id, str(len(self.records)), source_field)
        if provenance is None:
            provenance = (
                self.reader.profile.emission_provenance
                if role in {"assistant", "document"}
                else self.reader.profile.observation_provenance
                if result
                else "natural"
            )
        records = make_records(
            payload,
            logical_id=logical_id,
            role=role,
            turn_index=turn,
            policy_mask=score,
            free_spans=tool_free_spans(
                payload, self.tools_json, max_tokens=self.reader.context.max_record_tokens or 4096
            )
            if action_id
            else None,
            provenance=provenance,
            max_bytes=self.reader.context.max_record_bytes,
            source_uri=self.reader.context.uri,
            source_field=source_field,
            transform=transform,
            action_id=action_id,
            result_of=result.action_id if result else None,
            execution_id=result.execution_id if result else None,
            codec=self.reader.context.codec,
            max_tokens=self.reader.context.max_record_tokens,
            chunking=self.reader.context.chunking,
            **self.content_hints(parse_hints or {}),
        )
        if source_offset:
            records = tuple(
                replace(
                    record,
                    source_span=replace(
                        record.source_span,
                        start=record.source_span.start + source_offset,
                        end=record.source_span.end + source_offset,
                    ),
                )
                for record in records
            )
        if chunk_diagnostics:
            records = tuple(
                replace(
                    record,
                    chunk_diagnostics=tuple(
                        sorted(set(record.chunk_diagnostics).union(chunk_diagnostics))
                    ),
                )
                for record in records
            )
        self.records.extend(records)
        return records

    def action(
        self,
        payload: bytes,
        *,
        call_id: str,
        tool_name: str,
        arguments: Any,
        turn: int,
        score: bool,
        source_field: str,
        transform: str = "identity",
        parse_hints: Mapping[str, Any] | None = None,
    ) -> None:
        if call_id in self.pending or call_id in self.completed:
            raise SchemaError(f"Tool call ID reused: {call_id}")
        raw_arguments = arguments if isinstance(arguments, str) else canonical_json(arguments)
        if not isinstance(_object(raw_arguments, "tool arguments"), dict):
            raise SchemaError("A complete tool argument object is required before execution")
        action_id = stable_id("action-v1", self.episode_id, call_id, payload)
        execution = stable_id("imported-execution-v1", self.episode_id, call_id, payload)
        records = self.append(
            payload,
            role="assistant",
            turn=turn,
            score=score,
            source_field=source_field,
            transform=transform,
            action_id=action_id,
            parse_hints=parse_hints,
        )
        action = LogicalAction(
            action_id=action_id,
            call_id=call_id,
            tool_name=tool_name,
            arguments_json=raw_arguments,
            record_ids=tuple(record.record_id for record in records),
            serialized_sha256=digest_bytes(payload),
            execution_id=execution,
            schema_id=self.reader.profile.harness,
        )
        self.actions.append(action)
        self.pending[call_id] = action

    def observation(
        self,
        payload: bytes,
        *,
        turn: int,
        call_id: str | None,
        source_field: str,
        parse_hints: Mapping[str, Any] | None = None,
    ) -> None:
        if call_id is None:
            if len(self.pending) != 1:
                raise SchemaError(
                    "Unlinked observation; call IDs may be inferred only for exactly one pending action"
                )
            call_id = next(iter(self.pending))
        if call_id not in self.pending:
            raise SchemaError(f"Tool result has no preceding unexecuted call: {call_id}")
        action = self.pending.pop(call_id)
        self.completed.add(call_id)
        self.append(
            payload,
            role="tool",
            turn=turn,
            result=action,
            source_field=source_field,
            parse_hints=parse_hints,
        )


def _harness_action(content: str, harness: str) -> tuple[str, dict[str, Any]] | None:
    if harness == "terminus2":
        stripped = content.strip()
        if stripped.startswith("```json") and stripped.endswith("```"):
            stripped = stripped[7:-3].strip()
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return None
        if (
            isinstance(parsed, dict)
            and isinstance(parsed.get("commands"), list)
            and parsed["commands"]
        ):
            return "terminal.batch", parsed
    if harness == "mini_swe":
        commands = re.findall(r"```(?:bash|sh)\s*\n(.*?)```", content, flags=re.DOTALL)
        if len(commands) == 1:
            return "bash", {"command": commands[0]}
    return None


def _chat(reader: _Reader, builder: _EpisodeBuilder) -> None:
    messages = _object(reader.get("messages", required=True), "messages")
    if not isinstance(messages, list) or not messages:
        raise SchemaError("messages must be a nonempty ordered list")
    turn_mask = reader.get("mask")
    if turn_mask is None and reader.profile.require_turn_mask:
        raise SchemaError(
            "This source requires metadata.train_turns_mask; do not train every assistant turn"
        )
    if turn_mask is not None:
        if (
            not isinstance(turn_mask, list)
            or len(turn_mask) != len(messages)
            or any(type(v) is not bool for v in turn_mask)
        ):
            raise SchemaError(
                "The training-turn mask must contain one boolean per original message"
            )
    for turn, message in enumerate(messages):
        if not isinstance(message, Mapping):
            raise SchemaError("Every message must be an object")
        role = message.get("role", message.get("from"))
        role = {"human": "user", "gpt": "assistant", "function": "tool"}.get(role, role)
        if role == "developer":
            builder.parser_annotations.setdefault("source_roles", {})[str(turn)] = role
            role = "system"
        hints = builder.content_hints(message)
        if role not in {"system", "user", "assistant", "tool", "environment"}:
            raise SchemaError(f"Unknown chat role {role!r}")
        own_mask = message.get("train", message.get("loss", True))
        if own_mask not in (True, False, 0, 1):
            raise SchemaError("Per-message train/loss field must be boolean")
        score = role == "assistant" and bool(own_mask) and (turn_mask is None or turn_mask[turn])
        base = f"{reader.origin('messages')}[{turn}]"
        content = message.get("content", message.get("value"))
        if reader.context.options.get("include_reasoning", False) and message.get(
            "reasoning_content"
        ):
            if role != "assistant":
                raise SchemaError("reasoning_content belongs to an assistant turn")
            builder.append(
                _bytes(message["reasoning_content"], "reasoning_content"),
                role=role,
                turn=turn,
                score=score,
                source_field=base + ".reasoning_content",
                parse_hints=hints,
            )
        if isinstance(content, list):
            # Preserve block order, including interleaved Anthropic tool blocks.
            for index, block in enumerate(content):
                if not isinstance(block, dict):
                    raise SchemaError("A content block must be an object")
                kind = block.get("type", "text")
                field_name = f"{base}.content[{index}]"
                block_hints = builder.content_hints(message, block)
                if kind in {"text", "input_text", "output_text"}:
                    builder.append(
                        _bytes(block.get("text", ""), field_name),
                        role=role,
                        turn=turn,
                        score=score,
                        source_field=field_name + ".text",
                        parse_hints=block_hints,
                    )
                elif kind == "tool_use" and role == "assistant":
                    builder.action(
                        canonical_json(block).encode(),
                        call_id=block["id"],
                        tool_name=block["name"],
                        arguments=block["input"],
                        turn=turn,
                        score=score,
                        source_field=field_name,
                        transform="canonical-tool-json-v1",
                        parse_hints=block_hints,
                    )
                elif kind == "tool_result":
                    builder.observation(
                        _bytes(block.get("content", ""), field_name),
                        turn=turn,
                        call_id=block.get("tool_use_id"),
                        source_field=field_name + ".content",
                        parse_hints=block_hints,
                    )
                else:
                    raise SchemaError(
                        f"Unsupported content block {kind!r}; this pipeline is text/bytes only"
                    )
        elif content is not None:
            payload = _bytes(content, base + ".content")
            harness_action = (
                _harness_action(content, reader.profile.harness)
                if isinstance(content, str) and role == "assistant"
                else None
            )
            if role in {"tool", "environment"}:
                builder.observation(
                    payload,
                    turn=turn,
                    call_id=message.get("tool_call_id"),
                    source_field=base + ".content",
                    parse_hints=hints,
                )
            elif (
                role == "user"
                and builder.pending
                and reader.profile.harness in {"terminus2", "mini_swe"}
            ):
                builder.observation(
                    payload,
                    turn=turn,
                    call_id=message.get("tool_call_id"),
                    source_field=base + ".content",
                    parse_hints=hints,
                )
            elif harness_action:
                tool, arguments = harness_action
                builder.action(
                    payload,
                    call_id=f"{builder.episode_id}:{turn}",
                    tool_name=tool,
                    arguments=arguments,
                    turn=turn,
                    score=score,
                    source_field=base + ".content",
                    parse_hints=hints,
                )
            elif payload or not message.get("tool_calls"):
                builder.append(
                    payload,
                    role=role,
                    turn=turn,
                    score=score,
                    source_field=base + ".content",
                    parse_hints=hints,
                )
        calls = message.get("tool_calls") or []
        if message.get("function_call"):
            if calls:
                raise SchemaError("Both function_call and tool_calls are present")
            calls = [{"id": f"{builder.episode_id}:{turn}", "function": message["function_call"]}]
        if calls and role != "assistant":
            raise SchemaError("Only assistant messages can issue tools")
        for index, call in enumerate(calls):
            function = call.get("function", call)
            call_id = call.get("id")
            if not call_id:
                raise SchemaError("Native tool calls require their original call ID")
            builder.action(
                canonical_json(call).encode(),
                call_id=call_id,
                tool_name=function["name"],
                arguments=function.get("arguments", {}),
                turn=turn,
                score=score,
                source_field=f"{base}.tool_calls[{index}]",
                transform="canonical-tool-json-v1",
                parse_hints=hints,
            )
    if builder.pending and not reader.context.options.get("allow_pending_final_action", False):
        raise SchemaError(
            "Trace ends with an action whose observation is missing; opt into incomplete traces explicitly"
        )


def _serialized_chat(
    reader: _Reader,
    builder: _EpisodeBuilder,
    *,
    payload: bytes | None = None,
    result: ChatParse | None = None,
) -> None:
    """Import explicit text chat dumps without manufacturing tool dependencies.

    Framing/header/footer fields are separate unscored logical inputs. The body
    is the scored field for a complete assistant turn. Each field has its own
    canonical reference encoding; physical parts never re-tokenize that field.
    All source spans still address the original serialized text, byte for byte.
    """
    from .parsing.chat import has_structural_marker

    source = reader.origin("text")
    if payload is None:
        payload = _bytes(reader.get("text", required=True), source)
    if result is None:
        result = _source_chat(reader, payload)
    if not result.detected:
        codes = ", ".join(item.code for item in result.diagnostics)
        raise ChatFramingError(
            f"Serialized chat has no unambiguous template ({codes}); select a known chat_template "
            "or use the document adapter for ordinary content",
            tuple(item.code for item in result.diagnostics),
        )
    framing_codes = {item.code for item in result.diagnostics}
    if any(frame.role is None for frame in result.frames):
        framing_codes.add("chat_unframed_tail")
    if framing_codes & {
        "chat_ambiguous_template",
        "chat_unknown_template",
        "chat_unrecognized_header",
        "chat_unframed_tail",
        "chat_frame_limit",
    }:
        # A bounded/recovered frame count is not the source's turn count. Queue
        # the framing failure before a valid original mask can look malformed.
        raise ChatFramingError(
            "Serialized chat has unparsed or ambiguous framing whose roles cannot be assigned safely",
            tuple(sorted(framing_codes)),
        )
    if not reader.context.options.get("allow_incomplete_serialized", False) and any(
        not frame.complete and frame.header_end != frame.body_end for frame in result.frames
    ):
        raise ChatFramingError(
            "Serialized chat contains an incomplete message; enable allow_incomplete_serialized "
            "to retain its bytes as unscored context",
            ("chat_incomplete_body",),
        )
    mask = reader.get("mask")
    if mask is not None and (
        not isinstance(mask, list)
        or len(mask) != len(result.frames)
        or any(type(value) is not bool for value in mask)
    ):
        raise SchemaError("Serialized training-turn mask must contain one boolean per parsed frame")
    builder.parser_annotations["chat_framing"] = {
        "template": result.template,
        "frames": [asdict(frame) for frame in result.frames],
        "diagnostics": [asdict(item) for item in result.diagnostics],
        "reference_encoding_unit": "header/body/footer logical field",
    }
    # Serialized tool markup lacks the explicit call/result IDs required by
    # LogicalAction. Guessing dependencies here would contaminate causal data.
    unsupported_tool_markup = (
        b"<tool_call>",
        b"[TOOL_CALLS]",
        b"[TOOL_RESULTS]",
        "<｜tool▁calls▁begin｜>".encode(),
        "<｜tool▁outputs▁begin｜>".encode(),
    )
    for turn, frame in enumerate(result.frames):
        if frame.role not in {
            "system",
            "developer",
            "user",
            "assistant",
        } or frame.recipient not in {None, "all"}:
            raise ChatFramingError(
                "Serialized tool, recipient, or unknown-role frames require structured messages "
                "with original call/result IDs",
                ("chat_missing_dependency_ids",),
            )
        if frame.role == "assistant" and has_structural_marker(
            payload[frame.header_end : frame.body_end], unsupported_tool_markup
        ):
            raise ChatFramingError(
                "Serialized tool markup requires structured original call/result IDs",
                ("chat_missing_dependency_ids",),
            )
        role = "system" if frame.role == "developer" else frame.role
        if role == "user" and builder.family_prompt is None:
            builder.family_prompt = payload[frame.header_end : frame.body_end]
        for part, (start, end) in (
            ("header", frame.header),
            ("body", frame.body),
            ("footer", frame.footer),
        ):
            if start == end:
                continue
            score = (
                part == "body"
                and role == "assistant"
                and frame.complete
                and (mask is None or mask[turn])
            )
            builder.append(
                payload[start:end],
                role=role,
                turn=turn,
                score=score,
                source_field=source,
                source_offset=start,
                parse_hints={"content_kind": "binary"} if part != "body" else None,
            )


def _documents(reader: _Reader, builder: _EpisodeBuilder) -> None:
    if reader.profile.kind == "document":
        payload = _bytes(reader.get("text", required=True), reader.origin("text"))
        framing = _source_chat(reader, payload)
        if framing.detected:
            _serialized_chat(reader, builder, payload=payload, result=framing)
            return
        builder.parser_annotations["chat_framing"] = {
            "template": framing.template,
            "frames": [],
            "diagnostics": [asdict(item) for item in framing.diagnostics],
            "reference_encoding_unit": "document logical field",
        }
        builder.append(
            payload,
            role="document",
            turn=0,
            score=True,
            source_field=reader.origin("text"),
            chunk_diagnostics=tuple(item.code for item in framing.diagnostics),
        )
    elif reader.profile.kind == "artifact":
        builder.append(
            _bytes(reader.get("prompt", required=True), "prompt"),
            role="user",
            turn=0,
            source_field=reader.origin("prompt"),
        )
        source = reader.row.get("source_document")
        if source is not None:
            builder.append(
                _bytes(source, "source_document"),
                role="document",
                turn=1,
                source_field="source_document",
                provenance="natural",
            )
        builder.append(
            _bytes(reader.get("target", required=True), "target"),
            role="assistant",
            turn=2,
            score=True,
            source_field=reader.origin("target"),
        )
    elif reader.profile.kind == "repository":
        builder.append(
            _bytes(reader.get("prompt", required=True), "prompt"),
            role="user",
            turn=0,
            source_field=reader.origin("prompt"),
        )
        builder.append(
            _bytes(reader.get("before", required=True), "before"),
            role="document",
            turn=1,
            source_field=reader.origin("before"),
            provenance="natural",
        )
        mode = reader.context.options.get("task_mode", "repair")
        if mode == "review":
            builder.append(
                _bytes(reader.get("patch", required=True), "patch"),
                role="document",
                turn=2,
                source_field=reader.origin("patch"),
            )
            builder.append(
                _bytes(reader.get("target", required=True), "target"),
                role="assistant",
                turn=3,
                score=True,
                source_field=reader.origin("target"),
            )
        elif mode == "repair":
            builder.append(
                _bytes(reader.get("patch", required=True), "patch"),
                role="assistant",
                turn=2,
                score=True,
                source_field=reader.origin("patch"),
            )
        else:
            raise SchemaError("Repository task_mode must be repair or review")
    elif reader.profile.kind == "fim":
        # A missing-span task has observed suffix conditioning; this is tagged FIM.
        for turn, name in enumerate(("prefix", "suffix", "middle")):
            builder.append(
                _bytes(reader.row[name], name),
                role="assistant" if name == "middle" else "document",
                turn=turn,
                score=name == "middle",
                source_field=name,
            )


def adapt_row(row: Mapping[str, Any], adapter: str, context: AdapterContext) -> Episode:
    """Convert one source row without rendering any supervision-side metadata."""
    if adapter not in PROFILES:
        raise SchemaError(f"Unknown adapter {adapter!r}; available: {', '.join(sorted(PROFILES))}")
    profile = PROFILES[adapter]
    if profile.kind == "native":
        from .masking import apply_tool_masks

        episode = Episode.from_dict(row)
        if context.codec is not None:
            for group in episode.iter_logical_records():
                cached = [item.cached_ids for item in group]
                if any(ids is not None for ids in cached):
                    if not all(ids is not None for ids in cached):
                        raise SchemaError("A logical record has incomplete reference meter chunks")
                    expected = tuple(context.codec.encode(b"".join(item.payload for item in group)))
                    actual = tuple(token for ids in cached for token in ids)
                    if actual != expected:
                        raise SchemaError(
                            "Native cached meter IDs differ from canonical logical-field tokenization"
                        )
        episode = apply_tool_masks(episode, max_tokens=context.max_record_tokens or 4096)
        return (
            episode
            if context.options.get("preserve_native_stream", False)
            else replace(episode, stream=context.stream)
        )
    if profile.kind == "seed":
        raise SchemaError(
            "Tool-use seed rows are TaskSeed inputs to collection, not completed trajectories"
        )
    reader = _Reader(row, profile, context)
    # Raw row hash audits transformations; source transport checksums are separate.
    raw_sha = digest_bytes(canonical_json(row).encode("utf-8"))
    episode_id = stable_id(
        "adapted-episode-v1",
        context.source_id,
        context.revision,
        str(reader.get("id", context.row_index)),
        str(context.row_index),
        raw_sha,
    )
    builder = _EpisodeBuilder(episode_id, reader)
    if profile.kind == "chat":
        _chat(reader, builder)
    elif profile.kind == "serialized_chat":
        _serialized_chat(reader, builder)
    else:
        _documents(reader, builder)
    supervision = _supervision(reader)
    if builder.parser_annotations:
        supervision = replace(
            supervision,
            annotations_json=canonical_json(
                json.loads(supervision.annotations_json) | {"parser": builder.parser_annotations}
            ),
        )
    if adapter == "swe_zero":
        supervision = replace(
            supervision,
            outcome="incomplete"
            if str(row.get("exit_status", "")).casefold() != "submitted"
            else "unknown",
            verification="none",
        )
    episode = Episode(
        episode_id=episode_id,
        records=tuple(builder.records),
        lineage=_lineage(reader, raw_sha, builder.family_prompt),
        stream=context.stream,
        actions=tuple(builder.actions),
        supervision=supervision,
        tools_json=builder.tools_json,
    )
    if profile.trajectory:
        # Never promote an unverified Submitted flag or a failed branch to success.
        episode = apply_recovery_policy(
            episode,
            imitate_unknown=bool(context.options.get("imitate_unknown", profile.unknown_imitation)),
        )
    dependencies = []
    # Explicit annotations use original message indices, then resolve after chunks.
    for item in row.get("dependencies", ()):
        writers = [record for record in episode.records if record.turn_index == item["writer_turn"]]
        readers = [
            record
            for record in episode.records
            if record.turn_index == item["reader_turn"] and record.policy_mask
        ]
        if not writers or not readers:
            raise SchemaError("Dependency has no matching original writer or scored reader turn")
        dependencies.append(
            Dependency(writers[0].record_id, readers[-1].record_id, item.get("kind", "evidence"))
        )
    return replace(episode, dependencies=tuple(dependencies))
