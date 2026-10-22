"""Source profiles over one strict causal conversation/document parser.

Adapters describe published schemas; they never assume a dataset's rows are
verified successes. Field maps can override dotted paths for pinned variants.
Malformed or unavailable fields are explicit rejects, not guessed targets.
"""

from __future__ import annotations

import datetime
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
    sequential_meridians,
    stable_id,
)

if TYPE_CHECKING:
    from .compression import DocumentCompressor
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
    content_kind: str = "auto"
    # A text convention for tool calls/results, converted by data/dialects.py.
    dialect: str = ""
    # Feedback turns after the task prompt come from the environment, not a user.
    environment_turns: bool = False
    # An artifact row may omit the request (a rewrite is its own request).
    prompt_required: bool = True


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
    "hints": ("hints_text",),
    "context": ("context", "context_items"),
    "patch": ("patch", "target_patch"),
    "benchmark": ("benchmark_tags", "metadata.benchmark_tags"),
    "dependencies": ("metadata.dependencies",),
    "document": ("source_document",),
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
    # A target the publisher checked (unit tests, an instruction checker, an answer match) and
    # recorded as passing.
    "tested_artifact": AdapterProfile(kind="artifact", outcome_verification="publisher_tests"),
    "human_tested_artifact": AdapterProfile(
        kind="artifact", outcome_verification="publisher_tests", emission_provenance="natural"
    ),
    # The original as an observed document, its rewrite as the target; no request is stored.
    "code_rewrite": AdapterProfile(kind="artifact", prompt_required=False),
    "human_code_rewrite": AdapterProfile(
        kind="artifact", prompt_required=False, emission_provenance="natural"
    ),
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
    "smoltalk2": AdapterProfile(fields=_fields(messages="messages")),
    "nemotron_chat": AdapterProfile(require_turn_mask=True),
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
    # Text scaffolds: the assistant message is the action and the next user message its result.
    "openhands_xml": AdapterProfile(
        trajectory=True,
        observation_provenance="executed",
        outcome_verification="publisher_tests",
        harness="openhands",
    ),
    "swe_agent_fenced": AdapterProfile(
        trajectory=True,
        observation_provenance="executed",
        outcome_verification="publisher_tests",
        harness="swe_agent",
        fields=_fields(messages="trajectory", outcome="target"),
    ),
    "codeact_agent": AdapterProfile(
        trajectory=True,
        observation_provenance="executed",
        outcome_verification="publisher_tests",
        harness="codeact",
    ),
    "mini_swe_agent": AdapterProfile(
        trajectory=True,
        observation_provenance="executed",
        outcome_verification="publisher_tests",
        harness="mini_swe",
    ),
    # Plain ReAct text (Thought/Action turns, environment replies): no structured calls exist.
    "react_agent": AdapterProfile(observation_provenance="executed", environment_turns=True),
    "react_function_agent": AdapterProfile(
        observation_provenance="executed", environment_turns=True, dialect="environment_roles"
    ),
    # Tool dialogues with an LLM-simulated tool backend.
    "hermes_tools": AdapterProfile(observation_provenance="simulated", dialect="hermes"),
    "python_call_tools": AdapterProfile(observation_provenance="simulated", dialect="python_calls"),
    # gpt-oss traces of a browsing agent whose search and page tools really ran.
    "harmony_tools": AdapterProfile(observation_provenance="executed", dialect="harmony"),
    "call_list_tools": AdapterProfile(observation_provenance="simulated", dialect="call_list"),
    # tau2 logs: the customer also calls tools of its own, which stay on the customer's side.
    "device_call_tools": AdapterProfile(observation_provenance="executed", dialect="device_calls"),
    # NeMo-Gym RL trajectories: Responses-API items plus the step that comes next.
    "responses_tools": AdapterProfile(
        observation_provenance="simulated",
        dialect="responses",
        fields=_fields(messages="responses_create_params.input"),
    ),
    "tool_dict_tools": AdapterProfile(
        observation_provenance="executed",
        dialect="tool_dict",
        fields=_fields(messages="conversation"),
    ),
    "glaive_tools": AdapterProfile(
        observation_provenance="simulated",
        dialect="glaive",
        fields=_fields(messages="chat"),
    ),
    "xlam_calls": AdapterProfile(observation_provenance="simulated", dialect="xlam"),
    "simulated_tool_roles": AdapterProfile(
        observation_provenance="simulated", dialect="role_calls"
    ),
    # Tool calls executed against real MCP servers.
    "mcp_tools": AdapterProfile(observation_provenance="executed", dialect="role_calls"),
    "turn_pairs": AdapterProfile(dialect="turn_pairs", fields=_fields(messages="conversation")),
    "human_assistant_text": AdapterProfile(dialect="human_assistant"),
    "alternating_turns": AdapterProfile(
        dialect="alternating_turns", fields=_fields(messages="data")
    ),
    "prompt_and_reply": AdapterProfile(
        dialect="prompt_and_reply", fields=_fields(messages="conversation")
    ),
    "prompt_completion": AdapterProfile(
        dialect="prompt_completion", fields=_fields(messages="prompt")
    ),
    # An exit_status of Submitted is explicitly NOT test verification.
    "swe_zero": AdapterProfile(
        trajectory=True,
        observation_provenance="executed",
        harness="mini_swe",
        require_parent=True,
        fields=_fields(outcome="__no_verified_outcome__"),
    ),
}


_HINT_OPTIONS = frozenset(
    {"content_kind", "language", "filename", "mime_type", "locale", "chat_template"}
)
# Read by the preparation pipeline itself rather than by an adapter.
_PIPELINE_OPTIONS = frozenset({"lineage_join", "row_filters", "quality", "read_batch_rows"})
_PROFILE_OPTIONS: dict[str, frozenset[str]] = {
    "native": frozenset({"preserve_native_stream"}),
    "chat": _HINT_OPTIONS
    | {
        "allow_pending_final_action",
        "observation_provenance",
        "assume_outcome",
        "positional_call_ids",
        "admit_unknown_outcome",
        "path_dependencies",
        "drop_empty_user_turns",
    },
    "serialized_chat": _HINT_OPTIONS | {"allow_incomplete_serialized"},
    "document": _HINT_OPTIONS | {"allow_incomplete_serialized"},
    "artifact": _HINT_OPTIONS,
    "repository": _HINT_OPTIONS,
    "fim": _HINT_OPTIONS,
}


def validate_options(adapter: str, options: Mapping[str, Any]) -> None:
    """Reject option names the selected adapter never reads; a typo must not pass silently."""
    if adapter not in PROFILES:
        raise SchemaError(f"Unknown adapter {adapter!r}; available: {', '.join(sorted(PROFILES))}")
    allowed = _PROFILE_OPTIONS[PROFILES[adapter].kind] | _PIPELINE_OPTIONS
    unknown = sorted(set(options) - allowed)
    if unknown:
        raise SchemaError(
            f"Adapter {adapter!r} has no option(s) {unknown}; allowed: {sorted(allowed)}"
        )


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
    # Reads long observed fields (reference documents, large tool results) as
    # teacher features instead of byte chunks; see data/compression.py.
    compressor: DocumentCompressor | None = None


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

    def present(self, key: str) -> list[tuple[str, Any]]:
        """Every configured path that holds a value, with its path, in order."""
        found = ((path, _path(self.row, path)) for path in self.fields.get(key, (key,)))
        return [(path, value) for path, value in found if value is not None]

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
    if normalized in {"resolved", "success", "passed", "pass", "true"}:
        return "success"
    if normalized in {"unresolved", "failure", "failed", "fail", "false"}:
        return "failure"
    if normalized in {"incomplete", "timeout", "interrupted"}:
        return "incomplete"
    try:
        score = float(normalized)  # a pass rate delivered as text: only 0 and 1 are verdicts
    except ValueError:
        return "unknown"
    return "success" if score == 1 else "failure" if score == 0 else "unknown"


def _list_of(value: Any, kind: type, name: str) -> tuple[Any, ...]:
    if not isinstance(value, (list, tuple)) or any(type(item) is not kind for item in value):
        raise SchemaError(f"{name} must be a list of {kind.__name__}")
    return tuple(value)


def _label(value: Any, name: str) -> str | None:
    if value is None or isinstance(value, str):
        return value
    if type(value) is int:
        return str(value)
    if isinstance(value, datetime.datetime):
        return value.isoformat()  # columnar sources deliver timestamps as datetimes
    raise SchemaError(f"{name} must be a string, got {type(value).__name__}")


def _supervision(reader: _Reader) -> Supervision:
    outcome = _outcome(reader.get("outcome"))
    verification = reader.profile.outcome_verification
    assumed = reader.context.options.get("assume_outcome")
    if assumed is not None and reader.get("outcome") is None:
        # The publisher kept only successful traces but gives no per-row label: success is
        # recorded, as curation and not as a test verdict, so it never fills a verified quota.
        if assumed != "success":
            raise SchemaError("assume_outcome can only be 'success'")
        outcome, verification = "success", "publisher_curated"
    benchmark = reader.get("benchmark", ())
    if isinstance(benchmark, str):
        benchmark = (benchmark,)
    reward = reader.get("reward")
    if reward is not None and (isinstance(reward, bool) or not isinstance(reward, (int, float))):
        raise SchemaError("reward must be a number")
    annotations = {
        key: reader.row[key]
        for key in ("metadata", "exit_status", "duration_sec", "license", "used_in")
        if key in reader.row
    }
    return Supervision(
        outcome=outcome,
        verification=verification if outcome != "unknown" else "none",
        reward=reward,
        recovery_turns=_list_of(reader.get("recovery", ()), int, "recovery_turns"),
        benchmark_tags=_list_of(benchmark, str, "benchmark_tags"),
        annotations_json=canonical_json(annotations),
    )


def _lineage(reader: _Reader, raw_sha: str, prompt_override: bytes | None = None) -> Lineage:
    repository = _label(reader.get("repository"), "repository")
    task = _label(reader.get("task"), "task")
    family = _label(reader.get("family"), "family")
    ancestors = reader.get("ancestors", ())
    if isinstance(ancestors, str):
        ancestors = (ancestors,)
    if not isinstance(ancestors, (list, tuple)):
        raise SchemaError("ancestor_ids must be a list")
    if not family:
        if repository:
            family = f"repository:{str(repository).casefold()}"
        elif task:
            family = f"task:{task}"
        elif reader.profile.require_parent:
            raise SchemaError(
                "Missing original parent/task lineage; supply field_map.family or a joined parent manifest"
            )
        else:
            # Prompt hashing groups repeated completions of the same prompt.
            prompt = prompt_override if prompt_override is not None else reader.get("prompt")
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
        created_at=_label(reader.get("created_at"), "created_at"),
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


_JSON_TYPES = frozenset({"string", "number", "integer", "boolean", "array", "object", "null"})
_PYTHON_TYPES = {
    "str": "string",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
    "list": "array",
    "tuple": "array",
    "set": "array",
    "dict": "object",
}


def _json_type(name: Any) -> Any:
    """A Python-style type name (``str``, ``List[float]``, ``int, optional``) as a JSON type."""
    if isinstance(name, list):
        return [_json_type(item) for item in name]
    if not isinstance(name, str) or name in _JSON_TYPES:
        return name
    head = name.split(",")[0].split("[")[0].strip().lower()
    return _PYTHON_TYPES.get(head, head if head in _JSON_TYPES else None)


def _clean_schema(value: Any) -> Any:
    """Make a tool's parameters valid JSON Schema without changing what they say.

    Columnar stores fill every schema key that a row lacks with null; older schemas spell
    ``required`` as a boolean on each property; many tool sets write Python type names.
    """
    if isinstance(value, dict):
        cleaned = {
            key: _clean_schema(item)
            for key, item in value.items()
            if (item is not None or key in {"default", "const"})
            and not (key == "required" and isinstance(item, bool))
        }
        if isinstance(cleaned.get("type"), (str, list)):
            kind = _json_type(cleaned["type"])
            if kind is None or (isinstance(kind, list) and None in kind):
                del cleaned["type"]  # an unknown spelling constrains nothing
            else:
                cleaned["type"] = kind
        return cleaned
    if isinstance(value, list):
        return [_clean_schema(item) for item in value]
    return value


def _tools(value: Any) -> str:
    schemas = _object(value, "tools") if value is not None else []
    if not isinstance(schemas, list):
        raise SchemaError("tools must be a list (possibly JSON encoded)")
    schemas = [_object(item, "tool schema") for item in schemas]
    if any(not isinstance(item, dict) for item in schemas):
        raise SchemaError("Every tool schema must be an object")
    return canonical_json(_clean_schema([_function_tool(item) for item in schemas]))


def _function_tool(tool: dict[str, Any]) -> dict[str, Any]:
    """A tool in the AI SDK spelling (``id`` and ``inputSchema.jsonSchema``) as an OpenAI one."""
    schema = tool.get("inputSchema")
    if not isinstance(schema, dict) or not isinstance(tool.get("id"), str):
        return tool
    function = {"name": tool["id"], "parameters": schema.get("jsonSchema", schema)}
    if isinstance(tool.get("description"), str):
        function["description"] = tool["description"]
    return {"type": "function", "function": function}


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
        self.observation_provenance = reader.context.options.get(
            "observation_provenance", reader.profile.observation_provenance
        )
        self.assistant_spoke = False
        # For sources whose calls carry no ID: calls are numbered by position and each
        # result answers the oldest unanswered call. Opt-in, because it is a convention.
        self.positional = bool(reader.context.options.get("positional_call_ids", False))

    def content_hints(self, *objects: Mapping[str, Any]) -> dict[str, Any]:
        """Specific block/message hints override source options and row defaults.

        Only an allowlist of parser hints leaves this method. Outcome labels and
        arbitrary metadata are never rendered or treated as content hints.
        """
        keys = ("content_kind", "language", "filename", "mime_type", "locale", "chat_template")
        # A row-level language, filename or MIME type describes one file. In a conversation it
        # would force that grammar onto every system prompt, question and answer, which then
        # fail to parse; only a message's own hints (below) or the source options set them.
        row_level = (
            ("content_kind", "locale", "chat_template")
            if self.reader.profile.kind == "chat"
            else keys
        )
        hints = {
            key: self.reader.context.options.get(
                key, self.reader.get(key) if key in row_level else None
            )
            for key in keys
        }
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
                else self.observation_provenance
                if result
                else "natural"
            )
        compressor = self.reader.context.compressor
        if (
            compressor is not None
            and action_id is None
            and compressor.applies(payload, role=role, score=score)
        ):
            records = compressor.records(
                payload,
                logical_id=logical_id,
                role=role,
                turn_index=turn,
                provenance=provenance,
                source_uri=self.reader.context.uri,
                source_field=source_field,
                transform=transform,
                result_of=result.action_id if result else None,
                execution_id=result.execution_id if result else None,
            )
            self.records.extend(records)
            return records
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
        transform: str = "identity",
        parse_hints: Mapping[str, Any] | None = None,
    ) -> None:
        if call_id is None:
            if len(self.pending) != 1 and not (self.positional and self.pending):
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
            transform=transform,
            parse_hints=parse_hints,
        )


def _call_id(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise SchemaError("Native tool calls require their original call ID as a nonempty string")
    return value


def _tool_call(call: Any, default_id: str | None = None) -> tuple[str, str, Any]:
    """Validate one native tool call; returns its ID, tool name and arguments."""
    if not isinstance(call, Mapping):
        raise SchemaError("Every tool call must be an object")
    function = call.get("function", call)
    if not isinstance(function, Mapping):
        raise SchemaError("A tool call's function must be an object")
    name = function.get("name")
    if not isinstance(name, str) or not name:
        raise SchemaError("A tool call needs a nonempty function name")
    call_id = call.get("id", default_id) if default_id is not None else call.get("id")
    return _call_id(call_id), name, function.get("arguments", {})


def _first_user_text(content: Any) -> bytes | None:
    """The text of a user turn, whether plain or split into content blocks."""
    if isinstance(content, (str, bytes)):
        return _bytes(content, "content")
    if isinstance(content, list):
        texts = [
            _bytes(block.get("text", ""), "content block")
            for block in content
            if isinstance(block, Mapping) and block.get("type", "text") in _TEXT_BLOCKS
        ]
        return b"".join(texts) if texts else None
    return None


_OPENHANDS_CALL = re.compile(r"<function=([^>\s]+)>(.*?)</function>\s*$", re.DOTALL)
_OPENHANDS_PARAMETER = re.compile(r"<parameter=([^>\s]+)>(.*?)</parameter>", re.DOTALL)
_FENCE = re.compile(r"```(?:bash|sh)?[ \t]*\n(.*?)```", re.DOTALL)
_EXECUTE = re.compile(r"<execute>\n?(.*?)\n?</execute>", re.DOTALL)
_TEXT_HARNESSES = frozenset({"terminus2", "mini_swe", "openhands", "swe_agent", "codeact"})


def _harness_action(content: str, harness: str) -> tuple[str, dict[str, Any]] | None:
    if harness == "terminus2":
        stripped = content.strip()
        if stripped.startswith("<think>") and "</think>" in stripped:
            # Reasoning models write their thinking first; the call is the JSON after it.
            stripped = stripped.partition("</think>")[2].strip()
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
    if harness == "openhands":
        # One <function=name><parameter=key>value</parameter></function> closes the message.
        call = _OPENHANDS_CALL.search(content)
        if call is not None and content.count("<function=") == 1:
            arguments = {
                key: value.removeprefix("\n").removesuffix("\n")
                for key, value in _OPENHANDS_PARAMETER.findall(call.group(2))
            }
            return call.group(1), arguments
    if harness == "codeact":
        # One <execute> code block per message; the next user message is its observation.
        blocks = _EXECUTE.findall(content)
        if len(blocks) == 1:
            return "execute", {"code": blocks[0]}
    if harness == "swe_agent":
        commands = _FENCE.findall(content)
        if len(commands) == 1:
            return "bash", {"command": commands[0]}
    return None


_TEXT_BLOCKS = frozenset({"text", "input_text", "output_text"})
_MASK_KEYS = ("train", "loss", "mask")
_CONTENT_KEYS = ("content", "value", "text", "system_prompt")


def _first(message: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    return next((message[key] for key in keys if message.get(key) is not None), None)


def _with_reasoning(message: Mapping[str, Any], role: str, content: Any) -> Any:
    """Spell a reasoning trace kept in its own field inline, as every other source spells it."""
    reasoning = message.get("reasoning_content") if role == "assistant" else None
    if not isinstance(reasoning, str) or not reasoning.strip():
        return content
    if content is not None and not isinstance(content, str):
        return content
    return f"<think>\n{reasoning.strip()}\n</think>\n\n{content or ''}"


def _result_id(message: Mapping[str, Any], role: str) -> str | None:
    """The call a result answers, under the spellings published sets use."""
    for key in ("tool_call_id", "tool_use_id"):
        if isinstance(message.get(key), str):
            return message[key]
    ids = _object(message["tool_call_ids"], "tool_call_ids") if message.get("tool_call_ids") else ()
    if isinstance(ids, list) and len(ids) == 1 and isinstance(ids[0], str):
        return ids[0]
    return message["id"] if role == "tool" and isinstance(message.get("id"), str) else None


def _chat(reader: _Reader, builder: _EpisodeBuilder) -> None:
    dialect = reader.profile.dialect
    # A dialect may build the whole dialogue from other row fields (xLAM has no messages).
    value = reader.get("messages", required=not dialect)
    if dialect:
        from .dialects import DIALECTS

        messages, tools = DIALECTS[dialect](reader.row, value)
        if tools is not None:
            builder.tools_json = _tools(tools)
    else:
        messages = _object(value, "messages")
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
    from .dialects import role_of

    drop_empty_users = bool(reader.context.options.get("drop_empty_user_turns", False))
    for turn, message in enumerate(messages):
        if isinstance(message, str):
            message = _object(message, "A message")  # some sets store one JSON string per message
        if not isinstance(message, Mapping):
            raise SchemaError("Every message must be an object")
        role = role_of(message)
        if message.get("role", message.get("from")) == "developer":
            builder.parser_annotations.setdefault("source_roles", {})[str(turn)] = "developer"
        hints = builder.content_hints(message)
        if role not in {"system", "user", "assistant", "tool", "environment"}:
            raise SchemaError(f"Unknown chat role {role!r}")
        own_mask = _first(message, _MASK_KEYS)
        if own_mask not in (None, True, False, 0, 1):
            raise SchemaError("Per-message train/loss field must be boolean")
        score = (
            role == "assistant"
            and own_mask in (None, True, 1)
            and (turn_mask is None or turn_mask[turn])
        )
        base = f"{reader.origin('messages')}[{turn}]"
        content = _with_reasoning(message, role, _first(message, _CONTENT_KEYS))
        if role == "user" and builder.family_prompt is None:
            # Repeated prompts must share one family however they are encoded.
            builder.family_prompt = _first_user_text(content)
        if (
            role == "tool"
            and isinstance(content, list)
            and content
            and all(
                isinstance(block, dict) and block.get("type", "text") in _TEXT_BLOCKS
                for block in content
            )
        ):
            # A result delivered as text blocks is still one observation.
            content = "".join(_bytes(block.get("text", ""), base).decode() for block in content)
        structured_result = role in {"tool", "environment"} and (
            isinstance(content, dict)
            or (
                isinstance(content, list)
                and not all(isinstance(block, dict) and "type" in block for block in content)
            )
        )
        if structured_result:
            # A JSON result object is delivered as its canonical serialization.
            content = canonical_json(content)
        calls = (
            _object(message.get("tool_calls"), "tool_calls") if message.get("tool_calls") else []
        )
        if not isinstance(calls, list):
            raise SchemaError("tool_calls must be a list")
        if message.get("function_call"):
            if calls:
                raise SchemaError("Both function_call and tool_calls are present")
            calls = [{"id": f"{builder.episode_id}:{turn}", "function": message["function_call"]}]
        if calls and role != "assistant":
            raise SchemaError("Only assistant messages can issue tools")
        blank_user = role == "user" and isinstance(content, str) and not content.strip()
        if blank_user and drop_empty_users and not calls:
            continue  # the publisher blanked a turn (a withheld device action), not the prompt
        if (content is None and not calls) or blank_user:
            raise SchemaError(
                f"Message {turn} has no content; a source that withholds prompts must be restored first"
            )
        if isinstance(content, list):
            # Preserve block order, including interleaved Anthropic tool blocks.
            for index, block in enumerate(content):
                if not isinstance(block, dict):
                    raise SchemaError("A content block must be an object")
                kind = block.get("type", "text")
                field_name = f"{base}.content[{index}]"
                block_hints = builder.content_hints(message, block)
                if kind in _TEXT_BLOCKS:
                    builder.append(
                        _bytes(block.get("text", ""), field_name),
                        role=role,
                        turn=turn,
                        score=score,
                        source_field=field_name + ".text",
                        parse_hints=block_hints,
                    )
                elif kind == "tool_use" and role == "assistant":
                    if not isinstance(block.get("name"), str) or "input" not in block:
                        raise SchemaError("A tool_use block needs a name and an input object")
                    builder.action(
                        canonical_json(block).encode(),
                        call_id=_call_id(block.get("id")),
                        tool_name=block["name"],
                        arguments=block["input"],
                        turn=turn,
                        score=score,
                        source_field=field_name,
                        transform="canonical-tool-json-v1",
                        parse_hints=block_hints,
                    )
                elif kind in {"tool_result", "tool-result"}:
                    # Anthropic spells the result's call "tool_use_id" and its text "content";
                    # the AI SDK spells them "toolCallId" and "output.value".
                    output = block.get("output")
                    value = output.get("value") if isinstance(output, Mapping) else None
                    builder.observation(
                        _bytes(
                            value if value is not None else block.get("content", ""), field_name
                        ),
                        turn=turn,
                        call_id=block.get("tool_use_id") or block.get("toolCallId"),
                        source_field=field_name
                        + (".output.value" if value is not None else ".content"),
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
                    call_id=_result_id(message, role),
                    source_field=base + ".content",
                    transform="canonical-json-v1" if structured_result else "identity",
                    parse_hints=hints,
                )
            elif role == "user" and builder.pending and reader.profile.harness in _TEXT_HARNESSES:
                builder.observation(
                    payload,
                    turn=turn,
                    call_id=_result_id(message, role),
                    source_field=base + ".content",
                    parse_hints=hints,
                )
            elif role == "user" and reader.profile.environment_turns and builder.assistant_spoke:
                # Replies after the task prompt come from the environment; no call is named.
                builder.append(
                    payload,
                    role="environment",
                    turn=turn,
                    source_field=base + ".content",
                    provenance=builder.observation_provenance,
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
            elif payload or not calls:
                builder.append(
                    payload,
                    role=role,
                    turn=turn,
                    score=score,
                    source_field=base + ".content",
                    parse_hints=hints,
                )
        if role == "assistant":
            builder.assistant_spoke = True
        for index, call in enumerate(calls):
            call_id, tool_name, arguments = _tool_call(
                call, f"{builder.episode_id}:{turn}:{index}" if builder.positional else None
            )
            canonical = {
                "function": {"arguments": arguments, "name": tool_name},
                "id": call_id,
                "type": "function",
            }
            builder.action(
                canonical_json(canonical).encode(),
                call_id=call_id,
                tool_name=tool_name,
                arguments=arguments,
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
        prompt = reader.get("prompt", required=reader.profile.prompt_required)
        if prompt is not None:
            builder.append(
                _bytes(prompt, "prompt"), role="user", turn=0, source_field=reader.origin("prompt")
            )
        _observed_files(reader, builder, "document", turn=1)
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
        # Discussion before the change, then the files as they were, then the change.
        _observed_files(reader, builder, "hints", turn=1)
        _observed_files(reader, builder, "before", turn=1)
        for path, payload, field_name in _files(reader, "patch", _PATCH_KEYS, required=True):
            builder.append(
                payload,
                role="assistant",
                turn=2,
                score=True,
                source_field=field_name,
                parse_hints={"filename": path} if path else None,
            )
    elif reader.profile.kind == "fim":
        # A missing-span task has observed suffix conditioning; this is tagged FIM.
        _observed_files(reader, builder, "context", turn=0)
        for turn, name in enumerate(("prefix", "suffix", "middle"), start=1):
            builder.append(
                _bytes(reader.row[name], name),
                role="assistant" if name == "middle" else "document",
                turn=turn,
                score=name == "middle",
                source_field=name,
            )


_FILE_PATH_KEYS = ("path", "file_path", "filename", "new_path", "old_path", "old_file", "new_file")
_FILE_CONTENT_KEYS = ("content", "text", "code", "before_content", "body", "comment")
_PATCH_KEYS = ("unified_patch", "patch", "diff", "resolving_diff", "content")


def _files(
    reader: _Reader,
    key: str,
    content_keys: tuple[str, ...] = _FILE_CONTENT_KEYS,
    *,
    required: bool = False,
) -> list[tuple[str | None, bytes, str]]:
    """The files or passages a row delivers under a field: text, or a list of either.

    Every configured path that holds a value is read, in the order listed, so a field map
    can name several row fields (a PR body, then its comments) for one role. A list item is
    a string or an object with a path and its content under one of ``content_keys``; an
    object with none of those keys is an error, one whose content is null or empty is
    skipped. Each item keeps its own source field and, when it names a path, a filename
    parser hint. A field that yields nothing is an error only when ``required``.
    """
    files = []
    for origin, value in reader.present(key):
        if isinstance(value, (str, bytes)):
            items, names = [value], [origin]
        elif isinstance(value, list):
            items, names = value, [f"{origin}[{index}]" for index in range(len(value))]
        else:
            raise SchemaError(f"{key} must be text or a list of files, got {type(value).__name__}")
        for item, name in zip(items, names, strict=True):
            path = None
            if isinstance(item, Mapping):
                if not any(candidate in item for candidate in content_keys):
                    raise SchemaError(f"{name} has none of {content_keys}")
                path = _first(item, _FILE_PATH_KEYS)
                item = _first(item, content_keys)
            if item is not None and (payload := _bytes(item, name)):
                files.append((path, payload, name))
    if required and not files:
        reader.get(key, required=True)  # a missing field names the field map to configure
        raise SchemaError(f"{key} is empty")
    return files


def _observed_files(reader: _Reader, builder: _EpisodeBuilder, key: str, *, turn: int) -> None:
    """Natural passages the model reads before it writes: documents, never scored."""
    for path, payload, name in _files(reader, key):
        builder.append(
            payload,
            role="document",
            turn=turn,
            source_field=name,
            provenance="natural",
            parse_hints={"filename": path} if path else None,
        )


_EDIT_ARGUMENTS = frozenset({"old_str", "new_str", "file_text", "insert_line"})


def _path_dependencies(episode: Episode) -> list[Dependency]:
    """Link an edit of a file to the first observation that showed the model that file.

    Only structured tool arguments are read: an action whose ``path`` argument names a
    file, answered by a result, makes that result the file's evidence; a later scored
    action that edits the same ``path`` (editor arguments ``old_str``, ``new_str``,
    ``file_text`` or ``insert_line``) depends on it. Shell commands carry no ``path``
    and so create no link; nothing is inferred from free text.
    """
    result_of = {
        record.result_of: record.record_id
        for record in episode.records
        if record.result_of and record.part_index == 0
    }
    records = {record.record_id: record for record in episode.records}
    evidence: dict[str, str] = {}
    dependencies = []
    for action in episode.actions:
        arguments = json.loads(action.arguments_json)
        path = arguments.get("path") if isinstance(arguments, dict) else None
        if not isinstance(path, str):
            continue
        reader = action.record_ids[0]
        if path in evidence and _EDIT_ARGUMENTS & arguments.keys() and records[reader].policy_mask:
            dependencies.append(Dependency(evidence[path], reader, "file_read_to_edit"))
        if action.action_id in result_of:
            evidence.setdefault(path, result_of[action.action_id])
    return dependencies


def adapt_row(row: Mapping[str, Any], adapter: str, context: AdapterContext) -> Episode:
    """Convert one source row without rendering any supervision-side metadata."""
    validate_options(adapter, context.options)
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
    admitted = episode.supervision.outcome == "unknown" and context.options.get(
        "admit_unknown_outcome", False
    )
    if profile.trajectory and not admitted:
        # Never promote an unverified Submitted flag or a failed branch to success. A source
        # that records no outcome at all can be admitted as unknown-quality imitation data
        # explicitly; a recorded failure is never admitted.
        episode = apply_recovery_policy(episode)
    dependencies = []
    # Explicit annotations use original message indices, then resolve after chunks.
    annotated = reader.get("dependencies", [])
    if not isinstance(annotated, list):
        raise SchemaError("Dependency annotations must be a list")
    for item in annotated:
        if (
            not isinstance(item, Mapping)
            or type(item.get("writer_turn")) is not int
            or type(item.get("reader_turn")) is not int
        ):
            raise SchemaError("A dependency needs integer writer_turn and reader_turn")
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
    if reader.context.options.get("path_dependencies", False):
        dependencies.extend(_path_dependencies(episode))
    return sequential_meridians(replace(episode, dependencies=tuple(dependencies)))
