"""Budgeted teacher collection against explicitly injected executed environments.

Every request is reserved before it is submitted, including rejected attempts.
An interrupted request keeps its reservation until its usage is reconciled; a
network retry cannot silently spend an unmetered second completion. Verifier
labels and simulator-private intent are never sent to the teacher/student.
"""

from __future__ import annotations

import base64
import json
import os
import sqlite3
import time
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Protocol

from megatoken.environments import Environment, EnvironmentResult
from megatoken.io import atomic_binary_writer, content_hash, write_json

from .chunking import ChunkingConfig
from .dedup import DedupStore
from .masking import tool_free_spans
from .recipes import load_recipes
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
from .sources import TermsReview, pin_local_source, write_source_manifest

_PILOT = load_recipes()["collection_pilot"]


class CollectionLimit(RuntimeError):
    """The predeclared collection/task/attempt budget has no available capacity."""


@dataclass(frozen=True, slots=True)
class TaskSeed:
    seed_id: str
    family_id: str
    category: str
    prompt: bytes
    environment_json: str
    tools_json: str
    revision: str
    ancestor_ids: tuple[str, ...] = ()
    mode: str = "executed"
    # Public source policy belongs in the initial causal control; customer intent
    # remains inside the injected environment until an actual user communicates it.
    policy: bytes = b""

    def __post_init__(self) -> None:
        if not all((self.seed_id, self.family_id, self.category, self.revision)):
            raise ValueError("A seed requires immutable source/task/family identities")
        if not isinstance(self.prompt, bytes) or not isinstance(self.policy, bytes):
            raise TypeError("Task prompt and public policy must be immutable bytes")
        if not isinstance(json.loads(self.environment_json), dict) or not isinstance(
            json.loads(self.tools_json), list
        ):
            raise ValueError("Seed environment/tools must be an object/list")
        if self.mode not in {"executed", "artifact", "simulated"}:
            raise ValueError("Unknown collection mode")

    @property
    def task_key(self) -> str:
        # Rephrasing only the prompt does not manufacture another distinct task.
        environment = dict(json.loads(self.environment_json))
        environment.pop("prompt", None)
        return stable_id("executable-task-v1", self.family_id, canonical_json(environment))

    @classmethod
    def from_tool_use_seed(
        cls,
        row: Mapping[str, Any],
        *,
        revision: str,
        environment_config: Mapping[str, Any],
        public_user_message: str,
    ) -> TaskSeed:
        """Import NVIDIA's policy/tool/scenario seed without leaking private intent.

        ``public_user_message`` must be the actual first communicated user turn;
        it is deliberately not inferred from ``customer_scenario``. Bind a NeMo
        Gym simulator through an injected environment adapter and retain its
        simulated provenance. It cannot fill executed-success quotas.
        """
        required = {"id", "policy", "tools", "customer_scenario"}
        if not required.issubset(row):
            raise ValueError(f"Missing seed fields: {sorted(required - set(row))}")
        tools = json.loads(row["tools"]) if isinstance(row["tools"], str) else row["tools"]
        config = {**environment_config, "private_customer_scenario": row["customer_scenario"]}
        return cls(
            seed_id=str(row["id"]),
            family_id=f"tool-seed:{row['id']}",
            category="structured_tools",
            prompt=public_user_message.encode(),
            policy=str(row["policy"]).encode(),
            tools_json=canonical_json(tools),
            environment_json=canonical_json(config),
            revision=revision,
            mode="simulated",
        )


@dataclass(frozen=True, slots=True)
class TeacherCall:
    call_id: str
    name: str
    arguments_json: str
    wire_payload_base64: str | None = None
    emitted_ids: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if (
            not self.call_id
            or not self.name
            or not isinstance(json.loads(self.arguments_json), dict)
        ):
            raise ValueError("Teacher tool calls must be complete, identified JSON objects")
        if self.wire_payload_base64 is not None:
            from megatoken.tool_schema import parse_tool_action

            name, arguments, _, _ = parse_tool_action(
                base64.b64decode(self.wire_payload_base64, validate=True)
            )
            if name != self.name or arguments != json.loads(self.arguments_json):
                raise ValueError("Native action bytes differ from their executable descriptor")

    @property
    def action_bytes(self) -> bytes:
        if self.wire_payload_base64 is not None:
            return base64.b64decode(self.wire_payload_base64, validate=True)
        return canonical_json(
            {"name": self.name, "arguments": json.loads(self.arguments_json)}
        ).encode()


@dataclass(frozen=True, slots=True)
class TeacherRequest:
    request_id: str
    messages_json: str
    tools_json: str
    max_output_tokens: int
    codec_fingerprint: str
    session_id: str = ""
    # Local student inference receives only public physical records, never the
    # Episode supervision/environment configuration. HTTP ignores these fields.
    records: tuple[PhysicalRecord, ...] = ()
    control_seed: bytes = b""


@dataclass(frozen=True, slots=True)
class TeacherResponse:
    content: bytes = b""
    calls: tuple[TeacherCall, ...] = ()
    finish_reason: str = "stop"
    reasoning: bytes = b""
    provider_input_tokens: int = 0
    provider_output_tokens: int = 0
    provider_cache_read_tokens: int = 0
    provider_cache_write_tokens: int = 0
    request_id: str = ""
    emitted_ids: tuple[int, ...] | None = None
    native_tape_json: str | None = None
    raw_output_complete: bool = True

    def output_bytes(self) -> bytes:
        # The raw meter convention includes exposed reasoning and exact generated
        # tool name/argument bytes, without inventing sampled fixed JSON syntax.
        return (
            self.reasoning
            + self.content
            + b"".join(call.name.encode() + call.arguments_json.encode() for call in self.calls)
        )


class Teacher(Protocol):
    model: str

    def respond(self, request: TeacherRequest) -> TeacherResponse: ...


class OpenAICompatibleTeacher:
    """Explicit HTTP client for local/server-hosted teachers; never called at import.

    The endpoint must be a Chat Completions URL. No credentials are stored in
    manifests. We meter raw pinned-codec input/output independently of provider
    usage (which may use different tokens or include hidden reasoning).
    """

    def __init__(
        self,
        endpoint: str,
        model: str,
        codec: Codec,
        *,
        token_env: str | None = None,
        timeout: float = 60.0,
        include_reasoning: bool = False,
    ):
        parsed = urllib.parse.urlparse(endpoint)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.username
            or parsed.password
        ):
            raise ValueError("Teacher endpoint must be an explicit HTTP(S) URL without credentials")
        if parsed.scheme == "http" and parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("Remote teacher endpoints require HTTPS")
        self.endpoint, self.model, self.codec = endpoint, model, codec
        self.token_env, self.timeout, self.include_reasoning = token_env, timeout, include_reasoning

    def respond(self, request: TeacherRequest) -> TeacherResponse:
        if request.codec_fingerprint != self.codec.fingerprint:
            raise ValueError("Teacher request uses an incompatible raw-meter codec")
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": json.loads(request.messages_json),
            "max_tokens": request.max_output_tokens,
            "temperature": 1.0,
            "stream": False,
        }
        schemas = json.loads(request.tools_json)
        if schemas:
            # Support the compact shared environment schema and native OpenAI schemas.
            payload["tools"] = [
                schema
                if schema.get("type") == "function"
                else {"type": "function", "function": schema}
                for schema in schemas
            ]
        headers = {"Content-Type": "application/json", "Idempotency-Key": request.request_id}
        if self.token_env:
            token = os.environ.get(self.token_env)
            if not token:
                raise ValueError(f"Teacher API token is absent from {self.token_env}")
            headers["Authorization"] = f"Bearer {token}"
        call = urllib.request.Request(
            self.endpoint, data=canonical_json(payload).encode(), headers=headers, method="POST"
        )
        with urllib.request.urlopen(call, timeout=self.timeout) as result:
            raw = result.read((32 << 20) + 1)
        if len(raw) > 32 << 20:
            raise ValueError("Teacher response exceeds the explicit 32-MiB transport limit")
        obj = json.loads(raw)
        choice = obj["choices"][0]
        message = choice["message"]
        usage = obj.get("usage", {})
        calls = tuple(
            TeacherCall(call["id"], call["function"]["name"], call["function"]["arguments"])
            for call in message.get("tool_calls", ())
        )
        reasoning = (message.get("reasoning_content") or "").encode()
        hidden_reasoning = int(
            usage.get("completion_tokens_details", {}).get("reasoning_tokens", 0)
        )
        # Returned reasoning is metered even when the target policy omits it.
        return TeacherResponse(
            content=(message.get("content") or "").encode(),
            calls=calls,
            finish_reason=choice.get("finish_reason", "stop"),
            reasoning=reasoning,
            provider_input_tokens=int(usage.get("prompt_tokens", 0)),
            provider_output_tokens=int(usage.get("completion_tokens", 0)),
            provider_cache_read_tokens=int(
                usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
            ),
            provider_cache_write_tokens=int(usage.get("cache_creation_input_tokens", 0)),
            request_id=obj.get("id", ""),
            raw_output_complete=not hidden_reasoning or bool(reasoning),
        )


@dataclass(frozen=True, slots=True)
class CollectionConfig:
    distinct_tasks: int = _PILOT["distinct_tasks"]
    attempts_per_task: int = _PILOT["attempts_per_task"]
    teacher_output_limit: int = _PILOT["teacher_output_T_limit"]
    max_output_per_request: int = 4096
    max_requests_per_attempt: int = 64
    max_record_bytes: int = 16384
    max_record_tokens: int = 1024
    chunking: ChunkingConfig = ChunkingConfig()
    include_reasoning: bool = False
    # Fractions are by independent task seeds, not rows or token length.
    seed_mix: tuple[tuple[str, float], ...] = tuple(_PILOT["seed_mix"].items())

    def __post_init__(self) -> None:
        if not isinstance(self.chunking, ChunkingConfig):
            raise TypeError("Collection chunking must be a ChunkingConfig")
        if (
            min(
                self.distinct_tasks,
                self.attempts_per_task,
                self.teacher_output_limit,
                self.max_output_per_request,
                self.max_requests_per_attempt,
                self.max_record_bytes,
                self.max_record_tokens,
            )
            <= 0
        ):
            raise ValueError("Collection limits must be positive")
        if abs(sum(weight for _, weight in self.seed_mix) - 1.0) > 1e-10 or any(
            weight < 0 for _, weight in self.seed_mix
        ):
            raise ValueError("Seed mix must be a probability distribution")

    @property
    def chunking_policy(self) -> ChunkingConfig:
        return self.chunking.with_hard_limit(
            self.max_record_tokens, max_bytes=self.max_record_bytes
        )


class CollectionLedger:
    """WAL journal with atomic task/attempt/output-token reservations."""

    def __init__(self, path: str | Path, config: CollectionConfig | None = None):
        config = config or CollectionConfig()
        self.path, self.config = Path(path), config
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path, timeout=60)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS tasks(task_key TEXT PRIMARY KEY,seed_id TEXT NOT NULL,
                family_id TEXT NOT NULL,category TEXT NOT NULL,split TEXT NOT NULL,seed_sha256 TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS attempts(attempt_id TEXT PRIMARY KEY,task_key TEXT NOT NULL,
                attempt_index INTEGER NOT NULL,status TEXT NOT NULL,accepted INTEGER NOT NULL DEFAULT 0,
                episode_json TEXT,student_snapshot TEXT,UNIQUE(task_key,attempt_index));
            CREATE TABLE IF NOT EXISTS requests(request_id TEXT PRIMARY KEY,attempt_id TEXT NOT NULL,
                reserved_output INTEGER NOT NULL,raw_input INTEGER NOT NULL,status TEXT NOT NULL,
                raw_output INTEGER NOT NULL DEFAULT 0,provider_input INTEGER NOT NULL DEFAULT 0,
                provider_output INTEGER NOT NULL DEFAULT 0,cache_read INTEGER NOT NULL DEFAULT 0,
                cache_write INTEGER NOT NULL DEFAULT 0,response_json TEXT);
            CREATE TABLE IF NOT EXISTS executions(action_id TEXT PRIMARY KEY,attempt_id TEXT NOT NULL,
                input_state TEXT NOT NULL,action_sha256 TEXT NOT NULL,output_state TEXT NOT NULL,
                execution_id TEXT NOT NULL,observation_sha256 TEXT NOT NULL,provenance TEXT NOT NULL);
        """)
        identity = canonical_json(
            asdict(config) | {"chunking_fingerprint": config.chunking_policy.fingerprint}
        )
        row = self.connection.execute("SELECT value FROM meta WHERE key='config'").fetchone()
        if row and row[0] != identity:
            raise ValueError(
                "Collection budget/config changed; use a new explicitly budgeted ledger"
            )
        self.connection.execute("INSERT OR IGNORE INTO meta VALUES('config',?)", (identity,))
        self.connection.commit()

    def _begin(self) -> None:
        self.connection.execute("BEGIN IMMEDIATE")

    def remaining_output(self) -> int:
        row = self.connection.execute(
            "SELECT COALESCE(SUM(CASE WHEN status='complete' THEN raw_output ELSE MAX(reserved_output,raw_output) END),0) FROM requests"
        ).fetchone()
        return max(0, self.config.teacher_output_limit - row[0])

    def begin_attempt(
        self, seed: TaskSeed, *, split: str, student_snapshot: str | None = None
    ) -> str:
        if split == "quarantine":
            raise ValueError("Benchmark-quarantined tasks cannot enter teacher collection")
        self._begin()
        try:
            task = self.connection.execute(
                "SELECT * FROM tasks WHERE task_key=?", (seed.task_key,)
            ).fetchone()
            seed_hash = content_hash(
                {
                    "environment": seed.environment_json,
                    "tools": seed.tools_json,
                    "policy": base64.b64encode(seed.policy).decode(),
                    "revision": seed.revision,
                }
            )
            if task is None:
                count = self.connection.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
                if count >= self.config.distinct_tasks:
                    raise CollectionLimit("Distinct-task budget is exhausted")
                shares = dict(self.config.seed_mix)
                if seed.category not in shares:
                    raise ValueError("Seed category is absent from the declared pilot mix")
                category_count = self.connection.execute(
                    "SELECT COUNT(*) FROM tasks WHERE category=?", (seed.category,)
                ).fetchone()[0]
                # Largest-remainder quotas are deterministic even in tiny tests.
                quotas = {
                    key: int(share * self.config.distinct_tasks) for key, share in shares.items()
                }
                remainder = self.config.distinct_tasks - sum(quotas.values())
                for key, _ in sorted(
                    shares.items(),
                    key=lambda item: (-(item[1] * self.config.distinct_tasks % 1), item[0]),
                )[:remainder]:
                    quotas[key] += 1
                if category_count >= quotas[seed.category]:
                    raise CollectionLimit(f"Pilot category quota is full: {seed.category}")
                self.connection.execute(
                    "INSERT INTO tasks VALUES(?,?,?,?,?,?)",
                    (seed.task_key, seed.seed_id, seed.family_id, seed.category, split, seed_hash),
                )
            elif task["seed_sha256"] != seed_hash or task["split"] != split:
                raise ValueError("An existing task seed changed its environment/tools/source/split")
            existing = self.connection.execute(
                "SELECT COUNT(*),MAX(accepted) FROM attempts WHERE task_key=?", (seed.task_key,)
            ).fetchone()
            if existing[1]:
                raise CollectionLimit("This task already has an accepted episode")
            attempt_index = existing[0]
            if attempt_index >= self.config.attempts_per_task:
                raise CollectionLimit(
                    "Per-task attempt budget is exhausted (including interrupted/rejected attempts)"
                )
            if self.connection.execute(
                "SELECT 1 FROM attempts WHERE task_key=? AND status='running'", (seed.task_key,)
            ).fetchone():
                raise ValueError(
                    "An attempt is already active; explicitly recover an interrupted worker before retrying"
                )
            attempt_id = stable_id("collection-attempt-v1", seed.task_key, str(attempt_index))
            self.connection.execute(
                "INSERT INTO attempts(attempt_id,task_key,attempt_index,status,student_snapshot) VALUES(?,?,?,'running',?)",
                (attempt_id, seed.task_key, attempt_index, student_snapshot),
            )
            self.connection.commit()
            return attempt_id
        except BaseException:
            self.connection.rollback()
            raise

    def reserve(
        self, attempt_id: str, request_id: str, *, output_limit: int, raw_input: int
    ) -> None:
        if min(output_limit, raw_input) < 0 or output_limit == 0:
            raise ValueError(
                "Request usage must be nonnegative, with a positive output reservation"
            )
        self._begin()
        try:
            if self.connection.execute(
                "SELECT 1 FROM requests WHERE request_id=?", (request_id,)
            ).fetchone():
                raise ValueError(
                    "A request ID cannot be submitted again; reconcile the durable journal"
                )
            if self.remaining_output() < output_limit:
                raise CollectionLimit("Teacher output budget lacks a complete request reservation")
            row = self.connection.execute(
                "SELECT status FROM attempts WHERE attempt_id=?", (attempt_id,)
            ).fetchone()
            if row is None or row[0] != "running":
                raise ValueError("Teacher requests require a running attempt")
            self.connection.execute(
                "INSERT INTO requests(request_id,attempt_id,reserved_output,raw_input,status) VALUES(?,?,?,?,'submitted')",
                (request_id, attempt_id, output_limit, raw_input),
            )
            self.connection.commit()
        except BaseException:
            self.connection.rollback()
            raise

    def settle(self, request_id: str, response: TeacherResponse, *, raw_output: int) -> None:
        if (
            min(
                raw_output,
                response.provider_input_tokens,
                response.provider_output_tokens,
                response.provider_cache_read_tokens,
                response.provider_cache_write_tokens,
            )
            < 0
        ):
            raise ValueError("Teacher usage cannot be negative")
        self._begin()
        try:
            row = self.connection.execute(
                "SELECT * FROM requests WHERE request_id=?", (request_id,)
            ).fetchone()
            if row is None or row["status"] not in {"submitted", "usage_incomplete"}:
                raise ValueError("Only an outstanding reserved request can be settled")
            payload = asdict(response)
            payload["content"] = base64.b64encode(response.content).decode()
            payload["reasoning"] = base64.b64encode(response.reasoning).decode()
            self.connection.execute(
                "UPDATE requests SET status=?,raw_output=?,provider_input=?,provider_output=?,cache_read=?,cache_write=?,response_json=? WHERE request_id=?",
                (
                    "complete" if response.raw_output_complete else "usage_incomplete",
                    raw_output,
                    response.provider_input_tokens,
                    response.provider_output_tokens,
                    response.provider_cache_read_tokens,
                    response.provider_cache_write_tokens,
                    canonical_json(payload),
                    request_id,
                ),
            )
            self.connection.commit()
        except BaseException:
            self.connection.rollback()
            raise
        if raw_output > row["reserved_output"]:
            # Actual usage is never clipped to make the accounting look compliant.
            raise CollectionLimit(
                "Teacher exceeded the reserved raw-codec output limit; actual usage was charged and collection stopped"
            )
        if not response.raw_output_complete:
            raise RuntimeError(
                "Teacher reports hidden output without complete raw bytes; its reservation remains charged until usage is reconciled"
            )

    def record_execution(
        self,
        attempt_id: str,
        action_id: str,
        payload: bytes,
        input_state: str,
        result: EnvironmentResult,
    ) -> None:
        if not result.execution_id or result.state_id == input_state:
            raise ValueError(
                "An executed/simulated step needs a fresh state identity and execution receipt"
            )
        with self.connection:
            self.connection.execute(
                "INSERT INTO executions VALUES(?,?,?,?,?,?,?,?)",
                (
                    action_id,
                    attempt_id,
                    input_state,
                    digest_bytes(payload),
                    result.state_id,
                    result.execution_id,
                    digest_bytes(result.observation),
                    result.provenance,
                ),
            )

    def finish(self, attempt_id: str, episode: Episode, *, accepted: bool) -> None:
        with self.connection:
            changed = self.connection.execute(
                "UPDATE attempts SET status='complete',accepted=?,episode_json=? WHERE attempt_id=? AND status='running'",
                (int(accepted), canonical_json(episode.to_dict()), attempt_id),
            ).rowcount
            if changed != 1:
                raise ValueError("Attempt is missing or has already completed")

    def recover_interrupted(self, attempt_id: str) -> None:
        """Explicit worker recovery; unknown request output retains its full debit."""
        with self.connection:
            self.connection.execute(
                "UPDATE attempts SET status='interrupted' WHERE attempt_id=? AND status='running'",
                (attempt_id,),
            )

    def meters(self) -> dict[str, Any]:
        row = self.connection.execute("""SELECT
            COALESCE(SUM(raw_output),0) AS T,
            COALESCE(SUM(raw_input),0) AS teacher_input,
            COALESCE(SUM(CASE WHEN status!='complete' THEN MAX(0,reserved_output-raw_output) ELSE 0 END),0) AS unresolved_output_reservations,
            COALESCE(SUM(provider_input),0) AS provider_input_tokens,
            COALESCE(SUM(provider_output),0) AS provider_output_tokens,
            COALESCE(SUM(cache_read),0) AS provider_cache_read_tokens,
            COALESCE(SUM(cache_write),0) AS provider_cache_write_tokens FROM requests""").fetchone()
        result = dict(row)
        result["accepted_episodes"] = self.connection.execute(
            "SELECT COUNT(*) FROM attempts WHERE accepted=1"
        ).fetchone()[0]
        result["attempts"] = self.connection.execute("SELECT COUNT(*) FROM attempts").fetchone()[0]
        result["distinct_tasks"] = self.connection.execute("SELECT COUNT(*) FROM tasks").fetchone()[
            0
        ]
        result["remaining_output_budget"] = self.remaining_output()
        result["T_complete"] = not self.connection.execute(
            "SELECT 1 FROM requests WHERE status!='complete' LIMIT 1"
        ).fetchone()
        result["T_includes_rejected"] = True
        result["usage_convention"] = (
            "T/raw teacher input use the pinned raw codec; provider/cache counts retain provider units. Unresolved requests debit full reservations."
        )
        return result

    def export(self, directory: str | Path) -> Path:
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        sources = []
        for accepted, name in ((True, "accepted"), (False, "failed")):
            path = root / f"{name}.jsonl"
            with atomic_binary_writer(path) as handle:
                for row in self.connection.execute(
                    "SELECT episode_json FROM attempts WHERE episode_json IS NOT NULL AND accepted=? ORDER BY task_key,attempt_index",
                    (int(accepted),),
                ):
                    handle.write(row[0].encode() + b"\n")
            source = pin_local_source(
                path,
                source_id=f"fresh-{name}",
                adapter="native",
                stream="mixed_collection",
                terms=TermsReview(
                    approved=True,
                    license="operator-collected",
                    reviewed_by="collection-operator",
                    reviewed_at=time.strftime("%Y-%m-%d", time.gmtime()),
                    notes="Environment, task-source and teacher use approvals belong in the upstream collection configuration.",
                ),
            )
            sources.append(replace(source, options={"preserve_native_stream": True}))
        write_json(root / "collection_census.json", self.meters())
        manifest = root / "sources.json"
        write_source_manifest(manifest, tuple(sources))
        return manifest

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> CollectionLedger:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


class Collector:
    def __init__(
        self,
        teacher: Teacher,
        environment_factory: Callable[[], Environment] | None,
        codec: Codec,
        ledger: CollectionLedger,
        family_registry: DedupStore,
        config: CollectionConfig | None = None,
        artifact_verifier: Callable[[bytes], EnvironmentResult] | None = None,
    ):
        self.teacher, self.environment_factory, self.codec = teacher, environment_factory, codec
        self.ledger, self.family_registry = ledger, family_registry
        self.config = config or ledger.config
        if self.config != ledger.config:
            raise ValueError("Collector and durable ledger must use the same budget configuration")
        if not family_registry.frozen:
            raise ValueError("Source/task families must be split before descendants are generated")
        with ledger.connection:
            for key, value in (
                ("codec_fingerprint", codec.fingerprint),
                ("teacher_model", teacher.model),
            ):
                existing = ledger.connection.execute(
                    "SELECT value FROM meta WHERE key=?", (key,)
                ).fetchone()
                if existing is not None and existing[0] != value:
                    raise ValueError("Collection ledger codec or teacher identity changed")
                ledger.connection.execute("INSERT OR IGNORE INTO meta VALUES(?,?)", (key, value))
        self.artifact_verifier = artifact_verifier

    def collect(
        self,
        seed: TaskSeed,
        *,
        student: Teacher | None = None,
        student_snapshot: str | None = None,
        student_prefix_steps: int = 0,
    ) -> Episode:
        if (
            student_prefix_steps < 0
            or student_prefix_steps
            and (student is None or not student_snapshot)
        ):
            raise ValueError(
                "Student-state collection needs a frozen snapshot and an actual student actor"
            )
        keys = (seed.family_id, *seed.ancestor_ids)
        splits = {self.family_registry.split_of(key) for key in keys}
        if len(splits) != 1:
            raise ValueError("Seed ancestors disagree about their frozen split")
        split = next(iter(splits))
        attempt_id = self.ledger.begin_attempt(seed, split=split, student_snapshot=student_snapshot)
        records: list[PhysicalRecord] = []
        actions: list[LogicalAction] = []
        messages: list[dict[str, Any]] = []
        dependencies: list[Dependency] = []
        turn = 0
        environment: Environment | None = None
        result: EnvironmentResult | None = None
        last_observation: str | None = None
        accepted = False
        native_tapes: list[dict[str, Any]] = []

        def append(
            payload: bytes,
            role: str,
            *,
            score: bool = False,
            provenance: str = "natural",
            action_id: str | None = None,
            result_of: str | None = None,
            execution_id: str | None = None,
            emitted_ids: tuple[int, ...] | None = None,
        ) -> tuple[PhysicalRecord, ...]:
            nonlocal turn
            values = make_records(
                payload,
                logical_id=f"{attempt_id}:{turn}",
                role=role,
                turn_index=turn,
                policy_mask=score,
                free_spans=tool_free_spans(
                    payload, seed.tools_json, max_tokens=self.config.max_record_tokens
                )
                if action_id
                else None,
                provenance=provenance,
                max_bytes=self.config.max_record_bytes
                if emitted_ids is None
                else max(1, len(payload)),
                source_uri=f"collection:{attempt_id}",
                source_field=f"turns[{turn}]",
                action_id=action_id,
                result_of=result_of,
                execution_id=execution_id,
                codec=self.codec if emitted_ids is None else None,
                max_tokens=self.config.max_record_tokens if emitted_ids is None else None,
                chunking=self.config.chunking_policy if emitted_ids is None else None,
            )
            if emitted_ids is not None:
                if self.codec.decode(emitted_ids) != payload:
                    raise ValueError("Native student IDs do not decode to the executed bytes")
                if (
                    len(emitted_ids) > self.config.chunking_policy.hard_max_tokens
                    or len(payload) > self.config.max_record_bytes
                ):
                    raise ValueError(
                        "A native generated action exceeds the configured physical record capacity"
                    )
                values = (
                    replace(
                        values[0], emitted_ids=emitted_ids, codec_fingerprint=self.codec.fingerprint
                    ),
                )
            records.extend(values)
            turn += 1
            return values

        try:
            if seed.policy:
                append(seed.policy, "system")
                messages.append({"role": "system", "content": seed.policy.decode("utf-8")})
            append(seed.prompt, "user")
            messages.append({"role": "user", "content": seed.prompt.decode("utf-8")})
            if seed.mode != "artifact":
                if self.environment_factory is None:
                    raise ValueError(
                        "Executed collection requires an explicitly injected environment"
                    )
                environment = self.environment_factory()
                result = environment.reset(json.loads(seed.environment_json))
                if result.provenance != ("simulated" if seed.mode == "simulated" else "executed"):
                    raise ValueError(
                        "The seed's declared execution provenance differs from its actual environment"
                    )
                values = append(result.observation, "environment", provenance=result.provenance)
                last_observation = values[0].record_id
                messages.append({"role": "user", "content": result.observation.decode("utf-8")})
            for request_index in range(self.config.max_requests_per_attempt + student_prefix_steps):
                is_student = request_index < student_prefix_steps
                actor = student if is_student else self.teacher
                assert actor is not None
                output_limit = self.config.max_output_per_request
                if not is_student:
                    output_limit = min(output_limit, self.ledger.remaining_output())
                    if output_limit <= 0:
                        raise CollectionLimit("Teacher output budget is exhausted")
                request_id = stable_id(
                    "teacher-request-v1",
                    attempt_id,
                    str(request_index),
                    "student" if is_student else "teacher",
                )
                request = TeacherRequest(
                    request_id=request_id,
                    messages_json=canonical_json(messages),
                    tools_json=seed.tools_json,
                    max_output_tokens=output_limit,
                    codec_fingerprint=self.codec.fingerprint,
                    session_id=attempt_id,
                    records=tuple(records),
                    control_seed=seed.policy,
                )
                raw_input = len(
                    self.codec.encode((request.messages_json + request.tools_json).encode())
                )
                if not is_student:
                    self.ledger.reserve(
                        attempt_id, request_id, output_limit=output_limit, raw_input=raw_input
                    )
                response = actor.respond(request)
                native_tape = None
                if response.native_tape_json is not None:
                    from megatoken.generation.structure import ActionTape

                    native_tape = ActionTape.from_json(response.native_tape_json)
                    if native_tape.codec_fingerprint != self.codec.fingerprint:
                        raise ValueError("Native action tape and collection meter codecs differ")
                    native_tapes.append(
                        {
                            "request_id": request_id,
                            "provenance": "student" if is_student else "teacher",
                            # A rejected decode advances latent state without
                            # producing an executable transcript record. This
                            # cursor retains its position among physical events.
                            "record_cursor": len(records),
                            "tape": json.loads(response.native_tape_json),
                        }
                    )
                if not is_student:
                    self.ledger.settle(
                        request_id,
                        response,
                        raw_output=len(
                            self.codec.encode(
                                native_tape.payload
                                if native_tape is not None
                                else response.output_bytes()
                            )
                        ),
                    )
                if native_tape is not None and not native_tape.valid:
                    # The actor already advanced its recurrent state. No tool
                    # execution or invented observation follows an invalid decode.
                    continue
                provenance = "student" if is_student else "teacher"
                scored = not is_student
                visible = (
                    response.reasoning if self.config.include_reasoning else b""
                ) + response.content
                if visible:
                    values = append(
                        visible,
                        "assistant",
                        score=scored,
                        provenance=provenance,
                        emitted_ids=response.emitted_ids,
                    )
                    if scored and last_observation:
                        dependencies.append(
                            Dependency(last_observation, values[-1].record_id, "environment_state")
                        )
                native_calls = [
                    {
                        "id": call.call_id,
                        "type": "function",
                        "function": {"name": call.name, "arguments": call.arguments_json},
                    }
                    for call in response.calls
                ]
                message: dict[str, Any] = {"role": "assistant", "content": visible.decode("utf-8")}
                if native_calls:
                    message["tool_calls"] = native_calls
                messages.append(message)
                if response.calls:
                    if environment is None or result is None:
                        raise ValueError(
                            "Artifact generation emitted an unavailable environment action"
                        )
                    pending = []
                    # All selected calls precede every result from this teacher draw.
                    for call in response.calls:
                        payload = call.action_bytes
                        action_id = stable_id(
                            "collected-action-v1", request_id, call.call_id, payload
                        )
                        values = append(
                            payload,
                            "assistant",
                            score=scored,
                            provenance=provenance,
                            action_id=action_id,
                            emitted_ids=call.emitted_ids,
                        )
                        if scored and last_observation:
                            dependencies.append(
                                Dependency(
                                    last_observation, values[-1].record_id, "environment_state"
                                )
                            )
                        pending.append((call, payload, action_id, values))
                    for call, payload, action_id, values in pending:
                        input_state = result.state_id
                        # Every changed student/teacher action is executed against
                        # its current state. There is no prerecorded-result input.
                        result = environment.step(
                            payload, action_id=action_id, expected_state_id=input_state
                        )
                        self.ledger.record_execution(
                            attempt_id, action_id, payload, input_state, result
                        )
                        actions.append(
                            LogicalAction(
                                action_id=action_id,
                                call_id=f"{request_id}:{call.call_id}",
                                tool_name=call.name,
                                arguments_json=call.arguments_json,
                                record_ids=tuple(record.record_id for record in values),
                                serialized_sha256=digest_bytes(payload),
                                execution_id=result.execution_id,
                                schema_id=digest_bytes(seed.tools_json.encode()),
                            )
                        )
                        observation = append(
                            result.observation,
                            "tool",
                            provenance=result.provenance,
                            result_of=action_id,
                            execution_id=result.execution_id,
                        )
                        last_observation = observation[0].record_id
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": call.call_id,
                                "content": result.observation.decode("utf-8"),
                            }
                        )
                    if result.done and is_student:
                        raise ValueError(
                            "Student terminated before the correction point; collect this task as successful replay"
                        )
                    if result.done:
                        break
                elif seed.mode == "artifact" and response.finish_reason == "length":
                    continuation = b"Continue the same artifact from its exact existing ending. Do not restart."
                    append(continuation, "user")
                    messages.append({"role": "user", "content": continuation.decode()})
                elif is_student:
                    # A student's natural-language finish still gets teacher
                    # correction from that exact state, without a gold outcome.
                    continue
                else:
                    break
            if environment is not None:
                verdict = environment.verify()
            elif self.artifact_verifier is not None:
                verdict = self.artifact_verifier(
                    b"".join(record.payload for record in records if record.role == "assistant")
                )
            else:
                verdict = EnvironmentResult(
                    b"", "unverified-artifact", done=True, verified=False, provenance="unknown"
                )
            expected_provenance = "simulated" if seed.mode == "simulated" else "executed"
            accepted = (
                verdict.verified
                and verdict.reward > 0
                and verdict.provenance == expected_provenance
                and any(record.policy_mask for record in records)
            )
            outcome = "success" if accepted else "failure" if verdict.verified else "unknown"
            verification = (
                ("simulation_judge" if verdict.provenance == "simulated" else "external_tests")
                if verdict.verified
                else "none"
            )
            supervised = Supervision(
                outcome=outcome,
                verification=verification,
                reward=verdict.reward,
                annotations_json=canonical_json(
                    {
                        "split": split,
                        "teacher": self.teacher.model,
                        "student_snapshot": student_snapshot,
                        "student_prefix_steps": student_prefix_steps,
                        "collection_mode": seed.mode,
                        "attempt_id": attempt_id,
                        "task_key": seed.task_key,
                        "native_action_tapes": native_tapes,
                    }
                ),
            )
            episode = Episode(
                episode_id=attempt_id,
                records=tuple(records),
                lineage=Lineage(
                    source_id=f"fresh:{self.teacher.model}",
                    revision=seed.revision,
                    family_id=seed.family_id,
                    ancestor_ids=seed.ancestor_ids,
                    parent_episode_id=seed.seed_id,
                    source_uri=f"collection:{attempt_id}",
                    frozen_split=split,
                ),
                actions=tuple(actions),
                supervision=supervised,
                dependencies=tuple(dependencies),
                stream="simulated_trajectories"
                if seed.mode == "simulated"
                else "student_state_corrections"
                if student_prefix_steps
                else "verified_trajectories"
                if accepted
                else "recovery_clarification",
                tools_json=seed.tools_json,
                control_seed=seed.policy,
            )
            if not accepted:
                episode = apply_recovery_policy(episode)
            self.ledger.finish(attempt_id, episode, accepted=accepted)
            return episode
        except BaseException:
            # No speculative retry: outstanding network completions continue to
            # debit their maximum reservation and interrupted attempts still count.
            self.ledger.recover_interrupted(attempt_id)
            raise

    def run_pilot(
        self,
        seeds: Iterable[TaskSeed],
        *,
        student: Teacher | None = None,
        student_snapshot: str | None = None,
        student_prefix_steps: int = 0,
    ) -> Iterable[Episode]:
        for seed in seeds:
            for _ in range(self.config.attempts_per_task):
                if not self.ledger.remaining_output():
                    return
                try:
                    episode = self.collect(
                        seed,
                        student=student,
                        student_snapshot=student_snapshot,
                        student_prefix_steps=student_prefix_steps,
                    )
                except CollectionLimit:
                    break
                yield episode
                if episode.supervision.verified_success:
                    break
