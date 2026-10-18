"""Versioned executed environments shared by collection, evaluation, and RL.

Model-produced tool calls are data. They are never passed to a host shell. Real
repository/build/database tasks use an explicitly configured isolated service;
the included fixture performs actual, verifiable edits in an in-memory file map.
"""

from __future__ import annotations

import base64
import json
import math
import os
import urllib.parse
import urllib.request
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Protocol

from megatoken.io import canonical_json, content_hash


@dataclass(frozen=True)
class EnvironmentResult:
    observation: bytes
    state_id: str
    execution_id: str | None = None
    done: bool = False
    reward: float = 0.0
    verified: bool = False
    provenance: str = "executed"

    def __post_init__(self) -> None:
        if not isinstance(self.observation, bytes) or not self.state_id:
            raise ValueError("Environment observations need exact bytes and a state identity")
        if not math.isfinite(self.reward):
            raise ValueError("Environment reward must be finite")


class Environment(Protocol):
    def reset(self, task: Mapping[str, Any]) -> EnvironmentResult: ...
    def step(
        self, action: bytes, *, action_id: str, expected_state_id: str
    ) -> EnvironmentResult: ...
    def verify(self) -> EnvironmentResult: ...


FIXTURE_TOOLS = [
    {
        "name": "list",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "read",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
            "additionalProperties": False,
        },
    },
    {
        "name": "write",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
            "additionalProperties": False,
        },
    },
    {
        "name": "finish",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    },
]


def _safe_key(value: Any) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError("path must be a nonempty string without NUL")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value:
        raise ValueError("path must be relative to the environment root")
    return str(path)


class FixtureEnvironment:
    """Small reproducible read→edit→verify tasks; useful for pipeline gates."""

    def __init__(self) -> None:
        self.files: dict[str, str] = {}
        self.expected: dict[str, str] = {}
        self.session = ""
        self.state_id = ""
        self.sequence = 0
        self.done = False
        self._responses: dict[str, tuple[str, EnvironmentResult]] = {}

    def _version(self) -> str:
        return content_hash(
            {
                "session": self.session,
                "sequence": self.sequence,
                "files": self.files,
                "done": self.done,
            }
        )

    def reset(self, task: Mapping[str, Any]) -> EnvironmentResult:
        self.session = uuid.uuid4().hex
        self.sequence = 0
        self.done = False
        self._responses.clear()
        self.files = {
            _safe_key(path): str(value)
            for path, value in task.get("initial_files", {"answer.txt": "red"}).items()
        }
        self.expected = {
            _safe_key(path): str(value)
            for path, value in task.get("expected_files", {"answer.txt": "blue"}).items()
        }
        self.state_id = self._version()
        public = {
            "task": task.get("prompt", "Change answer.txt from red to blue, then finish."),
            "paths": sorted(self.files),
            "tools": FIXTURE_TOOLS,
        }
        return EnvironmentResult(canonical_json(public), self.state_id)

    def step(self, action: bytes, *, action_id: str, expected_state_id: str) -> EnvironmentResult:
        if not self.session:
            raise RuntimeError("reset must precede step")
        if not action_id:
            raise ValueError("A stable action_id is required")
        identity = content_hash(
            {"action": base64.b64encode(action).decode(), "state": expected_state_id}
        )
        if action_id in self._responses:
            previous_identity, previous = self._responses[action_id]
            if previous_identity != identity:
                raise ValueError(
                    "An action_id cannot be reused for different bytes or source state"
                )
            return previous
        if expected_state_id != self.state_id:
            raise ValueError("Action was chosen from a stale environment state")
        if self.done:
            raise ValueError("The environment has already terminated")
        error = None
        try:
            from megatoken.tool_schema import parse_tool_action

            name, arguments, _, _ = parse_tool_action(action)
            tool = next((item for item in FIXTURE_TOOLS if item["name"] == name), None)
            if tool is None:
                raise ValueError("Unknown tool")
            import jsonschema

            jsonschema.validate(arguments, tool["parameters"])
            if name == "list":
                output: Any = {"paths": sorted(self.files)}
            elif name == "read":
                key = _safe_key(arguments["path"])
                output = {"path": key, "content": self.files[key]}
            elif name == "write":
                key = _safe_key(arguments["path"])
                self.files[key] = arguments["content"]
                output = {"written": key}
            else:
                self.done = True
                output = {"submitted": True}
        except (ValueError, KeyError, TypeError, UnicodeError) as exc:
            error = str(exc)
            output = {"error": error}
        except Exception as exc:
            # JSON Schema errors are public validation feedback, never Python
            # tracebacks or hidden verifier contents.
            import jsonschema

            if not isinstance(exc, jsonschema.ValidationError):
                raise
            error = exc.message
            output = {"error": error}
        self.sequence += 1
        self.state_id = self._version()
        success = self.files == self.expected
        result = EnvironmentResult(
            canonical_json(output),
            self.state_id,
            execution_id=f"{self.session}:{self.sequence}",
            done=self.done,
            reward=float(success) if self.done else 0.0,
            verified=self.done,
        )
        self._responses[action_id] = identity, result
        return result

    def verify(self) -> EnvironmentResult:
        if not self.session:
            raise RuntimeError("reset must precede verify")
        # Outcome stays in supervision metadata. This observation carries no
        # hidden expected answer, pass/fail label, or numeric reward.
        return EnvironmentResult(
            b'{"verification_complete":true}',
            self.state_id,
            done=True,
            reward=float(self.files == self.expected),
            verified=True,
        )


class HTTPEnvironment:
    """Client for an explicitly selected isolated environment service.

    POST /reset {task}; POST /step {session_id, action_base64, action_id,
    expected_state_id}; POST /verify {session_id}. Every response contains
    observation_base64/state_id; reset also returns session_id. A service must
    enforce action idempotency and state leases before executing any action.
    """

    def __init__(
        self, endpoint: str, *, token_env: str | None = None, timeout: float = 60.0
    ) -> None:
        parsed = urllib.parse.urlparse(endpoint)
        if (
            parsed.scheme not in {"https", "http"}
            or not parsed.netloc
            or parsed.username
            or parsed.password
        ):
            raise ValueError(
                "An HTTP(S) environment endpoint without embedded credentials is required"
            )
        if parsed.scheme == "http" and parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("Remote environments must use HTTPS")
        self.endpoint, self.timeout, self.token_env = endpoint.rstrip("/"), timeout, token_env
        self.session_id: str | None = None

    def _post(self, path: str, value: dict[str, Any]) -> tuple[EnvironmentResult, dict[str, Any]]:
        headers = {"Content-Type": "application/json"}
        if self.token_env:
            token = os.environ.get(self.token_env)
            if not token:
                raise ValueError(f"Missing environment service token in {self.token_env}")
            headers["Authorization"] = f"Bearer {token}"
        request = urllib.request.Request(
            self.endpoint + path, data=canonical_json(value), headers=headers, method="POST"
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            payload = response.read(64 << 20)
            if len(payload) >= 64 << 20:
                raise ValueError("Environment response exceeds the 64-MiB observation limit")
        decoded = json.loads(payload)
        result = EnvironmentResult(
            base64.b64decode(decoded["observation_base64"], validate=True),
            state_id=decoded["state_id"],
            execution_id=decoded.get("execution_id"),
            done=bool(decoded.get("done", False)),
            reward=float(decoded.get("reward", 0.0)),
            verified=bool(decoded.get("verified", False)),
            provenance=decoded.get("provenance", "executed"),
        )
        if result.provenance != "executed":
            raise ValueError("Executed-task training requires an executed environment response")
        return result, decoded

    def reset(self, task: Mapping[str, Any]) -> EnvironmentResult:
        result, value = self._post("/reset", {"task": dict(task)})
        self.session_id = value["session_id"]
        return result

    def step(self, action: bytes, *, action_id: str, expected_state_id: str) -> EnvironmentResult:
        if self.session_id is None:
            raise RuntimeError("reset must precede step")
        result, _ = self._post(
            "/step",
            {
                "session_id": self.session_id,
                "action_base64": base64.b64encode(action).decode(),
                "action_id": action_id,
                "expected_state_id": expected_state_id,
            },
        )
        return result

    def verify(self) -> EnvironmentResult:
        if self.session_id is None:
            raise RuntimeError("reset must precede verify")
        result, _ = self._post("/verify", {"session_id": self.session_id})
        return result


def environment_factory(endpoint: str):
    if endpoint == "fixture":
        return FixtureEnvironment
    return lambda: HTTPEnvironment(endpoint, token_env="MEGATOKEN_ENV_TOKEN")
