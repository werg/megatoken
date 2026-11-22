"""Exact, checkpoint-bound recovery of a durably acknowledged native commit.

This bridge never saves a speculative selection as reusable model state. A
completed commit is recoverable only by reconstructing its entire adapter under
the original execution lock and comparing the complete resulting checkpoint.
"""

from __future__ import annotations

import base64
import copy
import fcntl
import hashlib
import os
import re
import stat
import sys
from pathlib import Path

from megatoken.data.semantic_json import stable_algorithm_hash, stable_algorithm_json
from megatoken.io import strict_json_loads
from megatoken.semantic_request_journal_v8 import (
    DurableNativeRequestJournal,
    NativeRequestConflict,
    NativeRequestJournalError,
    NativeRequestOutcomeUnknown,
    _canonical,
    _validate_identity,
)

_HEX64 = re.compile(r"[0-9a-f]{64}")
_EXECUTION_ID = re.compile(r"[a-z][a-z0-9:_-]{0,255}")
_COMMITS = {"commit", "commit-orchestration"}
_CLOSES = {"cancel"}
_MUTATIONS = {
    "initialize",
    "select",
    "reject",
    "complete",
    "cancel",
    "restore",
    "open-orchestration",
    "select-orchestration",
    "bootstrap-orchestration-child",
    "cancel-orchestration",
}
MAX_COMMIT_REQUEST_BYTES = 40 << 20


def _fail(message):
    raise NativeRequestJournalError("Native commit recovery: " + message)


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _same(left, right):
    return stable_algorithm_json(left) == stable_algorithm_json(right)


def _reply_bytes(value):
    # The producer verifies these exact bytes with ECMAScript stable JSON.
    # Python json.dumps uses different spellings for 1.0 and some exponents.
    return stable_algorithm_json(value).encode("utf-8") + b"\n"


def _decode(value, maximum, label):
    if type(value) is not str or len(value) > ((maximum + 2) // 3) * 4:
        _fail(label + " must be bounded base64 bytes")
    try:
        result = base64.b64decode(value, validate=True)
    except ValueError as exc:
        raise NativeRequestJournalError(label + " is not base64") from exc
    if len(result) > maximum or base64.b64encode(result).decode("ascii") != value:
        _fail(label + " must use canonical bounded base64")
    return result


class NativeExecutionLease:
    """Own one duplicate of a trusted producer's inherited Linux flock FD.

    Closing this object closes only its descriptor, never unlocks the shared open
    file description. An outliving parent or child retains execution exclusion.
    """

    def __init__(self, fd, path, session_id, stream_id):
        if sys.platform != "linux" or type(fd) is not int or fd < 0:
            _fail("a Linux inherited execution descriptor is required")
        if any(
            type(x) is not str or not _EXECUTION_ID.fullmatch(x) for x in (session_id, stream_id)
        ):
            _fail("invalid execution session or stream")
        self.session_id, self.stream_id = session_id, stream_id
        self.path = Path(path)
        expected = stable_algorithm_hash(
            {
                "schema": "semantic-selected-orchestration-lock-key/1",
                "sessionId": session_id,
                "streamId": stream_id,
            }
        )
        if (
            not self.path.is_absolute()
            or ".." in self.path.parts
            or self.path.name != expected + ".lock"
            or self.path.parent.name != "selected-orchestration-execution-locks"
        ):
            _fail("execution lock path does not bind its exact session and stream")
        self.fd = os.dup(fd)
        self._inode = None
        try:
            self.assert_held()
            descriptor = os.fstat(self.fd)
            self._inode = (descriptor.st_dev, descriptor.st_ino)
        except BaseException:
            self.close()
            raise

    def assert_held(self):
        if self.fd is None:
            _fail("execution lease is closed")
        cursor = Path(self.path.anchor)
        for part in self.path.parts[1:]:
            cursor = cursor / part
            entry = cursor.lstat()
            if stat.S_ISLNK(entry.st_mode):
                _fail("execution lock path contains a symlink")
            if cursor in (self.path.parent, self.path.parent.parent) and (
                not stat.S_ISDIR(entry.st_mode)
                or entry.st_uid != os.geteuid()
                or stat.S_IMODE(entry.st_mode) & 0o077
            ):
                _fail("execution lock requires private owner directories")
        descriptor = os.fstat(self.fd)
        entry = self.path.lstat()
        inode = (descriptor.st_dev, descriptor.st_ino)
        if (
            not stat.S_ISREG(descriptor.st_mode)
            or descriptor.st_uid != os.geteuid()
            or descriptor.st_nlink != 1
            or stat.S_IMODE(descriptor.st_mode) & 0o077
            or (entry.st_dev, entry.st_ino) != inode
            or self._inode is not None
            and self._inode != inode
        ):
            _fail("execution descriptor no longer names its private stable inode")
        # The inherited descriptor shares the producer's open-file description.
        # An unrelated live description's lock makes this nonblocking call fail.
        fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None


class DurableNativeCommitBridge:
    """Commit-only journal and full-adapter restoration under execution ownership."""

    def __init__(self, adapter, journal, execution_lease, *, session_generation=1):
        if not isinstance(journal, DurableNativeRequestJournal):
            raise TypeError("A native request journal is required")
        if not isinstance(execution_lease, NativeExecutionLease):
            raise TypeError("An inherited native execution lease is required")
        execution_lease.assert_held()
        if type(session_generation) is not int or not 1 <= session_generation <= 2**53 - 1:
            raise ValueError("Native execution generation must be a positive safe integer")
        self.adapter, self.journal, self.execution_lease = adapter, journal, execution_lease
        self.session_generation = session_generation
        self.fenced = False
        self._recoverable = None
        self._started = False

    def _scope_identity(self):
        model = {
            "checkpoint": self.adapter.model_identity,
            "tokenizer": self.adapter.codec.fingerprint,
        }
        if self.adapter.training_identity is not None:
            model["training"] = self.adapter.training_identity
        return {
            "schema": "semantic-native-request-identity/1",
            "session_id": self.execution_lease.session_id,
            "stream_id": self.execution_lease.stream_id,
            "session_generation": self.session_generation,
            "ordinal": 1,
            "operation": "commit",
            "model_identity": model,
        }

    def assert_operation(self, operation, request=None, *, _durable_bootstrap=False):
        try:
            self.execution_lease.assert_held()
        except BaseException:
            self.fenced = True
            raise
        if operation in {"metrics", "status"}:
            return
        if self.fenced:
            _fail("this process is fenced after an uncertain durable operation")
        if operation in {"commit", "commit_orchestration", "commit-orchestration"}:
            _fail("configured durable sessions require durable-commit")
        if (
            operation in {"bootstrap_orchestration_child", "bootstrap-orchestration-child"}
            and not _durable_bootstrap
        ):
            _fail("configured durable sessions require durable-bootstrap-child")
        if operation == "restore":
            _fail(
                "configured durable sessions require explicit recover-commit, not ordinary restore"
            )
        normalized = operation.replace("_", "-") if isinstance(operation, str) else operation
        if normalized in _MUTATIONS and not self._started:
            if self.journal.has_history(self._scope_identity()):
                _fail(
                    "retained native generation requires explicit commit recovery before any mutation"
                )
            self._started = True
        if normalized in _MUTATIONS and self._recoverable is not None:
            if type(request) is not dict or request.get("op") != operation:
                _fail("a later mutation requires its complete original request")
            identity, frame = self._recoverable
            try:
                self.journal.supersede(identity, frame, copy.deepcopy(request))
            except BaseException:
                self.fenced = True
                raise
            self._recoverable = None

    def _input(self, identity, requestBase64, preCheckpointDigest):
        identity = copy.deepcopy(_validate_identity(identity))
        lease = self.execution_lease
        lease.assert_held()
        if identity["session_id"] != lease.session_id or identity["stream_id"] != lease.stream_id:
            _fail("request identity differs from execution ownership")
        if identity["session_generation"] != self.session_generation:
            _fail("request generation differs from configured native execution")
        if identity["operation"] not in _COMMITS:
            _fail("only exact commit operations are recoverable")
        expected = {
            "checkpoint": self.adapter.model_identity,
            "tokenizer": self.adapter.codec.fingerprint,
        }
        if self.adapter.training_identity is not None:
            expected["training"] = self.adapter.training_identity
        if identity["model_identity"] != expected:
            _fail("request model, tokenizer or training identity differs")
        if type(preCheckpointDigest) is not str or not _HEX64.fullmatch(preCheckpointDigest):
            _fail("a complete pre-request checkpoint digest is required")
        raw = _decode(requestBase64, MAX_COMMIT_REQUEST_BYTES, "commit request")
        if not raw.endswith(b"\n") or b"\n" in raw[:-1] or b"\r" in raw:
            _fail("commit request must be one exact JSONL frame ending in a newline")
        try:
            request = strict_json_loads(raw.decode("utf-8", errors="strict"))
        except (ValueError, UnicodeError) as exc:
            raise NativeRequestJournalError(
                "Native commit request is not strict UTF-8 JSON"
            ) from exc
        if (
            type(request) is not dict
            or type(request.get("id")) is not str
            or not request["id"]
            or request.get("op") != identity["operation"]
            or request.get("session") != identity["session_id"]
        ):
            _fail("exact request ID, operation or session differs")
        keys = {"id", "op", "session", "record", "selection"} | (
            {"public", "contexts", "invocationId"}
            if identity["operation"] == "commit"
            else {"track"}
        )
        if set(request) != keys:
            _fail("commit request fields differ")
        frame = _canonical(
            {
                "schema": "semantic-native-durable-commit-request/8",
                "preCheckpointDigest": preCheckpointDigest,
                "requestBase64": requestBase64,
            }
        )
        return identity, raw, request, frame

    def _bootstrap_input(
        self,
        identity,
        requestBase64,
        preCheckpointDigest,
        createIdentity,
        createRequestBase64,
        createPreCheckpointDigest,
    ):
        identity = copy.deepcopy(_validate_identity(identity))
        create_identity = copy.deepcopy(_validate_identity(createIdentity))
        lease = self.execution_lease
        lease.assert_held()
        if identity["session_id"] != lease.session_id or identity["stream_id"] != lease.stream_id:
            _fail("bootstrap request identity differs from execution ownership")
        if identity["session_generation"] != self.session_generation:
            _fail("bootstrap generation differs from configured native execution")
        if identity["operation"] != "bootstrap-orchestration-child":
            _fail("only child bootstrap operations use bootstrap recovery")
        expected = {
            "checkpoint": self.adapter.model_identity,
            "tokenizer": self.adapter.codec.fingerprint,
        }
        if self.adapter.training_identity is not None:
            expected["training"] = self.adapter.training_identity
        if identity["model_identity"] != expected:
            _fail("bootstrap model, tokenizer or training identity differs")
        if type(preCheckpointDigest) is not str or not _HEX64.fullmatch(preCheckpointDigest):
            _fail("a complete bootstrap precheckpoint digest is required")
        raw = _decode(requestBase64, MAX_COMMIT_REQUEST_BYTES, "bootstrap request")
        if not raw.endswith(b"\n") or b"\n" in raw[:-1] or b"\r" in raw:
            _fail("bootstrap request must be one exact JSONL frame ending in a newline")
        try:
            request = strict_json_loads(raw.decode("utf-8", errors="strict"))
        except (ValueError, UnicodeError) as exc:
            raise NativeRequestJournalError("Bootstrap request is not strict UTF-8 JSON") from exc
        if (
            type(request) is not dict
            or set(request) != {"id", "op", "session", "track", "childTrack"}
            or type(request.get("id")) is not str
            or not request["id"]
            or request.get("op") != "bootstrap-orchestration-child"
            or request.get("session") != identity["session_id"]
            or not all(
                type(request.get(key)) is str and request[key] for key in ("track", "childTrack")
            )
        ):
            _fail("bootstrap request fields, ID, session or child identity differ")
        if (
            create_identity["session_id"] != identity["session_id"]
            or create_identity["stream_id"] != identity["stream_id"]
            or create_identity["session_generation"] != identity["session_generation"]
            or create_identity["model_identity"] != identity["model_identity"]
            or create_identity["operation"] != "commit-orchestration"
            or create_identity["ordinal"] + 1 != identity["ordinal"]
        ):
            _fail("bootstrap must immediately follow its exact same-owner child-create commit")
        # The predecessor is causal evidence only. Its own latest-state recovery
        # authority is revoked by this later bootstrap intent.
        create_identity, create_raw, create_request, create_frame = self._input(
            create_identity, createRequestBase64, createPreCheckpointDigest
        )
        if (
            set(create_request) != {"id", "op", "session", "track", "record", "selection"}
            or create_request.get("op") != "commit-orchestration"
            or create_request.get("session") != request["session"]
            or create_request.get("track") != request["track"]
            or type(create_request.get("record")) is not str
        ):
            _fail("bootstrap predecessor is not the exact coordinator child-create commit")
        try:
            create_record = strict_json_loads(create_request["record"])
        except (ValueError, TypeError) as exc:
            raise NativeRequestJournalError(
                "Native commit recovery: child-create record is not strict JSON"
            ) from exc
        if type(create_record) is not dict or create_record.get("name") != "track_create":
            _fail("bootstrap predecessor is not the exact coordinator child-create commit")
        frame = _canonical(
            {
                "schema": "semantic-native-durable-bootstrap-child-request/8",
                "preCheckpointDigest": preCheckpointDigest,
                "requestBase64": requestBase64,
                "createIdentity": create_identity,
                "createRequestBase64": createRequestBase64,
                "createPreCheckpointDigest": createPreCheckpointDigest,
            }
        )
        return (
            identity,
            raw,
            request,
            frame,
            create_identity,
            create_raw,
            create_request,
            create_frame,
        )

    def _create_predecessor(
        self, create_identity, create_raw, create_request, create_frame, create_pre_digest
    ):
        readback = self.journal.inspect(create_identity, create_frame)
        if readback["status"] == "never-admitted":
            raise NativeRequestConflict("Child bootstrap has no admitted child-create commit")
        if readback["status"] != "completed":
            raise NativeRequestOutcomeUnknown("Child-create commit outcome is unknown")
        value, response_raw = self._envelope(
            readback["result_bytes"],
            create_request,
            create_raw,
            create_pre_digest,
        )
        response = strict_json_loads(response_raw.decode("utf-8", errors="strict"))
        return readback, value, response_raw, response

    def _close_input(self, identity, requestBase64, preCheckpointDigest):
        identity = copy.deepcopy(_validate_identity(identity))
        lease = self.execution_lease
        lease.assert_held()
        if identity["session_id"] != lease.session_id or identity["stream_id"] != lease.stream_id:
            _fail("request identity differs from execution ownership")
        if identity["session_generation"] != self.session_generation:
            _fail("request generation differs from configured native execution")
        if identity["operation"] not in _CLOSES:
            _fail("only exact terminal close operations are recoverable")
        expected = {
            "checkpoint": self.adapter.model_identity,
            "tokenizer": self.adapter.codec.fingerprint,
        }
        if self.adapter.training_identity is not None:
            expected["training"] = self.adapter.training_identity
        if identity["model_identity"] != expected:
            _fail("request model, tokenizer or training identity differs")
        if type(preCheckpointDigest) is not str or not _HEX64.fullmatch(preCheckpointDigest):
            _fail("a complete pre-request checkpoint digest is required")
        raw = _decode(requestBase64, MAX_COMMIT_REQUEST_BYTES, "close request")
        if not raw.endswith(b"\n") or b"\n" in raw[:-1] or b"\r" in raw:
            _fail("close request must be one exact JSONL frame ending in a newline")
        try:
            request = strict_json_loads(raw.decode("utf-8", errors="strict"))
        except (ValueError, UnicodeError) as exc:
            raise NativeRequestJournalError(
                "Native close request is not strict UTF-8 JSON"
            ) from exc
        if (
            type(request) is not dict
            or type(request.get("id")) is not str
            or not request["id"]
            or request.get("op") != "cancel"
            or request.get("session") != identity["session_id"]
            or set(request) != {"id", "op", "session", "reason"}
            or type(request.get("reason")) is not str
            or not request["reason"]
            or identity["operation"] != request["op"]
        ):
            _fail("exact close request ID, operation, session or reason differs")
        frame = _canonical(
            {
                "schema": "semantic-native-durable-close-request/8",
                "preCheckpointDigest": preCheckpointDigest,
                "requestBase64": requestBase64,
            }
        )
        return identity, raw, request, frame

    def _assert_terminal_close(self, request, checkpoint):
        sessions = checkpoint.get("sessions")
        closed = checkpoint.get("closedSessions")
        if type(sessions) is not dict or type(closed) is not dict:
            _fail("terminal close requires complete active and closed session inventories")
        if any(
            type(saved) is not dict or saved.get("unacknowledgedSelection") is not None
            for saved in sessions.values()
        ):
            _fail("terminal close cannot retain pending selections in any session")
        if request["session"] in closed:
            _fail("terminal close requires the exact active session")
        work = self.adapter.sessions.get(request["session"])
        if work is None or work.orchestration is None:
            _fail("durable close requires a terminal orchestration coordinator")
        state = work.orchestration
        coordinator = work.runtime.tracks.tracks.get(state.get("track"))
        if (
            coordinator is None
            or coordinator.status not in {"completed", "cancelled"}
            or work.pending is not None
            or any(
                track.status not in {"completed", "cancelled"}
                for track in work.runtime.tracks.tracks.values()
            )
        ):
            _fail("durable close requires every owned native track to be terminal")
        if coordinator.status == "cancelled" and not state.get("cancelled"):
            _fail("cancelled coordinator lacks its exact acknowledged cancellation")
        if coordinator.status == "completed" and state.get("cancelled") is not None:
            _fail("completed coordinator carries a cancellation outcome")
        if len(closed) >= self.adapter.max_attempts:
            _fail("closed-session history capacity exhausted")
        return work

    def _check_close_checkpoints(self, request, response, checkpoint, before, before_digest):
        if (
            type(before) is not dict
            or before.get("digest") != before_digest
            or before_digest
            != stable_algorithm_hash({key: value for key, value in before.items() if key != "digest"})
        ):
            _fail("saved precheckpoint does not bind the producer's original digest")
        if (
            type(checkpoint) is not dict
            or checkpoint.get("digest")
            != stable_algorithm_hash(
                {key: value for key, value in checkpoint.items() if key != "digest"}
            )
        ):
            _fail("saved postcheckpoint digest differs")
        before_sessions, sessions = before.get("sessions"), checkpoint.get("sessions")
        before_closed, closed = before.get("closedSessions"), checkpoint.get("closedSessions")
        if (
            type(before_sessions) is not dict
            or type(sessions) is not dict
            or request["session"] not in before_sessions
            or set(sessions) != set(before_sessions) - {request["session"]}
            or any(
                type(saved) is not dict or saved.get("unacknowledgedSelection") is not None
                for saved in before_sessions.values()
            )
            or any(
                type(saved) is not dict or saved.get("unacknowledgedSelection") is not None
                for saved in sessions.values()
            )
            or type(before_closed) is not dict
            or type(closed) is not dict
            or request["session"] in before_closed
            or set(closed) != set(before_closed) | {request["session"]}
        ):
            _fail("close changed the complete active or closed session inventory")
        for session, saved in before_sessions.items():
            if session != request["session"] and not _same(saved, sessions[session]):
                _fail("close changed a sibling session")
        for session, saved in before_closed.items():
            if not _same(saved, closed[session]):
                _fail("close changed an earlier closed-session archive")
        target = before_sessions[request["session"]]
        if (
            type(target) is not dict
            or target.get("unacknowledgedSelection") is not None
            or type(closed.get(request["session"])) is not dict
            or not _same(closed[request["session"]], target)
        ):
            _fail("closed-session archive differs from the exact terminal precheckpoint")
        for field in (
            "schema",
            "equivalence",
            "contractDigest",
            "initialization",
            "settings",
            "randomState",
            "attempts",
            "openings",
        ):
            if not _same(before.get(field), checkpoint.get(field)):
                _fail("close changed whole-adapter " + field)
        expected = {
            "id": request["id"],
            "ok": True,
            "result": {
                "schema": "semantic-native-close/8",
                "session": request["session"],
                "closed": True,
                "reason": request["reason"],
            },
        }
        if not _same(response, expected):
            _fail("stored response differs from the exact terminal close request")

    def _close_envelope(self, stored, request, raw, pre_digest):
        value = strict_json_loads(stored.decode("utf-8", errors="strict"))
        if (
            type(value) is not dict
            or set(value)
            != {
                "schema",
                "requestSha256",
                "preCheckpointDigest",
                "responseBase64",
                "postCheckpoint",
                "preCheckpoint",
            }
            or value["schema"] != "semantic-native-durable-close-outcome/8"
            or value["requestSha256"] != _sha(raw)
            or value["preCheckpointDigest"] != pre_digest
            or _canonical(value) != stored
        ):
            _fail("saved terminal close outcome binding differs")
        response_raw = _decode(value["responseBase64"], MAX_COMMIT_REQUEST_BYTES, "close response")
        response = strict_json_loads(response_raw.decode("utf-8", errors="strict"))
        if _reply_bytes(response) != response_raw:
            _fail("saved terminal close response bytes differ")
        self._check_close_checkpoints(
            request, response, value["postCheckpoint"], value["preCheckpoint"], pre_digest
        )
        return value, response_raw, response

    def _check_post(self, request, response, checkpoint, before, before_digest):
        if (
            type(before) is not dict
            or before.get("digest") != before_digest
            or before_digest
            != stable_algorithm_hash(
                {key: value for key, value in before.items() if key != "digest"}
            )
        ):
            _fail("saved precheckpoint does not bind the producer's original digest")
        if type(checkpoint) is not dict or checkpoint.get("digest") != stable_algorithm_hash(
            {key: value for key, value in checkpoint.items() if key != "digest"}
        ):
            _fail("saved postcheckpoint digest differs")
        sessions = checkpoint.get("sessions")
        if type(sessions) is not dict or any(
            type(saved) is not dict or saved.get("unacknowledgedSelection") is not None
            for saved in sessions.values()
        ):
            _fail("saved postcheckpoint contains an uncommitted sibling selection")
        selection = request["selection"]
        if type(selection) is not dict or selection.get("status") != "awaiting-commit":
            _fail("commit requires its exact pending selection")
        before_sessions = before.get("sessions")
        if type(before_sessions) is not dict or set(before_sessions) != set(sessions):
            _fail("commit changed the complete session inventory")
        current = before_sessions.get(request["session"])
        if type(current) is not dict or current.get("unacknowledgedSelection") != selection.get(
            "selectionId"
        ):
            _fail("precheckpoint does not contain the exact pending selection")
        for session, saved in before_sessions.items():
            if session != request["session"] and (
                saved.get("unacknowledgedSelection") is not None
                or not _same(saved, sessions[session])
            ):
                _fail("commit changed a sibling session or abandoned its pending selection")
        for field in (
            "schema",
            "equivalence",
            "contractDigest",
            "initialization",
            "settings",
            "closedSessions",
            "openings",
            "randomState",
        ):
            if not _same(before.get(field), checkpoint.get(field)):
                _fail("commit changed whole-adapter " + field)
        prior_attempts = before.get("attempts")
        if type(prior_attempts) is not list:
            _fail("precheckpoint attempts are missing")
        pending = [
            row for row in prior_attempts if row.get("selectionId") == selection.get("selectionId")
        ]
        if len(pending) != 1 or not _same(pending[0], selection):
            _fail("precheckpoint selection differs from the exact original request")
        expected_attempts = [
            {**row, "status": "committed"}
            if row.get("selectionId") == selection["selectionId"]
            else row
            for row in prior_attempts
        ]
        if not _same(expected_attempts, checkpoint.get("attempts")):
            _fail("commit changed sibling attempts or consumed a new selection")
        attempts = checkpoint.get("attempts")
        matches = [
            attempt
            for attempt in attempts or []
            if type(attempt) is dict and attempt.get("selectionId") == selection.get("selectionId")
        ]
        if len(matches) != 1 or not _same(matches[0], {**selection, "status": "committed"}):
            _fail("postcheckpoint does not contain the exact committed selection")
        saved = sessions.get(request["session"])
        if type(saved) is not dict:
            _fail("postcheckpoint does not contain the committed session")
        if request["op"] == "commit":
            expected = {
                "schema": "semantic-native-commit/8",
                "selectionId": selection["selectionId"],
                "publicDigest": stable_algorithm_hash(request["public"]),
                "committed": True,
            }
            if not _same(saved.get("public"), request["public"]) or not _same(
                saved.get("contexts"), request["contexts"]
            ):
                _fail("commit postcheckpoint public history differs")
        else:
            decisions = saved.get("orchestration", {}).get("decisions", [])
            matches = [
                row for row in decisions if row.get("selectionId") == selection["selectionId"]
            ]
            if len(matches) != 1 or matches[0].get("record") != request["record"]:
                _fail("postcheckpoint orchestration decision differs")
            expected = matches[0]["outcome"]
        if not _same(response, {"id": request["id"], "ok": True, "result": expected}):
            _fail("stored response differs from its exact native committed outcome")

    def _envelope(self, stored, request, raw, pre_digest):
        value = strict_json_loads(stored.decode("utf-8", errors="strict"))
        if (
            type(value) is not dict
            or set(value)
            != {
                "schema",
                "requestSha256",
                "preCheckpointDigest",
                "responseBase64",
                "postCheckpoint",
                "preCheckpoint",
            }
            or value["schema"] != "semantic-native-durable-commit-outcome/8"
            or value["requestSha256"] != _sha(raw)
            or value["preCheckpointDigest"] != pre_digest
            or _canonical(value) != stored
        ):
            _fail("saved commit outcome binding differs")
        response_raw = _decode(value["responseBase64"], MAX_COMMIT_REQUEST_BYTES, "commit response")
        response = strict_json_loads(response_raw.decode("utf-8", errors="strict"))
        if _reply_bytes(response) != response_raw:
            _fail("saved commit response bytes differ")
        self._check_post(
            request, response, value["postCheckpoint"], value["preCheckpoint"], pre_digest
        )
        return value, response_raw

    def _receipt(self, identity, raw, value, response_raw, request_id, restored=None):
        return {
            "schema": "semantic-native-durable-commit/8",
            "requestId": request_id,
            "identity": copy.deepcopy(identity),
            "requestSha256": _sha(raw),
            "preCheckpointDigest": value["preCheckpointDigest"],
            "postCheckpointDigest": value["postCheckpoint"]["digest"],
            "responseBase64": value["responseBase64"],
            "responseSha256": _sha(response_raw),
            "postCheckpoint": copy.deepcopy(value["postCheckpoint"]),
            "recovered": restored is not None,
            "restoreReceipt": restored,
        }

    def _check_bootstrap_pre(self, request, create_value, create_response, before, digest):
        if (
            type(before) is not dict
            or before.get("digest") != digest
            or digest
            != stable_algorithm_hash(
                {key: value for key, value in before.items() if key != "digest"}
            )
            or not _same(before, create_value["postCheckpoint"])
        ):
            _fail("bootstrap precheckpoint is not the complete acknowledged child-create state")
        if any(
            type(saved) is not dict or saved.get("unacknowledgedSelection") is not None
            for saved in before.get("sessions", {}).values()
        ):
            _fail("bootstrap precheckpoint contains an uncommitted sibling selection")
        sessions = before.get("sessions")
        saved = sessions.get(request["session"]) if type(sessions) is dict else None
        orchestration = saved.get("orchestration") if type(saved) is dict else None
        children = orchestration.get("children") if type(orchestration) is dict else None
        matching = [
            child
            for child in children or []
            if type(child) is dict and child.get("track") == request["childTrack"]
        ]
        create_result = create_response.get("result")
        created = create_result.get("track") if type(create_result) is dict else None
        if (
            len(matching) != 1
            or created is None
            or created.get("address") != request["childTrack"]
            or created.get("bootstrap_ready") is not False
            or orchestration.get("track") != request["track"]
        ):
            _fail("bootstrap request does not name its exact committed unbootstrapped child")

    def _check_bootstrap_post(
        self, request, response, checkpoint, before, before_digest, create_value, create_response
    ):
        self._check_bootstrap_pre(request, create_value, create_response, before, before_digest)
        if (
            type(checkpoint) is not dict
            or checkpoint.get("digest")
            != stable_algorithm_hash(
                {key: value for key, value in checkpoint.items() if key != "digest"}
            )
            or set(before.get("sessions", {})) != set(checkpoint.get("sessions", {}))
        ):
            _fail("bootstrap postcheckpoint digest or full session inventory differs")
        for field in (
            "schema",
            "equivalence",
            "contractDigest",
            "initialization",
            "settings",
            "attempts",
            "closedSessions",
            "openings",
            "randomState",
        ):
            if not _same(before.get(field), checkpoint.get(field)):
                _fail("child bootstrap changed whole-adapter " + field)
        for session, saved in before["sessions"].items():
            after = checkpoint["sessions"][session]
            if session != request["session"]:
                if not _same(saved, after):
                    _fail("child bootstrap changed a sibling session")
                continue
            for field in (
                "public",
                "contexts",
                "ingestedThrough",
                "completed",
                "openingRandom",
                "replay",
                "unacknowledgedSelection",
            ):
                if not _same(saved.get(field), after.get(field)):
                    _fail("child bootstrap changed coordinator accounting " + field)
            old_orchestration = saved.get("orchestration")
            new_orchestration = after.get("orchestration")
            if type(old_orchestration) is not dict or type(new_orchestration) is not dict:
                _fail("child bootstrap lost its orchestration state")
            operation = new_orchestration.get("operations", [])[-1:]
            if (
                operation
                and operation[0].get("kind") == "bootstrap"
                and operation[0].get("track") == request["childTrack"]
                and set(operation[0]) == {"kind", "track", "randomBefore"}
                and _same(operation[0].get("randomBefore"), before.get("randomState"))
                and _same(
                    {**new_orchestration, "operations": new_orchestration["operations"][:-1]},
                    old_orchestration,
                )
            ):
                pass
            else:
                _fail("child bootstrap changed coordinator history beyond its one exact receipt")
        target_session = checkpoint["sessions"][request["session"]]
        expected_track = target_session["orchestration"]
        child = next(
            child for child in expected_track["children"] if child["track"] == request["childTrack"]
        )
        if child.get("completion") is not None:
            _fail("child bootstrap changed reserved child completion")
        expected = {
            "schema": "semantic-orchestration-bootstrap/8",
            "track": response.get("result", {}).get("track"),
            "bootstrapped": True,
        }
        created = create_response.get("result", {}).get("track")
        returned_track = expected["track"]
        track_matches = False
        if type(created) is dict and type(returned_track) is dict:
            expected_track_response = {
                **created,
                "bootstrap_ready": True,
                "status": returned_track.get("status"),
            }
            track_matches = (
                returned_track.get("status") in {"active", "waiting"}
                and _same(returned_track, expected_track_response)
            )
        if (
            not _same(response, {"id": request["id"], "ok": True, "result": expected})
            or not track_matches
        ):
            _fail("bootstrap response differs from the exact postcheckpoint child state")

    def _live_bootstrap_result(self, request):
        work = self.adapter.sessions.get(request["session"])
        if work is None or work.orchestration is None or work.orchestration.get(
            "track"
        ) != request["track"]:
            _fail("restored bootstrap coordinator differs from its exact request")
        reservations = [
            child
            for child in work.orchestration.get("children", [])
            if child.get("track") == request["childTrack"]
        ]
        description = work.runtime.tracks.tracks.get(request["childTrack"])
        if (
            len(reservations) != 1
            or description is None
            or description.execution is None
            or description.bootstrap_ready is not True
            or description.status not in {"active", "waiting"}
            or work.runtime.execution_publication.get(request["childTrack"]) != 2
        ):
            _fail("restored child is not the exact fully published bootstrap")
        from megatoken.semantic_orchestration_v8 import _description

        return {
            "schema": "semantic-orchestration-bootstrap/8",
            "track": _description(description),
            "bootstrapped": True,
        }

    def _bootstrap_envelope(
        self,
        stored,
        request,
        raw,
        pre_digest,
        create_identity,
        create_raw,
        create_pre_digest,
        create_value,
        create_response,
    ):
        value = strict_json_loads(stored.decode("utf-8", errors="strict"))
        if (
            type(value) is not dict
            or set(value)
            != {
                "schema",
                "requestSha256",
                "preCheckpointDigest",
                "createIdentity",
                "createRequestSha256",
                "createPreCheckpointDigest",
                "createPostCheckpointDigest",
                "responseBase64",
                "postCheckpoint",
                "preCheckpoint",
            }
            or value["schema"] != "semantic-native-durable-bootstrap-child-outcome/8"
            or value["requestSha256"] != _sha(raw)
            or value["preCheckpointDigest"] != pre_digest
            or not _same(value["createIdentity"], create_identity)
            or value["createRequestSha256"] != _sha(create_raw)
            or value["createPreCheckpointDigest"] != create_pre_digest
            or value["createPostCheckpointDigest"] != create_value["postCheckpoint"]["digest"]
            or _canonical(value) != stored
        ):
            _fail("saved bootstrap outcome binding differs")
        response_raw = _decode(
            value["responseBase64"], MAX_COMMIT_REQUEST_BYTES, "bootstrap response"
        )
        response = strict_json_loads(response_raw.decode("utf-8", errors="strict"))
        if _reply_bytes(response) != response_raw:
            _fail("saved bootstrap response bytes differ")
        self._check_bootstrap_post(
            request,
            response,
            value["postCheckpoint"],
            value["preCheckpoint"],
            pre_digest,
            create_value,
            create_response,
        )
        return value, response_raw, response

    def _bootstrap_receipt(
        self,
        identity,
        raw,
        value,
        response_raw,
        request_id,
        create_identity,
        create_raw,
        create_readback,
        create_value,
        restored=None,
    ):
        return {
            "schema": "semantic-native-durable-bootstrap-child/8",
            "requestId": request_id,
            "identity": copy.deepcopy(identity),
            "requestSha256": _sha(raw),
            "preCheckpointDigest": value["preCheckpointDigest"],
            "postCheckpointDigest": value["postCheckpoint"]["digest"],
            "responseBase64": value["responseBase64"],
            "responseSha256": _sha(response_raw),
            "createIdentity": copy.deepcopy(create_identity),
            "createRequestId": create_readback["request_id"],
            "createRequestSha256": _sha(create_raw),
            "createPreCheckpointDigest": value["createPreCheckpointDigest"],
            "createPostCheckpointDigest": create_value["postCheckpoint"]["digest"],
            "createResponseSha256": _sha(
                _decode(
                    create_value["responseBase64"],
                    MAX_COMMIT_REQUEST_BYTES,
                    "create response",
                )
            ),
            "postCheckpoint": copy.deepcopy(value["postCheckpoint"]),
            "recovered": restored is not None,
            "restoreReceipt": restored,
        }

    def durable_bootstrap_child(
        self,
        *,
        identity,
        requestBase64,
        preCheckpointDigest,
        createIdentity,
        createRequestBase64,
        createPreCheckpointDigest,
    ):
        self.assert_operation("durable-bootstrap-child")
        identity, raw, request, frame, create_identity, create_raw, create_request, create_frame = (
            self._bootstrap_input(
                identity,
                requestBase64,
                preCheckpointDigest,
                createIdentity,
                createRequestBase64,
                createPreCheckpointDigest,
            )
        )
        with self.adapter.lock:
            create_readback, create_value, _, create_response = self._create_predecessor(
                create_identity,
                create_raw,
                create_request,
                create_frame,
                createPreCheckpointDigest,
            )
            existing = self.journal.inspect(identity, frame, require_latest=True)
            if existing["status"] == "never-admitted":
                self.journal.assert_not_superseded(create_identity, create_frame)
            else:
                self.journal.assert_superseded_by(create_identity, create_frame, request)
            if existing["status"] == "unknown":
                self.fenced = True
                raise NativeRequestOutcomeUnknown(
                    "Bootstrap outcome is unknown; native bootstrap cannot be retried"
                )
            if existing["status"] == "completed":
                value, response_raw, response = self._bootstrap_envelope(
                    existing["result_bytes"],
                    request,
                    raw,
                    preCheckpointDigest,
                    create_identity,
                    create_raw,
                    createPreCheckpointDigest,
                    create_value,
                    create_response,
                )
                if not _same(self.adapter.checkpoint(), value["postCheckpoint"]):
                    _fail("whole live adapter differs from saved bootstrap poststate")
                if not _same(response.get("result"), self._live_bootstrap_result(request)):
                    _fail("stored response differs from the complete live bootstrap state")
                self._recoverable = (copy.deepcopy(identity), frame)
                self._started = True
                return self._bootstrap_receipt(
                    identity,
                    raw,
                    value,
                    response_raw,
                    existing["request_id"],
                    create_identity,
                    create_raw,
                    create_readback,
                    create_value,
                )
            before = self.adapter.checkpoint()
            if before["digest"] != preCheckpointDigest or not _same(
                before, create_value["postCheckpoint"]
            ):
                raise NativeRequestConflict(
                    "Bootstrap precheckpoint differs from its exact committed create"
                )
            # Validate the requested child before revoking create recovery or
            # admitting an irreversible bootstrap intent.
            self._check_bootstrap_pre(
                request, create_value, create_response, before, preCheckpointDigest
            )
            self.assert_operation("bootstrap-orchestration-child", request, _durable_bootstrap=True)

            def execute(_frame):
                before = self.adapter.checkpoint()
                if before["digest"] != preCheckpointDigest or not _same(
                    before, create_value["postCheckpoint"]
                ):
                    raise NativeRequestConflict(
                        "Bootstrap precheckpoint differs from its exact committed create"
                    )
                self._check_bootstrap_pre(
                    request, create_value, create_response, before, preCheckpointDigest
                )
                work = self.adapter.sessions.get(request["session"])
                description = (
                    None if work is None else work.runtime.tracks.tracks.get(request["childTrack"])
                )
                if description is None or description.bootstrap_ready:
                    _fail("Committed child is not in its exact unbootstrapped native state")
                result = self.adapter.bootstrap_orchestration_child(
                    session=request["session"],
                    track=request["track"],
                    childTrack=request["childTrack"],
                )
                if not _same(result, self._live_bootstrap_result(request)):
                    _fail("native bootstrap response differs from its full live track state")
                post = self.adapter.checkpoint()
                response = {"id": request["id"], "ok": True, "result": result}
                self._check_bootstrap_post(
                    request,
                    response,
                    post,
                    before,
                    preCheckpointDigest,
                    create_value,
                    create_response,
                )
                return _canonical(
                    {
                        "schema": "semantic-native-durable-bootstrap-child-outcome/8",
                        "requestSha256": _sha(raw),
                        "preCheckpointDigest": preCheckpointDigest,
                        "createIdentity": create_identity,
                        "createRequestSha256": _sha(create_raw),
                        "createPreCheckpointDigest": createPreCheckpointDigest,
                        "createPostCheckpointDigest": create_value["postCheckpoint"]["digest"],
                        "responseBase64": base64.b64encode(_reply_bytes(response)).decode("ascii"),
                        "postCheckpoint": post,
                        "preCheckpoint": before,
                    }
                )

            try:
                stored = self.journal.execute(identity, frame, execute)
                value, response_raw, _ = self._bootstrap_envelope(
                    stored,
                    request,
                    raw,
                    preCheckpointDigest,
                    create_identity,
                    create_raw,
                    createPreCheckpointDigest,
                    create_value,
                    create_response,
                )
                if not _same(self.adapter.checkpoint(), value["postCheckpoint"]):
                    _fail("whole live adapter differs from saved bootstrap poststate")
                readback = self.journal.inspect(identity, frame, require_latest=True)
                if readback.get("result_bytes") != stored:
                    _fail("independent durable bootstrap readback differs")
                self._recoverable = (copy.deepcopy(identity), frame)
                self._started = True
                return self._bootstrap_receipt(
                    identity,
                    raw,
                    value,
                    response_raw,
                    readback["request_id"],
                    create_identity,
                    create_raw,
                    create_readback,
                    create_value,
                )
            except BaseException:
                self.fenced = True
                raise

    def recover_bootstrap_child(
        self,
        *,
        identity,
        requestBase64,
        preCheckpointDigest,
        createIdentity,
        createRequestBase64,
        createPreCheckpointDigest,
    ):
        self.assert_operation("recover-bootstrap-child")
        identity, raw, request, frame, create_identity, create_raw, create_request, create_frame = (
            self._bootstrap_input(
                identity,
                requestBase64,
                preCheckpointDigest,
                createIdentity,
                createRequestBase64,
                createPreCheckpointDigest,
            )
        )
        with self.adapter.lock:
            if (
                self.adapter.sessions
                or self.adapter.attempts
                or self.adapter.closed_sessions
                or self.adapter.openings
                or self.adapter.runtime.memory.sessions
            ):
                _fail("bootstrap recovery requires a fresh entire native adapter")
            create_readback, create_value, _, create_response = self._create_predecessor(
                create_identity,
                create_raw,
                create_request,
                create_frame,
                createPreCheckpointDigest,
            )
            readback = self.journal.inspect(identity, frame, require_latest=True)
            if readback["status"] == "never-admitted":
                raise NativeRequestConflict(
                    "Bootstrap was never admitted; recovery cannot execute it"
                )
            self.journal.assert_superseded_by(create_identity, create_frame, request)
            if readback["status"] != "completed":
                raise NativeRequestOutcomeUnknown(
                    "Bootstrap outcome is unknown; recovery cannot retry it"
                )
            try:
                value, response_raw, response = self._bootstrap_envelope(
                    readback["result_bytes"],
                    request,
                    raw,
                    preCheckpointDigest,
                    create_identity,
                    create_raw,
                    createPreCheckpointDigest,
                    create_value,
                    create_response,
                )
            except BaseException:
                self.fenced = True
                raise
            try:
                restored = self.adapter.restore(checkpoint=value["preCheckpoint"])
                if not _same(self.adapter.checkpoint(), value["preCheckpoint"]):
                    _fail("whole restored adapter differs from bootstrap precheckpoint")
                self._check_bootstrap_pre(
                    request,
                    create_value,
                    create_response,
                    self.adapter.checkpoint(),
                    preCheckpointDigest,
                )
                work = self.adapter.sessions.get(request["session"])
                description = (
                    None if work is None else work.runtime.tracks.tracks.get(request["childTrack"])
                )
                if description is None or description.bootstrap_ready:
                    _fail("Restored committed child is not unbootstrapped")
                replay_result = self.adapter.bootstrap_orchestration_child(
                    session=request["session"],
                    track=request["track"],
                    childTrack=request["childTrack"],
                )
                replay_response = {"id": request["id"], "ok": True, "result": replay_result}
                if not _same(replay_response, response) or not _same(
                    self.adapter.checkpoint(), value["postCheckpoint"]
                ):
                    _fail(
                        "independent full-state bootstrap reexecution differs from durable outcome"
                    )
                if not _same(response.get("result"), self._live_bootstrap_result(request)):
                    _fail("restored child differs from the exact bootstrap response")
                self._recoverable = (copy.deepcopy(identity), frame)
                self._started = True
                return self._bootstrap_receipt(
                    identity,
                    raw,
                    value,
                    response_raw,
                    readback["request_id"],
                    create_identity,
                    create_raw,
                    create_readback,
                    create_value,
                    restored,
                )
            except BaseException:
                self.fenced = True
                raise

    def durable_commit(self, *, identity, requestBase64, preCheckpointDigest):
        self.assert_operation("durable-commit")
        identity, raw, request, frame = self._input(identity, requestBase64, preCheckpointDigest)
        with self.adapter.lock:

            def execute(_frame):
                before = self.adapter.checkpoint()
                if before["digest"] != preCheckpointDigest:
                    raise NativeRequestConflict("Native pre-request checkpoint differs")
                if any(
                    session != identity["session_id"]
                    and saved["unacknowledgedSelection"] is not None
                    for session, saved in before["sessions"].items()
                ):
                    _fail("pending sibling selections cannot survive commit recovery")
                result = getattr(self.adapter, request["op"].replace("-", "_"))(
                    **{key: value for key, value in request.items() if key not in {"id", "op"}}
                )
                post = self.adapter.checkpoint()
                response = {"id": request["id"], "ok": True, "result": result}
                self._check_post(request, response, post, before, preCheckpointDigest)
                return _canonical(
                    {
                        "schema": "semantic-native-durable-commit-outcome/8",
                        "requestSha256": _sha(raw),
                        "preCheckpointDigest": preCheckpointDigest,
                        "responseBase64": base64.b64encode(_reply_bytes(response)).decode("ascii"),
                        "postCheckpoint": post,
                        "preCheckpoint": before,
                    }
                )

            try:
                stored = self.journal.execute(identity, frame, execute)
                value, response_raw = self._envelope(stored, request, raw, preCheckpointDigest)
                if not _same(self.adapter.checkpoint(), value["postCheckpoint"]):
                    _fail("whole live adapter differs from saved commit poststate")
                readback = self.journal.inspect(identity, frame, require_latest=True)
                if readback.get("result_bytes") != stored:
                    _fail("independent durable commit readback differs")
                self._recoverable = (copy.deepcopy(identity), frame)
                self._started = True
                return self._receipt(identity, raw, value, response_raw, readback["request_id"])
            except BaseException:
                self.fenced = True
                raise

    def recover_commit(self, *, identity, requestBase64, preCheckpointDigest):
        self.assert_operation("recover-commit")
        identity, raw, request, frame = self._input(identity, requestBase64, preCheckpointDigest)
        with self.adapter.lock:
            if (
                self.adapter.sessions
                or self.adapter.attempts
                or self.adapter.closed_sessions
                or self.adapter.openings
                or self.adapter.runtime.memory.sessions
            ):
                _fail("recovery requires a fresh entire native adapter")
            readback = self.journal.inspect(identity, frame, require_latest=True)
            if readback["status"] == "never-admitted":
                raise NativeRequestConflict(
                    "Native commit was never admitted; recovery cannot execute it"
                )
            if readback["status"] != "completed":
                raise NativeRequestOutcomeUnknown(
                    "Native commit outcome is unknown; recovery cannot retry it"
                )
            value, response_raw = self._envelope(
                readback["result_bytes"], request, raw, preCheckpointDigest
            )
            try:
                restored = self.adapter.restore(checkpoint=value["postCheckpoint"])
                if not _same(self.adapter.checkpoint(), value["postCheckpoint"]):
                    _fail("whole restored adapter differs from saved commit poststate")
                self._recoverable = (copy.deepcopy(identity), frame)
                self._started = True
                return self._receipt(
                    identity, raw, value, response_raw, readback["request_id"], restored
                )
            except BaseException:
                # Native restore is staged and atomic. A post-restore comparison
                # failure must still prevent all future dispatch in this process.
                self.fenced = True
                raise
    def _close_receipt(self, identity, raw, value, response_raw, request_id, restored=None):
        return {
            "schema": "semantic-native-durable-close/8",
            "requestId": request_id,
            "identity": copy.deepcopy(identity),
            "requestSha256": _sha(raw),
            "preCheckpointDigest": value["preCheckpointDigest"],
            "postCheckpointDigest": value["postCheckpoint"]["digest"],
            "responseBase64": value["responseBase64"],
            "responseSha256": _sha(response_raw),
            "postCheckpoint": copy.deepcopy(value["postCheckpoint"]),
            "recovered": restored is not None,
            "restoreReceipt": restored,
        }

    def _assert_closed_adapter(self, request, checkpoint, restore_receipt=None):
        if request["session"] in self.adapter.sessions:
            _fail("closed session was restored as a live native session")
        if request["session"] not in self.adapter.closed_sessions:
            _fail("closed session is missing its retained archive")
        expected_memories = {
            work.runtime.memory.session_id for work in self.adapter.sessions.values()
        }
        if set(self.adapter.runtime.memory.sessions) != expected_memories:
            _fail("closed session retained native runtime memory")
        if (
            set(self.adapter.sessions) != set(checkpoint["sessions"])
            or not _same(self.adapter.closed_sessions, checkpoint["closedSessions"])
            or not _same(self.adapter.attempts, checkpoint["attempts"])
            or not _same(self.adapter.openings, checkpoint["openings"])
        ):
            _fail("live closed adapter differs from its exact checkpoint archive")
        if restore_receipt is not None and (
            restore_receipt.get("hostDispatches") != 0
            or restore_receipt.get("restoredSessions") != len(checkpoint["sessions"])
            or restore_receipt.get("restored") is not True
        ):
            _fail("closed checkpoint recovery dispatched host work or retained a live target")

    def durable_close(self, *, identity, requestBase64, preCheckpointDigest):
        self.assert_operation("durable-close")
        identity, raw, request, frame = self._close_input(
            identity, requestBase64, preCheckpointDigest
        )
        with self.adapter.lock:
            try:
                readback = self.journal.inspect(identity, frame, require_latest=True)
                if readback["status"] == "unknown":
                    raise NativeRequestOutcomeUnknown(
                        "Native close outcome is unknown; durable close cannot be retried"
                    )
                if readback["status"] == "completed":
                    value, response_raw, _response = self._close_envelope(
                        readback["result_bytes"], request, raw, preCheckpointDigest
                    )
                    if not _same(self.adapter.checkpoint(), value["postCheckpoint"]):
                        _fail("live adapter differs from the retained terminal close poststate")
                    self._assert_closed_adapter(request, value["postCheckpoint"])
                    return self._close_receipt(
                        identity, raw, value, response_raw, readback["request_id"]
                    )

                before = self.adapter.checkpoint()
                if before["digest"] != preCheckpointDigest:
                    raise NativeRequestConflict("Native pre-request checkpoint differs")
                self._assert_terminal_close(request, before)

                def execute(_frame):
                    current = self.adapter.checkpoint()
                    if current["digest"] != preCheckpointDigest:
                        raise NativeRequestConflict("Native pre-request checkpoint differs")
                    self._assert_terminal_close(request, current)
                    response = self.adapter.cancel(
                        session=request["session"], reason=request["reason"]
                    )
                    post = self.adapter.checkpoint()
                    reply = {"id": request["id"], "ok": True, "result": response}
                    self._check_close_checkpoints(
                        request, reply, post, current, preCheckpointDigest
                    )
                    return _canonical(
                        {
                            "schema": "semantic-native-durable-close-outcome/8",
                            "requestSha256": _sha(raw),
                            "preCheckpointDigest": preCheckpointDigest,
                            "responseBase64": base64.b64encode(_reply_bytes(reply)).decode(
                                "ascii"
                            ),
                            "postCheckpoint": post,
                            "preCheckpoint": current,
                        }
                    )

                stored = self.journal.execute(identity, frame, execute)
                value, response_raw, _response = self._close_envelope(
                    stored, request, raw, preCheckpointDigest
                )
                if not _same(self.adapter.checkpoint(), value["postCheckpoint"]):
                    _fail("whole live adapter differs from saved terminal close poststate")
                self._assert_closed_adapter(request, value["postCheckpoint"])
                readback = self.journal.inspect(identity, frame, require_latest=True)
                if readback.get("result_bytes") != stored:
                    _fail("independent durable terminal close readback differs")
                self._recoverable = (copy.deepcopy(identity), frame)
                self._started = True
                return self._close_receipt(
                    identity, raw, value, response_raw, readback["request_id"]
                )
            except BaseException:
                self.fenced = True
                raise

    def recover_close(self, *, identity, requestBase64, preCheckpointDigest):
        self.assert_operation("recover-close")
        identity, raw, request, frame = self._close_input(
            identity, requestBase64, preCheckpointDigest
        )
        with self.adapter.lock:
            if (
                self.adapter.sessions
                or self.adapter.attempts
                or self.adapter.closed_sessions
                or self.adapter.openings
                or self.adapter.runtime.memory.sessions
            ):
                _fail("recovery requires a fresh entire native adapter")
            readback = self.journal.inspect(identity, frame, require_latest=True)
            if readback["status"] == "never-admitted":
                raise NativeRequestConflict(
                    "Native close was never admitted; recovery cannot execute it"
                )
            if readback["status"] != "completed":
                raise NativeRequestOutcomeUnknown(
                    "Native close outcome is unknown; recovery cannot retry it"
                )
            value, response_raw, _response = self._close_envelope(
                readback["result_bytes"], request, raw, preCheckpointDigest
            )
            try:
                restored = self.adapter.restore(checkpoint=value["postCheckpoint"])
                if not _same(self.adapter.checkpoint(), value["postCheckpoint"]):
                    _fail("whole restored adapter differs from saved terminal close poststate")
                self._assert_closed_adapter(request, value["postCheckpoint"], restored)
                self._recoverable = (copy.deepcopy(identity), frame)
                self._started = True
                return self._close_receipt(
                    identity,
                    raw,
                    value,
                    response_raw,
                    readback["request_id"],
                    restored,
                )
            except BaseException:
                self.fenced = True
                raise
