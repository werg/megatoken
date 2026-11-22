"""Checkpoint-bound readback for committed native v8 requests.

This bridge stores the exact response bytes together with the native adapter's
post-request checkpoint. It returns a stored response from a fresh adapter only
after that adapter has actually restored and re-captured the same checkpoint.
"""
from __future__ import annotations

import base64
import fcntl
import hashlib
import json
import os
import stat

from megatoken.data.semantic_json import stable_algorithm_hash
from megatoken.io import strict_json_loads
from megatoken.semantic_request_journal_v8 import (
    DurableNativeRequestJournal,
    NativeRequestJournalError,
    NativeRequestOutcomeUnknown,
)

_BRIDGE_SCHEMA = "semantic-native-committed-response/1"


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


class NativePoststateMismatch(NativeRequestJournalError):
    pass


def validate_execution_lock_fd(fd: int, expected_path: str) -> None:
    """Prove the inherited descriptor is the already locked expected inode."""
    from pathlib import Path

    if type(fd) is not int or fd < 4:
        raise NativeRequestJournalError("execution lock must use an inherited descriptor >= 4")
    if not isinstance(expected_path, str) or not os.path.isabs(expected_path):
        raise NativeRequestJournalError("execution lock path must be absolute")
    probe = None
    try:
        inherited = os.fstat(fd)
        if (
            not stat.S_ISREG(inherited.st_mode)
            or inherited.st_uid != os.geteuid()
            or stat.S_IMODE(inherited.st_mode) & 0o077
            or inherited.st_nlink != 1
        ):
            raise NativeRequestJournalError("unsafe inherited execution lock descriptor")
        flags = os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
        probe = os.open(Path(expected_path), flags)
        expected = os.fstat(probe)
        if (
            not stat.S_ISREG(expected.st_mode)
            or expected.st_uid != os.geteuid()
            or stat.S_IMODE(expected.st_mode) & 0o077
            or expected.st_nlink != 1
            or (expected.st_dev, expected.st_ino) != (inherited.st_dev, inherited.st_ino)
        ):
            raise NativeRequestJournalError("inherited lock descriptor does not match expected lock path")
        try:
            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            fcntl.flock(probe, fcntl.LOCK_UN)
            raise NativeRequestJournalError("execution lock inode was not already locked")
        # Reacquiring through the inherited descriptor must succeed on the same
        # open-file description; an independently opened contender must fail.
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        raise NativeRequestJournalError("inherited open-file-description lock proof failed") from exc
    finally:
        if probe is not None:
            os.close(probe)


def _checkpoint_digest(checkpoint: object) -> str:
    if not isinstance(checkpoint, dict) or type(checkpoint.get("digest")) is not str:
        raise NativePoststateMismatch("invalid native post-request checkpoint")
    body = {key: value for key, value in checkpoint.items() if key != "digest"}
    digest = stable_algorithm_hash(body)
    if checkpoint["digest"] != digest:
        raise NativePoststateMismatch("native post-request checkpoint digest differs")
    return digest


def _assert_safe_committed_checkpoint(checkpoint: dict, identity: dict, request: dict, reply: dict) -> None:
    _checkpoint_digest(checkpoint)
    if checkpoint.get("schema") != "semantic-model-checkpoint/8":
        raise NativePoststateMismatch("unsupported native post-request checkpoint")
    model = identity["model_identity"]
    initialized = checkpoint.get("initialization")
    if (
        not isinstance(initialized, dict)
        or initialized.get("checkpointIdentity") != model["checkpoint"]
        or initialized.get("tokenizerIdentity") != model["tokenizer"]
        or initialized.get("trainingIdentity") != model.get("training")
    ):
        raise NativePoststateMismatch("checkpoint model identity differs from request identity")
    if identity.get("operation") != "commit" or request.get("op") != "commit":
        raise NativePoststateMismatch("only native commit response readback is supported")
    if reply.get("id") != request.get("id"):
        raise NativePoststateMismatch("commit response ID differs from exact request")
    if reply.get("ok") is not True or not isinstance(reply.get("result"), dict):
        raise NativePoststateMismatch("only a successful native commit can be checkpointed")
    result = reply["result"]
    session = request.get("session")
    selection = request.get("selection")
    if (
        not isinstance(session, str)
        or not isinstance(selection, dict)
        or result.get("schema") != "semantic-native-commit/8"
        or result.get("committed") is not True
        or result.get("selectionId") != selection.get("selectionId")
    ):
        raise NativePoststateMismatch("native commit response binding differs")
    attempts = checkpoint.get("attempts")
    sessions = checkpoint.get("sessions")
    if not isinstance(attempts, list) or not isinstance(sessions, dict):
        raise NativePoststateMismatch("checkpoint lacks native attempt/session ledgers")
    saved = sessions.get(session)
    if not isinstance(saved, dict) or not isinstance(saved.get("public"), dict):
        raise NativePoststateMismatch("checkpoint lacks the committed public session")
    if stable_algorithm_hash(saved["public"]) != result.get("publicDigest"):
        raise NativePoststateMismatch("commit response public digest differs from checkpoint")
    matches = [
        row for row in attempts
        if isinstance(row, dict) and row.get("selectionId") == result["selectionId"]
    ]
    if (
        len(matches) != 1
        or matches[0].get("status") != "committed"
        or matches[0].get("session") != session
    ):
        raise NativePoststateMismatch("checkpoint does not contain this committed attempt")
    if any(
        isinstance(row, dict) and row.get("status") == "awaiting-commit"
        for row in attempts
    ) or any(
        isinstance(row, dict) and row.get("unacknowledgedSelection") is not None
        for row in sessions.values()
    ):
        raise NativePoststateMismatch("checkpoint contains another unacknowledged selection")


class CheckpointBoundNativeRequestBridge:
    """Bind exact commit response bytes to validated post-commit native state."""

    def __init__(self, journal: DurableNativeRequestJournal):
        if not isinstance(journal, DurableNativeRequestJournal):
            raise TypeError("journal must be a DurableNativeRequestJournal")
        self.journal = journal

    def execute_commit(self, adapter, identity: dict, request_bytes: bytes, callback) -> bytes:
        """Execute or read back one commit; never redispatch an admitted unknown slot.

        The callback performs the native operation and returns its exact JSON reply
        bytes. Its reply and the checkpoint captured immediately afterward are
        stored as one immutable journal outcome.
        """
        if not isinstance(identity, dict) or identity.get("operation") != "commit":
            raise NativePoststateMismatch("bridge accepts only commit requests")
        expected_model = identity.get("model_identity")
        adapter_model = {
            "checkpoint": getattr(adapter, "model_identity", None),
            "tokenizer": getattr(getattr(adapter, "codec", None), "fingerprint", None),
        }
        training_identity = getattr(adapter, "training_identity", None)
        if training_identity is not None:
            adapter_model["training"] = training_identity
        if expected_model != adapter_model:
            raise NativePoststateMismatch("request model identity differs before dispatch")
        try:
            request = strict_json_loads(request_bytes.decode("utf-8", errors="strict"))
        except (UnicodeError, ValueError) as exc:
            raise NativeRequestJournalError("commit request is not strict UTF-8 JSON") from exc
        if (
            not isinstance(request, dict)
            or not isinstance(request.get("id"), str)
            or not request["id"]
            or request.get("op") != "commit"
            or request.get("nativeRequest") != identity
        ):
            raise NativePoststateMismatch("request envelope or operation differs from commit identity")
        captured = False

        def execute_once(exact_request: bytes) -> bytes:
            nonlocal captured
            reply_bytes = callback(exact_request)
            if not isinstance(reply_bytes, bytes):
                raise NativeRequestJournalError("commit callback must return exact response bytes")
            try:
                reply = strict_json_loads(reply_bytes.decode("utf-8", errors="strict"))
            except (UnicodeError, ValueError) as exc:
                raise NativeRequestJournalError("commit callback response is not strict JSON") from exc
            checkpoint = adapter.checkpoint()
            _assert_safe_committed_checkpoint(checkpoint, identity, request, reply)
            body = {
                "schema": _BRIDGE_SCHEMA,
                "identity": identity,
                "request_sha256": _sha(exact_request),
                "response_b64": base64.b64encode(reply_bytes).decode("ascii"),
                "postCheckpoint": checkpoint,
            }
            body["digest"] = stable_algorithm_hash(body)
            captured = True
            return _canonical(body)

        try:
            envelope_bytes = self.journal.execute(identity, request_bytes, execute_once)
        except Exception as original_error:
            try:
                observed = self.journal.inspect(identity, request_bytes)
            except Exception as readback_error:
                raise NativeRequestOutcomeUnknown(
                    "commit may have been admitted but independent durable readback failed"
                ) from readback_error
            if observed["status"] == "unknown":
                raise NativeRequestOutcomeUnknown(
                    "commit intent is durable but no outcome is known; request will not be retried"
                ) from original_error
            if observed["status"] == "completed":
                envelope_bytes = observed["result_bytes"]
            else:
                raise
        envelope = self._decode_envelope(envelope_bytes, identity, request_bytes, request)
        checkpoint = envelope["postCheckpoint"]
        reply_bytes = base64.b64decode(envelope["response_b64"], validate=True)
        reply = strict_json_loads(reply_bytes.decode("utf-8", errors="strict"))
        _assert_safe_committed_checkpoint(checkpoint, identity, request, reply)
        if captured:
            current = adapter.checkpoint()
            if _canonical(current) != _canonical(checkpoint):
                raise NativePoststateMismatch("live native state differs from journaled poststate")
        else:
            runtime = getattr(adapter, "runtime", None)
            memory = getattr(runtime, "memory", None)
            runtime_sessions = getattr(memory, "sessions", {})
            pristine = (
                not getattr(adapter, "attempts", ())
                and not getattr(adapter, "sessions", {})
                and not runtime_sessions
            )
            try:
                current = adapter.checkpoint()
            except Exception:
                if not pristine:
                    raise
                current = None
            if current is None or _canonical(current) != _canonical(checkpoint):
                if not pristine:
                    raise NativePoststateMismatch(
                        "active native state differs from journaled poststate"
                    )
                adapter.restore(checkpoint=checkpoint)
                current = adapter.checkpoint()
            if _canonical(current) != _canonical(checkpoint):
                raise NativePoststateMismatch("restored native state differs from journaled poststate")
        return reply_bytes

    def _decode_envelope(self, raw: bytes, identity: dict, request_bytes: bytes, request: dict) -> dict:
        try:
            envelope = json.loads(raw.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise NativeRequestJournalError("corrupt checkpoint-bound response record") from exc
        if (
            not isinstance(envelope, dict)
            or set(envelope) != {
                "schema", "identity", "request_sha256", "response_b64",
                "postCheckpoint", "digest",
            }
            or _canonical(envelope) != raw
        ):
            raise NativeRequestJournalError("invalid checkpoint-bound response fields")
        body = {key: value for key, value in envelope.items() if key != "digest"}
        if (
            envelope["schema"] != _BRIDGE_SCHEMA
            or envelope["identity"] != identity
            or envelope["request_sha256"] != _sha(request_bytes)
            or envelope["digest"] != stable_algorithm_hash(body)
            or not isinstance(envelope["response_b64"], str)
        ):
            raise NativeRequestJournalError("checkpoint-bound response identity differs")
        try:
            response = base64.b64decode(envelope["response_b64"], validate=True)
        except ValueError as exc:
            raise NativeRequestJournalError("invalid exact native response bytes") from exc
        if base64.b64encode(response).decode("ascii") != envelope["response_b64"]:
            raise NativeRequestJournalError("noncanonical native response encoding")
        return envelope
