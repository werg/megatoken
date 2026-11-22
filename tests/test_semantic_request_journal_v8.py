from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from megatoken.semantic_request_journal_v8 import (
    DurableNativeRequestJournal,
    NativeRequestConflict,
    NativeRequestJournalError,
    NativeRequestOutcomeUnknown,
)

FIXTURE = Path(__file__).parent / "fixtures" / "semantic_request_journal_worker_v8.py"


def identity(ordinal=1, **changes):
    value = {
        "schema": "semantic-native-request-identity/1",
        "session_id": "session-a",
        "stream_id": "stream-a",
        "session_generation": 1,
        "ordinal": ordinal,
        "operation": "generate",
        "model_identity": {"checkpoint": "a" * 64, "tokenizer": "b" * 64},
    }
    value.update(changes)
    return value


def run_worker(root, mode, **env):
    child_env = dict(
        os.environ,
        PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"),
        JOURNAL_ROOT=str(root),
        JOURNAL_MODE=mode,
        **env,
    )
    return subprocess.run(
        [sys.executable, str(FIXTURE)], env=child_env, capture_output=True, text=True, timeout=15
    )


def test_exact_result_persisted_and_same_slot_is_idempotent(tmp_path):
    journal = DurableNativeRequestJournal(tmp_path / "journal")
    calls = []
    request = b'{"op":"generate","prompt":"x"}'
    result = journal.execute(
        identity(), request, lambda data: calls.append(data) or b'{"tokens":[1]}'
    )
    assert result == b'{"tokens":[1]}'
    inspected = journal.inspect(identity(), request)
    assert inspected["status"] == "completed"
    assert inspected["result_bytes"] == result
    assert len(inspected["request_id"]) == 64
    assert len(inspected["outcome_sha256"]) == 64
    assert (
        journal.execute(identity(), request, lambda _: pytest.fail("callback repeated")) == result
    )
    assert calls == [request]


def test_payload_operation_and_model_substitution_conflict(tmp_path):
    journal = DurableNativeRequestJournal(tmp_path / "journal")
    request = b"same"
    journal.execute(identity(), request, lambda _: b"ok")
    with pytest.raises(NativeRequestConflict):
        journal.execute(identity(), b"changed", lambda _: b"bad")
    with pytest.raises(NativeRequestConflict):
        journal.execute(identity(operation="score"), request, lambda _: b"bad")
    with pytest.raises(NativeRequestConflict):
        journal.execute(
            identity(model_identity={"checkpoint": "c" * 64, "tokenizer": "b" * 64}),
            request,
            lambda _: b"bad",
        )


def test_ordinals_are_contiguous_and_scopes_are_separate(tmp_path):
    journal = DurableNativeRequestJournal(tmp_path / "journal")
    with pytest.raises(NativeRequestConflict):
        journal.execute(identity(2), b"later", lambda _: b"bad")
    journal.execute(identity(), b"first", lambda _: b"one")
    journal.execute(identity(2), b"second", lambda _: b"two")
    other = identity(session_generation=2)
    assert journal.execute(other, b"first", lambda _: b"new") == b"new"


def test_crash_before_intent_publication_is_never_admitted(tmp_path):
    child = run_worker(tmp_path / "journal", "before-admission")
    assert child.returncode == 70
    journal = DurableNativeRequestJournal(tmp_path / "journal")
    request = b"request"
    scope = next((tmp_path / "journal").iterdir())
    assert any(item.name.startswith(".tmp-") for item in scope.iterdir())
    assert journal.inspect(identity(), request)["status"] == "never-admitted"
    assert not any(item.name.startswith(".tmp-") for item in scope.iterdir())
    assert journal.execute(identity(), request, lambda _: b"result") == b"result"


def test_crash_after_intent_leaves_unknown_and_never_reexecutes(tmp_path):
    child = run_worker(tmp_path / "journal", "after-admission")
    assert child.returncode == 71
    journal = DurableNativeRequestJournal(tmp_path / "journal")
    request = b"request"
    assert journal.inspect(identity(), request)["status"] == "unknown"
    with pytest.raises(NativeRequestOutcomeUnknown):
        journal.execute(identity(), request, lambda _: pytest.fail("unknown request redispatched"))


def test_crash_after_outcome_publication_is_readable_without_callback(tmp_path):
    child = run_worker(tmp_path / "journal", "after-outcome")
    assert child.returncode == 72
    journal = DurableNativeRequestJournal(tmp_path / "journal")
    request = b"request"
    result = journal.inspect(identity(), request)
    assert result["status"] == "completed"
    assert result["result_bytes"] == b"response"
    assert (
        journal.execute(
            identity(), request, lambda _: pytest.fail("completed request redispatched")
        )
        == b"response"
    )


def test_live_second_process_cannot_enter_same_scope(tmp_path):
    first = subprocess.Popen(
        [sys.executable, str(FIXTURE)],
        env=dict(
            os.environ,
            PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"),
            JOURNAL_ROOT=str(tmp_path / "journal"),
            JOURNAL_MODE="hold",
        ),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert first.stdout.readline().strip() == "ADMITTED"
        second = run_worker(tmp_path / "journal", "contend")
        assert second.returncode == 73
        first.stdin.write("continue\n")
        first.stdin.flush()
        assert first.stdout.readline().strip() == "RESULT:response"
        assert first.wait(timeout=10) == 0
    finally:
        if first.poll() is None:
            first.kill()
            first.wait(timeout=10)


def test_model_identity_is_frozen_for_session_generation(tmp_path):
    journal = DurableNativeRequestJournal(tmp_path / "journal")
    journal.execute(identity(), b"one", lambda _: b"one-result")
    with pytest.raises(NativeRequestConflict):
        journal.execute(
            identity(2, model_identity={"checkpoint": "c" * 64, "tokenizer": "b" * 64}),
            b"two",
            lambda _: b"wrong",
        )


def test_scope_creation_sync_failure_prevents_callback(tmp_path):
    class FailScopeDirectorySync(DurableNativeRequestJournal):
        def _after_directory_fsync(self, path, reason):
            if reason == "mkdir-self" and path.parent == self.root:
                raise OSError("injected scope directory sync failure")

    journal = FailScopeDirectorySync(tmp_path / "journal")
    calls = []
    with pytest.raises(NativeRequestJournalError):
        journal.execute(identity(), b"request", lambda _: calls.append("called") or b"result")
    assert calls == []


def test_supersession_authority_binds_the_exact_later_mutation(tmp_path):
    journal = DurableNativeRequestJournal(tmp_path / "journal")
    slot = identity()
    request = b"exact-frame"
    first = {"id": "later-a", "op": "bootstrap-orchestration-child"}
    other = {"id": "later-b", "op": "cancel-orchestration"}
    journal.execute(slot, request, lambda _: b"known-result")

    journal.assert_not_superseded(slot, request)
    with pytest.raises(NativeRequestConflict, match="not superseded"):
        journal.assert_superseded_by(slot, request, first)

    journal.supersede(slot, request, first)
    journal.assert_superseded_by(slot, request, first)
    journal.supersede(slot, request, first)
    with pytest.raises(NativeRequestConflict, match="different mutation"):
        journal.assert_superseded_by(slot, request, other)
    with pytest.raises(NativeRequestConflict, match="different mutation"):
        journal.supersede(slot, request, other)
    with pytest.raises(NativeRequestConflict, match="already superseded"):
        journal.assert_not_superseded(slot, request)


def test_lost_outcome_publication_ack_is_recovered_only_after_fresh_sync(tmp_path):
    class LoseOutcomeAck(DurableNativeRequestJournal):
        publications = 0

        def _after_directory_fsync(self, path, reason):
            if reason == "record-publish":
                self.publications += 1
                if self.publications == 2:
                    raise OSError("injected outcome publication acknowledgement loss")

    journal = LoseOutcomeAck(tmp_path / "journal")
    with pytest.raises(OSError):
        journal.execute(identity(), b"request", lambda _: b"response")
    fresh = DurableNativeRequestJournal(tmp_path / "journal")
    observed = fresh.inspect(identity(), b"request")
    assert observed["status"] == "completed"
    assert observed["result_bytes"] == b"response"


def test_readback_sync_failure_is_not_reported_as_completed(tmp_path):
    path = tmp_path / "journal"
    DurableNativeRequestJournal(path).execute(identity(), b"request", lambda _: b"response")

    class FailReadSync(DurableNativeRequestJournal):
        def _after_directory_fsync(self, directory, reason):
            if reason == "record-read":
                raise OSError("injected readback sync failure")

    with pytest.raises(OSError):
        FailReadSync(path).inspect(identity(), b"request")


def test_orphan_outcome_and_missing_lower_slot_refuse(tmp_path):
    journal = DurableNativeRequestJournal(tmp_path / "journal")
    scope, _ = journal._scope(identity())
    _, orphan = journal._paths(scope, 1)
    orphan.write_bytes(b"{}")
    orphan.chmod(0o600)
    with pytest.raises(NativeRequestJournalError, match="orphan outcome"):
        journal.execute(identity(), b"request", lambda _: pytest.fail("orphan dispatched"))

    orphan.unlink()
    journal.execute(identity(), b"first", lambda _: b"done")
    journal.execute(identity(2), b"second", lambda _: b"done")
    intent, _ = journal._paths(scope, 1)
    intent.unlink()
    with pytest.raises(NativeRequestJournalError, match="missing request intent"):
        journal.inspect(identity(3), b"third")
