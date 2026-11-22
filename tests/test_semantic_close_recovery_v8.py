"""Exact recovery of terminal coordinator closure on the native runtime.

Lifecycle decisions and lexical rows are supplied as explicit engineering
controls. These tests establish native close/recovery behavior, not learned
model quality or source admission.
"""

from __future__ import annotations

import base64
import hashlib
import json

import pytest

from megatoken.data.semantic_json import stable_algorithm_hash
from megatoken.semantic_commit_recovery_v8 import DurableNativeCommitBridge
from megatoken.semantic_request_journal_v8 import (
    DurableNativeRequestJournal,
    NativeRequestConflict,
    NativeRequestJournalError,
    NativeRequestOutcomeUnknown,
    _canonical,
)
from scripts.serve_semantic_v8 import dispatch
from tests.test_semantic_commit_recovery_v8 import execution_lease
from tests.test_semantic_v8_closed_session_restore import with_session
from tests.test_semantic_v8_orchestration import (
    acknowledge,
    chosen,
    commit_child,
    create_arguments,
    example,
    open_coordinator,
    real_runtime,
    select_child,
    targets,
)


def _terminal_coordinator(open_adapter, decoder, terminal):
    adapter = open_adapter()
    opened = open_coordinator(adapter)
    if terminal == "cancelled":
        adapter.cancel_orchestration(
            session=opened["session"],
            track=opened["track"],
            reason="fixture coordinator cancellation",
        )
    else:
        document = example()
        created = acknowledge(
            adapter,
            opened,
            chosen(adapter, opened, "track_create", create_arguments(opened)),
        )
        child = created["track"]["address"]
        adapter.bootstrap_orchestration_child(
            session=opened["session"], track=opened["track"], childTrack=child
        )
        acknowledge(
            adapter,
            opened,
            chosen(
                adapter,
                opened,
                "track_join",
                {
                    "address": opened["track"],
                    "completion_dependency": created["completion_dependency"],
                },
            ),
        )
        decoder["records"] = targets(document)
        for index in range(len(document["contexts"])):
            commit_child(adapter, document, index, select_child(adapter, document, index))
        completion = adapter.complete(
            session=opened["session"],
            public=document["public"],
            contexts=document["contexts"],
            invocationId=document["contexts"][0]["invocationId"],
        )["childCompletion"]
        acknowledge(
            adapter,
            opened,
            chosen(
                adapter,
                opened,
                "track_progress",
                {"address": opened["track"], "progress": json.dumps(completion["completion"])},
            ),
        )
        acknowledge(
            adapter,
            opened,
            chosen(adapter, opened, "track_complete", {"address": opened["track"]}),
        )
        assert adapter.status_orchestration(
            session=opened["session"], track=opened["track"]
        )["status"] == "completed"

    sibling = with_session(example(), "sibling")
    public = sibling["public"]
    public["timeline"] = public["timeline"][:2]
    adapter.open_orchestration(
        session="sibling", public=public, origin="controlled", childBudget=64
    )
    return adapter, opened, sibling if terminal == "completed" else None


def _close_arguments(adapter, lease, session, *, ordinal=1, reason="terminal coordinator close"):
    request = {"id": "terminal-close", "op": "cancel", "session": session, "reason": reason}
    return {
        "identity": {
            "schema": "semantic-native-request-identity/1",
            "session_id": session,
            "stream_id": lease.stream_id,
            "session_generation": 1,
            "ordinal": ordinal,
            "operation": "cancel",
            "model_identity": {
                "checkpoint": adapter.model_identity,
                "tokenizer": adapter.codec.fingerprint,
            },
        },
        "requestBase64": base64.b64encode(_canonical(request) + b"\n").decode("ascii"),
        "preCheckpointDigest": adapter.checkpoint()["digest"],
    }


def _tamper_outcome(journal, mutate):
    path = next(journal.root.glob("*/00000000000000000001.outcome.json"))
    outer = json.loads(path.read_text(encoding="utf-8"))
    stored = base64.b64decode(outer["result_b64"])
    outcome = json.loads(stored.decode("utf-8"))
    mutate(outcome)
    stored = _canonical(outcome)
    outer["result_b64"] = base64.b64encode(stored).decode("ascii")
    outer["result_sha256"] = hashlib.sha256(stored).hexdigest()
    path.write_bytes(_canonical(outer))


@pytest.mark.parametrize("terminal", ["completed", "cancelled"])
def test_terminal_close_recovers_closed_archive_without_live_target(tmp_path, terminal):
    with real_runtime() as (open_adapter, decoder):
        with execution_lease(tmp_path / "owner", session="teacher1") as lease:
            adapter, opened, _sibling = _terminal_coordinator(open_adapter, decoder, terminal)
            before = adapter.checkpoint()
            target_memory_id = adapter.sessions[opened["session"]].runtime.memory.session_id
            args = _close_arguments(adapter, lease, opened["session"])
            bridge = DurableNativeCommitBridge(
                adapter, DurableNativeRequestJournal(tmp_path / "journal"), lease
            )
            result = bridge.durable_close(**args)
            assert result["schema"] == "semantic-native-durable-close/8"
            assert result["recovered"] is False
            assert result["postCheckpoint"]["sessions"] == {"sibling": before["sessions"]["sibling"]}
            assert result["postCheckpoint"]["closedSessions"][opened["session"]] == before[
                "sessions"
            ][opened["session"]]
            assert set(adapter.sessions) == {"sibling"}
            assert target_memory_id not in adapter.runtime.memory.sessions

            if terminal == "completed":
                decoder["records"] = targets(example())
            fresh = open_adapter()
            fresh_bridge = DurableNativeCommitBridge(fresh, bridge.journal, lease)
            recovered = fresh_bridge.recover_close(**args)
            assert recovered["recovered"] is True
            assert recovered["responseBase64"] == result["responseBase64"]
            assert recovered["postCheckpoint"] == result["postCheckpoint"]
            assert recovered["restoreReceipt"]["restoredSessions"] == 1
            assert recovered["restoreReceipt"]["hostDispatches"] == 0
            assert opened["session"] not in fresh.sessions
            assert opened["session"] in fresh.closed_sessions
            assert set(fresh.sessions) == {"sibling"}
            assert set(fresh.runtime.memory.sessions) == {
                fresh.sessions["sibling"].runtime.memory.session_id
            }


def test_unknown_terminal_close_is_never_retried(tmp_path):
    class LoseCloseOutcome(DurableNativeRequestJournal):
        def _after_temp_fsync(self, target_name, temp_path):
            if target_name.endswith("00000000000000000001.outcome.json"):
                raise OSError("terminal close outcome not published")

    with real_runtime() as (open_adapter, decoder):
        with execution_lease(tmp_path / "owner", session="teacher1") as lease:
            adapter, opened, _sibling = _terminal_coordinator(open_adapter, decoder, "cancelled")
            args = _close_arguments(adapter, lease, opened["session"])
            bridge = DurableNativeCommitBridge(
                adapter, LoseCloseOutcome(tmp_path / "journal"), lease
            )
            with pytest.raises(OSError, match="outcome not published"):
                bridge.durable_close(**args)
            assert bridge.fenced
            assert opened["session"] not in adapter.sessions
            fresh = open_adapter()
            calls = []
            original = fresh.cancel

            def observed_cancel(**request):
                calls.append(request)
                return original(**request)

            fresh.cancel = observed_cancel
            fresh_bridge = DurableNativeCommitBridge(
                fresh, DurableNativeRequestJournal(tmp_path / "journal"), lease
            )
            with pytest.raises(NativeRequestOutcomeUnknown, match="cannot retry"):
                fresh_bridge.recover_close(**args)
            assert calls == []
            assert not fresh.sessions and not fresh.closed_sessions
            assert not fresh.runtime.memory.sessions


def test_active_session_cannot_enter_durable_terminal_close(tmp_path):
    with real_runtime() as (open_adapter, _decoder):
        with execution_lease(tmp_path / "owner", session="teacher1") as lease:
            adapter = open_adapter()
            opened = open_coordinator(adapter)
            before = adapter.checkpoint()
            journal = DurableNativeRequestJournal(tmp_path / "journal")
            bridge = DurableNativeCommitBridge(adapter, journal, lease)
            args = _close_arguments(adapter, lease, opened["session"])
            with pytest.raises(NativeRequestJournalError, match="terminal"):
                bridge.durable_close(**args)
            assert adapter.checkpoint() == before
            assert not list(journal.root.glob("*/*.intent.json"))


def test_terminal_close_recovery_refuses_a_never_admitted_request(tmp_path):
    with real_runtime() as (open_adapter, decoder):
        with execution_lease(tmp_path / "owner", session="teacher1") as lease:
            original, opened, _sibling = _terminal_coordinator(
                open_adapter, decoder, "cancelled"
            )
            args = _close_arguments(original, lease, opened["session"])
            journal = DurableNativeRequestJournal(tmp_path / "journal")
            fresh = open_adapter()
            calls = []
            cancel = fresh.cancel

            def observed_cancel(**request):
                calls.append(request)
                return cancel(**request)

            fresh.cancel = observed_cancel
            bridge = DurableNativeCommitBridge(fresh, journal, lease)
            with pytest.raises(NativeRequestConflict, match="never admitted"):
                bridge.recover_close(**args)
            assert calls == []
            assert not fresh.sessions and not fresh.closed_sessions
            assert not fresh.runtime.memory.sessions


def test_changed_close_reason_and_rehashed_closed_archive_are_refused(tmp_path):
    with real_runtime() as (open_adapter, decoder):
        with execution_lease(tmp_path / "owner", session="teacher1") as lease:
            adapter, opened, _sibling = _terminal_coordinator(open_adapter, decoder, "cancelled")
            args = _close_arguments(adapter, lease, opened["session"])
            bridge = DurableNativeCommitBridge(
                adapter, DurableNativeRequestJournal(tmp_path / "journal"), lease
            )
            bridge.durable_close(**args)
            changed = json.loads(base64.b64decode(args["requestBase64"]))
            changed["reason"] = "different exact reason"
            changed_args = {
                **args,
                "requestBase64": base64.b64encode(_canonical(changed) + b"\n").decode(
                    "ascii"
                ),
            }
            fresh = open_adapter()
            fresh_bridge = DurableNativeCommitBridge(fresh, bridge.journal, lease)
            with pytest.raises(NativeRequestConflict, match="exact bytes differ"):
                fresh_bridge.recover_close(**changed_args)
            assert not fresh.sessions and not fresh.closed_sessions

    with real_runtime() as (open_adapter, decoder):
        with execution_lease(tmp_path / "owner", session="teacher1") as lease:
            adapter, opened, _sibling = _terminal_coordinator(open_adapter, decoder, "cancelled")
            bridge = DurableNativeCommitBridge(
                adapter, DurableNativeRequestJournal(tmp_path / "tampered-journal"), lease
            )
            args = _close_arguments(adapter, lease, opened["session"])
            bridge.durable_close(**args)

            def corrupt(outcome):
                closed = outcome["postCheckpoint"]["closedSessions"][opened["session"]]
                closed["continuationDigest"] = "f" * 64
                checkpoint = outcome["postCheckpoint"]
                checkpoint["digest"] = stable_algorithm_hash(
                    {key: value for key, value in checkpoint.items() if key != "digest"}
                )

            _tamper_outcome(bridge.journal, corrupt)
            fresh = open_adapter()
            fresh_bridge = DurableNativeCommitBridge(fresh, bridge.journal, lease)
            with pytest.raises(NativeRequestJournalError, match="closed-session archive"):
                fresh_bridge.recover_close(**args)
            assert not fresh.sessions and not fresh.closed_sessions


def test_duplicate_terminal_close_returns_known_outcome_without_a_second_close(tmp_path):
    with real_runtime() as (open_adapter, decoder):
        with execution_lease(tmp_path / "owner", session="teacher1") as lease:
            adapter, opened, _sibling = _terminal_coordinator(open_adapter, decoder, "cancelled")
            args = _close_arguments(adapter, lease, opened["session"])
            bridge = DurableNativeCommitBridge(
                adapter, DurableNativeRequestJournal(tmp_path / "journal"), lease
            )
            first = bridge.durable_close(**args)
            calls = []
            original = adapter.cancel

            def observed_cancel(**request):
                calls.append(request)
                return original(**request)

            adapter.cancel = observed_cancel
            duplicate = dispatch(
                adapter,
                {"id": "duplicate-terminal-close", "op": "durable-close", **args},
                commit_bridge=bridge,
            )
            assert duplicate == first
            assert calls == []
