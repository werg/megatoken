"""Exact child-bootstrap durability on the real native runtime.

The coordinator and child lifecycle use the actual recurrent runtime. The
lifecycle action is explicitly controlled engineering input; it is not learned
model output or source-admission evidence.
"""

from __future__ import annotations

import base64
import json
import subprocess
import sys
from pathlib import Path

import pytest

from megatoken.semantic_commit_recovery_v8 import DurableNativeCommitBridge
from megatoken.semantic_request_journal_v8 import (
    DurableNativeRequestJournal,
    NativeRequestConflict,
    NativeRequestOutcomeUnknown,
    _canonical,
)
from scripts.serve_semantic_v8 import dispatch
from tests.test_semantic_commit_recovery_v8 import execution_lease
from tests.test_semantic_v8_orchestration import (
    chosen,
    create_arguments,
    open_coordinator,
    real_runtime,
)


def bootstrap_case(tmp_path, lease, open_adapter, journal=None):
    adapter = open_adapter()
    opened = open_coordinator(adapter)
    selected = chosen(adapter, opened, "track_create", create_arguments(opened))
    request = {
        "id": "create-ack",
        "op": "commit-orchestration",
        "session": opened["session"],
        "track": opened["track"],
        "record": selected["record"],
        "selection": selected["selection"],
    }
    if journal is None:
        journal = DurableNativeRequestJournal(tmp_path / "journal")
    bridge = DurableNativeCommitBridge(adapter, journal, lease)
    create_args = {
        "identity": {
            "schema": "semantic-native-request-identity/1",
            "session_id": opened["session"],
            "stream_id": lease.stream_id,
            "session_generation": 1,
            "ordinal": 1,
            "operation": "commit-orchestration",
            "model_identity": {
                "checkpoint": adapter.model_identity,
                "tokenizer": adapter.codec.fingerprint,
            },
        },
        "requestBase64": base64.b64encode(_canonical(request) + b"\n").decode("ascii"),
        "preCheckpointDigest": adapter.checkpoint()["digest"],
    }
    committed = bridge.durable_commit(**create_args)
    child = committed["postCheckpoint"]["sessions"][opened["session"]]["orchestration"][
        "children"
    ][0]["track"]
    bootstrap_request = {
        "id": "child-bootstrap",
        "op": "bootstrap-orchestration-child",
        "session": opened["session"],
        "track": opened["track"],
        "childTrack": child,
    }
    bootstrap_args = {
        "identity": {
            "schema": "semantic-native-request-identity/1",
            "session_id": opened["session"],
            "stream_id": lease.stream_id,
            "session_generation": 1,
            "ordinal": 2,
            "operation": "bootstrap-orchestration-child",
            "model_identity": {
                "checkpoint": adapter.model_identity,
                "tokenizer": adapter.codec.fingerprint,
            },
        },
        "requestBase64": base64.b64encode(_canonical(bootstrap_request) + b"\n").decode(
            "ascii"
        ),
        "preCheckpointDigest": adapter.checkpoint()["digest"],
    }
    return adapter, opened, bridge, bootstrap_args, committed


def test_bootstrap_outcome_recovers_in_a_fresh_process(tmp_path):
    with real_runtime() as (open_adapter, _decoder):
        with execution_lease(tmp_path / "owner", session="teacher1") as lease:
            adapter, opened, bridge, args, created = bootstrap_case(
                tmp_path, lease, open_adapter
            )
            completed = dispatch(
                adapter,
                {"id": "outer-bootstrap", "op": "durable-bootstrap", **args},
                commit_bridge=bridge,
            )
            assert completed["recovered"] is False
            assert completed["postCheckpoint"]["sessions"][opened["session"]]["orchestration"][
                "operations"
            ][-1]["kind"] == "bootstrap"
            assert completed["identity"]["ordinal"] == 2
            assert completed["preCheckpointDigest"] == created["postCheckpointDigest"]
            source = tmp_path / "bootstrap-request.json"
            source.write_bytes(_canonical(args))
            code = """
import json,sys
from pathlib import Path
from megatoken.semantic_commit_recovery_v8 import DurableNativeCommitBridge, NativeExecutionLease
from megatoken.semantic_request_journal_v8 import DurableNativeRequestJournal
from tests.test_semantic_v8_orchestration import real_runtime
args=json.loads(Path(sys.argv[1]).read_text())
lease=NativeExecutionLease(int(sys.argv[4]),sys.argv[3],'teacher1','stream-a')
with real_runtime() as (open_adapter,_):
    adapter=open_adapter()
    bridge=DurableNativeCommitBridge(adapter,DurableNativeRequestJournal(sys.argv[2]),lease)
    print(json.dumps(bridge.recover_bootstrap(**args)))
lease.close()
"""
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    code,
                    str(source),
                    str(tmp_path / "journal"),
                    str(lease.path),
                    str(lease.fd),
                ],
                cwd=Path(__file__).parents[1],
                pass_fds=(lease.fd,),
                capture_output=True,
                text=True,
                timeout=120,
                check=True,
            )
            recovered = json.loads(result.stdout)
            assert recovered["schema"] == "semantic-native-durable-bootstrap/8"
            assert recovered["recovered"] is True
            assert recovered["responseBase64"] == completed["responseBase64"]
            assert recovered["postCheckpoint"] == completed["postCheckpoint"]
            assert recovered["restoreReceipt"]["hostDispatches"] == 0


def test_bootstrap_recovery_never_admits_a_missing_outcome(tmp_path):
    with real_runtime() as (open_adapter, _decoder):
        with execution_lease(tmp_path / "owner", session="teacher1") as lease:
            adapter, _opened, bridge, args, _created = bootstrap_case(
                tmp_path, lease, open_adapter
            )
            fresh = open_adapter()
            fresh_bridge = DurableNativeCommitBridge(fresh, bridge.journal, lease)
            with pytest.raises(NativeRequestConflict, match="never admitted"):
                fresh_bridge.recover_bootstrap(**args)
            assert not fresh.sessions and not fresh.attempts
            assert not fresh.runtime.memory.sessions


def test_bootstrap_unknown_outcome_is_not_retried_in_fresh_adapter(tmp_path):
    class LoseBootstrapOutcome(DurableNativeRequestJournal):
        def _after_temp_fsync(self, target_name, temp_path):
            if target_name.endswith("00000000000000000002.outcome.json"):
                raise OSError("bootstrap outcome not published")

    with real_runtime() as (open_adapter, _decoder):
        with execution_lease(tmp_path / "owner", session="teacher1") as lease:
            journal = LoseBootstrapOutcome(tmp_path / "journal")
            adapter, _opened, bridge, args, _created = bootstrap_case(
                tmp_path, lease, open_adapter, journal=journal
            )
            with pytest.raises(OSError, match="outcome not published"):
                bridge.durable_bootstrap(**args)
            assert bridge.fenced
            assert adapter.sessions[args["identity"]["session_id"]].orchestration["children"][
                0
            ]["track"]
            fresh = open_adapter()
            calls = []
            original = fresh.bootstrap_orchestration_child

            def observed_bootstrap(**request):
                calls.append(request)
                return original(**request)

            fresh.bootstrap_orchestration_child = observed_bootstrap
            fresh_bridge = DurableNativeCommitBridge(
                fresh,
                DurableNativeRequestJournal(tmp_path / "journal"),
                lease,
            )
            with pytest.raises(NativeRequestOutcomeUnknown, match="cannot retry"):
                fresh_bridge.recover_bootstrap(**args)
            assert calls == []
            assert not fresh.sessions and not fresh.attempts
            assert not fresh.runtime.memory.sessions


def test_later_native_mutation_supersedes_bootstrap_recovery(tmp_path):
    with real_runtime() as (open_adapter, _decoder):
        with execution_lease(tmp_path / "owner", session="teacher1") as lease:
            adapter, opened, bridge, args, _created = bootstrap_case(
                tmp_path, lease, open_adapter
            )
            bridge.durable_bootstrap(**args)
            dispatch(
                adapter,
                {
                    "id": "later-cancel",
                    "op": "cancel",
                    "session": opened["session"],
                    "reason": "caller cancelled after bootstrap",
                },
                commit_bridge=bridge,
            )
            fresh = open_adapter()
            fresh_bridge = DurableNativeCommitBridge(fresh, bridge.journal, lease)
            with pytest.raises(NativeRequestConflict, match="superseded"):
                fresh_bridge.recover_bootstrap(**args)
            assert not fresh.sessions and not fresh.attempts
            assert not fresh.runtime.memory.sessions

