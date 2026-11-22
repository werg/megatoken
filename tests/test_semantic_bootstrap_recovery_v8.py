"""Exact child-bootstrap durability on the real native runtime.

The coordinator and child lifecycle use the actual recurrent runtime. The
lifecycle action is explicitly controlled engineering input; it is not learned
model output or source-admission evidence.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from megatoken.semantic_commit_recovery_v8 import DurableNativeCommitBridge, _reply_bytes
from megatoken.semantic_request_journal_v8 import (
    DurableNativeRequestJournal,
    NativeRequestConflict,
    NativeRequestJournalError,
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
        "createIdentity": create_args["identity"],
        "createRequestBase64": create_args["requestBase64"],
        "createPreCheckpointDigest": create_args["preCheckpointDigest"],
    }
    return adapter, opened, bridge, bootstrap_args, committed


def tamper_bootstrap_outcome(journal, tamper):
    path = next(journal.root.glob("*/00000000000000000002.outcome.json"))
    outer = json.loads(path.read_text(encoding="utf-8"))
    stored = base64.b64decode(outer["result_b64"])
    outcome = json.loads(stored.decode("utf-8"))
    tamper(outcome)
    stored = _canonical(outcome)
    outer["result_b64"] = base64.b64encode(stored).decode("ascii")
    outer["result_sha256"] = hashlib.sha256(stored).hexdigest()
    path.write_bytes(_canonical(outer))


@pytest.mark.parametrize(
    "operation",
    [
        "bootstrap_orchestration_child",
        "bootstrap-orchestration-child",
        "bootstrap_orchestration-child",
        "bootstrap-orchestration_child",
    ],
)
def test_all_raw_bootstrap_spellings_refuse_before_mutation(tmp_path, operation):
    with real_runtime() as (open_adapter, _decoder):
        with execution_lease(tmp_path / "owner", session="teacher1") as lease:
            adapter, _opened, bridge, args, _created = bootstrap_case(
                tmp_path, lease, open_adapter
            )
            request = json.loads(base64.b64decode(args["requestBase64"]))
            before = adapter.checkpoint()
            with pytest.raises(NativeRequestJournalError, match="require durable-bootstrap-child"):
                dispatch(
                    adapter,
                    {
                        "id": request["id"],
                        "op": operation,
                        **{
                            key: value
                            for key, value in request.items()
                            if key not in {"id", "op"}
                        },
                    },
                    commit_bridge=bridge,
                )
            assert adapter.checkpoint() == before
            assert bridge.journal.inspect(
                args["createIdentity"],
                _canonical(
                    {
                        "schema": "semantic-native-durable-commit-request/8",
                        "preCheckpointDigest": args["createPreCheckpointDigest"],
                        "requestBase64": args["createRequestBase64"],
                    }
                ),
                require_latest=True,
            )["status"] == "completed"


@pytest.mark.parametrize(
    ("field", "change"),
    [
        ("session_id", lambda value: "another-session"),
        ("stream_id", lambda value: "another-stream"),
        ("session_generation", lambda value: value + 1),
        ("ordinal", lambda value: value + 1),
        ("model_identity", lambda value: {**value, "checkpoint": "f" * 64}),
    ],
)
def test_bootstrap_requires_the_same_adjacent_create_identity(tmp_path, field, change):
    with real_runtime() as (open_adapter, _decoder):
        with execution_lease(tmp_path / "owner", session="teacher1") as lease:
            adapter, _opened, bridge, args, created = bootstrap_case(
                tmp_path, lease, open_adapter
            )
            changed = copy.deepcopy(args)
            changed["createIdentity"][field] = change(changed["createIdentity"][field])
            with pytest.raises((NativeRequestConflict, NativeRequestJournalError, TypeError)):
                bridge.durable_bootstrap_child(**changed)
            assert adapter.checkpoint() == created["postCheckpoint"]


@pytest.mark.parametrize("field", ["createRequestBase64", "createPreCheckpointDigest"])
def test_bootstrap_requires_exact_original_create_bytes_and_precheckpoint(tmp_path, field):
    with real_runtime() as (open_adapter, _decoder):
        with execution_lease(tmp_path / "owner", session="teacher1") as lease:
            adapter, _opened, bridge, args, created = bootstrap_case(
                tmp_path, lease, open_adapter
            )
            changed = copy.deepcopy(args)
            if field == "createRequestBase64":
                decoded = json.loads(base64.b64decode(changed[field]).decode("utf-8"))
                decoded["id"] = "different-create-id"
                changed[field] = base64.b64encode(_canonical(decoded) + b"\n").decode("ascii")
            else:
                changed[field] = "f" * 64
            with pytest.raises((NativeRequestConflict, NativeRequestJournalError, TypeError)):
                bridge.durable_bootstrap_child(**changed)
            assert adapter.checkpoint() == created["postCheckpoint"]


def test_bootstrap_requires_create_to_be_the_latest_completed_native_request(tmp_path):
    with real_runtime() as (open_adapter, _decoder):
        with execution_lease(tmp_path / "owner", session="teacher1") as lease:
            adapter, _opened, bridge, args, created = bootstrap_case(
                tmp_path, lease, open_adapter
            )
            extra_identity = {
                **args["createIdentity"],
                "ordinal": 2,
                "operation": "status-orchestration",
            }
            extra_bytes = b'{"id":"intervening","op":"status-orchestration"}\n'
            extra_frame = _canonical(
                {
                    "schema": "intervening-native-request/1",
                    "requestBase64": base64.b64encode(extra_bytes).decode("ascii"),
                }
            )
            bridge.journal.execute(extra_identity, extra_frame, lambda _frame: b"{}")
            changed = copy.deepcopy(args)
            changed["identity"]["ordinal"] = 3
            with pytest.raises(NativeRequestJournalError, match="immediately follow"):
                bridge.durable_bootstrap_child(**changed)
            assert adapter.checkpoint() == created["postCheckpoint"]


def test_bootstrap_requires_a_completed_create_outcome(tmp_path):
    with real_runtime() as (open_adapter, _decoder):
        with execution_lease(tmp_path / "owner", session="teacher1") as lease:
            adapter, _opened, bridge, args, created = bootstrap_case(
                tmp_path, lease, open_adapter
            )
            outcome = next(bridge.journal.root.glob("*/00000000000000000001.outcome.json"))
            outcome.unlink()
            with pytest.raises(NativeRequestOutcomeUnknown):
                bridge.durable_bootstrap_child(**args)
            assert adapter.checkpoint() == created["postCheckpoint"]


def test_bootstrap_precheckpoint_must_equal_the_committed_create_poststate(tmp_path):
    with real_runtime() as (open_adapter, _decoder):
        with execution_lease(tmp_path / "owner", session="teacher1") as lease:
            adapter, opened, bridge, args, created = bootstrap_case(
                tmp_path, lease, open_adapter
            )
            adapter.cancel_orchestration(
                session=opened["session"], track=opened["track"], reason="changed after create"
            )
            changed = copy.deepcopy(args)
            changed["preCheckpointDigest"] = adapter.checkpoint()["digest"]
            with pytest.raises(NativeRequestConflict, match="precheckpoint differs"):
                bridge.durable_bootstrap_child(**changed)
            assert adapter.checkpoint() != created["postCheckpoint"]
            assert not list(bridge.journal.root.glob("*/00000000000000000002.intent.json"))


def rewrite_response(outcome, mutate):
    response = json.loads(base64.b64decode(outcome["responseBase64"]).decode("utf-8"))
    mutate(response["result"]["track"])
    outcome["responseBase64"] = base64.b64encode(_reply_bytes(response)).decode("ascii")


def test_bootstrap_outcome_recovers_in_a_fresh_process(tmp_path):
    with real_runtime() as (open_adapter, _decoder):
        with execution_lease(tmp_path / "owner", session="teacher1") as lease:
            adapter, opened, bridge, args, created = bootstrap_case(
                tmp_path, lease, open_adapter
            )
            completed = dispatch(
                adapter,
                {"id": "outer-bootstrap", "op": "durable-bootstrap-child", **args},
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
    print(json.dumps(bridge.recover_bootstrap_child(**args)))
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
            assert recovered["schema"] == "semantic-native-durable-bootstrap-child/8"
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
                fresh_bridge.recover_bootstrap_child(**args)
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
                bridge.durable_bootstrap_child(**args)
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
                fresh_bridge.recover_bootstrap_child(**args)
            assert calls == []
            assert not fresh.sessions and not fresh.attempts
            assert not fresh.runtime.memory.sessions


def test_later_native_mutation_supersedes_bootstrap_recovery(tmp_path):
    with real_runtime() as (open_adapter, _decoder):
        with execution_lease(tmp_path / "owner", session="teacher1") as lease:
            adapter, opened, bridge, args, _created = bootstrap_case(
                tmp_path, lease, open_adapter
            )
            bridge.durable_bootstrap_child(**args)
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
                fresh_bridge.recover_bootstrap_child(**args)
            assert not fresh.sessions and not fresh.attempts
            assert not fresh.runtime.memory.sessions


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("generation", "generation differs"),
        ("model", "model, tokenizer or training identity differs"),
        ("operation", "only child bootstrap operations use bootstrap recovery"),
        ("child", "exact bytes differ"),
    ],
)
def test_bootstrap_recovery_rejects_identity_and_child_substitutions(
    tmp_path, mutation, message
):
    with real_runtime() as (open_adapter, _decoder):
        with execution_lease(tmp_path / "owner", session="teacher1") as lease:
            adapter, _opened, bridge, args, _created = bootstrap_case(
                tmp_path, lease, open_adapter
            )
            bridge.durable_bootstrap_child(**args)
            changed = copy.deepcopy(args)
            if mutation == "generation":
                changed["identity"]["session_generation"] = 2
            elif mutation == "model":
                changed["identity"]["model_identity"]["checkpoint"] = "b" * 64
            elif mutation == "operation":
                changed["identity"]["operation"] = "commit"
            else:
                request = json.loads(base64.b64decode(changed["requestBase64"]))
                request["childTrack"] += ":foreign"
                changed["requestBase64"] = base64.b64encode(
                    _canonical(request) + b"\n"
                ).decode("ascii")
            fresh = open_adapter()
            fresh_bridge = DurableNativeCommitBridge(fresh, bridge.journal, lease)
            with pytest.raises(NativeRequestJournalError, match=message):
                fresh_bridge.recover_bootstrap_child(**changed)
            assert not fresh.sessions and not fresh.attempts
            assert not fresh.runtime.memory.sessions


@pytest.mark.parametrize(
    "mutate",
    [
        lambda track: track.update(parent="foreign-parent"),
        lambda track: track.update(budget=track["budget"] + 1),
    ],
    ids=["wrong-parent", "wrong-budget"],
)
def test_rehashed_bootstrap_response_must_match_restored_track(tmp_path, mutate):
    with real_runtime() as (open_adapter, _decoder):
        with execution_lease(tmp_path / "owner", session="teacher1") as lease:
            adapter, _opened, bridge, args, _created = bootstrap_case(
                tmp_path, lease, open_adapter
            )
            completed = bridge.durable_bootstrap_child(**args)
            tamper_bootstrap_outcome(
                bridge.journal,
                lambda outcome: rewrite_response(outcome, mutate),
            )
            fresh = open_adapter()
            fresh_bridge = DurableNativeCommitBridge(fresh, bridge.journal, lease)
            with pytest.raises(NativeRequestJournalError, match="bootstrap response"):
                fresh_bridge.recover_bootstrap_child(**args)
            assert fresh_bridge.fenced
            # Exact create/bootstrap evidence is checked before recovery
            # opens model state, so a malformed response cannot restore an
            # otherwise valid postcheckpoint into this adapter.
            assert not fresh.sessions and not fresh.attempts


@pytest.mark.parametrize("field", ["operations", "attempts"])
def test_rehashed_bootstrap_poststate_rejects_mutated_history(tmp_path, field):
    with real_runtime() as (open_adapter, _decoder):
        with execution_lease(tmp_path / "owner", session="teacher1") as lease:
            adapter, _opened, bridge, args, _created = bootstrap_case(
                tmp_path, lease, open_adapter
            )
            bridge.durable_bootstrap_child(**args)

            def change(outcome):
                post = outcome["postCheckpoint"]
                if field == "operations":
                    operations = post["sessions"]["teacher1"]["orchestration"]["operations"]
                    operations.pop()
                else:
                    post["attempts"].append({"id": "forged-attempt"})
                post["digest"] = _checkpoint_digest(post)

            tamper_bootstrap_outcome(bridge.journal, change)
            fresh = open_adapter()
            fresh_bridge = DurableNativeCommitBridge(fresh, bridge.journal, lease)
            with pytest.raises(NativeRequestJournalError):
                fresh_bridge.recover_bootstrap_child(**args)
            assert not fresh.sessions and not fresh.attempts
            assert not fresh.runtime.memory.sessions


def _checkpoint_digest(checkpoint):
    from megatoken.data.semantic_json import stable_algorithm_hash

    return stable_algorithm_hash(
        {key: value for key, value in checkpoint.items() if key != "digest"}
    )


def test_duplicate_durable_bootstrap_returns_the_retained_result_without_dispatch(tmp_path):
    with real_runtime() as (open_adapter, _decoder):
        with execution_lease(tmp_path / "owner", session="teacher1") as lease:
            adapter, _opened, bridge, args, _created = bootstrap_case(
                tmp_path, lease, open_adapter
            )
            first = bridge.durable_bootstrap_child(**args)
            calls = []
            original = adapter.bootstrap_orchestration_child

            def observed_bootstrap(**request):
                calls.append(request)
                return original(**request)

            adapter.bootstrap_orchestration_child = observed_bootstrap
            duplicate = dispatch(
                adapter,
                {"id": "duplicate-bootstrap", "op": "durable-bootstrap-child", **args},
                commit_bridge=bridge,
            )
            assert duplicate == first
            assert calls == []

