"""Commit durability controls; supplied decoder bytes make no quality claim."""

from __future__ import annotations

import base64
import copy
import fcntl
import hashlib
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest

from megatoken.data.semantic_json import stable_algorithm_hash
from megatoken.semantic_commit_recovery_v8 import DurableNativeCommitBridge, NativeExecutionLease
from megatoken.semantic_request_journal_v8 import (
    DurableNativeRequestJournal,
    NativeRequestConflict,
    NativeRequestJournalError,
    NativeRequestOutcomeUnknown,
    _canonical,
)
from scripts.serve_semantic_v8 import dispatch
from tests.test_prediction_prefix_v8 import examples, prefix
from tests.test_semantic_v8_closed_session_restore import adapter_for_sessions, with_session
from tests.test_semantic_v8_serving_restore import commit, select, target


@contextmanager
def execution_lease(root, session="session-a", stream="stream-a"):
    root.mkdir(mode=0o700, exist_ok=True)
    directory = root / "selected-orchestration-execution-locks"
    directory.mkdir(mode=0o700, exist_ok=True)
    key = stable_algorithm_hash(
        {
            "schema": "semantic-selected-orchestration-lock-key/1",
            "sessionId": session,
            "streamId": stream,
        }
    )
    path = directory / (key + ".lock")
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    lease = NativeExecutionLease(fd, path, session, stream)
    try:
        yield lease
    finally:
        lease.close()
        os.close(fd)


def example():
    return with_session(examples("semantic-regions-v8.json.gz.b64")[0], "session-a")


def arguments(adapter, document, index, selected, ordinal=1):
    public = copy.deepcopy(document["public"])
    context = document["contexts"][index]
    public["timeline"] = public["timeline"][: context["visibleThrough"] + 1]
    request = {
        "id": f"v8-commit-1-{ordinal}",
        "op": "commit",
        "session": selected["selection"]["session"],
        "public": public,
        "contexts": document["contexts"][: index + 1],
        "invocationId": context["invocationId"],
        "record": selected["record"],
        "selection": selected["selection"],
    }
    return {
        "identity": {
            "schema": "semantic-native-request-identity/1",
            "session_id": request["session"],
            "stream_id": "stream-a",
            "session_generation": 1,
            "ordinal": ordinal,
            "operation": "commit",
            "model_identity": {
                "checkpoint": adapter.model_identity,
                "tokenizer": adapter.codec.fingerprint,
            },
        },
        "requestBase64": base64.b64encode((json.dumps(request) + "\n").encode()).decode(),
        "preCheckpointDigest": adapter.checkpoint()["digest"],
    }


def prepare(tmp_path, lease, *, journal_class=DurableNativeRequestJournal, fragments=None):
    document = example()
    adapter, runtime = adapter_for_sessions(fragments or [target(document, 0)])
    selected = select(adapter, document, 0)
    args = arguments(adapter, document, 0, selected)
    journal = journal_class(tmp_path / "journal")
    return adapter, runtime, DurableNativeCommitBridge(adapter, journal, lease), args


def test_exact_commit_bytes_recover_whole_native_adapter_without_recommitting(tmp_path):
    with execution_lease(tmp_path / "owner") as lease:
        adapter, _, bridge, args = prepare(tmp_path, lease)
        response = bridge.durable_commit(**args)
        assert response["recovered"] is False
        assert response["restoreReceipt"] is None
        assert response["postCheckpoint"] == adapter.checkpoint()
        assert json.loads(base64.b64decode(response["responseBase64"]))["result"]["committed"]
        assert bridge.durable_commit(**args) == response
        restored, _ = adapter_for_sessions([target(example(), 0), target(example(), 1)])
        new_bridge = DurableNativeCommitBridge(restored, bridge.journal, lease)
        receipt = new_bridge.recover_commit(**args)
        assert receipt["responseBase64"] == response["responseBase64"]
        assert receipt["responseSha256"] == response["responseSha256"]
        assert receipt["requestId"] == response["requestId"]
        assert receipt["recovered"] is True
        assert receipt["restoreReceipt"]["hostDispatches"] == 0
        assert receipt["restoreReceipt"]["reexecutedSelections"] == 1
        assert restored.checkpoint() == response["postCheckpoint"]
        result = select(restored, example(), 1)
        commit(restored, example(), 1, result)
        assert restored.metrics()["committed"] == 2


def test_outcome_lost_ack_fences_original_and_fresh_recovery_keeps_exact_response(tmp_path):
    class LoseAck(DurableNativeRequestJournal):
        def _after_published(self, name, path):
            if name.endswith(".outcome.json"):
                raise OSError("lost durable outcome acknowledgement")

    with execution_lease(tmp_path / "owner") as lease:
        adapter, _, bridge, args = prepare(tmp_path, lease, journal_class=LoseAck)
        with pytest.raises(OSError, match="lost durable"):
            bridge.durable_commit(**args)
        assert adapter.metrics()["committed"] == 1
        with pytest.raises(NativeRequestJournalError, match="fenced"):
            dispatch(adapter, {"id": "next", "op": "checkpoint"}, commit_bridge=bridge)
        restored, _ = adapter_for_sessions([target(example(), 0)])
        fresh = DurableNativeCommitBridge(
            restored, DurableNativeRequestJournal(tmp_path / "journal"), lease
        )
        receipt = fresh.recover_commit(**args)
        assert receipt["recovered"] and receipt["restoreReceipt"]["hostDispatches"] == 0
        assert restored.checkpoint() == adapter.checkpoint()


def test_missing_outcome_after_native_commit_remains_unknown_and_cannot_retry(tmp_path):
    class LoseOutcome(DurableNativeRequestJournal):
        def _after_temp_fsync(self, name, path):
            if name.endswith(".outcome.json"):
                raise OSError("crash before outcome publication")

    with execution_lease(tmp_path / "owner") as lease:
        adapter, _, bridge, args = prepare(tmp_path, lease, journal_class=LoseOutcome)
        with pytest.raises(OSError, match="before outcome"):
            bridge.durable_commit(**args)
        assert adapter.metrics()["committed"] == 1
        restored, runtime = adapter_for_sessions([target(example(), 0)])
        fresh = DurableNativeCommitBridge(
            restored, DurableNativeRequestJournal(tmp_path / "journal"), lease
        )
        with pytest.raises(NativeRequestOutcomeUnknown):
            fresh.recover_commit(**args)
        assert restored.metrics()["predictionAttempts"] == 0
        assert runtime.fragments == [(target(example(), 0), True)]


@pytest.mark.parametrize("mutation", ["operation", "bytes", "model", "checkpoint", "stream"])
def test_request_substitutions_never_reuse_saved_outcome(tmp_path, mutation):
    with execution_lease(tmp_path / "owner") as lease:
        adapter, _, bridge, args = prepare(tmp_path, lease)
        bridge.durable_commit(**args)
        changed = copy.deepcopy(args)
        if mutation == "operation":
            changed["identity"]["operation"] = "select"
        elif mutation == "bytes":
            raw = base64.b64decode(changed["requestBase64"])
            changed["requestBase64"] = base64.b64encode(raw + b" ").decode()
        elif mutation == "model":
            changed["identity"]["model_identity"]["checkpoint"] = "b" * 64
        elif mutation == "stream":
            changed["identity"]["stream_id"] = "another-stream"
        else:
            changed["preCheckpointDigest"] = "c" * 64
        restored, _ = adapter_for_sessions([target(example(), 0)])
        fresh = DurableNativeCommitBridge(restored, bridge.journal, lease)
        with pytest.raises(NativeRequestJournalError):
            fresh.recover_commit(**changed)
        assert not restored.sessions and not restored.attempts


def test_pending_sibling_is_refused_before_native_commit(tmp_path):
    with execution_lease(tmp_path / "owner") as lease:
        document = example()
        sibling = with_session(document, "sibling")
        adapter, _ = adapter_for_sessions([target(document, 0), target(sibling, 0)])
        selected = select(adapter, document, 0)
        select(adapter, sibling, 0)
        args = arguments(adapter, document, 0, selected)
        before = adapter.checkpoint()
        bridge = DurableNativeCommitBridge(
            adapter, DurableNativeRequestJournal(tmp_path / "journal"), lease
        )
        with pytest.raises(NativeRequestJournalError, match="pending sibling"):
            bridge.durable_commit(**args)
        assert adapter.checkpoint() == before
        assert adapter.metrics()["committed"] == 0


def test_current_adapter_with_advanced_sibling_never_receives_old_response(tmp_path):
    with execution_lease(tmp_path / "owner") as lease:
        document = example()
        adapter, _, bridge, args = prepare(
            tmp_path, lease, fragments=[target(document, 0), target(document, 0)]
        )
        bridge.durable_commit(**args)
        sibling = with_session(document, "sibling")
        commit(adapter, sibling, 0, select(adapter, sibling, 0))
        before = adapter.checkpoint()
        with pytest.raises(NativeRequestJournalError, match="whole live adapter"):
            bridge.durable_commit(**args)
        assert adapter.checkpoint() == before


@pytest.mark.parametrize("later_known", [False, True])
def test_earlier_commit_never_rolls_back_later_durable_intent(tmp_path, later_known):
    with execution_lease(tmp_path / "owner") as lease:
        document = example()
        adapter, _, bridge, args = prepare(
            tmp_path, lease, fragments=[target(document, 0), target(document, 1)]
        )
        bridge.durable_commit(**args)
        later = arguments(adapter, document, 1, select(adapter, document, 1), ordinal=2)
        if later_known:
            bridge.durable_commit(**later)
        else:
            _, _, _, frame = bridge._input(**later)
            with pytest.raises(RuntimeError):
                bridge.journal.execute(
                    later["identity"], frame, lambda _: (_ for _ in ()).throw(RuntimeError("crash"))
                )
        restored, _ = adapter_for_sessions([target(document, 0)])
        fresh = DurableNativeCommitBridge(restored, bridge.journal, lease)
        with pytest.raises(NativeRequestConflict, match="later native request"):
            fresh.recover_commit(**args)
        assert not restored.sessions and not restored.attempts


def test_rpc_requires_configured_bridge_and_blocks_unjournaled_commits(tmp_path):
    with execution_lease(tmp_path / "owner") as lease:
        adapter, _, bridge, args = prepare(tmp_path, lease)
        wrapped = {"id": "outer", "op": "durable-commit", **args}
        with pytest.raises(ValueError, match="configured execution"):
            dispatch(adapter, wrapped)
        with pytest.raises(NativeRequestJournalError, match="require durable-commit"):
            dispatch(
                adapter, json.loads(base64.b64decode(args["requestBase64"])), commit_bridge=bridge
            )
        assert (
            dispatch(adapter, wrapped, commit_bridge=bridge)["postCheckpointDigest"]
            == adapter.checkpoint()["digest"]
        )


def test_live_inherited_descriptor_excludes_second_process(tmp_path):
    with execution_lease(tmp_path / "owner") as lease:
        code = """
import fcntl,os,sys
fd=os.open(sys.argv[1],os.O_RDWR)
try:
    fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
except BlockingIOError:
    raise SystemExit(73)
raise SystemExit(0)
"""
        result = subprocess.run([sys.executable, "-c", code, str(lease.path)], timeout=10)
        assert result.returncode == 73
        lease.assert_held()


def test_outliving_native_child_retains_lock_after_every_parent_descriptor_closes(tmp_path):
    child = None
    contender = """
import fcntl,os,sys
fd=os.open(sys.argv[1],os.O_RDWR)
try:
    fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
except BlockingIOError:
    raise SystemExit(73)
os.close(fd)
"""
    holder = """
import fcntl,os,sys
fd=int(sys.argv[1])
fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
print('HELD',flush=True)
sys.stdin.readline()
os.close(fd)
"""
    try:
        with execution_lease(tmp_path / "owner") as lease:
            path = str(lease.path)
            child = subprocess.Popen(
                [sys.executable, "-c", holder, str(lease.fd)],
                pass_fds=(lease.fd,),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            assert child.stdout.readline().strip() == "HELD"
        # The context closed both the lease duplicate and its original owner fd.
        assert subprocess.run([sys.executable, "-c", contender, path], timeout=10).returncode == 73
        child.stdin.write("finish\n")
        child.stdin.flush()
        assert child.wait(timeout=10) == 0
        assert subprocess.run([sys.executable, "-c", contender, path], timeout=10).returncode == 0
    finally:
        if child is not None:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=10)
            child.stdin.close()
            child.stdout.close()
            child.stderr.close()


def test_execution_lock_refuses_replaced_inode_and_wrong_scope(tmp_path):
    with execution_lease(tmp_path / "owner") as lease:
        with pytest.raises(NativeRequestJournalError, match="exact session and stream"):
            NativeExecutionLease(lease.fd, lease.path, "different", lease.stream_id)
        moved = lease.path.with_suffix(".old")
        lease.path.rename(moved)
        lease.path.touch(mode=0o600)
        with pytest.raises(NativeRequestJournalError, match="stable inode"):
            lease.assert_held()


def test_fresh_subprocess_reconstructs_persisted_commit_under_inherited_owner(tmp_path):
    with execution_lease(tmp_path / "owner") as lease:
        adapter, _, bridge, args = prepare(tmp_path, lease)
        expected = bridge.durable_commit(**args)
        source = tmp_path / "request.json"
        source.write_bytes(_canonical(args))
        code = """
import json,sys
from pathlib import Path
from tests.test_semantic_commit_recovery_v8 import example
from tests.test_semantic_v8_closed_session_restore import adapter_for_sessions
from tests.test_semantic_v8_serving_restore import target
from megatoken.semantic_commit_recovery_v8 import DurableNativeCommitBridge, NativeExecutionLease
from megatoken.semantic_request_journal_v8 import DurableNativeRequestJournal
args=json.loads(Path(sys.argv[1]).read_text())
lease=NativeExecutionLease(int(sys.argv[4]),sys.argv[3],'session-a','stream-a')
adapter,_=adapter_for_sessions([target(example(),0)])
result=DurableNativeCommitBridge(adapter,DurableNativeRequestJournal(sys.argv[2]),lease).recover_commit(**args)
print(json.dumps(result))
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
            timeout=45,
            check=True,
        )
        restored = json.loads(result.stdout)
        assert restored["responseBase64"] == expected["responseBase64"]
        assert restored["postCheckpoint"] == expected["postCheckpoint"]
        assert restored["restoreReceipt"]["hostDispatches"] == 0


@pytest.mark.parametrize("ending", [b"", b"\n\n", b"\r\n"])
def test_original_request_must_be_one_exact_jsonl_frame(tmp_path, ending):
    with execution_lease(tmp_path / "owner") as lease:
        adapter, _, bridge, args = prepare(tmp_path, lease)
        raw = base64.b64decode(args["requestBase64"])
        args["requestBase64"] = base64.b64encode(raw[:-1] + ending).decode()
        with pytest.raises(NativeRequestJournalError, match="one exact JSONL frame"):
            bridge.durable_commit(**args)
        assert adapter.metrics()["committed"] == 0


def test_rehashed_saved_poststate_cannot_remove_prior_sibling(tmp_path):
    with execution_lease(tmp_path / "owner") as lease:
        document = example()
        sibling = with_session(document, "sibling")
        adapter, _ = adapter_for_sessions([target(sibling, 0), target(document, 0)])
        commit(adapter, sibling, 0, select(adapter, sibling, 0))
        selected = select(adapter, document, 0)
        args = arguments(adapter, document, 0, selected)
        bridge = DurableNativeCommitBridge(
            adapter, DurableNativeRequestJournal(tmp_path / "journal"), lease
        )
        bridge.durable_commit(**args)
        outcome_path = next((tmp_path / "journal").glob("*/*.outcome.json"))
        outer = json.loads(outcome_path.read_text())
        stored = json.loads(base64.b64decode(outer["result_b64"]))
        checkpoint = stored["postCheckpoint"]
        del checkpoint["sessions"]["sibling"]
        checkpoint["openings"] = [
            row for row in checkpoint["openings"] if row["session"] != "sibling"
        ]
        checkpoint["attempts"] = checkpoint["attempts"][1:]
        checkpoint["digest"] = stable_algorithm_hash(
            {key: value for key, value in checkpoint.items() if key != "digest"}
        )
        encoded = _canonical(stored)
        outer["result_b64"] = base64.b64encode(encoded).decode()
        outer["result_sha256"] = hashlib.sha256(encoded).hexdigest()
        outcome_path.write_bytes(_canonical(outer))
        restored, _ = adapter_for_sessions([target(document, 0)])
        fresh = DurableNativeCommitBridge(restored, bridge.journal, lease)
        with pytest.raises(NativeRequestJournalError, match="complete session inventory"):
            fresh.recover_commit(**args)
        assert not restored.sessions and not restored.attempts


def test_real_native_runtime_restores_committed_target_and_unchanged_sibling(tmp_path):
    from tests.test_semantic_v8_orchestration import real_runtime

    with execution_lease(tmp_path / "owner") as lease, real_runtime() as (open_adapter, decoder):
        document = example()
        sibling = with_session(document, "sibling")
        decoder["records"] = [target(sibling, 0), target(document, 0)]
        adapter = open_adapter()
        commit(adapter, sibling, 0, select(adapter, sibling, 0))
        selected = select(adapter, document, 0)
        args = arguments(adapter, document, 0, selected)
        bridge = DurableNativeCommitBridge(
            adapter, DurableNativeRequestJournal(tmp_path / "journal"), lease
        )
        committed = bridge.durable_commit(**args)
        decoder["records"] = [target(sibling, 0), target(document, 0)]
        restored = open_adapter()
        receipt = DurableNativeCommitBridge(restored, bridge.journal, lease).recover_commit(**args)
        assert receipt["postCheckpoint"] == committed["postCheckpoint"] == restored.checkpoint()
        assert receipt["restoreReceipt"]["restoredSessions"] == 2
        assert receipt["restoreReceipt"]["reexecutedSelections"] == 2
        assert receipt["restoreReceipt"]["hostDispatches"] == 0


def test_real_native_orchestration_commit_restores_reserved_child(tmp_path):
    from tests.test_semantic_v8_orchestration import (
        chosen,
        create_arguments,
        open_coordinator,
        real_runtime,
    )

    with real_runtime() as (open_adapter, _decoder):
        adapter = open_adapter()
        opened = open_coordinator(adapter)
        selected = chosen(adapter, opened, "track_create", create_arguments(opened))
        with execution_lease(tmp_path / "owner", session=opened["session"]) as lease:
            request = {
                "id": "v8-commit-1-1",
                "op": "commit-orchestration",
                "session": opened["session"],
                "track": opened["track"],
                "record": selected["record"],
                "selection": selected["selection"],
            }
            args = {
                "identity": {
                    "schema": "semantic-native-request-identity/1",
                    "session_id": opened["session"],
                    "stream_id": "stream-a",
                    "session_generation": 1,
                    "ordinal": 1,
                    "operation": "commit-orchestration",
                    "model_identity": {
                        "checkpoint": adapter.model_identity,
                        "tokenizer": adapter.codec.fingerprint,
                    },
                },
                "requestBase64": base64.b64encode(_canonical(request) + b"\n").decode(),
                "preCheckpointDigest": adapter.checkpoint()["digest"],
            }
            bridge = DurableNativeCommitBridge(
                adapter, DurableNativeRequestJournal(tmp_path / "journal"), lease
            )
            committed = bridge.durable_commit(**args)
            restored = open_adapter()
            receipt = DurableNativeCommitBridge(restored, bridge.journal, lease).recover_commit(
                **args
            )
            assert receipt["postCheckpoint"] == committed["postCheckpoint"] == restored.checkpoint()
            assert receipt["restoreReceipt"]["hostDispatches"] == 0
            outcome = json.loads(base64.b64decode(receipt["responseBase64"]))["result"]
            assert outcome["committed"] and outcome["track"]["execution"]["invocation_id"]


@pytest.mark.parametrize("operation", ["select", "cancel"])
def test_later_native_mutation_durably_supersedes_commit_before_dispatch(tmp_path, operation):
    with execution_lease(tmp_path / "owner") as lease:
        document = example()
        adapter, _, bridge, args = prepare(
            tmp_path, lease, fragments=[target(document, 0), target(document, 1)]
        )
        bridge.durable_commit(**args)
        if operation == "select":
            public, contexts, context = prefix(document, 1)
            mutation = {
                "id": "later-selection",
                "op": "select",
                "session": "session-a",
                "public": public,
                "contexts": contexts,
                "context": context,
            }
        else:
            mutation = {
                "id": "later-cancel",
                "op": "cancel",
                "session": "session-a",
                "reason": "explicit caller cancellation",
            }
        dispatch(adapter, mutation, commit_bridge=bridge)
        assert list((tmp_path / "journal").glob("*/*.superseded.json"))
        restored, _ = adapter_for_sessions([target(document, 0)])
        fresh = DurableNativeCommitBridge(restored, bridge.journal, lease)
        with pytest.raises(NativeRequestConflict, match="superseded"):
            fresh.recover_commit(**args)
        assert not restored.sessions and not restored.attempts


def test_lost_supersession_acknowledgement_prevents_later_native_mutation(tmp_path):
    class LoseMarkerAck(DurableNativeRequestJournal):
        def _after_published(self, name, path):
            if name.endswith(".superseded.json"):
                raise OSError("lost supersession acknowledgement")

    with execution_lease(tmp_path / "owner") as lease:
        adapter, _, bridge, args = prepare(tmp_path, lease, journal_class=LoseMarkerAck)
        bridge.durable_commit(**args)
        before = adapter.checkpoint()
        with pytest.raises(OSError, match="lost supersession"):
            dispatch(
                adapter,
                {"id": "cancel", "op": "cancel", "session": "session-a", "reason": "later"},
                commit_bridge=bridge,
            )
        assert adapter.checkpoint() == before
        assert bridge.fenced
        restored, _ = adapter_for_sessions([target(example(), 0)])
        fresh = DurableNativeCommitBridge(
            restored, DurableNativeRequestJournal(tmp_path / "journal"), lease
        )
        with pytest.raises(NativeRequestConflict, match="superseded"):
            fresh.recover_commit(**args)


def test_new_durable_commit_reestablishes_exact_recoverable_poststate(tmp_path):
    with execution_lease(tmp_path / "owner") as lease:
        document = example()
        adapter, _, bridge, first = prepare(
            tmp_path, lease, fragments=[target(document, 0), target(document, 1)]
        )
        bridge.durable_commit(**first)
        public, contexts, context = prefix(document, 1)
        selected = dispatch(
            adapter,
            {
                "id": "next",
                "op": "select",
                "session": "session-a",
                "public": public,
                "contexts": contexts,
                "context": context,
            },
            commit_bridge=bridge,
        )
        second = arguments(adapter, document, 1, selected, ordinal=2)
        committed = bridge.durable_commit(**second)
        restored, _ = adapter_for_sessions([target(document, 0), target(document, 1)])
        receipt = DurableNativeCommitBridge(restored, bridge.journal, lease).recover_commit(
            **second
        )
        assert receipt["postCheckpoint"] == committed["postCheckpoint"] == restored.checkpoint()
        assert receipt["restoreReceipt"]["reexecutedSelections"] == 2


def test_read_only_native_rpcs_keep_commit_recoverable(tmp_path):
    with execution_lease(tmp_path / "owner") as lease:
        adapter, _, bridge, args = prepare(tmp_path, lease)
        response = bridge.durable_commit(**args)
        for op in ("metrics", "status", "checkpoint"):
            dispatch(adapter, {"id": op, "op": op}, commit_bridge=bridge)
        assert not list((tmp_path / "journal").glob("*/*.superseded.json"))
        restored, _ = adapter_for_sessions([target(example(), 0)])
        receipt = DurableNativeCommitBridge(restored, bridge.journal, lease).recover_commit(**args)
        assert receipt["responseBase64"] == response["responseBase64"]


def test_ordinary_restore_cannot_bypass_durable_commit_reconciliation(tmp_path):
    with execution_lease(tmp_path / "owner") as lease:
        adapter, _, bridge, args = prepare(tmp_path, lease)
        result = bridge.durable_commit(**args)
        restored, _ = adapter_for_sessions([target(example(), 0)])
        fresh = DurableNativeCommitBridge(restored, bridge.journal, lease)
        with pytest.raises(NativeRequestJournalError, match="explicit recover-commit"):
            dispatch(
                restored,
                {"id": "old-cut", "op": "restore", "checkpoint": result["postCheckpoint"]},
                commit_bridge=fresh,
            )
        assert not restored.sessions and not restored.attempts
        assert fresh.recover_commit(**args)["responseBase64"] == result["responseBase64"]


@pytest.mark.parametrize("operation", ["initialize", "select", "open-orchestration"])
def test_fresh_native_generation_cannot_mutate_before_existing_commit_is_recovered(
    tmp_path, operation
):
    with execution_lease(tmp_path / "owner") as lease:
        _adapter, _, bridge, args = prepare(tmp_path, lease)
        committed = bridge.durable_commit(**args)
        restored, _ = adapter_for_sessions([target(example(), 0)])
        fresh = DurableNativeCommitBridge(restored, bridge.journal, lease)
        before = restored.checkpoint()
        with pytest.raises(NativeRequestJournalError, match="retained native generation"):
            dispatch(restored, {"id": "premature", "op": operation}, commit_bridge=fresh)
        assert restored.checkpoint() == before
        assert fresh.recover_commit(**args)["postCheckpoint"] == committed["postCheckpoint"]


def test_native_execution_generation_is_fixed_and_new_generation_is_explicit(tmp_path):
    with execution_lease(tmp_path / "owner") as lease:
        _adapter, _, bridge, args = prepare(tmp_path, lease)
        bridge.durable_commit(**args)
        second, _ = adapter_for_sessions([target(example(), 0)])
        newer = DurableNativeCommitBridge(second, bridge.journal, lease, session_generation=2)
        with pytest.raises(NativeRequestJournalError, match="generation differs"):
            newer.recover_commit(**args)
        dispatch(
            second,
            {
                "id": "new-generation",
                "op": "initialize",
                "seed": 17,
                "checkpointIdentity": second.model_identity,
                "tokenizerIdentity": second.codec.fingerprint,
            },
            commit_bridge=newer,
        )
        selected = select(second, example(), 0)
        next_args = arguments(second, example(), 0, selected)
        next_args["identity"]["session_generation"] = 2
        result = newer.durable_commit(**next_args)
        assert result["identity"]["session_generation"] == 2
        assert len(list((tmp_path / "journal").glob("*/00000000000000000001.intent.json"))) == 2
