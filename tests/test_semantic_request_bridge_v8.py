from __future__ import annotations

import copy
import json

import pytest

from megatoken.data.semantic_json import stable_algorithm_hash
from megatoken.semantic_request_bridge_v8 import (
    CheckpointBoundNativeRequestBridge,
    NativePoststateMismatch,
)
from megatoken.semantic_request_journal_v8 import (
    DurableNativeRequestJournal,
    NativeRequestJournalError,
    NativeRequestOutcomeUnknown,
)


def request_identity():
    return {
        "schema": "semantic-native-request-identity/1",
        "session_id": "session-1",
        "stream_id": "stream-1",
        "session_generation": 1,
        "ordinal": 1,
        "operation": "commit",
        "model_identity": {"checkpoint": "a" * 64, "tokenizer": "b" * 64},
    }


def checkpoint(*, pending=False):
    value = {
        "schema": "semantic-model-checkpoint/8",
        "initialization": {
            "checkpointIdentity": "a" * 64,
            "tokenizerIdentity": "b" * 64,
            "trainingIdentity": None,
        },
        "attempts": [
            {"selectionId": "v8:0", "status": "committed", "session": "session-1"}
        ],
        "sessions": {
            "session-1": {
                "public": {"timeline": [{"mode": "generated"}]},
                "unacknowledgedSelection": None,
            }
        },
    }
    if pending:
        value["attempts"].append(
            {"selectionId": "v8:other", "status": "awaiting-commit", "session": "other-session"}
        )
        value["sessions"]["other-session"] = {
            "public": {"timeline": []},
            "unacknowledgedSelection": "v8:other",
        }
    value["digest"] = stable_algorithm_hash(value)
    return value


class Adapter:
    def __init__(self, state=None):
        self.model_identity = "a" * 64
        self.codec = type("Codec", (), {"fingerprint": "b" * 64})()
        self.training_identity = None
        self.attempts = []
        self.sessions = {}
        self.runtime = type("Runtime", (), {"memory": type("Memory", (), {"sessions": {}})()})()
        self.state = copy.deepcopy(state)

    def checkpoint(self):
        if self.state is None:
            raise ValueError("no adapter checkpoint")
        return copy.deepcopy(self.state)

    def restore(self, *, checkpoint):
        if self.state is not None:
            raise ValueError("restore requires empty adapter")
        self.state = copy.deepcopy(checkpoint)


def request_bytes():
    return json.dumps(
        {
            "id": "rpc-2",
            "op": "commit",
            "nativeRequest": request_identity(),
            "session": "session-1",
            "selection": {"selectionId": "v8:0"},
        },
        separators=(",", ":"),
    ).encode()


def reply_bytes(cp=None):
    cp = cp or checkpoint()
    return json.dumps(
        {
            "id": "rpc-2",
            "ok": True,
            "result": {
                "schema": "semantic-native-commit/8",
                "selectionId": "v8:0",
                "publicDigest": stable_algorithm_hash(cp["sessions"]["session-1"]["public"]),
                "committed": True,
            },
        },
        separators=(",", ":"),
    ).encode()


def test_completed_commit_readback_restores_and_rechecks_poststate(tmp_path):
    journal = DurableNativeRequestJournal(tmp_path / "journal")
    bridge = CheckpointBoundNativeRequestBridge(journal)
    identity = request_identity()
    raw = request_bytes()
    first = Adapter()
    cp = checkpoint()
    output = bridge.execute_commit(
        first,
        identity,
        raw,
        lambda _: (setattr(first, "state", cp) or reply_bytes(cp)),
    )
    assert output == reply_bytes(cp)

    fresh = Adapter()
    recovered = bridge.execute_commit(
        fresh,
        identity,
        raw,
        lambda _: pytest.fail("completed commit was dispatched again"),
    )
    assert recovered == output
    assert fresh.checkpoint() == cp


def test_unknown_commit_outcome_is_not_reexecuted(tmp_path):
    journal = DurableNativeRequestJournal(tmp_path / "journal")
    bridge = CheckpointBoundNativeRequestBridge(journal)
    identity = request_identity()
    raw = request_bytes()
    with pytest.raises(NativeRequestOutcomeUnknown) as refused:
        bridge.execute_commit(
            Adapter(), identity, raw, lambda _: (_ for _ in ()).throw(RuntimeError("post-dispatch crash"))
        )
    assert isinstance(refused.value.__cause__, RuntimeError)
    assert str(refused.value.__cause__) == "post-dispatch crash"
    with pytest.raises(NativeRequestOutcomeUnknown):
        bridge.execute_commit(
            Adapter(), identity, raw, lambda _: pytest.fail("unknown commit redispatched")
        )


def test_fresh_restore_refuses_checkpoint_with_unacknowledged_selection(tmp_path):
    journal = DurableNativeRequestJournal(tmp_path / "journal")
    bridge = CheckpointBoundNativeRequestBridge(journal)
    identity = request_identity()
    raw = request_bytes()
    pending = checkpoint(pending=True)
    adapter = Adapter()
    with pytest.raises(NativeRequestOutcomeUnknown) as refused:
        bridge.execute_commit(
            adapter,
            identity,
            raw,
            lambda _: (setattr(adapter, "state", pending) or reply_bytes(pending)),
        )
    assert isinstance(refused.value.__cause__, NativePoststateMismatch)
    assert "unacknowledged selection" in str(refused.value.__cause__)
    with pytest.raises(NativeRequestOutcomeUnknown):
        bridge.execute_commit(
            Adapter(), identity, raw, lambda _: pytest.fail("unsafe checkpoint retried")
        )


def test_committed_reply_must_match_checkpoint_public_digest(tmp_path):
    journal = DurableNativeRequestJournal(tmp_path / "journal")
    bridge = CheckpointBoundNativeRequestBridge(journal)
    identity = request_identity()
    cp = checkpoint()
    wrong = json.dumps(
        {
            "id": "rpc-2",
            "ok": True,
            "result": {
                "schema": "semantic-native-commit/8",
                "selectionId": "v8:0",
                "publicDigest": "0" * 64,
                "committed": True,
            },
        },
        separators=(",", ":"),
    ).encode()
    first = Adapter()
    with pytest.raises(NativeRequestOutcomeUnknown) as refused:
        bridge.execute_commit(
            first,
            identity,
            request_bytes(),
            lambda _: (setattr(first, "state", cp) or wrong),
        )
    assert isinstance(refused.value.__cause__, NativePoststateMismatch)
    assert "public digest" in str(refused.value.__cause__)
    with pytest.raises(NativeRequestOutcomeUnknown):
        bridge.execute_commit(
            Adapter(), identity, request_bytes(), lambda _: pytest.fail("mismatched commit retried")
        )


def test_execution_lock_proves_existing_lock_on_exact_inherited_open_description(tmp_path):
    import fcntl
    import os

    from megatoken.semantic_request_bridge_v8 import validate_execution_lock_fd

    lock_path = tmp_path / "execution.lock"
    owner = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    contender = None
    try:
        with pytest.raises(NativeRequestJournalError, match="not already locked"):
            validate_execution_lock_fd(owner, str(lock_path))

        fcntl.flock(owner, fcntl.LOCK_EX)
        contender = os.open(lock_path, os.O_RDWR)
        with pytest.raises(NativeRequestJournalError, match="open-file-description lock proof"):
            validate_execution_lock_fd(contender, str(lock_path))
        validate_execution_lock_fd(owner, str(lock_path))
    finally:
        if contender is not None:
            os.close(contender)
        fcntl.flock(owner, fcntl.LOCK_UN)
        os.close(owner)


@pytest.mark.parametrize("reply", [
    lambda base: {**base, "id": "wrong-rpc"},
    lambda base: {**base, "ok": False, "error": {"category": "refused"}},
])
def test_changed_or_rejected_reply_becomes_unknown_without_retry(tmp_path, reply):
    journal = DurableNativeRequestJournal(tmp_path / "journal")
    bridge = CheckpointBoundNativeRequestBridge(journal)
    identity = request_identity()
    raw = request_bytes()
    cp = checkpoint()
    adapter = Adapter()
    base = json.loads(reply_bytes(cp))
    changed = json.dumps(reply(base), separators=(",", ":")).encode()
    with pytest.raises(NativeRequestOutcomeUnknown) as refused:
        bridge.execute_commit(
            adapter,
            identity,
            raw,
            lambda _: (setattr(adapter, "state", cp) or changed),
        )
    assert isinstance(refused.value.__cause__, NativePoststateMismatch)
    with pytest.raises(NativeRequestOutcomeUnknown):
        bridge.execute_commit(
            Adapter(), identity, raw, lambda _: pytest.fail("rejected commit was retried")
        )


def test_completed_readback_refuses_to_replace_existing_sibling_native_state(tmp_path):
    journal = DurableNativeRequestJournal(tmp_path / "journal")
    bridge = CheckpointBoundNativeRequestBridge(journal)
    identity = request_identity()
    raw = request_bytes()
    target = checkpoint()
    original = Adapter()
    bridge.execute_commit(
        original,
        identity,
        raw,
        lambda _: (setattr(original, "state", target) or reply_bytes(target)),
    )
    sibling = checkpoint()
    sibling["sessions"]["sibling-session"] = {
        "public": {"timeline": [{"mode": "observed"}]},
        "unacknowledgedSelection": None,
    }
    sibling["digest"] = stable_algorithm_hash({k: v for k, v in sibling.items() if k != "digest"})
    current = Adapter(sibling)
    current.attempts = [{"selectionId": "v8:other"}]
    with pytest.raises(NativePoststateMismatch, match="active native state differs"):
        bridge.execute_commit(current, identity, raw, lambda _: pytest.fail("callback repeated"))
