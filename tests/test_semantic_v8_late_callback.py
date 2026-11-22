"""Original listener input and actual native continuation, with supplied decoding.

The paired producer fixture executes the original JavaScript independently,
registers once through the production harness, checkpoints, journals a later
arrival, and restores that arrival before delivering the callback. These are
engineering controls, not learned execution or source-admission qualification.
"""

from __future__ import annotations

import base64
import copy
import gzip
import json
from pathlib import Path

import pytest

from megatoken.data.semantic_json import stable_algorithm_hash
from tests.test_semantic_v8_orchestration import (
    acknowledge,
    bootstrap,
    chosen,
    commit_child,
    real_runtime,
    recover,
    request,
    select_child,
)


def original_listener():
    path = Path(__file__).parent / "fixtures/late-callback-original-input-v8.json.gz.b64"
    return json.loads(gzip.decompress(base64.b64decode(path.read_bytes())))


def completed_root(adapter, decoder, document):
    public = document["initialPublic"]
    session = public["timeline"][0]["fields"]["session"]
    opened = request(
        adapter,
        "open-orchestration",
        session=session,
        public=public,
        origin="controlled",
        childBudget=64,
    )
    selected = chosen(
        adapter,
        opened,
        "track_create",
        {
            "purpose": "Run the original listener once",
            "completion": "Deliver its exact return",
            "budget": 64,
            "capabilities": ["text", "demo.listen"],
            "execution": copy.deepcopy(opened["children"][0]["execution"]),
        },
    )
    created = acknowledge(adapter, opened, selected)
    bootstrap(adapter, opened, created["track"]["address"])
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
    decoder["records"] = [record.encode() for record in document["records"][:4]]
    for index in range(4):
        selected = select_child(adapter, document, index)
        commit_child(adapter, document, index, selected)
    result = adapter.complete(
        session=session,
        public=document["rootPublic"],
        contexts=document["contexts"][:4],
        invocationId=document["rootInvocation"],
    )
    assert result["childCompletion"]["completion"] == {"kind": "return", "value": ["num", 1]}
    acknowledge(
        adapter,
        opened,
        chosen(
            adapter,
            opened,
            "track_progress",
            {
                "address": opened["track"],
                "progress": json.dumps(result["childCompletion"]["completion"]),
            },
        ),
    )
    packet = adapter.checkpoint()
    assert (
        adapter.status_orchestration(session=session, track=opened["track"])["status"] == "active"
    )
    return opened, packet


def test_actual_journalled_callback_can_continue_after_root_checkpoint():
    document = original_listener()
    assert document["reference"] == {"registrations": 1, "root": 1, "callback": 5}
    assert document["registrations"] == 1
    assert document["rootPublic"]["timeline"] == document["admissionPublic"]["timeline"][:-2]
    assert document["ingressRecords"][-1]["arrival"]["event"]["eventId"] == "after-checkpoint"
    with real_runtime() as (open_adapter, decoder):
        adapter = open_adapter()
        opened, packet = completed_root(adapter, decoder, document)
        decoder["records"] = [record.encode() for record in document["records"][:4]]
        adapter, receipt = recover(open_adapter, adapter, opened["session"], packet)
        assert receipt["hostDispatches"] == 0
        decoder["records"] = [record.encode() for record in document["records"][4:]]
        # The generic refusal remains. The callback cannot bypass admission.
        with pytest.raises(ValueError, match="Dynamic invocation admission"):
            select_child(adapter, document, 4)
        before_attempts = copy.deepcopy(adapter.attempts)
        admitted = adapter.admit_orchestration_callback(
            session=opened["session"],
            track=opened["track"],
            public=document["admissionPublic"],
            contexts=document["contexts"][:4],
        )
        assert admitted["execution"]["input_ref"] == stable_algorithm_hash(
            document["admissionPublic"]["timeline"][-1]["fields"]["inputGraph"]
        )
        assert admitted["parentInvocationId"] == "root"
        assert adapter.attempts == before_attempts
        state = adapter.status_orchestration(session=opened["session"], track=opened["track"])
        assert len(state["children"]) == 2 and len(state["tracks"]) == 2
        assert state["children"][-1]["track"] is None
        admitted_packet = adapter.checkpoint()
        assert (
            adapter.admit_orchestration_callback(
                session=opened["session"],
                track=opened["track"],
                public=document["admissionPublic"],
                contexts=document["contexts"][:4],
            )
            == admitted
        )
        assert adapter.checkpoint() == admitted_packet
        changed = copy.deepcopy(document["admissionPublic"])
        changed["timeline"][-1]["fields"]["inputGraph"]["roots"] = [["num", 6]]
        with pytest.raises(ValueError, match="changed history"):
            adapter.admit_orchestration_callback(
                session=opened["session"],
                track=opened["track"],
                public=changed,
                contexts=document["contexts"][:4],
            )
        assert adapter.checkpoint() == admitted_packet
        decoder["records"] = [record.encode() for record in document["records"][:4]]
        adapter, restored = recover(open_adapter, adapter, opened["session"], admitted_packet)
        assert restored["hostDispatches"] == 0
        decoder["records"] = [record.encode() for record in document["records"][4:]]
        with pytest.raises(ValueError, match="reservation"):
            select_child(adapter, document, 4)
        created = acknowledge(
            adapter,
            opened,
            chosen(
                adapter,
                opened,
                "track_create",
                {
                    "purpose": "Run the one observed retained callback",
                    "completion": "Deliver its exact return",
                    "budget": 64,
                    "capabilities": ["text", "demo.listen"],
                    "execution": admitted["execution"],
                },
            ),
        )
        bootstrap(adapter, opened, created["track"]["address"])
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
        for index in (4, 5):
            selected = select_child(adapter, document, index)
            commit_child(adapter, document, index, selected)
        complete = adapter.complete(
            session=opened["session"],
            public=document["public"],
            contexts=document["contexts"],
            invocationId=document["callbackInvocation"],
        )
        assert complete["childCompletion"]["completion"] == {"kind": "return", "value": ["num", 5]}
        acknowledge(
            adapter,
            opened,
            chosen(
                adapter,
                opened,
                "track_progress",
                {
                    "address": opened["track"],
                    "progress": json.dumps(complete["childCompletion"]["completion"]),
                },
            ),
        )
        final = adapter.checkpoint()
        decoder["records"] = [record.encode() for record in document["records"]]
        adapter, restored = recover(open_adapter, adapter, opened["session"], final)
        assert restored["hostDispatches"] == 0
        assert adapter.status_orchestration(session=opened["session"], track=opened["track"])[
            "published"
        ] == ["root", "event1"]


def test_original_callback_mutations_refuse_before_native_state_changes(subtests):
    document = original_listener()
    with real_runtime() as (open_adapter, decoder):
        adapter = open_adapter()
        opened, before = completed_root(adapter, decoder, document)
        changes = {
            "source": lambda rows: rows[-1]["fields"].update(sourceDigest="f" * 64),
            "parent": lambda rows: rows[-1]["fields"].update(parentInvocationId="another"),
            "realm": lambda rows: rows[-1]["fields"].update(realmId="another"),
            "closure": lambda rows: rows[-1]["fields"].update(closureId="another"),
            "entry": lambda rows: rows[-1]["fields"].update(entry="run"),
            "trigger": lambda rows: rows[-1]["fields"].update(triggerId="another"),
            "owner-track": lambda rows: rows[-1]["fields"].update(track="another"),
            "event": lambda rows: rows[-2]["fields"].update(subscriptionId="another"),
            "missing-event": lambda rows: rows.pop(-2),
            "reordered-event": lambda rows: rows.reverse(),
            "changed-root": lambda rows: rows[1]["fields"]["inputGraph"].update(roots=[["num", 9]]),
        }
        for name, mutate in changes.items():
            with subtests.test(name=name):
                public = copy.deepcopy(document["admissionPublic"])
                mutate(public["timeline"])
                with pytest.raises(ValueError):
                    adapter.admit_orchestration_callback(
                        session=opened["session"],
                        track=opened["track"],
                        public=public,
                        contexts=document["contexts"][:4],
                    )
                assert adapter.checkpoint() == before
        with subtests.test(name="caller-authored-binding"):
            with pytest.raises(TypeError):
                adapter.admit_orchestration_callback(
                    session=opened["session"],
                    track=opened["track"],
                    public=document["admissionPublic"],
                    contexts=document["contexts"][:4],
                    execution={"input_ref": "f" * 64},
                )
            assert adapter.checkpoint() == before


@pytest.mark.parametrize(
    "operation",
    [
        "admit-orchestration-callback",
        "admit_orchestration_callback",
        "admit-orchestration_callback",
        "admit_orchestration-callback",
    ],
)
def test_durable_sessions_refuse_direct_admission_spellings(tmp_path, operation):
    from megatoken.semantic_commit_recovery_v8 import DurableNativeCommitBridge
    from megatoken.semantic_request_journal_v8 import (
        DurableNativeRequestJournal,
        NativeRequestJournalError,
    )
    from scripts.serve_semantic_v8 import dispatch
    from tests.test_semantic_commit_recovery_v8 import execution_lease

    with execution_lease(tmp_path) as lease, real_runtime() as (open_adapter, _):
        adapter = open_adapter()
        journal = DurableNativeRequestJournal(tmp_path / "journal")
        bridge = DurableNativeCommitBridge(adapter, journal, lease)
        before = adapter.checkpoint()
        with pytest.raises(NativeRequestJournalError, match="durable-commit"):
            dispatch(adapter, {"id": "unguarded-admission", "op": operation}, commit_bridge=bridge)
        assert adapter.checkpoint() == before
        assert not journal.has_history(bridge._scope_identity())


def test_actual_admission_known_outcome_restores_and_unknown_outcome_stays_fenced(
    tmp_path, subtests
):
    from megatoken.semantic_commit_recovery_v8 import DurableNativeCommitBridge
    from megatoken.semantic_request_journal_v8 import (
        DurableNativeRequestJournal,
        NativeRequestOutcomeUnknown,
        _canonical,
    )
    from tests.test_semantic_commit_recovery_v8 import execution_lease

    document = original_listener()
    with (
        execution_lease(tmp_path, session="late-callback") as lease,
        real_runtime() as (open_adapter, decoder),
    ):
        adapter = open_adapter()
        opened, before = completed_root(adapter, decoder, document)
        journal = DurableNativeRequestJournal(tmp_path / "journal")
        bridge = DurableNativeCommitBridge(adapter, journal, lease)
        original = {
            "id": "one-observed-callback",
            "op": "admit-orchestration-callback",
            "session": opened["session"],
            "track": opened["track"],
            "public": document["admissionPublic"],
            "contexts": document["contexts"][:4],
        }
        arguments = {
            "identity": {
                **bridge._scope_identity(),
                "operation": original["op"],
                "ordinal": 1,
            },
            "requestBase64": base64.b64encode(_canonical(original) + b"\n").decode(),
            "preCheckpointDigest": before["digest"],
        }
        committed = bridge.durable_commit(**arguments)
        post = adapter.checkpoint()
        assert committed["postCheckpoint"] == post
        assert post["attempts"] == before["attempts"]
        assert len(post["sessions"][opened["session"]]["orchestration"]["children"]) == 2
        response = json.loads(base64.b64decode(committed["responseBase64"]))
        # The original reply can be lost. Recovery reads the retained outcome,
        # replays the whole adapter and never repeats host registration/intake.
        adapter.cancel(
            session=opened["session"], reason="Original process ended after the durable outcome"
        )
        fresh = open_adapter()
        restored_bridge = DurableNativeCommitBridge(fresh, journal, lease)
        decoder["records"] = [record.encode() for record in document["records"][:4]]
        recovered = restored_bridge.recover_commit(**arguments)
        assert recovered["restoreReceipt"]["hostDispatches"] == 0
        assert recovered["responseBase64"] == committed["responseBase64"]
        assert fresh.checkpoint() == post
        assert restored_bridge.durable_commit(**arguments)["postCheckpoint"] == post
        for field in ("programId", "parentInvocationId", "input_ref"):
            with subtests.test(changed_post_binding=field):
                changed = copy.deepcopy(post)
                child = changed["sessions"][opened["session"]]["orchestration"]["children"][-1]
                if field == "input_ref":
                    child["execution"][field] = "f" * 64
                else:
                    child["invocation"]["fields"][field] = "different"
                changed["digest"] = stable_algorithm_hash(
                    {key: value for key, value in changed.items() if key != "digest"}
                )
                with pytest.raises(Exception, match="exact observed child"):
                    bridge._check_post(original, response, changed, before, before["digest"])
        unknown = DurableNativeRequestJournal(tmp_path / "unknown-journal")
        empty = open_adapter()
        unknown_bridge = DurableNativeCommitBridge(empty, unknown, lease)
        identity, _, _, frame = unknown_bridge._input(**arguments)

        def lose_outcome(_):
            raise RuntimeError("Original operation outcome was not durably recorded")

        with pytest.raises(RuntimeError, match="not durably recorded"):
            unknown.execute(identity, frame, lose_outcome)
        empty_before = empty.checkpoint()
        with pytest.raises(NativeRequestOutcomeUnknown, match="unknown"):
            unknown_bridge.recover_commit(**arguments)
        assert empty.checkpoint() == empty_before
        assert not empty.sessions and not empty.attempts
