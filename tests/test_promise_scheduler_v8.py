import copy
import unittest

from megatoken.data.native_v8 import NativeV8Error, _apply_detail, _initial_state
from megatoken.data.protocol_v8 import (
    ProtocolV8Error,
    _external_machine,
    _internal_machine,
    apply_protocol_record,
    assert_detailed_record_compatible,
    create_protocol_state,
)
from tests.test_number_v8 import allocate, machine


def shared_state():
    return _initial_state(
        {},
        {"inputGraph": {"roots": [], "objects": []}, "invocationId": "root", "entry": "run"},
    )


def detail(state, record_kind, **fields):
    _apply_detail(
        state,
        record_kind,
        fields,
        {},
        root="root",
        functions=[{"name": "callback", "params": ["value"]}],
    )


def create(state, identity):
    detail(state, "promise", action="create", promise=identity)


def callback(state):
    detail(
        state,
        "allocate",
        name="callback",
        kind="closure",
        properties=[],
        captures=[],
        code="callback",
    )


class PromiseSchedulerTests(unittest.TestCase):
    def test_protocol_reaction_receiver_survives_await_and_restores_owner(self):
        state = shared_state()
        for identity in ("owner", "lexical"):
            detail(state, "allocate", name=identity, kind="object", properties=[])
        callback(state)
        node = dict(state["heap"])["callback"]
        node.update(functionKind="arrow", lexicalReceiver=["ref", "lexical"])
        state["receiver"] = ["ref", "owner"]
        state["protocol"] = create_protocol_state()
        create(state, "source")
        create(state, "pause")
        for identity in ("source", "pause"):
            detail(state, "promise", action="resolve", promise=identity, value=["num", 1])
        detail(
            state,
            "promise",
            action="react",
            promise="source",
            job="reaction",
            reaction="then",
            closure="callback",
            target="result",
        )
        detail(state, "await", promise="result", continuation="outer", invocation="root")
        detail(state, "job", action="dispatch", job="reaction", owner="root")
        self.assertEqual(state["receiver"], ["ref", "lexical"])
        detail(state, "await", promise="pause", continuation="inner", invocation="job:reaction")
        self.assertEqual(state["receiver"], ["ref", "owner"])
        self.assertEqual(dict(state["continuations"])["inner"]["receiver"], ["ref", "lexical"])
        before = copy.deepcopy(state)
        with self.assertRaisesRegex(NativeV8Error, "unresumed"):
            detail(state, "job", action="complete", job="reaction", owner="root")
        self.assertEqual(state, before)
        detail(
            state,
            "resume",
            continuation="inner",
            invocation="job:reaction",
            status="fulfilled",
            result=["num", 1],
        )
        self.assertEqual(state["receiver"], ["ref", "lexical"])
        detail(state, "return", invocation="job:reaction", value=["num", 7])
        detail(state, "job", action="complete", job="reaction", owner="root")
        self.assertEqual(state["receiver"], ["ref", "owner"])
        detail(
            state,
            "resume",
            continuation="outer",
            invocation="root",
            status="fulfilled",
            result=["num", 7],
        )
        self.assertEqual(state["receiver"], ["ref", "owner"])

    def test_protocol_job_return_cannot_invent_promise_assimilation(self):
        state = shared_state()
        create(state, "returned")
        state["protocol"] = create_protocol_state()
        state["jobs"] = [
            [
                "reaction",
                {
                    "status": "running",
                    "target": "result",
                    "returned": {"kind": "return", "value": ["ref", "returned"]},
                },
            ]
        ]
        before = copy.deepcopy(state)
        with self.assertRaisesRegex(ProtocolV8Error, "assimilation job"):
            assert_detailed_record_compatible(
                state, "job", {"action": "complete", "job": "reaction", "owner": "root"}
            )
        self.assertEqual(state, before)

    def test_adoption_and_continuations_share_registration_order(self):
        state = shared_state()
        for identity in ("source", "adopter"):
            create(state, identity)
        detail(state, "promise", action="adopt", promise="adopter", target="source")
        detail(state, "promise", action="react", promise="adopter", job="reaction", reaction="then")
        detail(state, "await", promise="source", continuation="resume", invocation="root")
        detail(state, "promise", action="resolve", promise="source", value=["num", 7])
        self.assertEqual(state["ready"], ["reaction", "resume"])
        self.assertEqual(dict(state["heap"])["adopter"]["result"], ["num", 7])
        before = copy.deepcopy(state)
        with self.assertRaisesRegex(NativeV8Error, "FIFO"):
            detail(
                state,
                "resume",
                continuation="resume",
                invocation="root",
                status="fulfilled",
                result=["num", 7],
            )
        self.assertEqual(state, before)
        detail(state, "job", action="dispatch", job="reaction", owner="root")
        detail(state, "job", action="complete", job="reaction", owner="root")
        before = copy.deepcopy(state)
        with self.assertRaisesRegex(NativeV8Error, "settlement"):
            detail(
                state,
                "resume",
                continuation="resume",
                invocation="root",
                status="fulfilled",
                result=["num", 99],
            )
        self.assertEqual(state, before)
        detail(
            state,
            "resume",
            continuation="resume",
            invocation="root",
            status="fulfilled",
            result=["num", 7],
        )
        self.assertEqual(state["stack"], ["root"])

    def test_adoption_cycle_and_external_settlement_are_rejected_atomically(self):
        state = shared_state()
        create(state, "left")
        create(state, "right")
        detail(state, "promise", action="adopt", promise="left", target="right")
        before = copy.deepcopy(state)
        with self.assertRaisesRegex(NativeV8Error, "cycle"):
            detail(state, "promise", action="adopt", promise="right", target="left")
        self.assertEqual(state, before)
        dict(state["heap"])["right"]["external"] = "request"
        before = copy.deepcopy(state)
        with self.assertRaisesRegex(NativeV8Error, "observed"):
            detail(state, "promise", action="reject", promise="right", value=["num", 1])
        self.assertEqual(state, before)

    def test_finally_waits_for_cleanup_and_preserves_rejection(self):
        state = shared_state()
        for identity in ("source", "cleanup"):
            create(state, identity)
        callback(state)
        detail(state, "promise", action="resolve", promise="source", value=["num", 1])
        detail(
            state,
            "promise",
            action="react",
            promise="source",
            job="cleanup-job",
            reaction="finally",
            closure="callback",
            target="result",
        )
        detail(state, "await", promise="result", continuation="resume", invocation="root")
        detail(state, "job", action="dispatch", job="cleanup-job", owner="root")
        self.assertEqual(dict(state["locals"])["jcleanup-jobp0"], ["undefined"])
        detail(state, "return", invocation="job:cleanup-job", value=["ref", "cleanup"])
        detail(state, "job", action="complete", job="cleanup-job", owner="root")
        self.assertEqual(dict(state["jobs"])["cleanup-job"]["status"], "waiting-cleanup")
        self.assertEqual(state["captureAliases"], [])
        self.assertEqual(state["ready"], [])
        detail(
            state, "promise", action="reject", promise="cleanup", value=["str", "cleanup failed"]
        )
        self.assertEqual(dict(state["jobs"])["cleanup-job"]["status"], "completed")
        self.assertEqual(dict(state["heap"])["result"]["state"], "rejected")
        self.assertEqual(dict(state["heap"])["result"]["result"], ["str", "cleanup failed"])
        self.assertEqual(state["ready"], ["resume"])

    def test_dispatch_waits_for_stack_and_terminal_waits_for_job(self):
        state = shared_state()
        callback(state)
        detail(state, "job", action="enqueue", job="job", owner="root", closure="callback")
        before = copy.deepcopy(state)
        with self.assertRaisesRegex(NativeV8Error, "stack turn"):
            detail(state, "job", action="dispatch", job="job", owner="root")
        self.assertEqual(state, before)
        detail(state, "return", invocation="root", value=["num", 1])
        with self.assertRaisesRegex(NativeV8Error, "outstanding"):
            detail(state, "end", kind="return")
        detail(state, "job", action="dispatch", job="job", owner="root")
        detail(state, "throw", invocation="job:job", value=["str", "failed"])
        detail(state, "job", action="complete", job="job", owner="root")
        _apply_detail(state, "end", {"kind": "throw"}, {}, completion_policy="owned-quiescence")
        self.assertEqual(state["completion"], {"kind": "throw", "value": ["str", "failed"]})

    def test_protocol_helpers_can_complete_inside_job_but_cannot_close_job(self):
        state = machine()
        allocate(state, "helper", "closure")
        state["stack"] = ["job:reaction"]
        state["frameFunctions"]["job:reaction"] = "callback"
        state["activeJob"] = "reaction"
        state["jobs"] = [["reaction", {"status": "running"}]]
        apply_protocol_record(
            state,
            {
                "kind": "call",
                "callee": ["ref", "helper"],
                "receiver": ["undefined"],
                "arguments": [],
                "child": "nested",
            },
        )
        apply_protocol_record(
            state, {"kind": "return", "invocation": "nested", "value": ["num", 7]}
        )
        self.assertEqual(state["stack"], ["job:reaction"])
        self.assertEqual(state["activeJob"], "reaction")
        self.assertEqual(state["lastReturn"], {"kind": "return", "value": ["num", 7]})
        before = _external_machine(state)
        state = _internal_machine(before)
        with self.assertRaisesRegex(ProtocolV8Error, "scheduler"):
            apply_protocol_record(
                state, {"kind": "return", "invocation": "job:reaction", "value": ["num", 9]}
            )
        self.assertEqual(_external_machine(state), before)
