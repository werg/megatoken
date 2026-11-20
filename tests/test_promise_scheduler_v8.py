import copy
import unittest

from megatoken.data.native_v8 import NativeV8Error, _apply_detail, _initial_state
from megatoken.data.promise_scheduler_v8 import settle_promise
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
    def test_thenable_resolution_checks_descriptors_without_invoking_them(self):
        for inherited in (False, True):
            for accessor in (False, True):
                state = shared_state()
                create(state, "promise")
                callback(state)
                detail(state, "allocate", name="parent", kind="object", properties=[])
                detail(state, "allocate", name="value", kind="object", properties=[])
                nodes = dict(state["heap"])
                owner = nodes["parent"] if inherited else nodes["value"]
                if inherited:
                    nodes["value"]["prototype"] = ["ref", "parent"]
                if accessor:
                    owner["descriptors"] = [
                        [
                            "then",
                            {
                                "kind": "accessor",
                                "get": ["ref", "callback"],
                                "set": ["undefined"],
                                "enumerable": True,
                                "configurable": True,
                            },
                        ]
                    ]
                else:
                    owner["properties"].append(["then", ["ref", "callback"]])
                before = copy.deepcopy(state)
                with self.assertRaisesRegex(
                    NativeV8Error, "then.*(?:resolution protocol|assimilation jobs)"
                ):
                    detail(
                        state,
                        "promise",
                        action="resolve",
                        promise="promise",
                        value=["ref", "value"],
                    )
                self.assertEqual(state, before)
                machine = _internal_machine(state)
                machine["heap"]["promise"]["external"] = "request"
                before = copy.deepcopy(machine)
                with self.assertRaises(ProtocolV8Error):
                    settle_promise(machine, "promise", "fulfilled", ["ref", "value"], external=True)
                self.assertEqual(machine, before)
        # A data property with a noncallable then remains an ordinary value.
        state = shared_state()
        create(state, "promise")
        detail(
            state,
            "allocate",
            name="value",
            kind="object",
            properties=[{"key": "then", "value": ["num", 3]}],
        )
        detail(state, "promise", action="resolve", promise="promise", value=["ref", "value"])
        self.assertEqual(dict(state["heap"])["promise"]["result"], ["ref", "value"])

    def test_resolution_cannot_treat_unknown_intrinsic_prototypes_as_absent_then(self):
        state = shared_state()
        create(state, "promise")
        detail(state, "allocate", name="value", kind="object", properties=[])
        dict(state["heap"])["value"]["prototype"] = "unmodeled.prototype"
        before = copy.deepcopy(state)
        with self.assertRaisesRegex(NativeV8Error, "unmodeled prototype"):
            detail(state, "promise", action="resolve", promise="promise", value=["ref", "value"])
        self.assertEqual(state, before)

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

    def test_bare_adoption_requires_assimilation_jobs_without_mutating_readiness(self):
        for settled in (False, True):
            for action in ("adopt", "resolve"):
                with self.subTest(settled=settled, action=action):
                    state = shared_state()
                    for identity in ("source", "adopter"):
                        create(state, identity)
                    detail(
                        state,
                        "promise",
                        action="react",
                        promise="adopter",
                        job="early",
                        reaction="then",
                    )
                    detail(
                        state,
                        "promise",
                        action="react",
                        promise="source",
                        job="direct",
                        reaction="then",
                    )
                    if settled:
                        detail(
                            state, "promise", action="resolve", promise="source", value=["num", 7]
                        )
                    before = copy.deepcopy(state)
                    fields = (
                        {"target": "source"} if action == "adopt" else {"value": ["ref", "source"]}
                    )
                    with self.assertRaisesRegex(NativeV8Error, "assimilation jobs"):
                        detail(state, "promise", action=action, promise="adopter", **fields)
                    self.assertEqual(state, before)

    def test_adoption_lock_targets_and_external_settlements_fail_closed(self):
        state = shared_state()
        create(state, "left")
        create(state, "right")
        dict(state["heap"])["left"]["adopted"] = "right"
        for action in ("resolve", "reject", "adopt"):
            before = copy.deepcopy(state)
            fields = {"target": "right"} if action == "adopt" else {"value": ["num", 1]}
            with self.assertRaisesRegex(NativeV8Error, "adoption lock"):
                detail(state, "promise", action=action, promise="left", **fields)
            self.assertEqual(state, before)
        state = shared_state()
        create(state, "left")
        detail(state, "allocate", name="ordinary", kind="object", properties=[])
        before = copy.deepcopy(state)
        with self.assertRaisesRegex(NativeV8Error, "Promise identity"):
            detail(state, "promise", action="adopt", promise="left", target="ordinary")
        self.assertEqual(state, before)
        # Ordinary reference fulfillment and opaque Promise rejection reasons
        # retain their meanings; neither is an adoption record.
        detail(state, "promise", action="resolve", promise="left", value=["ref", "ordinary"])
        create(state, "right")
        detail(state, "promise", action="reject", promise="right", value=["ref", "left"])
        self.assertEqual(dict(state["heap"])["right"]["result"], ["ref", "left"])
        create(state, "external")
        dict(state["heap"])["external"]["external"] = "request"
        before = copy.deepcopy(state)
        with self.assertRaisesRegex(NativeV8Error, "observed"):
            detail(state, "promise", action="reject", promise="external", value=["num", 1])
        self.assertEqual(state, before)

    def test_generator_and_resolver_callbacks_cannot_use_ordinary_jobs(self):
        for metadata in (
            {"generatorFunction": True},
            {"promiseResolver": {"promise": "source", "mode": "resolve"}},
        ):
            for field in ("closure", "rejectionClosure", "enqueue", "dispatch"):
                with self.subTest(metadata=metadata, field=field):
                    state = shared_state()
                    callback(state)
                    create(state, "source")
                    if field == "dispatch":
                        detail(
                            state,
                            "promise",
                            action="react",
                            promise="source",
                            job="reaction",
                            reaction="then",
                            closure="callback",
                        )
                        detail(
                            state, "promise", action="resolve", promise="source", value=["num", 1]
                        )
                        detail(state, "return", invocation="root", value=["num", 0])
                    dict(state["heap"])["callback"].update(metadata)
                    before = copy.deepcopy(state)
                    with self.assertRaisesRegex(NativeV8Error, "explicit call protocol"):
                        if field == "enqueue":
                            detail(
                                state,
                                "job",
                                action="enqueue",
                                job="reaction",
                                owner="root",
                                closure="callback",
                            )
                        elif field == "dispatch":
                            detail(state, "job", action="dispatch", job="reaction", owner="root")
                        else:
                            detail(
                                state,
                                "promise",
                                action="react",
                                promise="source",
                                job="reaction",
                                reaction="then",
                                **{field: "callback"},
                            )
                    self.assertEqual(state, before)

    def test_normal_finally_completion_requires_cleanup_jobs_but_throw_is_direct(self):
        for value in (["num", 7], ["undefined"], ["ref", "cleanup"]):
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
            detail(state, "return", invocation="job:cleanup-job", value=value)
            before = copy.deepcopy(state)
            with self.assertRaisesRegex(NativeV8Error, "cleanup jobs"):
                detail(state, "job", action="complete", job="cleanup-job", owner="root")
            self.assertEqual(state, before)
        state = shared_state()
        create(state, "source")
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
        detail(state, "throw", invocation="job:cleanup-job", value=["str", "cleanup failed"])
        detail(state, "job", action="complete", job="cleanup-job", owner="root")
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
