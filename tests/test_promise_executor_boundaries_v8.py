import copy
import unittest

from megatoken.data.promise_executor_v8 import validate_state
from megatoken.data.promise_scheduler_v8 import create_promise
from megatoken.data.protocol_v8 import (
    ProtocolV8Error,
    _external_machine,
    _internal_machine,
    apply_protocol_record,
    assert_detailed_record_compatible,
    decode_protocol_record,
    encode_protocol_record,
    validate_protocol_state,
)
from tests.test_number_v8 import allocate, machine


def executor_machine(active_job=False):
    state = machine()
    state.update(jobs=[], continuations=[], ready=[], scheduler={"registration": 0, "ready": 0})
    create_promise(state, "promise")
    allocate(state, "executor", "closure")
    if active_job:
        state["activeJob"] = "reaction"
        state["stack"] = ["job:reaction"]
        state["frameFunctions"]["job:reaction"] = "callback"
        state["jobs"] = [["reaction", {"status": "running"}]]
    return state


def enter(state):
    apply_protocol_record(
        state,
        {
            "kind": "promise.executor.enter",
            "promise": "promise",
            "executor": ["ref", "executor"],
            "child": "executor-call",
            "resolve": "resolve",
            "reject": "reject",
        },
    )


def resolve(state, value, resolver="resolve"):
    record = {"kind": "promise.resolver.call", "resolver": ["ref", resolver], "value": value}
    apply_protocol_record(state, decode_protocol_record(encode_protocol_record(record)))


def finish(state, kind="return", value=None):
    apply_protocol_record(
        state,
        {
            "kind": "promise.executor.finish",
            "promise": "promise",
            "invocation": "executor-call",
            "completion": {"kind": kind, "value": ["undefined"] if value is None else value},
        },
    )


def restore(state):
    shared = _external_machine(state)
    validate_protocol_state(shared["protocol"], shared["heap"], shared)
    return _internal_machine(shared)


class PromiseExecutorTests(unittest.TestCase):
    def test_generator_executor_is_refused_before_opening_an_ordinary_call(self):
        state = executor_machine()
        apply_protocol_record(
            state,
            {
                "kind": "generator.function",
                "object": "executor",
                "prototype": "generator-prototype",
                "name": "executor",
                "length": 2,
            },
        )
        before = copy.deepcopy(state)
        with self.assertRaisesRegex(ProtocolV8Error, "owned source closure"):
            enter(state)
        self.assertEqual(state, before)

    def test_join_cannot_reuse_executor_result_but_can_observe_it_as_input(self):
        state = executor_machine()
        enter(state)
        finish(state)
        record = {
            "kind": "promise.join",
            "mode": "all",
            "result": "promise",
            "inputs": [],
            "jobs": [],
            "output": "output",
            "error": None,
        }
        before = copy.deepcopy(state)
        with self.assertRaisesRegex(ProtocolV8Error, "join result"):
            apply_protocol_record(state, record)
        self.assertEqual(state, before)
        create_promise(state, "join-result")
        record.update(result="join-result", inputs=["promise"], jobs=["join-job"], output=None)
        apply_protocol_record(state, record)
        self.assertTrue(validate_state(restore(state)))
        resolve(state, ["num", 7])
        self.assertEqual(state["heap"]["promise"]["result"], ["num", 7])
        self.assertEqual(state["ready"], ["join-job"])

    def test_restored_executor_capability_cannot_share_a_join_result(self):
        state = executor_machine()
        enter(state)
        finish(state)
        state["heap"]["promise"]["join"] = {
            "schema": "semantic-promise-join-state/1",
            "mode": "race",
            "inputs": [],
            "jobs": [],
            "observed": [],
            "values": [],
            "output": None,
            "error": None,
            "settled": False,
        }
        with self.assertRaisesRegex(ProtocolV8Error, "cannot own a join"):
            validate_state(state)

    def test_malformed_executor_identities_raise_protocol_errors_on_restore(self):
        state = executor_machine()
        enter(state)
        for identity in ([], {}, None, True, 7):
            with self.subTest(identity=identity):
                malformed = copy.deepcopy(state)
                malformed["protocol"]["promiseExecutors"]["executors"][0]["promise"] = identity
                with self.assertRaisesRegex(ProtocolV8Error, "identity"):
                    validate_state(malformed)

    def test_first_resolving_call_wins_over_opposite_handle_and_executor_throw(self):
        state = executor_machine()
        enter(state)
        state = restore(state)
        resolve(state, ["num", 7])
        state = restore(state)
        resolve(state, ["str", "ignored"], "reject")
        finish(state, "throw", ["str", "also ignored"])
        state = restore(state)
        self.assertEqual(state["heap"]["promise"]["state"], "fulfilled")
        self.assertEqual(state["heap"]["promise"]["result"], ["num", 7])
        self.assertEqual(state["lastReturn"], {"kind": "return", "value": ["ref", "promise"]})
        self.assertEqual(state["stack"], ["root"])
        self.assertEqual(state["protocol"]["callFrames"], [])

    def test_resolving_functions_survive_executor_completion(self):
        for thrown in (False, True):
            with self.subTest(thrown=thrown):
                state = executor_machine()
                enter(state)
                finish(state, "throw" if thrown else "return", ["str", "boom"])
                state = restore(state)
                resolve(state, ["num", 9])
                resolve(state, ["ref", "executor"])
                state = restore(state)
                self.assertEqual(
                    state["heap"]["promise"]["state"], "rejected" if thrown else "fulfilled"
                )
                self.assertEqual(
                    state["heap"]["promise"]["result"], ["str", "boom"] if thrown else ["num", 9]
                )

    def test_executor_throw_consumes_the_resolving_pair_before_later_calls(self):
        state = executor_machine()
        enter(state)
        finish(state, "throw", ["str", "boom"])
        capability = state["protocol"]["promiseExecutors"]["capabilities"][0]
        self.assertTrue(capability["called"])
        self.assertEqual(capability["resolution"], {"mode": "reject", "value": ["str", "boom"]})
        state = restore(state)
        before = copy.deepcopy(state)
        for resolver, value in (("resolve", ["ref", "executor"]), ("reject", ["str", "late"])):
            resolve(state, value, resolver)
            before["protocol"]["sequence"] += 1
            self.assertEqual(state, before)
            state = restore(state)

    def test_throw_completion_cannot_restore_an_unconsumed_or_false_first_call(self):
        state = executor_machine()
        enter(state)
        finish(state, "throw", ["str", "boom"])
        for mutation in ("uncalled", "false-value", "pending"):
            with self.subTest(mutation=mutation):
                changed = copy.deepcopy(state)
                capability = changed["protocol"]["promiseExecutors"]["capabilities"][0]
                if mutation == "false-value":
                    capability.update(
                        called=True, resolution={"mode": "resolve", "value": ["num", 9]}
                    )
                else:
                    capability.update(called=False, resolution=None)
                    if mutation == "pending":
                        changed["heap"]["promise"].update(state="pending", result=None)
                with self.assertRaises(ProtocolV8Error):
                    restore(changed)

    def test_reference_resolution_is_refused_before_first_call_but_rejection_is_owned(self):
        state = executor_machine()
        allocate(state, "object")
        create_promise(state, "other-promise")
        enter(state)
        before = copy.deepcopy(state)
        with self.assertRaisesRegex(ProtocolV8Error, "assimilation"):
            resolve(state, ["ref", "object"])
        self.assertEqual(state, before)
        resolve(state, ["ref", "object"], "reject")
        finish(state)
        self.assertTrue(validate_state(restore(state)))
        self.assertEqual(state["heap"]["promise"]["result"], ["ref", "object"])

        state = executor_machine()
        create_promise(state, "other-promise")
        enter(state)
        resolve(state, ["ref", "other-promise"])
        self.assertEqual(state["heap"]["promise"]["state"], "pending")
        apply_protocol_record(
            state,
            {
                "kind": "promise.adopt.enqueue",
                "promise": "promise",
                "target": "other-promise",
                "job": "adopt-job",
            },
        )
        finish(state)
        self.assertTrue(validate_state(restore(state)))

    def test_executor_inside_reaction_preserves_job_and_nested_helper_scope(self):
        state = executor_machine(active_job=True)
        allocate(state, "helper", "closure")
        enter(state)
        apply_protocol_record(
            state,
            {
                "kind": "call",
                "callee": ["ref", "helper"],
                "child": "helper-call",
                "receiver": ["undefined"],
                "arguments": [],
            },
        )
        apply_protocol_record(
            state, {"kind": "return", "invocation": "helper-call", "value": ["num", 4]}
        )
        resolve(state, ["num", 4])
        finish(state)
        state = restore(state)
        self.assertEqual(state["activeJob"], "reaction")
        self.assertEqual(state["stack"], ["job:reaction"])
        self.assertEqual(state["heap"]["promise"]["result"], ["num", 4])

    def test_unscoped_completion_settlement_and_resolver_calls_are_rejected(self):
        state = executor_machine()
        enter(state)
        records = [
            {"kind": "return", "invocation": "executor-call", "value": ["undefined"]},
            {
                "kind": "call",
                "child": "illegal",
                "callee": ["ref", "resolve"],
                "receiver": ["undefined"],
                "arguments": [["num", 1]],
            },
            {
                "kind": "property.set",
                "object": "resolve",
                "receiver": ["ref", "resolve"],
                "key": ["str", "x"],
                "value": ["num", 1],
                "success": True,
            },
        ]
        before = copy.deepcopy(state)
        for record in records:
            with self.subTest(kind=record["kind"]), self.assertRaises(ProtocolV8Error):
                apply_protocol_record(state, record)
            self.assertEqual(state, before)
        with self.assertRaisesRegex(ProtocolV8Error, "resolver record"):
            assert_detailed_record_compatible(
                _external_machine(state),
                "promise",
                {"action": "resolve", "promise": "promise", "value": ["num", 1]},
            )

    def test_restoration_checks_first_call_cell_and_original_resolving_pair(self):
        state = executor_machine()
        enter(state)
        cases = []
        changed = copy.deepcopy(state)
        changed["protocol"]["callFrames"][0]["arguments"].reverse()
        cases.append(changed)
        changed = copy.deepcopy(state)
        changed["heap"]["resolve"]["promiseResolver"]["mode"] = "reject"
        cases.append(changed)
        changed = copy.deepcopy(state)
        changed["protocol"]["promiseExecutors"]["capabilities"][0]["called"] = True
        cases.append(changed)
        for changed in cases:
            with self.assertRaises(ProtocolV8Error):
                validate_state(changed)
        resolve(state, ["num", 1])
        finish(state)
        changed = copy.deepcopy(state)
        changed["heap"]["promise"]["result"] = ["num", 2]
        with self.assertRaisesRegex(ProtocolV8Error, "contradicts"):
            validate_state(changed)

    def test_rejected_first_entry_and_unknown_resolver_leave_state_unchanged(self):
        state = executor_machine()
        allocate(state, "resolve")
        before = copy.deepcopy(state)
        with self.assertRaisesRegex(ProtocolV8Error, "identity"):
            enter(state)
        self.assertEqual(state, before)
        with self.assertRaisesRegex(ProtocolV8Error, "undeclared"):
            resolve(state, ["num", 1], "executor")
        self.assertEqual(state, before)
