import copy
import unittest

from megatoken.data.protocol_v8 import (
    ProtocolV8Error,
    _external_machine,
    apply_protocol_record,
    create_protocol_state,
    validate_protocol_state,
)


def machine():
    return {
        "protocol": create_protocol_state(),
        "heap": {
            "promise": {
                "kind": "promise",
                "properties": {},
                "state": "pending",
                "result": None,
                "adopted": None,
                "external": None,
                "reactions": [],
            }
        },
        "locals": {},
        "stack": ["root"],
        "frameFunctions": {"root": "root-code"},
        "pending": None,
        "lastReturn": None,
        "completion": None,
        "captureAliases": {},
        "receiver": ["undefined"],
    }


def enter(state):
    apply_protocol_record(
        state,
        {
            "kind": "promise.executor.enter",
            "promise": "promise",
            "executor": ["ref", "executor"],
            "child": "executor-call",
            "resolve": "resolve-id",
            "reject": "reject-id",
        },
    )


def executor_fixture():
    state = machine()
    apply_protocol_record(
        state,
        {
            "kind": "allocate",
            "name": "executor",
            "objectKind": "closure",
            "code": "executor-code",
            "captures": [],
            "functionKind": "ordinary",
            "lexicalReceiver": None,
            "length": None,
            "properties": [],
        },
    )
    return state


class PromiseExecutorV8Tests(unittest.TestCase):
    def test_synchronous_executor_resolves_once_and_finishes_with_promise(self):
        state = executor_fixture()
        enter(state)
        self.assertEqual(state["stack"], ["root", "executor-call"])
        self.assertEqual(state["locals"]["executor-call:argument:0"], ["ref", "resolve-id"])
        apply_protocol_record(
            state,
            {
                "kind": "promise.resolver.call",
                "resolver": ["ref", "resolve-id"],
                "value": ["num", 7],
            },
        )
        before = copy.deepcopy(state["heap"]["promise"])
        apply_protocol_record(
            state,
            {
                "kind": "promise.resolver.call",
                "resolver": ["ref", "reject-id"],
                "value": ["str", "late"],
            },
        )
        self.assertEqual(state["heap"]["promise"], before)
        apply_protocol_record(
            state,
            {
                "kind": "promise.executor.finish",
                "promise": "promise",
                "invocation": "executor-call",
                "completion": {"kind": "return", "value": ["undefined"]},
            },
        )
        self.assertEqual(state["heap"]["promise"]["state"], "fulfilled")
        self.assertEqual(state["lastReturn"], {"kind": "return", "value": ["ref", "promise"]})
        self.assertTrue(
            validate_protocol_state(
                _external_machine(state)["protocol"],
                _external_machine(state)["heap"],
                _external_machine(state),
            )
        )

    def test_throwing_executor_rejects_pending_promise(self):
        state = executor_fixture()
        enter(state)
        apply_protocol_record(
            state,
            {
                "kind": "promise.executor.finish",
                "promise": "promise",
                "invocation": "executor-call",
                "completion": {"kind": "throw", "value": ["str", "failure"]},
            },
        )
        self.assertEqual(state["heap"]["promise"]["state"], "rejected")
        self.assertEqual(state["heap"]["promise"]["result"], ["str", "failure"])
        self.assertTrue(
            validate_protocol_state(
                _external_machine(state)["protocol"],
                _external_machine(state)["heap"],
                _external_machine(state),
            )
        )

    def test_resolution_adopts_no_objects_and_reject_first_call_wins(self):
        state = executor_fixture()
        apply_protocol_record(
            state,
            {
                "kind": "allocate",
                "name": "other",
                "objectKind": "object",
                "code": None,
                "captures": [],
                "functionKind": None,
                "lexicalReceiver": None,
                "length": None,
                "properties": [],
            },
        )
        enter(state)
        before = copy.deepcopy(state)
        with self.assertRaisesRegex(ProtocolV8Error, "assimilation protocol"):
            apply_protocol_record(
                state,
                {
                    "kind": "promise.resolver.call",
                    "resolver": ["ref", "resolve-id"],
                    "value": ["ref", "other"],
                },
            )
        self.assertEqual(state, before)
        apply_protocol_record(
            state,
            {
                "kind": "promise.resolver.call",
                "resolver": ["ref", "reject-id"],
                "value": ["str", "first rejection"],
            },
        )
        apply_protocol_record(
            state,
            {
                "kind": "promise.resolver.call",
                "resolver": ["ref", "resolve-id"],
                "value": ["num", 99],
            },
        )
        self.assertEqual(state["heap"]["promise"]["state"], "rejected")
        self.assertEqual(state["heap"]["promise"]["result"], ["str", "first rejection"])

    def test_generic_return_and_direct_call_cannot_bypass_executor_records(self):
        state = executor_fixture()
        enter(state)
        before = copy.deepcopy(state)
        with self.assertRaises(ProtocolV8Error):
            apply_protocol_record(
                state, {"kind": "return", "invocation": "executor-call", "value": ["undefined"]}
            )
        self.assertEqual(state, before)
        with self.assertRaises(ProtocolV8Error):
            apply_protocol_record(
                state,
                {
                    "kind": "call",
                    "callee": ["ref", "resolve-id"],
                    "receiver": ["undefined"],
                    "arguments": [["num", 1]],
                    "child": "resolver-call",
                },
            )
        self.assertEqual(state, before)


if __name__ == "__main__":
    unittest.main()
