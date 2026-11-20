"""Generator continuations preserve identities and already-consumed decisions."""

import copy
import json
import unittest
from pathlib import Path

from megatoken.data.protocol_v8 import (
    ProtocolV8Error,
    _external_machine,
    _internal_machine,
    apply_protocol_record,
    assert_detailed_record_compatible,
    decode_protocol_record,
    validate_protocol_state,
)

FIXTURE = Path(__file__).parent / "fixtures" / "semantic-generator-v8.json"


class GeneratorBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = {case["id"]: case for case in json.loads(FIXTURE.read_text())["cases"]}

    def prefix(self, case_id, count):
        case = self.cases[case_id]
        state = _internal_machine(case["initialState"])
        records = [decode_protocol_record(line) for line in case["lines"]]
        for record in records[:count]:
            apply_protocol_record(state, record)
        return state, records

    def test_creation_rejects_existing_heap_symbol_and_private_identities_atomically(self):
        state, records = self.prefix("return-before-body-start", 2)
        apply_protocol_record(
            state, {"kind": "symbol.create", "id": "o7", "description": ["undefined"]}
        )
        state["protocol"]["construction"] = {
            "schema": "semantic-construction-state/1",
            "privateNames": [{"id": "private"}],
            "frames": [],
            "lastCompletion": None,
        }
        for identity in ("fn", "proto", "o7", "private"):
            with self.subTest(identity=identity):
                before = copy.deepcopy(state)
                with self.assertRaisesRegex(ProtocolV8Error, "heap identity"):
                    apply_protocol_record(state, {**records[2], "name": identity})
                self.assertEqual(state, before)

    def test_active_generator_rejects_generic_completion_but_allows_ordinary_children(self):
        state, _ = self.prefix("delegate-array-preserves-result-identity", 6)
        for kind in ("return", "throw"):
            record = {"kind": kind, "invocation": "generator1", "value": ["num", 7]}
            before = copy.deepcopy(state)
            with self.assertRaisesRegex(ProtocolV8Error, "generator completion"):
                apply_protocol_record(state, record)
            self.assertEqual(state, before)
            with self.assertRaisesRegex(ProtocolV8Error, "generator completion"):
                assert_detailed_record_compatible(_external_machine(state), kind, record)
        ordinary = copy.deepcopy(state)
        apply_protocol_record(
            ordinary,
            {
                "kind": "allocate",
                "name": "ordinary",
                "objectKind": "closure",
                "code": "ordinary",
                "captures": [],
                "functionKind": "ordinary",
                "lexicalReceiver": None,
                "length": None,
                "properties": [],
            },
        )
        apply_protocol_record(
            ordinary,
            {
                "kind": "call",
                "child": "ordinary-child",
                "callee": ["ref", "ordinary"],
                "receiver": ["undefined"],
                "arguments": [],
            },
        )
        apply_protocol_record(
            ordinary, {"kind": "return", "invocation": "ordinary-child", "value": ["num", 7]}
        )
        self.assertEqual(ordinary["stack"][-1], "generator1")

    def test_published_result_mutation_survives_checkpoint_and_resume(self):
        state, records = self.prefix("delegate-array-preserves-result-identity", 12)
        apply_protocol_record(
            state,
            {
                "kind": "property.set",
                "object": "step1",
                "key": ["str", "done"],
                "receiver": ["ref", "step1"],
                "value": ["str", "changed after yield"],
                "success": True,
            },
        )
        shared = json.loads(json.dumps(_external_machine(state)))
        self.assertTrue(validate_protocol_state(shared["protocol"], shared["heap"], shared))
        state = _internal_machine(shared)
        for record in records[12:]:
            apply_protocol_record(state, record)
        self.assertEqual(
            state["heap"]["step1"]["properties"]["done"], ["str", "changed after yield"]
        )
        self.assertEqual(state["heap"]["it"]["phase"], "completed")

    def test_ordinary_generator_properties_preserve_aliases_and_intrinsic_guards(self):
        state, _ = self.prefix("return-before-body-start", 4)
        for key in (["str", "tag"], ["symbol", "well-known:match"]):
            apply_protocol_record(
                state,
                {
                    "kind": "property.set",
                    "object": "it",
                    "key": key,
                    "receiver": ["ref", "it"],
                    "value": ["ref", "fn"],
                    "success": True,
                },
            )
            apply_protocol_record(
                state,
                {
                    "kind": "property.get",
                    "object": "it",
                    "key": key,
                    "receiver": ["ref", "it"],
                    "result": ["ref", "fn"],
                },
            )
        before = copy.deepcopy(state)
        with self.assertRaisesRegex(ProtocolV8Error, "intrinsic property"):
            apply_protocol_record(
                state,
                {
                    "kind": "property.get",
                    "object": "it",
                    "key": ["str", "next"],
                    "receiver": ["ref", "it"],
                    "result": ["undefined"],
                },
            )
        self.assertEqual(state, before)
        tag = ["symbol", "well-known:toStringTag"]
        apply_protocol_record(
            state,
            {
                "kind": "property.set",
                "object": "it",
                "key": tag,
                "receiver": ["ref", "it"],
                "value": ["str", "changed"],
                "success": False,
            },
        )
        apply_protocol_record(
            state,
            {
                "kind": "property.get",
                "object": "it",
                "key": tag,
                "receiver": ["ref", "it"],
                "result": ["str", "Generator"],
            },
        )
        shared = _external_machine(state)
        self.assertTrue(validate_protocol_state(shared["protocol"], shared["heap"], shared))

    def test_unpublished_result_completion_must_match_saved_phase(self):
        for count, done in ((11, True), (16, False)):
            with self.subTest(count=count):
                state, _ = self.prefix("delegate-array-preserves-result-identity", count)
                result = state["heap"]["it"]["delegation"]["result"][1]
                state["heap"][result]["properties"]["done"] = ["bool", done]
                shared = _external_machine(state)
                with self.assertRaisesRegex(ProtocolV8Error, "saved delegate completion"):
                    validate_protocol_state(shared["protocol"], shared["heap"], shared)

    def test_restore_rejects_fabricated_frame_resume_and_handler_state(self):
        state, _ = self.prefix("normal-sent-value-and-completion", 7)
        mutations = {
            "unknown field": lambda node: node.update(pc=42),
            "unknown invocation": lambda node: node.update(invocation="invented"),
            "unknown saved argument": lambda node: node.update(arguments=[["ref", "missing"]]),
            "unknown receiver": lambda node: node.update(receiver=["ref", "missing"]),
            "resume outside active frame": lambda node: node.update(
                resume={"kind": "next", "from": "suspendedYield", "value": ["num", 3]}
            ),
            "pending completion while suspended": lambda node: node.update(
                pending={"kind": "return", "value": ["num", 3]}
            ),
            "fake handler capability": lambda node: node.update(
                handlers=[
                    {
                        "name": "h",
                        "catch": 1,
                        "finally": False,
                        "phase": "try",
                        "caught": None,
                        "pending": None,
                    }
                ]
            ),
            "primitive prototype": lambda node: node.update(prototype=["num", 3]),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                broken = copy.deepcopy(state)
                mutate(broken["heap"]["it"])
                shared = _external_machine(broken)
                with self.assertRaises(ProtocolV8Error):
                    validate_protocol_state(shared["protocol"], shared["heap"], shared)

    def test_restore_rejects_changed_initial_next_input_and_cleanup_obligations(self):
        state, _ = self.prefix("normal-sent-value-and-completion", 5)
        state["heap"]["it"]["resume"]["value"] = ["num", 99]
        shared = _external_machine(state)
        with self.assertRaisesRegex(ProtocolV8Error, "initial next input"):
            validate_protocol_state(shared["protocol"], shared["heap"], shared)
        state, _ = self.prefix("cleanup-retains-return-across-yield", 11)
        mutations = (
            lambda handler: handler.update(pending={"kind": "normal", "value": ["num", 3]}),
            lambda handler: handler.update(caught=["num", 7]),
            lambda handler: handler.update(phase="try"),
        )
        for mutate in mutations:
            broken = copy.deepcopy(state)
            mutate(broken["heap"]["it"]["handlers"][0])
            shared = _external_machine(broken)
            with self.assertRaises(ProtocolV8Error):
                validate_protocol_state(shared["protocol"], shared["heap"], shared)

    def test_restore_requires_exact_delegate_phase_and_known_request(self):
        state, _ = self.prefix("delegate-array-preserves-result-identity", 12)
        for change in (
            {"iterator": ["ref", "it"]},
            {"request": {"kind": "next", "value": ["ref", "missing"]}},
            {"request": {"kind": "next", "value": ["num", 3], "extra": True}},
            {"phase": "ready"},
            {"result": None},
        ):
            with self.subTest(change=change):
                broken = copy.deepcopy(state)
                broken["heap"]["it"]["delegation"].update(change)
                shared = _external_machine(broken)
                with self.assertRaises(ProtocolV8Error):
                    validate_protocol_state(shared["protocol"], shared["heap"], shared)

    def test_generator_prototype_may_become_nonwritable_and_bad_resume_kind_is_protocol_error(self):
        state, _ = self.prefix("normal-sent-value-and-completion", 5)
        state["heap"]["fn"]["descriptors"]["prototype"]["writable"] = False
        shared = _external_machine(state)
        self.assertTrue(validate_protocol_state(shared["protocol"], shared["heap"], shared))

        broken = copy.deepcopy(state)
        broken["heap"]["it"]["resume"]["kind"] = []
        shared = _external_machine(broken)
        with self.assertRaises(ProtocolV8Error):
            validate_protocol_state(shared["protocol"], shared["heap"], shared)
