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

    def test_unpublished_result_completion_must_match_saved_phase(self):
        for count, done in ((11, True), (16, False)):
            with self.subTest(count=count):
                state, _ = self.prefix("delegate-array-preserves-result-identity", count)
                result = state["heap"]["it"]["delegation"]["result"][1]
                state["heap"][result]["properties"]["done"] = ["bool", done]
                shared = _external_machine(state)
                with self.assertRaisesRegex(ProtocolV8Error, "saved delegate completion"):
                    validate_protocol_state(shared["protocol"], shared["heap"], shared)
