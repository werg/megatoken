"""Malformed or exhausted protocol boundaries cannot alter the shared machine."""

import copy
import json
import unittest

from megatoken.data.construction_v8 import _call
from megatoken.data.native_v8 import NativeV8Error, _apply_algorithm, _apply_detail
from megatoken.data.protocol_v8 import (
    ProtocolV8Error,
    _external_machine,
    _internal_machine,
    apply_protocol_record,
    assert_detailed_record_compatible,
    validate_protocol_record,
    validate_protocol_state,
)
from tests.test_number_v8 import allocate, machine


class ProtocolBoundaryTests(unittest.TestCase):
    def test_non_string_record_kind_is_a_protocol_error(self):
        for kind in ([], {}, None, 3):
            with self.subTest(kind=kind), self.assertRaisesRegex(ProtocolV8Error, "kind"):
                validate_protocol_record({"kind": kind})

    def test_missing_terminal_completion_is_rejected_transactionally(self):
        state = machine()
        state["stack"] = []
        before = copy.deepcopy(state)
        with self.assertRaisesRegex(ProtocolV8Error, "terminal completion"):
            apply_protocol_record(state, {"kind": "end", "completion": "return"})
        self.assertEqual(state, before)
        for kind in ("return", "throw"):
            state = copy.deepcopy(before)
            state["pending"] = {"kind": kind, "value": ["num", 7]}
            apply_protocol_record(state, {"kind": "end", "completion": kind})
            self.assertEqual(state["completion"], state["pending"])

    def test_property_capacity_counts_accessors_once_and_permits_updates(self):
        state = machine()
        allocate(state, "object")
        node = state["heap"]["object"]
        node["properties"] = {f"k{i}": ["num", i] for i in range(4095)}
        node.setdefault("descriptors", {})["getter"] = {
            "kind": "accessor",
            "get": ["undefined"],
            "set": ["undefined"],
            "enumerable": True,
            "configurable": True,
        }
        descriptor = {
            "kind": "data",
            "value": ["num", 8],
            "writable": True,
            "enumerable": True,
            "configurable": True,
        }
        for kind in ("property.define", "property.set"):
            row = {"kind": kind, "object": "object", "key": ["str", "new"], "success": True}
            row.update(
                {"descriptor": descriptor}
                if kind == "property.define"
                else {"receiver": ["ref", "object"], "value": ["num", 8]}
            )
            before = copy.deepcopy(state)
            with self.assertRaisesRegex(ProtocolV8Error, "property bound"):
                apply_protocol_record(state, row)
            self.assertEqual(state, before)
            row["key"] = ["str", "k0"]
            apply_protocol_record(state, row)
            self.assertEqual(state["heap"]["object"]["properties"]["k0"], ["num", 8])
        apply_protocol_record(
            state,
            {
                "kind": "property.define",
                "object": "object",
                "key": ["str", "getter"],
                "descriptor": descriptor,
                "success": True,
            },
        )

    def test_call_capacity_rejects_entry_and_allows_return(self):
        state = machine()
        allocate(state, "callee", "closure")
        for i in range(64):
            apply_protocol_record(
                state,
                {
                    "kind": "call",
                    "child": f"child{i}",
                    "callee": ["ref", "callee"],
                    "receiver": ["undefined"],
                    "arguments": [],
                },
            )
        before = copy.deepcopy(state)
        with self.assertRaisesRegex(ProtocolV8Error, "call frame bound"):
            apply_protocol_record(
                state,
                {
                    "kind": "call",
                    "child": "overflow",
                    "callee": ["ref", "callee"],
                    "receiver": ["undefined"],
                    "arguments": [],
                },
            )
        self.assertEqual(state, before)
        with self.assertRaisesRegex(ProtocolV8Error, "call frame bound"):
            _call(state, "overflow", ["ref", "callee"], ["undefined"], [], construction=True)
        self.assertEqual(state, before)
        apply_protocol_record(
            state, {"kind": "return", "invocation": "child63", "value": ["num", 7]}
        )
        self.assertEqual(len(state["protocol"]["callFrames"]), 63)

    def test_allocation_and_restore_apply_the_same_property_capacity(self):
        state = machine()
        row = {
            "kind": "allocate",
            "name": "array",
            "objectKind": "array",
            "length": 4096,
            "properties": [{"key": ["str", str(i)], "value": ["num", i]} for i in range(4096)],
            "code": None,
            "captures": [],
            "functionKind": None,
            "lexicalReceiver": None,
        }
        before = copy.deepcopy(state)
        with self.assertRaisesRegex(ProtocolV8Error, "allocation properties"):
            apply_protocol_record(state, row)
        self.assertEqual(state, before)
        row["properties"].pop()
        apply_protocol_record(state, row)
        shared = _external_machine(state)
        self.assertTrue(validate_protocol_state(shared["protocol"], shared["heap"], shared))
        state["heap"]["array"]["properties"]["extra"] = ["num", 9]
        shared = _external_machine(state)
        with self.assertRaisesRegex(ProtocolV8Error, "property bound"):
            validate_protocol_state(shared["protocol"], shared["heap"], shared)

    def test_active_symbol_and_protocol_completion_guards_are_exercised_directly(self):
        state = machine()
        allocate(state, "callee", "closure")
        shared = _external_machine(state)
        with self.assertRaisesRegex(ProtocolV8Error, "Symbol data must use protocol"):
            assert_detailed_record_compatible(
                shared, "bind", {"name": "s", "value": ["symbol", "o7"]}
            )
        apply_protocol_record(
            state,
            {
                "kind": "call",
                "child": "child",
                "callee": ["ref", "callee"],
                "receiver": ["undefined"],
                "arguments": [],
            },
        )
        for kind in ("return", "throw"):
            with self.assertRaisesRegex(ProtocolV8Error, "Protocol-owned invocation"):
                assert_detailed_record_compatible(
                    _external_machine(state), kind, {"invocation": "child", "value": ["num", 7]}
                )
            with self.assertRaisesRegex(ProtocolV8Error, "name the active"):
                assert_detailed_record_compatible(
                    _external_machine(state), kind, {"invocation": "root", "value": ["num", 7]}
                )
        state["stack"].append("detail-child")
        state["frameFunctions"]["detail-child"] = "detail"
        self.assertTrue(
            assert_detailed_record_compatible(
                _external_machine(state),
                "return",
                {"invocation": "detail-child", "value": ["num", 7]},
            )
        )

    def test_allocation_paths_reject_reserved_or_existing_identities_before_writes(self):
        state = machine()
        allocate(state, "existing")
        apply_protocol_record(
            state, {"kind": "symbol.create", "id": "o7", "description": ["undefined"]}
        )
        state["protocol"]["construction"] = {"privateNames": [{"id": "private"}]}
        for identity in ("existing", "o7", "private"):
            shared = _external_machine(state)
            before = copy.deepcopy(shared)
            with self.assertRaisesRegex(NativeV8Error, "heap identity"):
                _apply_detail(
                    shared, "allocate", {"name": identity, "kind": "object", "properties": []}, {}
                )
            self.assertEqual(shared, before)
            line = "algorithm " + json.dumps(
                {
                    "allocations": [
                        {"id": "fresh", "kind": "object", "properties": []},
                        {"id": identity, "kind": "object", "properties": []},
                    ],
                    "completion": {"kind": "return", "value": ["num", 7]},
                }
            )
            with self.assertRaisesRegex(NativeV8Error, "heap identity"):
                _apply_algorithm(shared, {}, {}, line)
            self.assertEqual(shared, before)
            for kind, fields in (
                ("promise", {"action": "create", "promise": identity}),
                (
                    "request",
                    {
                        "operation": "demo.number",
                        "invocation": "root",
                        "effect": "e1",
                        "arguments": [],
                        "payload": "[]",
                        "promise": identity,
                        "site": {"start": 0, "end": 1},
                    },
                ),
            ):
                with (
                    self.subTest(kind=kind),
                    self.assertRaisesRegex(NativeV8Error, "heap identity"),
                ):
                    _apply_detail(shared, kind, fields, {})
                self.assertEqual(shared, before)

    def test_detailed_writes_preserve_key_order_and_capacity_boundaries(self):
        state = machine()
        allocate(state, "object")
        node = state["heap"]["object"]
        node["keyOrder"] = ["x"]
        node["properties"] = {"x": ["num", 1]}
        fields = {"object": "object", "key": "x", "present": False, "value": ["undefined"]}
        with self.assertRaisesRegex(ProtocolV8Error, "bypasses rich"):
            assert_detailed_record_compatible(_external_machine(state), "write", fields)
        node.pop("keyOrder")
        node["properties"] = {f"k{i}": ["num", i] for i in range(4096)}
        shared = _external_machine(state)
        before = copy.deepcopy(shared)
        with self.assertRaisesRegex(NativeV8Error, "property bound"):
            _apply_detail(
                shared,
                "write",
                {"object": "object", "key": "extra", "present": True, "value": ["num", 1]},
                {},
            )
        self.assertEqual(shared, before)

    def test_empty_descriptor_metadata_survives_deletion_and_checkpoint(self):
        state = machine()
        allocate(state, "object")
        apply_protocol_record(
            state,
            {
                "kind": "property.define",
                "object": "object",
                "key": ["str", "x"],
                "descriptor": {
                    "kind": "data",
                    "value": ["num", 1],
                    "writable": True,
                    "enumerable": True,
                    "configurable": True,
                },
                "success": True,
            },
        )
        fields = {"object": "object", "key": "x", "present": True, "value": ["num", 1]}
        shared = _external_machine(state)
        self.assertTrue(assert_detailed_record_compatible(shared, "read", fields))
        with self.assertRaisesRegex(ProtocolV8Error, "bypasses rich"):
            assert_detailed_record_compatible(shared, "write", fields)

        apply_protocol_record(
            state,
            {"kind": "property.delete", "object": "object", "key": ["str", "x"], "success": True},
        )
        shared = _external_machine(state)
        node = dict(shared["heap"])["object"]
        self.assertEqual(node["descriptors"], [])
        self.assertEqual(node["keyOrder"], [])
        self.assertIs(node["extensible"], True)
        self.assertEqual(_external_machine(_internal_machine(shared)), shared)
        with self.assertRaisesRegex(ProtocolV8Error, "bypasses rich"):
            assert_detailed_record_compatible(shared, "write", fields)

    def test_existing_property_update_preserves_enumeration_order(self):
        state = machine()
        allocate(state, "object")
        state["heap"]["object"]["properties"] = {"a": ["num", 1], "b": ["num", 2]}
        apply_protocol_record(
            state,
            {
                "kind": "property.set",
                "object": "object",
                "receiver": ["ref", "object"],
                "key": ["str", "a"],
                "value": ["num", 3],
                "success": True,
            },
        )
        self.assertEqual(list(state["heap"]["object"]["properties"]), ["b", "a"])
        self.assertEqual(state["heap"]["object"]["keyOrder"], ["a", "b"])
        apply_protocol_record(
            state,
            {
                "kind": "property.keys",
                "object": "object",
                "mode": "own",
                "name": "keys",
                "keys": [["str", "a"], ["str", "b"]],
            },
        )

    def test_detail_array_allocation_counts_implicit_length(self):
        shared = _external_machine(machine())
        before = copy.deepcopy(shared)
        fields = {
            "name": "array",
            "kind": "array",
            "length": 4096,
            "properties": [{"key": str(i), "value": ["num", i]} for i in range(4096)],
        }
        with self.assertRaisesRegex(NativeV8Error, "property bound"):
            _apply_detail(shared, "allocate", fields, {})
        self.assertEqual(shared, before)


if __name__ == "__main__":
    unittest.main()
