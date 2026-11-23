"""Own and inherited existence are checked without reading accessor values."""

import copy
import json
import shutil
import subprocess
import unittest

from megatoken.data.protocol_v8 import (
    ProtocolV8Error,
    _external_machine,
    _internal_machine,
    apply_protocol_record,
    decode_protocol_record,
    encode_protocol_record,
    validate_protocol_state,
)
from tests.test_enumeration_v8 import define
from tests.test_number_v8 import allocate, machine


def has(state, identity, key, expected):
    record = {
        "kind": "property.has",
        "object": identity,
        "key": key,
        "result": ["bool", expected],
    }
    apply_protocol_record(state, decode_protocol_record(encode_protocol_record(record)))


def prototype(state, identity, parent):
    apply_protocol_record(
        state,
        {"kind": "property.prototype", "object": identity, "prototype": parent, "success": True},
    )


def delete(state, identity, key):
    apply_protocol_record(
        state,
        {"kind": "property.delete", "object": identity, "key": key, "success": True},
    )


class PropertyPresenceV8Tests(unittest.TestCase):
    def test_inherited_accessors_nonenumerable_and_symbol_properties_do_not_call_getters(self):
        state = machine()
        allocate(state, "parent")
        allocate(state, "child")
        allocate(state, "getter", "closure")
        prototype(state, "child", ["ref", "parent"])
        define(state, "child", ["str", "own"], ["undefined"])
        define(state, "parent", ["str", "hidden"], ["num", 7], enumerable=False)
        define(state, "parent", ["symbol", "well-known:iterator"], ["undefined"])
        apply_protocol_record(
            state,
            {
                "kind": "property.define",
                "object": "parent",
                "key": ["str", "accessor"],
                "descriptor": {
                    "kind": "accessor",
                    "get": ["ref", "getter"],
                    "set": ["undefined"],
                    "enumerable": True,
                    "configurable": True,
                },
                "success": True,
            },
        )
        before = copy.deepcopy(state)
        for key in (
            ["str", "own"],
            ["str", "hidden"],
            ["str", "accessor"],
            ["symbol", "well-known:iterator"],
        ):
            has(state, "child", key, True)
        has(state, "child", ["str", "missing"], False)
        self.assertEqual(state["heap"], before["heap"])
        self.assertEqual(state["locals"], before["locals"])
        self.assertEqual(state["stack"], ["root"])
        self.assertEqual(state["protocol"]["pendingCalls"], [])
        self.assertEqual(state["protocol"]["callFrames"], [])

    def test_presence_follows_live_deletion_and_restored_prototypes(self):
        state = machine()
        allocate(state, "parent")
        allocate(state, "child")
        prototype(state, "child", ["ref", "parent"])
        define(state, "child", ["str", "key"], ["num", 1])
        define(state, "parent", ["str", "key"], ["num", 2])
        delete(state, "child", ["str", "key"])
        external = json.loads(json.dumps(_external_machine(state)))
        self.assertTrue(validate_protocol_state(external["protocol"], external["heap"], external))
        restored = _internal_machine(external)
        has(restored, "child", ["str", "key"], True)
        delete(restored, "parent", ["str", "key"])
        has(restored, "child", ["str", "key"], False)
        prototype(restored, "child", None)
        has(restored, "child", ["str", "toString"], False)

    def test_implicit_intrinsic_existence_and_mutated_explicit_intrinsic_state(self):
        state = machine()
        allocate(state, "object")
        for key in ("toString", "constructor", "__proto__", "hasOwnProperty"):
            has(state, "object", ["str", key], True)
        has(state, "object", ["str", "notAnIntrinsic"], False)
        apply_protocol_record(state, {"kind": "promise.intrinsics"})
        delete(state, "intrinsicobjectprototype", ["str", "toString"])
        has(state, "object", ["str", "toString"], False)
        define(state, "intrinsicobjectprototype", ["str", "added"], ["num", 4])
        has(state, "object", ["str", "added"], True)

    def test_sparse_arrays_distinguish_absence_from_undefined(self):
        state = machine()
        apply_protocol_record(
            state,
            {
                "kind": "allocate",
                "name": "array",
                "objectKind": "array",
                "length": 3,
                "properties": [{"key": ["str", "1"], "value": ["undefined"]}],
                "code": None,
                "captures": [],
                "functionKind": None,
                "lexicalReceiver": None,
            },
        )
        for key, expected in (("0", False), ("1", True), ("2", False), ("length", True)):
            has(state, "array", ["str", key], expected)
        has(state, "array", ["str", "map"], True)
        has(state, "array", ["symbol", "well-known:iterator"], True)
        prototype(state, "array", None)
        has(state, "array", ["str", "map"], False)
        has(state, "array", ["symbol", "well-known:iterator"], False)

    def test_false_results_unknown_keys_receivers_and_cycles_are_rejected_atomically(self):
        state = machine()
        allocate(state, "object")
        define(state, "object", ["str", "key"], ["num", 1])
        correct = {
            "kind": "property.has",
            "object": "object",
            "key": ["str", "key"],
            "result": ["bool", True],
        }
        for fields in (
            {"result": ["bool", False]},
            {"result": ["num", 1]},
            {"object": "absent"},
            {"key": ["symbol", "undeclared"]},
            {"key": ["num", 1]},
            {"receiver": ["ref", "object"]},
        ):
            with self.subTest(fields=fields):
                before = copy.deepcopy(state)
                with self.assertRaises(ProtocolV8Error):
                    apply_protocol_record(state, {**correct, **fields})
                self.assertEqual(state, before)
        state["heap"]["object"]["prototype"] = ["ref", "object"]
        before = copy.deepcopy(state)
        with self.assertRaisesRegex(ProtocolV8Error, "cycle"):
            has(state, "object", ["str", "missing"], False)
        self.assertEqual(state, before)

    def test_unrepresented_callable_prototype_is_an_explicit_refusal(self):
        state = machine()
        allocate(state, "function", "closure")
        for expected in (True, False):
            with self.assertRaisesRegex(ProtocolV8Error, "intrinsic"):
                has(state, "function", ["str", "prototype"], expected)

    def test_implicit_promise_prototype_requires_its_owned_intrinsic_state(self):
        from megatoken.data.promise_scheduler_v8 import create_promise

        state = machine()
        create_promise(state, "promise")
        for key in ("then", "catch", "finally", "missing"):
            for expected in (True, False):
                with self.assertRaisesRegex(ProtocolV8Error, "initialized intrinsic prototype"):
                    has(state, "promise", ["str", key], expected)
        apply_protocol_record(state, {"kind": "promise.intrinsics"})
        for key in ("then", "catch", "finally"):
            has(state, "promise", ["str", key], True)
        has(state, "promise", ["symbol", "well-known:toStringTag"], True)
        has(state, "promise", ["str", "missing"], False)

    @unittest.skipUnless(shutil.which("node"), "native JavaScript runtime is not available")
    def test_native_has_property_agrees_without_observable_getter_calls(self):
        script = """
            let reads=0;
            const key=Symbol.iterator;
            const parent=Object.create(null, {
                inherited: {get(){reads++;throw new Error('getter called');}},
                hidden: {value:1},
                [key]: {value:undefined}
            });
            const object=Object.create(parent);
            object.own=undefined;
            process.stdout.write(JSON.stringify([
                'own' in object, 'inherited' in object, 'hidden' in object,
                key in object, 'missing' in object, reads
            ]));
        """
        actual = json.loads(subprocess.check_output(["node", "-e", script], text=True))
        self.assertEqual(actual, [True, True, True, True, False, 0])
