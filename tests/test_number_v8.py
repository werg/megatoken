import copy
import json
import shutil
import subprocess
import unittest

from megatoken.data import number_v8 as number
from megatoken.data.protocol_v8 import ProtocolV8Error, apply_protocol_record, create_protocol_state


def machine():
    return {
        "heap": {},
        "locals": {},
        "captureAliases": {},
        "receiver": ["undefined"],
        "stack": ["root"],
        "frameFunctions": {"root": "run"},
        "pending": None,
        "lastReturn": None,
        "completion": None,
        "protocol": create_protocol_state(),
    }


def apply(state, record):
    staged = copy.deepcopy(state)
    number.assert_next(staged, record)
    number.apply_record(staged, record)
    state.clear()
    state.update(staged)


def allocate(state, identity, kind="object"):
    apply_protocol_record(
        state,
        {
            "kind": "allocate",
            "name": identity,
            "objectKind": kind,
            "code": identity if kind == "closure" else None,
            "captures": [],
            "functionKind": "ordinary" if kind == "closure" else None,
            "lexicalReceiver": None,
            "length": None,
            "properties": [],
        },
    )


def define(state, identity, key, value):
    apply_protocol_record(
        state,
        {
            "kind": "property.define",
            "object": identity,
            "key": key,
            "descriptor": {
                "kind": "data",
                "value": value,
                "writable": True,
                "enumerable": True,
                "configurable": True,
            },
            "success": True,
        },
    )


class NumberProtocolTests(unittest.TestCase):
    def test_retained_number_protocol_cannot_invent_function_prototype_stringification(self):
        state = machine()
        allocate(state, "prototype", "closure")
        allocate(state, "input")
        apply_protocol_record(
            state,
            {
                "kind": "property.prototype",
                "object": "input",
                "prototype": ["ref", "prototype"],
                "success": True,
            },
        )
        apply(state, {"kind": "number.begin", "id": "n", "value": ["ref", "input"]})
        apply(
            state,
            {
                "kind": "number.select",
                "id": "n",
                "key": ["symbol", "well-known:toPrimitive"],
                "method": ["undefined"],
                "builtin": None,
            },
        )
        apply(
            state,
            {
                "kind": "number.select",
                "id": "n",
                "key": ["str", "valueOf"],
                "method": ["undefined"],
                "builtin": "Object.valueOf",
            },
        )
        apply(state, {"kind": "number.builtin", "id": "n", "result": ["ref", "input"]})
        before = copy.deepcopy(state)
        with self.assertRaisesRegex(ProtocolV8Error, "function intrinsic prototype"):
            apply(
                state,
                {
                    "kind": "number.select",
                    "id": "n",
                    "key": ["str", "toString"],
                    "method": ["undefined"],
                    "builtin": "Object.toString",
                },
            )
        self.assertEqual(state, before)

    def test_cleanup_scope_transfer_is_bounded_and_cannot_skip_number_conversion(self):
        from megatoken.data.protocol_v8 import (
            _external_machine,
            assert_detailed_record_compatible,
            validate_completion_detail,
        )

        state = machine()
        apply(state, {"kind": "number.begin", "id": "n", "value": ["num", 3]})
        public = _external_machine(state)
        fields = {
            "kind": "cleanup",
            "value": ["undefined"],
            "handler": {
                "schema": "semantic-handler-scopes/1",
                "frames": [
                    {
                        "invocation": "root",
                        "handlers": [
                            {
                                "id": "handler1",
                                "catch": True,
                                "finally": True,
                                "phase": "finally",
                                "caught": None,
                                "pending": {"kind": "throw", "value": ["num", 7]},
                            }
                        ],
                    }
                ],
            },
        }
        self.assertTrue(validate_completion_detail(public, "transfer", fields))
        self.assertTrue(assert_detailed_record_compatible(public, "transfer", fields))
        self.assertEqual(
            state["protocol"]["numberConversions"]["operations"][0]["phase"], "convert"
        )
        with self.assertRaisesRegex(ProtocolV8Error, "number.finish"):
            assert_detailed_record_compatible(
                public, "transfer", {"kind": "return", "value": ["num", 3], "handler": "finally"}
            )
        mutations = [
            lambda f: f["handler"]["frames"][0].update(invocation="absent"),
            lambda f: f["handler"]["frames"].append(copy.deepcopy(f["handler"]["frames"][0])),
            lambda f: f["handler"]["frames"][0]["handlers"][0].update(pending=None),
            lambda f: f["handler"]["frames"][0]["handlers"][0].update(caught=["num", 7]),
            lambda f: f["handler"]["frames"][0]["handlers"][0]["pending"].update(
                value=["ref", "absent"]
            ),
            lambda f: f["handler"]["frames"][0]["handlers"][0].update(bytecode=[]),
        ]
        before = copy.deepcopy(public)
        for mutate in mutations:
            malformed = copy.deepcopy(fields)
            mutate(malformed)
            with self.assertRaises(ProtocolV8Error):
                validate_completion_detail(public, "transfer", malformed)
            self.assertEqual(public, before)
        apply(state, {"kind": "number.finish", "id": "n", "result": ["num", 3]})

    def test_abrupt_callback_is_terminal_exact_and_restorable(self):
        state = machine()
        allocate(state, "object")
        allocate(state, "method", "closure")
        define(state, "object", ["symbol", "well-known:toPrimitive"], ["ref", "method"])
        apply(state, {"kind": "number.begin", "id": "n", "value": ["ref", "object"]})
        apply(
            state,
            {
                "kind": "number.select",
                "id": "n",
                "key": ["symbol", "well-known:toPrimitive"],
                "method": ["ref", "method"],
                "builtin": None,
            },
        )
        apply(state, {"kind": "number.call", "id": "n", "child": "callback"})
        before = copy.deepcopy(state)
        with self.assertRaisesRegex(ProtocolV8Error, "outside caller"):
            apply(state, {"kind": "number.throw", "id": "n", "value": ["num", 7]})
        self.assertEqual(state, before)
        apply_protocol_record(
            state, {"kind": "throw", "invocation": "callback", "value": ["num", 7]}
        )
        before = copy.deepcopy(state)
        with self.assertRaisesRegex(ProtocolV8Error, "number.throw"):
            apply(state, {"kind": "number.return", "id": "n", "value": ["num", 7]})
        with self.assertRaisesRegex(ProtocolV8Error, "abrupt completion differs"):
            apply(state, {"kind": "number.throw", "id": "n", "value": ["num", 99]})
        self.assertEqual(state, before)
        apply(state, {"kind": "number.throw", "id": "n", "value": ["num", 7]})
        self.assertEqual(state["lastReturn"], {"kind": "throw", "value": ["num", 7]})
        self.assertEqual(state["protocol"]["numberConversions"]["operations"][0]["phase"], "abrupt")
        self.assertTrue(number.validate_state(state))
        before = copy.deepcopy(state)
        with self.assertRaisesRegex(ProtocolV8Error, "primitive conversion phase"):
            apply(state, {"kind": "number.finish", "id": "n", "result": ["num", 7]})
        with self.assertRaisesRegex(ProtocolV8Error, "method order"):
            apply(
                state,
                {
                    "kind": "number.select",
                    "id": "n",
                    "key": ["str", "toString"],
                    "method": ["undefined"],
                    "builtin": "Object.toString",
                },
            )
        self.assertEqual(state, before)
        for field, bad in (("primitive", ["num", 7]), ("child", None), ("result", None)):
            malformed = copy.deepcopy(state)
            malformed["protocol"]["numberConversions"]["operations"][0][field] = bad
            with self.assertRaises(ProtocolV8Error):
                number.validate_state(malformed)

    def test_generator_conversion_callback_is_refused_before_entering_its_body(self):
        state = machine()
        allocate(state, "object")
        allocate(state, "method", "closure")
        apply_protocol_record(
            state,
            {
                "kind": "generator.function",
                "object": "method",
                "name": "method",
                "length": 0,
                "prototype": "generator-prototype",
            },
        )
        define(state, "object", ["symbol", "well-known:toPrimitive"], ["ref", "method"])
        apply(state, {"kind": "number.begin", "id": "n", "value": ["ref", "object"]})
        apply(
            state,
            {
                "kind": "number.select",
                "id": "n",
                "key": ["symbol", "well-known:toPrimitive"],
                "method": ["ref", "method"],
                "builtin": None,
            },
        )
        before = copy.deepcopy(state)
        with self.assertRaisesRegex(ProtocolV8Error, "generator callback"):
            apply(state, {"kind": "number.call", "id": "n", "child": "callback"})
        self.assertEqual(state, before)

    @unittest.skipUnless(shutil.which("node"), "Node is the independent Number oracle")
    def test_scalar_conversion_matches_actual_javascript_including_unicode_and_rounding(self):
        values = [
            ["undefined"],
            ["null"],
            ["bool", False],
            ["bool", True],
            ["num", "-0"],
            ["bigint", "9007199254740993"],
            ["bigint", "9" * 10000],
        ]
        values += [
            ["str", text]
            for text in [
                "",
                " ",
                "0x10",
                "0b11",
                "0o17",
                "+0x10",
                "-0",
                "+Infinity",
                "-Infinity",
                "NaN",
                "bad",
                "1e9999",
                "1_0",
                "123.5",
                "\ufeff7\ufeff",
                "\u00857",
                "\u001c7",
                "\u180e7",
                "\u200b7",
                "\u00a07\u2029",
            ]
        ]
        program = """const fs=require('fs'); const input=JSON.parse(fs.readFileSync(0,'utf8'));
        console.log(JSON.stringify(input.map(v=>{const x=v[0]==='undefined'?undefined:v[0]==='null'?null:
          v[0]==='bigint'?BigInt(v[1]):v[0]==='num'&&v[1]==='-0'?-0:v[1];const n=Number(x);
          return ['num',Object.is(n,-0)?'-0':Number.isNaN(n)?'NaN':n===Infinity?'+Infinity':n===-Infinity?'-Infinity':n];})));"""
        actual = json.loads(
            subprocess.check_output(
                [shutil.which("node"), "-e", program], input=json.dumps(values).encode(), timeout=20
            )
        )
        for value, expected in zip(values, actual, strict=True):
            with self.subTest(value=value if len(str(value)) < 100 else value[0]):
                state = machine()
                apply(state, {"kind": "number.begin", "id": "n", "value": value})
                apply(state, {"kind": "number.finish", "id": "n", "result": expected})
                self.assertEqual(state["lastReturn"], {"kind": "return", "value": expected})
                self.assertTrue(number.validate_state(state))

    def test_method_selection_preserves_receiver_hint_and_live_fallback(self):
        state = machine()
        allocate(state, "object")
        allocate(state, "method", "closure")
        define(state, "object", ["symbol", "well-known:toPrimitive"], ["ref", "method"])
        apply(state, {"kind": "number.begin", "id": "n", "value": ["ref", "object"]})
        apply(
            state,
            {
                "kind": "number.select",
                "id": "n",
                "key": ["symbol", "well-known:toPrimitive"],
                "method": ["ref", "method"],
                "builtin": None,
            },
        )
        apply(state, {"kind": "number.call", "id": "n", "child": "callback"})
        self.assertEqual(state["receiver"], ["ref", "object"])
        self.assertEqual(state["protocol"]["callFrames"][-1]["arguments"], [["str", "number"]])
        self.assertTrue(number.validate_state(state))
        apply_protocol_record(
            state, {"kind": "return", "invocation": "callback", "value": ["str", "12"]}
        )
        apply(state, {"kind": "number.return", "id": "n", "value": ["str", "12"]})
        apply(state, {"kind": "number.finish", "id": "n", "result": ["num", 12]})
        self.assertEqual(state["receiver"], ["undefined"])

    def test_ordinary_fallback_skips_noncallable_method_and_checks_intrinsics(self):
        state = machine()
        allocate(state, "object")
        define(state, "object", ["str", "valueOf"], ["num", 1])
        define(state, "object", ["symbol", "well-known:toStringTag"], ["str", "Tagged"])
        apply(state, {"kind": "number.begin", "id": "n", "value": ["ref", "object"]})
        apply(
            state,
            {
                "kind": "number.select",
                "id": "n",
                "key": ["symbol", "well-known:toPrimitive"],
                "method": ["undefined"],
                "builtin": None,
            },
        )
        apply(
            state,
            {
                "kind": "number.select",
                "id": "n",
                "key": ["str", "valueOf"],
                "method": ["num", 1],
                "builtin": None,
            },
        )
        apply(
            state,
            {
                "kind": "number.select",
                "id": "n",
                "key": ["str", "toString"],
                "method": ["undefined"],
                "builtin": "Object.toString",
            },
        )
        apply(state, {"kind": "number.builtin", "id": "n", "result": ["str", "[object Tagged]"]})
        apply(state, {"kind": "number.finish", "id": "n", "result": ["num", "NaN"]})
        self.assertTrue(number.validate_state(state))

    def test_wrong_order_false_claims_and_corrupt_retained_state_fail_atomically(self):
        state = machine()
        allocate(state, "object")
        apply(state, {"kind": "number.begin", "id": "n", "value": ["ref", "object"]})
        before = copy.deepcopy(state)
        with self.assertRaises(ProtocolV8Error):
            apply(
                state,
                {
                    "kind": "number.select",
                    "id": "n",
                    "key": ["str", "toString"],
                    "method": ["undefined"],
                    "builtin": "Object.toString",
                },
            )
        self.assertEqual(state, before)
        with self.assertRaises(ProtocolV8Error):
            number.assert_next(state, {"kind": "end"})
        scalar_state = machine()
        apply(scalar_state, {"kind": "number.begin", "id": "n", "value": ["str", "3"]})
        before = copy.deepcopy(scalar_state)
        with self.assertRaises(ProtocolV8Error):
            apply(scalar_state, {"kind": "number.finish", "id": "n", "result": ["num", 4]})
        self.assertEqual(scalar_state, before)
        for change in (
            {"parent": "missing"},
            {"primitive": None},
            {"phase": "complete", "result": ["num", 4]},
            {"extra": True},
        ):
            altered = copy.deepcopy(scalar_state)
            altered["protocol"]["numberConversions"]["operations"][0].update(change)
            with self.assertRaises(ProtocolV8Error):
                number.validate_state(altered)

    def test_rejected_direct_entry_does_not_initialize_retained_state(self):
        for value in (["ref", "missing"], ["str", "3" * 10001]):
            state = machine()
            before = copy.deepcopy(state)
            with self.assertRaises(ProtocolV8Error):
                number.apply_record(state, {"kind": "number.begin", "id": "n", "value": value})
            self.assertEqual(state, before)
        state = machine()
        state["stack"] = []
        before = copy.deepcopy(state)
        with self.assertRaisesRegex(ProtocolV8Error, "active invocation"):
            number.apply_record(state, {"kind": "number.begin", "id": "n", "value": ["num", 3]})
        self.assertEqual(state, before)
        state = machine()
        number.apply_record(
            state, {"kind": "number.begin", "id": "n", "value": ["str", "3" * 10000]}
        )
        self.assertTrue(number.validate_state(state))

    def test_oversized_callback_primitive_is_rejected_before_retention(self):
        state = machine()
        allocate(state, "object")
        allocate(state, "method", "closure")
        define(state, "object", ["symbol", "well-known:toPrimitive"], ["ref", "method"])
        for record in (
            {"kind": "number.begin", "id": "n", "value": ["ref", "object"]},
            {
                "kind": "number.select",
                "id": "n",
                "key": ["symbol", "well-known:toPrimitive"],
                "method": ["ref", "method"],
                "builtin": None,
            },
            {"kind": "number.call", "id": "n", "child": "convert"},
            {"kind": "return", "invocation": "convert", "value": ["str", "3" * 10001]},
        ):
            apply_protocol_record(state, record)
        before = copy.deepcopy(state)
        with self.assertRaisesRegex(ProtocolV8Error, "scalar work bound"):
            number.apply_record(
                state, {"kind": "number.return", "id": "n", "value": ["str", "3" * 10001]}
            )
        self.assertEqual(state, before)

    def test_noncallable_exotic_valueof_falls_back_to_tostring(self):
        state = machine()
        allocate(state, "object")
        allocate(state, "method", "closure")
        state["heap"]["promise"] = {
            "kind": "promise",
            "properties": {},
            "state": "fulfilled",
            "result": ["num", 1],
            "reactions": [],
        }
        define(state, "object", ["str", "valueOf"], ["ref", "promise"])
        define(state, "object", ["str", "toString"], ["ref", "method"])
        for record in (
            {"kind": "number.begin", "id": "n", "value": ["ref", "object"]},
            {
                "kind": "number.select",
                "id": "n",
                "key": ["symbol", "well-known:toPrimitive"],
                "method": ["undefined"],
                "builtin": None,
            },
            {
                "kind": "number.select",
                "id": "n",
                "key": ["str", "valueOf"],
                "method": ["ref", "promise"],
                "builtin": None,
            },
            {
                "kind": "number.select",
                "id": "n",
                "key": ["str", "toString"],
                "method": ["ref", "method"],
                "builtin": None,
            },
            {"kind": "number.call", "id": "n", "child": "convert"},
            {"kind": "return", "invocation": "convert", "value": ["str", "3"]},
            {"kind": "number.return", "id": "n", "value": ["str", "3"]},
            {"kind": "number.finish", "id": "n", "result": ["num", 3]},
        ):
            apply_protocol_record(state, record)
        self.assertEqual(state["lastReturn"], {"kind": "return", "value": ["num", 3]})
        self.assertTrue(number.validate_state(state))

    def test_unfinished_conversion_requires_a_live_parent_after_restore(self):
        state = machine()
        apply_protocol_record(state, {"kind": "number.begin", "id": "n", "value": ["str", "3"]})
        altered = copy.deepcopy(state)
        altered["stack"] = []
        with self.assertRaisesRegex(ProtocolV8Error, "live parent"):
            number.validate_state(altered)
        apply_protocol_record(state, {"kind": "number.finish", "id": "n", "result": ["num", 3]})
        state["stack"] = []
        self.assertTrue(number.validate_state(state))

    def test_noncallable_toprimitive_rejection_preserves_selected_state(self):
        state = machine()
        allocate(state, "object")
        define(state, "object", ["symbol", "well-known:toPrimitive"], ["num", 1])
        apply_protocol_record(
            state, {"kind": "number.begin", "id": "n", "value": ["ref", "object"]}
        )
        before = copy.deepcopy(state)
        with self.assertRaisesRegex(ProtocolV8Error, "noncallable method"):
            number.apply_record(
                state,
                {
                    "kind": "number.select",
                    "id": "n",
                    "key": ["symbol", "well-known:toPrimitive"],
                    "method": ["num", 1],
                    "builtin": None,
                },
            )
        self.assertEqual(state, before)


if __name__ == "__main__":
    unittest.main()
