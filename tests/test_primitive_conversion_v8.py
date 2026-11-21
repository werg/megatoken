import copy
import json
import shutil
import subprocess
import unittest

from megatoken.data import primitive_conversion_v8 as conversion
from megatoken.data.protocol_v8 import (
    ProtocolV8Error,
    _external_machine,
    _internal_machine,
    apply_protocol_record,
    assert_detailed_record_compatible,
    create_protocol_state,
    decode_protocol_record,
    encode_protocol_record,
    validate_protocol_state,
)

UNDEFINED = ["undefined"]
EXOTIC = ["symbol", "well-known:toPrimitive"]


def machine():
    return {
        "heap": {},
        "locals": {},
        "captureAliases": {},
        "receiver": UNDEFINED,
        "stack": ["root"],
        "frameFunctions": {"root": "run"},
        "pending": None,
        "lastReturn": None,
        "completion": None,
        "protocol": create_protocol_state(),
    }


def allocate(identity, properties=(), *, kind="object", length=None, **extra):
    return {
        "kind": "allocate",
        "name": identity,
        "objectKind": kind,
        "length": length,
        "code": identity if kind == "closure" else None,
        "captures": [],
        "functionKind": "ordinary" if kind == "closure" else None,
        "lexicalReceiver": None,
        "properties": [{"key": key, "value": value} for key, value in properties],
        **extra,
    }


def define(identity, key, value=None, *, getter=None):
    descriptor = (
        {"kind": "accessor", "get": getter, "set": UNDEFINED}
        if getter is not None
        else {"kind": "data", "value": value, "writable": True}
    )
    return {
        "kind": "property.define",
        "object": identity,
        "key": key,
        "descriptor": {**descriptor, "enumerable": True, "configurable": True},
        "success": True,
    }


def begin(identity, goal, value):
    return {"kind": "conversion.begin", "id": identity, "goal": goal, "value": value}


def lookup(identity, key, descriptor="missing", value=UNDEFINED, builtin=None):
    return {
        "kind": "conversion.lookup",
        "id": identity,
        "key": key,
        "descriptor": descriptor,
        "value": value,
        "builtin": builtin,
    }


def finish(identity, result):
    return {"kind": "conversion.finish", "id": identity, "result": result}


def apply(state, *records):
    for record in records:
        # Exercise the public codec/registry, not the component entrypoint alone.
        apply_protocol_record(state, decode_protocol_record(encode_protocol_record(record)))
        snapshot = json.loads(json.dumps(_external_machine(state)))
        restored = _internal_machine(snapshot)
        conversion.validate_state(restored)
        validate_protocol_state(restored["protocol"], restored["heap"], snapshot)
        state.clear()
        state.update(restored)
    return state


def returned(state, identity, child, value, *, getter=False, thrown=False):
    apply(
        state,
        {"kind": "throw" if thrown else "return", "invocation": child, "value": value},
        {
            "kind": "conversion.throw"
            if thrown
            else "conversion.getter.return"
            if getter
            else "conversion.return",
            "id": identity,
            "value": value,
        },
    )


class PrimitiveConversionTests(unittest.TestCase):
    def refuse(self, state, record, message=None):
        before = copy.deepcopy(state)
        with self.assertRaisesRegex((ValueError, TypeError), message or ".*"):
            apply_protocol_record(state, record)
        self.assertEqual(state, before)

    def ordinary(self, state, identity="key", goal="PropertyKey"):
        apply(state, begin(identity, goal, ["ref", "input"]), lookup(identity, EXOTIC))

    def test_registered_scalar_leaves_symbols_and_truthful_faults(self):
        state = machine()
        apply(state, {"kind": "symbol.create", "id": "o1", "description": ["str", "key"]})
        values = [
            (UNDEFINED, ["num", "NaN"], ["str", "undefined"]),
            (["null"], ["num", 0], ["str", "null"]),
            (["bool", True], ["num", 1], ["str", "true"]),
            (["num", "-0"], ["num", "-0"], ["str", "0"]),
            (["str", " 2.5 "], ["num", 2.5], ["str", " 2.5 "]),
            (
                ["bigint", "9007199254740993"],
                ["num", 9007199254740992],
                ["str", "9007199254740993"],
            ),
        ]
        for index, (value, number, key) in enumerate(values):
            for goal, result in (("Number", number), ("PropertyKey", key)):
                identity = f"scalar{index}{goal}"
                apply(state, begin(identity, goal, value))
                self.refuse(state, finish(identity, ["str", "invented"]), "false primitive scalar")
                apply(state, finish(identity, result))
        apply(
            state,
            begin("symbolKey", "PropertyKey", ["symbol", "o1"]),
            finish("symbolKey", ["symbol", "o1"]),
        )
        apply(state, begin("symbolNumber", "Number", ["symbol", "o1"]))
        self.refuse(state, finish("symbolNumber", ["num", 0]), "conversion.fault")
        apply(state, {"kind": "conversion.fault", "id": "symbolNumber", "error": "typeError"})
        self.assertEqual(state["lastReturn"], {"kind": "throw", "value": ["ref", "typeError"]})
        self.assertEqual(
            state["heap"]["typeError"]["properties"]["message"],
            ["str", conversion.FAULTS["symbol-to-number"]],
        )

    def test_goal_controls_hint_and_checkpoint_binds_receiver_and_callback(self):
        for goal in ("Number", "PropertyKey"):
            state = machine()
            apply(
                state,
                allocate("method", kind="closure"),
                allocate("input", [(EXOTIC, ["ref", "method"])]),
                begin("convert", goal, ["ref", "input"]),
            )
            self.refuse(
                state,
                lookup("convert", ["str", "valueOf"], "data", ["ref", "method"]),
                "method order",
            )
            apply(
                state,
                lookup("convert", EXOTIC, "data", ["ref", "method"]),
                {"kind": "conversion.call", "id": "convert", "child": "child"},
            )
            self.assertEqual(state["receiver"], ["ref", "input"])
            self.assertEqual(
                state["protocol"]["callFrames"][-1]["arguments"],
                [["str", "number" if goal == "Number" else "string"]],
            )
            for field, value in (
                ("goal", "PropertyKey" if goal == "Number" else "Number"),
                ("index", 0),
                ("index", True),
                ("lookupKind", "missing"),
                ("child", "root"),
            ):
                malformed = copy.deepcopy(state)
                malformed["protocol"]["primitiveConversions"]["operations"][0][field] = value
                with self.subTest(goal=goal, field=field), self.assertRaises(ProtocolV8Error):
                    conversion.validate_state(malformed)
            for target, field, value in (
                ("locals", "child:this", UNDEFINED),
                ("frameFunctions", "child", "unrelated"),
            ):
                malformed = copy.deepcopy(state)
                malformed[target][field] = value
                with self.assertRaises(ProtocolV8Error):
                    conversion.validate_state(malformed)
            returned(state, "convert", "child", ["str", "7"])
            apply(state, finish("convert", ["num", 7] if goal == "Number" else ["str", "7"]))

    def test_getter_is_captured_once_and_method_uses_original_receiver(self):
        state = machine()
        apply(
            state,
            allocate("getter", kind="closure"),
            allocate("method", kind="closure"),
            allocate("replacement", kind="closure"),
            allocate("input"),
            define("input", ["str", "toString"], getter=["ref", "getter"]),
        )
        self.ordinary(state)
        apply(
            state,
            lookup("key", ["str", "toString"], "accessor", ["ref", "getter"]),
            {"kind": "conversion.getter.call", "id": "key", "child": "getterCall"},
            define("input", ["str", "toString"], ["ref", "replacement"]),
        )
        returned(state, "key", "getterCall", ["ref", "method"], getter=True)
        self.refuse(
            state,
            lookup("key", ["str", "toString"], "data", ["ref", "replacement"]),
            "conversion.call",
        )
        self.refuse(
            state,
            {"kind": "conversion.call", "id": "key", "child": "getterCall"},
            "duplicate invocation",
        )
        apply(state, {"kind": "conversion.call", "id": "key", "child": "methodCall"})
        self.assertEqual(state["frameFunctions"]["methodCall"], "method")
        self.assertEqual(state["receiver"], ["ref", "input"])
        returned(state, "key", "methodCall", ["str", "x"])
        apply(state, finish("key", ["str", "x"]))

    def test_fallback_rechecks_live_heap_after_first_method_mutates_it(self):
        state = machine()
        apply(
            state,
            *(allocate(name, kind="closure") for name in ("first", "old", "fresh")),
            allocate("nonprimitive"),
            allocate(
                "input",
                [(["str", "toString"], ["ref", "first"]), (["str", "valueOf"], ["ref", "old"])],
            ),
        )
        self.ordinary(state)
        apply(
            state,
            lookup("key", ["str", "toString"], "data", ["ref", "first"]),
            {"kind": "conversion.call", "id": "key", "child": "firstCall"},
            define("input", ["str", "valueOf"], ["ref", "fresh"]),
        )
        returned(state, "key", "firstCall", ["ref", "nonprimitive"])
        self.refuse(
            state,
            lookup("key", ["str", "valueOf"], "data", ["ref", "old"]),
            "false primitive conversion lookup",
        )
        apply(
            state,
            lookup("key", ["str", "valueOf"], "data", ["ref", "fresh"]),
            {"kind": "conversion.call", "id": "key", "child": "secondCall"},
        )
        returned(state, "key", "secondCall", ["str", "x"])
        apply(state, finish("key", ["str", "x"]))

    def test_nested_conversions_and_original_getter_method_throws(self):
        for getter in (True, False):
            state = machine()
            apply(
                state,
                allocate("method", kind="closure"),
                allocate("thrown"),
                allocate("input"),
                define(
                    "input",
                    ["str", "toString"],
                    ["ref", "method"],
                    getter=["ref", "method"] if getter else None,
                ),
            )
            self.ordinary(state, "outer")
            apply(
                state,
                lookup(
                    "outer",
                    ["str", "toString"],
                    "accessor" if getter else "data",
                    ["ref", "method"],
                ),
                {
                    "kind": "conversion.getter.call" if getter else "conversion.call",
                    "id": "outer",
                    "child": "child",
                },
                begin("inner", "Number", ["str", "7"]),
            )
            malformed = copy.deepcopy(state)
            malformed["protocol"]["primitiveConversions"]["operations"][1]["parent"] = "root"
            with self.assertRaisesRegex(ProtocolV8Error, "nested parent"):
                conversion.validate_state(malformed)
            apply(
                state,
                finish("inner", ["num", 7]),
                {"kind": "throw", "invocation": "child", "value": ["ref", "thrown"]},
            )
            self.refuse(
                state,
                {"kind": "conversion.throw", "id": "outer", "value": ["str", "replacement"]},
                "abrupt completion differs",
            )
            apply(state, {"kind": "conversion.throw", "id": "outer", "value": ["ref", "thrown"]})
            row = state["protocol"]["primitiveConversions"]["operations"][0]
            self.assertEqual(
                (row["phase"], row["child"], row["result"], row["fault"]),
                ("abrupt", "child", ["ref", "thrown"], None),
            )

    def test_intrinsic_fault_causes_are_derived_allocation_is_atomic_and_error_is_mutable(self):
        for cause in ("exotic-not-callable", "exotic-nonprimitive", "ordinary-exhausted"):
            state = machine()
            apply(
                state,
                allocate("method", kind="closure"),
                allocate("input"),
                allocate("nonprimitive"),
                allocate("occupied"),
            )
            if cause == "exotic-not-callable":
                apply(
                    state,
                    define("input", EXOTIC, ["num", 1]),
                    begin("convert", "Number", ["ref", "input"]),
                    lookup("convert", EXOTIC, "data", ["num", 1]),
                )
            elif cause == "exotic-nonprimitive":
                apply(
                    state,
                    define("input", EXOTIC, ["ref", "method"]),
                    begin("convert", "Number", ["ref", "input"]),
                    lookup("convert", EXOTIC, "data", ["ref", "method"]),
                    {"kind": "conversion.call", "id": "convert", "child": "child"},
                )
                returned(state, "convert", "child", ["ref", "nonprimitive"])
            else:
                apply(
                    state,
                    {
                        "kind": "property.prototype",
                        "object": "input",
                        "prototype": None,
                        "success": True,
                    },
                )
                self.ordinary(state, "convert")
                apply(
                    state,
                    lookup("convert", ["str", "toString"]),
                    lookup("convert", ["str", "valueOf"]),
                )
            self.assertEqual(
                state["protocol"]["primitiveConversions"]["operations"][0]["fault"], cause
            )
            self.refuse(
                state,
                {"kind": "conversion.fault", "id": "convert", "error": "occupied"},
                "heap identity",
            )
            self.refuse(
                state,
                {
                    "kind": "conversion.fault",
                    "id": "convert",
                    "error": "error",
                    "message": "invented",
                },
                "fields",
            )
            malformed = copy.deepcopy(state)
            malformed["protocol"]["primitiveConversions"]["operations"][0]["fault"] = (
                "symbol-to-number"
            )
            with self.assertRaises(ProtocolV8Error):
                conversion.validate_state(malformed)
            apply(state, {"kind": "conversion.fault", "id": "convert", "error": "error"})
            self.assertEqual(
                state["heap"]["error"]["properties"],
                {"name": ["str", "TypeError"], "message": ["str", conversion.FAULTS[cause]]},
            )
            apply(state, define("error", ["str", "message"], ["str", "changed by catch"]))
            self.assertTrue(conversion.validate_state(state))

    def test_fault_allocation_requires_a_valid_reference_before_mutation(self):
        state = machine()
        apply(state, begin("convert", "Number", ["symbol", "well-known:iterator"]))
        for name in (
            "bad/name",
            "bad name",
            "0error",
            "error\n",
            "error\r",
            "error\u2028",
            "error\u2029",
        ):
            record = {"kind": "conversion.fault", "id": "convert", "error": name}
            self.refuse(state, record, "value reference")
            with self.assertRaisesRegex((ValueError, TypeError), "value reference"):
                encode_protocol_record(record)
            self.assertTrue(conversion.validate_state(state))
        apply(state, {"kind": "conversion.fault", "id": "convert", "error": "typeError"})
        self.assertEqual(state["lastReturn"], {"kind": "throw", "value": ["ref", "typeError"]})

    def test_object_and_dense_array_builtins_are_derived(self):
        for name, props, options, expected in (
            ("Object.toString", [], {}, ["str", "[object Object]"]),
            (
                "Object.toString",
                [(["symbol", "well-known:toStringTag"], ["str", "Tagged"])],
                {},
                ["str", "[object Tagged]"],
            ),
            (
                "Array.toString",
                [(["str", "0"], ["num", 7]), (["str", "1"], ["null"])],
                {"kind": "array", "length": 2},
                ["str", "7,"],
            ),
        ):
            state = machine()
            apply(state, allocate("input", props, **options))
            self.ordinary(state)
            apply(state, lookup("key", ["str", "toString"], "builtin", builtin=name))
            self.refuse(
                state,
                {"kind": "conversion.builtin", "id": "key", "result": ["str", "fabricated"]},
                "false primitive conversion builtin",
            )
            apply(
                state,
                {"kind": "conversion.builtin", "id": "key", "result": expected},
                finish("key", expected),
            )

    def test_builtin_getter_join_sparse_and_symbol_boundaries_cannot_be_skipped(self):
        for boundary in ("tag-getter", "join-override", "sparse", "symbol-element"):
            state = machine()
            apply(
                state,
                allocate("getter", kind="closure"),
                allocate(
                    "input",
                    [(["str", "0"], ["symbol", "well-known:iterator"])]
                    if boundary == "symbol-element"
                    else [],
                    **({} if boundary == "tag-getter" else {"kind": "array", "length": 1}),
                ),
            )
            if boundary == "tag-getter":
                apply(
                    state,
                    define("input", ["symbol", "well-known:toStringTag"], getter=["ref", "getter"]),
                )
            if boundary == "join-override":
                apply(state, define("input", ["str", "join"], ["ref", "getter"]))
            self.ordinary(state)
            apply(
                state,
                lookup(
                    "key",
                    ["str", "toString"],
                    "builtin",
                    builtin="Object.toString" if boundary == "tag-getter" else "Array.toString",
                ),
            )
            self.refuse(
                state,
                {"kind": "conversion.builtin", "id": "key", "result": ["str", "fabricated"]},
                "getter|join override|dense own scalar",
            )

    def test_function_prototype_fallback_cannot_invent_object_stringification(self):
        for goal in ("Number", "PropertyKey"):
            state = machine()
            apply(
                state,
                allocate("prototype", kind="closure"),
                allocate("input"),
                {
                    "kind": "property.prototype",
                    "object": "input",
                    "prototype": ["ref", "prototype"],
                    "success": True,
                },
            )
            self.ordinary(state, "convert", goal)
            if goal == "Number":
                apply(
                    state,
                    lookup("convert", ["str", "valueOf"], "builtin", builtin="Object.valueOf"),
                    {"kind": "conversion.builtin", "id": "convert", "result": ["ref", "input"]},
                )
            self.refuse(
                state,
                lookup("convert", ["str", "toString"], "builtin", builtin="Object.toString"),
                "function intrinsic prototype",
            )
        if shutil.which("node"):
            native = subprocess.check_output(
                [
                    shutil.which("node"),
                    "-e",
                    "const input=Object.create(function(){});for(const f of [x=>Number(x),x=>({[x]:1})]){"
                    "try{f(input);console.log('accepted')}catch(e){console.log(e.name)}}",
                ],
                text=True,
                timeout=10,
            )
            self.assertEqual(native.splitlines(), ["TypeError", "TypeError"])

    def test_function_prototype_owned_method_can_still_supply_the_key(self):
        state = machine()
        apply(
            state,
            allocate("method", kind="closure"),
            allocate("prototype", [(["str", "toString"], ["ref", "method"])], kind="closure"),
            allocate("input"),
            {
                "kind": "property.prototype",
                "object": "input",
                "prototype": ["ref", "prototype"],
                "success": True,
            },
        )
        self.ordinary(state)
        apply(
            state,
            lookup("key", ["str", "toString"], "data", ["ref", "method"]),
            {"kind": "conversion.call", "id": "key", "child": "child"},
        )
        returned(state, "key", "child", ["str", "x"])
        apply(state, finish("key", ["str", "x"]))

    def test_detail_cannot_skip_pending_conversion_and_cleanup_is_narrow(self):
        state = machine()
        apply(state, begin("number", "Number", ["num", 7]))
        public = _external_machine(state)
        with self.assertRaisesRegex(ProtocolV8Error, "conversion.finish"):
            assert_detailed_record_compatible(
                public, "transfer", {"kind": "return", "value": ["num", 7], "handler": None}
            )
        with self.assertRaisesRegex(ProtocolV8Error, "conversion.finish"):
            conversion.assert_next(
                state,
                {
                    "kind": "transfer",
                    "fields": {
                        "kind": "cleanup",
                        "value": ["num", 7],
                        "handler": {"schema": "semantic-handler-scopes/1"},
                    },
                },
            )
        self.assertTrue(
            conversion.assert_next(
                state,
                {
                    "kind": "transfer",
                    "fields": {
                        "kind": "cleanup",
                        "value": UNDEFINED,
                        "handler": {"schema": "semantic-handler-scopes/1"},
                    },
                },
            )
        )

    def test_work_bounds_use_javascript_utf16_units(self):
        state = machine()
        self.refuse(
            state, begin("tooLong", "PropertyKey", ["str", "🚀" * 5001]), "scalar conversion bound"
        )
        self.refuse(state, begin("🚀" * 129, "Number", ["num", 1]), "identity bound")
        apply(
            state,
            begin("atBound", "PropertyKey", ["str", "🚀" * 5000]),
            finish("atBound", ["str", "🚀" * 5000]),
        )

    def test_json_numeric_equivalence_preserves_valid_index_but_rejects_booleans(self):
        state = machine()
        apply(state, allocate("input"))
        self.ordinary(state)
        state["protocol"]["primitiveConversions"]["operations"][0]["index"] = 0.0
        self.assertTrue(conversion.validate_state(state))
        state["protocol"]["primitiveConversions"]["operations"][0]["index"] = False
        with self.assertRaises(ProtocolV8Error):
            conversion.validate_state(state)

    def test_snapshot_cannot_omit_shared_authority_or_substitute_an_unchecked_snapshot(self):
        state = machine()
        apply(state, begin("number", "Number", ["num", 7]))
        public = _external_machine(state)
        with self.assertRaisesRegex(ProtocolV8Error, "complete shared state"):
            validate_protocol_state(public["protocol"], public["heap"])
        malformed = copy.deepcopy(public["protocol"])
        malformed["primitiveConversions"]["operations"][0]["candidate"] = ["num", 9]
        with self.assertRaisesRegex(ProtocolV8Error, "entry state"):
            validate_protocol_state(malformed, public["heap"], public)

    @unittest.skipUnless(shutil.which("node"), "Node is the independent scalar oracle")
    def test_scalar_conversion_matches_native_javascript(self):
        values = [UNDEFINED, ["null"], ["bool", False], ["bool", True]]
        values += [
            ["num", n]
            for n in (
                "-0",
                "NaN",
                "+Infinity",
                "-Infinity",
                1e-7,
                1e-6,
                1e20,
                1e21,
                0.1,
                1000000000000000100,
                9007199254740992,
                5e-324,
            )
        ]
        values += [
            ["str", s]
            for s in (
                "",
                " ",
                "0x10",
                "+0x10",
                "0b11",
                "0o17",
                "-0",
                "+Infinity",
                "NaN",
                "bad",
                "1e9999",
                "1_0",
                "\ufeff7\ufeff",
                "\u00857",
                "\u001c7",
                "\u180e7",
            )
        ]
        values += [["bigint", "9007199254740993"], ["bigint", "9" * 10000]]
        program = """const fs=require('fs'); const values=JSON.parse(fs.readFileSync(0,'utf8'));
        const decode=v=>v[0]==='undefined'?undefined:v[0]==='null'?null:v[0]==='bigint'?BigInt(v[1]):
          v[0]==='num'&&typeof v[1]==='string'?({'-0':-0,NaN:NaN,'+Infinity':Infinity,'-Infinity':-Infinity})[v[1]]:v[1];
        const enc=n=>['num',Object.is(n,-0)?'-0':Number.isNaN(n)?'NaN':n===Infinity?'+Infinity':n===-Infinity?'-Infinity':n];
        console.log(JSON.stringify(values.map(v=>{const x=decode(v);return [enc(Number(x)),['str',String(x)]]})));"""
        native = json.loads(
            subprocess.check_output(
                [shutil.which("node"), "-e", program], input=json.dumps(values).encode(), timeout=20
            )
        )
        for value, (number, key) in zip(values, native, strict=True):
            with self.subTest(value=str(value)[:80]):
                # JSON decimal spellings can differ while denoting the same JS binary64 value.
                self.assertTrue(
                    conversion._eq(
                        conversion.scalar_result({"goal": "Number", "candidate": value}), number
                    )
                )
                self.assertEqual(
                    conversion.scalar_result({"goal": "PropertyKey", "candidate": value}), key
                )


if __name__ == "__main__":
    unittest.main()
