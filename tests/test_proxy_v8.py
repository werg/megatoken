"""Native replay checks for bounded ordinary-data Proxy operations."""

import copy
import json
import shutil
import subprocess
import unittest
from pathlib import Path

from megatoken.data.protocol_v8 import (
    ProtocolV8Error,
    apply_protocol_to_shared_state,
    create_protocol_state,
    encode_protocol_record,
    validate_protocol_state,
)

FIXTURE = Path(__file__).parent / "fixtures" / "semantic-protocol-v8.json"
KEY = ["str", "x"]


def REF(name):
    return ["ref", name]


def initial_state():
    fixture = json.loads(FIXTURE.read_text())["cases"][0]["initialState"]
    fixture = copy.deepcopy(fixture)
    fixture["protocol"] = create_protocol_state()
    fixture["protocol"]["proxyCalls"] = None
    return fixture


def step(state, record):
    updated = apply_protocol_to_shared_state(state, encode_protocol_record(record))
    state.clear()
    state.update(updated)
    return state


def base_state(traps=()):
    state = initial_state()
    for name in ("target", "handler"):
        step(
            state,
            {
                "kind": "allocate",
                "name": name,
                "objectKind": "object",
                "length": None,
                "properties": [],
                "code": None,
                "captures": [],
                "functionKind": None,
                "lexicalReceiver": None,
            },
        )
    step(
        state,
        {
            "kind": "property.set",
            "object": "target",
            "key": KEY,
            "receiver": REF("target"),
            "value": ["num", 2],
            "success": True,
        },
    )
    for name, value in traps:
        step(
            state,
            {
                "kind": "allocate",
                "name": name,
                "objectKind": "closure",
                "length": None,
                "properties": [],
                "code": name,
                "captures": [],
                "functionKind": "ordinary",
                "lexicalReceiver": None,
            },
        )
        step(
            state,
            {
                "kind": "property.set",
                "object": "handler",
                "key": [
                    "str",
                    next(
                        trap
                        for prefix, trap in (
                            ("get", "get"),
                            ("set", "set"),
                            ("has", "has"),
                            ("delete", "deleteProperty"),
                        )
                        if name.startswith(prefix)
                    ),
                ],
                "receiver": REF("handler"),
                "value": value,
                "success": True,
            },
        )
    step(
        state,
        {
            "kind": "proxy.create",
            "name": "proxy",
            "target": REF("target"),
            "handler": REF("handler"),
        },
    )
    return state


class ProxyV8Tests(unittest.TestCase):
    def test_generator_trap_cannot_be_entered_as_an_ordinary_callback(self):
        state = base_state((("getTrap", REF("getTrap")),))
        step(
            state,
            {
                "kind": "generator.function",
                "object": "getTrap",
                "name": "getTrap",
                "length": 0,
                "prototype": "generator-prototype",
            },
        )
        before = copy.deepcopy(state)
        with self.assertRaisesRegex(ProtocolV8Error, "generator callback"):
            step(
                state,
                {
                    "kind": "proxy.get.call",
                    "id": "read",
                    "object": "proxy",
                    "key": KEY,
                    "child": "reader",
                },
            )
        self.assertEqual(state, before)

    def test_transparent_operations_keep_target_alias_and_proxy_identity(self):
        state = base_state()
        step(state, {"kind": "proxy.get", "object": "proxy", "key": KEY, "result": ["num", 2]})
        step(
            state,
            {
                "kind": "proxy.set",
                "object": "proxy",
                "key": KEY,
                "value": ["num", 7],
                "success": True,
            },
        )
        step(
            state,
            {
                "kind": "proxy.set",
                "object": "proxy",
                "key": ["str", "self"],
                "value": REF("proxy"),
                "success": True,
            },
        )
        self.assertEqual(validate_protocol_state(state["protocol"], state["heap"], state), True)
        heap = dict(state["heap"])
        self.assertEqual(dict(heap["target"]["properties"])["x"], ["num", 7])
        self.assertEqual(dict(heap["target"]["properties"])["self"], REF("proxy"))
        self.assertEqual(heap["proxy"]["properties"], [])

    def test_callback_entry_restore_and_finish_preserve_receiver_arguments_and_live_selection(self):
        state = base_state((("getTrap", REF("getTrap")),))
        step(
            state,
            {
                "kind": "proxy.get.call",
                "id": "read",
                "object": "proxy",
                "key": KEY,
                "child": "reader",
            },
        )
        validate_protocol_state(state["protocol"], state["heap"], state)
        frame = state["protocol"]["callFrames"][0]
        self.assertEqual(frame["arguments"], [REF("target"), KEY, REF("proxy")])
        self.assertEqual(state["receiver"], REF("handler"))
        # Replacing the live handler field does not change the already selected callback.
        step(
            state,
            {
                "kind": "property.set",
                "object": "handler",
                "key": ["str", "get"],
                "receiver": REF("handler"),
                "value": ["null"],
                "success": True,
            },
        )
        state = step(state, {"kind": "return", "invocation": "reader", "value": ["num", 3]})
        with self.assertRaisesRegex(ProtocolV8Error, "finish"):
            step(state, {"kind": "bind", "name": "skip", "value": ["num", 1]})
        step(
            state,
            {
                "kind": "proxy.finish",
                "id": "read",
                "completion": {"kind": "normal", "value": ["num", 3]},
            },
        )
        self.assertEqual(state["receiver"], ["undefined"])
        validate_protocol_state(state["protocol"], state["heap"], state)

    def test_restore_rejects_forged_callback_source_and_arguments(self):
        state = base_state((("getTrap", REF("getTrap")),))
        step(
            state,
            {
                "kind": "proxy.get.call",
                "id": "read",
                "object": "proxy",
                "key": KEY,
                "child": "reader",
            },
        )
        for mutation in (
            lambda snapshot: snapshot["protocol"]["proxyCalls"]["pending"][0].__setitem__(
                "callee", REF("target")
            ),
            lambda snapshot: snapshot["protocol"]["callFrames"][0]["arguments"].__setitem__(
                1, ["str", "wrong"]
            ),
        ):
            forged = copy.deepcopy(state)
            mutation(forged)
            with self.assertRaises(ProtocolV8Error):
                validate_protocol_state(forged["protocol"], forged["heap"], forged)

    def test_set_trap_uses_bigint_and_symbol_truthiness(self):
        for returned, accepted in (
            (["bigint", "0"], False),
            (["bigint", "2"], True),
            (["bigint", "9" * 5000], True),
            (["symbol", "well-known:iterator"], True),
        ):
            state = base_state((("setTrap", REF("setTrap")),))
            step(
                state,
                {
                    "kind": "proxy.set.call",
                    "id": "write",
                    "object": "proxy",
                    "key": KEY,
                    "value": ["num", 9],
                    "child": "writer",
                },
            )
            state = step(state, {"kind": "return", "invocation": "writer", "value": returned})
            finish = {
                "kind": "proxy.finish",
                "id": "write",
                "completion": {"kind": "normal", "value": ["num", 9]},
            }
            if accepted:
                step(state, finish)
            else:
                before = copy.deepcopy(state)
                with self.assertRaisesRegex(ProtocolV8Error, "abrupt completion"):
                    step(state, finish)
                self.assertEqual(state, before)

    def test_restore_rejects_reversed_nested_callback_order(self):
        state = base_state((("getTrap", REF("getTrap")),))
        for identity in ("outer", "inner"):
            step(
                state,
                {
                    "kind": "proxy.get.call",
                    "id": identity,
                    "object": "proxy",
                    "key": KEY,
                    "child": identity + "Trap",
                },
            )
        self.assertTrue(validate_protocol_state(state["protocol"], state["heap"], state))
        forged = copy.deepcopy(state)
        forged["protocol"]["proxyCalls"]["pending"].reverse()
        with self.assertRaisesRegex(ProtocolV8Error, "stack order"):
            validate_protocol_state(forged["protocol"], forged["heap"], forged)
        step(state, {"kind": "return", "invocation": "innerTrap", "value": ["num", 3]})
        step(
            state,
            {
                "kind": "proxy.finish",
                "id": "inner",
                "completion": {"kind": "normal", "value": ["num", 3]},
            },
        )
        self.assertTrue(validate_protocol_state(state["protocol"], state["heap"], state))

    def test_restore_checks_symbol_keys_and_creation_reserves_symbol_identity(self):
        state = base_state()
        state["protocol"]["proxyCalls"] = {"schema": "semantic-proxy-call-state/1", "pending": []}
        # A symbol key must name a symbol declared by this protocol state.
        from megatoken.data.protocol_v8 import _internal_machine
        from megatoken.data.proxy_v8 import validate_state

        machine = _internal_machine(state)
        machine["protocol"]["proxyCalls"]["pending"] = [
            {
                "id": "pending",
                "operation": "get",
                "parent": "invocation1",
                "child": "reader",
                "object": "proxy",
                "target": REF("target"),
                "handler": REF("handler"),
                "key": ["symbol", "missing"],
                "value": None,
                "callee": REF("getTrap"),
                "strict": False,
            }
        ]
        with self.assertRaises(ProtocolV8Error):
            validate_state(machine)
        state = initial_state()
        step(state, {"kind": "symbol.create", "id": "o50", "description": ["undefined"]})
        for name in ("target", "handler"):
            step(
                state,
                {
                    "kind": "allocate",
                    "name": name,
                    "objectKind": "object",
                    "length": None,
                    "properties": [],
                    "code": None,
                    "captures": [],
                    "functionKind": None,
                    "lexicalReceiver": None,
                },
            )
        with self.assertRaises(ProtocolV8Error):
            step(
                state,
                {
                    "kind": "proxy.create",
                    "name": "o50",
                    "target": REF("target"),
                    "handler": REF("handler"),
                },
            )


class ProxyPresenceDeletionTests(unittest.TestCase):
    def begin(self, operation, key=KEY, *, strict=False):
        name = operation + "Trap"
        state = base_state(((name, REF(name)),))
        step(
            state,
            {
                "kind": "proxy." + operation + ".call",
                "id": "operation",
                **({"strict": strict} if operation == "delete" else {}),
                "object": "proxy",
                "key": key,
                "child": "callback",
            },
        )
        self.assertEqual(state["receiver"], REF("handler"))
        self.assertEqual(state["protocol"]["callFrames"][-1]["arguments"], [REF("target"), key])
        validate_protocol_state(state["protocol"], state["heap"], state)
        return state

    def finish(self, state, returned, result, *, thrown=False):
        step(
            state,
            {"kind": "throw" if thrown else "return", "invocation": "callback", "value": returned},
        )
        record = {
            "kind": "proxy.finish",
            "id": "operation",
            "completion": {"kind": "throw" if thrown else "normal", "value": result},
        }
        return step(state, record)

    def test_transparent_presence_and_deletion_keep_identity_and_own_order(self):
        state = base_state()
        step(
            state,
            {
                "kind": "proxy.has",
                "object": "proxy",
                "key": ["str", "toString"],
                "result": ["bool", True],
            },
        )
        step(state, {"kind": "proxy.delete", "object": "proxy", "key": KEY, "success": True})
        step(state, {"kind": "proxy.has", "object": "proxy", "key": KEY, "result": ["bool", False]})
        step(
            state,
            {
                "kind": "proxy.delete",
                "object": "proxy",
                "key": ["str", "toString"],
                "success": True,
            },
        )
        step(
            state,
            {
                "kind": "proxy.has",
                "object": "proxy",
                "key": ["str", "toString"],
                "result": ["bool", True],
            },
        )
        self.assertEqual(dict(state["heap"])["proxy"]["properties"], [])
        validate_protocol_state(state["protocol"], state["heap"], state)

    def test_accessors_are_present_and_deleted_without_getter_calls(self):
        state = base_state()
        step(
            state,
            {
                "kind": "property.define",
                "object": "target",
                "key": KEY,
                "descriptor": {
                    "kind": "accessor",
                    "get": ["undefined"],
                    "set": ["undefined"],
                    "enumerable": True,
                    "configurable": True,
                },
                "success": True,
            },
        )
        step(state, {"kind": "proxy.has", "object": "proxy", "key": KEY, "result": ["bool", True]})
        step(state, {"kind": "proxy.delete", "object": "proxy", "key": KEY, "success": True})
        self.assertEqual(state["protocol"]["callFrames"], [])

    def test_traps_coerce_boolean_once_and_keep_target_unmodified(self):
        for operation in ("has", "delete"):
            for value, truthy in (
                (["undefined"], False),
                (["num", 0], False),
                (["str", "yes"], True),
                (["bigint", "0"], False),
                (["bigint", "9" * 5000], True),
                (["symbol", "well-known:iterator"], True),
                (REF("target"), True),
            ):
                with self.subTest(operation=operation, value=value[0], truthy=truthy):
                    state = self.begin(operation)
                    self.finish(state, value, ["bool", truthy])
                    self.assertEqual(
                        dict(dict(state["heap"])["target"]["properties"])["x"], ["num", 2]
                    )
                    self.assertEqual(state["protocol"]["proxyCalls"]["pending"], [])

    def test_selected_trap_survives_noncallable_replacement_and_throw_identity(self):
        for operation, trap in (("has", "has"), ("delete", "deleteProperty")):
            state = self.begin(operation)
            step(
                state,
                {
                    "kind": "property.set",
                    "object": "handler",
                    "key": ["str", trap],
                    "receiver": REF("handler"),
                    "value": ["num", 42],
                    "success": True,
                },
            )
            validate_protocol_state(state["protocol"], state["heap"], state)
            self.finish(state, REF("target"), REF("target"), thrown=True)
            validate_protocol_state(state["protocol"], state["heap"], state)
            before = copy.deepcopy(state)
            with self.assertRaises(ProtocolV8Error):
                step(
                    state,
                    {
                        "kind": "proxy." + operation + ".call",
                        "id": "again",
                        **({"strict": False} if operation == "delete" else {}),
                        "object": "proxy",
                        "key": KEY,
                        "child": "second",
                    },
                )
            self.assertEqual(state, before)

    def test_invariants_observe_descriptor_after_callback_mutation(self):
        for operation, returned in (("has", ["bool", False]), ("delete", ["bool", True])):
            state = self.begin(operation)
            step(
                state, {"kind": "property.delete", "object": "target", "key": KEY, "success": True}
            )
            step(state, {"kind": "property.preventExtensions", "object": "target"})
            self.finish(state, returned, returned)
            self.assertNotIn("x", dict(dict(state["heap"])["target"]["properties"]))

            for frozen in (True, False):
                state = self.begin(operation)
                if frozen:
                    step(
                        state,
                        {
                            "kind": "property.define",
                            "object": "target",
                            "key": KEY,
                            "descriptor": {
                                "kind": "data",
                                "value": ["num", 2],
                                "writable": True,
                                "enumerable": True,
                                "configurable": False,
                            },
                            "success": True,
                        },
                    )
                else:
                    step(state, {"kind": "property.preventExtensions", "object": "target"})
                step(state, {"kind": "return", "invocation": "callback", "value": returned})
                before = copy.deepcopy(state)
                with self.assertRaisesRegex(ProtocolV8Error, "target invariant"):
                    step(
                        state,
                        {
                            "kind": "proxy.finish",
                            "id": "operation",
                            "completion": {"kind": "normal", "value": returned},
                        },
                    )
                self.assertEqual(state, before)

    def test_restore_authenticates_operation_key_and_argument_count(self):
        for operation in ("has", "delete"):
            state = self.begin(operation)
            for mutate in (
                lambda s: s["protocol"]["proxyCalls"]["pending"][-1].__setitem__(
                    "operation", "get"
                ),
                lambda s: s["protocol"]["proxyCalls"]["pending"][-1].__setitem__(
                    "key", ["str", "wrong"]
                ),
                lambda s: s["protocol"]["callFrames"][-1]["arguments"].append(REF("proxy")),
                lambda s: s["protocol"]["proxyCalls"]["pending"][-1].__setitem__(
                    "value", ["undefined"]
                ),
            ):
                forged = copy.deepcopy(state)
                mutate(forged)
                with self.assertRaises(ProtocolV8Error):
                    validate_protocol_state(forged["protocol"], forged["heap"], forged)
            for result in (["bool", False], ["num", 1]):
                candidate = copy.deepcopy(state)
                step(
                    candidate, {"kind": "return", "invocation": "callback", "value": ["str", "yes"]}
                )
                before = copy.deepcopy(candidate)
                with self.assertRaises(ProtocolV8Error):
                    step(
                        candidate,
                        {
                            "kind": "proxy.finish",
                            "id": "operation",
                            "completion": {"kind": "normal", "value": result},
                        },
                    )
                self.assertEqual(candidate, before)


class ProxyConstructorArrowTests(unittest.TestCase):
    def test_pre_super_arrow_trap_reads_live_receiver_after_constructor_substitution(self):
        state = base_state()

        def allocate(name, kind="object", arrow=False):
            step(
                state,
                {
                    "kind": "allocate",
                    "name": name,
                    "objectKind": kind,
                    "length": None,
                    "properties": [],
                    "code": name if kind == "closure" else None,
                    "captures": [],
                    "functionKind": "arrow" if arrow else "ordinary" if kind == "closure" else None,
                    "lexicalReceiver": ["undefined"] if arrow else None,
                },
            )

        for name, parent in (("Base", None), ("Derived", "Base")):
            allocate(name, "closure")
            allocate(name + "-prototype")
            step(
                state,
                {
                    "kind": "callable.define",
                    "object": name,
                    "constructorKind": "derived" if parent else "base",
                    "classConstructor": True,
                },
            )
            step(
                state,
                {
                    "kind": "class.link",
                    "object": name,
                    "prototype": REF(name + "-prototype"),
                    "heritage": REF(parent) if parent else ["undefined"],
                },
            )
        step(
            state,
            {
                "kind": "construct.enter",
                "id": "derived-construction",
                "child": "derived-body",
                "callee": REF("Derived"),
                "newTarget": REF("Derived"),
                "arguments": [],
                "receiver": ["undefined"],
                "mode": "ordinary",
                "owner": None,
            },
        )
        allocate("captured-trap", "closure", arrow=True)
        step(
            state,
            {
                "kind": "callable.define",
                "object": "captured-trap",
                "constructorKind": None,
                "classConstructor": False,
            },
        )
        step(
            state,
            {
                "kind": "callable.capture",
                "object": "captured-trap",
                "frame": "derived-construction",
                "thisCell": "derived-body:this",
            },
        )
        step(
            state, {"kind": "super.select", "owner": "derived-construction", "callee": REF("Base")}
        )
        step(
            state,
            {
                "kind": "construct.enter",
                "id": "base-construction",
                "child": "base-body",
                "callee": REF("Base"),
                "newTarget": REF("Derived"),
                "arguments": [],
                "receiver": REF("instance"),
                "mode": "super",
                "owner": "derived-construction",
            },
        )
        step(state, {"kind": "return", "invocation": "base-body", "value": ["undefined"]})
        step(
            state,
            {
                "kind": "construct.finish",
                "id": "base-construction",
                "completion": {"kind": "normal", "value": REF("instance")},
            },
        )
        step(
            state,
            {
                "kind": "super.bind",
                "owner": "derived-construction",
                "receiver": REF("instance"),
                "child": None,
            },
        )
        step(state, {"kind": "return", "invocation": "derived-body", "value": REF("target")})
        step(
            state,
            {
                "kind": "construct.finish",
                "id": "derived-construction",
                "completion": {"kind": "normal", "value": REF("target")},
            },
        )
        step(
            state,
            {
                "kind": "property.set",
                "object": "handler",
                "key": ["str", "has"],
                "receiver": REF("handler"),
                "value": REF("captured-trap"),
                "success": True,
            },
        )
        step(
            state,
            {
                "kind": "proxy.has.call",
                "id": "presence",
                "object": "proxy",
                "key": KEY,
                "child": "captured-callback",
            },
        )
        self.assertEqual(state["receiver"], REF("instance"))
        self.assertEqual(state["protocol"]["callFrames"][-1]["callee"], REF("captured-trap"))
        self.assertEqual(dict(state["heap"])["captured-trap"]["lexicalReceiver"], ["undefined"])
        step(
            state,
            {
                "kind": "construct.this",
                "invocation": "captured-callback",
                "result": REF("instance"),
            },
        )
        step(
            state,
            {
                "kind": "construct.target",
                "invocation": "captured-callback",
                "result": REF("Derived"),
            },
        )
        validate_protocol_state(state["protocol"], state["heap"], state)
        for mutate in (
            lambda snapshot: snapshot.__setitem__("receiver", REF("target")),
            lambda snapshot: snapshot.__setitem__(
                "captureAliases", [["forged", "derived-body:this"]]
            ),
            lambda snapshot: snapshot["protocol"]["callFrames"][-1].__setitem__("parent", "absent"),
        ):
            forged = copy.deepcopy(state)
            mutate(forged)
            with self.assertRaises(ProtocolV8Error):
                validate_protocol_state(forged["protocol"], forged["heap"], forged)
        step(state, {"kind": "return", "invocation": "captured-callback", "value": ["bool", True]})
        step(
            state,
            {
                "kind": "proxy.finish",
                "id": "presence",
                "completion": {"kind": "normal", "value": ["bool", True]},
            },
        )
        validate_protocol_state(state["protocol"], state["heap"], state)


class ProxyFaultTests(unittest.TestCase):
    begin = ProxyPresenceDeletionTests.begin

    def test_fault_is_derived_and_allocates_exact_fresh_identity(self):
        for operation, returned in (("has", ["bool", False]), ("delete", ["bool", True])):
            state = self.begin(operation)
            step(state, {"kind": "property.preventExtensions", "object": "target"})
            step(state, {"kind": "return", "invocation": "callback", "value": returned})
            before = copy.deepcopy(state)
            for error, identity in (
                ("target", "operation"),
                ("fault", "forged"),
                ("well-known:iterator", "operation"),
            ):
                with self.assertRaises(ProtocolV8Error):
                    step(state, {"kind": "proxy.fault", "id": identity, "error": error})
                self.assertEqual(state, before)
            step(state, {"kind": "proxy.fault", "id": "operation", "error": "fault"})
            self.assertEqual(state["lastReturn"], {"kind": "throw", "value": REF("fault")})
            self.assertEqual(state["protocol"]["proxyCalls"]["pending"], [])
            self.assertFalse(dict(state["heap"])["target"]["extensible"])
            error = dict(state["heap"])["fault"]
            self.assertEqual(error["declaredError"], "TypeError")
            self.assertEqual(dict(error["properties"])["name"], ["str", "TypeError"])
            validate_protocol_state(state["protocol"], state["heap"], state)

    def test_no_fault_can_replace_success_or_a_guest_thrown_identity(self):
        for returned, thrown in ((["bool", True], False), (REF("target"), True)):
            state = self.begin("has")
            step(
                state,
                {
                    "kind": "throw" if thrown else "return",
                    "invocation": "callback",
                    "value": returned,
                },
            )
            before = copy.deepcopy(state)
            with self.assertRaisesRegex(ProtocolV8Error, "derived"):
                step(state, {"kind": "proxy.fault", "id": "operation", "error": "invented"})
            self.assertEqual(state, before)

    @unittest.skipUnless(shutil.which("node"), "Node is the independent Proxy diagnostic oracle")
    def test_has_and_delete_invariant_messages_match_native_with_live_restrictions(self):
        inputs, messages = [], []
        for operation in ("has", "delete"):
            for key in (["str", "x"], ["str", "a\0b"], ["symbol", "well-known:iterator"]):
                for restriction in ("nonconfigurable", "nonextensible", "both"):
                    state = self.begin(operation, key)
                    step(
                        state,
                        {
                            "kind": "property.define",
                            "object": "target",
                            "key": key,
                            "descriptor": {
                                "kind": "data",
                                "value": ["num", 2],
                                "writable": True,
                                "enumerable": True,
                                "configurable": restriction == "nonextensible",
                            },
                            "success": True,
                        },
                    )
                    if restriction != "nonconfigurable":
                        step(state, {"kind": "property.preventExtensions", "object": "target"})
                    step(
                        state,
                        {
                            "kind": "return",
                            "invocation": "callback",
                            "value": ["bool", operation == "delete"],
                        },
                    )
                    step(state, {"kind": "proxy.fault", "id": "operation", "error": "fault"})
                    messages.append(dict(dict(state["heap"])["fault"]["properties"])["message"][1])
                    inputs.append([operation, key, restriction])
        program = """const fs=require('node:fs');const inputs=JSON.parse(fs.readFileSync(0,'utf8'));
        const messages=inputs.map(([operation,tag,restriction])=>{
          const key=tag[0]==='symbol'?Symbol.iterator:tag[1],target={};
          Object.defineProperty(target,key,{value:2,writable:true,enumerable:true,
            configurable:restriction==='nonextensible'});
          if(restriction!=='nonconfigurable')Object.preventExtensions(target);
          const proxy=new Proxy(target,operation==='has'?{has(){return false;}}:{deleteProperty(){return true;}});
          try{operation==='has'?Reflect.has(proxy,key):Reflect.deleteProperty(proxy,key);return null;}
          catch(error){return error.message;}
        });console.log(JSON.stringify(messages));"""
        native = json.loads(
            subprocess.check_output(
                [shutil.which("node"), "-e", program], input=json.dumps(inputs).encode(), timeout=20
            )
        )
        self.assertEqual(messages, native)

    def test_frozen_get_and_set_faults_derive_from_return_and_live_descriptor(self):
        for operation in ("get", "set"):
            for descriptor in (
                {
                    "kind": "data",
                    "value": ["num", 2],
                    "writable": False,
                    "enumerable": True,
                    "configurable": False,
                },
                {
                    "kind": "accessor",
                    "get": ["undefined"],
                    "set": ["undefined"],
                    "enumerable": True,
                    "configurable": False,
                },
            ):
                name = operation + "Trap"
                state = base_state(((name, REF(name)),))
                step(
                    state,
                    {
                        "kind": "proxy." + operation + ".call",
                        "id": "operation",
                        "object": "proxy",
                        "key": KEY,
                        "child": "callback",
                        **({"value": ["num", 9]} if operation == "set" else {}),
                    },
                )
                step(
                    state,
                    {
                        "kind": "property.define",
                        "object": "target",
                        "key": KEY,
                        "descriptor": descriptor,
                        "success": True,
                    },
                )
                step(state, {"kind": "return", "invocation": "callback", "value": ["num", 3]})
                before = copy.deepcopy(state)
                with self.assertRaisesRegex(ProtocolV8Error, "fault required"):
                    step(
                        state,
                        {
                            "kind": "proxy.finish",
                            "id": "operation",
                            "completion": {
                                "kind": "normal",
                                "value": ["num", 3 if operation == "get" else 9],
                            },
                        },
                    )
                self.assertEqual(state, before)
                step(state, {"kind": "proxy.fault", "id": "operation", "error": "fault"})
                self.assertEqual(state["lastReturn"], {"kind": "throw", "value": REF("fault")})
                self.assertEqual(dict(state["heap"])["fault"]["declaredError"], "TypeError")
                validate_protocol_state(state["protocol"], state["heap"], state)

    def test_false_strict_delete_is_a_caller_error_even_when_property_is_absent(self):
        for key, display in (
            (["str", "absent"], "absent"),
            (["str", "\0symbol:test"], "\0symbol:test"),
            (["symbol", "well-known:iterator"], "Symbol(Symbol.iterator)"),
        ):
            state = self.begin("delete", key, strict=True)
            step(state, {"kind": "return", "invocation": "callback", "value": ["num", 0]})
            before = copy.deepcopy(state)
            with self.assertRaisesRegex(ProtocolV8Error, "fault required"):
                step(
                    state,
                    {
                        "kind": "proxy.finish",
                        "id": "operation",
                        "completion": {"kind": "normal", "value": ["bool", False]},
                    },
                )
            self.assertEqual(state, before)
            step(state, {"kind": "proxy.fault", "id": "operation", "error": "fault"})
            message = dict(dict(state["heap"])["fault"]["properties"])["message"]
            self.assertEqual(
                message,
                [
                    "str",
                    "'deleteProperty' on proxy: trap returned falsish for property '"
                    + display
                    + "'",
                ],
            )
            self.assertEqual(state["lastReturn"], {"kind": "throw", "value": REF("fault")})

    def test_transparent_strict_delete_fault_requires_live_nonconfigurable_property(self):
        for key, display in (
            (["str", "x"], "x"),
            (["str", "a\0b"], "a\0b"),
            (["symbol", "well-known:iterator"], "Symbol(Symbol.iterator)"),
        ):
            state = base_state()
            fault = {"kind": "proxy.delete.fault", "object": "proxy", "key": key, "error": "fault"}
            before = copy.deepcopy(state)
            with self.assertRaisesRegex(ProtocolV8Error, "nonconfigurable"):
                step(state, fault)
            self.assertEqual(state, before)
            step(
                state,
                {
                    "kind": "property.define",
                    "object": "target",
                    "key": key,
                    "descriptor": {
                        "kind": "data",
                        "value": ["num", 2],
                        "writable": True,
                        "enumerable": True,
                        "configurable": False,
                    },
                    "success": True,
                },
            )
            step(state, {"kind": "proxy.delete", "object": "proxy", "key": key, "success": False})
            before = copy.deepcopy(state)
            for forged in ({**fault, "error": "target"}, {**fault, "message": "invented"}):
                with self.assertRaises(ProtocolV8Error):
                    step(state, forged)
                self.assertEqual(state, before)
            target_before = copy.deepcopy(dict(state["heap"])["target"])
            step(state, fault)
            self.assertEqual(dict(state["heap"])["target"], target_before)
            error = dict(state["heap"])["fault"]
            self.assertEqual(error["declaredError"], "TypeError")
            self.assertEqual(
                dict(error["properties"])["message"],
                ["str", "Cannot delete property '" + display + "' of #<Object>"],
            )
            self.assertEqual(state["lastReturn"], {"kind": "throw", "value": REF("fault")})
            validate_protocol_state(state["protocol"], state["heap"], state)

    def test_transparent_delete_fault_cannot_bypass_selected_trap(self):
        state = base_state((("deleteTrap", REF("deleteTrap")),))
        step(
            state,
            {
                "kind": "property.define",
                "object": "target",
                "key": KEY,
                "descriptor": {
                    "kind": "data",
                    "value": ["num", 2],
                    "writable": True,
                    "enumerable": True,
                    "configurable": False,
                },
                "success": True,
            },
        )
        before = copy.deepcopy(state)
        with self.assertRaisesRegex(ProtocolV8Error, "explicit call boundary"):
            step(
                state,
                {"kind": "proxy.delete.fault", "object": "proxy", "key": KEY, "error": "fault"},
            )
        self.assertEqual(state, before)

    def test_non_delete_operations_cannot_retain_strict_delete_context(self):
        state = self.begin("has")
        forged = copy.deepcopy(state)
        forged["protocol"]["proxyCalls"]["pending"][-1]["strict"] = True
        with self.assertRaisesRegex(ProtocolV8Error, "strict"):
            validate_protocol_state(forged["protocol"], forged["heap"], forged)


if __name__ == "__main__":
    unittest.main()
