"""Native replay checks for bounded ordinary-data Proxy operations."""

import copy
import json
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
        step(state, {"kind": "allocate", "name": name, "objectKind": "object", "length": None,
                     "properties": [], "code": None, "captures": [], "functionKind": None,
                     "lexicalReceiver": None})
    step(state, {"kind": "property.set", "object": "target", "key": KEY, "receiver": REF("target"),
                 "value": ["num", 2], "success": True})
    for name, value in traps:
        step(state, {"kind": "allocate", "name": name, "objectKind": "closure", "length": None,
                     "properties": [], "code": name, "captures": [], "functionKind": "ordinary",
                     "lexicalReceiver": None})
        step(state, {"kind": "property.set", "object": "handler", "key": ["str", "get" if name.startswith("get") else "set"],
                     "receiver": REF("handler"), "value": value, "success": True})
    step(state, {"kind": "proxy.create", "name": "proxy", "target": REF("target"), "handler": REF("handler")})
    return state


class ProxyV8Tests(unittest.TestCase):
    def test_transparent_operations_keep_target_alias_and_proxy_identity(self):
        state = base_state()
        step(state, {"kind": "proxy.get", "object": "proxy", "key": KEY, "result": ["num", 2]})
        step(state, {"kind": "proxy.set", "object": "proxy", "key": KEY, "value": ["num", 7], "success": True})
        step(state, {"kind": "proxy.set", "object": "proxy", "key": ["str", "self"], "value": REF("proxy"), "success": True})
        self.assertEqual(validate_protocol_state(state["protocol"], state["heap"], state), True)
        heap = dict(state["heap"])
        self.assertEqual(dict(heap["target"]["properties"])["x"], ["num", 7])
        self.assertEqual(dict(heap["target"]["properties"])["self"], REF("proxy"))
        self.assertEqual(heap["proxy"]["properties"], [])

    def test_callback_entry_restore_and_finish_preserve_receiver_arguments_and_live_selection(self):
        state = base_state((("getTrap", REF("getTrap")),))
        step(state, {"kind": "proxy.get.call", "id": "read", "object": "proxy", "key": KEY, "child": "reader"})
        validate_protocol_state(state["protocol"], state["heap"], state)
        frame = state["protocol"]["callFrames"][0]
        self.assertEqual(frame["arguments"], [REF("target"), KEY, REF("proxy")])
        self.assertEqual(state["receiver"], REF("handler"))
        # Replacing the live handler field does not change the already selected callback.
        step(state, {"kind": "property.set", "object": "handler", "key": ["str", "get"], "receiver": REF("handler"),
                     "value": ["null"], "success": True})
        state = step(state, {"kind": "return", "invocation": "reader", "value": ["num", 3]})
        with self.assertRaisesRegex(ProtocolV8Error, "finish"):
            step(state, {"kind": "bind", "name": "skip", "value": ["num", 1]})
        step(state, {"kind": "proxy.finish", "id": "read", "completion": {"kind": "normal", "value": ["num", 3]}})
        self.assertEqual(state["receiver"], ["undefined"])
        validate_protocol_state(state["protocol"], state["heap"], state)

    def test_restore_rejects_forged_callback_source_and_arguments(self):
        state = base_state((("getTrap", REF("getTrap")),))
        step(state, {"kind": "proxy.get.call", "id": "read", "object": "proxy", "key": KEY, "child": "reader"})
        for mutation in (
            lambda snapshot: snapshot["protocol"]["proxyCalls"]["pending"][0].__setitem__("callee", REF("target")),
            lambda snapshot: snapshot["protocol"]["callFrames"][0]["arguments"].__setitem__(1, ["str", "wrong"]),
        ):
            forged = copy.deepcopy(state)
            mutation(forged)
            with self.assertRaises(ProtocolV8Error):
                validate_protocol_state(forged["protocol"], forged["heap"], forged)

    def test_set_trap_false_bigint_is_rejected_and_true_bigint_is_accepted(self):
        for returned, accepted in ((["bigint", "0"], False), (["bigint", "2"], True)):
            state = base_state((("setTrap", REF("setTrap")),))
            step(state, {"kind": "proxy.set.call", "id": "write", "object": "proxy", "key": KEY,
                         "value": ["num", 9], "child": "writer"})
            state = step(state, {"kind": "return", "invocation": "writer", "value": returned})
            finish = {"kind": "proxy.finish", "id": "write", "completion": {"kind": "normal", "value": ["num", 9]}}
            if accepted:
                step(state, finish)
            else:
                before = copy.deepcopy(state)
                with self.assertRaisesRegex(ProtocolV8Error, "abrupt completion"):
                    step(state, finish)
                self.assertEqual(state, before)

    def test_restore_checks_symbol_keys_and_creation_reserves_symbol_identity(self):
        state = base_state()
        state["protocol"]["proxyCalls"] = {"schema": "semantic-proxy-call-state/1", "pending": []}
        # A symbol key must name a symbol declared by this protocol state.
        from megatoken.data.protocol_v8 import _internal_machine
        from megatoken.data.proxy_v8 import validate_state
        machine = _internal_machine(state)
        machine["protocol"]["proxyCalls"]["pending"] = [{
            "id": "pending", "operation": "get", "parent": "invocation1", "child": "reader",
            "object": "proxy", "target": REF("target"), "handler": REF("handler"),
            "key": ["symbol", "missing"], "value": None, "callee": REF("getTrap")
        }]
        with self.assertRaises(ProtocolV8Error):
            validate_state(machine)
        state = initial_state()
        step(state, {"kind": "symbol.create", "id": "o50", "description": ["undefined"]})
        for name in ("target", "handler"):
            step(state, {"kind": "allocate", "name": name, "objectKind": "object", "length": None,
                         "properties": [], "code": None, "captures": [], "functionKind": None,
                         "lexicalReceiver": None})
        with self.assertRaises(ProtocolV8Error):
            step(state, {"kind": "proxy.create", "name": "o50", "target": REF("target"), "handler": REF("handler")})


if __name__ == "__main__":
    unittest.main()
