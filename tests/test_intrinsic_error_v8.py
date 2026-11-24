"""Implicit error claims must follow a derived public-machine fault."""

import copy
import unittest

from megatoken.data.protocol_v8 import ProtocolV8Error, validate_protocol_state
from tests.test_proxy_v8 import REF, base_state, step


class IntrinsicErrorTests(unittest.TestCase):
    def test_invalid_has_messages_are_derived_from_values_without_conversion(self):
        cases = (
            (["str", "x"], ["null"], "Cannot use 'in' operator to search for 'x' in null"),
            (["num", "-0"], ["bool", False], "Cannot use 'in' operator to search for '0' in false"),
            (
                ["symbol", "well-known:iterator"],
                ["undefined"],
                "Cannot use 'in' operator to search for 'Symbol(Symbol.iterator)' in undefined",
            ),
            (
                REF("target"),
                ["bigint", "7"],
                "Cannot use 'in' operator to search for '#<Object>' in 7",
            ),
            (
                ["str", "a\n\x00b"],
                ["str", "text"],
                "Cannot use 'in' operator to search for 'a\n\x00b' in text",
            ),
        )
        for key, target, message in cases:
            with self.subTest(key=key, target=target):
                state = base_state()
                step(
                    state,
                    {
                        "kind": "property.fault",
                        "operation": "has",
                        "target": target,
                        "key": key,
                        "error": "fault",
                    },
                )
                step(
                    state,
                    {
                        "kind": "property.get",
                        "object": "fault",
                        "key": ["str", "message"],
                        "receiver": REF("fault"),
                        "result": ["str", message],
                    },
                )
                self.assertEqual(state["protocol"]["callFrames"], [])
                before = copy.deepcopy(state)
                with self.assertRaises(ProtocolV8Error):
                    step(
                        state,
                        {
                            "kind": "property.get",
                            "object": "fault",
                            "key": ["str", "message"],
                            "receiver": REF("fault"),
                            "result": ["str", "forged"],
                        },
                    )
                self.assertEqual(state, before)

    def test_invalid_has_rhs_allocates_a_new_error_before_key_conversion(self):
        for target in (
            ["null"],
            ["undefined"],
            ["bool", False],
            ["num", 1],
            ["str", "text"],
            ["bigint", "7"],
            ["symbol", "well-known:iterator"],
        ):
            with self.subTest(target=target):
                state = base_state((("getTrap", REF("getTrap")),))
                before = copy.deepcopy(dict(state["heap"])["target"])
                step(
                    state,
                    {
                        "kind": "property.fault",
                        "operation": "has",
                        "target": target,
                        "key": REF("target"),
                        "error": "fault",
                    },
                )
                self.assertEqual(state["lastReturn"], {"kind": "throw", "value": REF("fault")})
                self.assertEqual(dict(state["heap"])["target"], before)
                self.assertEqual(state["protocol"]["callFrames"], [])
                step(
                    state,
                    {
                        "kind": "property.get",
                        "object": "fault",
                        "key": ["str", "name"],
                        "receiver": REF("fault"),
                        "result": ["str", "TypeError"],
                    },
                )
                step(state, {"kind": "bind", "name": "caught", "value": REF("fault")})
                validate_protocol_state(state["protocol"], state["heap"], state)

    def test_valid_rhs_unknown_key_and_reused_identity_refuse_without_partial_allocation(self):
        state = base_state()
        record = {
            "kind": "property.fault",
            "operation": "has",
            "target": ["null"],
            "key": REF("target"),
            "error": "fault",
        }
        for changed in (
            {"target": REF("target")},
            {"key": REF("unknown")},
            {"error": "target"},
            {"error": "well-known:iterator"},
            {"operation": "delete"},
            {"message": "invented"},
        ):
            before = copy.deepcopy(state)
            with self.assertRaises(ProtocolV8Error):
                step(state, {**record, **changed})
            self.assertEqual(state, before)
        with self.assertRaises(ProtocolV8Error):
            step(
                state,
                {
                    "kind": "error.create",
                    "id": "invented",
                    "errorType": "TypeError",
                    "message": "invented",
                },
            )

    def test_error_brand_is_retained_when_guest_changes_ordinary_name(self):
        state = base_state()
        step(
            state,
            {
                "kind": "property.fault",
                "operation": "has",
                "target": ["null"],
                "key": ["str", "x"],
                "error": "fault",
            },
        )
        step(
            state,
            {
                "kind": "property.set",
                "object": "fault",
                "key": ["str", "name"],
                "receiver": REF("fault"),
                "value": ["str", "changed"],
                "success": True,
            },
        )
        self.assertEqual(dict(state["heap"])["fault"]["declaredError"], "TypeError")
        validate_protocol_state(state["protocol"], state["heap"], state)
        for brand in ("InventedError", {}, [], None, False):
            forged = copy.deepcopy(state)
            dict(forged["heap"])["fault"]["declaredError"] = brand
            with self.assertRaises(ProtocolV8Error):
                validate_protocol_state(forged["protocol"], forged["heap"], forged)

    def test_known_error_brand_requires_its_derived_allocation_registry(self):
        state = base_state()
        forged = copy.deepcopy(state)
        dict(forged["heap"])["target"]["declaredError"] = "TypeError"
        with self.assertRaisesRegex(ProtocolV8Error, "allocation registry"):
            validate_protocol_state(forged["protocol"], forged["heap"], forged)
        before = copy.deepcopy(forged)
        with self.assertRaisesRegex(ProtocolV8Error, "allocation registry"):
            step(
                forged,
                {
                    "kind": "property.fault",
                    "operation": "has",
                    "target": ["null"],
                    "key": ["str", "x"],
                    "error": "second",
                },
            )
        self.assertEqual(forged, before)
        step(
            state,
            {
                "kind": "property.fault",
                "operation": "has",
                "target": ["null"],
                "key": ["str", "x"],
                "error": "fault",
            },
        )
        self.assertEqual(
            state["protocol"]["intrinsicErrors"],
            {
                "schema": "semantic-intrinsic-error-state/1",
                "errors": [{"id": "fault", "errorType": "TypeError"}],
            },
        )
        for mutate in (
            lambda snapshot: snapshot["protocol"].__setitem__("intrinsicErrors", None),
            lambda snapshot: snapshot["protocol"]["intrinsicErrors"].__setitem__("errors", []),
            lambda snapshot: snapshot["protocol"]["intrinsicErrors"]["errors"][0].__setitem__(
                "id", "target"
            ),
            lambda snapshot: snapshot["protocol"]["intrinsicErrors"]["errors"][0].__setitem__(
                "errorType", "RangeError"
            ),
            lambda snapshot: snapshot["protocol"]["intrinsicErrors"]["errors"].append(
                {"id": "fault", "errorType": "TypeError"}
            ),
            lambda snapshot: dict(snapshot["heap"])["fault"].pop("declaredError"),
            lambda snapshot: dict(snapshot["heap"])["target"].__setitem__(
                "declaredError", "TypeError"
            ),
        ):
            forged = copy.deepcopy(state)
            mutate(forged)
            with self.assertRaises(ProtocolV8Error):
                validate_protocol_state(forged["protocol"], forged["heap"], forged)

    def test_error_identity_and_message_bounds_match_javascript_utf16_units(self):
        from megatoken.data.intrinsic_error_v8 import allocate_intrinsic_error_v8
        from megatoken.data.protocol_v8 import _internal_machine

        state = base_state()
        record = {
            "kind": "property.fault",
            "operation": "has",
            "target": ["null"],
            "key": ["str", "x"],
            "error": "🚀" * 128,
        }
        step(state, record)
        before = copy.deepcopy(state)
        with self.assertRaisesRegex(ProtocolV8Error, "identity"):
            step(state, {**record, "error": "🚀" * 129})
        self.assertEqual(state, before)
        machine = _internal_machine(state)
        before = copy.deepcopy(machine)
        with self.assertRaisesRegex(ProtocolV8Error, "message"):
            allocate_intrinsic_error_v8(machine, "error", "TypeError", "🚀" * 2049)
        self.assertEqual(machine, before)
        allocate_intrinsic_error_v8(machine, "error", "TypeError", "🚀" * 2048)
        self.assertEqual(machine["heap"]["error"]["declaredError"], "TypeError")


class ConversionErrorMutationTests(unittest.TestCase):
    def test_error_brand_is_stable_but_name_and_message_remain_guest_writable(self):
        from megatoken.data import primitive_conversion_v8 as conversion
        from tests.test_primitive_conversion_v8 import allocate, apply, begin, machine

        state = machine()
        apply(
            state,
            begin("conversion", "Number", ["symbol", "well-known:iterator"]),
            {"kind": "conversion.fault", "id": "conversion", "error": "error"},
        )
        for key in ("name", "message"):
            apply(
                state,
                {
                    "kind": "property.set",
                    "object": "error",
                    "key": ["str", key],
                    "receiver": REF("error"),
                    "value": ["str", "changed"],
                    "success": True,
                },
            )
        self.assertEqual(state["heap"]["error"]["declaredError"], "TypeError")
        apply(state, allocate("ordinary", ((["str", "name"], ["str", "TypeError"]),)))
        forged = copy.deepcopy(state)
        forged["protocol"]["primitiveConversions"]["operations"][-1]["result"] = REF("ordinary")
        with self.assertRaisesRegex(ProtocolV8Error, "identity/brand"):
            conversion.validate_state(forged)


if __name__ == "__main__":
    unittest.main()
