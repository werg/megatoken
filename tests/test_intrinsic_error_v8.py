"""Implicit error claims must follow a derived public-machine fault."""

import copy
import unittest

from megatoken.data.protocol_v8 import ProtocolV8Error, validate_protocol_state
from tests.test_proxy_v8 import REF, base_state, step


class IntrinsicErrorTests(unittest.TestCase):
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
        forged = copy.deepcopy(state)
        dict(forged["heap"])["fault"]["declaredError"] = "InventedError"
        with self.assertRaises(ProtocolV8Error):
            validate_protocol_state(forged["protocol"], forged["heap"], forged)


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
