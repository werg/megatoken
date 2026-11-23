"""Retained handler scopes are checked without externalizing the full machine."""

import copy
import unittest
from unittest.mock import patch

from megatoken.data.protocol_v8 import (
    ProtocolV8Error,
    _internal_machine,
    create_protocol_state,
    validate_completion_detail,
    validate_pending_scope,
)


def machine():
    external = {
        "heap": [],
        "locals": [],
        "captureAliases": [],
        "receiver": ["undefined"],
        "stack": ["root", "close"],
        "frameFunctions": [["root", "run"], ["close", "return"]],
        "pending": None,
        "lastReturn": None,
        "completion": None,
        "protocol": create_protocol_state(),
    }
    return _internal_machine(external)


def pending(invocation="root"):
    return {
        "kind": "cleanup",
        "value": ["undefined"],
        "handler": {
            "schema": "semantic-handler-scopes/1",
            "frames": [
                {
                    "invocation": invocation,
                    "targets": [{"id": "outer", "continue": True}],
                    "handlers": [
                        {
                            "id": "inner-close",
                            "catch": False,
                            "finally": True,
                            "phase": "finally",
                            "caught": None,
                            "pending": {
                                "kind": "break",
                                "value": ["undefined"],
                                "target": "outer",
                            },
                        }
                    ],
                }
            ],
        },
    }


class PendingScopeValidationTests(unittest.TestCase):
    def test_retained_scope_uses_internal_machine_without_round_trip_conversion(self):
        selected = machine()
        with (
            patch(
                "megatoken.data.protocol_v8._external_machine",
                side_effect=AssertionError("unexpected external machine conversion"),
            ),
            patch(
                "megatoken.data.protocol_v8._internal_machine",
                side_effect=AssertionError("unexpected internal machine conversion"),
            ),
        ):
            self.assertTrue(validate_pending_scope(selected, pending(), ["root"]))

    def test_retained_scope_still_rejects_invocations_outside_selected_stack(self):
        with self.assertRaises(ProtocolV8Error):
            validate_pending_scope(machine(), pending(), ["close"])

    def test_public_handler_shape_is_rejected_before_machine_conversion(self):
        malformed = []
        wrong_kind = pending()
        wrong_kind["kind"] = "return"
        malformed.append(wrong_kind)
        wrong_frames = pending()
        wrong_frames["handler"]["frames"] = None
        malformed.append(wrong_frames)
        for detail in malformed:
            with (
                self.subTest(detail=detail),
                patch(
                    "megatoken.data.protocol_v8._internal_machine",
                    side_effect=AssertionError("invalid fields must reject before conversion"),
                ),
                self.assertRaises(ProtocolV8Error),
            ):
                validate_completion_detail({}, "transfer", detail)

    def test_retained_scope_rejects_unknown_references_and_loop_targets(self):
        missing_reference = pending()
        missing_reference["handler"]["frames"][0]["handlers"][0]["pending"] = {
            "kind": "return",
            "value": ["ref", "absent"],
        }
        missing_target = pending()
        missing_target["handler"]["frames"][0]["handlers"][0]["pending"]["target"] = "absent"
        for detail in (missing_reference, missing_target):
            with self.subTest(detail=detail), self.assertRaises(ProtocolV8Error):
                validate_pending_scope(machine(), detail, ["root"])

    def test_retained_scope_uses_selected_stack_order(self):
        detail = pending()
        second = copy.deepcopy(detail["handler"]["frames"][0])
        second["invocation"] = "close"
        second["targets"][0]["id"] = "second-target"
        second["handlers"][0]["id"] = "second-handler"
        second["handlers"][0]["pending"]["target"] = "second-target"
        detail["handler"]["frames"].append(second)
        self.assertTrue(validate_pending_scope(machine(), detail, ["root", "close"]))
        with self.assertRaisesRegex(ProtocolV8Error, "stack order"):
            validate_pending_scope(machine(), detail, ["close", "root"])


if __name__ == "__main__":
    unittest.main()
