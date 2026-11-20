"""IteratorClose keeps a bounded destination while guest cleanup calls run."""

import copy
import json
import unittest

from megatoken.data.native_v8 import NativeV8Error, _apply_detail
from megatoken.data.protocol_v8 import (
    ProtocolV8Error,
    _internal_machine,
    apply_protocol_record,
    create_protocol_state,
    validate_completion_detail,
)


def state():
    return {
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


def cleanup(kind="break"):
    return {
        "kind": "cleanup",
        "value": ["undefined"],
        "handler": {
            "schema": "semantic-handler-scopes/1",
            "frames": [
                {
                    "invocation": "root",
                    "targets": [{"id": "outer", "continue": True}],
                    "handlers": [
                        {
                            "id": "inner-close",
                            "catch": False,
                            "finally": True,
                            "phase": "finally",
                            "caught": None,
                            "pending": {"kind": kind, "value": ["undefined"], "target": "outer"},
                        }
                    ],
                }
            ],
        },
    }


def array_state():
    selected = _internal_machine(state())
    selected["stack"] = ["root"]
    apply_protocol_record(
        selected,
        {
            "kind": "allocate",
            "name": "array",
            "objectKind": "array",
            "length": 0,
            "properties": [],
            "code": None,
            "captures": [],
            "functionKind": None,
            "lexicalReceiver": None,
        },
    )
    return selected


class CustomIteratorContinuationTests(unittest.TestCase):
    def test_missing_intrinsic_values_cannot_be_fabricated_while_writable_iterator_can_be_shadowed(
        self,
    ):
        selected = array_state()
        before = copy.deepcopy(selected)
        for key in (["str", "values"], ["symbol", "well-known:iterator"], ["str", "toString"]):
            with self.assertRaisesRegex(ProtocolV8Error, "inherited intrinsic"):
                apply_protocol_record(
                    selected,
                    {
                        "kind": "property.get",
                        "object": "array",
                        "key": key,
                        "receiver": ["ref", "array"],
                        "result": ["undefined"],
                    },
                )
            self.assertEqual(selected, before)
        apply_protocol_record(
            selected,
            {
                "kind": "property.set",
                "object": "array",
                "key": ["symbol", "well-known:iterator"],
                "receiver": ["ref", "array"],
                "value": ["num", 3],
                "success": True,
            },
        )
        apply_protocol_record(
            selected,
            {
                "kind": "property.get",
                "object": "array",
                "key": ["symbol", "well-known:iterator"],
                "receiver": ["ref", "array"],
                "result": ["num", 3],
            },
        )
        apply_protocol_record(
            selected, {"kind": "iterator.arrayGuard", "iterated": ["ref", "array"], "result": False}
        )
        before = copy.deepcopy(selected)
        with self.assertRaisesRegex(ProtocolV8Error, "inherited intrinsic"):
            apply_protocol_record(
                selected,
                {
                    "kind": "property.set",
                    "object": "array",
                    "key": ["symbol", "well-known:unscopables"],
                    "receiver": ["ref", "array"],
                    "value": ["num", 3],
                    "success": True,
                },
            )
        self.assertEqual(selected, before)

    def test_null_prototype_array_does_not_inherit_builtin_iteration(self):
        selected = array_state()
        apply_protocol_record(
            selected,
            {"kind": "property.prototype", "object": "array", "prototype": None, "success": True},
        )
        apply_protocol_record(
            selected, {"kind": "iterator.arrayGuard", "iterated": ["ref", "array"], "result": False}
        )
        before = copy.deepcopy(selected)
        for record in (
            {"kind": "iterator.arrayGuard", "iterated": ["ref", "array"], "result": True},
            {
                "kind": "iterator.lookup",
                "receiver": ["ref", "array"],
                "key": ["symbol", "well-known:iterator"],
                "method": "Array.prototype.values",
            },
            {
                "kind": "iterator.lookup",
                "receiver": ["ref", "array"],
                "key": ["str", "values"],
                "method": "Array.prototype.values",
            },
        ):
            with self.assertRaises(ProtocolV8Error):
                apply_protocol_record(selected, record)
            self.assertEqual(selected, before)

    def test_break_and_continue_keep_their_target_during_a_nested_close_callback(self):
        for kind in ("break", "continue"):
            with self.subTest(kind=kind):
                selected = state()
                fields = cleanup(kind)
                self.assertTrue(validate_completion_detail(selected, "transfer", fields))
                _apply_detail(selected, "transfer", fields, {})
                restored = json.loads(json.dumps(selected))
                self.assertEqual(restored, selected)
                self.assertEqual(restored["stack"], ["root", "close"])
                self.assertEqual(
                    restored["pending"]["handler"]["frames"][0]["handlers"][0]["pending"],
                    {"kind": kind, "value": ["undefined"], "target": "outer"},
                )
                self.assertTrue(validate_completion_detail(restored, "transfer", fields))

    def test_unknown_duplicate_or_executable_targets_reject_before_selected_state_changes(self):
        selected = state()
        before = copy.deepcopy(selected)
        mutations = [
            lambda frame: frame["handlers"][0]["pending"].update(target="absent"),
            lambda frame: frame["handlers"][0]["pending"].update(target=[]),
            lambda frame: frame["handlers"][0]["pending"].update(value=["num", 1]),
            lambda frame: frame["targets"].append(copy.deepcopy(frame["targets"][0])),
            lambda frame: frame["targets"][0].update(targetPc=12),
            lambda frame: frame["targets"][0].update(id="inner-close"),
            lambda frame: frame.update(invocation="absent"),
            lambda frame: frame["handlers"][0].update(pending=3),
        ]
        for mutate in mutations:
            fields = cleanup()
            mutate(fields["handler"]["frames"][0])
            with self.assertRaises(ProtocolV8Error):
                validate_completion_detail(selected, "transfer", fields)
            with self.assertRaises(NativeV8Error):
                _apply_detail(selected, "transfer", fields, {})
            self.assertEqual(selected, before)

    def test_a_break_only_target_cannot_accept_continue(self):
        fields = cleanup("continue")
        fields["handler"]["frames"][0]["targets"][0]["continue"] = False
        with self.assertRaisesRegex(ProtocolV8Error, "cannot continue"):
            validate_completion_detail(state(), "transfer", fields)
        fields["handler"]["frames"][0]["handlers"][0]["pending"]["kind"] = "break"
        self.assertTrue(validate_completion_detail(state(), "transfer", fields))


if __name__ == "__main__":
    unittest.main()
