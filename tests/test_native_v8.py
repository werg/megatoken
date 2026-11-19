import base64
import copy
import gzip
import json
import unittest
from pathlib import Path

from megatoken.data.native_v8 import (
    NativeV8Error,
    _apply_detail,
    _apply_observation,
    _native_checkpoint,
    _restore_native_checkpoint,
    validate_v8_artifact,
)

FIXTURE = Path(__file__).parent / "fixtures" / "semantic-regions-v8.json.gz.b64"


class NativeV8Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.document = json.loads(gzip.decompress(base64.b64decode(FIXTURE.read_bytes())))

    def test_public_interoperability_examples_are_independently_checked(self):
        for example in self.document["examples"]:
            with self.subTest(example=example["id"]):
                receipt = validate_v8_artifact(example)
                self.assertFalse(receipt["admitted"])
                self.assertEqual(receipt["checks"]["contexts"], "passed")
                self.assertEqual(receipt["checks"]["continuation"], "passed")
                self.assertEqual(receipt["checks"]["replay"], "passed")

    def test_future_context_prefix_is_rejected(self):
        example = copy.deepcopy(self.document["examples"][0])
        example["contexts"][0]["visibleThrough"] = example["public"]["timeline"][-1]["sequence"]
        with self.assertRaises(NativeV8Error):
            validate_v8_artifact(example)

    def test_old_trace_label_is_rejected(self):
        example = copy.deepcopy(self.document["examples"][0])
        example["public"]["traceFormat"] = "semantic-trace/7"
        with self.assertRaises(NativeV8Error):
            validate_v8_artifact(example)

    def test_false_region_entry_state_is_rejected(self):
        example = copy.deepcopy(self.document["examples"][1])
        region = next(row for row in example["public"]["timeline"] if row["kind"] == "region")
        region["fields"]["entry"]["stateDigest"] = "0" * 64
        with self.assertRaises(NativeV8Error):
            validate_v8_artifact(example)

    def test_false_shared_state_is_rejected_at_next_prediction(self):
        example = copy.deepcopy(self.document["examples"][1])
        # The second target's entry claims a binding value that the previous
        # local block did not commit.
        example["contexts"][2]["state"]["locals"][-1][1] = ["num", 88]
        with self.assertRaises(NativeV8Error):
            validate_v8_artifact(example)

    def test_false_region_cursor_is_rejected_at_next_prediction(self):
        example = copy.deepcopy(self.document["examples"][1])
        example["contexts"][2]["regionCursor"]["consumed"].append(
            {"invocationId": "invocation1", "frame": "invocation1", "region": "r1"}
        )
        with self.assertRaises(NativeV8Error):
            validate_v8_artifact(example)

    def test_false_request_obligation_is_rejected_at_next_prediction(self):
        example = copy.deepcopy(self.document["examples"][2])
        example["contexts"][3]["obligations"]["intents"][0][1]["payloadDigest"] = "0" * 64
        with self.assertRaises(NativeV8Error):
            validate_v8_artifact(example)

    def test_choose_and_completion_transfer_replay(self):
        state = {
            "locals": [["n", ["num", 8]]],
            "captureAliases": [],
            "heap": [],
            "stack": ["invocation1"],
            "pending": None,
            "lastReturn": None,
            "completion": None,
            "frameFunctions": [["invocation1", "run"]],
        }
        choose = {
            "expr": {
                "op": "binary",
                "operator": ">",
                "left": {"op": "local", "name": "n"},
                "right": {"op": "literal", "value": ["num", 3]},
            },
            "decision": ["bool", True],
            "path": "then",
        }
        _apply_detail(state, "choose", choose, {"sequence": 1})
        self.assertEqual(state["locals"], [["n", ["num", 8]]])
        transfer = {"kind": "throw", "value": ["num", 2], "handler": "catch-or-finally"}
        _apply_detail(state, "transfer", transfer, {"sequence": 2})
        self.assertEqual(
            state["pending"], {"kind": "throw", "value": ["num", 2], "handler": "catch-or-finally"}
        )
        choose["path"] = "else"
        with self.assertRaises(NativeV8Error):
            _apply_detail(state, "choose", choose, {"sequence": 3})

    def test_callback_request_dispatch_event_and_close_are_replayed(self):
        state = {
            "heap": [
                ["o2", {"kind": "closure", "code": "function:77", "captures": [], "properties": []}]
            ],
            "locals": [],
            "captureAliases": [],
            "effects": [],
            "stack": ["invocation1"],
            "scheduler": {"ready": 0},
        }
        obligations = {"intents": [], "handlers": [], "callbacks": [], "closureQueue": []}
        request = {
            "effect": "e1",
            "operation": "demo.withCallback",
            "arguments": [["str", "value"], ["ref", "o2"]],
            "invocation": "invocation1",
            "site": {"start": 51, "end": 131},
            "payload": '["value",{"$ownedCallback":"o2"}]',
            "promise": "p2",
        }
        _apply_detail(
            state,
            "request",
            request,
            {"sequence": 6},
            obligations,
            {"source": "host.withCallback()"},
            {"invocation1": 1},
        )
        dispatch = {
            "sequence": 7,
            "kind": "dispatch",
            "fields": {
                "invocationId": "invocation1",
                "effectId": "e1",
                "key": "invocation1/e1",
                "attempt": 1,
            },
        }
        _apply_observation(state, dispatch, set(), set(), obligations, {})
        handler = {
            "sequence": 8,
            "kind": "handler",
            "fields": {
                "subscriptionId": "callback1",
                "programId": "program1",
                "realmId": "realm1",
                "trackId": "worker1",
                "realmPolicy": "retained",
                "entry": "function:77",
                "queueLimit": 64,
                "closureId": "o2",
                "lifetime": "call",
                "ownerInvocationId": "invocation1",
            },
        }
        _apply_observation(state, handler, set(), set(), obligations, {})
        self.assertEqual(obligations["callbacks"][0][1]["state"], "active")
        event = {
            "sequence": 9,
            "kind": "event",
            "fields": {"subscriptionId": "callback1", "eventId": "event1"},
        }
        _apply_observation(state, event, set(), set(), obligations, {})
        self.assertEqual(obligations["handlers"][0][1]["seen"], ["event1"])
        close = {
            "sequence": 10,
            "kind": "handler-close",
            "fields": {"subscriptionId": "callback1", "discarded": []},
        }
        _apply_observation(state, close, set(), set(), obligations, {})
        self.assertEqual(obligations["callbacks"][0][1]["state"], "revoked")
        self.assertEqual(obligations["handlers"][0][1]["state"], "closed")

    def test_native_checkpoint_roundtrip_and_tamper_detection(self):
        context = self.document["examples"][2]["contexts"][4]
        prefix = self.document["examples"][2]["public"]["timeline"][: context["visibleThrough"]]
        checkpoint = _native_checkpoint(prefix, context)
        restored = _restore_native_checkpoint(
            checkpoint, expected_prefix=prefix, expected_state=context["state"]
        )
        self.assertEqual(restored, context["state"])
        checkpoint["state"]["stack"].append("future")
        with self.assertRaises(NativeV8Error):
            _restore_native_checkpoint(
                checkpoint, expected_prefix=prefix, expected_state=context["state"]
            )


if __name__ == "__main__":
    unittest.main()
