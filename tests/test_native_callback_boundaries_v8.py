"""Public callback bindings and resumption agree with the selected producer."""

import base64
import copy
import gzip
import json
import unittest
from pathlib import Path

from megatoken.data.algorithm_v7 import stable_algorithm_hash
from megatoken.data.native_v8 import NativeV8Error, _replay_shared_state, validate_v8_artifact
from tests.test_prediction_prefix_v8 import examples


class NativeCallbackBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).parent / "fixtures/semantic-callback-regions-v8.json.gz.b64"
        cls.original = json.loads(gzip.decompress(base64.b64decode(path.read_bytes())))

    def variant(self, policy="await-returned-promise", bindings=None):
        document = copy.deepcopy(self.original)
        for row in document["public"]["timeline"]:
            if row["kind"] != "invocation":
                continue
            if row["fields"]["activation"] == "callback":
                if bindings is not None:
                    row["fields"]["bindings"] = bindings
            else:
                row["fields"]["completionPolicy"] = policy
        for context in document["contexts"]:
            context["prefixDigest"] = stable_algorithm_hash(
                document["public"]["timeline"][: context["visibleThrough"]]
            )
        return document

    def test_callback_explicit_bindings_validate_observed_arguments_without_duplicate_cells(self):
        # Direct child invocations may declare observed argument aliases. Event
        # deliveries use the handler's exact activation with no extra bindings.
        document = examples("semantic-retained-invocations-v8.json.gz.b64")[1]
        for row in document["public"]["timeline"]:
            if row["kind"] == "invocation" and row["fields"]["activation"] == "callback":
                row["fields"]["bindings"] = {"value": 0}
        for context in document["contexts"]:
            context["prefixDigest"] = stable_algorithm_hash(
                document["public"]["timeline"][: context["visibleThrough"]]
            )
        receipt = validate_v8_artifact(document)
        self.assertEqual(receipt["checks"]["contexts"], "passed")
        self.assertEqual(receipt["checks"]["continuation"], "passed")
        self.assertFalse(receipt["admitted"])

    def test_event_callback_cannot_add_an_explicit_binding_after_delivery(self):
        with self.assertRaisesRegex(NativeV8Error, "differs from its observed event delivery"):
            validate_v8_artifact(self.variant(bindings={"n": 0}))

    def test_callback_bindings_cannot_name_unobserved_arguments(self):
        for index in (-1, 1, True, "0"):
            with self.subTest(index=index), self.assertRaises(NativeV8Error):
                validate_v8_artifact(self.variant(bindings={"n": index}))

    def test_owner_completion_policy_is_restored_after_callback(self):
        # The producer's public harness accepts these three root policies on
        # the same callback trace. Only await-returned-promise unwraps p1.
        for policy in ("call-return", "owned-quiescence", "await-returned-promise"):
            with self.subTest(policy=policy):
                document = self.variant(policy=policy)
                validate_v8_artifact(document)
                for roundtrip in (False, True):
                    state = _replay_shared_state(
                        document["public"]["timeline"],
                        document["contexts"],
                        {},
                        roundtrip=roundtrip,
                    )
                    expected = ["num", 7] if policy == "await-returned-promise" else ["ref", "p1"]
                    self.assertEqual(state["completion"], {"kind": "return", "value": expected})


if __name__ == "__main__":
    unittest.main()
