import copy
import base64
import gzip
import json
import unittest
from pathlib import Path

from megatoken.data.native_v8 import NativeV8Error, validate_v8_artifact


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
                self.assertEqual(receipt["checks"]["continuation"], "untested")
                self.assertEqual(receipt["checks"]["replay"], "unsupported")

    def test_future_context_prefix_is_rejected(self):
        example = copy.deepcopy(self.document["examples"][0])
        example["contexts"][0]["visibleThrough"] = example["public"]["timeline"][-1]["sequence"]
        with self.assertRaises(NativeV8Error): validate_v8_artifact(example)

    def test_old_trace_label_is_rejected(self):
        example = copy.deepcopy(self.document["examples"][0])
        example["public"]["traceFormat"] = "semantic-trace/7"
        with self.assertRaises(NativeV8Error): validate_v8_artifact(example)

    def test_false_region_entry_state_is_rejected(self):
        example = copy.deepcopy(self.document["examples"][1])
        region = next(row for row in example["public"]["timeline"] if row["kind"] == "region")
        region["fields"]["entry"]["stateDigest"] = "0" * 64
        with self.assertRaises(NativeV8Error): validate_v8_artifact(example)


if __name__ == "__main__": unittest.main()
