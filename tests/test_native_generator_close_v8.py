"""The frozen generator-close population case retains exact causal heap rows."""

import base64
import gzip
import json
import unittest
from pathlib import Path

from megatoken.data.native_v8 import validate_v8_artifact


class NativeGeneratorCloseTests(unittest.TestCase):
    def test_source_qualified_generator_close_replays_every_checkpoint(self):
        fixture = Path(__file__).parent / "fixtures/semantic-generator-close-v8.json.gz.b64"
        document = json.loads(gzip.decompress(base64.b64decode(fixture.read_bytes())))
        receipt = validate_v8_artifact(document)
        self.assertEqual(receipt["checks"]["replay"], "passed")
        self.assertEqual(receipt["checks"]["contexts"], "passed")
        self.assertEqual(receipt["checks"]["continuation"], "passed")
        self.assertFalse(receipt["admitted"])


if __name__ == "__main__":
    unittest.main()
