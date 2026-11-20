import base64
import copy
import gzip
import json
import unittest
from pathlib import Path

from megatoken.controls import semantic_selection_instruction_v8
from megatoken.data.algorithm_v7 import stable_algorithm_hash
from megatoken.data.prefix_v8 import validate_v8_prediction_prefix

FIXTURES = Path(__file__).parent / "fixtures"


def examples(name):
    raw = (FIXTURES / name).read_bytes()
    if name.endswith(".gz.b64"):
        raw = gzip.decompress(base64.b64decode(raw))
    document = json.loads(raw)
    return [
        item.get("artifact", item)
        for item in document.get("examples", document.get("cases", [document]))
    ]


def prefix(example, index):
    context = example["contexts"][index]
    public = {
        **example["public"],
        "timeline": example["public"]["timeline"][: context["visibleThrough"]],
    }
    return copy.deepcopy((public, example["contexts"][:index], context))


class PredictionPrefixV8Tests(unittest.TestCase):
    def test_retained_prediction_boundaries_need_no_future_target(self):
        for name in (
            "semantic-regions-v8.json.gz.b64",
            "semantic-callback-regions-v8.json.gz.b64",
            "semantic-protocol-region-v8.json.gz.b64",
            "semantic-generator-close-v8.json.gz.b64",
        ):
            for example in examples(name):
                count = len(example["contexts"])
                indices = (
                    range(count)
                    if "callback" in name
                    else sorted({0, 1, count // 2, count - 1} & set(range(count)))
                )
                for index in indices:
                    with self.subTest(fixture=name, case=example.get("id"), boundary=index):
                        public, contexts, context = prefix(example, index)
                        checked = validate_v8_prediction_prefix(public, contexts, context)
                        self.assertEqual(checked["context"], context)
                        self.assertEqual(checked["publicDigest"], stable_algorithm_hash(public))

    def test_context_mutations_are_not_authority(self):
        example = examples("semantic-regions-v8.json.gz.b64")[1]
        for field in ("state", "input", "programs", "regionCursor", "obligations"):
            public, contexts, context = prefix(example, 2)
            if field == "programs":
                context[field] = []
            else:
                context[field] = {**context[field], "futureAnswer": 42}
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "context differs"):
                validate_v8_prediction_prefix(public, contexts, context)

    def test_future_or_missing_context_and_extra_observations_are_rejected(self):
        example = examples("semantic-regions-v8.json.gz.b64")[1]
        public, contexts, context = prefix(example, 2)
        with self.assertRaises(ValueError):
            validate_v8_prediction_prefix(public, example["contexts"], context)
        with self.assertRaises(ValueError):
            validate_v8_prediction_prefix(public, [], context)
        public["timeline"][0]["fields"]["expectedResult"] = 42
        context["prefixDigest"] = stable_algorithm_hash(public["timeline"])
        with self.assertRaisesRegex(ValueError, "unexpected fields"):
            validate_v8_prediction_prefix(public, contexts, context)

    def test_v8_training_and_serving_instruction_names_its_own_contract(self):
        first = semantic_selection_instruction_v8()
        continued = semantic_selection_instruction_v8(continuation=True)
        self.assertIn("semantic-trace/8", first)
        self.assertNotIn("semantic-trace/6", first)
        self.assertTrue(continued.startswith(first))
        self.assertIn("Continue the exact current record bytes", continued)


if __name__ == "__main__":
    unittest.main()
