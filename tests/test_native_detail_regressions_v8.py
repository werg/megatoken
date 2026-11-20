"""Replay retained producer cases and reject false state/transition claims."""

import base64
import copy
import gzip
import json
import unittest
from pathlib import Path

from megatoken.data.native_v8 import (
    NativeV8Error,
    _apply_detail,
    _initial_state,
    validate_v8_artifact,
)


def invocation(roots, bindings=None):
    return {
        "entry": "run",
        "invocationId": "invocation1",
        "inputGraph": {"roots": roots, "objects": []},
        "bindings": bindings or {},
    }


class NativeV8DetailedRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture = Path(__file__).parent / "fixtures" / "semantic-details-v8.json.gz.b64"
        cls.examples = json.loads(gzip.decompress(base64.b64decode(fixture.read_bytes())))["cases"]

    def test_retained_defaults_decisions_captures_and_seeded_random_replay(self):
        for example in self.examples:
            with self.subTest(example=example["id"]):
                receipt = validate_v8_artifact(example)
                self.assertEqual(receipt["checks"]["replay"], "passed")
                self.assertEqual(receipt["checks"]["contexts"], "passed")
                self.assertEqual(receipt["checks"]["continuation"], "passed")
                self.assertFalse(receipt["admitted"])

    def test_initial_cells_follow_named_parameters_and_explicit_bindings(self):
        functions = [{"name": "run", "params": [None, "y", "z"]}]
        state = _initial_state({}, invocation([["num", 7], ["num", 8]]), functions)
        self.assertEqual(state["locals"], [["a0b0", ["num", 8]], ["a0b1", ["undefined"]]])
        self.assertEqual(state["captureAliases"], [["y", "a0b0"], ["z", "a0b1"]])
        state = _initial_state(
            {}, invocation([["num", 7], ["num", 8]], {"extra": 0, "y": 0}), functions
        )
        self.assertEqual(
            state["locals"],
            [["a0b0", ["num", 7]], ["a0b1", ["undefined"]], ["a0b2", ["num", 7]]],
        )
        self.assertEqual(state["captureAliases"], [["y", "a0b0"], ["z", "a0b1"], ["extra", "a0b2"]])

    def test_missing_duplicate_and_extra_arguments_do_not_invent_cells(self):
        state = _initial_state({}, invocation([]), [{"name": "run", "params": ["x"]}])
        self.assertEqual(state["locals"], [["a0b0", ["undefined"]]])
        for params, expected in ((["x"], 1), (["x", "x"], 2)):
            state = _initial_state(
                {}, invocation([["num", 1], ["num", 2]]), [{"name": "run", "params": params}]
            )
            self.assertEqual(state["locals"], [["a0b0", ["num", expected]]])
            self.assertEqual(state["captureAliases"], [["x", "a0b0"]])

    def test_unobserved_bindings_and_invalid_parameter_metadata_are_rejected(self):
        for index in (-1, 1, True, "0"):
            with (
                self.subTest(index=index),
                self.assertRaisesRegex(NativeV8Error, "observed argument"),
            ):
                _initial_state({}, invocation([["num", 1]], {"x": index}))
        for params in ([False], [1], [""], {"x": 0}):
            with self.subTest(params=params), self.assertRaisesRegex(NativeV8Error, "parameter"):
                _initial_state({}, invocation([]), [{"name": "run", "params": params}])

    def test_observed_null_prototype_and_zero_random_seed_are_preserved(self):
        entry = invocation([["ref", "o0"]])
        entry["inputGraph"]["objects"] = [
            {"id": "o0", "kind": "object", "prototype": None, "properties": []}
        ]
        state = _initial_state({}, entry, [{"name": "run", "params": ["x"]}], random_seed=0)
        self.assertIsNone(state["heap"][0][1]["prototype"])
        self.assertEqual(state["randomState"], 0)
        for seed in (-1, 2**32, True, 0.5):
            with self.subTest(seed=seed), self.assertRaisesRegex(NativeV8Error, "seed"):
                _initial_state({}, entry, random_seed=seed)

    def test_closure_capture_uses_anchored_cell_and_rejects_dangling_capture(self):
        state = _initial_state({}, invocation([["num", 7]]), [{"name": "run", "params": ["x"]}])
        fields = {
            "kind": "closure",
            "name": "o0",
            "properties": [],
            "code": "function:1",
            "captures": [{"binding": "x", "cell": "x"}],
        }
        _apply_detail(state, "allocate", fields, {"sequence": 1})
        self.assertEqual(state["heap"][0][1]["captures"], [{"binding": "x", "cell": "a0b0"}])
        self.assertEqual(fields["captures"], [{"binding": "x", "cell": "x"}])
        before = copy.deepcopy(state)
        bad = {**fields, "name": "o1", "captures": [{"binding": "x", "cell": "missing"}]}
        with self.assertRaisesRegex(NativeV8Error, "cell not yet committed"):
            _apply_detail(state, "allocate", bad, {"sequence": 2})
        self.assertEqual(state, before)

    def test_literal_lazy_witness_is_checked_without_inventing_an_operator(self):
        state = {"locals": [], "captureAliases": [], "heap": []}
        for value, path in ((["bool", True], "use-left"), (["bool", False], "evaluate-right")):
            fields = {"expr": {"op": "literal", "value": value}, "decision": value, "path": path}
            _apply_detail(state, "choose", fields, {"sequence": 1})
            fields["decision"] = ["bool", not value[1]]
            with self.assertRaisesRegex(NativeV8Error, "expression claim"):
                _apply_detail(state, "choose", fields, {"sequence": 1})

    def test_random_claims_advance_once_and_false_claims_leave_state_unchanged(self):
        state = {"locals": [], "captureAliases": [], "heap": [], "randomState": 42}
        for expected_seed in (1083814273, 378494188):
            fields = {"expr": {"op": "random"}, "result": ["num", expected_seed / 4294967296]}
            _apply_detail(state, "compute", fields, {"sequence": 1})
            self.assertEqual(state["randomState"], expected_seed)
        before = copy.deepcopy(state)
        with self.assertRaisesRegex(NativeV8Error, "expression claim"):
            _apply_detail(state, "compute", {"expr": {"op": "random"}, "result": ["num", 0]}, {})
        self.assertEqual(state, before)
        for seed in (None, -1, True, 2**32):
            with (
                self.subTest(seed=seed),
                self.assertRaisesRegex(NativeV8Error, "committed deterministic seed"),
            ):
                _apply_detail(
                    {**state, "randomState": seed},
                    "compute",
                    {"expr": {"op": "random"}, "result": ["num", 0]},
                    {},
                )


if __name__ == "__main__":
    unittest.main()
