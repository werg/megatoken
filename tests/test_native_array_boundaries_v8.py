"""Detailed writes preserve JavaScript array length and ordinary string keys."""

import unittest

from megatoken.data.native_v8 import _apply_detail


def array_state():
    return {
        "heap": [
            ["a", {"kind": "array", "properties": [["length", ["num", 1]], ["0", ["num", 7]]]}]
        ]
    }


def write(state, key, value, present=True):
    _apply_detail(
        state, "write", {"object": "a", "key": key, "value": value, "present": present}, {}
    )


def properties(state):
    return dict(state["heap"][0][1]["properties"])


class NativeArrayBoundaryTests(unittest.TestCase):
    def test_deleting_an_absent_or_existing_element_preserves_length(self):
        state = array_state()
        for key in ("3", "0"):
            write(state, key, ["undefined"], present=False)
            self.assertEqual(properties(state)["length"], ["num", 1])
            self.assertNotIn(key, properties(state))

    def test_only_canonical_ecmascript_array_indices_extend_length(self):
        state = array_state()
        for key in ("4294967295", "4294967296", "01", "-0", "²", "١", "1" * 5000):
            with self.subTest(key=key[:20]):
                write(state, key, ["num", 9])
                self.assertEqual(properties(state)[key], ["num", 9])
                self.assertEqual(properties(state)["length"], ["num", 1])
        write(state, "3", ["num", 4])
        self.assertEqual(properties(state)["length"], ["num", 4])

    def test_length_truncation_preserves_non_index_properties(self):
        state = array_state()
        # Start from an independently valid heap so truncation is checked
        # separately from the preceding write classification.
        props = state["heap"][0][1]["properties"]
        props.extend([[key, ["num", 8]] for key in ("3", "4294967295", "²", "01")])
        props[0][1] = ["num", 4]
        write(state, "length", ["num", 0])
        self.assertEqual(
            properties(state),
            {"length": ["num", 0], "4294967295": ["num", 8], "²": ["num", 8], "01": ["num", 8]},
        )


if __name__ == "__main__":
    unittest.main()
