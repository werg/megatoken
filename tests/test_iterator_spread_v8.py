"""Independent verification of bounded Array/String spread snapshots."""
import copy
import unittest

from megatoken.data.protocol_v8 import ProtocolV8Error
from tests.test_proxy_v8 import initial_state, step


class IteratorSpreadV8Tests(unittest.TestCase):
    def _array(self, state, identity, values, length=None):
        step(
            state,
            {
                "kind": "allocate",
                "name": identity,
                "objectKind": "array",
                "length": len(values) if length is None else length,
                "properties": [
                    {"key": ["str", str(index)], "value": value}
                    for index, value in values
                ],
                "code": None,
                "captures": [],
                "functionKind": None,
                "lexicalReceiver": None,
            },
        )

    def test_dense_array_spread_is_derived_and_rejects_forged_snapshot_atomically(self):
        state = initial_state()
        self._array(state, "items", [(0, ["num", 1]), (1, ["str", "x"])])
        step(
            state,
            {
                "kind": "iterator.spread",
                "iterated": ["ref", "items"],
                "values": [["num", 1], ["str", "x"]],
            },
        )
        before = copy.deepcopy(state)
        with self.assertRaises(ProtocolV8Error):
            step(
                state,
                {
                    "kind": "iterator.spread",
                    "iterated": ["ref", "items"],
                    "values": [["num", 1], ["str", "forged"]],
                },
            )
        self.assertEqual(state, before)

    def test_string_spread_uses_code_points_and_sparse_arrays_refuse(self):
        state = initial_state()
        text = "A😀\ud800z"
        step(
            state,
            {
                "kind": "iterator.spread",
                "iterated": ["str", text],
                "values": [["str", "A"], ["str", "😀"], ["str", "\ud800"], ["str", "z"]],
            },
        )
        self._array(state, "sparse", [(1, ["num", 1])], length=2)
        before = copy.deepcopy(state)
        with self.assertRaises(ProtocolV8Error):
            step(
                state,
                {
                    "kind": "iterator.spread",
                    "iterated": ["ref", "sparse"],
                    "values": [["undefined"], ["num", 1]],
                },
            )
        self.assertEqual(state, before)


if __name__ == "__main__":
    unittest.main()
