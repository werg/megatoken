"""Independent replay tests for public collection allocation and membership."""
import copy
import unittest

from megatoken.data.protocol_v8 import ProtocolV8Error, validate_protocol_state
from tests.test_proxy_v8 import REF, initial_state, step


class CollectionV8Tests(unittest.TestCase):
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

    def test_set_constructor_deduplicates_same_value_zero_and_retains_order(self):
        state = initial_state()
        self._array(
            state,
            "items",
            [
                (0, ["num", "NaN"]),
                (1, ["num", "-0"]),
                (2, ["num", 0]),
                (3, ["num", "NaN"]),
                (4, ["str", "x"]),
            ],
        )
        step(state, {"kind": "set.create", "name": "selected", "iterable": REF("items")})
        self.assertEqual(
            state["heap"][-1][1]["entries"],
            [["num", "NaN"], ["num", 0], ["str", "x"]],
        )
        step(
            state,
            {
                "kind": "set.lookup",
                "object": "selected",
                "key": ["str", "has"],
                "method": "Set.prototype.has",
            },
        )
        step(
            state,
            {
                "kind": "set.has",
                "object": "selected",
                "value": ["num", "NaN"],
                "result": ["bool", True],
            },
        )
        step(
            state,
            {
                "kind": "set.add",
                "object": "selected",
                "value": ["num", 7],
                "result": REF("selected"),
            },
        )
        self.assertEqual(state["heap"][-1][1]["entries"][-1], ["num", 7])
        validate_protocol_state(state["protocol"], state["heap"], state)

    def test_sparse_iterables_and_false_membership_claims_are_atomic(self):
        state = initial_state()
        self._array(state, "sparse", [(0, ["num", 1])], length=2)
        before = copy.deepcopy(state)
        with self.assertRaisesRegex(ProtocolV8Error, "sparse"):
            step(state, {"kind": "set.create", "name": "set", "iterable": REF("sparse")})
        self.assertEqual(state, before)

        self._array(state, "dense", [(0, ["num", 1])])
        step(state, {"kind": "set.create", "name": "set", "iterable": REF("dense")})
        before = copy.deepcopy(state)
        with self.assertRaisesRegex(ProtocolV8Error, "false Set.has result"):
            step(
                state,
                {
                    "kind": "set.has",
                    "object": "set",
                    "value": ["num", 1],
                    "result": ["bool", False],
                },
            )
        self.assertEqual(state, before)

    def test_retained_allocation_registry_rejects_set_node_forgery(self):
        state = initial_state()
        step(state, {"kind": "set.create", "name": "set", "iterable": ["undefined"]})
        for mutate in (
            lambda forged: forged["protocol"].__setitem__("collections", None),
            lambda forged: forged["protocol"]["collections"]["collections"].clear(),
            lambda forged: forged["protocol"]["collections"]["collections"][0].__setitem__(
                "kind", "map"
            ),
            lambda forged: forged["heap"][-1][1].__setitem__("entries", [["num", 1], ["num", 1]]),
            lambda forged: forged["heap"][-1][1].__setitem__("undeclared", True),
        ):
            forged = copy.deepcopy(state)
            mutate(forged)
            with self.assertRaises(ProtocolV8Error):
                validate_protocol_state(forged["protocol"], forged["heap"], forged)


if __name__ == "__main__":
    unittest.main()
