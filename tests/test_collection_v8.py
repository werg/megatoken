"""Independent replay tests for public Set/Map construction and mutations."""
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

    def test_set_constructor_same_value_zero_order_delete_and_readd(self):
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
        step(state, {"kind": "set.size", "object": "selected", "result": ["num", 3]})
        step(
            state,
            {
                "kind": "set.delete",
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
                "value": ["num", "NaN"],
                "result": REF("selected"),
            },
        )
        self.assertEqual(
            state["heap"][-1][1]["entries"],
            [None, ["num", 0], ["str", "x"], ["num", "NaN"]],
        )
        step(
            state,
            {
                "kind": "set.clear",
                "object": "selected",
                "result": ["undefined"],
            },
        )
        self.assertEqual(state["heap"][-1][1]["entries"], [None, None, None, None])
        validate_protocol_state(state["protocol"], state["heap"], state)

    def test_map_constructor_updates_duplicate_keys_without_reordering_then_mutates(self):
        state = initial_state()
        self._array(state, "pairA", [(0, ["num", 1]), (1, ["str", "first"])])
        self._array(state, "pairB", [(0, ["num", 1]), (1, ["str", "updated"])])
        self._array(state, "pairC", [(0, ["str", "x"]), (1, ["num", "NaN"])])
        self._array(state, "items", [(0, REF("pairA")), (1, REF("pairB")), (2, REF("pairC"))])
        step(state, {"kind": "map.create", "name": "selected", "iterable": REF("items")})
        node = state["heap"][-1][1]
        self.assertEqual(node["entries"], [[["num", 1], ["str", "updated"]], [["str", "x"], ["num", "NaN"]]])
        step(state, {"kind": "map.size", "object": "selected", "result": ["num", 2]})
        step(
            state,
            {
                "kind": "map.get",
                "object": "selected",
                "key": ["num", 1],
                "result": ["str", "updated"],
            },
        )
        step(
            state,
            {
                "kind": "map.set",
                "object": "selected",
                "key": ["num", 1],
                "value": ["str", "latest"],
                "result": REF("selected"),
            },
        )
        step(
            state,
            {
                "kind": "map.delete",
                "object": "selected",
                "key": ["str", "x"],
                "result": ["bool", True],
            },
        )
        step(
            state,
            {
                "kind": "map.set",
                "object": "selected",
                "key": ["str", "x"],
                "value": ["num", 5],
                "result": REF("selected"),
            },
        )
        self.assertEqual(
            state["heap"][-1][1]["entries"],
            [[["num", 1], ["str", "latest"]], None, [["str", "x"], ["num", 5]]],
        )
        step(state, {"kind": "map.clear", "object": "selected", "result": ["undefined"]})
        self.assertEqual(state["heap"][-1][1]["entries"], [None, None, None])
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
        with self.assertRaisesRegex(ProtocolV8Error, "false collection membership"):
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

    def test_map_rejects_bad_constructor_pair_and_false_get_result_transactionally(self):
        state = initial_state()
        self._array(state, "badPair", [(0, ["num", 1])])
        self._array(state, "items", [(0, REF("badPair"))])
        before = copy.deepcopy(state)
        with self.assertRaisesRegex(ProtocolV8Error, "exactly two"):
            step(state, {"kind": "map.create", "name": "map", "iterable": REF("items")})
        self.assertEqual(state, before)

        self._array(state, "pair", [(0, ["num", 1]), (1, ["str", "one"])])
        self._array(state, "dense", [(0, REF("pair"))])
        step(state, {"kind": "map.create", "name": "map", "iterable": REF("dense")})
        before = copy.deepcopy(state)
        with self.assertRaisesRegex(ProtocolV8Error, "false Map.get"):
            step(
                state,
                {
                    "kind": "map.get",
                    "object": "map",
                    "key": ["num", 1],
                    "result": ["str", "wrong"],
                },
            )
        self.assertEqual(state, before)

    def test_retained_allocation_registry_rejects_collection_node_forgery(self):
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
