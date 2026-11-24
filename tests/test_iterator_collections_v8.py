"""Independent iterator replay for bounded Set/Map cursors."""

import unittest

from tests.test_proxy_v8 import REF, initial_state, step


class CollectionIteratorV8Tests(unittest.TestCase):
    def test_set_iterator_observes_delete_and_readdition_after_creation(self):
        state = initial_state()
        step(state, {"kind": "set.create", "name": "set", "iterable": ["undefined"]})
        step(
            state,
            {
                "kind": "iterator.lookup",
                "receiver": REF("set"),
                "key": ["str", "values"],
                "method": "Set.prototype.values",
            },
        )
        step(
            state,
            {
                "kind": "iterator.create",
                "name": "iterator",
                "iterated": REF("set"),
                "iterationKind": "set-values",
            },
        )
        step(
            state,
            {
                "kind": "set.add",
                "object": "set",
                "value": ["str", "first"],
                "result": REF("set"),
            },
        )
        step(
            state,
            {
                "kind": "set.delete",
                "object": "set",
                "value": ["str", "first"],
                "result": ["bool", True],
            },
        )
        step(
            state,
            {
                "kind": "set.add",
                "object": "set",
                "value": ["str", "second"],
                "result": REF("set"),
            },
        )
        step(
            state,
            {
                "kind": "iterator.next",
                "object": "iterator",
                "name": "result",
                "entry": None,
                "result": ["str", "second"],
                "done": False,
            },
        )
        self.assertEqual(state["heap"][-2][1]["nextIndex"], 2)
        step(
            state,
            {
                "kind": "iterator.next",
                "object": "iterator",
                "name": "end",
                "entry": None,
                "result": ["undefined"],
                "done": True,
            },
        )
        step(
            state,
            {
                "kind": "set.add",
                "object": "set",
                "value": ["str", "late"],
                "result": REF("set"),
            },
        )
        step(
            state,
            {
                "kind": "iterator.next",
                "object": "iterator",
                "name": "stillEnd",
                "entry": None,
                "result": ["undefined"],
                "done": True,
            },
        )

    def test_set_entries_duplicate_values_and_map_entries_track_live_mutation(self):
        state = initial_state()
        step(state, {"kind": "set.create", "name": "set", "iterable": ["undefined"]})
        step(
            state,
            {
                "kind": "set.lookup",
                "object": "set",
                "key": ["str", "add"],
                "method": "Set.prototype.add",
            },
        )
        step(state, {"kind": "set.add", "object": "set", "value": ["num", 4], "result": REF("set")})
        step(
            state,
            {
                "kind": "iterator.create",
                "name": "setIterator",
                "iterated": REF("set"),
                "iterationKind": "set-entries",
            },
        )
        step(
            state,
            {
                "kind": "iterator.next",
                "object": "setIterator",
                "name": "setResult",
                "entry": "setPair",
                "result": REF("setPair"),
                "done": False,
            },
        )
        set_pair = state["heap"][-2][1]
        self.assertEqual(dict(set_pair["properties"])["0"], ["num", 4])
        self.assertEqual(dict(set_pair["properties"])["1"], ["num", 4])

        step(state, {"kind": "map.create", "name": "map", "iterable": ["undefined"]})
        step(
            state,
            {
                "kind": "map.set",
                "object": "map",
                "key": ["str", "x"],
                "value": ["num", 1],
                "result": REF("map"),
            },
        )
        step(
            state,
            {
                "kind": "iterator.create",
                "name": "mapIterator",
                "iterated": REF("map"),
                "iterationKind": "map-entries",
            },
        )
        step(
            state,
            {
                "kind": "map.set",
                "object": "map",
                "key": ["str", "x"],
                "value": ["num", 2],
                "result": REF("map"),
            },
        )
        step(
            state,
            {
                "kind": "map.set",
                "object": "map",
                "key": ["str", "y"],
                "value": ["num", 3],
                "result": REF("map"),
            },
        )
        step(
            state,
            {
                "kind": "iterator.next",
                "object": "mapIterator",
                "name": "mapResult",
                "entry": "mapPair",
                "result": REF("mapPair"),
                "done": False,
            },
        )
        map_pair = state["heap"][-2][1]
        self.assertEqual(dict(map_pair["properties"])["0"], ["str", "x"])
        self.assertEqual(dict(map_pair["properties"])["1"], ["num", 2])
        step(
            state,
            {
                "kind": "iterator.next",
                "object": "mapIterator",
                "name": "mapResult2",
                "entry": "mapPair2",
                "result": REF("mapPair2"),
                "done": False,
            },
        )
        map_pair2 = state["heap"][-2][1]
        self.assertEqual(dict(map_pair2["properties"])["0"], ["str", "y"])
        self.assertEqual(dict(map_pair2["properties"])["1"], ["num", 3])


if __name__ == "__main__":
    unittest.main()
