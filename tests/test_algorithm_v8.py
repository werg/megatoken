import unittest

from megatoken.data.algorithm_v8 import (
    AlgorithmV8Error,
    evaluate_algorithm,
    evaluate_term,
    replay_algorithm_line,
    validate_algorithm,
)


def callback_term(mode="map"):
    return {
        "op": "traverse",
        "mode": mode,
        "input": {"op": "input", "id": "a0"},
        "initial": None,
        "callback": {
            "arity": 3,
            "body": {
                "op": "multiply",
                "args": [
                    {"op": "bound", "depth": 0, "index": 0},
                    {"op": "literal", "value": ["num", 2]},
                ],
            },
        },
        "profile": "dense-primitive-array/1",
        "order": "ascending",
    }


def traversal(mode, body, initial=None, input_term=None):
    return {
        "op": "traverse",
        "mode": mode,
        "input": input_term or {"op": "input", "id": "a0"},
        "initial": initial,
        "callback": {"arity": 4 if mode in ("fold", "scan") else 3, "body": body},
        "profile": "dense-primitive-array/1",
        "order": "ascending",
    }


def array_heap(items, identity="a"):
    return {
        identity: {
            "kind": "array",
            "properties": {
                **{str(i): v for i, v in enumerate(items)},
                "length": ["num", len(items)],
            },
        }
    }


def bound(index, depth=0):
    return {"op": "bound", "depth": depth, "index": index}


def lit(value):
    return {"op": "literal", "value": value}


def operation(op, *args):
    return {"op": op, "args": list(args)}


class AlgorithmV8Tests(unittest.TestCase):
    def setUp(self):
        self.heap = {
            "o2": {
                "kind": "array",
                "properties": [
                    ["0", ["num", 2]],
                    ["1", ["num", 3]],
                    ["2", ["num", 4]],
                    ["length", ["num", 3]],
                ],
            }
        }
        self.core = {
            "schema": "semantic-algorithm/2",
            "profile": "ecmascript-pure-regions/1",
            "inputs": [{"id": "a0", "type": "dense-number-array"}],
            "steps": [
                {"kind": "compute", "id": "v0", "term": callback_term()},
                {"kind": "return", "value": {"op": "ref", "id": "v0"}},
            ],
        }

    def test_map_preserves_input_identity_and_allocates_result(self):
        result = evaluate_algorithm(self.core, [["ref", "o2"]], heap=self.heap)
        self.assertEqual(result["completion"], {"kind": "return", "value": ["ref", "v8:0"]})
        self.assertEqual(result["heap"]["o2"]["properties"]["2"], ["num", 4])
        self.assertEqual(result["heap"]["v8:0"]["properties"]["2"], ["num", 8])

    def test_callback_bound_depth_and_index_are_checked(self):
        self.core["steps"][0]["term"]["callback"]["body"]["args"][0] = {
            "op": "bound",
            "depth": 0,
            "index": 3,
        }
        with self.assertRaises(AlgorithmV8Error):
            validate_algorithm(self.core)

    def test_sparse_or_accessor_like_array_data_is_refused(self):
        del self.heap["o2"]["properties"][1]
        with self.assertRaises(ValueError):
            evaluate_algorithm(self.core, [["ref", "o2"]], heap=self.heap)

    def test_old_algorithm_version_is_not_relabelled(self):
        self.core["schema"] = "semantic-algorithm/1"
        with self.assertRaises(AlgorithmV8Error):
            validate_algorithm(self.core)

    def test_traversal_evaluates_allocating_input_once_and_preserves_aliases(self):
        term = {
            "op": "traverse",
            "mode": "map",
            "input": {
                "op": "array",
                "items": [
                    {"op": "literal", "value": ["num", 1]},
                    {"op": "literal", "value": ["num", 2]},
                ],
            },
            "initial": None,
            "callback": {"arity": 3, "body": {"op": "bound", "depth": 0, "index": 2}},
            "profile": "dense-primitive-array/1",
            "order": "ascending",
        }
        result, heap, _ = evaluate_term(term, allocator={"prefix": "v8", "next": 0})
        self.assertEqual(result, ["ref", "v8:1"])
        self.assertEqual(heap["v8:1"]["properties"]["0"], ["ref", "v8:0"])
        self.assertEqual(heap["v8:1"]["properties"]["1"], ["ref", "v8:0"])

    def test_object_allocation_uses_ecmascript_integer_key_order(self):
        term = {
            "op": "object",
            "properties": [
                {"key": "z", "value": {"op": "literal", "value": ["num", 0]}},
                {"key": "10", "value": {"op": "literal", "value": ["num", 10]}},
                {"key": "2", "value": {"op": "literal", "value": ["num", 2]}},
            ],
        }
        result, heap, _ = evaluate_term(term, allocator={"prefix": "v8", "next": 0})
        self.assertEqual(result, ["ref", "v8:0"])
        self.assertEqual(list(heap["v8:0"]["properties"]), ["2", "10", "z"])

    def test_work_budget_is_shared_across_callback_iterations(self):
        term = callback_term()
        with self.assertRaises(AlgorithmV8Error):
            evaluate_term(
                term, inputs={"a0": ["ref", "o2"]}, heap=self.heap, budget={"remaining": 6}
            )

    def test_generated_claims_and_allocations_are_replayed(self):
        line = 'algorithm {"allocations":[{"id":"o0","kind":"array","properties":[{"key":"0","value":["num",4]},{"key":"1","value":["num",6]},{"key":"2","value":["num",8]},{"key":"length","value":["num",3]}]}],"claims":[{"id":"v0","kind":"compute","result":["ref","o0"],"step":0},{"kind":"return","step":1,"value":["ref","o0"]}],"completion":{"kind":"return","value":["ref","o0"]}}'
        graph = {
            "roots": [["ref", "o2"]],
            "objects": [
                {
                    "id": "o2",
                    "kind": "array",
                    "properties": [
                        {"key": "0", "value": ["num", 2]},
                        {"key": "1", "value": ["num", 3]},
                        {"key": "2", "value": ["num", 4]},
                        {"key": "length", "value": ["num", 3]},
                    ],
                }
            ],
        }
        self.assertEqual(
            replay_algorithm_line(self.core, graph, line)["completion"]["value"], ["ref", "o0"]
        )
        forged = line.replace('["num",8]', '["num",9]')
        with self.assertRaises(AlgorithmV8Error):
            replay_algorithm_line(self.core, graph, forged)

    def test_all_traversal_modes_match_ordered_ecmascript_behavior(self):
        values = [2, 3, 4]
        heap = array_heap([["num", v] for v in values], "xs")
        source = {"op": "input", "id": "xs"}
        greater = operation("greater", bound(0), lit(["num", 2]))
        cases = {
            "fold": (operation("subtract", bound(0), bound(1)), lit(["num", 10]), ["num", 1]),
            "map": (operation("multiply", bound(0), lit(["num", 2])), None, ["ref", "v8:0"]),
            "filter": (greater, None, ["ref", "v8:0"]),
            "find": (greater, None, ["num", 3]),
            "find_index": (greater, None, ["num", 1]),
            "some": (greater, None, ["bool", True]),
            "every": (operation("greater", bound(0), lit(["num", 1])), None, ["bool", True]),
            "scan": (operation("add", bound(0), bound(1)), lit(["num", 0]), ["ref", "v8:0"]),
        }
        for mode, (body, initial, expected) in cases.items():
            with self.subTest(mode=mode):
                result, allocated, _ = evaluate_term(
                    traversal(mode, body, initial, source), inputs={"xs": ["ref", "xs"]}, heap=heap
                )
                self.assertEqual(result, expected)
                if mode == "map":
                    self.assertEqual(
                        [allocated["v8:0"]["properties"][str(i)] for i in range(3)],
                        [["num", 4], ["num", 6], ["num", 8]],
                    )
                if mode == "filter":
                    self.assertEqual(allocated["v8:0"]["properties"]["1"], ["num", 4])
                if mode == "scan":
                    self.assertEqual(
                        [allocated["v8:0"]["properties"][str(i)] for i in range(3)],
                        [["num", 2], ["num", 5], ["num", 9]],
                    )
        product = traversal(
            "fold", operation("multiply", bound(0), bound(1)), lit(["num", 1]), source
        )
        self.assertEqual(
            evaluate_term(product, inputs={"xs": ["ref", "xs"]}, heap=heap)[0], ["num", 24]
        )
        cancellation = traversal(
            "fold", operation("add", bound(0), bound(1)), lit(["num", 0]), source
        )
        cancel_heap = array_heap([["num", 1e16], ["num", 1], ["num", -1e16]], "xs")
        self.assertEqual(
            evaluate_term(cancellation, inputs={"xs": ["ref", "xs"]}, heap=cancel_heap)[0],
            ["num", 0],
        )

    def test_empty_traversals_and_short_circuit_modes(self):
        empty_heap = array_heap([], "xs")
        source = {"op": "input", "id": "xs"}
        for mode, body, initial, expected in [
            ("find", lit(["bool", True]), None, ["undefined"]),
            ("find_index", lit(["bool", True]), None, ["num", -1]),
            ("some", lit(["bool", True]), None, ["bool", False]),
            ("every", lit(["bool", False]), None, ["bool", True]),
            ("fold", bound(0), lit(["num", 9]), ["num", 9]),
        ]:
            result, _, _ = evaluate_term(
                traversal(mode, body, initial, source),
                inputs={"xs": ["ref", "xs"]},
                heap=empty_heap,
            )
            self.assertEqual(result, expected)
        one_heap = array_heap([["num", 1], ["num", 2], ["num", 3]], "xs")
        early = traversal("find", operation("greater", bound(0), lit(["num", 0])), None, source)
        result, _, _ = evaluate_term(
            early, inputs={"xs": ["ref", "xs"]}, heap=one_heap, budget={"remaining": 7}
        )
        self.assertEqual(result, ["num", 1])

    def test_lazy_defaults_select_and_logical_operators_do_not_evaluate_skipped_terms(self):
        bad = {"op": "get", "object": lit(["num", 1]), "key": lit(["str", "missing"])}
        for term, expected in [
            ({"op": "default_undefined", "args": [lit(["null"]), bad]}, ["null"]),
            ({"op": "default_undefined", "args": [lit(["num", 0]), bad]}, ["num", 0]),
            (operation("select", lit(["bool", True]), lit(["str", "yes"]), bad), ["str", "yes"]),
            (operation("and", lit(["bool", False]), bad), ["bool", False]),
            (operation("or", lit(["num", 4]), bad), ["num", 4]),
            (operation("coalesce", lit(["num", 0]), bad), ["num", 0]),
        ]:
            self.assertEqual(evaluate_term(term)[0], expected)

    def test_lexical_let_depth_and_eager_binding_order(self):
        term = {
            "op": "let",
            "bindings": [
                lit(["num", 7]),
                operation("add", bound(0), lit(["num", 1])),
            ],
            "body": {
                "op": "let",
                "bindings": [operation("multiply", bound(0, 1), bound(1, 1))],
                "body": operation("add", bound(0), bound(1, 1)),
            },
        }
        self.assertEqual(evaluate_term(term)[0], ["num", 64])
        for bindings in ([bound(0)], [bound(1), lit(["num", 1])]):
            with (
                self.subTest(bindings=bindings),
                self.assertRaisesRegex(AlgorithmV8Error, "bound reference"),
            ):
                evaluate_term({"op": "let", "bindings": bindings, "body": bound(0)})
        with self.assertRaises(AlgorithmV8Error):
            evaluate_term(
                {"op": "let", "bindings": [{"id": "x", "value": lit(["num", 1])}], "body": bound(0)}
            )
        capture = {
            "op": "let",
            "bindings": [lit(["num", 10])],
            "body": traversal(
                "map", operation("add", bound(0), bound(0, 1)), None, {"op": "input", "id": "a0"}
            ),
        }
        result, heap, _ = evaluate_term(
            capture, inputs={"a0": ["ref", "xs"]}, heap=array_heap([["num", 1], ["num", 2]], "xs")
        )
        self.assertEqual(
            [heap[result[1]]["properties"][str(i)] for i in range(2)], [["num", 11], ["num", 12]]
        )

    def test_supplied_lexical_scopes_are_nearest_first_and_keep_committed_references(self):
        scopes = [[["num", 2]], [["num", 9], ["ref", "xs"]]]
        heap = array_heap([["num", 4]], "xs")
        self.assertEqual(
            evaluate_term(operation("subtract", bound(0, 1), bound(0)), scopes=scopes, heap=heap)[
                0
            ],
            ["num", 7],
        )
        term = {"op": "let", "bindings": [bound(1, 2)], "body": bound(0)}
        self.assertEqual(evaluate_term(term, scopes=scopes, heap=heap)[0], ["ref", "xs"])
        with self.assertRaises(ValueError):
            evaluate_term(bound(1, 1), scopes=scopes)

    def test_number_specials_rounding_and_multi_object_allocation_aliases(self):
        rounded = operation("add", lit(["num", 1e16]), lit(["num", 1]))
        self.assertEqual(evaluate_term(rounded)[0], ["num", 1e16])
        self.assertEqual(
            evaluate_term(operation("divide", lit(["num", 1]), lit(["num", 0])))[0],
            ["num", "+Infinity"],
        )
        self.assertEqual(
            evaluate_term(operation("multiply", lit(["num", "-0"]), lit(["num", 1])))[0],
            ["num", "-0"],
        )
        obj = {
            "op": "object",
            "properties": [
                {"key": "z", "value": bound(0)},
                {"key": "10", "value": bound(0)},
                {"key": "2", "value": bound(0)},
                {"key": "a", "value": bound(0)},
            ],
        }
        term = {
            "op": "let",
            "bindings": [{"op": "array", "items": [lit(["num", 5])]}],
            "body": obj,
        }
        value, heap, _ = evaluate_term(term)
        obj = heap[value[1]]
        self.assertEqual(list(obj["properties"]), ["2", "10", "z", "a"])
        self.assertEqual(obj["properties"]["2"], obj["properties"]["10"])
        self.assertEqual(obj["properties"]["2"], obj["properties"]["z"])
        self.assertEqual(len(heap), 2)

    def test_legacy_numeric_fold_uses_v8_lexical_terms_and_shared_work_budget(self):
        fold = {
            "op": "fold_left",
            "input": bound(0),
            "initial": bound(1),
            "step": "add",
            "order": "ascending",
            "profile": "dense-number-array/1",
        }
        term = {
            "op": "let",
            "bindings": [
                {
                    "op": "array",
                    "items": [lit(["num", 1e16]), lit(["num", 1]), lit(["num", -1e16])],
                },
                lit(["num", 2]),
            ],
            "body": fold,
        }
        result, heap, allocator = evaluate_term(term)
        self.assertEqual(result, ["num", 4])
        self.assertEqual(heap["v8:0"]["properties"]["length"], ["num", 3])
        self.assertEqual(allocator["next"], 1)
        with self.assertRaisesRegex(AlgorithmV8Error, "semantic work budget"):
            evaluate_term(term, budget={"remaining": 8})
        for field, invalid in (
            ("step", "multiply"),
            ("order", "descending"),
            ("profile", "dense-primitive-array/1"),
        ):
            with (
                self.subTest(field=field),
                self.assertRaisesRegex(AlgorithmV8Error, "legacy fold profile"),
            ):
                evaluate_term({**term, "body": {**fold, field: invalid}})

    def test_resource_limits_and_scope_errors_fail_closed(self):
        with self.assertRaises(AlgorithmV8Error):
            evaluate_term(
                operation("add", lit(["num", 1]), lit(["num", 2])), budget={"remaining": 1}
            )
        with self.assertRaises(AlgorithmV8Error):
            evaluate_term(
                traversal("every", bound(3)), inputs={"a0": ["ref", "o2"]}, heap=self.heap
            )

    def test_search_short_circuit_and_object_value_evaluation_order(self):
        bad = operation("divide", lit(["bigint", "1"]), lit(["bigint", "0"]))
        predicate = operation(
            "select", operation("equal", bound(0), lit(["num", 1])), lit(["bool", True]), bad
        )
        array = array_heap([["num", 1], ["num", 2]], "xs")
        for mode in ("some", "find"):
            term = traversal(mode, predicate, None, {"op": "input", "id": "xs"})
            result, _, _ = evaluate_term(term, inputs={"xs": ["ref", "xs"]}, heap=array)
            self.assertEqual(result, ["bool", True] if mode == "some" else ["num", 1])
        with self.assertRaisesRegex(ArithmeticError, "BigInt division by zero"):
            evaluate_term(
                traversal("map", predicate, None, {"op": "input", "id": "xs"}),
                inputs={"xs": ["ref", "xs"]},
                heap=array,
            )
        term = {
            "op": "object",
            "properties": [
                {"key": "2", "value": {"op": "array", "items": [lit(["num", 2])]}},
                {"key": "1", "value": {"op": "array", "items": [lit(["num", 1])]}},
                {"key": "last", "value": lit(["num", 3])},
            ],
        }
        result, heap, _ = evaluate_term(term)
        self.assertEqual(list(heap[result[1]]["properties"]), ["1", "2", "last"])
        self.assertEqual(heap[result[1]]["properties"]["1"], ["ref", "v8:1"])
        self.assertEqual(heap[result[1]]["properties"]["2"], ["ref", "v8:0"])


if __name__ == "__main__":
    unittest.main()
