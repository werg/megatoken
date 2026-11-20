"""The shared detailed and compact paths use the same primitive semantics."""

import json
import unittest

from megatoken.data.algorithm_v7 import stable_algorithm_hash
from megatoken.data.native_v8 import (
    NativeV8Error,
    _apply_detail,
    _apply_region,
    _eval_detail_expression,
    _replay_region,
)


def literal(value):
    return {"op": "literal", "value": value}


class NativeV8PrimitiveTests(unittest.TestCase):
    def replay_region(self, term, result, decisions=(), state=None):
        state = state or {"locals": [], "captureAliases": []}
        definition = {
            "id": "r1",
            "completion": "return",
            "operations": [{"kind": "return", "term": term}],
        }
        definition["digest"] = stable_algorithm_hash(definition)
        program = {"regions": [definition]}
        program["definitionDigest"] = stable_algorithm_hash(program)
        context = {"state": state, "visibleThrough": 2}
        entry = {
            "definition": program["definitionDigest"],
            "region": "r1",
            "regionDigest": definition["digest"],
            "visibleThrough": 2,
            "stateDigest": stable_algorithm_hash(state),
        }
        target = {
            "region": "r1",
            "bindings": [],
            "decisions": list(decisions),
            "completion": {"kind": "return", "value": result},
        }
        return _replay_region(program, context, "region " + json.dumps(target), entry)

    def test_compact_selection_uses_ecmascript_falsy_values(self):
        for value in (["num", "NaN"], ["num", "-0"], ["bigint", "0"], ["undefined"]):
            with self.subTest(value=value):
                term = {
                    "op": "select",
                    "test": literal(value),
                    "consequent": literal(["num", 1]),
                    "alternate": literal(["num", 2]),
                }
                self.replay_region(term, ["num", 2], [{"kind": "select", "selected": "alternate"}])
                with self.assertRaises(NativeV8Error):
                    self.replay_region(
                        term, ["num", 1], [{"kind": "select", "selected": "consequent"}]
                    )

    def test_compact_logical_and_skips_right_operand_for_zero_bigint(self):
        term = {
            "op": "logical",
            "operator": "&&",
            "left": literal(["bigint", "0"]),
            "right": {
                "op": "binary",
                "operator": "/",
                "left": literal(["bigint", "1"]),
                "right": literal(["bigint", "0"]),
            },
        }
        self.replay_region(term, ["bigint", "0"], [{"kind": "&&", "selected": "left"}])

    def test_compact_unary_operators_do_not_use_binary_arity(self):
        for operator, value, expected in (
            ("-", ["num", 3], ["num", -3]),
            ("+", ["str", "3"], ["num", 3]),
            ("-", ["bigint", "3"], ["bigint", "-3"]),
        ):
            with self.subTest(operator=operator, value=value):
                self.replay_region(
                    {"op": "unary", "operator": operator, "arg": literal(value)}, expected
                )

    def test_detailed_branch_treats_zero_bigint_as_false(self):
        state = {"locals": [], "captureAliases": [], "heap": []}
        fields = {"expr": literal(["bigint", "0"]), "decision": ["bigint", "0"], "path": "else"}
        _apply_detail(state, "choose", fields, {"sequence": 1})
        fields["path"] = "then"
        with self.assertRaisesRegex(NativeV8Error, "Branch decision"):
            _apply_detail(state, "choose", fields, {"sequence": 1})

    def test_compact_binding_updates_captured_cell(self):
        state = {"locals": [["cell", ["num", 1]]], "captureAliases": [["x", "cell"]]}
        _apply_region(state, {"bindings": [["x", ["num", 7]]], "completion": {"kind": "normal"}})
        self.assertEqual(state["locals"], [["cell", ["num", 7]]])
        self.assertEqual(_eval_detail_expression({"op": "local", "name": "x"}, state), ["num", 7])
        self.replay_region({"op": "local", "name": "x"}, ["num", 7], state=state)

    def test_compact_select_and_logical_paths_treat_owned_symbols_as_truthy(self):
        state = {"locals": [["s", ["symbol", "s1"]]], "captureAliases": []}
        local = {"op": "local", "name": "s"}
        self.replay_region(
            {
                "op": "select",
                "test": local,
                "consequent": literal(["num", 1]),
                "alternate": literal(["num", 2]),
            },
            ["num", 1],
            [{"kind": "select", "selected": "consequent"}],
            state=state,
        )
        for operator, expected, selected in (
            ("&&", ["num", 7], "right"),
            ("||", ["symbol", "s1"], "left"),
        ):
            self.replay_region(
                {
                    "op": "logical",
                    "operator": operator,
                    "left": local,
                    "right": literal(["num", 7]),
                },
                expected,
                [{"kind": operator, "selected": selected}],
                state=state,
            )

    def test_detailed_lazy_path_uses_left_operand(self):
        state = {"locals": [], "captureAliases": [], "heap": []}
        for op, left, right, result, path in (
            ("and", ["bool", True], ["bool", False], ["bool", False], "evaluate-right"),
            ("or", ["bool", False], ["bool", True], ["bool", True], "evaluate-right"),
            ("coalesce", ["null"], ["num", 0], ["num", 0], "evaluate-right"),
            ("and", ["num", 0], ["num", 7], ["num", 0], "use-left"),
            ("or", ["num", 7], ["num", 0], ["num", 7], "use-left"),
        ):
            fields = {
                "expr": {"op": op, "args": [literal(left), literal(right)]},
                "decision": result,
                "path": path,
            }
            _apply_detail(state, "choose", fields, {"sequence": 1})
            fields["path"] = "use-left" if path == "evaluate-right" else "evaluate-right"
            with self.assertRaisesRegex(NativeV8Error, "Logical decision path"):
                _apply_detail(state, "choose", fields, {"sequence": 1})


if __name__ == "__main__":
    unittest.main()
