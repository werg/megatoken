import copy
import unittest

from megatoken.data.detailed_v7 import encode_record
from megatoken.data.detailed_v8 import parse_record as parse_detailed
from megatoken.data.native_v8 import NativeV8Error, _apply_observation, _initial_state


def pending_delivery():
    state = _initial_state(
        {},
        {"inputGraph": {"roots": [], "objects": []}, "invocationId": "root", "entry": "run"},
    )
    state["heap"].append(
        [
            "p1",
            {
                "kind": "promise",
                "properties": [],
                "state": "pending",
                "result": None,
                "adopted": None,
                "external": "e1",
                "reactions": [],
            },
        ]
    )
    state["effects"].append(
        [
            "e1",
            {
                "operation": "natlang.call",
                "invocation": "root",
                "status": "requested",
                "promise": "p1",
            },
        ]
    )
    obligations = {
        "intents": [["root/e1", {"status": "dispatched", "attempts": 1, "result": None}]],
        "callbacks": [],
        "handlers": [],
        "closureQueue": [],
    }
    return state, {"root/e1"}, set(), obligations


def result_row(result, *, status="fulfilled", graph=None):
    observed = {"effect": "e1", "result": result, "status": status}
    if graph is not None:
        observed["graph"] = graph
    return {
        "kind": "result",
        "sequence": 3,
        "fields": {
            "session": "session1",
            "track": "worker1",
            "invocationId": "root",
            "effectId": "e1",
            "result": result,
            "status": status,
            "line": encode_record("observe", observed),
        },
    }


class NativeObservationBoundaryTests(unittest.TestCase):
    def apply(self, holders, row):
        state, dispatched, settled, obligations = holders
        _apply_observation(state, row, dispatched, settled, obligations, {3: 1})

    def assert_refused_without_mutation(self, holders, row, message):
        before = copy.deepcopy(holders)
        with self.assertRaisesRegex(NativeV8Error, message):
            self.apply(holders, row)
        self.assertEqual(holders, before)

    def test_failed_promise_resolution_leaves_receipts_and_effect_outstanding(self):
        for behavior in ("promise", "own", "inherited", "accessor", "unknown-prototype"):
            with self.subTest(behavior=behavior):
                holders = pending_delivery()
                state = holders[0]
                value = {"kind": "object", "properties": []}
                state["heap"].extend(
                    [
                        ["o1", value],
                        [
                            "f1",
                            {
                                "kind": "closure",
                                "code": "callback",
                                "captures": [],
                                "properties": [],
                            },
                        ],
                    ]
                )
                if behavior == "promise":
                    value.update(kind="promise", state="pending", adopted=None, external=None)
                elif behavior == "unknown-prototype":
                    value["prototype"] = "unmodeled.prototype"
                elif behavior == "accessor":
                    value["descriptors"] = [
                        [
                            "then",
                            {
                                "kind": "accessor",
                                "get": ["ref", "f1"],
                                "set": ["undefined"],
                                "enumerable": True,
                                "configurable": True,
                            },
                        ]
                    ]
                elif behavior == "inherited":
                    value["prototype"] = ["ref", "o2"]
                    state["heap"].append(
                        ["o2", {"kind": "object", "properties": [["then", ["ref", "f1"]]]}]
                    )
                else:
                    value["properties"].append(["then", ["ref", "f1"]])
                self.assert_refused_without_mutation(
                    holders,
                    result_row(["ref", "o1"]),
                    "assimilation jobs|resolution protocol|unmodeled prototype",
                )

    def test_late_obligation_failure_does_not_consume_dispatch_or_result(self):
        holders = pending_delivery()
        holders[1].clear()
        self.assert_refused_without_mutation(
            holders,
            {
                "kind": "dispatch",
                "fields": {
                    "invocationId": "root",
                    "effectId": "e1",
                    "key": "root/e1",
                    "attempt": 1,
                },
            },
            "requested intent",
        )
        holders[1].add("root/e1")
        holders[3]["intents"].clear()
        self.assert_refused_without_mutation(
            holders, result_row(["num", 7]), "dispatched obligation"
        )

    def test_failed_handler_close_does_not_revoke_callback(self):
        holders = pending_delivery()
        holders[3]["callbacks"] = [["root/e1/0", {"state": "active", "subscriptionId": "handler1"}]]
        holders[3]["handlers"] = [["handler1", {"state": "active", "queue": ["event1"]}]]
        self.assert_refused_without_mutation(
            holders,
            {"kind": "handler-close", "fields": {"subscriptionId": "handler1", "discarded": []}},
            "queued events",
        )

    def test_fresh_data_graph_and_receipt_commit_together_with_aliases_and_cycles(self):
        graph = {
            "roots": [["ref", "o1"]],
            "objects": [
                {
                    "id": "o1",
                    "kind": "object",
                    "prototype": None,
                    "properties": [
                        {"key": "self", "value": ["ref", "o1"]},
                        {"key": "alias", "value": ["ref", "o2"]},
                        {"key": "then", "value": ["ref", "o2"]},
                    ],
                },
                {
                    "id": "o2",
                    "kind": "array",
                    "properties": [
                        {"key": "0", "value": ["ref", "o1"]},
                        {"key": "length", "value": ["num", 1]},
                    ],
                },
            ],
        }
        for status in ("fulfilled", "rejected"):
            with self.subTest(status=status):
                holders = pending_delivery()
                row = result_row(["ref", "o1"], status=status, graph=graph)
                self.apply(holders, row)
                state, dispatched, settled, obligations = holders
                heap = dict(state["heap"])
                self.assertEqual(heap["p1"]["state"], status)
                self.assertEqual(heap["p1"]["result"], ["ref", "o1"])
                self.assertEqual(
                    heap["o1"]["properties"],
                    [[p["key"], p["value"]] for p in graph["objects"][0]["properties"]],
                )
                self.assertEqual(heap["o2"]["properties"][0], ["0", ["ref", "o1"]])
                self.assertIsNone(heap["o1"]["prototype"])
                self.assertEqual(dispatched, {"root/e1"})
                self.assertEqual(settled, {"root/e1"})
                intent = dict(obligations["intents"])["root/e1"]
                self.assertEqual(intent["status"], "settled")
                self.assertEqual(intent["result"]["ownerSequence"], 1)
                self.assertEqual(
                    parse_detailed(intent["result"]["line"], role="observed")[1]["graph"],
                    graph,
                )
                self.assert_refused_without_mutation(holders, row, "outstanding dispatch")

    def test_graph_collision_rolls_back_earlier_fresh_nodes(self):
        graph = {
            "roots": [["ref", "o1"]],
            "objects": [
                {"id": "o1", "kind": "object", "properties": []},
                {"id": "p1", "kind": "object", "properties": []},
            ],
        }
        self.assert_refused_without_mutation(
            pending_delivery(), result_row(["ref", "o1"], graph=graph), "duplicate or reserved"
        )

    def test_graph_requires_declared_result_codec(self):
        holders = pending_delivery()
        holders[0]["effects"][0][1]["operation"] = "demo.withCallback"
        graph = {
            "roots": [["ref", "o1"]],
            "objects": [{"id": "o1", "kind": "object", "properties": []}],
        }
        self.assert_refused_without_mutation(
            holders, result_row(["ref", "o1"], graph=graph), "declared owned result codec"
        )

    def test_optional_obligation_ledger_preserves_atomicity(self):
        state, dispatched, settled, _ = pending_delivery()
        holders = state, dispatched, settled, None
        self.assert_refused_without_mutation(
            holders, result_row(["ref", "missing"]), "unknown identity"
        )
        self.apply(holders, result_row(["num", 7]))
        self.assertEqual(dict(state["heap"])["p1"]["result"], ["num", 7])
        self.assertEqual(settled, {"root/e1"})


if __name__ == "__main__":
    unittest.main()
