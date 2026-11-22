import copy
import unittest

from megatoken.data.promise_join_v8 import validate_state
from megatoken.data.protocol_v8 import (
    ProtocolV8Error,
    _external_machine,
    apply_protocol_record,
    assert_detailed_record_compatible,
)
from tests.test_number_v8 import machine


def joined_machine(empty=False):
    state = machine()
    state["heap"].update(
        {
            "input": {
                "kind": "promise",
                "properties": {},
                "state": "fulfilled",
                "result": ["num", 7],
                "reactions": [],
            },
            "result": {
                "kind": "promise",
                "properties": {},
                "state": "pending",
                "result": None,
                "reactions": [],
            },
        }
    )
    state.update(jobs=[], ready=[], scheduler={"registration": 0, "ready": 0})
    row = {
        "kind": "promise.join",
        "mode": "all",
        "result": "result",
        "inputs": [] if empty else ["input"],
        "jobs": [] if empty else ["job"],
        "output": "output" if empty else None,
        "error": None,
    }
    return state, row


def finish_row(output="output"):
    return {
        "kind": "promise.join.step",
        "result": "result",
        "job": "job",
        "entry": None,
        "output": output,
        "error": None,
        "status": "fulfilled",
        "value": ["ref", output],
    }


class PromiseJoinBoundaryTests(unittest.TestCase):
    def test_all_settled_keeps_rejections_pending_and_duplicate_entries_distinct(self):
        state, row = joined_machine()
        state["heap"]["later"] = {
            "kind": "promise",
            "properties": {},
            "state": "pending",
            "result": None,
            "reactions": [],
        }
        state["heap"]["input"].update(state="rejected", result=["str", "no"])
        row.update(mode="allSettled", inputs=["later", "input", "input"], jobs=["j0", "j1", "j2"])
        apply_protocol_record(state, row)
        state["stack"] = []
        for index in (1, 2):
            apply_protocol_record(
                state,
                {
                    **finish_row(None),
                    "job": "j" + str(index),
                    "entry": "entry" + str(index),
                    "status": "pending",
                    "value": None,
                },
            )
            self.assertTrue(validate_state(state))
            state = copy.deepcopy(state)
        self.assertEqual(state["heap"]["result"]["state"], "pending")
        self.assertEqual(
            state["heap"]["entry1"]["properties"],
            {
                "status": ["str", "rejected"],
                "reason": ["str", "no"],
            },
        )
        from megatoken.data.promise_scheduler_v8 import settle_promise

        settle_promise(state, "later", "fulfilled", ["undefined"])
        apply_protocol_record(state, {**finish_row(), "job": "j0", "entry": "entry0"})
        self.assertEqual(
            state["heap"]["output"]["properties"],
            {
                "0": ["ref", "entry0"],
                "1": ["ref", "entry1"],
                "2": ["ref", "entry2"],
                "length": ["num", 3],
            },
        )
        self.assertTrue(validate_state(state))
        state["heap"]["entry1"]["properties"]["status"] = ["str", "changed"]
        state["heap"]["output"]["properties"]["length"] = ["num", 0]
        self.assertTrue(validate_state(state))

    def test_all_settled_invalid_entry_or_settlement_is_atomic(self):
        state, row = joined_machine()
        state["heap"]["later"] = {
            "kind": "promise",
            "properties": {},
            "state": "pending",
            "result": None,
            "reactions": [],
        }
        row.update(mode="allSettled", inputs=["input", "later"], jobs=["job", "later-job"])
        apply_protocol_record(state, row)
        state["stack"] = []
        step = {**finish_row(None), "entry": "entry", "status": "pending", "value": None}
        for change in (
            {"entry": None},
            {"entry": "later-job"},
            {"output": "output", "status": "fulfilled", "value": ["ref", "output"]},
            {"status": "rejected", "value": ["num", 7]},
        ):
            before = copy.deepcopy(state)
            with self.subTest(change=change), self.assertRaises(ProtocolV8Error):
                apply_protocol_record(state, {**step, **change})
            self.assertEqual(state, before)
        apply_protocol_record(state, step)
        changed = copy.deepcopy(state)
        changed["heap"]["entry"]["properties"]["value"] = ["num", 8]
        with self.assertRaisesRegex(ProtocolV8Error, "pending entry"):
            validate_state(changed)
        changed = copy.deepcopy(state)
        changed["heap"]["result"].update(state="fulfilled", result=["ref", "entry"])
        changed["heap"]["result"]["join"]["settled"] = True
        with self.assertRaisesRegex(ProtocolV8Error, "allSettled"):
            validate_state(changed)

    def test_all_settled_empty_result_and_bound(self):
        state, row = joined_machine(empty=True)
        row["mode"] = "allSettled"
        apply_protocol_record(state, row)
        self.assertEqual(state["heap"]["result"]["result"], ["ref", "output"])
        self.assertTrue(validate_state(state))
        state["heap"]["output"]["properties"].update({"0": ["num", 7], "length": ["num", 1]})
        self.assertTrue(validate_state(state))
        state, row = joined_machine()
        row.update(
            mode="allSettled", inputs=["input"] * 4096, jobs=["job" + str(i) for i in range(4096)]
        )
        with self.assertRaisesRegex(ProtocolV8Error, "array length"):
            apply_protocol_record(state, row)

    def test_shared_join_heap_retains_controller_ownership_after_root_return(self):
        state, row = joined_machine()
        apply_protocol_record(state, row)
        self.assertEqual(state["heap"]["result"]["join"]["owner"], "root")
        state["stack"] = []
        self.assertTrue(validate_state(state))

        callback = copy.deepcopy(state)
        callback.update(
            stack=["callback"], frameFunctions={"callback": "handler"}, jobs=[], ready=[]
        )
        self.assertTrue(validate_state(callback))
        apply_protocol_record(state, finish_row())
        callback["heap"] = copy.deepcopy(state["heap"])
        self.assertTrue(validate_state(callback))
        self.assertEqual(callback["heap"]["result"]["result"], ["ref", "output"])

    def test_join_jobs_cannot_migrate_between_controllers(self):
        state, row = joined_machine()
        apply_protocol_record(state, row)
        for mutation in ("join-owner", "job-owner", "controller-owner"):
            changed = copy.deepcopy(state)
            if mutation == "join-owner":
                changed["heap"]["result"]["join"]["owner"] = "callback"
            elif mutation == "job-owner":
                changed["jobs"][0][1]["owner"] = "callback"
            else:
                changed["frameFunctions"] = {"callback": "handler"}
                changed["stack"] = ["callback"]
            with (
                self.subTest(mutation=mutation),
                self.assertRaisesRegex(ProtocolV8Error, "another invocation|scheduler linkage"),
            ):
                validate_state(changed)

        changed = copy.deepcopy(state)
        changed["stack"] = []
        changed["frameFunctions"] = {"callback": "handler"}
        before = copy.deepcopy(changed)
        with self.assertRaisesRegex(ProtocolV8Error, "unknown or stale"):
            apply_protocol_record(changed, finish_row())
        self.assertEqual(changed, before)

    def test_legacy_settlement_cannot_bypass_join_result_ownership(self):
        state, row = joined_machine()
        apply_protocol_record(state, row)
        before = copy.deepcopy(state)
        for action in ("resolve", "reject", "adopt"):
            fields = {"action": action, "promise": "result"}
            fields.update(target="input") if action == "adopt" else fields.update(value=["num", 9])
            with self.subTest(action=action), self.assertRaisesRegex(ProtocolV8Error, "join step"):
                assert_detailed_record_compatible(_external_machine(state), "promise", fields)
            self.assertEqual(state, before)

    def test_empty_all_must_restore_fulfilled_while_empty_race_remains_pending(self):
        state, row = joined_machine(empty=True)
        apply_protocol_record(state, row)
        self.assertTrue(validate_state(state))
        target = state["heap"]["result"]
        target.update(state="pending", result=None)
        target["join"]["settled"] = False
        with self.assertRaisesRegex(ProtocolV8Error, "restored join metadata"):
            validate_state(state)
        state, row = joined_machine(empty=True)
        row.update(mode="race", output=None)
        apply_protocol_record(state, row)
        self.assertTrue(validate_state(state))
        self.assertEqual(state["heap"]["result"]["state"], "pending")
        state, row = joined_machine()
        state["heap"]["input"].update(state="pending", result=None)
        apply_protocol_record(state, row)
        self.assertTrue(validate_state(state))

    def test_join_result_cannot_already_be_adopted_or_external(self):
        for field, value in (("adopted", "input"), ("external", "request")):
            state, row = joined_machine(empty=True)
            state["heap"]["result"][field] = value
            before = copy.deepcopy(state)
            with self.assertRaisesRegex(ProtocolV8Error, "fresh pending"):
                apply_protocol_record(state, row)
            self.assertEqual(state, before)
            state, row = joined_machine()
            apply_protocol_record(state, row)
            state["heap"]["result"][field] = value
            with self.assertRaisesRegex(ProtocolV8Error, "restored join metadata"):
                validate_state(state)

    def test_all_bound_reserves_array_length_without_reducing_race_bound(self):
        state, row = joined_machine()
        row["inputs"] = ["input"] * 4096
        row["jobs"] = ["job" + str(index) for index in range(4096)]
        before = copy.deepcopy(state)
        with self.assertRaisesRegex(ProtocolV8Error, "array length"):
            apply_protocol_record(state, row)
        self.assertEqual(state, before)
        row["mode"] = "race"
        apply_protocol_record(state, row)
        self.assertEqual(len(state["jobs"]), 4096)

    def test_observed_race_cannot_be_restored_to_pending(self):
        state, row = joined_machine()
        row["mode"] = "race"
        apply_protocol_record(state, row)
        self.assertTrue(validate_state(state))
        state["stack"] = []
        step = finish_row(None)
        step["value"] = ["num", 7]
        apply_protocol_record(state, step)
        self.assertTrue(validate_state(state))
        result = state["heap"]["result"]
        result["state"], result["result"], result["join"]["settled"] = "pending", None, False
        with self.assertRaisesRegex(ProtocolV8Error, "restored join metadata"):
            validate_state(state)

    def test_empty_and_final_output_reject_reserved_identities_atomically(self):
        for empty in (True, False):
            for reservation in ("symbol", "private", "job"):
                with self.subTest(empty=empty, reservation=reservation):
                    state, row = joined_machine(empty)
                    if reservation == "symbol":
                        apply_protocol_record(
                            state,
                            {"kind": "symbol.create", "id": "o7", "description": ["undefined"]},
                        )
                        identity = "o7"
                    elif reservation == "private":
                        state["protocol"]["construction"] = {
                            "privateNames": [{"id": "private"}],
                            "frames": [],
                            "lastCompletion": None,
                        }
                        identity = "private"
                    else:
                        state["jobs"] = [["other-job", {"status": "completed"}]]
                        identity = "other-job"
                    if empty:
                        row["output"] = identity
                    else:
                        apply_protocol_record(state, row)
                        state["stack"] = []
                        row = finish_row(identity)
                    before = copy.deepcopy(state)
                    with self.assertRaisesRegex(ProtocolV8Error, "identity"):
                        apply_protocol_record(state, row)
                    self.assertEqual(state, before)

    def test_restore_requires_observation_and_job_completion_to_agree(self):
        state, row = joined_machine()
        apply_protocol_record(state, row)
        self.assertTrue(validate_state(state))
        altered = copy.deepcopy(state)
        join = altered["heap"]["result"]["join"]
        join["observed"][0], join["values"][0] = True, ["num", 7]
        with self.assertRaisesRegex(ProtocolV8Error, "completion differs"):
            validate_state(altered)
        altered = copy.deepcopy(state)
        altered["jobs"][0][1]["status"] = "completed"
        with self.assertRaisesRegex(ProtocolV8Error, "completion differs"):
            validate_state(altered)
        state["stack"] = []
        apply_protocol_record(state, finish_row())
        self.assertTrue(validate_state(state))
        altered = copy.deepcopy(state)
        altered["jobs"][0][1]["status"] = "queued"
        altered["ready"] = ["job"]
        with self.assertRaisesRegex(ProtocolV8Error, "completion differs"):
            validate_state(altered)


if __name__ == "__main__":
    unittest.main()
