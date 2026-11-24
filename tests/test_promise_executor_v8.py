import copy
import unittest

from megatoken.data.promise_object_v8 import intrinsic_id
from megatoken.data.protocol_v8 import (
    ProtocolV8Error,
    _external_machine,
    apply_protocol_record,
    assert_detailed_record_compatible,
    create_protocol_state,
    validate_protocol_state,
)


def machine():
    return {
        "protocol": create_protocol_state(),
        "heap": {
            "promise": {
                "kind": "promise",
                "properties": {},
                "state": "pending",
                "result": None,
                "adopted": None,
                "external": None,
                "reactions": [],
            }
        },
        "locals": {},
        "stack": ["root"],
        "frameFunctions": {"root": "root-code"},
        "pending": None,
        "lastReturn": None,
        "completion": None,
        "captureAliases": {},
        "receiver": ["undefined"],
    }


def enter(state):
    apply_protocol_record(
        state,
        {
            "kind": "promise.executor.enter",
            "promise": "promise",
            "executor": ["ref", "executor"],
            "child": "executor-call",
            "resolve": "resolve-id",
            "reject": "reject-id",
        },
    )


def executor_fixture():
    state = machine()
    apply_protocol_record(
        state,
        {
            "kind": "allocate",
            "name": "executor",
            "objectKind": "closure",
            "code": "executor-code",
            "captures": [],
            "functionKind": "ordinary",
            "lexicalReceiver": None,
            "length": None,
            "properties": [],
        },
    )
    return state


class PromiseExecutorV8Tests(unittest.TestCase):
    def _constructor_getter_adoption(self):
        from megatoken.data.promise_scheduler_v8 import create_promise

        state = executor_fixture()
        apply_protocol_record(state, {"kind": "promise.intrinsics"})
        create_promise(state, "source")
        apply_protocol_record(
            state,
            {
                "kind": "allocate",
                "name": "constructor-getter",
                "objectKind": "closure",
                "code": "constructor-getter-code",
                "captures": [],
                "functionKind": "ordinary",
                "lexicalReceiver": None,
                "length": None,
                "properties": [],
            },
        )
        apply_protocol_record(
            state,
            {
                "kind": "allocate",
                "name": "error",
                "objectKind": "object",
                "code": None,
                "captures": [],
                "functionKind": None,
                "lexicalReceiver": None,
                "length": None,
                "properties": [],
            },
        )
        apply_protocol_record(
            state,
            {
                "kind": "property.define",
                "object": "source",
                "key": ["str", "constructor"],
                "descriptor": {
                    "kind": "accessor",
                    "get": ["ref", "constructor-getter"],
                    "set": ["undefined"],
                    "enumerable": False,
                    "configurable": True,
                },
                "success": True,
            },
        )
        apply_protocol_record(
            state,
            {
                "kind": "property.define",
                "object": "source",
                "key": ["str", "then"],
                "descriptor": {
                    "kind": "data",
                    "value": ["ref", "intrinsicpromiseprototypethen"],
                    "writable": True,
                    "enumerable": True,
                    "configurable": True,
                },
                "success": True,
            },
        )
        enter(state)
        apply_protocol_record(
            state,
            {
                "kind": "promise.resolver.call",
                "resolver": ["ref", "resolve-id"],
                "value": ["ref", "source"],
            },
        )
        apply_protocol_record(
            state,
            {
                "kind": "promise.adopt.enqueue",
                "promise": "promise",
                "target": "source",
                "job": "adopt-job",
            },
        )
        apply_protocol_record(
            state,
            {
                "kind": "promise.executor.finish",
                "promise": "promise",
                "invocation": "executor-call",
                "completion": {"kind": "return", "value": ["undefined"]},
            },
        )
        state["stack"] = []
        state["heap"]["source"]["properties"]["then"] = [
            "str",
            "replaced after resolution",
        ]
        apply_protocol_record(
            state,
            {
                "kind": "promise.adopt.get",
                "promise": "promise",
                "target": "source",
                "job": "adopt-job",
                "child": "constructor-getter-call",
            },
        )
        return state

    def test_queued_adoption_keeps_intrinsic_then_selection_and_reading_is_pending(self):
        state = self._constructor_getter_adoption()
        self.assertEqual(state["heap"]["promise"]["adoption"]["phase"], "reading")
        saved = _external_machine(state)
        self.assertTrue(validate_protocol_state(saved["protocol"], saved["heap"], saved))

        tampered = copy.deepcopy(state)
        tampered["heap"]["promise"].update(state="fulfilled", result=["num", 9])
        forged = _external_machine(tampered)
        with self.assertRaisesRegex(
            ProtocolV8Error, "reading adoption requires a pending|resolver first call"
        ):
            validate_protocol_state(forged["protocol"], forged["heap"], forged)

        state["lastReturn"] = {"kind": "throw", "value": ["ref", "error"]}
        from megatoken.data.promise_adoption_v8 import apply_record as apply_adoption_record

        apply_adoption_record(
            state,
            {
                "kind": "promise.adopt.finish",
                "promise": "promise",
                "target": "source",
                "job": "adopt-job",
                "child": "constructor-getter-call",
                "completion": {"kind": "throw", "value": ["ref", "error"]},
                "forwarding": None,
            },
        )
        saved_rejected = _external_machine(state)
        self.assertTrue(
            validate_protocol_state(
                saved_rejected["protocol"], saved_rejected["heap"], saved_rejected
            )
        )

    def test_foreign_shared_realm_adoption_keeps_common_heap_invariants(self):
        state = machine()
        state["stack"] = []
        state["heap"]["target"] = {
            "kind": "promise",
            "properties": {},
            "state": "fulfilled",
            "result": ["num", 1],
            "adopted": None,
            "external": None,
            "reactions": [],
        }
        state["heap"]["promise"].update(state="fulfilled", result=["num", 1])
        state["heap"]["promise"]["adoption"] = {
            "schema": "semantic-promise-adoption-state/1",
            "owner": "foreign-root",
            "target": "target",
            "phase": "completed",
            "cause": {"kind": "async", "invocation": "foreign-call"},
            "job": "foreign-adopt-job",
            "forwarding": "foreign-forward-job",
            "error": None,
        }
        saved = _external_machine(state)
        self.assertTrue(validate_protocol_state(saved["protocol"], saved["heap"], saved))

        # Controller-local job/cause rows are absent here, but shared heap
        # settlement consistency is still mandatory for every owner.
        bad = copy.deepcopy(saved)
        target = dict(bad["heap"])["target"]
        target["state"] = "fulfilled"
        target["result"] = ["num", 2]
        bad["heap"] = [[identity, node] for identity, node in dict(bad["heap"]).items()]
        with self.assertRaises(ProtocolV8Error):
            validate_protocol_state(bad["protocol"], bad["heap"], bad)

    def test_owned_promise_resolution_uses_fifo_adoption_and_forwarding_jobs(self):
        state = executor_fixture()
        state["heap"]["source"] = {
            "kind": "promise",
            "properties": {},
            "state": "pending",
            "result": None,
            "adopted": None,
            "external": None,
            "reactions": [],
        }
        enter(state)
        apply_protocol_record(
            state,
            {
                "kind": "promise.resolver.call",
                "resolver": ["ref", "resolve-id"],
                "value": ["ref", "source"],
            },
        )
        self.assertEqual(state["heap"]["promise"]["state"], "pending")
        self.assertEqual(state["heap"]["promise"]["adoption"]["phase"], "intent")
        saved_intent = _external_machine(state)
        self.assertTrue(
            validate_protocol_state(saved_intent["protocol"], saved_intent["heap"], saved_intent)
        )
        apply_protocol_record(
            state,
            {
                "kind": "promise.adopt.enqueue",
                "promise": "promise",
                "target": "source",
                "job": "adopt-job",
            },
        )
        self.assertEqual(state["ready"], ["adopt-job"])
        before = copy.deepcopy(state)
        with self.assertRaisesRegex(ProtocolV8Error, "scheduler record|managed Promise settlement"):
            assert_detailed_record_compatible(
                _external_machine(state),
                "promise",
                {"action": "resolve", "promise": "promise", "value": ["num", 99]},
            )
        self.assertEqual(state, before)
        apply_protocol_record(
            state,
            {
                "kind": "promise.executor.finish",
                "promise": "promise",
                "invocation": "executor-call",
                "completion": {"kind": "return", "value": ["undefined"]},
            },
        )
        state["stack"] = []
        saved_finished = _external_machine(state)
        self.assertTrue(
            validate_protocol_state(
                saved_finished["protocol"], saved_finished["heap"], saved_finished
            )
        )
        saved = _external_machine(state)
        self.assertTrue(validate_protocol_state(saved["protocol"], saved["heap"], saved))
        tampered = copy.deepcopy(state)
        next(row for row in tampered["jobs"] if row[0] == "adopt-job")[1]["adoptionTarget"] = (
            "other"
        )
        saved_tampered = _external_machine(tampered)
        with self.assertRaises(ProtocolV8Error):
            validate_protocol_state(
                saved_tampered["protocol"], saved_tampered["heap"], saved_tampered
            )
        tampered = copy.deepcopy(state)
        next(row for row in tampered["jobs"] if row[0] == "adopt-job")[1]["extra"] = None
        saved_tampered = _external_machine(tampered)
        with self.assertRaisesRegex(ProtocolV8Error, "noncanonical retained fields"):
            validate_protocol_state(
                saved_tampered["protocol"], saved_tampered["heap"], saved_tampered
            )
        apply_protocol_record(
            state,
            {
                "kind": "promise.adopt.enter",
                "promise": "promise",
                "target": "source",
                "job": "adopt-job",
                "forwarding": "forward-job",
            },
        )
        self.assertEqual(state["heap"]["promise"]["adoption"]["phase"], "observing")
        saved_observing = _external_machine(state)
        self.assertTrue(
            validate_protocol_state(
                saved_observing["protocol"], saved_observing["heap"], saved_observing
            )
        )
        from megatoken.data.promise_scheduler_v8 import settle_promise

        settle_promise(state, "source", "fulfilled", ["num", 17])
        self.assertEqual(state["ready"], ["forward-job"])
        saved_ready = _external_machine(state)
        self.assertTrue(
            validate_protocol_state(saved_ready["protocol"], saved_ready["heap"], saved_ready)
        )
        apply_protocol_record(
            state,
            {
                "kind": "promise.adopt.forward",
                "promise": "promise",
                "target": "source",
                "job": "forward-job",
                "status": "fulfilled",
                "value": ["num", 17],
            },
        )
        self.assertEqual(state["heap"]["promise"]["state"], "fulfilled")
        self.assertEqual(state["heap"]["promise"]["result"], ["num", 17])
        self.assertTrue(
            validate_protocol_state(
                _external_machine(state)["protocol"],
                _external_machine(state)["heap"],
                _external_machine(state),
            )
        )

    def test_resolution_intent_cannot_be_skipped_or_retargeted(self):
        state = executor_fixture()
        state["heap"]["source"] = {
            "kind": "promise",
            "properties": {},
            "state": "pending",
            "result": None,
            "adopted": None,
            "external": None,
            "reactions": [],
        }
        state["heap"]["other"] = copy.deepcopy(state["heap"]["source"])
        enter(state)
        apply_protocol_record(
            state,
            {
                "kind": "promise.resolver.call",
                "resolver": ["ref", "resolve-id"],
                "value": ["ref", "source"],
            },
        )
        before = copy.deepcopy(state)
        with self.assertRaises(ProtocolV8Error):
            apply_protocol_record(state, {"kind": "end"})
        self.assertEqual(state, before)
        with self.assertRaises(ProtocolV8Error):
            apply_protocol_record(
                state,
                {
                    "kind": "promise.adopt.enqueue",
                    "promise": "promise",
                    "target": "other",
                    "job": "adopt-job",
                },
            )
        self.assertEqual(state, before)
        with self.assertRaisesRegex(ProtocolV8Error, "same Promise and target"):
            apply_protocol_record(
                state,
                {
                    "kind": "promise.adopt.enqueue",
                    "promise": "other",
                    "target": "source",
                    "job": "other-job",
                },
            )
        self.assertEqual(state, before)

    def test_overridden_promise_then_is_refused_before_resolver_mutation(self):
        for key in ("properties", "descriptors", "prototype"):
            state = executor_fixture()
            source = {
                "kind": "promise",
                "properties": {},
                "state": "pending",
                "result": None,
                "adopted": None,
                "adoption": None,
                "external": None,
                "reactions": [],
            }
            if key == "properties":
                source[key] = {"then": ["undefined"]}
            elif key == "descriptors":
                source[key] = {"then": {"kind": "accessor"}}
            else:
                source[key] = None
            state["heap"]["source"] = source
            enter(state)
            before = copy.deepcopy(state)
            with self.assertRaisesRegex(ProtocolV8Error, "then lookup"):
                apply_protocol_record(
                    state,
                    {
                        "kind": "promise.resolver.call",
                        "resolver": ["ref", "resolve-id"],
                        "value": ["ref", "source"],
                    },
                )
            self.assertEqual(state, before)

    def test_self_resolution_allocates_only_the_declared_typeerror(self):
        state = executor_fixture()
        enter(state)
        apply_protocol_record(
            state,
            {
                "kind": "promise.resolver.call",
                "resolver": ["ref", "resolve-id"],
                "value": ["ref", "promise"],
            },
        )
        before = copy.deepcopy(state)
        with self.assertRaisesRegex(ProtocolV8Error, "identity"):
            apply_protocol_record(
                state, {"kind": "promise.adopt.self", "promise": "promise", "error": "executor"}
            )
        self.assertEqual(state, before)
        apply_protocol_record(
            state, {"kind": "promise.adopt.self", "promise": "promise", "error": "self-typeerror"}
        )
        self.assertEqual(state["heap"]["promise"]["state"], "rejected")
        self.assertEqual(state["heap"]["promise"]["result"], ["ref", "self-typeerror"])
        self.assertEqual(state["heap"]["self-typeerror"]["kind"], "object")
        self.assertEqual(
            state["heap"]["self-typeerror"]["properties"],
            {"message": ["str", "Chaining cycle detected for promise"]},
        )
        self.assertNotIn("name", state["heap"]["self-typeerror"]["properties"])
        self.assertEqual(
            state["heap"]["self-typeerror"]["prototype"],
            ["ref", intrinsic_id("TypeError.prototype")],
        )
        external = _external_machine(state)
        self.assertTrue(validate_protocol_state(external["protocol"], external["heap"], external))
        state["heap"]["self-typeerror"]["properties"]["message"] = ["str", "mutated by guest"]
        external_after_write = _external_machine(state)
        self.assertTrue(
            validate_protocol_state(
                external_after_write["protocol"],
                external_after_write["heap"],
                external_after_write,
            )
        )

    def test_nested_and_mutual_adoption_use_jobs_without_synchronous_cascade(self):
        def owned_machine():
            state = machine()
            for identity in ("left", "right", "leaf"):
                state["heap"][identity] = {
                    "kind": "promise",
                    "properties": {},
                    "state": "pending",
                    "result": None,
                    "adopted": None,
                    "external": None,
                    "reactions": [],
                }
            return state

        state = owned_machine()
        for promise, target, job in (("left", "right", "left-job"), ("right", "leaf", "right-job")):
            apply_protocol_record(
                state,
                {"kind": "promise.adopt.enqueue", "promise": promise, "target": target, "job": job},
            )
        with self.assertRaisesRegex(ProtocolV8Error, "scheduler record"):
            assert_detailed_record_compatible(
                _external_machine(state),
                "promise",
                {"action": "resolve", "promise": "left", "value": ["num", 99]},
            )
        state["stack"] = []
        apply_protocol_record(
            state,
            {
                "kind": "promise.adopt.enter",
                "promise": "left",
                "target": "right",
                "job": "left-job",
                "forwarding": "left-forward",
            },
        )
        apply_protocol_record(
            state,
            {
                "kind": "promise.adopt.enter",
                "promise": "right",
                "target": "leaf",
                "job": "right-job",
                "forwarding": "right-forward",
            },
        )
        from megatoken.data.promise_scheduler_v8 import settle_promise

        settle_promise(state, "leaf", "fulfilled", ["num", 41])
        self.assertEqual(state["ready"], ["right-forward"])
        apply_protocol_record(
            state,
            {
                "kind": "promise.adopt.forward",
                "promise": "right",
                "target": "leaf",
                "job": "right-forward",
                "status": "fulfilled",
                "value": ["num", 41],
            },
        )
        self.assertEqual(state["ready"], ["left-forward"])
        apply_protocol_record(
            state,
            {
                "kind": "promise.adopt.forward",
                "promise": "left",
                "target": "right",
                "job": "left-forward",
                "status": "fulfilled",
                "value": ["num", 41],
            },
        )
        self.assertEqual(state["heap"]["left"]["result"], ["num", 41])

        cycle = owned_machine()
        for promise, target, job in (("left", "right", "left-job"), ("right", "left", "right-job")):
            apply_protocol_record(
                cycle,
                {"kind": "promise.adopt.enqueue", "promise": promise, "target": target, "job": job},
            )
        cycle["stack"] = []
        apply_protocol_record(
            cycle,
            {
                "kind": "promise.adopt.enter",
                "promise": "left",
                "target": "right",
                "job": "left-job",
                "forwarding": "left-forward",
            },
        )
        apply_protocol_record(
            cycle,
            {
                "kind": "promise.adopt.enter",
                "promise": "right",
                "target": "left",
                "job": "right-job",
                "forwarding": "right-forward",
            },
        )
        self.assertEqual(cycle["ready"], [])
        self.assertEqual(cycle["heap"]["left"]["state"], "pending")
        self.assertEqual(cycle["heap"]["right"]["state"], "pending")

    def test_reaction_returned_promise_is_bound_before_its_adoption_enqueue(self):
        state = executor_fixture()
        for identity in ("derived", "source"):
            state["heap"][identity] = {
                "kind": "promise",
                "properties": {},
                "state": "pending",
                "result": None,
                "adopted": None,
                "external": None,
                "reactions": [],
            }
        state["stack"] = []
        state["activeJob"] = "reaction"
        state["jobs"] = {
            "reaction": {
                "status": "running",
                "owner": "root",
                "promise": "input",
                "target": "derived",
                "reaction": "then",
                "returned": {"kind": "return", "value": ["ref", "source"]},
                "priorAliases": [],
            }
        }
        apply_protocol_record(
            state,
            {
                "kind": "promise.reaction.complete",
                "job": "reaction",
                "owner": "root",
                "promise": "derived",
                "target": "source",
            },
        )
        self.assertEqual(state["activeJob"], None)
        self.assertEqual(
            state["heap"]["derived"]["adoption"]["cause"], {"kind": "reaction", "job": "reaction"}
        )
        apply_protocol_record(
            state,
            {
                "kind": "promise.adopt.enqueue",
                "promise": "derived",
                "target": "source",
                "job": "reaction-adopt",
            },
        )
        self.assertEqual(state["ready"], ["reaction-adopt"])

    def test_synchronous_executor_resolves_once_and_finishes_with_promise(self):
        state = executor_fixture()
        enter(state)
        self.assertEqual(state["stack"], ["root", "executor-call"])
        self.assertEqual(state["locals"]["executor-call:argument:0"], ["ref", "resolve-id"])
        apply_protocol_record(
            state,
            {
                "kind": "promise.resolver.call",
                "resolver": ["ref", "resolve-id"],
                "value": ["num", 7],
            },
        )
        before = copy.deepcopy(state["heap"]["promise"])
        apply_protocol_record(
            state,
            {
                "kind": "promise.resolver.call",
                "resolver": ["ref", "reject-id"],
                "value": ["str", "late"],
            },
        )
        self.assertEqual(state["heap"]["promise"], before)
        apply_protocol_record(
            state,
            {
                "kind": "promise.executor.finish",
                "promise": "promise",
                "invocation": "executor-call",
                "completion": {"kind": "return", "value": ["undefined"]},
            },
        )
        self.assertEqual(state["heap"]["promise"]["state"], "fulfilled")
        self.assertEqual(state["lastReturn"], {"kind": "return", "value": ["ref", "promise"]})
        self.assertTrue(
            validate_protocol_state(
                _external_machine(state)["protocol"],
                _external_machine(state)["heap"],
                _external_machine(state),
            )
        )

    def test_throwing_executor_rejects_pending_promise(self):
        state = executor_fixture()
        enter(state)
        apply_protocol_record(
            state,
            {
                "kind": "promise.executor.finish",
                "promise": "promise",
                "invocation": "executor-call",
                "completion": {"kind": "throw", "value": ["str", "failure"]},
            },
        )
        self.assertEqual(state["heap"]["promise"]["state"], "rejected")
        self.assertEqual(state["heap"]["promise"]["result"], ["str", "failure"])
        self.assertTrue(
            validate_protocol_state(
                _external_machine(state)["protocol"],
                _external_machine(state)["heap"],
                _external_machine(state),
            )
        )

    def test_resolution_adopts_no_objects_and_reject_first_call_wins(self):
        state = executor_fixture()
        apply_protocol_record(
            state,
            {
                "kind": "allocate",
                "name": "other",
                "objectKind": "object",
                "code": None,
                "captures": [],
                "functionKind": None,
                "lexicalReceiver": None,
                "length": None,
                "properties": [],
            },
        )
        enter(state)
        before = copy.deepcopy(state)
        with self.assertRaisesRegex(ProtocolV8Error, "assimilation protocol"):
            apply_protocol_record(
                state,
                {
                    "kind": "promise.resolver.call",
                    "resolver": ["ref", "resolve-id"],
                    "value": ["ref", "other"],
                },
            )
        self.assertEqual(state, before)
        apply_protocol_record(
            state,
            {
                "kind": "promise.resolver.call",
                "resolver": ["ref", "reject-id"],
                "value": ["str", "first rejection"],
            },
        )
        apply_protocol_record(
            state,
            {
                "kind": "promise.resolver.call",
                "resolver": ["ref", "resolve-id"],
                "value": ["num", 99],
            },
        )
        self.assertEqual(state["heap"]["promise"]["state"], "rejected")
        self.assertEqual(state["heap"]["promise"]["result"], ["str", "first rejection"])

    def test_generic_return_and_direct_call_cannot_bypass_executor_records(self):
        state = executor_fixture()
        enter(state)
        before = copy.deepcopy(state)
        with self.assertRaises(ProtocolV8Error):
            apply_protocol_record(
                state, {"kind": "return", "invocation": "executor-call", "value": ["undefined"]}
            )
        self.assertEqual(state, before)
        with self.assertRaises(ProtocolV8Error):
            apply_protocol_record(
                state,
                {
                    "kind": "call",
                    "callee": ["ref", "resolve-id"],
                    "receiver": ["undefined"],
                    "arguments": [["num", 1]],
                    "child": "resolver-call",
                },
            )
        self.assertEqual(state, before)


if __name__ == "__main__":
    unittest.main()
