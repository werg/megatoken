"""For-in keys, shadowing, and cursors are checked independently of their claims."""

import base64
import copy
import gzip
import json
import shutil
import subprocess
import unittest
from pathlib import Path

from megatoken.data.algorithm_v7 import stable_algorithm_hash
from megatoken.data.enumeration_v8 import validate_state
from megatoken.data.native_v8 import NativeV8Error, validate_v8_artifact
from megatoken.data.protocol_v8 import (
    ProtocolV8Error,
    _external_machine,
    _internal_machine,
    apply_protocol_record,
    assert_detailed_record_compatible,
    decode_protocol_record,
    encode_protocol_record,
    validate_protocol_state,
)
from tests.test_number_v8 import allocate, machine


def define(state, identity, key, value, *, enumerable=True):
    apply_protocol_record(
        state,
        {
            "kind": "property.define",
            "object": identity,
            "key": key,
            "descriptor": {
                "kind": "data",
                "value": value,
                "writable": True,
                "enumerable": enumerable,
                "configurable": True,
            },
            "success": True,
        },
    )


def create(state, source, name="cursor"):
    record = {"kind": "enumeration.create", "name": name, "source": source}
    apply_protocol_record(state, decode_protocol_record(encode_protocol_record(record)))


def advance(state, key, name="result", *, cursor="cursor"):
    apply_protocol_record(
        state,
        {
            "kind": "enumeration.next",
            "object": cursor,
            "name": name,
            "result": ["undefined"] if key is None else ["str", key],
            "done": key is None,
        },
    )


class EnumerationV8Tests(unittest.TestCase):
    def test_source_golden_replays_every_context_and_matches_the_contract(self):
        fixture = Path(__file__).parent / "fixtures/semantic-enumeration-v8.json.gz.b64"
        artifact = json.loads(gzip.decompress(base64.b64decode(fixture.read_bytes())))
        from megatoken.data.semantic_contract_v8 import CONTRACT

        declaration = next(
            row
            for row in CONTRACT["interoperability"]
            if row["id"] == "protocol-region:source-for-in-own-keys"
        )
        self.assertEqual(stable_algorithm_hash(artifact["public"]), declaration["publicDigest"])
        self.assertEqual(stable_algorithm_hash(artifact["contexts"]), declaration["contextDigest"])
        receipt = validate_v8_artifact(artifact)
        for check in ("replay", "contexts", "continuation"):
            self.assertEqual(receipt["checks"][check], "passed")
        self.assertEqual(receipt["counts"]["contexts"], 40)
        self.assertEqual(receipt["counts"]["protocolRecords"], 21)
        self.assertEqual(receipt["counts"]["detailedRecords"], 19)
        self.assertFalse(receipt["admitted"])
        mutated = copy.deepcopy(artifact)
        context = next(
            row
            for row in mutated["contexts"]
            if any(node.get("kind") == "enumerator" for _, node in row["state"]["heap"])
        )
        cursor = next(node for _, node in context["state"]["heap"] if node["kind"] == "enumerator")
        cursor["nextIndex"] = len(cursor["keys"]) + 1
        with self.assertRaises(NativeV8Error):
            validate_v8_artifact(mutated)

    def test_own_property_checks_distinguish_inherited_and_nonenumerable_keys(self):
        state = machine()
        allocate(state, "parent")
        allocate(state, "child")
        define(state, "parent", ["str", "inherited"], ["num", 1])
        define(state, "child", ["str", "hidden"], ["num", 2], enumerable=False)
        define(state, "child", ["symbol", "well-known:iterator"], ["num", 3])
        apply_protocol_record(
            state,
            {
                "kind": "property.prototype",
                "object": "child",
                "prototype": ["ref", "parent"],
                "success": True,
            },
        )
        for key, expected in (
            (["str", "inherited"], False),
            (["str", "hidden"], True),
            (["symbol", "well-known:iterator"], True),
            (["str", "missing"], False),
        ):
            record = {
                "kind": "property.hasOwn",
                "object": "child",
                "key": key,
                "result": ["bool", expected],
            }
            apply_protocol_record(state, record)
            record["result"] = ["bool", not expected]
            before = copy.deepcopy(state)
            with self.assertRaisesRegex(ProtocolV8Error, "false own property"):
                apply_protocol_record(state, record)
            self.assertEqual(state, before)

    def test_key_order_and_nonenumerable_shadowing_across_prototypes(self):
        state = machine()
        allocate(state, "parent")
        allocate(state, "child")
        for key in ("a", "2", "hidden", "tail"):
            define(state, "parent", ["str", key], ["num", 3])
        for key in ("own", "10", "1"):
            define(state, "child", ["str", key], ["num", 1])
        define(state, "child", ["str", "hidden"], ["num", 9], enumerable=False)
        define(state, "child", ["symbol", "well-known:iterator"], ["num", 5])
        apply_protocol_record(
            state,
            {
                "kind": "property.prototype",
                "object": "child",
                "prototype": ["ref", "parent"],
                "success": True,
            },
        )
        create(state, ["ref", "child"])
        keys = ["1", "10", "own", "2", "a", "tail"]
        self.assertEqual(state["heap"]["cursor"]["keys"], keys)
        for index, key in enumerate(keys):
            advance(state, key, f"result{index}")
        advance(state, None)
        advance(state, None, "again")
        self.assertEqual(list(state["heap"]["result"]["properties"]), ["value", "done"])
        self.assertEqual(state["heap"]["cursor"]["nextIndex"], len(keys))
        self.assertTrue(state["heap"]["cursor"]["done"])

    def test_value_writes_and_new_properties_do_not_recapture_candidates_after_restore(self):
        state = machine()
        allocate(state, "source")
        define(state, "source", ["str", "key"], ["num", 1])
        create(state, ["ref", "source"])
        define(state, "source", ["str", "key"], ["num", 2])
        advance(state, "key")
        define(state, "source", ["str", "new"], ["num", 3])
        shared = json.loads(json.dumps(_external_machine(state)))
        self.assertTrue(validate_protocol_state(shared["protocol"], shared["heap"], shared))
        restored = _internal_machine(shared)
        advance(restored, None, "end")
        self.assertEqual(restored["heap"]["cursor"]["keys"], ["key"])
        self.assertEqual(restored["heap"]["cursor"]["nextIndex"], 1)

    def test_deleting_unvisited_own_and_inherited_properties_skips_live_absence(self):
        state = machine()
        allocate(state, "parent")
        allocate(state, "child")
        for identity, keys in (("child", ("a", "b")), ("parent", ("c", "d"))):
            for key in keys:
                define(state, identity, ["str", key], ["num", 1])
        apply_protocol_record(
            state,
            {
                "kind": "property.prototype",
                "object": "child",
                "prototype": ["ref", "parent"],
                "success": True,
            },
        )
        create(state, ["ref", "child"])
        advance(state, "a", "first")
        for identity, key in (("child", "b"), ("parent", "c")):
            apply_protocol_record(
                state,
                {
                    "kind": "property.delete",
                    "object": identity,
                    "key": ["str", key],
                    "success": True,
                },
            )
        shared = json.loads(json.dumps(_external_machine(state)))
        self.assertTrue(validate_protocol_state(shared["protocol"], shared["heap"], shared))
        state = _internal_machine(shared)
        before = copy.deepcopy(state)
        with self.assertRaisesRegex(ProtocolV8Error, "false next result"):
            advance(state, "b", "forged")
        self.assertEqual(state, before)
        advance(state, "d", "second")
        self.assertEqual(state["heap"]["cursor"]["nextIndex"], 4)
        advance(state, None, "end")

    def test_deleted_shadow_falls_back_to_inherited_candidate_without_reordering(self):
        state = machine()
        allocate(state, "parent")
        allocate(state, "child")
        define(state, "child", ["str", "a"], ["num", 1])
        define(state, "child", ["str", "b"], ["num", 2])
        define(state, "parent", ["str", "b"], ["num", 3])
        define(state, "parent", ["str", "c"], ["num", 4])
        apply_protocol_record(
            state,
            {
                "kind": "property.prototype",
                "object": "child",
                "prototype": ["ref", "parent"],
                "success": True,
            },
        )
        create(state, ["ref", "child"])
        advance(state, "a", "first")
        apply_protocol_record(
            state,
            {"kind": "property.delete", "object": "child", "key": ["str", "b"], "success": True},
        )
        advance(state, "b", "second")
        advance(state, "c", "third")
        advance(state, None, "end")

    def test_readded_candidate_keeps_its_captured_position_and_getters_are_not_read(self):
        state = machine()
        allocate(state, "source")
        allocate(state, "getter", "closure")
        for key in ("a", "b", "c"):
            define(state, "source", ["str", key], ["num", 1])
        create(state, ["ref", "source"])
        advance(state, "a", "first")
        apply_protocol_record(
            state,
            {"kind": "property.delete", "object": "source", "key": ["str", "b"], "success": True},
        )
        apply_protocol_record(
            state,
            {
                "kind": "property.define",
                "object": "source",
                "key": ["str", "b"],
                "descriptor": {
                    "kind": "accessor",
                    "get": ["ref", "getter"],
                    "set": ["undefined"],
                    "enumerable": True,
                    "configurable": True,
                },
                "success": True,
            },
        )
        advance(state, "b", "second")
        advance(state, "c", "third")
        advance(state, None, "end")
        self.assertEqual(state["stack"], ["root"])
        self.assertEqual(state["protocol"]["pendingCalls"], [])

    def test_deleted_candidate_falls_back_to_implicit_intrinsic_without_calling_it(self):
        for initialize in (False, True):
            with self.subTest(initialized_intrinsics=initialize):
                state = machine()
                if initialize:
                    apply_protocol_record(state, {"kind": "promise.intrinsics"})
                allocate(state, "source")
                define(state, "source", ["str", "a"], ["num", 1])
                define(state, "source", ["str", "toString"], ["num", 2])
                create(state, ["ref", "source"])
                advance(state, "a", "first")
                apply_protocol_record(
                    state,
                    {
                        "kind": "property.delete",
                        "object": "source",
                        "key": ["str", "toString"],
                        "success": True,
                    },
                )
                advance(state, "toString", "second")
                advance(state, None, "end")
                self.assertEqual(state["stack"], ["root"])

    def test_enumerable_intrinsic_keys_require_an_explicit_chain_at_creation(self):
        state = machine()
        apply_protocol_record(state, {"kind": "promise.intrinsics"})
        allocate(state, "source")
        define(state, "intrinsicobjectprototype", ["str", "inherited"], ["num", 1])
        before = copy.deepcopy(state)
        with self.assertRaisesRegex(ProtocolV8Error, "implicit intrinsic enumeration"):
            create(state, ["ref", "source"])
        self.assertEqual(state, before)
        apply_protocol_record(
            state,
            {
                "kind": "property.prototype",
                "object": "source",
                "prototype": ["ref", "intrinsicobjectprototype"],
                "success": True,
            },
        )
        create(state, ["ref", "source"])
        advance(state, "inherited", "first")
        advance(state, None, "end")

    def test_new_intrinsic_names_do_not_extend_candidates_but_can_supply_deleted_fallback(self):
        state = machine()
        apply_protocol_record(state, {"kind": "promise.intrinsics"})
        allocate(state, "source")
        define(state, "source", ["str", "a"], ["num", 1])
        define(state, "source", ["str", "b"], ["num", 2])
        create(state, ["ref", "source"])
        advance(state, "a", "first")
        define(state, "intrinsicobjectprototype", ["str", "b"], ["num", 3])
        define(state, "intrinsicobjectprototype", ["str", "new"], ["num", 4])
        apply_protocol_record(
            state,
            {"kind": "property.delete", "object": "source", "key": ["str", "b"], "success": True},
        )
        advance(state, "b", "second")
        advance(state, None, "end")
        self.assertEqual(state["heap"]["cursor"]["keys"], ["a", "b"])

    def test_removed_hidden_shadow_does_not_add_a_previously_excluded_candidate(self):
        state = machine()
        allocate(state, "parent")
        allocate(state, "child")
        define(state, "child", ["str", "a"], ["num", 1])
        define(state, "child", ["str", "hidden"], ["num", 2], enumerable=False)
        define(state, "parent", ["str", "hidden"], ["num", 3])
        apply_protocol_record(
            state,
            {
                "kind": "property.prototype",
                "object": "child",
                "prototype": ["ref", "parent"],
                "success": True,
            },
        )
        create(state, ["ref", "child"])
        advance(state, "a", "first")
        apply_protocol_record(
            state,
            {
                "kind": "property.delete",
                "object": "child",
                "key": ["str", "hidden"],
                "success": True,
            },
        )
        advance(state, None, "end")

    def test_changed_enumerability_and_prototypes_keep_captured_candidates(self):
        original = machine()
        allocate(original, "source")
        allocate(original, "parent")
        define(original, "source", ["str", "key"], ["num", 1])
        create(original, ["ref", "source"])
        for change in ("enumerability", "prototype"):
            with self.subTest(change=change):
                state = copy.deepcopy(original)
                if change == "enumerability":
                    define(state, "source", ["str", "key"], ["num", 1], enumerable=False)
                else:
                    apply_protocol_record(
                        state,
                        {
                            "kind": "property.prototype",
                            "object": "source",
                            "prototype": ["ref", "parent"],
                            "success": True,
                        },
                    )
                shared = _external_machine(state)
                self.assertTrue(validate_protocol_state(shared["protocol"], shared["heap"], shared))
                resumed = copy.deepcopy(state)
                advance(resumed, "key")
                advance(resumed, None, "done")
                self.assertEqual(resumed["heap"]["cursor"]["keys"], ["key"])

        advance(original, "key", "first")
        advance(original, None, "end")
        define(original, "source", ["str", "key"], ["num", 2], enumerable=False)
        apply_protocol_record(
            original,
            {
                "kind": "property.prototype",
                "object": "source",
                "prototype": ["ref", "parent"],
                "success": True,
            },
        )
        advance(original, None, "stillDone")
        self.assertTrue(original["heap"]["cursor"]["done"])

    @unittest.skipUnless(shutil.which("node"), "native JavaScript runtime is not available")
    def test_mutation_policy_matches_independent_original_native_loops(self):
        script = """
            const runs=[];
            function run(object, mutate) {
                const keys=[];
                for(const key in object) { keys.push(key); if(keys.length===1) mutate(object); }
                runs.push(keys);
            }
            run({a:1,b:2,c:3}, object=>delete object.b);
            run({a:1,b:2}, object=>{object.c=3;});
            run(Object.assign(Object.create({b:20,c:30}),{a:1,b:2}), object=>delete object.b);
            run(Object.assign(Object.create({b:20,c:30}),{a:1}), object=>delete Object.getPrototypeOf(object).b);
            run({a:1,b:2,c:3}, object=>{
                delete object.b;
                Object.defineProperty(object,'b',{get(){throw new Error('read');},enumerable:true});
            });
            run(Object.defineProperty(Object.assign(Object.create({b:20}),{a:1}),
                'b',{value:2,enumerable:false,configurable:true}), object=>delete object.b);
            run({a:1,toString:2}, object=>delete object.toString);
            process.stdout.write(JSON.stringify(runs));
        """
        actual = json.loads(subprocess.check_output(["node", "-e", script], text=True))
        self.assertEqual(
            actual,
            [
                ["a", "c"],
                ["a", "b"],
                ["a", "b", "c"],
                ["a", "c"],
                ["a", "b", "c"],
                ["a"],
                ["a", "toString"],
            ],
        )

    def test_utf16_string_keys_and_empty_primitive_sources(self):
        state = machine()
        create(state, ["str", "A\U0001f642"])
        self.assertEqual(state["heap"]["cursor"]["keys"], ["0", "1", "2"])
        for index in range(3):
            advance(state, str(index), f"r{index}")
        advance(state, None)
        for source in (
            ["undefined"],
            ["null"],
            ["bool", True],
            ["num", 7],
            ["bigint", "9"],
            ["symbol", "well-known:iterator"],
        ):
            with self.subTest(source=source):
                state = machine()
                create(state, source)
                advance(state, None)
                self.assertEqual(state["heap"]["cursor"]["chain"], [])

    def test_function_sources_and_function_prototypes_require_a_separate_profile(self):
        state = machine()
        allocate(state, "object")
        allocate(state, "function", "closure")
        with self.assertRaisesRegex(ProtocolV8Error, "ordinary"):
            create(state, ["ref", "function"])
        apply_protocol_record(
            state,
            {
                "kind": "property.prototype",
                "object": "object",
                "prototype": ["ref", "function"],
                "success": True,
            },
        )
        with self.assertRaisesRegex(ProtocolV8Error, "ordinary"):
            create(state, ["ref", "object"])

    def test_false_results_and_identity_collisions_leave_cursor_unchanged(self):
        state = machine()
        allocate(state, "source")
        define(state, "source", ["str", "correct"], ["num", 1])
        create(state, ["ref", "source"])
        for key, name in (("invented", "result"), (None, "result"), ("correct", "source")):
            with self.subTest(key=key, name=name):
                before = copy.deepcopy(state)
                with self.assertRaises(ProtocolV8Error):
                    advance(state, key, name)
                self.assertEqual(state, before)
        with self.assertRaisesRegex(ProtocolV8Error, "bypasses"):
            assert_detailed_record_compatible(
                _external_machine(state), "write", {"object": "cursor", "key": "nextIndex"}
            )

    def test_restore_rejects_forged_shadowing_linkage_and_cursor(self):
        state = machine()
        allocate(state, "source")
        for key in ("2", "1", "tail"):
            define(state, "source", ["str", key], ["num", 1])
        create(state, ["ref", "source"])
        mutations = (
            lambda node: node.update(nextIndex=True),
            lambda node: node.update(nextIndex=4),
            lambda node: node.update(done=True),
            lambda node: node.update(keys=["2", "1", "tail"]),
            lambda node: node.update(chain=[None]),
            lambda node: node["chain"][0].update(prototype=["ref", "source"]),
            lambda node: node["chain"][0]["keys"].append(["tail", False]),
            lambda node: node["chain"][0]["keys"].reverse(),
            lambda node: node["properties"].update(extra=["num", 1]),
        )
        for index, mutate in enumerate(mutations):
            with self.subTest(mutation=index):
                forged = copy.deepcopy(state)
                mutate(forged["heap"]["cursor"])
                with self.assertRaises(ProtocolV8Error):
                    validate_state(forged)

    def test_snapshot_bounds_and_unsupported_heap_sources_are_explicit(self):
        state = machine()
        with self.assertRaisesRegex(ProtocolV8Error, "bound"):
            create(state, ["str", "a" * 4097])
        state["heap"]["exotic"] = {"kind": "proxy", "properties": {}}
        with self.assertRaisesRegex(ProtocolV8Error, "ordinary"):
            create(state, ["ref", "exotic"])
        for index in range(129):
            state["heap"][f"p{index}"] = {
                "kind": "object",
                "properties": {},
                "prototype": ["ref", f"p{index + 1}"] if index < 128 else None,
            }
        with self.assertRaisesRegex(ProtocolV8Error, "bound"):
            create(state, ["ref", "p0"])


if __name__ == "__main__":
    unittest.main()
