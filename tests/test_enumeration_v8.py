"""For-in keys, shadowing, and cursors are checked independently of their claims."""

import copy
import json
import unittest

from megatoken.data.enumeration_v8 import validate_state
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

    def test_value_writes_preserve_cursor_but_structural_changes_refuse_next(self):
        state = machine()
        allocate(state, "source")
        define(state, "source", ["str", "key"], ["num", 1])
        create(state, ["ref", "source"])
        define(state, "source", ["str", "key"], ["num", 2])
        advance(state, "key")
        define(state, "source", ["str", "new"], ["num", 3])
        # This prefix may end in break; checkpoints preserve it without
        # inventing another enumeration step or silently updating the snapshot.
        shared = json.loads(json.dumps(_external_machine(state)))
        self.assertTrue(validate_protocol_state(shared["protocol"], shared["heap"], shared))
        restored = _internal_machine(shared)
        before = copy.deepcopy(restored)
        with self.assertRaisesRegex(ProtocolV8Error, "structure changed"):
            advance(restored, None, "end")
        self.assertEqual(restored, before)

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
