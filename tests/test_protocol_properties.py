import copy
import unittest

from megatoken.data.protocol_v8 import (
    ProtocolV8Error,
    _external_machine,
    apply_protocol_record,
    create_protocol_state,
    validate_protocol_state,
)


class OrdinaryArrayPropertyTests(unittest.TestCase):
    def setUp(self):
        self.machine = {
            "protocol": create_protocol_state(),
            "heap": {},
            "stack": ["root"],
            "locals": {},
            "captureAliases": {},
            "frameFunctions": {"root": "root"},
            "receiver": ["undefined"],
            "pending": None,
            "lastReturn": None,
            "completion": None,
        }

    def send(self, kind, **fields):
        return apply_protocol_record(self.machine, {"kind": kind, **fields})

    def array(self, length=0):
        self.send(
            "allocate",
            name="a",
            objectKind="array",
            length=length,
            properties=[],
            code=None,
            captures=[],
            functionKind=None,
            lexicalReceiver=None,
        )

    def set(self, key, value, success=True):
        self.send(
            "property.set",
            object="a",
            key=["str", key],
            receiver=["ref", "a"],
            value=value,
            success=success,
        )

    def define(self, key, descriptor, success=True):
        self.send(
            "property.define", object="a", key=["str", key], descriptor=descriptor, success=success
        )

    def data(self, value, writable=True, enumerable=True, configurable=True):
        return {
            "kind": "data",
            "value": value,
            "writable": writable,
            "enumerable": enumerable,
            "configurable": configurable,
        }

    def test_length_write_truncates_and_growth_updates_length(self):
        self.array(4)
        self.set("1", ["str", "one"])
        self.set("3", ["str", "three"])
        self.set("length", ["num", 2])
        self.assertNotIn("3", self.machine["heap"]["a"]["properties"])
        self.assertEqual(self.machine["heap"]["a"]["properties"]["length"], ["num", 2])
        self.set("5", ["str", "five"])
        self.assertEqual(self.machine["heap"]["a"]["properties"]["length"], ["num", 6])

    def test_array_length_rejects_invalid_and_nonwritable_changes(self):
        self.array(2)
        with self.assertRaisesRegex(ProtocolV8Error, "invalid array length"):
            self.set("length", ["num", 1.5], success=False)
        self.define(
            "length", self.data(["num", 2], writable=False, enumerable=False, configurable=False)
        )
        self.set("length", ["num", 3], success=False)
        self.set("2", ["str", "no"], success=False)
        self.assertNotIn("2", self.machine["heap"]["a"]["properties"])

    def test_defining_index_grows_length_but_obeys_nonwritable_length(self):
        self.array(1)
        self.define("3", self.data(["str", "three"]))
        self.assertEqual(self.machine["heap"]["a"]["properties"]["length"], ["num", 4])
        self.define(
            "length", self.data(["num", 4], writable=False, enumerable=False, configurable=False)
        )
        self.define("5", self.data(["str", "five"]), success=False)
        self.assertNotIn("5", self.machine["heap"]["a"]["properties"])

    def test_length_shrink_stops_at_nonconfigurable_index(self):
        self.array(4)
        self.define("1", self.data(["str", "fixed"], configurable=False))
        self.set("3", ["str", "removed"])
        self.set("length", ["num", 1], success=False)
        properties = self.machine["heap"]["a"]["properties"]
        self.assertNotIn("3", properties)
        self.assertEqual(properties["length"], ["num", 2])
        self.assertEqual(properties["1"], ["str", "fixed"])

    def test_define_length_shrink_partial_failure_is_preserved(self):
        self.array(4)
        self.define("1", self.data(["str", "fixed"], configurable=False))
        self.define("3", self.data(["str", "removed"]))
        self.define(
            "length",
            self.data(["num", 1], writable=False, enumerable=False, configurable=False),
            success=False,
        )
        properties = self.machine["heap"]["a"]["properties"]
        self.assertNotIn("3", properties)
        self.assertEqual(properties["length"], ["num", 2])
        self.assertFalse(self.machine["heap"]["a"]["descriptors"]["length"]["writable"])

    def test_inherited_data_write_targets_receiver(self):
        self.array()
        self.send(
            "allocate",
            name="prototype",
            objectKind="object",
            length=None,
            properties=[],
            code=None,
            captures=[],
            functionKind=None,
            lexicalReceiver=None,
        )
        self.send(
            "allocate",
            name="receiver",
            objectKind="object",
            length=None,
            properties=[],
            code=None,
            captures=[],
            functionKind=None,
            lexicalReceiver=None,
        )
        self.send(
            "property.define",
            object="prototype",
            key=["str", "x"],
            descriptor=self.data(["num", 1]),
            success=True,
        )
        self.send("property.prototype", object="a", prototype=["ref", "prototype"], success=True)
        self.send(
            "property.set",
            object="a",
            key=["str", "x"],
            receiver=["ref", "receiver"],
            value=["num", 2],
            success=True,
        )
        self.assertEqual(self.machine["heap"]["prototype"]["properties"]["x"], ["num", 1])
        self.assertEqual(self.machine["heap"]["receiver"]["properties"]["x"], ["num", 2])

    def test_inherited_getter_call_uses_receiver(self):
        self.array()
        self.send(
            "allocate",
            name="prototype",
            objectKind="object",
            length=None,
            properties=[],
            code=None,
            captures=[],
            functionKind=None,
            lexicalReceiver=None,
        )
        self.send(
            "allocate",
            name="getter",
            objectKind="closure",
            length=None,
            properties=[],
            code="getter-code",
            captures=[],
            functionKind="ordinary",
            lexicalReceiver=None,
        )
        self.send(
            "property.define",
            object="prototype",
            key=["str", "x"],
            descriptor={
                "kind": "accessor",
                "get": ["ref", "getter"],
                "set": ["undefined"],
                "enumerable": True,
                "configurable": True,
            },
            success=True,
        )
        self.send("property.prototype", object="a", prototype=["ref", "prototype"], success=True)
        self.send(
            "property.get.call",
            id="getter-call",
            child="getter-body",
            object="a",
            key=["str", "x"],
            receiver=["ref", "a"],
        )
        self.assertEqual(self.machine["receiver"], ["ref", "a"])
        self.send("return", value=["num", 7], invocation="getter-body")
        self.send(
            "property.finish", id="getter-call", completion={"kind": "normal", "value": ["num", 7]}
        )

    def test_branch_path_matches_truthiness_and_rejects_unmodeled_paths(self):
        expr = {"op": "literal", "value": ["num", 0]}
        self.send("choose", expr=expr, decision=["num", 0], path="else")
        with self.assertRaisesRegex(ProtocolV8Error, "false branch path"):
            self.send(
                "choose",
                expr={"op": "literal", "value": ["num", 1]},
                decision=["num", 1],
                path="else",
            )
        with self.assertRaisesRegex(ProtocolV8Error, "unsupported short-circuit path"):
            self.send("choose", expr=expr, decision=["num", 0], path="evaluate-right")

    def test_array_length_definition_requires_fixed_descriptor_flags(self):
        self.array(2)
        descriptor = self.data(["num", 2], enumerable=True, configurable=False)
        self.define("length", descriptor, success=False)
        self.assertEqual(self.machine["heap"]["a"]["properties"]["length"], ["num", 2])

    def test_property_and_arrow_values_require_declared_identities(self):
        self.array()
        with self.assertRaisesRegex(ProtocolV8Error, "unallocated reference"):
            self.define("bad", self.data(["ref", "missing"]))
        with self.assertRaisesRegex(ProtocolV8Error, "unallocated reference"):
            self.send(
                "allocate",
                name="bad-arrow",
                objectKind="closure",
                length=None,
                properties=[],
                code="arrow-code",
                captures=[],
                functionKind="arrow",
                lexicalReceiver=["ref", "missing"],
            )

    def test_array_length_coercion_and_writes_through_another_lookup_object(self):
        self.array(3)
        self.set("2", ["num", 7])
        self.send(
            "allocate",
            name="lookup",
            objectKind="object",
            length=None,
            properties=[],
            code=None,
            captures=[],
            functionKind=None,
            lexicalReceiver=None,
        )
        self.send(
            "property.set",
            object="lookup",
            key=["str", "length"],
            receiver=["ref", "a"],
            value=["str", "1"],
            success=True,
        )
        self.send(
            "property.get",
            object="a",
            key=["str", "2"],
            receiver=["ref", "a"],
            result=["undefined"],
        )
        self.assertEqual(self.machine["heap"]["a"]["properties"]["length"], ["num", 1])
        self.set("length", ["num", "-0"])
        self.assertEqual(self.machine["heap"]["a"]["properties"]["length"], ["num", 0])

    def test_failed_claim_does_not_commit_a_partial_truncation(self):
        self.array(4)
        self.define("1", self.data(["num", 1], configurable=False))
        self.set("3", ["num", 3])
        before = copy.deepcopy(self.machine)
        with self.assertRaisesRegex(ProtocolV8Error, "false property write"):
            self.set("length", ["num", 0], success=True)
        self.assertEqual(self.machine, before)

    def test_symbol_keys_must_exist_on_writes_and_restoration(self):
        self.array()
        with self.assertRaisesRegex(ProtocolV8Error, "undeclared Symbol"):
            self.send(
                "property.set",
                object="a",
                key=["symbol", "o99"],
                receiver=["ref", "a"],
                value=["num", 7],
                success=True,
            )
        self.machine["heap"]["a"]["properties"][("symbol", "o99")] = ["num", 7]
        external = _external_machine(self.machine)
        with self.assertRaisesRegex(ProtocolV8Error, "undeclared Symbol"):
            validate_protocol_state(external["protocol"], external["heap"], external)

    def test_arrow_receiver_and_heap_symbol_collisions_cannot_be_restored(self):
        self.array()
        self.send(
            "allocate",
            name="arrow",
            objectKind="closure",
            length=None,
            properties=[],
            code="arrow-code",
            captures=[],
            functionKind="arrow",
            lexicalReceiver=["ref", "a"],
        )
        self.machine["heap"]["arrow"]["lexicalReceiver"] = ["ref", "missing"]
        external = _external_machine(self.machine)
        with self.assertRaisesRegex(ProtocolV8Error, "unallocated reference"):
            validate_protocol_state(external["protocol"], external["heap"], external)
        self.machine["heap"]["arrow"]["lexicalReceiver"] = ["ref", "a"]
        self.machine["heap"]["well-known:iterator"] = self.machine["heap"]["a"]
        external = _external_machine(self.machine)
        with self.assertRaisesRegex(ProtocolV8Error, "symbol/heap identity collision"):
            validate_protocol_state(external["protocol"], external["heap"], external)

    def test_noncanonical_index_strings_do_not_change_array_length(self):
        self.array()
        for key in ("01", "4294967295", "9" * 5000):
            self.set(key, ["num", 1])
        self.assertEqual(self.machine["heap"]["a"]["properties"]["length"], ["num", 0])

    def test_prototype_cycles_are_rejected_without_installing_them(self):
        self.array()
        self.send(
            "allocate",
            name="parent",
            objectKind="object",
            length=None,
            properties=[],
            code=None,
            captures=[],
            functionKind=None,
            lexicalReceiver=None,
        )
        self.send("property.prototype", object="a", prototype=["ref", "parent"], success=True)
        self.send("property.prototype", object="parent", prototype=["ref", "a"], success=False)
        self.assertEqual(self.machine["heap"]["parent"]["prototype"], "ordinary")


if __name__ == "__main__":
    unittest.main()
