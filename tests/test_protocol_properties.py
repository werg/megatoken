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

    def test_explicit_length_writes_preserve_key_order_and_reinstall_checkpoint_storage(self):
        # The producer installs an explicit length write as a descriptor. Its
        # Map storage order changes, while the JavaScript key order does not.
        for length in (1, 3):
            with self.subTest(length=length):
                self.setUp()
                self.array(1)
                self.set("0", ["str", "kept"])
                self.set("length", ["num", length])
                node = self.machine["heap"]["a"]
                self.assertEqual(list(node["properties"]), ["0", "length"])
                self.assertEqual(list(node["descriptors"]), ["0", "length"])
                self.assertEqual(node["keyOrder"], ["length", "0"])

        self.setUp()
        self.array(3)
        self.set("0", ["str", "kept"])
        self.set("2", ["str", "removed"])
        self.set("length", ["num", 2])
        node = self.machine["heap"]["a"]
        self.assertEqual(list(node["properties"]), ["0", "length"])
        self.assertEqual(list(node["descriptors"]), ["0", "length"])
        self.assertEqual(node["keyOrder"], ["length", "0"])

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

    def test_branch_path_matches_truthiness(self):
        expr = {"op": "literal", "value": ["num", 0]}
        self.send("choose", expr=expr, decision=["num", 0], path="else")
        with self.assertRaisesRegex(ProtocolV8Error, "false branch path"):
            self.send(
                "choose",
                expr={"op": "literal", "value": ["num", 1]},
                decision=["num", 1],
                path="else",
            )

    def test_short_circuit_choices_check_left_operand_before_right_execution(self):
        for operator, value, skip in (
            ("&&", ["bool", True], False),
            ("&&", ["num", 0], True),
            ("||", ["bool", False], False),
            ("||", ["str", "present"], True),
            ("??", ["null"], False),
            ("??", ["undefined"], False),
            ("??", ["num", 0], True),
            ("??", ["bool", False], True),
            ("&&", ["bigint", "0"], True),
            ("||", ["num", "NaN"], False),
            ("&&", ["symbol", "well-known:iterator"], False),
        ):
            with self.subTest(operator=operator, value=value):
                self.send("bind", name="left", value=value)
                expr = {
                    "op": "logical-test",
                    "operator": operator,
                    "arg": {"op": "local", "name": "left"},
                }
                self.send(
                    "choose",
                    expr=expr,
                    decision=["bool", skip],
                    path="use-left" if skip else "evaluate-right",
                )
                before = copy.deepcopy(self.machine)
                with self.assertRaisesRegex(ProtocolV8Error, "false short-circuit path"):
                    self.send(
                        "choose",
                        expr=expr,
                        decision=["bool", skip],
                        path="evaluate-right" if skip else "use-left",
                    )
                self.assertEqual(self.machine, before)
                with self.assertRaisesRegex(ProtocolV8Error, "false protocol computation"):
                    self.send(
                        "choose",
                        expr=expr,
                        decision=["bool", not skip],
                        path="evaluate-right" if skip else "use-left",
                    )
                self.assertEqual(self.machine, before)

    def test_short_circuit_choice_refuses_unbound_boolean_and_future_operand(self):
        for path in ("use-left", "evaluate-right"):
            with self.assertRaisesRegex(ProtocolV8Error, "requires its logical test"):
                self.send(
                    "choose",
                    expr={"op": "literal", "value": ["bool", path == "use-left"]},
                    decision=["bool", path == "use-left"],
                    path=path,
                )
        with self.assertRaisesRegex(ProtocolV8Error, "logical test fields"):
            self.send(
                "choose",
                expr={
                    "op": "logical-test",
                    "operator": "&&",
                    "arg": {"op": "literal", "value": ["bool", True]},
                    "right": {"op": "local", "name": "not-yet-evaluated"},
                },
                decision=["bool", False],
                path="evaluate-right",
            )

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


class PromiseOrdinaryPropertyTests(unittest.TestCase):
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
        apply_protocol_record(self.machine, {"kind": "promise.intrinsics"})
        from megatoken.data.promise_scheduler_v8 import create_promise

        create_promise(self.machine, "p")

    def test_promise_uses_intrinsic_prototype_and_ordinary_own_properties(self):
        self.assertEqual(
            self.machine["heap"]["p"]["prototype"], ["ref", "intrinsicpromiseprototype"]
        )
        apply_protocol_record(
            self.machine,
            {
                "kind": "property.set",
                "object": "p",
                "key": ["str", "label"],
                "receiver": ["ref", "p"],
                "value": ["str", "ready"],
                "success": True,
            },
        )
        apply_protocol_record(
            self.machine,
            {
                "kind": "property.get",
                "object": "p",
                "key": ["str", "then"],
                "receiver": ["ref", "p"],
                "result": ["ref", "intrinsicpromiseprototypethen"],
            },
        )
        self.assertEqual(self.machine["heap"]["p"]["properties"]["label"], ["str", "ready"])

    def test_intrinsic_record_round_trips_descriptor_reflection(self):
        apply_protocol_record(
            self.machine,
            {
                "kind": "promise.intrinsic.get",
                "base": "Promise",
                "key": ["str", "prototype"],
                "result": ["ref", "intrinsicpromiseprototype"],
            },
        )
        apply_protocol_record(
            self.machine,
            {
                "kind": "property.getPrototypeOf",
                "object": "p",
                "result": ["ref", "intrinsicpromiseprototype"],
            },
        )
        apply_protocol_record(
            self.machine,
            {"kind": "property.isExtensible", "object": "p", "result": ["bool", True]},
        )
        record = {
            "kind": "property.descriptor",
            "object": "intrinsicpromiseprototype",
            "key": ["str", "then"],
            "descriptor": {
                "kind": "data",
                "value": ["ref", "intrinsicpromiseprototypethen"],
                "writable": True,
                "enumerable": False,
                "configurable": True,
            },
            "name": "descriptor",
        }
        apply_protocol_record(self.machine, record)
        self.assertEqual(
            self.machine["heap"]["descriptor"]["properties"]["writable"], ["bool", True]
        )

    def test_adoption_checks_effective_then_and_tolerates_unrelated_own_keys(self):
        from megatoken.data.promise_adoption_v8 import _assert_intrinsic_then

        _assert_intrinsic_then(self.machine, "p")
        self.machine["heap"]["p"]["properties"]["label"] = ["str", "ordinary"]
        _assert_intrinsic_then(self.machine, "p")
        self.machine["heap"]["p"]["properties"]["then"] = [
            "ref",
            "intrinsicpromiseprototypethen",
        ]
        _assert_intrinsic_then(self.machine, "p")
        self.machine["heap"]["p"]["properties"]["then"] = ["str", "changed"]
        with self.assertRaisesRegex(ProtocolV8Error, "then lookup"):
            _assert_intrinsic_then(self.machine, "p")

    def test_intrinsic_then_method_cannot_be_used_as_plain_resolution_value(self):
        from megatoken.data.promise_scheduler_v8 import assert_plain_resolution

        self.machine["heap"]["thenable"] = {
            "kind": "object",
            "properties": {"then": ["ref", "intrinsicpromiseprototypethen"]},
            "prototype": "ordinary",
        }
        with self.assertRaisesRegex(ProtocolV8Error, "callable then"):
            assert_plain_resolution(self.machine, ["ref", "thenable"])

    def test_default_array_intrinsic_overrides_are_not_fabricated(self):
        for key in ("constructor", "toString", "toLocaleString"):
            with self.subTest(key=key):
                self.machine["heap"]["array"] = {
                    "kind": "array",
                    "properties": {"length": ["num", 0]},
                    "prototype": "ordinary",
                    "descriptors": {
                        "length": {
                            "kind": "data",
                            "value": ["num", 0],
                            "writable": True,
                            "enumerable": False,
                            "configurable": False,
                        }
                    },
                }
                with self.assertRaisesRegex(ProtocolV8Error, "inherited intrinsic"):
                    apply_protocol_record(
                        self.machine,
                        {
                            "kind": "property.get",
                            "object": "array",
                            "key": ["str", key],
                            "receiver": ["ref", "array"],
                            "result": ["undefined"],
                        },
                    )

    def test_default_function_intrinsic_overrides_are_not_fabricated(self):
        self.machine["heap"]["function"] = {
            "kind": "closure",
            "properties": {},
            "code": None,
            "captures": [],
            "functionKind": "ordinary",
            "lexicalReceiver": None,
            "prototype": "ordinary",
        }
        for key in ("constructor", "toString"):
            with (
                self.subTest(key=key),
                self.assertRaisesRegex(ProtocolV8Error, "inherited intrinsic"),
            ):
                apply_protocol_record(
                    self.machine,
                    {
                        "kind": "property.get",
                        "object": "function",
                        "key": ["str", key],
                        "receiver": ["ref", "function"],
                        "result": ["undefined"],
                    },
                )

    def test_adoption_follows_actual_prototype_chain_and_refuses_getter_order(self):
        from megatoken.data.promise_adoption_v8 import _assert_intrinsic_then

        self.machine["heap"]["custom-prototype"] = {
            "kind": "object",
            "code": None,
            "captures": [],
            "prototype": ["ref", "intrinsicpromiseprototype"],
            "properties": {},
        }
        self.machine["heap"]["p"]["prototype"] = ["ref", "custom-prototype"]
        _assert_intrinsic_then(self.machine, "p")
        self.machine["heap"]["custom-prototype"]["properties"]["then"] = [
            "ref",
            "intrinsicpromiseprototypecatch",
        ]
        with self.assertRaisesRegex(ProtocolV8Error, "then lookup"):
            _assert_intrinsic_then(self.machine, "p")
        self.machine["heap"]["custom-prototype"]["properties"].pop("then")
        self.machine["heap"]["custom-prototype"].setdefault("descriptors", {})["then"] = {
            "kind": "accessor",
            "get": ["ref", "getter"],
            "set": ["undefined"],
            "enumerable": True,
            "configurable": True,
        }
        with self.assertRaisesRegex(ProtocolV8Error, "then lookup"):
            _assert_intrinsic_then(self.machine, "p")

    def test_materialized_object_prototype_lookup_preserves_unmodeled_array_methods(self):
        from megatoken.data.protocol_v8 import _lookup_descriptor

        self.machine["heap"]["o"] = {
            "kind": "object",
            "code": None,
            "captures": [],
            "prototype": "ordinary",
            "properties": {},
        }
        _, descriptor = _lookup_descriptor(self.machine, "o", "toString")
        self.assertEqual(descriptor["value"], ["ref", "intrinsicobjectprototypetostring"])
        self.machine["heap"]["a"] = {
            "kind": "array",
            "code": None,
            "captures": [],
            "prototype": "ordinary",
            "properties": {"length": ["num", 0]},
        }
        with self.assertRaisesRegex(ProtocolV8Error, "callable protocol"):
            _lookup_descriptor(self.machine, "a", "map")

    def test_implicit_prototype_same_value_and_object_prototype_immutability(self):
        self.machine["heap"]["o"] = {
            "kind": "object",
            "code": None,
            "captures": [],
            "prototype": "ordinary",
            "properties": {},
            "extensible": False,
        }
        apply_protocol_record(
            self.machine,
            {
                "kind": "property.prototype",
                "object": "o",
                "prototype": ["ref", "intrinsicobjectprototype"],
                "success": True,
            },
        )
        apply_protocol_record(
            self.machine,
            {
                "kind": "property.prototype",
                "object": "intrinsicobjectprototype",
                "prototype": ["ref", "o"],
                "success": False,
            },
        )

    def test_restored_accessor_receiver_must_match_retained_invocation(self):
        from megatoken.data.protocol_v8 import _external_machine, validate_protocol_state

        apply_protocol_record(
            self.machine,
            {
                "kind": "allocate",
                "name": "getter",
                "objectKind": "closure",
                "length": None,
                "properties": [],
                "code": "getter-code",
                "captures": [],
                "functionKind": "ordinary",
                "lexicalReceiver": None,
            },
        )
        apply_protocol_record(
            self.machine,
            {
                "kind": "allocate",
                "name": "object",
                "objectKind": "object",
                "length": None,
                "properties": [],
                "code": None,
                "captures": [],
                "functionKind": None,
                "lexicalReceiver": None,
            },
        )
        apply_protocol_record(
            self.machine,
            {
                "kind": "allocate",
                "name": "other",
                "objectKind": "object",
                "length": None,
                "properties": [],
                "code": None,
                "captures": [],
                "functionKind": None,
                "lexicalReceiver": None,
            },
        )
        apply_protocol_record(
            self.machine,
            {
                "kind": "property.define",
                "object": "object",
                "key": ["str", "x"],
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
        apply_protocol_record(
            self.machine,
            {
                "kind": "property.get.call",
                "id": "pending-get",
                "child": "getter-invocation",
                "object": "object",
                "key": ["str", "x"],
                "receiver": ["ref", "object"],
            },
        )
        valid = _external_machine(self.machine)
        self.assertTrue(validate_protocol_state(valid["protocol"], valid["heap"], valid))
        changed = copy.deepcopy(valid)
        changed["protocol"]["pendingCalls"][0]["receiver"] = ["ref", "other"]
        with self.assertRaisesRegex(ProtocolV8Error, "retained invocation"):
            validate_protocol_state(changed["protocol"], changed["heap"], changed)
        apply_protocol_record(
            self.machine,
            {
                "kind": "return",
                "invocation": "getter-invocation",
                "value": ["str", "returned"],
            },
        )
        returned = _external_machine(self.machine)
        self.assertTrue(validate_protocol_state(returned["protocol"], returned["heap"], returned))
        changed_returned = copy.deepcopy(returned)
        changed_returned["protocol"]["pendingCalls"][0]["receiver"] = ["ref", "other"]
        with self.assertRaisesRegex(ProtocolV8Error, "retained invocation"):
            validate_protocol_state(
                changed_returned["protocol"], changed_returned["heap"], changed_returned
            )

    def test_restored_mutable_then_is_checked_at_adoption(self):
        from megatoken.data.promise_adoption_v8 import _assert_intrinsic_then
        from megatoken.data.protocol_v8 import _external_machine, validate_protocol_state

        before = _external_machine(self.machine)
        self.assertTrue(validate_protocol_state(before["protocol"], before["heap"], before))
        self.machine["heap"]["intrinsicpromiseprototype"]["properties"]["then"] = [
            "ref",
            "intrinsicpromiseprototypecatch",
        ]
        changed = _external_machine(self.machine)
        self.assertTrue(validate_protocol_state(changed["protocol"], changed["heap"], changed))
        with self.assertRaisesRegex(ProtocolV8Error, "then lookup"):
            _assert_intrinsic_then(self.machine, "p")
        intrinsic = dict(changed["heap"])["intrinsicpromiseprototypethen"]
        intrinsic["intrinsicName"] = "forged"
        with self.assertRaisesRegex(ProtocolV8Error, "intrinsic callable identity changed"):
            validate_protocol_state(changed["protocol"], changed["heap"], changed)

    def test_restored_intrinsic_property_order_must_match_descriptors(self):
        from megatoken.data.protocol_v8 import _external_machine, validate_protocol_state

        changed = _external_machine(self.machine)
        heap = dict(changed["heap"])
        prototype = heap["intrinsicpromiseprototype"]
        prototype["keyOrder"].pop()
        with self.assertRaisesRegex(ProtocolV8Error, "property order"):
            validate_protocol_state(changed["protocol"], changed["heap"], changed)


if __name__ == "__main__":
    unittest.main()
