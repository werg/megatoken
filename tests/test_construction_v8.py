import copy
import unittest

from megatoken.data.protocol_v8 import (
    ProtocolV8Error,
    _external_machine,
    _internal_machine,
    apply_protocol_record,
    create_protocol_state,
    validate_protocol_state,
)
from megatoken.data.construction_v8 import _id as _construction_id


class ConstructionIdentityBoundsTests(unittest.TestCase):
    def test_constructor_identity_bound_counts_utf16_code_units(self):
        _construction_id("😀" * 128)
        with self.assertRaisesRegex(ProtocolV8Error, "identity"):
            _construction_id("😀" * 129)


class ConstructionInitializerTests(unittest.TestCase):
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

    def set_up_class_and_arrow_initializer(self):
        self.send(
            "allocate",
            name="C",
            objectKind="closure",
            length=None,
            properties=[],
            code="class-code",
            captures=[],
            functionKind="ordinary",
            lexicalReceiver=None,
        )
        self.send(
            "allocate",
            name="C-prototype",
            objectKind="object",
            length=None,
            properties=[],
            code=None,
            captures=[],
            functionKind=None,
            lexicalReceiver=None,
        )
        self.send("callable.define", object="C", constructorKind="base", classConstructor=True)
        self.send(
            "class.link", object="C", prototype=["ref", "C-prototype"], heritage=["undefined"]
        )
        self.send(
            "allocate",
            name="other",
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
            name="arrow-init",
            objectKind="closure",
            length=None,
            properties=[],
            code="arrow-init-code",
            captures=[],
            functionKind="arrow",
            lexicalReceiver=["ref", "other"],
        )
        self.send(
            "callable.define", object="arrow-init", constructorKind=None, classConstructor=False
        )
        # callable.home already rejects arrow closures. Model malformed or
        # restored metadata to exercise the initializer invariant itself.
        self.machine["heap"]["arrow-init"]["callable"]["homeObject"] = ["ref", "C-prototype"]

    def test_arrow_instance_initializer_is_rejected_at_declaration(self):
        self.set_up_class_and_arrow_initializer()
        with self.assertRaisesRegex(ProtocolV8Error, "class initializer cannot have lexical this"):
            self.send(
                "class.initializer",
                object="C",
                placement="instance",
                initializer=["ref", "arrow-init"],
            )

    def test_checkpoint_cannot_restore_arrow_initializer_metadata(self):
        self.set_up_class_and_arrow_initializer()
        self.machine["heap"]["C"]["callable"]["instanceInitializer"] = ["ref", "arrow-init"]
        external = _external_machine(self.machine)
        with self.assertRaisesRegex(ProtocolV8Error, "class initializer cannot have lexical this"):
            validate_protocol_state(external["protocol"], external["heap"], external)


class ConstructionAbruptCompletionTests(unittest.TestCase):
    def setUp(self):
        from tests.test_number_v8 import machine

        self.machine = machine()

    def send(self, kind, **fields):
        return apply_protocol_record(self.machine, {"kind": kind, **fields})

    def allocate(self, name, kind):
        from tests.test_number_v8 import allocate

        allocate(self.machine, name, kind)

    def class_(self, name, parent=None):
        self.allocate(name, "closure")
        self.allocate(name + "-prototype", "object")
        self.send(
            "callable.define",
            object=name,
            constructorKind="derived" if parent else "base",
            classConstructor=True,
        )
        self.send(
            "class.link",
            object=name,
            prototype=["ref", name + "-prototype"],
            heritage=["ref", parent] if parent else ["undefined"],
        )

    def initializer(self, owner):
        name = owner + "-init"
        self.allocate(name, "closure")
        self.send("callable.define", object=name, constructorKind=None, classConstructor=False)
        self.send("callable.home", object=name, home=["ref", owner + "-prototype"])
        self.send(
            "class.initializer", object=owner, placement="instance", initializer=["ref", name]
        )

    def checkpoint(self):
        shared = _external_machine(self.machine)
        self.assertTrue(validate_protocol_state(shared["protocol"], shared["heap"], shared))
        self.machine = _internal_machine(shared)

    def test_throwing_base_initializer_finishes_without_entering_body(self):
        self.class_("Base")
        self.initializer("Base")
        self.send(
            "construct.enter",
            id="construction",
            child="init-body",
            callee=["ref", "Base"],
            newTarget=["ref", "Base"],
            arguments=[],
            receiver=["ref", "instance"],
            mode="ordinary",
            owner=None,
        )
        self.send("throw", invocation="init-body", value=["str", "boom"])
        self.checkpoint()
        before = copy.deepcopy(self.machine)
        with self.assertRaisesRegex(ProtocolV8Error, "requires construct.finish"):
            self.send("construct.body", id="construction", child="body")
        self.assertEqual(self.machine, before)
        with self.assertRaisesRegex(ProtocolV8Error, "false constructor completion"):
            self.send(
                "construct.finish",
                id="construction",
                completion={"kind": "normal", "value": ["ref", "instance"]},
            )
        self.assertEqual(self.machine, before)
        self.send(
            "construct.finish",
            id="construction",
            completion={"kind": "throw", "value": ["str", "boom"]},
        )
        self.checkpoint()
        self.assertNotIn("body", self.machine["frameFunctions"])
        self.send("throw", invocation="root", value=["str", "boom"])
        self.send("end", completion="throw")
        self.assertEqual(self.machine["completion"], {"kind": "throw", "value": ["str", "boom"]})

    def test_throwing_derived_initializer_resumes_abrupt_completion(self):
        self.class_("Base")
        self.class_("Derived", "Base")
        self.initializer("Derived")
        self.send(
            "construct.enter",
            id="derived",
            child="derived-body",
            callee=["ref", "Derived"],
            newTarget=["ref", "Derived"],
            arguments=[],
            receiver=["undefined"],
            mode="ordinary",
            owner=None,
        )
        self.send("super.select", owner="derived", callee=["ref", "Base"])
        self.send(
            "construct.enter",
            id="base",
            child="base-body",
            callee=["ref", "Base"],
            newTarget=["ref", "Derived"],
            arguments=[],
            receiver=["ref", "instance"],
            mode="super",
            owner="derived",
        )
        self.send("return", invocation="base-body", value=["undefined"])
        self.send(
            "construct.finish",
            id="base",
            completion={"kind": "normal", "value": ["ref", "instance"]},
        )
        self.send("super.bind", owner="derived", child="init-body", receiver=["ref", "instance"])
        self.send("throw", invocation="init-body", value=["str", "boom"])
        self.checkpoint()
        before = copy.deepcopy(self.machine)
        with self.assertRaisesRegex(ProtocolV8Error, "false derived initializer completion"):
            self.send(
                "super.finish",
                owner="derived",
                completion={"kind": "normal", "value": ["ref", "instance"]},
            )
        self.assertEqual(self.machine, before)
        self.send(
            "super.finish", owner="derived", completion={"kind": "throw", "value": ["str", "boom"]}
        )
        self.checkpoint()
        self.send("throw", invocation="derived-body", value=["str", "boom"])
        self.send(
            "construct.finish", id="derived", completion={"kind": "throw", "value": ["str", "boom"]}
        )
        self.checkpoint()
        self.send("throw", invocation="root", value=["str", "boom"])
        self.send("end", completion="throw")
        self.assertEqual(self.machine["completion"], {"kind": "throw", "value": ["str", "boom"]})


if __name__ == "__main__":
    unittest.main()
