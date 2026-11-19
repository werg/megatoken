import unittest

from megatoken.data.protocol_v8 import (
    ProtocolV8Error,
    _external_machine,
    apply_protocol_record,
    create_protocol_state,
    validate_protocol_state,
)


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


if __name__ == "__main__":
    unittest.main()
