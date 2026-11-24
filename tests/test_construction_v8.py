import copy
import unittest

from megatoken.data.construction_v8 import _id as _construction_id
from megatoken.data.protocol_v8 import (
    ProtocolV8Error,
    _external_machine,
    _internal_machine,
    apply_protocol_record,
    create_protocol_state,
    validate_protocol_state,
)


class ConstructionIdentityBoundsTests(unittest.TestCase):
    def test_constructor_identity_bound_counts_utf16_code_units(self):
        _construction_id("😀" * 128)
        with self.assertRaisesRegex(ProtocolV8Error, "identity"):
            _construction_id("😀" * 129)


class ScheduledConstructorArrowTests(unittest.TestCase):
    def _running_job(self):
        from megatoken.data.native_v8 import _apply_detail, _initial_state

        state = _initial_state(
            {}, {"inputGraph": {"roots": [], "objects": []}, "invocationId": "root", "entry": "run"}
        )
        state["protocol"] = create_protocol_state()

        def protocol(record):
            internal = _internal_machine(state)
            apply_protocol_record(internal, record)
            state.clear()
            state.update(_external_machine(internal))

        def detail(kind, **fields):
            _apply_detail(
                state,
                kind,
                fields,
                {},
                root="root",
                functions=[{"name": "callback", "params": []}],
            )

        def allocate(identity, object_kind="object", function_kind=None):
            protocol({
                "kind": "allocate", "name": identity, "objectKind": object_kind,
                "length": None, "properties": [], "code": "callback" if function_kind else None,
                "captures": [], "functionKind": function_kind,
                "lexicalReceiver": ["undefined"] if function_kind == "arrow" else None,
            })

        allocate("Base", "closure", "ordinary")
        allocate("Base-prototype")
        protocol({"kind": "callable.define", "object": "Base", "constructorKind": "base", "classConstructor": True})
        protocol({"kind": "class.link", "object": "Base", "prototype": ["ref", "Base-prototype"], "heritage": ["undefined"]})
        allocate("callback", "closure", "arrow")
        protocol({"kind": "callable.define", "object": "callback", "constructorKind": None, "classConstructor": False})
        allocate("alternate", "closure", "arrow")
        protocol({"kind": "callable.define", "object": "alternate", "constructorKind": None, "classConstructor": False})
        allocate("otherReceiver")
        protocol({
            "kind": "construct.enter", "id": "construction1", "child": "constructor-body",
            "callee": ["ref", "Base"], "newTarget": ["ref", "Base"],
            "arguments": [], "receiver": ["ref", "constructedReceiver"], "mode": "ordinary", "owner": None,
        })
        for identity in ("callback", "alternate"):
            protocol({"kind": "callable.capture", "object": identity, "frame": "construction1", "thisCell": "constructor-body:this"})
        detail("promise", action="create", promise="source")
        detail("promise", action="resolve", promise="source", value=["num", 1])
        detail("promise", action="react", promise="source", job="reaction", reaction="then", closure="callback", target="result")
        protocol({"kind": "return", "invocation": "constructor-body", "value": ["undefined"]})
        entry = next(row for row in state["protocol"]["construction"]["frames"] if row["id"] == "construction1")
        receiver = copy.deepcopy(entry["receiver"])
        protocol({"kind": "construct.finish", "id": "construction1", "completion": {"kind": "normal", "value": receiver}})
        detail("return", invocation="root", value=["num", 23])
        detail("job", action="dispatch", job="reaction", owner="root")
        return state, detail, protocol, receiver

    def test_constructor_arrow_job_uses_retained_this_new_target_and_restores_caller_completion(self):
        state, detail, protocol, receiver = self._running_job()
        self.assertEqual(state["receiver"], receiver)
        self.assertEqual(dict(state["locals"])["job:reaction:this"], receiver)
        self.assertTrue(validate_protocol_state(state["protocol"], state["heap"], state))
        protocol({"kind": "construct.target", "invocation": "job:reaction", "result": ["ref", "Base"]})
        self.assertEqual(dict(state["locals"])["job:reaction:lexical-newTarget"], ["ref", "Base"])
        protocol({"kind": "return", "invocation": "job:reaction", "value": ["undefined"]})
        detail("job", action="complete", job="reaction", owner="root")
        job = dict(state["jobs"])["reaction"]
        self.assertEqual(job["status"], "completed")
        self.assertEqual(job["returned"], {"kind": "return", "value": ["undefined"]})
        self.assertEqual(state["pending"], job["priorPending"])
        self.assertEqual(state["lastReturn"], job["priorLastReturn"])
        self.assertEqual(state["pending"], {"kind": "return", "value": ["num", 23]})
        self.assertTrue(validate_protocol_state(state["protocol"], state["heap"], state))

    def test_constructor_arrow_job_checkpoint_rejects_forged_active_receiver_or_callee(self):
        state, _detail, _protocol, receiver = self._running_job()
        for mutate in (
            lambda forged: forged.update(receiver=["undefined"]),
            lambda forged: forged["locals"].__setitem__(slice(None), [[k, ["ref", "otherReceiver"] if k == "job:reaction:this" else v] for k, v in forged["locals"]]),
            lambda forged: forged["protocol"]["callFrames"][-1].update(callee=["ref", "alternate"]),
            lambda forged: dict(forged["heap"])["callback"]["callable"]["lexicalConstruction"].update(frame="missing"),
        ):
            forged = copy.deepcopy(state)
            mutate(forged)
            with self.assertRaises(ProtocolV8Error):
                validate_protocol_state(forged["protocol"], forged["heap"], forged)
        self.assertEqual(state["receiver"], receiver)


class ConstructionThisFaultTests(unittest.TestCase):
    MESSAGE = "Must call super constructor in derived class before accessing 'this' or returning from derived constructor"

    def setUp(self):
        from tests.test_number_v8 import machine

        self.machine = machine()

    def send(self, record):
        return apply_protocol_record(self.machine, record)

    def allocate(self, identity, function_kind=None, lexical_receiver=None):
        return self.send(
            {
                "kind": "allocate",
                "name": identity,
                "objectKind": "closure" if function_kind else "object",
                "length": None,
                "properties": [],
                "code": identity if function_kind else None,
                "captures": [],
                "functionKind": function_kind,
                "lexicalReceiver": lexical_receiver,
            }
        )

    def enter_derived(self):
        self.allocate("Base", "ordinary", None)
        self.allocate("Base-prototype")
        self.send({"kind": "callable.define", "object": "Base", "constructorKind": "base", "classConstructor": True})
        self.send({"kind": "class.link", "object": "Base", "prototype": ["ref", "Base-prototype"], "heritage": ["undefined"]})
        self.allocate("Derived", "ordinary", None)
        self.allocate("Derived-prototype")
        self.send({"kind": "callable.define", "object": "Derived", "constructorKind": "derived", "classConstructor": True})
        self.send({"kind": "class.link", "object": "Derived", "prototype": ["ref", "Derived-prototype"], "heritage": ["ref", "Base"]})
        self.send({
            "kind": "construct.enter",
            "id": "derived-construction",
            "child": "derived-body",
            "callee": ["ref", "Derived"],
            "newTarget": ["ref", "Derived"],
            "arguments": [],
            "receiver": ["undefined"],
            "mode": "ordinary",
            "owner": None,
        })

    def test_direct_derived_this_fault_allocates_a_branded_reference_error(self):
        self.enter_derived()
        self.send({"kind": "construct.this.fault", "invocation": "derived-body", "error": "reference-error"})
        error = self.machine["heap"]["reference-error"]
        self.assertEqual(error["declaredError"], "ReferenceError")
        self.assertEqual(error["properties"]["message"], ["str", self.MESSAGE])
        self.assertEqual(self.machine["lastReturn"], {"kind": "throw", "value": ["ref", "reference-error"]})
        external = _external_machine(self.machine)
        self.assertTrue(validate_protocol_state(external["protocol"], external["heap"], external))

    def test_captured_arrow_fault_requires_exact_live_constructor_cell_and_call_frame(self):
        self.enter_derived()
        self.allocate("reader", "arrow", ["undefined"])
        self.send({"kind": "callable.define", "object": "reader", "constructorKind": None, "classConstructor": False})
        self.send({"kind": "callable.capture", "object": "reader", "frame": "derived-construction", "thisCell": "derived-body:this"})
        self.send({"kind": "call", "child": "arrow-body", "callee": ["ref", "reader"], "receiver": ["undefined"], "arguments": []})
        before = copy.deepcopy(self.machine)
        self.send({"kind": "construct.this.fault", "invocation": "arrow-body", "error": "reference-error"})
        self.assertEqual(self.machine["heap"]["reference-error"]["declaredError"], "ReferenceError")
        self.assertEqual(self.machine["lastReturn"], {"kind": "throw", "value": ["ref", "reference-error"]})

        self.machine = copy.deepcopy(before)
        self.machine["protocol"]["callFrames"][-1]["callee"] = ["ref", "Derived"]
        with self.assertRaisesRegex(ProtocolV8Error, "active constructor body|captured arrow|constructor activation"):
            self.send({"kind": "construct.this.fault", "invocation": "arrow-body", "error": "forged-error"})

        # The public frame can only fault while its retained this cell remains uninitialized.
        self.machine = copy.deepcopy(before)
        self.machine["locals"]["derived-body:this"] = ["ref", "Derived-prototype"]
        with self.assertRaisesRegex(ProtocolV8Error, "uninitialized derived receiver"):
            self.send({"kind": "construct.this.fault", "invocation": "arrow-body", "error": "forged-error"})

    def test_this_fault_rejects_wrong_invocation_and_nonfresh_error_id(self):
        self.enter_derived()
        before = copy.deepcopy(self.machine)
        with self.assertRaisesRegex(ProtocolV8Error, "invocation"):
            self.send({"kind": "construct.this.fault", "invocation": "root", "error": "reference-error"})
        self.assertEqual(self.machine, before)
        with self.assertRaisesRegex(ProtocolV8Error, "identity|fresh"):
            self.send({"kind": "construct.this.fault", "invocation": "derived-body", "error": "Derived"})
        self.assertEqual(self.machine, before)


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
