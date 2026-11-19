import base64
import gzip
import json
import unittest
from pathlib import Path

from megatoken.data.algorithm_v7 import stable_algorithm_hash
from megatoken.data.native_v8 import NativeV8Error, validate_v8_artifact
from megatoken.data.protocol_v8 import (
    ProtocolV8Error,
    _external_machine,
    apply_protocol_record,
    create_protocol_state,
    decode_protocol_record,
    encode_protocol_record,
    replay_protocol_case,
    validate_protocol_state,
)

FIXTURE = Path(__file__).parent / "fixtures" / "semantic-protocol-v8.json"
REGION_FIXTURE = Path(__file__).parent / "fixtures" / "semantic-protocol-region-v8.json.gz.b64"
ITERATOR_FIXTURE = Path(__file__).parent / "fixtures" / "semantic-iterator-v8.json"


class ProtocolV8CodecTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.document = json.loads(FIXTURE.read_text())

    def test_interoperability_machine_cases_replay_and_restore_checkpoints(self):
        self.assertEqual(self.document["schema"], "semantic-protocol-fixtures/1")
        for case in self.document["cases"]:
            with self.subTest(case=case["id"]):
                self.assertEqual(case["qualification"], "controller-contract-fixture")
                self.assertEqual(replay_protocol_case(case), case["expected"])

    def test_roundtrips_symbol_property_and_completion_records(self):
        records = [
            {"kind": "symbol.create", "id": "o3", "description": ["str", "tag"]},
            {"kind": "symbol.for", "id": "o4", "key": "shared"},
            {
                "kind": "property.define",
                "object": "o1",
                "key": ["symbol", "o3"],
                "descriptor": {
                    "kind": "data",
                    "value": ["num", 4],
                    "writable": True,
                    "enumerable": False,
                    "configurable": True,
                },
                "success": True,
            },
            {
                "kind": "property.finish",
                "id": "g1",
                "completion": {"kind": "throw", "value": ["str", "boom"]},
            },
        ]
        for record in records:
            with self.subTest(kind=record["kind"]):
                line = encode_protocol_record(record)
                self.assertTrue(line.startswith("protocol "))
                self.assertEqual(decode_protocol_record(line), record)

    def test_rejects_symbol_values_in_legacy_shape_and_malformed_protocol(self):
        for record in (
            {"kind": "bind", "name": "s", "value": ["symbol", "o3", "extra"]},
            {
                "kind": "property.get",
                "object": "o1",
                "key": ["symbol"],
                "receiver": ["ref", "o1"],
                "result": ["undefined"],
            },
            {
                "kind": "symbol.read",
                "symbol": ["ref", "o3"],
                "operation": "description",
                "result": ["undefined"],
                "bind": None,
            },
            {"kind": "end", "completion": "normal"},
        ):
            with self.subTest(record=record), self.assertRaises(ProtocolV8Error):
                encode_protocol_record(record)

    def test_rejects_noncanonical_or_oversized_protocol_lines(self):
        record = {"kind": "bind", "name": "s", "value": ["symbol", "o3"]}
        line = encode_protocol_record(record)
        with self.assertRaises(ProtocolV8Error):
            decode_protocol_record(
                'protocol { "kind": "bind", "name": "s", "value": ["symbol","o3"] }'
            )
        with self.assertRaises(ProtocolV8Error):
            decode_protocol_record(line + " ")
        with self.assertRaises(ProtocolV8Error):
            decode_protocol_record("protocol " + " " * 65540)

    def test_protocol_snapshot_requires_full_symbols_and_exact_state_fields(self):
        snapshot = self.document["cases"][0]["expected"]["protocol"]
        self.assertTrue(
            validate_protocol_state(snapshot, dict(self.document["cases"][0]["expected"]["heap"]))
        )
        bad = json.loads(json.dumps(snapshot))
        bad["symbols"] = [
            row for row in bad["symbols"] if row.get("id") != "well-known:unscopables"
        ]
        with self.assertRaises(ProtocolV8Error):
            validate_protocol_state(bad)
        bad = json.loads(json.dumps(snapshot))
        bad["unchecked"] = True
        with self.assertRaises(ProtocolV8Error):
            validate_protocol_state(bad)


class PromiseJoinNativeTests(unittest.TestCase):
    def test_bounded_promise_all_join_uses_fifo_and_checks_settlement_claims(self):
        machine = {
            "protocol": create_protocol_state(), "heap": {
                "p1": {"kind": "promise", "properties": {}, "state": "fulfilled", "result": ["num", 7], "reactions": []},
                "p2": {"kind": "promise", "properties": {}, "state": "fulfilled", "result": ["str", "ok"], "reactions": []},
                "joined": {"kind": "promise", "properties": {}, "state": "pending", "result": None, "reactions": []},
            },
            "locals": {}, "stack": ["invocation"], "effects": [], "jobs": [], "ready": [],
            "scheduler": {"registration": 0, "ready": 0}, "continuations": [],
            "captureAliases": {}, "frameFunctions": {},
            "completion": None, "activeJob": None,
        }
        apply_protocol_record(machine, {"kind": "promise.join", "mode": "all", "result": "joined",
                                        "inputs": ["p1", "p2"], "jobs": ["j1", "j2"],
                                        "output": None, "error": None})
        self.assertEqual(machine["ready"], ["j1", "j2"])
        machine["stack"] = []
        first = {"kind": "promise.join.step", "result": "joined", "job": "j1", "entry": None,
                 "output": None, "error": None, "status": "pending", "value": None}
        apply_protocol_record(machine, first)
        second = {"kind": "promise.join.step", "result": "joined", "job": "j2", "entry": None,
                  "output": "out", "error": None, "status": "fulfilled", "value": ["ref", "out"]}
        apply_protocol_record(machine, second)
        self.assertEqual(machine["heap"]["joined"]["state"], "fulfilled")
        self.assertEqual(machine["heap"]["out"]["properties"]["1"], ["str", "ok"])
        external = _external_machine(machine)
        self.assertTrue(validate_protocol_state(external["protocol"], external["heap"], external))
        broken = json.loads(json.dumps(external))
        joined = dict(broken["heap"])["joined"]
        joined["join"]["jobs"][0] = "forged"
        with self.assertRaises(ProtocolV8Error):
            validate_protocol_state(broken["protocol"], broken["heap"], broken)
        with self.assertRaises(ProtocolV8Error):
            apply_protocol_record(machine, {**second, "job": "j1"})


class IteratorProtocolNativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.document = json.loads(ITERATOR_FIXTURE.read_text())

    def test_builtin_iterator_golden_cases_replay_exact_cursor_heap_and_results(self):
        self.assertEqual(self.document["schema"], "semantic-protocol-fixtures/1")
        self.assertEqual(self.document["profile"], "builtin-iterator/1")
        self.assertEqual(len(self.document["cases"]), 3)
        for case in self.document["cases"]:
            with self.subTest(case=case["id"]):
                self.assertEqual(case["qualification"], "controller-contract-fixture")
                self.assertEqual(replay_protocol_case(case), case["expected"])

    def test_noncanonical_and_false_iterator_identity_are_rejected(self):
        with self.assertRaises(ProtocolV8Error):
            encode_protocol_record(
                {
                    "kind": "iterator.create",
                    "name": "it",
                    "iterated": ["str", "x"],
                    "iterationKind": "function",
                }
            )
        bad = json.loads(
            json.dumps(
                next(
                    case for case in self.document["cases"] if case["id"] == "unicode-string-cursor"
                )
            )
        )
        position = next(
            i for i, line in enumerate(bad["lines"]) if '"kind":"iterator.identity"' in line
        )
        record = decode_protocol_record(bad["lines"][position])
        record["result"] = ["str", "different"]
        bad["lines"][position] = "protocol " + __import__(
            "megatoken.data.algorithm_v7", fromlist=["stable_algorithm_json"]
        ).stable_algorithm_json(record)
        with self.assertRaises(ProtocolV8Error):
            replay_protocol_case(bad)


class ProtocolV8NativeReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        raw = gzip.decompress(base64.b64decode(REGION_FIXTURE.read_bytes()))
        cls.artifact = json.loads(raw)

    def test_source_qualified_protocol_region_public_golden_replays_and_continues(self):
        self.assertEqual(self.artifact["schema"], "semantic-protocol-region-fixtures/1")
        expected = {
            "source-symbol-getter-arithmetic": (
                "f0f48dc8fcdc5e97e03334dc19d7f076f02f9f9c39095603070af4a0b2fc6c30",
                "00a245053b8a2c70089a755792a5149637afa9dadd59c2480bb10c4f3255c55a",
                11,
                3,
                8,
            ),
            "typed-views-shared-backing": (
                "bf88479ca947f7ae6250c493e458db1edac052b1ecd3ebdb8f8fe4f8b52e86d9",
                "519dce4e0d95f63c1070ec27c7d1c6f977110fb543f44b790b476006509def23",
                13,
                0,
                8,
            ),
            "source-derived-private-arithmetic": (
                "ea92a427b7c99365502f6b1df70d5bc60241079b987aea6695457187debb8f2d",
                "72825d06e63761462e0f00f01575bfa7d8126d20e3078a2e2c31a88f3a5cf80a",
                64,
                2,
                21,
            ),
            "builtin-iterator-mixed-arithmetic": (
                "a2e355eed02c63c41850dbff17cac4d5c8fc07b71ae5fccf214ab4d4b9de5c33",
                "9f0bc0d24903e7f3a1390745b5f59b6a41b76320d453709d677713bd76e03d8e",
                16,
                2,
                12,
            ),
        }
        self.assertEqual({case["id"] for case in self.artifact["cases"]}, set(expected))
        for golden in self.artifact["cases"]:
            case = golden["artifact"]
            with self.subTest(case=golden["id"]):
                public_digest, context_digest, protocol_records, regions, details = expected[
                    golden["id"]
                ]
                artifact = {"public": case["public"], "contexts": case["contexts"]}
                self.assertEqual(stable_algorithm_hash(case["public"]), public_digest)
                self.assertEqual(stable_algorithm_hash(case["contexts"]), context_digest)
                result = validate_v8_artifact(artifact)
                self.assertTrue(
                    all(
                        result["checks"][name] == "passed"
                        for name in ("replay", "contexts", "continuation")
                    )
                )
                self.assertEqual(result["counts"]["protocolRecords"], protocol_records)
                self.assertEqual(result["counts"]["regionRecords"], regions)
                self.assertEqual(result["counts"]["detailedRecords"], details)

    def test_protocol_context_tampering_is_rejected(self):
        case = self.artifact["cases"][0]["artifact"]
        bad = {
            "public": json.loads(json.dumps(case["public"])),
            "contexts": json.loads(json.dumps(case["contexts"])),
        }
        first = next(c for c in bad["contexts"] if c["state"]["protocol"] is not None)
        first["state"]["protocol"]["sequence"] += 1
        with self.assertRaises(NativeV8Error):
            validate_v8_artifact(bad)

    def test_protocol_value_cannot_be_relabelled_as_legacy_detail(self):
        case = self.artifact["cases"][0]["artifact"]
        bad = {
            "public": json.loads(json.dumps(case["public"])),
            "contexts": json.loads(json.dumps(case["contexts"])),
        }
        generated = bad["public"]["timeline"]
        target = next(r for r in generated if r["kind"] == "protocol")
        target["kind"] = "semantic"
        target["fields"]["line"] = "detail end {}"
        with self.assertRaises(NativeV8Error):
            validate_v8_artifact(bad)


class ConstructionProtocolNativeTests(unittest.TestCase):
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

    def allocate(self, name, kind="object", code=None):
        self.send(
            "allocate",
            name=name,
            objectKind=kind,
            length=None,
            properties=[],
            code=code,
            captures=[],
            functionKind="ordinary" if kind == "closure" else None,
            lexicalReceiver=None,
        )

    def class_(self, name, heritage=("undefined",)):
        self.allocate(name, "closure", name + "-code")
        self.allocate(name + "-prototype")
        self.send(
            "callable.define",
            object=name,
            constructorKind="base" if heritage[0] == "undefined" else "derived",
            classConstructor=True,
        )
        self.send(
            "class.link",
            object=name,
            prototype=["ref", name + "-prototype"],
            heritage=list(heritage),
        )

    def test_class_descriptor_and_constructor_receiver_follow_new_target(self):
        self.class_("Base")
        self.class_("Child", ("ref", "Base"))
        self.send(
            "construct.enter",
            id="construct-1",
            child="child-body",
            callee=["ref", "Base"],
            newTarget=["ref", "Child"],
            arguments=[],
            receiver=["ref", "instance"],
            mode="ordinary",
            owner=None,
        )
        self.assertEqual(self.machine["heap"]["instance"]["prototype"], ["ref", "Child-prototype"])
        self.send("construct.target", invocation="child-body", result=["ref", "Child"])
        self.send("construct.this", invocation="child-body", result=["ref", "instance"])
        self.send("return", value=["num", 2], invocation="child-body")
        with self.assertRaisesRegex(ProtocolV8Error, "pending constructor"):
            self.send("bind", name="bad", value=["num", 1])
        self.send(
            "construct.finish",
            id="construct-1",
            completion={"kind": "normal", "value": ["ref", "instance"]},
        )
        node = self.machine["heap"]["Base"]
        self.assertEqual(
            node["descriptors"]["prototype"],
            {"kind": "data", "writable": False, "enumerable": False, "configurable": False},
        )
        external = _external_machine(self.machine)
        validate_protocol_state(external["protocol"], external["heap"], external)

    def test_private_names_and_fields_share_heap_and_reject_false_claims(self):
        self.class_("C")
        self.allocate("instance")
        self.allocate("other")
        self.send("private.declare", id="o20", owner="C", name="secret")
        self.send(
            "private.define",
            object="instance",
            privateId="o20",
            elementKind="field",
            value=["num", 1],
        )
        self.send("private.has", object="instance", privateId="o20", result=True)
        self.send("private.set", object="instance", privateId="o20", value=["num", 3])
        self.send("private.get", object="instance", privateId="o20", result=["num", 3])
        self.send("private.has", object="other", privateId="o20", result=False)
        with self.assertRaisesRegex(ProtocolV8Error, "false private read"):
            self.send("private.get", object="instance", privateId="o20", result=["num", 99])
        serialized = self.machine["heap"]["instance"]["privateElements"]
        self.assertEqual(serialized["o20"], {"kind": "field", "value": ["num", 3], "version": 2})
        external = _external_machine(self.machine)
        validate_protocol_state(external["protocol"], external["heap"], external)

    def test_derived_super_binding_runs_initializer_before_body_resumes(self):
        self.class_("Base")
        self.class_("Derived", ("ref", "Base"))
        self.allocate("derived-init", "closure", "derived-init-code")
        self.send(
            "callable.define", object="derived-init", constructorKind=None, classConstructor=False
        )
        self.send("callable.home", object="derived-init", home=["ref", "Derived-prototype"])
        self.send(
            "class.initializer",
            object="Derived",
            placement="instance",
            initializer=["ref", "derived-init"],
        )
        self.send(
            "construct.enter",
            id="derived-construction",
            child="derived-body",
            callee=["ref", "Derived"],
            newTarget=["ref", "Derived"],
            arguments=[],
            receiver=["undefined"],
            mode="ordinary",
            owner=None,
        )
        self.send("super.select", owner="derived-construction", callee=["ref", "Base"])
        self.send(
            "construct.enter",
            id="base-construction",
            child="base-body",
            callee=["ref", "Base"],
            newTarget=["ref", "Derived"],
            arguments=[],
            receiver=["ref", "base-instance"],
            mode="super",
            owner="derived-construction",
        )
        self.send("return", value=["undefined"], invocation="base-body")
        self.send(
            "construct.finish",
            id="base-construction",
            completion={"kind": "normal", "value": ["ref", "base-instance"]},
        )
        self.send(
            "super.bind",
            owner="derived-construction",
            receiver=["ref", "base-instance"],
            child="derived-init-body",
        )
        external = _external_machine(self.machine)
        validate_protocol_state(external["protocol"], external["heap"], external)
        self.send("return", value=["undefined"], invocation="derived-init-body")
        self.send("super.finish", owner="derived-construction", result=["ref", "base-instance"])
        self.send("construct.this", invocation="derived-body", result=["ref", "base-instance"])
        self.send("return", value=["undefined"], invocation="derived-body")
        self.send(
            "construct.finish",
            id="derived-construction",
            completion={"kind": "normal", "value": ["ref", "base-instance"]},
        )
        external = _external_machine(self.machine)
        validate_protocol_state(external["protocol"], external["heap"], external)


class MemoryProtocolNativeTests(unittest.TestCase):
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

    def construct(self, constructor, name, args, backing=None):
        self.send(
            "memory.construct", constructor=constructor, name=name, backing=backing, arguments=args
        )

    def call(self, method, receiver, args, result, name=None):
        self.send(
            "memory.call",
            method=method,
            receiver=receiver,
            arguments=args,
            result=result,
            name=name,
        )

    def get(self, object, key, result):
        self.send("memory.get", object=object, key=["str", key], result=result)

    def set(self, object, key, value):
        self.send("memory.set", object=object, key=["str", key], value=value)

    def test_aliased_views_and_data_view_byte_order(self):
        self.construct("ArrayBuffer", "backing", [["num", 8]])
        self.construct("Uint8Array", "bytes", [["ref", "backing"]])
        self.construct("DataView", "view", [["ref", "backing"]])
        self.set("bytes", "0", ["num", 257])
        self.call(
            "DataView.prototype.setUint16",
            ["ref", "view"],
            [["num", 2], ["num", 0x1234], ["bool", True]],
            ["undefined"],
        )
        self.get("bytes", "0", ["num", 1])
        self.get("bytes", "2", ["num", 0x34])
        self.get("bytes", "3", ["num", 0x12])
        self.call(
            "DataView.prototype.getUint16",
            ["ref", "view"],
            [["num", 2], ["bool", False]],
            ["num", 0x3412],
        )
        self.get("view", "buffer", ["ref", "backing"])
        before = json.loads(json.dumps(self.machine))
        with self.assertRaisesRegex(ProtocolV8Error, "false binary-memory read"):
            self.get("bytes", "0", ["num", 257])
        self.assertEqual(self.machine, before)
        self.call(
            "ArrayBuffer.prototype.slice",
            ["ref", "backing"],
            [["num", 1], ["num", 4]],
            ["ref", "copy"],
            "copy",
        )
        self.construct("Uint8Array", "copied", [["ref", "copy"]])
        self.set("bytes", "2", ["num", 9])
        self.get("copied", "1", ["num", 0x34])
        external = _external_machine(self.machine)
        validate_protocol_state(external["protocol"], external["heap"], external)

    def test_shared_growable_buffer_retains_tracking_view_and_atomics(self):
        self.machine["heap"]["options"] = {
            "kind": "object",
            "properties": {"maxByteLength": ["num", 16]},
        }
        self.construct("SharedArrayBuffer", "shared", [["num", 4], ["ref", "options"]])
        self.construct("Int32Array", "words", [["ref", "shared"]])
        self.construct("Uint8Array", "fixed", [["ref", "shared"], ["num", 0], ["num", 4]])
        self.call("Atomics.store", None, [["ref", "words"], ["num", 0], ["num", 257]], ["num", 257])
        self.get("fixed", "0", ["num", 1])
        self.call(
            "SharedArrayBuffer.prototype.grow", ["ref", "shared"], [["num", 12]], ["undefined"]
        )
        self.get("words", "length", ["num", 3])
        self.get("fixed", "length", ["num", 4])
        self.get("words", "2", ["num", 0])
        external = _external_machine(self.machine)
        validate_protocol_state(external["protocol"], external["heap"], external)

    def test_binary_state_rejects_forged_bytes_views_and_ordinary_properties(self):
        self.construct("ArrayBuffer", "buffer", [["num", 8]])
        self.construct("Uint16Array", "view", [["ref", "buffer"], ["num", 2], ["num", 2]])
        external = _external_machine(self.machine)
        validate_protocol_state(external["protocol"], external["heap"], external)

        def node(state, identity):
            return next(row[1] for row in state["heap"] if row[0] == identity)

        for mutate in (
            lambda state: node(state, "buffer")["bytes"].__setitem__(0, 256),
            lambda state: node(state, "view").__setitem__("bufferId", "missing"),
            lambda state: node(state, "view").__setitem__("byteOffset", 3),
            lambda state: node(state, "view")["properties"].append(["custom", ["num", 1]]),
            lambda state: node(state, "buffer").__setitem__("version", 0),
            lambda state: node(state, "buffer").__setitem__("resizable", True),
        ):
            bad = json.loads(json.dumps(external))
            mutate(bad)
            with self.assertRaises(ProtocolV8Error):
                validate_protocol_state(bad["protocol"], bad["heap"], bad)

    def test_false_binary_setter_completion_does_not_commit_backing_bytes(self):
        self.construct("ArrayBuffer", "buffer", [["num", 4]])
        self.construct("DataView", "view", [["ref", "buffer"]])
        before = json.loads(json.dumps(self.machine))
        with self.assertRaisesRegex(ProtocolV8Error, "false binary-memory call result"):
            self.call(
                "DataView.prototype.setUint8", ["ref", "view"], [["num", 0], ["num", 9]], ["num", 9]
            )
        self.assertEqual(self.machine, before)


if __name__ == "__main__":
    unittest.main()
