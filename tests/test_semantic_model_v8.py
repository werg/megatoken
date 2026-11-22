"""Protocol tests use an explicit fake decoder; no model-quality claim follows."""

import base64
import copy
import gzip
import json
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from megatoken.contracts import ROLE_TO_ID
from megatoken.controls import semantic_selection_instruction_v8
from megatoken.data.native_v8 import prepare_native_v8_training
from megatoken.generation.codec import ByteCodec
from megatoken.multitrack_runtime import TrackSession
from megatoken.semantic_model_v8 import NativeSelectionV8Error, NativeSemanticInferenceV8
from tests.test_prediction_prefix_v8 import examples, prefix


@dataclass
class Memory:
    session_id: str
    state: object
    agenda: object
    scheduler: object
    writes: object = None
    sources: object = None
    lock: object = None


class FakeRuntime:
    """Only physical I/O and transaction boundaries; outputs are supplied by this test."""

    def __init__(self, fragments):
        self.codec = ByteCodec.byte_fallback()
        self.generator = SimpleNamespace(
            generator=SimpleNamespace(codec=self.codec, config=SimpleNamespace(max_capacity=256))
        )
        self.memory = SimpleNamespace(sessions={}, close_session=self.close_session)
        self.fragments = list(fragments)
        self.observations = []
        self.controls = []

    def open(self, **kwargs):
        self.metadata = kwargs["metadata"]
        memory = Memory("memory1", 0, {}, SimpleNamespace(lookups=0))
        self.memory.sessions[memory.session_id] = memory
        return TrackSession(memory)

    def observe(self, session, events, **kwargs):
        for event in events:
            self.observations.append(event)
            session.context_archive.put(event)
            session.occupied_events.add(event.event_id)
            session.history.append(event.payload)
        session.memory.state += 1
        session.revision += 1

    def generate(self, session, *, controls, event_ids, **kwargs):
        payload, terminal = self.fragments.pop(0)
        self.controls.extend(controls)
        session.memory.state += 1
        session.revision += 1
        session.occupied_events.update(event_ids)
        return SimpleNamespace(
            tape=SimpleNamespace(
                payloads=(payload,),
                terminal=(terminal,),
                valid=(True,),
                modalities=("text",),
                tracks=("assistant",),
                emitted_ids=(tuple(self.codec.encode(payload)),),
                event_ids=event_ids,
                meridian_id=session.revision,
            )
        )

    def release_generation(self, session, revision):
        pass

    def context_status(self, session, **kwargs):
        return {"visible_events": sorted(session.occupied_events), "cache_bytes": session.state * 8}

    def close_session(self, memory):
        self.memory.sessions.pop(memory.session_id, None)


class NativeServingV8Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.example = examples("semantic-regions-v8.json.gz.b64")[0]

    def setup_adapter(self, fragments):
        runtime = FakeRuntime(fragments)
        adapter = NativeSemanticInferenceV8(runtime, model_identity="a" * 64)
        adapter.initialize(
            seed=17, checkpointIdentity="a" * 64, tokenizerIdentity=runtime.codec.fingerprint
        )
        return adapter, runtime

    def request(self, adapter, index=0):
        public, contexts, context = prefix(self.example, index)
        return adapter.select(
            session=public["timeline"][0]["fields"]["session"],
            public=public,
            contexts=contexts,
            context=context,
        )

    def acknowledge(self, adapter, result, index=0):
        before, contexts, context = prefix(self.example, index)
        public = copy.deepcopy(self.example["public"])
        public["timeline"] = public["timeline"][: context["visibleThrough"] + 1]
        return adapter.commit(
            session=before["timeline"][0]["fields"]["session"],
            public=public,
            contexts=contexts + [context],
            invocationId=context["invocationId"],
            record=result["record"],
            selection=result["selection"],
        )

    def target(self, index=0):
        context = self.example["contexts"][index]
        return self.example["public"]["timeline"][context["visibleThrough"]]["fields"][
            "line"
        ].encode()

    def test_original_fragment_ids_and_bytes_survive_explicit_commit(self):
        target = self.target()
        adapter, runtime = self.setup_adapter([(target[:20], False), (target[20:], True)])
        result = self.request(adapter)
        identity = result["selection"]["session"]
        self.assertEqual(adapter.sessions[identity].runtime.state, 0)
        self.assertEqual(result["selection"]["payloadHex"], target.hex())
        self.assertEqual(
            b"".join(
                runtime.codec.decode(row["emittedTokenIds"])
                for row in result["selection"]["fragments"]
            ),
            target,
        )
        self.assertEqual(len(runtime.controls), 2)
        self.assertIn(semantic_selection_instruction_v8().encode(), runtime.controls[0])
        self.assertIn(b"Continue the exact current record bytes", runtime.controls[1])
        with self.assertRaisesRegex(ValueError, "requires commit or rejection"):
            self.request(adapter)
        self.assertTrue(self.acknowledge(adapter, result)["committed"])
        self.assertGreater(adapter.sessions[identity].runtime.state, 0)
        self.assertEqual(adapter.metrics()["committed"], 1)

    def test_rejected_candidate_leaves_native_state_available_for_retry(self):
        target = self.target()
        adapter, runtime = self.setup_adapter([(target, True), (target, True)])
        first = self.request(adapter)
        identity = first["selection"]["session"]
        adapter.reject(
            session=identity, selectionId=first["selection"]["selectionId"], reason="test refusal"
        )
        self.assertEqual(adapter.sessions[identity].runtime.state, 0)
        second = self.request(adapter)
        self.acknowledge(adapter, second)
        metrics = adapter.metrics()
        self.assertEqual(metrics["predictionAttempts"], 2)
        self.assertEqual(metrics["failedAttempts"], 1)
        self.assertEqual(metrics["generatedTokens"], 2 * len(runtime.codec.encode(target)))

    def test_native_commit_independently_rejects_false_semantics(self):
        target = self.target().replace(b'"num",4', b'"num",9', 1)
        self.assertNotEqual(target, self.target())
        adapter, _ = self.setup_adapter([(target, True)])
        result = self.request(adapter)
        before, contexts, context = prefix(self.example, 0)
        public = copy.deepcopy(self.example["public"])
        public["timeline"] = public["timeline"][: context["visibleThrough"] + 1]
        public["timeline"][-1]["fields"]["line"] = target.decode()
        with self.assertRaises(ValueError):
            adapter.commit(
                session=result["selection"]["session"],
                public=public,
                contexts=contexts + [context],
                invocationId=context["invocationId"],
                record=result["record"],
                selection=result["selection"],
            )
        self.assertEqual(adapter.sessions[result["selection"]["session"]].runtime.state, 0)

    def test_failed_emission_retains_original_cost_and_can_retry(self):
        adapter, runtime = self.setup_adapter([(b"\xff", True), (self.target(), True)])
        with self.assertRaises(NativeSelectionV8Error) as error:
            self.request(adapter)
        self.assertEqual(error.exception.evidence["payloadHex"], "ff")
        self.assertEqual(error.exception.evidence["generatedTokens"], 1)
        self.assertEqual(adapter.metrics()["failedAttempts"], 1)
        self.acknowledge(adapter, self.request(adapter))
        self.assertEqual(adapter.metrics()["committed"], 1)

    def test_runtime_invalid_first_and_later_fragments_keep_all_original_bytes(self):
        for earlier in (b"", b"prefix "):
            with self.subTest(earlier=earlier):
                fragments = ([(earlier, False)] if earlier else []) + [
                    (b"\xff", True),
                    (self.target(), True),
                ]
                adapter, runtime = self.setup_adapter(fragments)
                generate = runtime.generate

                def reject_utf8(*args, generate=generate, **kwargs):
                    result = generate(*args, **kwargs)
                    if result.tape.payloads == (b"\xff",):
                        result.tape.valid = (False,)
                    return result

                runtime.generate = reject_utf8
                with self.assertRaisesRegex(
                    NativeSelectionV8Error, "invalid v8 text fragment"
                ) as caught:
                    self.request(adapter)
                evidence = caught.exception.evidence
                self.assertEqual(evidence["payloadHex"], (earlier + b"\xff").hex())
                self.assertEqual(evidence["rawPayloadHex"], evidence["payloadHex"])
                self.assertEqual(
                    evidence["payloadHex"],
                    "".join(part["payloadHex"] for part in evidence["fragments"]),
                )
                self.assertEqual(evidence["generatedTokens"], len(earlier) + 1)
                validation = evidence["fragments"][-1]["validation"]
                self.assertEqual(
                    validation,
                    {
                        "shapeValid": True,
                        "runtimeValid": False,
                        "tokenIdsValid": True,
                        "tokenBytesMatch": True,
                        "tokenDecodeError": None,
                        "utf8Valid": False,
                    },
                )
                self.assertEqual(
                    evidence["fragments"][-1]["rejectedNativeOutputs"]["payloadsHex"], ["ff"]
                )
                self.assertEqual(evidence["status"], "rejected")
                work = adapter.sessions[evidence["session"]]
                self.assertEqual(work.runtime.state, 0)
                self.assertIsNone(work.pending)
                self.assertEqual(adapter.metrics()["committed"], 0)
                self.acknowledge(adapter, self.request(adapter))
                self.assertEqual(adapter.metrics()["committed"], 1)

    def test_malformed_rejected_outputs_retain_raw_evidence_without_aggregate_provenance(self):
        changes = {
            "multiple-outputs": {
                "payloads": (b"a", b"b"),
                "emitted_ids": ((97,), (98,)),
                "terminal": (True, True),
                "event_ids": ("a", "b"),
            },
            "not-bytes": {"payloads": ("not bytes",)},
            "token-byte-disagreement": {"payloads": (b"a",), "emitted_ids": ((98,),)},
            "negative-token": {"emitted_ids": ((-1,),)},
            "boolean-token": {"emitted_ids": ((True,),)},
            "out-of-range-token": {"emitted_ids": ((256,),)},
            "missing-terminal": {"terminal": ()},
            "wrong-track": {"tracks": ("other",)},
        }
        for label, changed in changes.items():
            with self.subTest(label=label):
                adapter, runtime = self.setup_adapter([(b"prefix", False), (b"a", True)])
                generate = runtime.generate

                def malformed(*args, changed=changed, generate=generate, **kwargs):
                    result = generate(*args, **kwargs)
                    if result.tape.payloads == (b"a",):
                        for key, value in changed.items():
                            setattr(result.tape, key, value)
                    return result

                runtime.generate = malformed
                with self.assertRaisesRegex(
                    NativeSelectionV8Error, "invalid v8 text fragment"
                ) as caught:
                    self.request(adapter)
                evidence = caught.exception.evidence
                self.assertIsNone(evidence["payloadHex"])
                self.assertEqual(len(evidence["fragments"]), 2)
                raw = evidence["fragments"][-1]["rejectedNativeOutputs"]
                payloads = changed.get("payloads", (b"a",))
                self.assertEqual(
                    raw["payloadsHex"],
                    [value.hex() if isinstance(value, bytes) else None for value in payloads],
                )
                self.assertEqual(
                    raw["emittedTokenIds"],
                    [list(values) for values in changed.get("emitted_ids", ((97,),))],
                )
                self.assertEqual(
                    evidence["generatedTokens"],
                    6 + sum(len(values) for values in raw["emittedTokenIds"]),
                )
                self.assertEqual(adapter.metrics()["committed"], 0)
                self.assertEqual(adapter.sessions[evidence["session"]].runtime.state, 0)
                self.assertIsNone(adapter.sessions[evidence["session"]].pending)

    def test_all_eight_original348_invalid_outputs_keep_rejected_final_bytes(self):
        # These are retained actual failed output bytes, supplied to the explicit
        # FakeRuntime. Its byte-fallback IDs are not the original model token IDs;
        # those remain separately retained in the fixture and original ZIP.
        path = (
            Path(__file__).parent / "fixtures/semantic-v8-original348-invalid-fragments.json.gz.b64"
        )
        fixture = json.loads(gzip.decompress(base64.b64decode(path.read_bytes(), validate=False)))
        self.assertEqual(fixture["originalRun"], 45046754653)
        self.assertEqual(len(fixture["cases"]), 8)
        self.assertEqual(len({row["caseId"] for row in fixture["cases"]}), 8)
        for original in fixture["cases"]:
            with self.subTest(case=original["caseId"]):
                fragments = [
                    (bytes.fromhex(row["payloadHex"]), row["terminal"])
                    for row in original["fragments"]
                ]
                prefix_bytes = b"".join(payload for payload, _ in fragments[:-1])
                self.assertEqual(
                    bytes.fromhex(original["originalAggregatePayloadHex"] or ""), prefix_bytes
                )
                self.assertFalse(original["originalProvenanceValidated"])
                for payload, _ in fragments[:-1]:
                    payload.decode("utf-8", errors="strict")
                with self.assertRaises(UnicodeDecodeError):
                    fragments[-1][0].decode("utf-8", errors="strict")
                adapter, runtime = self.setup_adapter(fragments)
                generate = runtime.generate

                def reproduce_invalid(*args, generate=generate, **kwargs):
                    result = generate(*args, **kwargs)
                    try:
                        result.tape.payloads[0].decode("utf-8", errors="strict")
                    except UnicodeDecodeError:
                        result.tape.valid = (False,)
                    return result

                runtime.generate = reproduce_invalid
                with self.assertRaisesRegex(
                    NativeSelectionV8Error, "invalid v8 text fragment"
                ) as caught:
                    self.request(adapter)
                evidence = caught.exception.evidence
                self.assertEqual(
                    evidence["payloadHex"], b"".join(payload for payload, _ in fragments).hex()
                )
                self.assertEqual(
                    [row["payloadHex"] for row in evidence["fragments"]],
                    [row["payloadHex"] for row in original["fragments"]],
                )
                self.assertEqual(
                    evidence["generatedTokens"], sum(len(payload) for payload, _ in fragments)
                )
                self.assertFalse(evidence["fragments"][-1]["validation"]["runtimeValid"])
                self.assertTrue(evidence["fragments"][-1]["validation"]["tokenBytesMatch"])
                self.assertEqual(adapter.metrics()["committed"], 0)
                self.assertEqual(adapter.sessions[evidence["session"]].runtime.state, 0)
                self.assertIsNone(adapter.sessions[evidence["session"]].pending)

    def test_continuation_requires_visible_context_and_every_earlier_fragment(self):
        target = self.target()
        for missing in ("context", "fragment"):
            with self.subTest(missing=missing):
                adapter, runtime = self.setup_adapter([(target[:20], False), (target[20:], True)])
                original_status = runtime.context_status

                def context_status(
                    session,
                    *,
                    runtime=runtime,
                    original_status=original_status,
                    missing=missing,
                    **kwargs,
                ):
                    status = original_status(session, **kwargs)
                    if runtime.controls:
                        event = (
                            next(
                                row.event_id
                                for row in runtime.observations
                                if row.role_id == ROLE_TO_ID["system"]
                            )
                            if missing == "context"
                            else "v8:0:fragment:0"
                        )
                        status["visible_events"].remove(event)
                    return status

                runtime.context_status = context_status
                with self.assertRaisesRegex(NativeSelectionV8Error, "no longer visible") as error:
                    self.request(adapter)
                self.assertEqual(len(runtime.controls), 1)
                self.assertEqual(len(error.exception.evidence["fragments"]), 1)
                self.assertEqual(error.exception.evidence["payloadHex"], target[:20].hex())
                identity = error.exception.evidence["session"]
                self.assertEqual(adapter.sessions[identity].runtime.state, 0)
                self.assertIsNone(adapter.sessions[identity].pending)

    def test_observed_native_inputs_match_training_preparation(self):
        adapter, runtime = self.setup_adapter([(self.target(), True)])
        self.request(adapter)
        episode, _ = prepare_native_v8_training(
            self.example, codec=runtime.codec, max_positions=256, max_history=1024
        )
        expected = []
        for record in episode.records:
            if record.policy_mask:
                break
            expected.append(
                (
                    record.payload,
                    ROLE_TO_ID[record.role],
                    record.part_index + 1 == record.part_count,
                )
            )
        self.assertEqual(
            [(row.payload, row.role_id, row.terminal) for row in runtime.observations], expected
        )
        self.assertEqual(
            json.loads(episode.supervision.annotations_json)["session_metadata"], runtime.metadata
        )
        target_instruction = next(
            event.instruction
            for group in episode.meridians
            for event in group.events
            if event.instruction
        )
        self.assertEqual(target_instruction, semantic_selection_instruction_v8())

    def test_wrong_initialization_and_unacknowledged_teacher_history_are_refused(self):
        adapter, runtime = self.setup_adapter([(self.target(), True)])
        with self.assertRaisesRegex(ValueError, "acknowledged native history"):
            self.request(adapter, 1)
        self.assertEqual(adapter.sessions, {})
        self.assertEqual(runtime.memory.sessions, {})
        self.assertEqual(adapter.metrics()["sessionOpenings"], [])
        self.assertFalse(hasattr(runtime, "metadata"))
        self.assertEqual(runtime.observations, [])
        self.assertEqual(runtime.controls, [])
        with self.assertRaisesRegex(ValueError, "identity differs"):
            adapter.initialize(seed=17, checkpointIdentity="a" * 64, tokenizerIdentity="b" * 64)
        with self.assertRaisesRegex(ValueError, "Initialize"):
            self.request(adapter)

    def test_acknowledgement_requires_the_complete_original_selection_evidence(self):
        adapter, _ = self.setup_adapter([(self.target(), True)])
        result = self.request(adapter)
        identity = result["selection"]["session"]
        alterations = [
            ("origin", "controlled"),
            ("modelSelected", 1),
            ("fragments", []),
            ("session", "another-session"),
            ("generatedTokens", 0),
            ("extra", "rewritten"),
        ]
        for key, value in alterations:
            with self.subTest(field=key):
                changed = copy.deepcopy(result)
                changed["selection"][key] = value
                with self.assertRaisesRegex(ValueError, "original native selection"):
                    self.acknowledge(adapter, changed)
                self.assertIsNotNone(adapter.sessions[identity].pending)
                self.assertEqual(adapter.sessions[identity].runtime.state, 0)
                with self.assertRaisesRegex(ValueError, "requires commit or rejection"):
                    self.request(adapter)
        self.assertTrue(self.acknowledge(adapter, result)["committed"])

    def test_acknowledged_host_suffix_cannot_be_dropped_and_is_ingested_once(self):
        from megatoken.controls import canonical_semantic_observation
        from megatoken.data.algorithm_v7 import stable_algorithm_hash
        from megatoken.data.native_v8 import _replay_shared_state
        from megatoken.data.schema import make_records

        adapter, runtime = self.setup_adapter([(self.target(0), True), (self.target(1), True)])
        selected = self.request(adapter)
        public, contexts, context = prefix(self.example, 0)
        public["timeline"].append(
            copy.deepcopy(self.example["public"]["timeline"][context["visibleThrough"]])
        )
        publication = copy.deepcopy(public["timeline"][0])
        publication["sequence"] = len(public["timeline"]) + 1
        publication["fields"]["programId"] = "other-program"
        public["timeline"].append(publication)
        identity = selected["selection"]["session"]
        contexts.append(context)
        acknowledgement = adapter.commit(
            session=identity,
            public=public,
            contexts=contexts,
            invocationId=context["invocationId"],
            record=selected["record"],
            selection=selected["selection"],
        )
        self.assertEqual(acknowledgement["publicDigest"], stable_algorithm_hash(public))
        observed = len(runtime.observations)
        with self.assertRaisesRegex(ValueError, "rewrites acknowledged native history"):
            self.request(adapter, 1)
        self.assertEqual(len(runtime.observations), observed)
        self.assertEqual(len(runtime.controls), 1)

        # The acknowledged observation is valid, but is not neural input until
        # the next independently reconstructed public prediction boundary.
        following = copy.deepcopy(self.example["contexts"][1])
        following.update(
            _replay_shared_state(
                public["timeline"],
                contexts,
                {},
                roundtrip=False,
                boundary_invocation=context["invocationId"],
            ),
            visibleThrough=len(public["timeline"]),
            prefixDigest=stable_algorithm_hash(public["timeline"]),
        )
        final = adapter.select(
            session=identity, public=public, contexts=contexts, context=following
        )
        terminal = copy.deepcopy(self.example["public"]["timeline"][-1])
        terminal["sequence"] = len(public["timeline"]) + 1
        public["timeline"].append(terminal)
        contexts.append(following)
        adapter.commit(
            session=identity,
            public=public,
            contexts=contexts,
            invocationId=context["invocationId"],
            record=final["record"],
            selection=final["selection"],
        )
        adapter.complete(
            session=identity, public=public, contexts=contexts, invocationId=context["invocationId"]
        )
        chunks = make_records(
            canonical_semantic_observation(publication),
            logical_id=f"semantic:v8:observation:{publication['sequence']}",
            role="user",
            codec=runtime.codec,
            max_tokens=256,
            content_kind="text",
        )
        for chunk in chunks:
            self.assertEqual(
                sum(row.event_id == chunk.record_id for row in runtime.observations), 1
            )

    def test_completion_requires_terminal_acknowledgement_and_explicit_session_close(self):
        adapter, _ = self.setup_adapter([(self.target(0), True), (self.target(1), True)])
        first = self.request(adapter)
        self.acknowledge(adapter, first)
        identity = first["selection"]["session"]
        before, contexts, _ = prefix(self.example, 1)
        with self.assertRaisesRegex(ValueError, "terminal record"):
            adapter.complete(
                session=identity,
                public=before,
                contexts=contexts,
                invocationId=first["selection"]["invocationId"],
            )
        final = self.request(adapter, 1)
        self.acknowledge(adapter, final, 1)
        completed_public = copy.deepcopy(self.example["public"])
        publication = copy.deepcopy(completed_public["timeline"][0])
        publication["sequence"] = len(completed_public["timeline"]) + 1
        publication["fields"]["programId"] = "completion-program"
        completed_public["timeline"].append(publication)
        completed = adapter.complete(
            session=identity,
            public=completed_public,
            contexts=self.example["contexts"],
            invocationId=final["selection"]["invocationId"],
            completion={"kind": "return"},
        )
        self.assertTrue(completed["completed"])
        self.assertTrue(completed["sessionRetained"])
        with self.assertRaisesRegex(ValueError, "completion rewrites acknowledged"):
            adapter.complete(
                session=identity,
                public=self.example["public"],
                contexts=self.example["contexts"],
                invocationId=final["selection"]["invocationId"],
            )
        self.assertEqual(adapter.metrics()["activeSessions"], 1)
        adapter.cancel(session=identity, reason="completed")
        self.assertEqual(adapter.metrics()["activeSessions"], 0)

    def test_callback_completions_preserve_native_tape_for_later_callback_and_parent(self):
        from megatoken.data.detailed_v7 import parse_record

        self.example = examples("semantic-callback-regions-v8.json.gz.b64")[0]
        adapter, runtime = self.setup_adapter(
            [(self.target(index), True) for index in range(len(self.example["contexts"]))]
        )
        completed = []
        last_revision = 0
        for index, context in enumerate(self.example["contexts"]):
            result = self.request(adapter, index)
            self.acknowledge(adapter, result, index)
            identity = result["selection"]["session"]
            current = adapter.sessions[identity].runtime
            self.assertGreater(current.revision, last_revision)
            last_revision = current.revision
            if (
                result["record"].startswith("detail ")
                and parse_record(result["record"])[0] == "end"
            ):
                public = copy.deepcopy(self.example["public"])
                public["timeline"] = public["timeline"][: context["visibleThrough"] + 1]
                response = adapter.complete(
                    session=identity,
                    public=public,
                    contexts=self.example["contexts"][: index + 1],
                    invocationId=context["invocationId"],
                )
                self.assertTrue(response["sessionRetained"])
                self.assertIs(adapter.sessions[identity].runtime, current)
                self.assertIn(current.memory.session_id, runtime.memory.sessions)
                completed.append(context["invocationId"])
        self.assertEqual(completed, ["event1", "event2", "invocation1"])
        self.assertEqual(adapter.metrics()["predictionAttempts"], len(self.example["contexts"]))
        self.assertEqual(len(adapter.metrics()["sessionOpenings"]), 1)
        adapter.cancel(session=identity, reason="completed")

    def test_commit_cannot_smuggle_extra_routing_payload(self):
        adapter, _ = self.setup_adapter([(self.target(), True)])
        result = self.request(adapter)
        public, contexts, context = prefix(self.example, 0)
        public["timeline"].append(
            copy.deepcopy(self.example["public"]["timeline"][context["visibleThrough"]])
        )
        public["timeline"][-1]["fields"]["futureAnswer"] = 42
        with self.assertRaisesRegex(ValueError, "unexpected generated routing"):
            adapter.commit(
                session=result["selection"]["session"],
                public=public,
                contexts=contexts + [context],
                invocationId=context["invocationId"],
                record=result["record"],
                selection=result["selection"],
            )

    def test_actual_native_runtime_generates_without_mutating_unacknowledged_state(self):
        import torch

        from megatoken.generation.policy import FacetedGenerator
        from megatoken.model import FacetedCore, ModelConfig
        from megatoken.model.multitrack import MultiTrackGenerator
        from megatoken.multitrack_runtime import MultiTrackRuntime

        torch.set_num_threads(1)
        torch.manual_seed(17)
        model = FacetedGenerator(
            FacetedCore(ModelConfig.tiny(max_history_chunks=128, max_capacity=512)),
            ByteCodec.byte_fallback(),
        ).eval()
        weights = {key: value.clone() for key, value in model.state_dict().items()}
        runtime = MultiTrackRuntime(MultiTrackGenerator.of(model))
        adapter = NativeSemanticInferenceV8(
            runtime, model_identity="a" * 64, max_tokens=4, max_fragments=1
        )
        adapter.initialize(
            seed=17, checkpointIdentity="a" * 64, tokenizerIdentity=model.codec.fingerprint
        )
        public, contexts, context = prefix(self.example, 0)
        identity = public["timeline"][0]["fields"]["session"]
        self.assertEqual(
            adapter.initialization["generationSettings"],
            {"device": "cpu", "maxTokens": 4, "maxFragments": 1, "effort": 1},
        )
        try:
            with torch.inference_mode():
                original = adapter._opened(identity).runtime
                before = original.state.raw.clone()
                try:
                    self.request(adapter)
                except NativeSelectionV8Error as error:
                    self.assertIsNotNone(error.evidence)
                self.assertEqual(original.revision, 0)
                self.assertTrue(torch.equal(original.state.raw, before))
                measured = adapter.metrics()
                self.assertEqual(len(measured["attempts"][0]["fragments"]), 1)
                self.assertGreater(measured["generatedTokens"], 0)
                self.assertIsNone(measured["learnedSuccess"])
                self.assertTrue(
                    all(
                        torch.equal(value, model.state_dict()[key])
                        for key, value in weights.items()
                    )
                )
        finally:
            for session in list(adapter.sessions):
                adapter.cancel(session=session, reason="test-complete")
            runtime.close()

    def test_real_training_and_serving_prediction_state_and_controls_match_exactly(self):
        import tempfile

        import torch

        from megatoken.data.chunking import ChunkingConfig
        from megatoken.data.storage import PreparedCorpus
        from megatoken.loading import build_model
        from megatoken.model import ModelConfig
        from megatoken.model.multitrack import MultiTrackGenerator
        from megatoken.multitrack_runtime import MultiTrackRuntime
        from megatoken.training.config import TrainConfig
        from scripts.fit_semantic_v8 import capture_boundaries
        from tests.test_semantic_v8_training_policy import prepared_v8_fixture

        torch.set_num_threads(1)
        torch.manual_seed(17)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        _, codec, policy, prepared = prepared_v8_fixture(
            Path(temporary.name), max_positions=128, max_history=128
        )
        model = build_model(
            ModelConfig.tiny(max_history_chunks=128, max_capacity=512), codec
        ).eval()
        with PreparedCorpus([str(prepared)], split="train") as corpus:
            episode = next(corpus.iter_episodes())
            chunking = ChunkingConfig.from_identity(corpus.chunking, codec)
        self.assertLess(chunking.hard_max_tokens, model.config.max_capacity)
        config = TrainConfig(
            purpose="diagnostic",
            phase="P1",
            model="tiny",
            codec="injected-tokenizer",
            datasets=[str(prepared)],
            semantic_v8_policy=policy,
            device="cpu",
            precision="fp32",
            representation_mode="bootstrap",
            effort=1,
            generated_capacity=128,
            max_window_records=128,
            max_history_tokens=16384,
            max_record_bytes=16384,
            scored_decisions=3,
            max_steps=1,
            gradient_checkpointing=False,
            expert_backend="torch",
        )
        boundaries, _ = capture_boundaries(model, episode, config, Path(__file__).parent)
        runtime = MultiTrackRuntime(MultiTrackGenerator.of(model), chunking=chunking)
        adapter = NativeSemanticInferenceV8(
            runtime, model_identity="a" * 64, max_tokens=4, max_fragments=1
        )
        actual_generate = runtime.generate
        seen = []

        def capture(session, **kwargs):
            seen.append((session.state.detach(), kwargs))
            return actual_generate(session, **kwargs)

        runtime.generate = capture
        try:
            with torch.inference_mode():
                adapter.initialize(
                    seed=17, checkpointIdentity="a" * 64, tokenizerIdentity=codec.fingerprint
                )
                try:
                    self.request(adapter)
                except NativeSelectionV8Error:
                    pass  # Random model output is not the assertion under test.
            self.assertEqual(len(seen), 1)
            trained, training_request = boundaries[0]
            served, serving_request = seen[0]
            for left, right in (
                (trained.raw, served.raw),
                (trained.history.keys, served.history.keys),
                (trained.history.values, served.history.values),
            ):
                self.assertTrue(torch.equal(left, right))
            self.assertEqual(trained.history_event_ids, served.history_event_ids)
            self.assertEqual(trained.history_event_tracks, served.history_event_tracks)
            self.assertEqual(trained.track_descriptions, served.track_descriptions)
            self.assertEqual(training_request["control_payloads"], serving_request["controls"])
        finally:
            for session in list(adapter.sessions):
                adapter.cancel(session=session, reason="test-complete")
            runtime.close()


if __name__ == "__main__":
    unittest.main()
