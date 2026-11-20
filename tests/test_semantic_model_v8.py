"""Protocol tests use an explicit fake decoder; no model-quality claim follows."""

import copy
import json
import unittest
from dataclasses import dataclass
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
        adapter, _ = self.setup_adapter([(self.target(), True)])
        with self.assertRaisesRegex(ValueError, "acknowledged native history"):
            self.request(adapter, 1)
        with self.assertRaisesRegex(ValueError, "active v8 sessions"):
            adapter.initialize(seed=17, checkpointIdentity="a" * 64, tokenizerIdentity="b" * 64)
        with self.assertRaisesRegex(ValueError, "Initialize"):
            self.request(adapter)

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
        completed = adapter.complete(
            session=identity,
            public=self.example["public"],
            contexts=self.example["contexts"],
            invocationId=final["selection"]["invocationId"],
            completion={"kind": "return"},
        )
        self.assertTrue(completed["completed"])
        self.assertTrue(completed["sessionRetained"])
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


if __name__ == "__main__":
    unittest.main()
