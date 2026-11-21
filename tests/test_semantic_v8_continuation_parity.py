"""Actual recurrent training/serving parity with explicitly controlled decoding.

The source/tokenizer audit boundaries use the shared unit fixture; every prepared
row still passes normal V8 batch admission. Only the lexical decoder is replaced
with exact prepared fragments. Native prediction, encoding, cached continuation,
retrieval, acknowledgement and reconstruction are real. No learned-success claim
follows from these engineering controls.
"""

from dataclasses import fields, is_dataclass

import pytest
import torch

from megatoken.data.chunking import ChunkingConfig
from megatoken.data.detailed_v8 import parse_record
from megatoken.data.storage import PreparedCorpus
from megatoken.loading import build_model
from megatoken.model import ModelConfig
from megatoken.model.multitrack import MultiTrackGenerator
from megatoken.multitrack_runtime import MultiTrackRuntime
from megatoken.semantic_checkpoint_v8 import continuation_fingerprint
from megatoken.semantic_model_v8 import NativeSelectionV8Error, NativeSemanticInferenceV8
from megatoken.training.config import TrainConfig
from megatoken.training.engine import window_config
from megatoken.training.stream import _WindowPlan
from scripts.fit_semantic_v8 import capture_boundaries
from tests.test_prediction_prefix_v8 import examples
from tests.test_semantic_v8_serving_restore import commit, select
from tests.test_semantic_v8_training_policy import prepared_v8_fixture


def assert_same_state(prepared, live, live_to_prepared, path="state"):
    """Compare every state field; only generated event addresses have a bijection.

    Prepared physical IDs hash their known target bytes. Live event IDs identify
    an attempt before its bytes exist. The mapping is established solely from
    equal original emitted fragments; observed/context addresses stay exact.
    Nothing is renamed inside either state or fed back into the model.
    """
    if isinstance(prepared, torch.Tensor):
        assert isinstance(live, torch.Tensor), path
        assert prepared.dtype == live.dtype and prepared.shape == live.shape, path
        assert torch.equal(prepared, live), path
    elif is_dataclass(prepared):
        assert type(prepared) is type(live), path
        for field in fields(prepared):
            assert_same_state(
                getattr(prepared, field.name),
                getattr(live, field.name),
                live_to_prepared,
                f"{path}.{field.name}",
            )
    elif isinstance(prepared, (list, tuple)):
        assert type(prepared) is type(live) and len(prepared) == len(live), path
        for index, (left, right) in enumerate(zip(prepared, live, strict=True)):
            assert_same_state(left, right, live_to_prepared, f"{path}[{index}]")
    elif isinstance(prepared, str):
        assert prepared == live_to_prepared.get(live, live), path
    else:
        assert prepared == live, path


@pytest.mark.parametrize(
    ("fixture", "example_index", "physical_limit", "mode", "effort", "recover"),
    [
        ("semantic-regions-v8.json.gz.b64", 0, 128, "bootstrap", 1, True),
        ("semantic-regions-v8.json.gz.b64", 2, 512, "recurrent", 2, False),
        ("semantic-callback-regions-v8.json.gz.b64", 0, 512, "recurrent", 1, False),
    ],
    ids=["split-record-rejection-restore", "host-results-eviction", "retained-callbacks"],
)
def test_actual_recurrent_state_controls_and_ownership_match_at_every_boundary(
    tmp_path, monkeypatch, fixture, example_index, physical_limit, mode, effort, recover
):
    torch.set_num_threads(1)
    torch.manual_seed(17)
    example, codec, policy, prepared = prepared_v8_fixture(
        tmp_path,
        max_positions=physical_limit,
        max_history=128,
        document=examples(fixture)[example_index],
    )
    model = build_model(ModelConfig.tiny(max_history_chunks=128, max_capacity=512), codec).eval()
    with PreparedCorpus([str(prepared)], split="train") as corpus:
        episode = next(corpus.iter_episodes())
        chunking = ChunkingConfig.from_identity(corpus.chunking, codec)
    config = TrainConfig(
        purpose="diagnostic",
        phase="P1",
        model="tiny",
        codec="injected-tokenizer",
        datasets=[str(prepared)],
        semantic_v8_policy=policy,
        device="cpu",
        precision="fp32",
        representation_mode=mode,
        effort=effort,
        use_effort_mixture=False,
        generated_capacity=physical_limit,
        max_window_records=16 if recover else 512,
        max_history_tokens=65536,
        max_record_bytes=16384,
        scored_decisions=128,
        max_steps=1,
        gradient_checkpointing=False,
        expert_backend="torch",
    )
    if recover or "callback" in fixture:
        assert len(_WindowPlan(episode, codec, window_config(config))) > 1
    boundaries, _ = capture_boundaries(model, episode, config, tmp_path)
    targets = [record for record in episode.records if record.policy_mask]
    assert len(boundaries) == len(targets)
    if recover:
        assert len(targets) > len(example["contexts"])
    else:
        assert len(targets) >= 11
    cursor = 0
    refuse_next = recover
    live_to_prepared = {}
    compared = []
    runtimes = []

    def controlled_decoder(_multi, _raw, _specs, _control, **_kwargs):
        nonlocal cursor, refuse_next
        if refuse_next:
            refuse_next = False
            return (b"\xff",), ((255,),), (True,), (True,), (None,)
        target = targets[cursor]
        cursor += 1
        return (
            (target.payload,),
            (tuple(codec.encode(target.payload)),),
            (True,),
            (target.part_index + 1 == target.part_count,),
            (None,),
        )

    monkeypatch.setattr(MultiTrackGenerator, "_realize", controlled_decoder)

    def open_adapter():
        runtime = MultiTrackRuntime(MultiTrackGenerator.of(model), chunking=chunking)
        runtimes.append(runtime)
        actual_generate = runtime.generate

        def compare_and_generate(session, **request):
            index, failing = cursor, refuse_next
            state, trained_request = boundaries[index]
            assert_same_state(state, session.state, live_to_prepared)
            assert trained_request["control_payloads"] == request["controls"]
            assert trained_request["tracks"] == request["tracks"]
            plan = trained_request["plans"][0]
            assert plan.mode_specs == request["specs"]
            assert plan.free_masks == ((True,) * len(plan.ids[0]),)
            generated = actual_generate(session, **request)
            if not failing:
                target = targets[index]
                tape = generated.tape
                assert tape.payloads == plan.payloads == (target.payload,)
                assert tape.emitted_ids == plan.ids
                assert tape.terminal == plan.terminals
                assert tape.roles == plan.roles
                assert tape.continuation_mode == "encoded_text"
                live_id = tape.event_ids[0]
                assert live_id not in live_to_prepared
                assert target.record_id not in live_to_prepared.values()
                live_to_prepared[live_id] = target.record_id
                compared.append(index)
            return generated

        runtime.generate = compare_and_generate
        adapter = NativeSemanticInferenceV8(
            runtime,
            model_identity="a" * 64,
            effort=effort,
            max_tokens=physical_limit,
            max_fragments=16,
        )
        return adapter

    adapter = open_adapter()
    try:
        with torch.inference_mode():
            adapter.initialize(
                seed=17, checkpointIdentity="a" * 64, tokenizerIdentity=codec.fingerprint
            )
            if recover:
                with pytest.raises(NativeSelectionV8Error, match="utf-8"):
                    select(adapter, example, 0)
                assert adapter.metrics()["failedAttempts"] == 1 and cursor == 0
            for index, context in enumerate(example["contexts"]):
                chosen = select(adapter, example, index)
                commit(adapter, example, index, chosen)
                public = example["public"]
                row = public["timeline"][context["visibleThrough"]]
                line = row["fields"]["line"]
                if line.startswith("detail ") and parse_record(line)[0] == "end":
                    adapter.complete(
                        session=chosen["selection"]["session"],
                        public={**public, "timeline": public["timeline"][: row["sequence"]]},
                        contexts=example["contexts"][: index + 1],
                        invocationId=context["invocationId"],
                    )
                if recover and index == 0:
                    packet = adapter.checkpoint()
                    session = chosen["selection"]["session"]
                    expected = continuation_fingerprint(adapter.sessions[session].runtime)
                    adapter.cancel(session=session, reason="Controlled recovery boundary")
                    cursor = 0
                    live_to_prepared.clear()
                    adapter = open_adapter()
                    receipt = adapter.restore(checkpoint=packet)
                    assert receipt["reexecutedSelections"] == 1
                    assert receipt["consumedAttempts"] == 2
                    assert continuation_fingerprint(adapter.sessions[session].runtime) == expected
            assert cursor == len(targets)
            assert set(compared) == set(range(len(targets)))
            assert adapter.metrics()["committed"] == len(example["contexts"])
            if not recover:
                assert any(len(state.history_event_ids) == 128 for state, _ in boundaries)
    finally:
        for session in list(adapter.sessions):
            adapter.cancel(session=session, reason="Engineering parity control completed")
        for runtime in runtimes:
            runtime.close()
