"""Live recovery controls; supplied decoder outputs make no model-quality claim."""

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from megatoken.data.semantic_json import stable_algorithm_hash
from megatoken.semantic_checkpoint_v8 import capture_random_state, continuation_fingerprint
from megatoken.semantic_model_v8 import NativeSelectionV8Error, NativeSemanticInferenceV8
from tests.test_prediction_prefix_v8 import examples, prefix
from tests.test_semantic_model_v8 import FakeRuntime


def target(example, index):
    return example["public"]["timeline"][example["contexts"][index]["visibleThrough"]]["fields"][
        "line"
    ].encode()


def adapter_for(fragments, **options):
    runtime = FakeRuntime([(item, True) for item in fragments])
    adapter = NativeSemanticInferenceV8(runtime, model_identity="a" * 64, **options)
    adapter.initialize(
        seed=17, checkpointIdentity="a" * 64, tokenizerIdentity=runtime.codec.fingerprint
    )
    return adapter, runtime


def select(adapter, example, index):
    public, contexts, context = prefix(example, index)
    return adapter.select(
        session=public["timeline"][0]["fields"]["session"],
        public=public,
        contexts=contexts,
        context=context,
    )


def commit(adapter, example, index, selected):
    context = example["contexts"][index]
    public = copy.deepcopy(example["public"])
    public["timeline"] = public["timeline"][: context["visibleThrough"] + 1]
    return adapter.commit(
        session=selected["selection"]["session"],
        public=public,
        contexts=example["contexts"][: index + 1],
        invocationId=context["invocationId"],
        record=selected["record"],
        selection=selected["selection"],
    )


def rehash(packet):
    packet["digest"] = stable_algorithm_hash(
        {key: value for key, value in packet.items() if key != "digest"}
    )
    return packet


def test_restore_preserves_accepted_cache_rng_cost_and_budget_but_excludes_pending_candidate():
    example = examples("semantic-regions-v8.json.gz.b64")[0]
    adapter, _ = adapter_for([b"\xff", target(example, 0), target(example, 1)], max_attempts=4)
    with pytest.raises(NativeSelectionV8Error):
        select(adapter, example, 0)
    committed = select(adapter, example, 0)
    commit(adapter, example, 0, committed)
    pending = select(adapter, example, 1)
    session = pending["selection"]["session"]
    saved = json.loads(json.dumps(adapter.checkpoint()))
    before = continuation_fingerprint(adapter.sessions[session].runtime)
    restored, _ = adapter_for([target(example, 0), target(example, 1)], max_attempts=4)
    receipt = restored.restore(checkpoint=saved)
    assert receipt == {
        "schema": "semantic-model-restore/8",
        "checkpointDigest": saved["digest"],
        "restoredSessions": 1,
        "reexecutedSelections": 1,
        "consumedAttempts": 3,
        "hostDispatches": 0,
        "equivalence": "verified-model-reexecution",
        "restored": True,
    }
    assert continuation_fingerprint(restored.sessions[session].runtime) == before
    assert capture_random_state() == saved["randomState"]
    assert restored.sessions[session].pending is None
    assert restored.attempts[-1]["status"] == "abandoned-on-restore"
    assert restored.metrics()["generatedTokens"] == adapter.metrics()["generatedTokens"]
    next_output = select(restored, example, 1)
    assert next_output["selection"]["selectionId"] == "v8:3"
    commit(restored, example, 1, next_output)
    with pytest.raises(NativeSelectionV8Error, match="capacity exhausted"):
        select(restored, example, 1)


def test_fresh_process_rpc_restore_continues_original_records(tmp_path):
    example = examples("semantic-regions-v8.json.gz.b64")[0]
    adapter, _ = adapter_for([target(example, 0)])
    commit(adapter, example, 0, select(adapter, example, 0))
    saved = adapter.checkpoint()
    source = tmp_path / "checkpoint.json"
    source.write_text(json.dumps(saved))
    code = """
import json, sys
from tests.test_semantic_v8_serving_restore import adapter_for, target, select, commit
from tests.test_prediction_prefix_v8 import examples
from scripts.serve_semantic_v8 import dispatch
example=examples("semantic-regions-v8.json.gz.b64")[0]
adapter,_=adapter_for([target(example,0),target(example,1)])
receipt=dispatch(adapter,{"id":"restore-1","op":"restore","checkpoint":json.load(open(sys.argv[1]))})
result=select(adapter,example,1)
committed=commit(adapter,example,1,result)
packet=dispatch(adapter,{"id":"checkpoint-2","op":"checkpoint"})
print(json.dumps({"receipt":receipt,"committed":committed,"attempts":len(packet["attempts"])}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code, str(source)],
        cwd=Path(__file__).resolve().parents[1],
        env=dict(os.environ),
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    assert result["receipt"]["restored"] and result["committed"]["committed"]
    assert result["attempts"] == 2


def test_rehashed_false_history_cursor_tokens_cache_and_missing_receipts_are_refused_atomically():
    example = examples("semantic-regions-v8.json.gz.b64")[0]
    adapter, _ = adapter_for([target(example, 0)])
    output = select(adapter, example, 0)
    commit(adapter, example, 0, output)
    saved = adapter.checkpoint()
    session = output["selection"]["session"]
    mutations = [
        lambda p: p["sessions"][session].update(ingestedThrough=0),
        lambda p: p["sessions"][session].update(continuationDigest="0" * 64),
        lambda p: p["sessions"][session]["replay"].clear(),
        lambda p: p["sessions"][session]["contexts"][0].update(visibleThrough=0),
        lambda p: p["attempts"][0].update(payloadHex="00"),
        lambda p: p["attempts"][0]["fragments"][0]["tokenIds"].append(0),
        lambda p: p["sessions"][session]["replay"][0].update(acknowledgedThrough=0),
        lambda p: p["sessions"][session]["openingRandom"].update(torch="invalid"),
        lambda p: p["settings"].update(maxAttempts=10001),
    ]
    for mutation in mutations:
        tampered = copy.deepcopy(saved)
        mutation(tampered)
        rehash(tampered)
        restored, runtime = adapter_for([target(example, 0)])
        before_random = capture_random_state()
        with pytest.raises((NativeSelectionV8Error, ValueError)):
            restored.restore(checkpoint=tampered)
        assert not restored.sessions and not restored.attempts and not runtime.memory.sessions
        assert capture_random_state() == before_random
    restored, _ = adapter_for([target(example, 0)])
    restored.restore(checkpoint=saved)
    with pytest.raises(NativeSelectionV8Error, match="fresh empty"):
        restored.restore(checkpoint=saved)


def test_callbacks_and_parent_continue_after_restoring_completed_child():
    from megatoken.data.detailed_v8 import parse_record

    example = examples("semantic-callback-regions-v8.json.gz.b64")[0]
    outputs = [target(example, index) for index in range(len(example["contexts"]))]
    adapter, _ = adapter_for(outputs)
    cutoff = next(
        index
        for index, line in enumerate(outputs)
        if line.startswith(b"detail ") and parse_record(line.decode())[0] == "end"
    )
    for index in range(cutoff + 1):
        result = select(adapter, example, index)
        commit(adapter, example, index, result)
    session = result["selection"]["session"]
    work = adapter.sessions[session]
    completed_child = example["contexts"][cutoff]["invocationId"]
    adapter.complete(
        session=session, public=work.public, contexts=work.contexts, invocationId=completed_child
    )
    saved = adapter.checkpoint()
    restored, _ = adapter_for(outputs)
    receipt = restored.restore(checkpoint=json.loads(json.dumps(saved)))
    assert receipt["hostDispatches"] == 0
    assert restored.sessions[session].completed == {completed_child}
    for index in range(cutoff + 1, len(outputs)):
        result = select(restored, example, index)
        commit(restored, example, index, result)
    assert restored.sessions[session].public == example["public"]
    assert len(restored.attempts) == len(outputs)


def actual_adapter():
    """Real tiny CPU model; random outputs are never labeled semantic success."""
    import torch

    from megatoken.generation.codec import ByteCodec
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
    runtime = MultiTrackRuntime(MultiTrackGenerator.of(model))
    adapter = NativeSemanticInferenceV8(
        runtime, model_identity="a" * 64, max_tokens=4, max_fragments=1
    )
    adapter.initialize(
        seed=17, checkpointIdentity="a" * 64, tokenizerIdentity=model.codec.fingerprint
    )
    return adapter, runtime


def actual_attempt(adapter, example):
    try:
        return select(adapter, example, 0)["selection"]
    except NativeSelectionV8Error as error:
        assert error.evidence is not None
        return error.evidence


def test_actual_model_rng_and_cache_resume_in_a_fresh_process(tmp_path):
    import torch

    example = examples("semantic-regions-v8.json.gz.b64")[0]
    adapter, runtime = actual_adapter()
    try:
        with torch.inference_mode():
            actual_attempt(adapter, example)
            packet = adapter.checkpoint()
            for session, work in adapter.sessions.items():
                if work.pending:
                    adapter.reject(
                        session=session,
                        selectionId=work.pending["attempt"]["selectionId"],
                        reason="Compare the next attempt after discarding pending output",
                    )
            expected = actual_attempt(adapter, example)
        checkpoint_file = tmp_path / "actual-model-checkpoint.json"
        checkpoint_file.write_text(json.dumps(packet))
        code = """
import json, sys, torch
from tests.test_semantic_v8_serving_restore import actual_adapter, actual_attempt
from tests.test_prediction_prefix_v8 import examples
adapter,runtime=actual_adapter()
try:
    with torch.inference_mode():
        receipt=adapter.restore(checkpoint=json.load(open(sys.argv[1])))
        output=actual_attempt(adapter,examples("semantic-regions-v8.json.gz.b64")[0])
    print(json.dumps({"receipt":receipt,"selectionId":output["selectionId"],
                     "fragments":output["fragments"],"payloadHex":output["payloadHex"]}))
finally:
    for session in list(adapter.sessions): adapter.cancel(session=session,reason="test")
    runtime.close()
"""
        completed = subprocess.run(
            [sys.executable, "-c", code, str(checkpoint_file)],
            cwd=Path(__file__).resolve().parents[1],
            env=dict(os.environ),
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        )
        restored = json.loads(completed.stdout.strip().splitlines()[-1])
        assert restored["receipt"]["restored"]
        assert restored["receipt"]["reexecutedSelections"] == 0
        for key in ("selectionId", "fragments", "payloadHex"):
            assert restored[key] == expected[key]
    finally:
        for session in list(adapter.sessions):
            adapter.cancel(session=session, reason="test")
        runtime.close()


def test_failed_runtime_open_during_restore_releases_partial_allocation():
    example = examples("semantic-regions-v8.json.gz.b64")[0]
    original, _ = adapter_for([target(example, 0)])
    commit(original, example, 0, select(original, example, 0))
    packet = original.checkpoint()
    restored, runtime = adapter_for([target(example, 0)])
    before_random = capture_random_state()
    open_runtime = runtime.open

    def fail_after_allocation(**kwargs):
        open_runtime(**kwargs)
        raise RuntimeError("simulated initialization failure after allocation")

    runtime.open = fail_after_allocation
    with pytest.raises(NativeSelectionV8Error, match="initialization failure"):
        restored.restore(checkpoint=packet)
    assert not restored.sessions and not restored.attempts and not runtime.memory.sessions
    assert capture_random_state() == before_random
