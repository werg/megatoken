"""Runtime binding controls; supplied records are not a model-quality claim."""

import copy

import pytest

from megatoken.semantic_model_v8 import NativeSelectionV8Error, NativeSemanticInferenceV8
from tests.test_prediction_prefix_v8 import examples
from tests.test_semantic_model_v8 import FakeRuntime
from tests.test_semantic_v8_serving_restore import commit, rehash, select, target

LINEAGE = "b" * 64


def adapter_for(fragments, *, lineage=LINEAGE):
    runtime = FakeRuntime([(fragment, True) for fragment in fragments])
    return (
        NativeSemanticInferenceV8(runtime, model_identity="a" * 64, training_identity=lineage),
        runtime,
    )


def initialize(adapter, runtime, **overrides):
    return adapter.initialize(
        **{
            "seed": 17,
            "checkpointIdentity": "a" * 64,
            "tokenizerIdentity": runtime.codec.fingerprint,
            "trainingIdentity": LINEAGE,
            **overrides,
        }
    )


def test_ordinary_model_requires_external_matching_training_identity():
    adapter, runtime = adapter_for([])
    for value in (None, "c" * 64, "malformed"):
        with pytest.raises(NativeSelectionV8Error, match="training-lineage identity"):
            initialize(adapter, runtime, trainingIdentity=value)
        assert adapter.initialization is None
        assert not runtime.memory.sessions
    assert initialize(adapter, runtime)["trainingIdentity"] == LINEAGE
    with pytest.raises(NativeSelectionV8Error, match="training-lineage identity"):
        initialize(adapter, runtime, trainingIdentity=None)
    assert adapter.initialization is None


def test_null_diagnostic_identity_does_not_impersonate_ordinary_training():
    adapter, runtime = adapter_for([], lineage=None)
    assert initialize(adapter, runtime, trainingIdentity=None)["trainingIdentity"] is None
    with pytest.raises(NativeSelectionV8Error, match="training-lineage identity"):
        initialize(adapter, runtime)
    for invalid in ("", {}, 17, "A" * 64):
        with pytest.raises(ValueError, match="training-lineage identity"):
            adapter_for([], lineage=invalid)


def test_training_identity_survives_selected_history_and_verified_restore():
    example = examples("semantic-regions-v8.json.gz.b64")[0]
    adapter, runtime = adapter_for([target(example, 0)])
    initialize(adapter, runtime)
    selected = select(adapter, example, 0)
    assert selected["selection"]["trainingIdentity"] == LINEAGE
    commit(adapter, example, 0, selected)
    saved = adapter.checkpoint()
    restored, restored_runtime = adapter_for([target(example, 0), target(example, 1)])
    assert restored.restore(checkpoint=saved)["restored"]
    assert restored.initialization["trainingIdentity"] == LINEAGE
    assert select(restored, example, 1)["selection"]["trainingIdentity"] == LINEAGE

    for mutate in (
        lambda packet: packet["initialization"].update(trainingIdentity="c" * 64),
        lambda packet: packet["initialization"].pop("trainingIdentity"),
        lambda packet: packet["attempts"][0].update(trainingIdentity="c" * 64),
    ):
        changed = copy.deepcopy(saved)
        mutate(changed)
        candidate, candidate_runtime = adapter_for([target(example, 0)])
        with pytest.raises(NativeSelectionV8Error, match="identity|accounting"):
            candidate.restore(checkpoint=rehash(changed))
        assert not candidate.sessions
        assert not candidate_runtime.memory.sessions
    candidate, candidate_runtime = adapter_for([target(example, 0)], lineage="c" * 64)
    with pytest.raises(NativeSelectionV8Error, match="training-lineage identity"):
        candidate.restore(checkpoint=saved)
    assert not candidate_runtime.memory.sessions
