"""Experiment checkpoint accounting without constructing an optimizer."""

import copy

import pytest

from megatoken.training.checkpoint import (
    load_checkpoint,
    save_checkpoint,
    transducer_attachment,
)
from megatoken.training.semantic_budget import (
    SemanticExposure,
    restore_semantic_training_state,
    semantic_training_state,
)


@pytest.mark.parametrize("phase", ["B0", "P1"])
def test_nonsemantic_checkpoint_has_no_experiment_counters(tmp_path, phase):
    payload = {
        "train_config": {"phase": phase, "semantic_experiment": None},
        **semantic_training_state(None, cohort_digest=None),
    }
    saved = save_checkpoint(
        tmp_path,
        phase,
        payload,
        {"phase": phase},
        attachments=transducer_attachment(lambda: {"bank": {}}),
    )
    restored, _ = load_checkpoint(saved)
    assert "semantic_exposure" not in restored
    assert "semantic_cohort_digest" not in restored
    assert restore_semantic_training_state(restored, cohort_digest=None) is None
    with pytest.raises(ValueError, match="nonsemantic checkpoint"):
        restore_semantic_training_state(
            {**restored, "semantic_exposure": SemanticExposure().checkpoint()},
            cohort_digest=None,
        )


def test_semantic_checkpoint_restores_exact_counters_and_cohort(tmp_path):
    exposure = SemanticExposure()
    exposure.consume(examples=2, generatedTokens=31, observedTokens=79, computeSeconds=0.75)
    fields = semantic_training_state(exposure, cohort_digest="a" * 64)
    saved = save_checkpoint(
        tmp_path,
        "SE",
        fields,
        {"phase": "SE"},
        attachments=transducer_attachment(lambda: {"bank": {}}),
    )
    payload, _ = load_checkpoint(saved)
    restored = restore_semantic_training_state(payload, cohort_digest="a" * 64)
    assert restored.checkpoint() == exposure.checkpoint()
    restored.consume(examples=1, generatedTokens=7, observedTokens=11, computeSeconds=0.25)
    assert restored.examples == 3 and restored.computeSeconds == 1
    for missing in ("semantic_exposure", "semantic_cohort_digest"):
        bad = copy.deepcopy(payload)
        del bad[missing]
        with pytest.raises(ValueError, match="exact exposure state and frozen cohort"):
            restore_semantic_training_state(bad, cohort_digest="a" * 64)
    with pytest.raises(ValueError, match="exact exposure state and frozen cohort"):
        restore_semantic_training_state(payload, cohort_digest="b" * 64)
