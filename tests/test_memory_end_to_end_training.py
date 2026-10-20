"""train() end-to-end on the mandatory retrieval-memory path."""

from dataclasses import replace

import pytest
import torch
from retrieval_fixture import make_episode, retrieval_train_config, write_retrieval_artifacts

from megatoken.training.checkpoint import load_checkpoint
from megatoken.training.engine import train


@pytest.fixture
def config(tmp_path):
    torch.set_num_threads(1)
    return retrieval_train_config(tmp_path)


def test_train_checkpoint_and_exact_resume(config):
    first = train(config)
    payload, _ = load_checkpoint(first["checkpoint"])
    assert first["global_step"] == payload["global_step"] == 1
    assert first["counters"]["P"] > 0
    for key in (
        "retrieval_sidecar_sha256",
        "retrieval_document_sha256",
        "retrieval_recipe_fingerprint",
    ):
        assert payload[key]
    resumed = train(config, resume=first["checkpoint"])
    assert resumed["global_step"] == 2
    again, _ = load_checkpoint(resumed["checkpoint"])
    assert again["retrieval_sidecar_sha256"] == payload["retrieval_sidecar_sha256"]


def test_resume_rejects_changed_sidecar_content(config, tmp_path):
    first = train(config)
    write_retrieval_artifacts(
        tmp_path / "retrieval",
        [make_episode(i) for i in range(4)],
        seed=1,
    )
    with pytest.raises(ValueError, match="changed retrieval supervision"):
        train(config, resume=first["checkpoint"])


def test_training_requires_all_retrieval_artifacts(tmp_path):
    with pytest.raises(ValueError, match="sidecars"):
        replace(retrieval_train_config(tmp_path), retrieval_sidecar="")
