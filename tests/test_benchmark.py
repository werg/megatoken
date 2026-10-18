import json

import pytest
import torch

from megatoken.benchmark import benchmark


@pytest.fixture(autouse=True)
def single_thread_cpu():
    torch.set_num_threads(1)


def test_core_benchmark_counts_reconstruction_without_inventing_policy_tokens():
    result = benchmark(
        steps=2, warmup=1, batch=2, record_bytes=8, capacity=4, core_only=True, history_records=1
    )
    assert result["counts"]["source_bytes"] == 2 * 2 * 8 * 2
    assert result["counts"]["record_ingestions"] == 2 * 2 * 2
    assert result["counts"]["reconstruction_records"] == 2 * 2
    assert result["counts"]["D"] is None and result["counts"]["I"] is None
    assert result["counts"]["P"] == 0
    assert result["dispatch"]["p"]["total_rows"] == 2 * 2 * 2 * 6 * 3 * 16
    assert sum(result["dispatch"]["p"]["learning_rows"]) == 0
    assert result["elapsed_seconds"] > 0
    json.dumps(result, allow_nan=False)


def test_full_generator_training_measures_real_policy_and_optimizer_work():
    result = benchmark(steps=1, warmup=0, batch=2, record_bytes=8, capacity=4, backward=True)
    assert result["counts"]["D"] == result["counts"]["I"] == 24
    assert result["counts"]["P"] == 8
    assert result["counts"]["record_ingestions"] == 4
    assert result["counts"]["optimizer_steps"] == 1
    assert result["phase_seconds"]["backward"] > 0
    assert result["phase_seconds"]["optimizer"] > 0
    assert result["workload"] == "supervised_training"
    json.dumps(result, allow_nan=False)


def test_full_prior_inference_commits_actual_sampled_bytes():
    result = benchmark(steps=1, warmup=0, batch=2, record_bytes=8, capacity=4)
    assert result["counts"]["P"] == 8
    assert result["counts"]["generated_bytes"] == 8
    assert result["counts"]["record_ingestions"] == 4
    assert result["counts"]["optimizer_steps"] == 0
    assert result["configuration"]["batch_execution"] == "serial_requests"
    assert "optimizer" not in result["phase_seconds"]


def test_no_silent_cuda_or_fake_vocabulary_benchmark():
    with pytest.raises(ValueError, match="actual CUDA"):
        benchmark(backend="triton")
    with pytest.raises(ValueError, match="real codec"):
        benchmark(config="v13", steps=1, warmup=0)
