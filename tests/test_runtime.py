"""Fail-closed device preflight and actual-width CUDA attention gates."""

import os
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from megatoken.model.attention import AttentionBranch, AttentionFeatures
from megatoken.model.config import ModelConfig
from megatoken.runtime import doctor


def fake_cuda(monkeypatch, *, capability=(10, 0), finite=True, failure=False):
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index: capability)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda index: SimpleNamespace(
            name="test-device",
            total_memory=80 * 1024**3,
        ),
    )
    monkeypatch.setattr(torch.cuda, "device", lambda index: nullcontext())
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda index: None)

    def matrix(*args, **kwargs):
        if failure:
            raise RuntimeError("test kernel cannot execute")
        return torch.full((2, 2), 1.0 if finite else torch.inf)

    monkeypatch.setattr(torch, "randn", matrix)


@pytest.mark.parametrize("capability", [(10, 0), (10, 3), (12, 0), (12, 1)])
def test_preflight_accepts_recognized_blackwell_only_after_execution(monkeypatch, capability):
    fake_cuda(monkeypatch, capability=capability)
    report = doctor(require_blackwell=True)
    assert report["ok"] and report["devices"][0]["matmul_ok"]


@pytest.mark.parametrize("failure", [False, True])
def test_preflight_rejects_nonfinite_or_failed_bf16_execution(monkeypatch, failure):
    fake_cuda(monkeypatch, finite=False, failure=failure)
    report = doctor(require_blackwell=True)
    assert not report["ok"] and not report["devices"][0]["matmul_ok"]
    assert any("BF16" in error for error in report["errors"])


def test_preflight_cuda_requirement_is_not_satisfied_by_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    assert doctor()["ok"]
    assert not doctor(require_cuda=True)["ok"]
    assert not doctor(require_blackwell=True)["ok"]


def test_preflight_does_not_label_hopper_blackwell(monkeypatch):
    fake_cuda(monkeypatch, capability=(9, 0))
    assert doctor(require_cuda=True)["ok"]
    assert not doctor(require_blackwell=True)["ok"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA attention gate requires a GPU")
@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("source", [False, True])
def test_primary_width_unmasked_flash_gqa_values_and_gradients(compact, source):
    """A forced fused backend prevents a hidden math fallback from passing the gate."""
    torch.manual_seed(51)
    config = ModelConfig.compact32() if compact else ModelConfig()
    source_dim = config.source_dim if source else config.facet_dim
    branch = AttentionBranch(
        config.facet_dim,
        source_dim,
        config.query_heads,
        config.kv_heads,
        config.attention_head_dim,
    ).to(device="cuda", dtype=torch.bfloat16)
    query = torch.randn(
        2, 32, config.facet_dim, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    memory = torch.randn(
        2,
        129 if source else 32,
        source_dim,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        actual = branch(query, memory)
    with sdpa_kernel(SDPBackend.MATH):
        expected = branch(query, memory)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-3)
    upstream = torch.randn_like(actual) / actual.numel() ** 0.5
    inputs = (query, memory, *branch.parameters())
    actual_grads = torch.autograd.grad(actual, inputs, upstream, retain_graph=True)
    expected_grads = torch.autograd.grad(expected, inputs, upstream)
    for result, reference in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(result, reference, rtol=3e-2, atol=2e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA masked-source gate requires a GPU")
@pytest.mark.parametrize("compact", [False, True])
def test_masked_and_empty_source_gqa_has_finite_correct_gradients(compact):
    torch.manual_seed(52)
    config = ModelConfig.compact32() if compact else ModelConfig()
    branch = AttentionBranch(
        config.facet_dim,
        config.source_dim,
        config.query_heads,
        config.kv_heads,
        config.attention_head_dim,
    ).to(device="cuda", dtype=torch.bfloat16)
    query = torch.randn(
        2, 32, config.facet_dim, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    memory = torch.randn(
        2, 17, config.source_dim, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    mask = torch.arange(17, device="cuda")[None] < torch.tensor([13, 0], device="cuda")[:, None]
    actual = branch(query, memory, mask)
    with sdpa_kernel(SDPBackend.MATH):
        expected = branch(query, memory, mask)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-3)
    assert actual[1].count_nonzero() == 0
    actual.float().square().mean().backward()
    assert all(torch.isfinite(parameter.grad).all() for parameter in branch.parameters())
    assert memory.grad[0, 13:].count_nonzero() == 0
    assert memory.grad[1].count_nonzero() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA historical-KV gate requires a GPU")
@pytest.mark.parametrize("compact", [False, True])
def test_full_history_projected_flash_gqa_values_and_gradients(compact):
    """Gate the full 1,024-chunk KV geometry, not only small live attention."""
    torch.manual_seed(53)
    config = ModelConfig.compact32() if compact else ModelConfig()
    branch = AttentionBranch(
        config.facet_dim,
        config.facet_dim,
        config.query_heads,
        config.kv_heads,
        config.attention_head_dim,
    ).to(device="cuda", dtype=torch.bfloat16)
    query = torch.randn(
        1,
        config.n_facets,
        config.facet_dim,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    shape = (
        1,
        config.kv_heads,
        config.n_facets * config.max_history_chunks,
        config.attention_head_dim,
    )
    keys = torch.randn(shape, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    values = torch.randn_like(keys, requires_grad=True)
    features = AttentionFeatures(keys, values, None)
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        actual = branch.read(query, features)
    with sdpa_kernel(SDPBackend.MATH):
        expected = branch.read(query, features)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-3)
    upstream = torch.randn_like(actual) / actual.numel() ** 0.5
    inputs = (
        query,
        keys,
        values,
        branch.query_norm.weight,
        branch.query.weight,
        branch.output.weight,
    )
    actual_grads = torch.autograd.grad(actual, inputs, upstream, retain_graph=True)
    expected_grads = torch.autograd.grad(expected, inputs, upstream)
    for result, reference in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(result, reference, rtol=3e-2, atol=2e-3)


def test_deterministic_runs_fix_the_cublas_workspace_before_cuda_starts(monkeypatch):
    from megatoken.runtime import seed_everything

    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    try:
        seed_everything(1, deterministic=False)
        assert "CUBLAS_WORKSPACE_CONFIG" not in os.environ
        seed_everything(1, deterministic=True)
        assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
        # A value the operator chose is respected.
        monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
        seed_everything(1, deterministic=True)
        assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":16:8"
    finally:
        torch.use_deterministic_algorithms(False)
