import pytest
import torch

from megatoken.kernels.grouped import grouped_linear


def test_grouped_linear_reference_matches_dense_values_and_gradients():
    torch.manual_seed(11)
    x = torch.randn(17, 5, dtype=torch.float64, requires_grad=True)
    w = torch.randn(7, 9, 5, dtype=torch.float64, requires_grad=True)
    ids = torch.tensor([0, 3, 3, 0, 5, 1, 3, 5, 1, 1, 3, 0, 5, 3, 1, 0, 3])
    expected = torch.bmm(w[ids], x.unsqueeze(-1)).squeeze(-1)
    actual = grouped_linear(x, w, ids)
    torch.testing.assert_close(actual, expected)
    cotangent = torch.randn_like(actual)
    actual_grads = torch.autograd.grad(actual, (x, w), cotangent, retain_graph=True)
    expected_grads = torch.autograd.grad(expected, (x, w), cotangent)
    for result, reference in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(result, reference)
    assert actual_grads[1][2].count_nonzero() == 0  # unused expert is not fabricated learning


def test_grouped_double_gradcheck():
    x = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
    w = torch.randn(4, 2, 3, dtype=torch.float64, requires_grad=True)
    ids = torch.tensor([3, 1, 1, 3, 0])
    assert torch.autograd.gradcheck(lambda a, b: grouped_linear(a, b, ids), (x, w))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA parity gate requires a GPU")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_triton_grouped_forward_backward(dtype):
    pytest.importorskip("triton")
    torch.manual_seed(13)
    x = torch.randn(193, 64, device="cuda", dtype=dtype, requires_grad=True)
    w = torch.randn(16, 96, 64, device="cuda", dtype=dtype, requires_grad=True) * 0.1
    ids = torch.randint(0, 8, (193,), device="cuda") * 2
    actual = grouped_linear(x, w, ids, "triton")
    expected = grouped_linear(x, w, ids, "torch")
    tolerance = 3e-2 if dtype == torch.bfloat16 else 3e-5
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    cotangent = torch.randn_like(actual)
    actual_grads = torch.autograd.grad(actual, (x, w), cotangent, retain_graph=True)
    expected_grads = torch.autograd.grad(expected, (x, w), cotangent)
    for result, reference in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(result, reference, atol=tolerance, rtol=tolerance)


def test_explicit_cuda_backend_does_not_silently_fall_back():
    with pytest.raises(RuntimeError, match="requires CUDA"):
        grouped_linear(
            torch.randn(1, 2), torch.randn(1, 3, 2), torch.zeros(1, dtype=torch.long), "triton"
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA primary-width gate requires a GPU")
@pytest.mark.parametrize("inner,output", [(1024, 3072), (512, 2048)])
def test_triton_primary_pair_and_output_widths(inner, output):
    """Exercise real P/U projection widths with uneven and unused expert groups."""
    pytest.importorskip("triton")
    torch.manual_seed(19)
    dtype = torch.bfloat16
    x = torch.randn(97, inner, device="cuda", dtype=dtype, requires_grad=True)
    w = torch.randn(4, output, inner, device="cuda", dtype=dtype) / inner**0.5
    w.requires_grad_()
    ids = torch.tensor([0] * 64 + [2] * 32 + [3], device="cuda")
    ids = ids[torch.randperm(ids.numel(), device="cuda")]
    actual = grouped_linear(x, w, ids, "triton")
    expected = grouped_linear(x, w, ids, "torch")
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
    cotangent = torch.randn_like(actual) / output**0.5
    actual_grads = torch.autograd.grad(actual, (x, w), cotangent, retain_graph=True)
    expected_grads = torch.autograd.grad(expected, (x, w), cotangent)
    for result, reference in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(result, reference, atol=2e-2, rtol=2e-2)
    assert actual_grads[1][1].count_nonzero() == 0


def test_auto_grouped_dispatch_keeps_cpu_reference_values_and_gradients():
    torch.manual_seed(41)
    x = torch.randn(11, 13, dtype=torch.float64, requires_grad=True)
    weights = torch.randn(5, 17, 13, dtype=torch.float64, requires_grad=True)
    ids = torch.tensor([0, 2, 4, 4, 2, 0, 1, 2, 4, 0, 3])
    selected = grouped_linear(x, weights, ids, "auto")
    expected = grouped_linear(x, weights, ids, "torch")
    torch.testing.assert_close(selected, expected)
    cotangent = torch.randn_like(selected)
    actual_grads = torch.autograd.grad(selected, (x, weights), cotangent, retain_graph=True)
    reference_grads = torch.autograd.grad(expected, (x, weights), cotangent)
    for actual, reference in zip(actual_grads, reference_grads, strict=True):
        torch.testing.assert_close(actual, reference)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA backend selection needs a GPU")
def test_auto_dispatch_uses_triton_on_cuda(monkeypatch):
    fast = pytest.importorskip("megatoken.kernels.triton_grouped")
    calls = []
    original = fast.grouped_linear_triton

    def recorded(*args):
        calls.append(1)
        return original(*args)

    monkeypatch.setattr(fast, "grouped_linear_triton", recorded)
    x = torch.randn(43, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weights = torch.randn(8, 96, 64, device="cuda", dtype=torch.bfloat16)
    weights.requires_grad_()
    ids = torch.randint(0, 8, (43,), device="cuda")
    output = grouped_linear(x, weights, ids, "auto")
    output.square().mean().backward()
    assert calls == [1]
    assert x.grad is not None and weights.grad is not None
