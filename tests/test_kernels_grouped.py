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
