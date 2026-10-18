import pytest
import torch

from megatoken.kernels.recurrent import delta_rule_step


def recurrence_inputs(device="cpu", dtype=torch.float64):
    leading = (2, 3, 2)
    return (
        torch.randn(*leading, 4, 4, device=device, dtype=dtype, requires_grad=True) * 0.1,
        torch.randn(*leading, 4, device=device, dtype=dtype, requires_grad=True) * 0.1,
        torch.randn(*leading, 4, device=device, dtype=dtype, requires_grad=True) * 0.1,
        torch.randn(*leading, 4, device=device, dtype=dtype, requires_grad=True) * 0.1,
        torch.full(leading, 0.9, device=device, dtype=dtype, requires_grad=True),
        torch.full(leading, 0.2, device=device, dtype=dtype, requires_grad=True),
    )


def test_record_recurrence_gradcheck_and_stream_independence():
    inputs = recurrence_inputs()
    assert torch.autograd.gradcheck(delta_rule_step, inputs, fast_mode=True)
    states, outputs = delta_rule_step(*inputs)
    changed = list(inputs)
    changed[3] = changed[3].clone()
    changed[3][0, 1, 0] = 9
    next_states, next_outputs = delta_rule_step(*changed)
    mask = torch.ones(inputs[0].shape[:-2], dtype=torch.bool)
    mask[0, 1, 0] = False
    torch.testing.assert_close(states[mask], next_states[mask], rtol=0, atol=0)
    torch.testing.assert_close(outputs[mask], next_outputs[mask], rtol=0, atol=0)
    assert not torch.equal(states[~mask], next_states[~mask])


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA recurrence parity gate requires a GPU"
)
def test_triton_recurrence_forward_and_all_six_gradients():
    pytest.importorskip("triton")
    inputs = recurrence_inputs("cuda", torch.float32)
    actual = delta_rule_step(*inputs, backend="triton")
    expected = delta_rule_step(*inputs, backend="torch")
    for result, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(result, reference, rtol=3e-5, atol=3e-6)
    cotangents = tuple(torch.randn_like(result) for result in actual)
    actual_grads = torch.autograd.grad(actual, inputs, cotangents, retain_graph=True)
    reference_grads = torch.autograd.grad(expected, inputs, cotangents)
    for result, reference in zip(actual_grads, reference_grads, strict=True):
        torch.testing.assert_close(result, reference, rtol=3e-5, atol=3e-6)
