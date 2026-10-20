"""Optimizer-step pooled router balance must use executed prefixes and local grads."""
import torch

from megatoken.model.routing import EpochRouter
from megatoken.training.router_balance import RouterBalanceWindow


class _SingleRank:
    device = torch.device("cpu")

    @staticmethod
    def sum_tensors(values):
        return values


def test_global_balance_matches_one_step_proxy_and_does_not_backprop_upstream():
    torch.manual_seed(7)
    router = EpochRouter(8, 12, experts=5, max_eligible=3)
    window = RouterBalanceWindow()
    router.balance_recorder = window
    x = torch.randn(2, 8, requires_grad=True)
    route = router.route(x, stage=2, rows=4)
    # Only one of the four compiled rows actually ran on each example.
    window.capture(router, route.prefix(1))
    chosen = route.expert_ids[:, :1].detach().flatten()
    counts = torch.bincount(chosen, minlength=router.experts).float()
    fraction = counts / counts.sum()
    reference = 0.3 * router.experts * (
        fraction * router.balance_probabilities(x.detach(), 2).mean(0)
    ).sum()
    parameters = list(router.parameters())
    expected = torch.autograd.grad(reference, parameters, allow_unused=True)
    metrics = window.backward(_SingleRank(), routers={"p": router}, coefficient=0.3)
    assert metrics["balance_p_assignments"] == 2.0
    assert x.grad is None
    for p, grad in zip(parameters, expected):
        if grad is None:
            assert p.grad is None
        else:
            torch.testing.assert_close(p.grad, grad, atol=1e-6, rtol=1e-5)
    assert not window.states


def _reference_backward(captures, router, coefficient):
    """The removed receipt-replay objective: one cloned context per stage route."""
    counts = torch.zeros(router.experts, dtype=torch.float64)
    for _, selected, _ in captures:
        counts.scatter_add_(0, selected.flatten(), torch.ones(selected.numel(), dtype=torch.float64))
    a = counts.sum()
    fraction = (counts / a).float()
    soft = sum((router.balance_probabilities(x, st) * sel.shape[1]).sum(0)
               for x, sel, st in captures)
    loss = coefficient * router.experts * (fraction * soft / float(a)).sum()
    return torch.autograd.grad(loss, list(router.parameters()), allow_unused=True)


def test_shared_context_statistics_match_per_stage_replay_gradient_without_cap():
    torch.manual_seed(3)
    router = EpochRouter(8, 12, experts=6, max_eligible=3)
    window = RouterBalanceWindow()
    router.balance_recorder = window
    captures = []
    for batch in range(3):  # several windows' worth of routes, many stages each
        x = torch.randn(4, 8)
        for stage in (1, 2, 3, 4):
            route = router.route(x, stage=stage, rows=5)
            part = route.prefix(1 + stage % 3)
            window.capture(router, part)
            captures.append((x, part.expert_ids.detach(), stage))
    # One stored context per compile, not per stage route.
    assert len(window.states[id(router)].contexts) == 3
    assert window.assignments == sum(s.numel() for _, s, _ in captures)
    expected = _reference_backward(captures, router, 0.7)
    window.backward(_SingleRank(), routers={"p": router}, coefficient=0.7)
    for p, grad in zip(router.parameters(), expected):
        if grad is None:
            assert p.grad is None
        else:
            torch.testing.assert_close(p.grad, grad, atol=1e-6, rtol=1e-5)
