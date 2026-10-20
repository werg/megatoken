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
    window = RouterBalanceWindow(max_assignments=10)
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
    assert not window.receipts


def test_replay_capacity_counts_executed_prefix_not_compiled_suffix():
    router = EpochRouter(8, 12, experts=5, max_eligible=3)
    route = router.route(torch.randn(2, 8), stage=1, rows=9)
    window = RouterBalanceWindow(max_assignments=2)
    window.capture(router, route.prefix(1))
    assert sum(r.assignments for r in window.receipts) == 2
