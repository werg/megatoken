from fractions import Fraction

import pytest
import torch

from megatoken.controls import effective_event_payloads
from megatoken.generation.latents import DiagonalGaussian
from megatoken.omni.clocks import TimeWindow, available_windows
from megatoken.omni.relations import coordinate_kl, endpoint_targets


def test_distinct_event_requests_are_causal_and_repeatable():
    prompts = effective_event_payloads(
        b"global", ("speech", "caption"), (("Speak English", 2), ("Describe visuals", 7))
    )
    assert prompts[0] != prompts[1]
    assert prompts == effective_event_payloads(
        b"global", ("speech", "caption"), (("Speak English", 2), ("Describe visuals", 7))
    )
    assert b"Describe visuals" not in prompts[0]
    with pytest.raises(ValueError):
        effective_event_payloads(b"", ("same", "same"), (("a", 0), ("b", 1)))


def test_analytic_kl_count_and_duplicate_invariance():
    q = DiagonalGaussian(torch.ones(1, 32, 2), torch.zeros(1, 32, 2))
    p = DiagonalGaussian(torch.zeros(1, 32, 2), torch.zeros(1, 32, 2))
    n, c = coordinate_kl(q, p, q.mean.numel())
    assert torch.allclose(n / c, torch.tensor(0.5))
    q2 = DiagonalGaussian(q.mean.repeat(3, 1, 1), q.log_scale.repeat(3, 1, 1))
    p2 = DiagonalGaussian(p.mean.repeat(3, 1, 1), p.log_scale.repeat(3, 1, 1))
    n2, c2 = coordinate_kl(q2, p2, q2.mean.numel())
    assert torch.allclose(n2 / c2, n / c)


def test_endpoints_and_prefix_limits():
    video = torch.randn(1, 5, 3, 16, 16)
    first, last = endpoint_targets(video, 1, 3)
    assert torch.equal(first, video[:, 1]) and torch.equal(last, video[:, 3])
    with pytest.raises(ValueError):
        available_windows((TimeWindow(Fraction(0), Fraction(2), Fraction(3)),), Fraction(2))
