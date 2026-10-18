import pytest
import torch
import torch.nn.functional as F

from megatoken.generation.distributions import tiled_argmax, tiled_cross_entropy
from megatoken.generation.latents import FacetGaussian


def test_tiled_masked_ce_and_gradients_match_dense_reference():
    torch.manual_seed(41)
    hidden = torch.randn(7, 3, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(11, 3, dtype=torch.float64, requires_grad=True)
    valid = torch.tensor([True, False, True, True, False, True, True, True, False, True, True])
    targets = torch.tensor([0, 2, 3, 5, 6, 9, 10])
    scale = torch.arange(1, 8, dtype=torch.float64) / 7
    actual = tiled_cross_entropy(hidden, weight, targets, valid, position_tile=3, vocabulary_tile=4)
    dense = F.cross_entropy(
        (hidden @ weight.T).masked_fill(~valid, -torch.inf), targets, reduction="none"
    )
    assert torch.allclose(actual, dense, atol=1e-12, rtol=1e-12)
    actual_grad = torch.autograd.grad((actual * scale).sum(), (hidden, weight), retain_graph=True)
    dense_grad = torch.autograd.grad((dense * scale).sum(), (hidden, weight))
    for candidate, reference in zip(actual_grad, dense_grad, strict=True):
        assert torch.allclose(candidate, reference, atol=1e-12, rtol=1e-12)
    assert torch.equal(actual_grad[1][~valid], torch.zeros_like(actual_grad[1][~valid]))


def test_tiled_loss_does_not_save_a_positions_by_vocabulary_matrix():
    hidden = torch.randn(19, 7, requires_grad=True)
    weight = torch.randn(101, 7, requires_grad=True)
    sizes = []
    with torch.autograd.graph.saved_tensors_hooks(
        lambda value: sizes.append(tuple(value.shape)) or value, lambda value: value
    ):
        loss = tiled_cross_entropy(
            hidden,
            weight,
            torch.zeros(19, dtype=torch.long),
            torch.ones(101, dtype=torch.bool),
            position_tile=4,
            vocabulary_tile=17,
        ).sum()
    loss.backward()
    assert (19, 101) not in sizes
    assert all(len(shape) != 2 or shape[1] <= 7 for shape in sizes)


def test_tiled_argmax_is_exact_masked_and_uses_lowest_id_for_ties():
    torch.manual_seed(117)
    hidden = torch.randn(37, 9)
    weight = torch.randn(79, 9)
    valid = torch.ones(79, dtype=torch.bool)
    valid[::7] = False
    rng_before = torch.random.get_rng_state()
    ids = tiled_argmax(hidden, weight, valid, position_tile=11, vocabulary_tile=13)
    expected = (hidden @ weight.T).masked_fill(~valid, -torch.inf).argmax(-1)
    assert torch.equal(ids, expected)
    assert torch.equal(rng_before, torch.random.get_rng_state())
    tied = tiled_argmax(torch.zeros(3, 9), weight, valid, vocabulary_tile=2)
    assert torch.equal(tied, torch.ones(3, dtype=torch.long))


def test_masked_invalid_targets_and_empty_support_are_rejected():
    hidden, weight = torch.ones(1, 2), torch.ones(3, 2)
    with pytest.raises(ValueError, match="empty support"):
        tiled_argmax(hidden, weight, torch.zeros(3, dtype=torch.bool))
    with pytest.raises(ValueError, match="masked support"):
        tiled_cross_entropy(hidden, weight, torch.tensor([1]), torch.tensor([True, False, True]))


def test_gaussian_statistics_remain_float32_in_bfloat16_model():
    head = FacetGaussian(8).bfloat16()
    mean = torch.randn(2, 4, 8).bfloat16()
    distribution = head(mean)
    assert torch.equal(distribution.mean, mean.float())
    assert distribution.mean.dtype == distribution.log_scale.dtype == torch.float32
    assert bool((distribution.log_scale >= -4).all()) and bool((distribution.log_scale <= 1).all())
    assert torch.isfinite(distribution.log_prob(distribution.sample())).all()


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA Tensor Core parity requires GPU hardware"
)
def test_cuda_bfloat16_tile_path_matches_float32_reference():
    torch.manual_seed(997)
    hidden = torch.randn(67, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(521, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    valid = torch.ones(521, device="cuda", dtype=torch.bool)
    valid[::11] = False
    target = torch.ones(67, device="cuda", dtype=torch.long)
    fast = tiled_cross_entropy(hidden, weight, target, valid, position_tile=32, vocabulary_tile=128)
    reference = tiled_cross_entropy(
        hidden, weight, target, valid, position_tile=32, vocabulary_tile=128, tensorcore=False
    )
    assert torch.allclose(fast, reference, atol=0.02, rtol=0.002)
    fast_grad = torch.autograd.grad(fast.mean(), (hidden, weight), retain_graph=True)
    ref_grad = torch.autograd.grad(reference.mean(), (hidden, weight))
    for actual, expected in zip(fast_grad, ref_grad, strict=True):
        assert torch.allclose(actual.float(), expected.float(), atol=0.005, rtol=0.04)


def test_gaussian_density_sums_exactly_the_full_canonical_tensor():
    head = FacetGaussian(5)
    distribution = head(torch.randn(3, 4, 5))
    sample = distribution.sample(torch.Generator().manual_seed(73))
    expected = torch.distributions.Independent(
        torch.distributions.Normal(distribution.mean, distribution.log_scale.exp()), 2
    ).log_prob(sample)
    torch.testing.assert_close(distribution.log_prob(sample), expected)
    assert distribution.log_prob(sample).shape == (3,)
