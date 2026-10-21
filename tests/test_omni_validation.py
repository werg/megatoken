"""Realizer pre-flight and the content-only replay invariant."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from megatoken.model.media import MediaOutputSpec
from megatoken.omni.realizers import FacetCondition, Realizer
from megatoken.omni.validation import assert_content_only_invariance, check_realizer
from tests.media_support import LATENT, FakeRealizer

FACETS, WIDTH = 32, 8


def raw(seed=0):
    return torch.randn(1, FACETS, WIDTH, generator=torch.Generator().manual_seed(seed))


def latents(modality="audio"):
    return torch.randn(1, *LATENT[modality])


def test_preflight_reports_a_frozen_donor_and_a_trainable_condition():
    r = FakeRealizer("audio", FACETS, WIDTH)
    result = check_realizer(r, raw(), latents())
    assert result.realizer == r.identity and result.latent_shape == (1, 6, 4)
    assert result.loss > 0
    assert result.trainable_parameters == sum(p.numel() for p in r.condition.parameters())
    # Pre-flight reads gradients only: no weight changed, none accumulated.
    assert all(p.grad is None for p in r.parameters())


@pytest.mark.parametrize(
    ("modality", "mask"),
    [("audio", [[1, 1, 1, 0, 0, 0]]), ("image", None), ("video", [[1, 1, 0]])],
)
def test_preflight_covers_every_modality_with_a_validity_mask(modality, mask):
    r = FakeRealizer(modality, FACETS, WIDTH)
    mask = None if mask is None else torch.tensor(mask, dtype=torch.bool)
    assert check_realizer(r, raw(), latents(modality), mask=mask).realizer == r.identity


def test_preflight_rejects_a_registered_donor_parameter():
    class Leaky(FakeRealizer):
        def __init__(self):
            super().__init__("audio", FACETS, WIDTH)
            self.donor_copy = nn.Linear(2, 2)  # a donor weight in the module tree

    with pytest.raises(ValueError, match="Only the facet condition"):
        check_realizer(Leaky(), raw(), latents())


def test_preflight_rejects_a_realizer_without_trainable_state():
    class Bare(Realizer):
        name, modality, _donor = "pocket", "audio", "d"
        checkpoint_identity = "00"

        def loss(self, raw, latents, mask, generator=None):
            return latents.mean()

    with pytest.raises(ValueError, match="Only the facet condition"):
        check_realizer(Bare(), raw(), latents())


def test_preflight_rejects_a_non_finite_or_non_scalar_objective():
    class Broken(FakeRealizer):
        def __init__(self, value):
            super().__init__("audio", FACETS, WIDTH)
            self.value = value

        def loss(self, raw, latents, mask, generator=None):
            return self.value * self.condition(raw).sum()

    with pytest.raises(ValueError, match="finite scalar"):
        check_realizer(Broken(float("nan")), raw(), latents())

    class Vector(Broken):
        def loss(self, raw, latents, mask, generator=None):
            return self.condition(raw).mean(dim=(0, 1))

    with pytest.raises(ValueError, match="finite scalar"):
        check_realizer(Vector(1.0), raw(), latents())


def test_preflight_requires_the_objective_to_reach_the_condition():
    class Disconnected(FakeRealizer):
        def loss(self, raw, latents, mask, generator=None):
            # Depends on the condition parameters, but with exactly zero gradient.
            return (0 * self.condition(raw)).sum() + latents.mean()

    with pytest.raises(ValueError, match="no realizer objective gradient"):
        check_realizer(Disconnected("audio", FACETS, WIDTH), raw(), latents())


def test_preflight_requires_a_locked_identity():
    r = FakeRealizer("audio", FACETS, WIDTH)
    r.checkpoint_identity = ""
    with pytest.raises(ValueError, match="locked pack manifest"):
        check_realizer(r, raw(), latents())


def test_condition_rejects_partial_facet_blocks():
    condition = FacetCondition(FACETS, WIDTH, 5)
    with pytest.raises(ValueError, match="complete raw facet"):
        condition(torch.randn(1, FACETS - 1, WIDTH))
    with pytest.raises(ValueError, match="finite"):
        condition(torch.full((1, FACETS, WIDTH), float("inf")))
    assert condition(raw()).shape == (1, FACETS, 5)


def test_replay_invariance_accepts_a_deterministic_realizer():
    r = FakeRealizer("audio", FACETS, WIDTH)
    spec = MediaOutputSpec("audio", sample_rate=24000, sample_count=48)
    out = assert_content_only_invariance(r, raw(), spec)
    assert out.shape == (1, 48)


def test_replay_invariance_gives_each_call_the_same_explicit_noise_and_a_private_copy():
    seen, original = [], raw()

    class Sampler:
        def realize(self, raw, spec, generator=None):
            seen.append(generator.initial_seed())
            noise = torch.randn(raw.shape, generator=generator)
            raw.add_(1.0)  # scribbling on its input must not leak into the next call
            return (raw + noise).mean(dim=1)

    out = assert_content_only_invariance(Sampler(), original, None, generator_seed=17)
    assert seen == [17, 17] and out.shape == (1, WIDTH)
    assert torch.equal(original, raw())


def test_replay_invariance_rejects_hidden_state_and_global_randomness():
    class GlobalRandom:
        def realize(self, raw, spec, generator=None):
            return raw.mean(1) + torch.randn(1, WIDTH)

    with pytest.raises(ValueError, match="deterministic"):
        assert_content_only_invariance(GlobalRandom(), raw(), None)

    class Counter:
        calls = 0

        def realize(self, raw, spec, generator=None):
            self.calls += 1
            return raw.mean(1) * self.calls

    with pytest.raises(ValueError, match="deterministic"):
        assert_content_only_invariance(Counter(), raw(), None)

    class Shape:
        calls = 0

        def realize(self, raw, spec, generator=None):
            self.calls += 1
            return raw.mean(1)[:, : WIDTH - self.calls]

    with pytest.raises(ValueError, match="deterministic"):
        assert_content_only_invariance(Shape(), raw(), None)
