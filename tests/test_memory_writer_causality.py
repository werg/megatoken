"""Causal noninterference gates for the source writer.

These use deterministic CPU weights, not trained retrieval quality.
"""
import pytest
import torch

from megatoken.memory.writers import SourceWriter


def _writer():
    torch.manual_seed(31)
    return SourceWriter(
        facets=4, facet_dim=8, semantic_dim=12, value_dim=10,
        hidden=32, heads=4, max_blocks=8,
    ).eval()


@pytest.mark.parametrize("changed", [0, 1, 2])
def test_suffix_writer_ignores_current_and_prefix_content(changed):
    model = _writer()
    original = torch.randn(1, 6, 4, 8)
    available = torch.ones(1, 6, dtype=torch.bool)
    with torch.no_grad():
        baseline = model(original, mode="suffix", valid_blocks=available)
        perturbed = original.clone()
        perturbed[:, changed] += torch.randn_like(perturbed[:, changed]) * 100
        changed_output = model(perturbed, mode="suffix", valid_blocks=available)
    assert baseline.valid[0, 0]
    for field in ("key", "value", "strength", "lifetime_logit"):
        torch.testing.assert_close(
            getattr(baseline, field)[:, 0],
            getattr(changed_output, field)[:, 0],
            rtol=0, atol=1e-6,
        )


def test_suffix_writer_reads_permitted_future_evidence_and_keeps_invalid_slots_masked():
    model = _writer()
    original = torch.randn(1, 6, 4, 8)
    available = torch.ones(1, 6, dtype=torch.bool)
    with torch.no_grad():
        baseline = model(original, mode="suffix", valid_blocks=available)
        future = original.clone()
        future[:, 3, 0, 0] += 40
        changed = model(future, mode="suffix", valid_blocks=available)
    assert baseline.valid.tolist() == [[True, True, True, True, False, False]]
    assert not torch.allclose(baseline.value[:, 0], changed.value[:, 0])


def test_suffix_writer_requires_positive_gap():
    model = _writer()
    with pytest.raises(ValueError, match="positive gap"):
        model(
            torch.randn(1, 4, 4, 8), mode="suffix",
            valid_blocks=torch.ones(1, 4, dtype=torch.bool), local_gap=0,
        )
