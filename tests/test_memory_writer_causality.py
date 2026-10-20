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


# With local_gap=2, slot 0 may read block 2 onward; blocks 0 and 1 are forbidden.
@pytest.mark.parametrize("changed", [0, 1])
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


def test_scored_assignment_does_not_silently_clamp_to_wrong_source():
    from megatoken.memory.writers import AssignmentTargets, writer_distillation
    model = _writer()
    inputs = torch.randn(1, 4, 4, 8)
    with torch.no_grad():
        output = model(
            inputs, mode="full",
            valid_blocks=torch.ones(1, 4, dtype=torch.bool),
        )
    bad = AssignmentTargets(
        slots=torch.tensor([[99]], dtype=torch.long),
        key=torch.randn(1, 1, 12),
        value=torch.randn(1, 1, 10),
        strength=torch.ones(1, 1),
        lifetime_logit=torch.zeros(1, 1),
        distance=torch.zeros(1, 1),
        valid=torch.ones(1, 1, dtype=torch.bool),
    )
    with pytest.raises(ValueError, match="missing output slot"):
        writer_distillation(output, bad)


def test_suffix_writer_has_no_gradient_to_forbidden_prefix():
    model = _writer()
    encoded = torch.randn(1, 6, 4, 8, requires_grad=True)
    output = model(
        encoded, mode="suffix",
        valid_blocks=torch.ones(1, 6, dtype=torch.bool),
        local_gap=2,
    )
    output.value[:, 0].square().sum().backward()
    assert encoded.grad is not None
    assert torch.count_nonzero(encoded.grad[:, :2]) == 0
    assert torch.count_nonzero(encoded.grad[:, 2:]) > 0
