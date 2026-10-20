"""Post-pretraining writers may learn through frozen downstream readers."""
import torch
from torch import nn

from megatoken.memory.writers import AssignmentTargets, RelationWriter, SourceWriter
from megatoken.training.hindsight import (
    PairWriterBatch,
    SourceWriterBatch,
    WriterFitSettings,
    fit_relation_writer_epoch,
    fit_source_writer_epoch,
    frozen_readers,
)


def _source_batch():
    encoded = torch.randn(1, 5, 4, 8)
    targets = AssignmentTargets(
        slots=torch.tensor([[0]], dtype=torch.long),
        key=torch.randn(1, 1, 12),
        value=torch.randn(1, 1, 10),
        strength=torch.ones(1, 1) * 0.5,
        lifetime_logit=torch.ones(1, 1) * -8,
        distance=torch.zeros(1, 1),
        valid=torch.ones(1, 1, dtype=torch.bool),
    )
    return SourceWriterBatch(
        encoded, torch.ones(1, 5, dtype=torch.bool), targets, mode="suffix",
    )


def test_writer_updates_through_frozen_reader_without_reader_parameter_gradients():
    torch.manual_seed(4)
    writer = SourceWriter(
        facets=4, facet_dim=8, semantic_dim=12, value_dim=10,
        hidden=32, heads=4, max_blocks=8,
    )
    reader = nn.Linear(10, 1)
    original_reader = reader.weight.detach().clone()
    original_writer = writer.value.weight.detach().clone()
    optimizer = torch.optim.AdamW(writer.parameters(), lr=1e-3)

    def benefit(output, batch):
        return reader(output.value[:, 0]).square().sum(), 1

    result = fit_source_writer_epoch(
        writer, [_source_batch()], optimizer=optimizer,
        settings=WriterFitSettings(
            distillation_weight=0, behavior_weight=1,
        ), frozen_owners=(reader,), behavior=benefit,
    )
    assert result["updates"] == 1 and result["valid_targets"] == 1
    assert not torch.equal(writer.value.weight, original_writer)
    torch.testing.assert_close(reader.weight, original_reader)
    assert reader.weight.grad is None
    assert reader.weight.requires_grad


def test_invalid_post_pretraining_suffix_gap_rejected():
    data = _source_batch()
    try:
        SourceWriterBatch(
            data.encoded, data.valid_blocks, data.assignments,
            mode="suffix", local_gap=1,
        )
    except ValueError as exc:
        assert "gap" in str(exc)
    else:
        raise AssertionError("Writer may not read adjacent target source blocks")


def test_pair_writer_trains_both_directional_dependence_coordinates():
    torch.manual_seed(2)
    writer = RelationWriter(
        key_dim=6, value_dim=8, statistics_dim=3, hidden=24,
    )
    optimizer = torch.optim.AdamW(writer.parameters(), lr=1e-3)
    batch = PairWriterBatch(
        left_key=torch.randn(2, 6), right_key=torch.randn(2, 6),
        left_value=torch.randn(2, 8), right_value=torch.randn(2, 8),
        statistics=torch.randn(2, 3),
        target_value=torch.randn(2, 8),
        target_strength=torch.ones(2) * 0.5,
        target_lifetime=torch.ones(2) * -6,
        target_dependence=torch.tensor([[1., -2.], [-1., 2.]]),
        valid=torch.ones(2, dtype=torch.bool),
        distance=torch.zeros(2),
    )
    original = writer.dependence.weight.detach().clone()
    result = fit_relation_writer_epoch(
        writer, [batch], settings=WriterFitSettings(), optimizer=optimizer,
    )
    assert result["valid_targets"] == 2
    assert not torch.equal(writer.dependence.weight, original)


def test_reader_context_restores_requires_grad_on_exception():
    reader = nn.Linear(2, 3)
    try:
        with frozen_readers(reader):
            assert not reader.weight.requires_grad
            raise RuntimeError("injected failure")
    except RuntimeError:
        pass
    assert reader.weight.requires_grad
