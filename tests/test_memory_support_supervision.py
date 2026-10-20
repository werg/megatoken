"""Retrieved evidence misses must never become negative/no-answer supervision."""

import torch

from megatoken.memory.reader import local_source_value_loss, selected_evidence
from megatoken.training.memory import annotated_ranking_loss


def _scores(mask):
    query = torch.tensor([[1.0, 0.0]])
    keys = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    return selected_evidence(query, keys, masks=torch.tensor([mask]))


def test_masked_positive_is_not_an_explicit_null_target():
    scores = _scores([False, True])
    positive = torch.tensor([[True, False]])
    complete = torch.tensor([True])
    features = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    source, n_source = local_source_value_loss(
        scores, features, positive=positive, complete_annotations=complete
    )
    rank, n_rank = annotated_ranking_loss(scores, positive, complete)
    assert n_source == n_rank == 0
    assert source == 0
    assert rank == 0


def test_truly_empty_complete_annotation_trains_null():
    scores = _scores([True, True])
    positive = torch.tensor([[False, False]])
    complete = torch.tensor([True])
    features = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    source, n_source = local_source_value_loss(
        scores, features, positive=positive, complete_annotations=complete
    )
    rank, n_rank = annotated_ranking_loss(scores, positive, complete)
    assert n_source == n_rank == 1
    assert source.item() > 0
    assert rank.item() > 0


def test_visible_useful_source_keeps_bilateral_gradients():
    query = torch.tensor([[0.8, 0.6]], requires_grad=True)
    key = torch.tensor([[[0.2, 0.9], [0.9, 0.1]]], requires_grad=True)
    scores = selected_evidence(query, key)
    source, count = local_source_value_loss(
        scores,
        torch.tensor([[[0.0, 1.0], [1.0, 0.0]]]),
        positive=torch.tensor([[True, False]]),
        complete_annotations=torch.tensor([True]),
    )
    assert count == 1
    source.backward()
    assert query.grad is not None and query.grad.abs().sum() > 0
    assert key.grad is not None and key.grad.abs().sum() > 0


def test_empty_selected_evidence_is_a_valid_explicit_null():
    scores = selected_evidence(
        torch.tensor([[1.0, 0.0]]),
        torch.empty(1, 0, 2),
    )
    assert scores.weights.shape == (1, 0)
    torch.testing.assert_close(scores.null_weight, torch.ones(1))
    loss, n = local_source_value_loss(
        scores,
        torch.empty(1, 0, 3),
        positive=torch.empty(1, 0, dtype=torch.bool),
        complete_annotations=torch.ones(1, dtype=torch.bool),
    )
    assert n == 1 and loss.item() == 0
