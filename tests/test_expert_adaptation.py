import torch

from megatoken.model.experts import ExpertBank, LinearExpertBank


def _setup_bank(bank):
    with torch.no_grad():
        bank.gate_up_adapter_up.fill_(0.03)
        bank.down_adapter_up.fill_(0.03)


def test_expert_bank_zero_weights_are_exact_baseline():
    torch.manual_seed(3)
    bank = ExpertBank(3, 5, 7, adapter_experts=4, adapter_rank=2)
    rows = torch.randn(2, 4, 5)
    ids = torch.tensor([[0, 1, 2, 0], [2, 1, 0, 2]])
    confidence = torch.rand(2, 4)
    weights = torch.zeros(2, 4, 4)

    baseline = bank(rows, ids, confidence)
    adapted = bank(rows, ids, confidence, weights)
    torch.testing.assert_close(adapted, baseline, rtol=0, atol=0)


def test_expert_bank_adapters_are_specific_to_base_expert():
    bank = ExpertBank(2, 3, 4, adapter_experts=2, adapter_rank=1)
    with torch.no_grad():
        bank.gate_up.zero_()
        bank.down.zero_()
        bank.gate_up_adapter_down.zero_()
        bank.gate_up_adapter_up.zero_()
        bank.down_adapter_down.zero_()
        bank.down_adapter_up.zero_()
        # Expert 0 has a gate-up correction; expert 1 has none.
        bank.gate_up_adapter_down[0, 0, 0, 0] = 1
        bank.gate_up_adapter_up[0, 0, 0, 0] = 1
        bank.gate_up_adapter_up[0, 0, 4, 0] = 1
    rows = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    ids = torch.tensor([0, 1])
    weights = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    output = bank(rows, ids, torch.ones(2), weights)

    assert output[0].abs().sum() > 0
    assert output[1].abs().sum() == 0


def test_expert_bank_gradients_reach_both_matrix_adapters_and_mixture():
    torch.manual_seed(5)
    bank = ExpertBank(2, 4, 5, adapter_experts=3, adapter_rank=2)
    _setup_bank(bank)
    rows = torch.randn(2, 3, 4)
    ids = torch.tensor([[0, 1, 0], [1, 0, 1]])
    confidence = torch.ones(2, 3)
    weights = torch.tensor(
        [
            [[0.7, 0.0, 0.3], [0.0, 0.6, 0.4], [0.4, 0.6, 0.0]],
            [[0.3, 0.7, 0.0], [0.8, 0.0, 0.2], [0.0, 0.4, 0.6]],
        ],
        requires_grad=True,
    )
    bank(rows, ids, confidence, weights).square().sum().backward()

    assert weights.grad is not None and weights.grad.abs().sum() > 0
    for parameter in (
        bank.gate_up_adapter_down,
        bank.gate_up_adapter_up,
        bank.down_adapter_down,
        bank.down_adapter_up,
    ):
        assert parameter.grad is not None
        assert parameter.grad.abs().sum() > 0


def test_linear_expert_bank_batched_mixtures_and_all_experts_addressable():
    torch.manual_seed(7)
    bank = LinearExpertBank(
        4, 3, 6, adapter_experts=5, adapter_rank=2, adapter_topk=2
    )
    with torch.no_grad():
        bank.adapter_up.fill_(0.02)
    rows = torch.randn(2, 4, 3)
    ids = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]])
    confidence = torch.ones(2, 4)
    # Exercise a different adapter on each row, including adapter 4 and all
    # four base experts. Sparse zeros must not cause extra factor dispatches.
    weights = torch.zeros(2, 4, 5)
    for batch in range(2):
        for event in range(4):
            weights[batch, event, (batch + event) % 5] = 1
    output = bank(rows, ids, confidence, weights)
    assert output.shape == (2, 4, 6)
    assert output.isfinite().all()
    assert bank.weight.shape == (4, 6, 3)
    output.sum().backward()
    assert bank.adapter_up.grad is not None
    assert bank.adapter_up.grad.abs().sum() > 0


def test_linear_expert_bank_zero_adaptation_matches_grouped_linear():
    torch.manual_seed(11)
    bank = LinearExpertBank(3, 4, 2, adapter_experts=4, adapter_rank=2)
    rows = torch.randn(2, 3, 4)
    ids = torch.tensor([[2, 0, 1], [1, 2, 0]])
    confidence = torch.tensor([[1.0, 0.5, 0.8], [0.2, 0.9, 0.7]])
    weights = torch.zeros(2, 3, 4)
    torch.testing.assert_close(
        bank(rows, ids, confidence), bank(rows, ids, confidence, weights), rtol=0, atol=0
    )
