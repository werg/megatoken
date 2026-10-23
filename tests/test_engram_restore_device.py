"""Restored additional donors must follow an already placed native table."""

import os
import warnings

import pytest
import torch

from megatoken.model.config import ModelConfig
from megatoken.model.frontend import NativeNGramTable


@pytest.mark.parametrize("device", ["cpu", "meta", "cuda"])
def test_multiple_donors_restore_to_existing_table_device(device):
    if device == "cuda" and (
        os.environ.get("MEGATOKEN_RUN_CUDA_TESTS") != "1" or not torch.cuda.is_available()
    ):
        pytest.skip("Explicit CUDA test execution is disabled or CUDA is unavailable")
    source = NativeNGramTable(ModelConfig.tiny()).bfloat16()
    rows = torch.tensor([3, 11, 29])
    for _ in range(2):
        source.install_import(rows, torch.randn(3, 128), torch.ones(3), torch.ones(3))
    state = source.state_dict()
    target = NativeNGramTable(ModelConfig.tiny()).to(device=device, dtype=torch.bfloat16)
    # Meta reproduces placement without requiring an accelerator on the test host.
    # Value equality is additionally checked on CPU and, when present, CUDA.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*copying from a non-meta.*")
        target.load_state_dict(state)
    expected = target.embedding.weight.device
    assert len(target.import_branches) == 1
    for name, value in (*target.named_parameters(), *target.named_buffers()):
        assert value.device == expected, name
    branch = target.import_branches[0]
    assert branch.import_rows.dtype == torch.long
    assert branch.descriptors.dtype == torch.float32
    assert branch.import_weights.dtype == torch.float32
    if device != "meta":
        actual = target.lookup(rows.to(device))
        torch.testing.assert_close(actual.cpu(), source.lookup(rows), rtol=0, atol=0)
        actual.float().square().sum().backward()
        assert branch.import_map.weight.grad is not None
