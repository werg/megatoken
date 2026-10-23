"""Packing a wide batch in row blocks gives exactly the bytes of packing it in one piece."""

import torch

from megatoken.memory import transducer_packing as packing


def test_blocked_packing_is_bit_identical(monkeypatch):
    torch.manual_seed(3)
    values = torch.randn(37, 300)
    seeds = packing.row_seeds(torch.arange(37), "value", 5, 1)
    whole4, whole8 = packing.pack_fp4(values, seeds), packing.pack_fp8(values, seeds)
    plain4 = packing.pack_fp4(values)
    monkeypatch.setattr(packing, "_BLOCK_ELEMENTS", 300 * 4)  # four rows per block
    blocked4, blocked8 = packing.pack_fp4(values, seeds), packing.pack_fp8(values, seeds)
    assert torch.equal(whole4.codes, blocked4.codes)
    assert torch.equal(
        whole4.group_scales.view(torch.uint8), blocked4.group_scales.view(torch.uint8)
    )
    assert torch.equal(whole4.row_scales, blocked4.row_scales)
    assert torch.equal(whole8.codes.view(torch.uint8), blocked8.codes.view(torch.uint8))
    assert torch.equal(whole8.scales, blocked8.scales)
    assert torch.equal(plain4.codes, packing.pack_fp4(values).codes)
