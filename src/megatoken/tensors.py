"""Shared host-to-device packing for native byte records."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor


def byte_tensor(payloads: Sequence[bytes], device: torch.device | str) -> tuple[Tensor, Tensor]:
    """Pack bytes once on the host, then transfer independent record rows together.

    Zero-byte records retain their boundary through lengths=0; the padded byte is
    never visible to the source encoder. Pinned CUDA transfers avoid per-byte
    device construction and are shared by training, policy commitment, and serve.
    """
    if not payloads:
        raise ValueError("cannot pack an empty record batch")
    device = torch.device(device)
    lengths = torch.tensor([len(payload) for payload in payloads], dtype=torch.long)
    packed = torch.zeros((len(payloads), max(1, max(map(len, payloads)))), dtype=torch.long)
    for index, payload in enumerate(payloads):
        if payload:
            packed[index, : len(payload)] = torch.frombuffer(
                bytearray(payload), dtype=torch.uint8
            ).long()
    if device.type == "cuda":
        packed, lengths = packed.pin_memory(), lengths.pin_memory()
    return (
        packed.to(device, non_blocking=device.type == "cuda"),
        lengths.to(device, non_blocking=device.type == "cuda"),
    )
