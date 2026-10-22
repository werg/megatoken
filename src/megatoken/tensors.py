"""Shared host-to-device packing for native byte records."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor


def byte_tensor(payloads: Sequence[bytes], device: torch.device | str) -> tuple[Tensor, Tensor]:
    """Pack bytes once on the host, then move the whole batch in one transfer.

    Zero-byte records retain their boundary through lengths=0; the padded byte is
    never visible to the source encoder. Records are a few kilobytes, so a plain
    copy beats allocating pinned memory on every call; the lengths ride in the
    first column so a single host-to-device copy carries both. Values are bytes
    and lengths are measured here, so they are valid by construction; consumers
    validate only host tensors (``tensors.is_host``) and never synchronize a device.
    """
    if not payloads:
        raise ValueError("cannot pack an empty record batch")
    device = torch.device(device)
    width = max(1, max(map(len, payloads)))
    block = torch.zeros((len(payloads), 1 + width), dtype=torch.long)
    for index, payload in enumerate(payloads):
        if payload:
            block[index, 0] = len(payload)
            block[index, 1 : 1 + len(payload)] = torch.frombuffer(
                bytearray(payload), dtype=torch.uint8
            ).long()
    block = block.to(device)
    return block[:, 1:].contiguous(), block[:, 0].contiguous()


def is_host(tensor: Tensor) -> bool:
    """Whether inspecting this tensor's values costs no device synchronization."""
    return tensor.device.type == "cpu"
