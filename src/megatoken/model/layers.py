"""Small shared numerical building blocks."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    """Accumulate normalization in FP32, preserving the activation dtype."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        return (normalized * self.weight.float()).to(x.dtype)


class SwiGLU(nn.Module):
    """Bias-free three-matrix FFN; packed gate/up are one parameter owner."""

    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.gate_up = nn.Linear(dim, 2 * hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)


def position_features(length: int, dim: int, device: torch.device, *, offset: int = 0) -> Tensor:
    """Fixed positional descriptors never introduce a live-state routing dependency."""
    positions = torch.arange(offset, offset + length, device=device, dtype=torch.float32)
    return position_features_at(positions, dim)


def position_features_at(positions: Tensor, dim: int) -> Tensor:
    """Sinusoidal descriptors for an explicit position tensor of any shape."""
    frequencies = torch.exp(
        -math.log(10_000.0) * torch.arange(0, dim, 2, device=positions.device).float() / dim
    )
    angles = positions.float()[..., None] * frequencies
    return torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(-2)[..., :dim]


def masked_mean(x: Tensor, mask: Tensor, dim: int = 1) -> Tensor:
    weights = mask.to(x.dtype).unsqueeze(-1)
    return (x * weights).sum(dim) / weights.sum(dim).clamp_min(1)


def initialize_module(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, std=1 / math.sqrt(module.in_features))
        if module.bias is not None:
            nn.init.zeros_(module.bias)
