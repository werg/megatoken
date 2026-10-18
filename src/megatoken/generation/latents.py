"""One Gaussian in the same raw coordinates as the canonical chunk encoder."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class DiagonalGaussian:
    mean: Tensor
    log_scale: Tensor

    def sample(self, generator: torch.Generator | None = None) -> Tensor:
        """The policy's only random draw, including every facet coordinate."""
        noise = torch.randn(
            self.mean.shape, device=self.mean.device, dtype=torch.float32, generator=generator
        )
        return self.mean + self.log_scale.exp() * noise

    def elementwise_log_prob(self, value: Tensor) -> Tensor:
        if value.shape != self.mean.shape:
            raise ValueError("Gaussian action geometry differs from its mean")
        return (
            -0.5 * ((value.float() - self.mean) * (-self.log_scale).exp()).square()
            - self.log_scale
            - 0.5 * math.log(2 * math.pi)
        )

    def log_prob(self, value: Tensor) -> Tensor:
        """Per-example density, summing all raw facet coordinates."""
        return self.elementwise_log_prob(value).flatten(1).sum(-1)


class FacetGaussian(nn.Module):
    """Preserve the predictor's mean exactly; learn only a bounded FP32 scale.

    There is no private target encoder, mean projection, root latent, or regional
    latent. The scale head reads the predicted facets before the single draw.
    """

    def __init__(self, width: int) -> None:
        super().__init__()
        self.log_scale = nn.Linear(width, width)
        nn.init.zeros_(self.log_scale.weight)
        nn.init.constant_(self.log_scale.bias, math.log(4.0))

    def forward(self, mean: Tensor) -> DiagonalGaussian:
        # Casting a served model to BF16 must not reduce variance precision.
        with torch.autocast(device_type=mean.device.type, enabled=False):
            mean = mean.float()
            raw = F.linear(mean, self.log_scale.weight.float(), self.log_scale.bias.float())
            return DiagonalGaussian(mean, -4.0 + 5.0 * raw.sigmoid())
