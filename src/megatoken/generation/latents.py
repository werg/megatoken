"""One Gaussian in the same raw coordinates as the canonical chunk encoder."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn


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

    def kl(self, other: DiagonalGaussian) -> Tensor:
        """Exact per-example KL(self || other), including encoder noise variance."""
        if self.mean.shape != other.mean.shape:
            raise ValueError("Gaussian correspondence requires identical facet geometry")
        residual = (self.mean.float() - other.mean.float()) * (-other.log_scale).exp()
        variance_ratio = (2 * (self.log_scale - other.log_scale)).exp()
        elements = other.log_scale - self.log_scale + 0.5 * (variance_ratio + residual.square() - 1)
        return elements.flatten(1).sum(-1)

    def detach(self) -> DiagonalGaussian:
        return DiagonalGaussian(self.mean.detach(), self.log_scale.detach())


class FacetGaussian(nn.Module):
    """Preserve the predictor's mean exactly; learn only a bounded FP32 scale.

    There is no private target encoder, mean projection, root latent, or regional
    latent. The scale head reads the predicted facets before the single draw.
    """

    def __init__(self, width: int, initial_std: float = 0.1) -> None:
        super().__init__()
        if not math.exp(-4) < initial_std < math.exp(1):
            raise ValueError("initial Gaussian scale must lie strictly inside its finite bounds")
        self.log_scale = nn.Linear(width, width)
        nn.init.zeros_(self.log_scale.weight)
        fraction = (math.log(initial_std) + 4.0) / 5.0
        nn.init.constant_(self.log_scale.bias, math.log(fraction / (1.0 - fraction)))

    def forward(self, mean: Tensor) -> DiagonalGaussian:
        # Casting a served model to BF16 must not reduce variance precision.
        with torch.autocast(device_type=mean.device.type, enabled=False):
            mean = mean.float()
            # ``log_scale`` may be a routed shared-pool projection. Call the
            # module so its expert dispatch and gradients are preserved; run
            # its input in the pool's storage dtype, then return to FP32 for
            # the bounded scale calculation and all density math.
            projection_dtype = self.log_scale.weight.dtype
            raw = self.log_scale(mean.to(projection_dtype)).float()
            return DiagonalGaussian(mean, -4.0 + 5.0 * raw.sigmoid())

