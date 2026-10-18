"""Separated target posterior and causal prior, with FP32 Gaussian statistics."""

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
        noise = torch.randn(
            self.mean.shape, device=self.mean.device, dtype=torch.float32, generator=generator
        )
        return self.mean + self.log_scale.exp() * noise

    def log_prob(self, value: Tensor) -> Tensor:
        return (
            -0.5 * ((value.float() - self.mean) * (-self.log_scale).exp()).square()
            - self.log_scale
            - 0.5 * math.log(2 * math.pi)
        ).sum(-1)

    def detached(self) -> DiagonalGaussian:
        return DiagonalGaussian(self.mean.detach(), self.log_scale.detach())

    def kl(self, other: DiagonalGaussian) -> Tensor:
        """Per-example KL(self || other), summed across innovation coordinates."""
        variance = (2 * (self.log_scale - other.log_scale)).exp()
        difference = (self.mean - other.mean).square() * (-2 * other.log_scale).exp()
        return (other.log_scale - self.log_scale + 0.5 * (variance + difference - 1)).sum(-1)


class GaussianHead(nn.Module):
    def __init__(self, input_width: int, dimensions: int, hidden_width: int) -> None:
        super().__init__()
        self.hidden = nn.Linear(input_width, hidden_width)
        self.statistics = nn.Linear(hidden_width, dimensions * 2)

    def forward(self, context: Tensor) -> DiagonalGaussian:
        # module.to(bfloat16) must not silently turn variance arithmetic into BF16.
        with torch.autocast(device_type=context.device.type, enabled=False):
            hidden = F.silu(
                F.linear(context.float(), self.hidden.weight.float(), self.hidden.bias.float())
            )
            mean, raw = F.linear(
                hidden, self.statistics.weight.float(), self.statistics.bias.float()
            ).chunk(2, -1)
            log_scale = -4.0 + 6.0 * torch.sigmoid(raw)  # scales in [exp(-4), exp(2)]
            return DiagonalGaussian(mean, log_scale)


class TargetPosterior(nn.Module):
    """The only module allowed to see the current target before commitment.

    Byte embeddings are private training parameters. Pooling masks padding and
    ordered convolutions retain local content; no writer/state is called here.
    """

    def __init__(
        self, facet_width: int, source_width: int, root_dim: int, regional_dim: int
    ) -> None:
        super().__init__()
        self.byte_embedding = nn.Embedding(256, source_width)
        self.content = nn.Conv1d(source_width, source_width, kernel_size=3, padding=1)
        self.root = GaussianHead(facet_width + source_width, root_dim, source_width)
        self.regional = GaussianHead(
            facet_width + source_width + root_dim, regional_dim, source_width
        )
        self.alignment = nn.Linear(source_width, facet_width, bias=False)

    def encode(self, bytes_: Tensor, mask: Tensor) -> Tensor:
        if bytes_.shape[1] == 0:
            return self.byte_embedding.weight.new_zeros(
                (bytes_.shape[0], self.byte_embedding.embedding_dim)
            )
        embeddings = self.byte_embedding(bytes_.clamp(0, 255)) * mask[..., None]
        values = F.silu(self.content(embeddings.transpose(1, 2))).transpose(1, 2)
        return (values * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
