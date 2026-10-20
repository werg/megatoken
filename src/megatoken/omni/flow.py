"""Facets-only continuous native-latent flow objective and sampler.

A matched codec's decoder is deliberately outside this module: it may only
consume the generated latents, never a gold prompt, transcript, or source U.
"""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F

def flow_matching_loss(prediction, target, valid_mask=None):
    if prediction.shape != target.shape:
        raise ValueError("Flow prediction/target geometry mismatch")
    error = (prediction.float() - target.float()).square()
    if valid_mask is not None:
        mask = valid_mask.to(device=error.device, dtype=torch.bool)
        while mask.ndim < error.ndim:
            mask = mask.unsqueeze(-1)
        mask = mask.expand_as(error)
        count = mask.sum()
        return (error * mask).sum(), count
    return error.sum(), torch.tensor(error.numel(), device=error.device)

class FacetFlow(nn.Module):
    """Generic trainable latent flow adapter; donor weights require explicit import."""
    def __init__(self, facet_dim: int, latent_width: int, hidden: int = 512, facets: int = 32):
        super().__init__()
        self.facets = facets
        self.proj = nn.Linear(facet_dim, hidden)
        self.net = nn.Sequential(nn.Linear(latent_width + hidden + 1, hidden),
                                 nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, latent_width))
    def velocity(self, noisy, time, raw_facets):
        if raw_facets.ndim != 3 or raw_facets.shape[1] != self.facets or noisy.shape[0] != raw_facets.shape[0]:
            raise ValueError("Native flow requires one full raw facet block per event")
        if noisy.ndim != 3:
            raise ValueError("Latents must be [batch, time, channels]")
        condition = self.proj(raw_facets).mean(dim=1)[:, None, :].expand(-1, noisy.shape[1], -1)
        t = torch.as_tensor(time, device=noisy.device, dtype=noisy.dtype).expand(noisy.shape[0], noisy.shape[1], 1)
        return self.net(torch.cat((noisy, condition, t), dim=-1))
    def training_loss(self, target, raw_facets, valid_mask=None, generator=None):
        if target.ndim != 3:
            raise ValueError("Codec target must be a 3D latent sequence")
        noise = torch.randn(target.shape, device=target.device, dtype=target.dtype, generator=generator)
        time = torch.rand((target.shape[0], 1, 1), device=target.device, generator=generator)
        mixed = (1 - time) * noise + time * target
        return flow_matching_loss(self.velocity(mixed, time, raw_facets), target - noise, valid_mask)
    @torch.no_grad()
    def sample(self, raw_facets, shape, steps=32, generator=None):
        if len(shape) != 3 or shape[0] != raw_facets.shape[0] or steps < 1:
            raise ValueError("Invalid latent rollout shape or steps")
        x = torch.randn(shape, device=raw_facets.device, dtype=raw_facets.dtype, generator=generator)
        for n in range(steps):
            x = x + self.velocity(x, (n + 0.5) / steps, raw_facets) / steps
        return x
