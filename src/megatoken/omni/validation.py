"""Pre-flight checks on installed pretrained realizers (no data, no weight changes)."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ProbeResult:
    realizer: str
    latent_shape: tuple[int, ...]
    loss: float
    trainable_parameters: int


def check_realizer(realizer, raw_facets, latents, *, mask=None) -> ProbeResult:
    """The donor is unregistered and frozen; the objective is finite and trains only
    the facet condition."""
    trainable = list(realizer.parameters())
    if not trainable or any(
        not name.startswith("condition.") for name, _ in realizer.named_parameters()
    ):
        raise ValueError("Only the facet condition may be a registered realizer parameter")
    loss = realizer.loss(raw_facets, latents, mask)
    if loss.ndim or not torch.isfinite(loss):
        raise ValueError("Realizer objective must be a finite scalar")
    grads = torch.autograd.grad(loss, trainable, allow_unused=True)
    if not any(grad is not None and bool(grad.abs().sum() > 0) for grad in grads):
        raise ValueError("The facet condition receives no realizer objective gradient")
    return ProbeResult(
        realizer.identity,
        tuple(latents.shape),
        float(loss.detach()),
        sum(p.numel() for p in trainable),
    )


@torch.no_grad()
def assert_content_only_invariance(renderer, raw, spec, *, generator_seed: int = 0):
    """A repeated raw action with equal spec and noise must realize identically."""
    outputs = []
    for _ in range(2):
        generator = torch.Generator(device=raw.device).manual_seed(generator_seed)
        outputs.append(renderer.realize(raw.clone(), spec, generator=generator))
    if outputs[0].shape != outputs[1].shape or not torch.equal(*outputs):
        raise ValueError("Renderer is not deterministic under explicit fixed noise")
    return outputs[0]
