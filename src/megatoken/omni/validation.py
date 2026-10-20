"""Pre-flight checks on installed, trained pretrained media components.

Run after loading exact donor checkpoints and before the first optimizer step.
Tests contracts and gradients without acquiring data or changing weights.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch

REQUIRED_FACETS = 32

@dataclass(frozen=True)
class ProbeResult:
    pack: str
    latent_shape: tuple[int, ...]
    loss: float
    trainable_parameters: int
    frozen_parameters: int

def check_pretrained_student(name, student, raw_facets, target_latents, *, valid_mask=None):
    """Check flow objective, facets-only conditioning, and donor codec freeze."""
    if raw_facets.ndim != 3 or raw_facets.shape[1] != REQUIRED_FACETS:
        raise ValueError("Full [B,32,D] raw event required")
    if not raw_facets.is_floating_point() or not torch.isfinite(raw_facets).all():
        raise ValueError("Nonfinite raw facets")
    if target_latents.shape[0] != raw_facets.shape[0] or not torch.isfinite(target_latents).all():
        raise ValueError("Nonfinite or unmatched donor target")
    if name not in ("pocket", "sana", "sana_video"):
        raise ValueError("Unsupported resident pack")
    codec = student.mimi if name == "pocket" else student.vae
    if any(param.requires_grad for param in codec.parameters()):
        raise ValueError("Pretrained codec is not frozen")
    trainable = [p for p in student.parameters() if p.requires_grad]
    if not trainable:
        raise ValueError("No trainable facets-to-donor conditioning projection")
    numerator, count = student.native_loss(raw_facets, target_latents, valid_mask=valid_mask)
    if numerator.ndim or count.ndim or not torch.isfinite(numerator) or count.item() <= 0:
        raise ValueError("Native donor objective must report finite numerator/count")
    grads = torch.autograd.grad(numerator / count, trainable, allow_unused=True)
    if not any(grad is not None and grad.abs().sum() > 0 for grad in grads):
        raise ValueError("Facets-only adapter receives no native objective gradient")
    return ProbeResult(name, tuple(target_latents.shape),
        float((numerator / count).detach().cpu()),
        sum(p.numel() for p in trainable),
        sum(p.numel() for p in codec.parameters()))

@torch.no_grad()
def assert_content_only_invariance(renderer, raw, spec, *, seeded_noise=None):
    """A repeated raw action with equal spec/noise must realize identically.

    The renderer must not have a transcript, reference, peer or core-history
    parameter. It is responsible for honoring an explicit renderer RNG.
    """
    first = renderer.realize(raw, spec, seeded_noise)
    second = renderer.realize(raw.clone(), spec, seeded_noise)
    if first.shape != second.shape or not torch.equal(first, second):
        raise ValueError("Renderer is not deterministic under explicit fixed noise")
    return first
