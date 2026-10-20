"""Post-pretraining writer fitting in pinned, frozen native/KB coordinates.

A separate owner deliberately does not mutate the generator, its optimizer, or
the free KB. The input must be actual frozen E source facets with complete
permitted source-view footprints prepared by the data pipeline. No teacher key,
target semantic value or answer enters the writer's source input.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator

import torch
from torch import Tensor, nn

from megatoken.memory.writers import (
    AssignmentTargets,
    PairWriterOutput,
    RelationWriter,
    SourceWriter,
    WriterOutput,
    relation_distillation,
    writer_distillation,
)


@dataclass(frozen=True)
class SourceWriterBatch:
    encoded: Tensor  # frozen, actual E(source) [B, blocks, facets, width]
    valid_blocks: Tensor
    assignments: AssignmentTargets
    mode: str  # explicit physical "suffix" or "full" view
    local_gap: int = 2

    def __post_init__(self):
        if self.encoded.ndim != 4 or self.encoded.requires_grad:
            raise ValueError("Source-writer inputs must be frozen native E facet blocks")
        if self.mode not in {"suffix", "full"}:
            raise ValueError("Explicit physical source view is required")
        if self.mode == "suffix" and self.local_gap < 2:
            raise ValueError("Local hindsight requires the declared j >= i+2 gap")


@dataclass(frozen=True)
class PairWriterBatch:
    left_key: Tensor
    right_key: Tensor
    left_value: Tensor
    right_value: Tensor
    statistics: Tensor
    target_value: Tensor
    target_strength: Tensor
    target_lifetime: Tensor
    target_dependence: Tensor
    valid: Tensor
    distance: Tensor


@dataclass(frozen=True)
class WriterFitSettings:
    learning_rate: float = 1e-4
    grad_clip: float = 1.0
    distillation_weight: float = 1.0
    behavior_weight: float = 0.0

    def __post_init__(self):
        if self.learning_rate <= 0 or self.grad_clip <= 0:
            raise ValueError("Writer optimizer rates/clip must be positive")
        if self.distillation_weight < 0 or self.behavior_weight < 0:
            raise ValueError("Writer loss weights must be nonnegative")
        if not self.distillation_weight and not self.behavior_weight:
            raise ValueError("At least one writer supervision path is required")


@contextmanager
def frozen_readers(*readers: nn.Module) -> Iterator[None]:
    """Freeze parameters while retaining autograd THROUGH their operations.

    Do not use torch.no_grad around reader evaluation: that would also remove
    gradients needed to learn the input writer's semantic values.
    """
    owned: list[tuple[nn.Parameter, bool]] = []
    seen: set[int] = set()
    for reader in readers:
        for parameter in reader.parameters():
            if id(parameter) not in seen:
                owned.append((parameter, parameter.requires_grad))
                seen.add(id(parameter))
                parameter.requires_grad_(False)
    try:
        yield
    finally:
        for parameter, old in owned:
            parameter.requires_grad_(old)


# Behavior callbacks evaluate a frozen downstream C/P/R/KB reader with the
# writer's live outputs and return (unnormalized numerator, true valid count).
SourceBehavior = Callable[[WriterOutput, SourceWriterBatch], tuple[Tensor, int]]
PairBehavior = Callable[[PairWriterOutput, PairWriterBatch], tuple[Tensor, int]]


def _normalized_loss(numerator: Tensor, count: int, label: str) -> Tensor:
    if numerator.ndim != 0 or not bool(torch.isfinite(numerator).all()):
        raise ValueError(f"{label} must return a finite scalar numerator")
    if count < 0:
        raise ValueError(f"{label} returned negative evidence count")
    return numerator / max(1, count)


def _update(writer: nn.Module, optimizer: torch.optim.Optimizer,
            loss: Tensor, *, grad_clip: float) -> None:
    if not loss.requires_grad:
        raise RuntimeError("Writer loss has no live gradient path")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradients = [p.grad for p in writer.parameters() if p.grad is not None]
    if not gradients or any(not bool(torch.isfinite(g).all()) for g in gradients):
        optimizer.zero_grad(set_to_none=True)
        raise FloatingPointError("Writer gradient is missing or nonfinite")
    torch.nn.utils.clip_grad_norm_(writer.parameters(), grad_clip, error_if_nonfinite=True)
    optimizer.step()


def fit_source_writer_epoch(
    writer: SourceWriter,
    batches: Iterable[SourceWriterBatch], *,
    settings: WriterFitSettings,
    optimizer: torch.optim.Optimizer,
    frozen_owners: tuple[nn.Module, ...] = (),
    behavior: SourceBehavior | None = None,
) -> dict[str, float]:
    if settings.behavior_weight and behavior is None:
        raise ValueError("Behavioral writer training requires a frozen-reader callback")
    writer.train()
    updates = 0
    examples = 0
    weighted_loss = 0.
    with frozen_readers(*frozen_owners), torch.enable_grad():
        for batch in batches:
            prediction = writer(
                batch.encoded.detach(), mode=batch.mode,
                valid_blocks=batch.valid_blocks, local_gap=batch.local_gap,
            )
            d_sum, d_count = writer_distillation(prediction, batch.assignments)
            loss = settings.distillation_weight * _normalized_loss(
                d_sum, d_count, "assignment distillation",
            )
            # Count supervised examples once, only for objectives that train.
            counted = d_count if settings.distillation_weight else 0
            if settings.behavior_weight:
                assert behavior is not None
                b_sum, b_count = behavior(prediction, batch)
                loss = loss + settings.behavior_weight * _normalized_loss(
                    b_sum, b_count, "frozen-reader behavior",
                )
                counted = max(counted, b_count)
            # No-update windows are legal; they cannot fabricate examples.
            if counted == 0:
                continue
            _update(writer, optimizer, loss, grad_clip=settings.grad_clip)
            updates += 1
            examples += counted
            weighted_loss += float(loss.detach())
    return {"updates": updates, "valid_targets": examples,
            "mean_step_loss": weighted_loss / max(1, updates)}


def fit_relation_writer_epoch(
    writer: RelationWriter,
    batches: Iterable[PairWriterBatch], *,
    settings: WriterFitSettings,
    optimizer: torch.optim.Optimizer,
    frozen_owners: tuple[nn.Module, ...] = (),
    behavior: PairBehavior | None = None,
) -> dict[str, float]:
    if settings.behavior_weight and behavior is None:
        raise ValueError("Behavioral pair training requires a frozen-reader callback")
    writer.train()
    updates = 0
    examples = 0
    loss_sum = 0.
    with frozen_readers(*frozen_owners), torch.enable_grad():
        for batch in batches:
            # Only source writer outputs and permitted source statistics enter
            # the relation writer. Optimized target fields are strictly labels.
            prediction = writer(
                batch.left_key.detach(), batch.right_key.detach(),
                batch.left_value.detach(), batch.right_value.detach(),
                batch.statistics.detach(),
            )
            d_sum, d_count = relation_distillation(
                prediction, value=batch.target_value,
                strength=batch.target_strength, lifetime_logit=batch.target_lifetime,
                dependence_logits=batch.target_dependence,
                valid=batch.valid, distance=batch.distance,
            )
            loss = settings.distillation_weight * _normalized_loss(
                d_sum, d_count, "pair distillation",
            )
            # Count supervised examples once, only for objectives that train.
            counted = d_count if settings.distillation_weight else 0
            if settings.behavior_weight:
                assert behavior is not None
                b_sum, b_count = behavior(prediction, batch)
                loss = loss + settings.behavior_weight * _normalized_loss(
                    b_sum, b_count, "frozen pair-reader behavior",
                )
                counted = max(counted, b_count)
            if counted == 0:
                continue
            _update(writer, optimizer, loss, grad_clip=settings.grad_clip)
            updates += 1
            examples += counted
            loss_sum += float(loss.detach())
    return {"updates": updates, "valid_targets": examples,
            "mean_step_loss": loss_sum / max(1, updates)}
