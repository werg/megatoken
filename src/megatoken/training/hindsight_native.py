"""Actual native grouped-meridian continuation loss for local/cross-source writers.

Keep one chronological prediction stream per frozen native prefix. Later
assistant targets are scored with ordinary native P/C/R and the exact
source-associated writer-produced semantic value; observations advance their
saved selected raw facets. Frozen reader PARAMETERS remain in autograd.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import torch
from torch import Tensor

from megatoken.memory.context import admit_kb
from megatoken.memory.semantic import AdmittedGroup, Constituent
from megatoken.memory.writers import WriterOutput
from megatoken.model.multitrack import MultiTrackGenerator, MultiTrackState
from megatoken.training.hindsight import SourceWriterBatch


@dataclass(frozen=True)
class ContinuationMeridian:
    """A chronological native group, scored once or replayed as original raw."""
    meridian: int
    tracks: tuple[str, ...]
    roles: tuple[int, ...]
    raw: Tensor | None = None       # [1, n_events, facets, width] frozen observation
    plans: tuple | None = None      # prepared native teacher-forced action plans
    controls: tuple[bytes, ...] = ()
    effort: int = 6
    teacher_forced_action: bool = False

    def __post_init__(self):
        if len(self.tracks) < 1 or len(set(self.tracks)) != len(self.tracks):
            raise ValueError("Native continuation needs unique nonempty tracks")
        if len(self.roles) != len(self.tracks) or (
            self.controls and len(self.controls) != len(self.tracks)
        ):
            raise ValueError("All continuation group fields must match tracks")
        if (self.raw is None) == (self.plans is None):
            raise ValueError("A native continuation is either scored or a fixed observation")
        if self.plans is not None and len(self.plans) != len(self.tracks):
            raise ValueError("Every scored native event requires an original action plan")


@dataclass(frozen=True)
class WriterContinuation:
    batch_index: int
    source_slot: int
    source_meridian: int
    initial_state: MultiTrackState
    events: tuple[ContinuationMeridian, ...]
    scope: str
    identity: str

    def __post_init__(self):
        if self.batch_index < 0 or self.source_slot < 0 or (
            self.source_meridian < 0 or not self.scope or not self.identity
        ):
            raise ValueError("Native writer target requires source assignment and lineage")
        clock = int(self.initial_state.meridians_seen[0])
        if self.initial_state.batch_size != 1:
            raise ValueError("Each writer trajectory must preserve an independent native history")
        for event in self.events:
            if event.meridian != clock:
                raise ValueError("Native writer event sequence has a gap or replay duplicate")
            if event.plans is not None and clock < self.source_meridian + 2:
                raise ValueError("Local hindsight cannot score j < i+2")
            clock += 1


def native_source_continuation(
    generator: MultiTrackGenerator,
    trajectories: Sequence[WriterContinuation],
) -> Callable[[WriterOutput, SourceWriterBatch], tuple[Tensor, int]]:
    """Construct a callback usable in fit_source_writer_epoch(behavior=...).

    The caller supplies the SAME frozen native trajectory/control schedule as
    supervised training and pins its assignment/source scopes. This is not an
    answer-conditioned writer: writer.forward reads only permitted E(source).
    """
    targets = tuple(trajectories)

    def evaluate(
        predicted: WriterOutput, batch: SourceWriterBatch,
    ) -> tuple[Tensor, int]:
        total = predicted.value.sum() * 0
        count = 0
        for target in targets:
            if (
                target.batch_index >= predicted.value.shape[0]
                or target.source_slot >= predicted.value.shape[1]
            ):
                raise ValueError("Native writer continuation has invalid source output slot")
            index = target.batch_index
            slot = target.source_slot
            if not bool(predicted.valid[index, slot]):
                continue
            state = target.initial_state
            source_value = predicted.value[index, slot][None]
            strength = predicted.strength[index, slot].reshape(1)
            lifetime = predicted.lifetime_logit[index, slot].reshape(1)
            anchor = int(state.meridians_seen[0])
            group = AdmittedGroup(
                group_id=f"{target.scope}/{target.identity}",
                constituents=(Constituent(
                    identity=target.identity, semantic=source_value,
                    coefficient=strength, lifetime_logit=lifetime,
                    admitted_at=anchor, version="source-writer-frozen-space",
                ),),
                anchor=anchor, revision="post-pretraining-writer-v1",
                compatible_group=target.scope,
            )
            state = admit_kb(state, group)
            for event in target.events:
                if event.meridian != int(state.meridians_seen[0]):
                    raise RuntimeError("Native writer replay altered the committed clock")
                if event.raw is None:
                    step = generator.supervised_meridian(
                        state, tracks=event.tracks, plans=event.plans,
                        control_payloads=event.controls or (
                            (b"",) * len(event.tracks)
                        ), effort=event.effort,
                        teacher_forced_action=event.teacher_forced_action,
                        memory_dropout=(0., 0.),
                    )
                    total = total + step.loss
                    state = step.state
                    count += len(event.tracks)
                else:
                    raw = event.raw.detach().to(state.device)
                    if raw.shape[0:2] != (1, len(event.tracks)):
                        raise ValueError("Frozen observed raw group has incorrect event count")
                    state = generator.tracks.advance_group(
                        state, raw, tracks=event.tracks,
                        role_ids=torch.tensor(
                            [event.roles], device=state.device, dtype=torch.long,
                        ),
                    )
        return total, count

    return evaluate
