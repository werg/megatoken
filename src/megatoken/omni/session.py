"""In-memory atomic group delivery, speculation and replay on v8 raw facets.

This is a thin stateful server-side adapter around the existing grouped
predictor/realizers, NOT a second scheduler, semantic model or transport token.
One selected group is published to C once; unaccepted candidates never become
history. Delivery cursors record physical bytes, not central megatokens.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from typing import TYPE_CHECKING, Any

import torch

from megatoken.controls import effective_event_payloads

if TYPE_CHECKING:
    from megatoken.model.multitrack import (
        GroupActionTape, GroupGeneration, MultiTrackGenerator, MultiTrackState,
        ObservedEvent,
    )


@dataclass(frozen=True)
class PendingGroup:
    epoch: int
    generation: GroupGeneration
    specs: tuple[Any, ...]
    candidate_id: str


@dataclass
class PublishedGroup:
    tape: GroupActionTape
    specs: tuple[Any, ...]
    delivery_cursors: dict[str, int] = field(default_factory=dict)
    interruptions: list[tuple[str, str, int]] = field(default_factory=list)


class OmniSession:
    """One session with copy-on-write candidates, committed inputs and outputs."""

    def __init__(self, multi: MultiTrackGenerator, initial: MultiTrackState):
        multi.tracks._validate(initial)
        if initial.batch_size != 1:
            raise ValueError("OmniSession requires one independent causal episode")
        self.multi = multi
        self.initial = initial
        self.state = initial
        self.pending: PendingGroup | None = None
        self.history: list[tuple[str, Any]] = []
        self.epoch = 0
        self.published: dict[int, PublishedGroup] = {}

    def invalidate(self) -> None:
        """Reject only unaccepted work, never erase published output."""
        self.pending = None
        self.epoch += 1

    @torch.no_grad()
    def observe(self, events: tuple[ObservedEvent, ...]) -> MultiTrackState:
        """New observed evidence invalidates speculative P/R work."""
        if not events:
            raise ValueError("Observation must contain real causal events")
        self.invalidate()
        self.state = self.multi.tracks.observe_group(
            self.state, events, encoder_grad=False,
        )
        self.history.append(("observation", tuple(events)))
        return self.state

    @torch.no_grad()
    def propose(
        self, *, tracks: tuple[str, ...], specs: tuple[Any, ...],
        session_payload: bytes = b"",
        requests: tuple[tuple[str, int], ...],
        effort: int = 6,
        generator=None,
    ) -> PendingGroup:
        """Prepare whole group on current immutable state; no public commit yet."""
        if self.pending is not None:
            raise ValueError("Reject or accept existing candidate before proposing another")
        controls = effective_event_payloads(session_payload, tracks, requests)
        result = self.multi.generate_group(
            self.state, tracks=tracks, specs=specs, controls=controls,
            effort=effort, generator=generator,
        )
        digest = sha256(result.tape.to_json().encode("utf-8")).hexdigest()
        candidate = PendingGroup(self.epoch, result, tuple(specs), digest)
        self.pending = candidate
        return candidate

    def reject(self, candidate_id: str) -> None:
        """Drop a speculative complete group with no changes to shared history."""
        self._candidate(candidate_id)
        self.invalidate()

    def _candidate(self, candidate_id: str) -> PendingGroup:
        if (self.pending is None or self.pending.epoch != self.epoch
                or self.pending.candidate_id != candidate_id):
            raise ValueError("Speculative group is stale, rejected or superseded")
        return self.pending

    def accept(self, candidate_id: str) -> PublishedGroup:
        """Publish all peers to future C history atomically, exactly once."""
        candidate = self._candidate(candidate_id)
        tape = candidate.generation.tape
        mid = tape.meridian_id
        if mid in self.published:
            raise ValueError("The meridian has already been accepted")
        published = PublishedGroup(
            tape, candidate.specs, {name: 0 for name in tape.tracks},
        )
        self.state = candidate.generation.state
        self.published[mid] = published
        self.history.append(("selection", published))
        self.pending = None
        self.epoch += 1
        return published

    def deliver(self, meridian_id: int, track: str, byte_count: int) -> int:
        """Advance an actual transport cursor, not an additional semantic event."""
        published = self.published[meridian_id]
        if track not in published.delivery_cursors:
            raise ValueError("Unknown track in published group")
        if type(byte_count) is not int or byte_count < 0:
            raise ValueError("Delivery progress must be nonnegative bytes")
        cursor = published.delivery_cursors[track] + byte_count
        index = published.tape.tracks.index(track)
        if cursor > len(published.tape.payloads[index]):
            raise ValueError("Transport cursor exceeds immutable rendered artifact")
        published.delivery_cursors[track] = cursor
        return cursor

    def interrupt(self, meridian_id: int, track: str, reason: str) -> None:
        """Physical output already delivered cannot be erased by speculation."""
        published = self.published[meridian_id]
        if not reason or track not in published.delivery_cursors:
            raise ValueError("Interruption needs a delivered track and reason")
        published.interruptions.append(
            (track, reason, published.delivery_cursors[track]),
        )

    @torch.no_grad()
    def rebuild(self) -> MultiTrackState:
        """Rebuild history from the same initial state, observed inputs and raw tapes."""
        state = self.initial
        for kind, item in self.history:
            if kind == "observation":
                state = self.multi.tracks.observe_group(
                    state, item, encoder_grad=False,
                )
            else:
                replayed = self.multi.replay_group(
                    state, item.tape, specs=item.specs, effort=item.tape.effort,
                )
                state = replayed.state
        return state
