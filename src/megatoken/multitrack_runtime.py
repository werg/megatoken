"""One canonical causal runtime for grouped observations and media generation.

A session owns its committed multitrack state and its transducer agenda: every
committed ordinary meridian may make a bank lookup due, served before the
next one. Ready tool results are observed in later meridians. Speculative branches are copy-on-write and can be rejected
without disturbing the committed session. Delivery is an irreversible ledger.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field

import torch

from megatoken.generation.structure import ActionSpec
from megatoken.model.media import MediaOutputSpec
from megatoken.model.multitrack import (
    GroupGeneration,
    MultiTrackGenerator,
    MultiTrackState,
    ObservedEvent,
)
from megatoken.transducer_service import TransducerInferenceEngine, TransducerSession


@dataclass(frozen=True)
class PendingGroup:
    base_revision: int
    result: GroupGeneration


@dataclass(frozen=True)
class Delivery:
    """Physical output delivery never rewinds committed semantic history."""

    event_id: str
    transmitted_bytes: int
    complete: bool


@dataclass
class TrackSession:
    """A single shared-history context, with no private per-agent transformers."""

    memory: TransducerSession
    revision: int = 0
    pending: PendingGroup | None = None
    delivered: dict[str, Delivery] = field(default_factory=dict)
    events: list[tuple[int, tuple[str, ...]]] = field(default_factory=list)

    @property
    def state(self) -> MultiTrackState:
        return self.memory.state


class MultiTrackRuntime:
    """Reference scheduler: one ready meridian at a time, no global track barrier."""

    def __init__(self, generator: MultiTrackGenerator) -> None:
        self.generator = generator
        self.memory = TransducerInferenceEngine(generator)

    def open(self, metadata: dict[str, object] | None = None) -> TrackSession:
        """Commit the metadata prefix and the first lookup, before any answer."""
        return TrackSession(self.memory.new_session(metadata or {}))

    def observe(self, session: TrackSession, events: Sequence[ObservedEvent]) -> MultiTrackState:
        """Only already available inputs may become shared causal observations."""
        ready = tuple(events)
        if not ready:
            raise ValueError("A committed meridian must contain one or more events")
        self.memory.observe(session.memory, ready)
        session.revision += 1
        session.pending = None
        session.events.append((session.revision, tuple(x.track for x in ready)))
        return session.state

    def speculate(
        self,
        session: TrackSession,
        *,
        tracks: Sequence[str],
        specs: Sequence[ActionSpec | MediaOutputSpec] | None = None,
        controls: Sequence[bytes] | None = None,
        effort: int = 6,
        generator: torch.Generator | None = None,
    ) -> PendingGroup:
        """Produce a candidate without committing raw state, KV or output."""
        result = self.generator.generate_group(
            session.state,
            tracks=tracks,
            specs=specs,
            controls=controls,
            effort=effort,
            generator=generator,
        )
        branch = PendingGroup(session.revision, result)
        session.pending = branch
        return branch

    def accept(self, session: TrackSession, branch: PendingGroup) -> GroupGeneration:
        """Atomic group acceptance; stale or foreign branches are never merged."""
        if session.pending is not branch or branch.base_revision != session.revision:
            raise ValueError("Speculation is stale or no longer the active candidate")
        result = branch.result
        if result.tape.meridian_id != int(session.state.meridians_seen[0]):
            raise ValueError("Speculative branch has the wrong causal history")
        self.memory.generated(session.memory, result.state)
        session.revision += 1
        session.pending = None
        session.events.append((session.revision, result.tape.tracks))
        return result

    def generate(
        self,
        session: TrackSession,
        *,
        tracks: Sequence[str],
        specs: Sequence[ActionSpec | MediaOutputSpec] | None = None,
        controls: Sequence[bytes] | None = None,
        effort: int = 6,
        generator: torch.Generator | None = None,
    ) -> GroupGeneration:
        return self.accept(
            session,
            self.speculate(
                session,
                tracks=tracks,
                specs=specs,
                controls=controls,
                effort=effort,
                generator=generator,
            ),
        )

    def cancel(self, session: TrackSession) -> None:
        session.pending = None

    @staticmethod
    def record_delivery(
        session: TrackSession,
        event_id: str,
        transmitted_bytes: int,
        *,
        complete: bool,
    ) -> None:
        if not event_id or transmitted_bytes < 0:
            raise ValueError("Delivered output requires identity and byte count")
        prior = session.delivered.get(event_id)
        if prior is not None:
            if transmitted_bytes < prior.transmitted_bytes:
                raise ValueError("External delivery cannot be rolled back")
            if prior.complete and not complete:
                raise ValueError("A completed delivery cannot become incomplete")
        session.delivered[event_id] = Delivery(event_id, transmitted_bytes, complete)


class ConcurrentToolRunner:
    """IO-only parallel calls; results enter KV only through observe()."""

    def __init__(self, max_workers: int = 4) -> None:
        if not 1 <= max_workers <= 32:
            raise ValueError("Tool runner requires 1 to 32 IO workers")
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.pending: dict[str, Future[bytes]] = {}

    def submit(self, action_id: str, function: Callable[[], bytes]) -> None:
        if not action_id or action_id in self.pending:
            raise ValueError("Parallel action requires a new stable action ID")
        self.pending[action_id] = self.executor.submit(function)

    def collect_ready(self) -> dict[str, bytes]:
        ready: dict[str, bytes] = {}
        for action_id, future in tuple(self.pending.items()):
            if not future.done():
                continue
            value = future.result()
            if not isinstance(value, bytes):
                raise TypeError("Executed tool result must return raw bytes")
            ready[action_id] = value
            del self.pending[action_id]
        return ready

    def close(self) -> None:
        self.executor.shutdown(wait=True)
