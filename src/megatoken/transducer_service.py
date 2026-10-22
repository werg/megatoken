"""Serving sessions for the v10 transducer: one agenda per session, native C history.

No teacher or training hindsight exists on this path. A
query/knowledge pair is one cancellable transaction: on timeout or
cancellation neither the agenda nor the native state changes. Only ordinary
meridians (observed or generated) schedule lookups, and the roles of a
meridian decide whether it is a real boundary, exactly as in training. A
meridian and the lookup it makes due commit together: an error leaves the
session as it was. Every call on a session is serialized by that session's lock.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import torch

from megatoken.memory.transducer import Lookup, SearchAgenda
from megatoken.model.config import RESERVED_TRACKS
from megatoken.model.multitrack import MultiTrackGenerator, MultiTrackState, ObservedEvent
from megatoken.model.transducer import metadata_payload, real_boundary
from megatoken.training.transducer import BoundaryScheduler


@dataclass
class TransducerSession:
    session_id: str
    state: MultiTrackState
    agenda: SearchAgenda
    scheduler: BoundaryScheduler
    owner: int
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


class TransducerInferenceEngine:
    """Owns serving sessions for one multitrack generator and its transducer."""

    def __init__(self, multi: MultiTrackGenerator) -> None:
        model = multi.generator.eval()
        self.native = model.transducer
        self.bank = self.native.bank
        if model.core.initial_facets.device != self.bank.device:
            raise ValueError("The v10 model and packed bank must be on one device")
        self.multi = multi

    def _check(self, session: TransducerSession) -> None:
        if session.owner != id(self):
            raise ValueError("Transducer session belongs to a different serving owner")
        if session.scheduler.lookups != session.agenda.lookups:
            raise ValueError("Transducer scheduler and search agenda cursors diverged")
        self.multi.tracks._validate(session.state)

    @torch.inference_mode()
    def new_session(
        self,
        metadata: dict[str, object],
        *,
        effort: int = 6,
        deadline: float | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> TransducerSession:
        """Commit the metadata prefix, then the first lookup, before any answer."""
        state = self.multi.tracks.initial_state(1, device=self.bank.device)
        state, agenda = self.native.begin(self.multi, state, metadata_payload(metadata))
        scheduler = BoundaryScheduler(max_lookups=self.bank.geometry.max_lookups)
        session = TransducerSession(uuid.uuid4().hex, state, agenda, scheduler, id(self))
        if scheduler.first_metadata():
            with session.lock:
                session.state, _ = self._run(session, state, effort, deadline, cancelled)
        return session

    @torch.inference_mode()
    def observe(
        self,
        session: TransducerSession,
        events: Sequence[ObservedEvent],
        *,
        effort: int = 6,
        deadline: float | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> Lookup | None:
        """Commit one observed ordinary meridian, then any lookup it makes due."""
        if not events or any(event.track in RESERVED_TRACKS for event in events):
            raise ValueError("Only nonempty ordinary meridians may be observed")
        with session.lock:
            self._check(session)
            state = self.multi.tracks.observe_group(session.state, events)
            boundary = real_boundary([e.role_id for e in events], [e.terminal for e in events])
            return self._commit(session, state, boundary, effort, deadline, cancelled)

    @torch.inference_mode()
    def generated(
        self,
        session: TransducerSession,
        state: MultiTrackState,
        terminal: Sequence[bool],
        *,
        effort: int = 6,
        deadline: float | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> Lookup | None:
        """Adopt the state after one generated ordinary meridian; it counts like any other.

        ``terminal`` says which of its events end their message (its tape's flags).
        """
        with session.lock:
            self._check(session)
            if int(state.meridians_seen[0]) != int(session.state.meridians_seen[0]) + 1:
                raise ValueError("A generated meridian must extend the session by exactly one")
            if any(track in RESERVED_TRACKS for track in state.frontier_tracks):
                raise ValueError("Internal events are not generated ordinary meridians")
            boundary = real_boundary(state.frontier_roles[0].tolist(), terminal)
            return self._commit(session, state, boundary, effort, deadline, cancelled)

    def _commit(self, session, state, boundary, effort, deadline, cancelled) -> Lookup | None:
        """Adopt ``state`` with the lookup it makes due; a failure leaves the session as it was."""
        scheduler = session.scheduler
        waited = scheduler.ordinary_since
        lookup = None
        try:
            if scheduler.ordinary_commit(real_boundary=boundary):
                state, lookup = self._run(session, state, effort, deadline, cancelled)
        except BaseException:
            scheduler.ordinary_since = waited
            raise
        session.state = state
        return lookup

    def _run(self, session, state, effort, deadline, cancelled):
        """Serve one due lookup; a cancelled or failed one is closed without committing."""

        def stopped() -> bool:
            return (cancelled is not None and cancelled()) or (
                deadline is not None and time.monotonic() >= deadline
            )

        lookup = None
        try:
            if not stopped():
                state, lookup = self.native.boundary(
                    self.multi, state, session.agenda, effort=effort, cancelled=stopped
                )
        finally:
            session.scheduler.finish(committed=lookup is not None)
        return state, lookup
