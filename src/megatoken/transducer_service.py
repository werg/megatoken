"""Serving sessions for the v10 transducer: one agenda per session, native C history.

No teacher, v9 point/pair KB or training hindsight exists on this path. A
query/knowledge pair is one cancellable transaction: on timeout or
cancellation neither the agenda nor the native state changes. Only ordinary
meridians (observed or generated) schedule lookups; every call on a session
is serialized by that session's lock.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from megatoken.memory.transducer import Lookup, SearchAgenda
from megatoken.model.config import RESERVED_TRACKS
from megatoken.model.multitrack import MultiTrackGenerator, MultiTrackState, ObservedEvent
from megatoken.model.transducer import metadata_payload
from megatoken.model.transducer_state import restore_causal_state, snapshot_causal_state
from megatoken.training.transducer import BoundaryScheduler

_SESSION_IDENTITY = "v10-serving-session"


@dataclass
class TransducerSession:
    session_id: str
    state: MultiTrackState
    agenda: SearchAgenda
    scheduler: BoundaryScheduler
    owner: int
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


class TransducerInferenceEngine:
    """Owns serving sessions for one generator with an attached transducer."""

    def __init__(self, model: Any) -> None:
        if model.transducer is None:
            raise ValueError("This model has no v10 transducer")
        self.model = model.eval()
        self.native = model.transducer
        self.bank = self.native.bank
        if model.core.initial_facets.device != self.bank.device:
            raise ValueError("The v10 model and packed bank must be on one device")
        self.multi = MultiTrackGenerator(model)

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
                self._run(session, deadline, cancelled)
        return session

    @torch.inference_mode()
    def observe(
        self,
        session: TransducerSession,
        events: Sequence[ObservedEvent],
        *,
        real_boundary: bool = False,
        deadline: float | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> Lookup | None:
        """Commit one observed ordinary meridian, then any lookup it makes due."""
        if not events or any(event.track in RESERVED_TRACKS for event in events):
            raise ValueError("Only nonempty ordinary meridians may be observed")
        with session.lock:
            self._check(session)
            session.state = self.multi.tracks.observe_group(session.state, events)
            return self._after_ordinary(session, real_boundary, deadline, cancelled)

    @torch.inference_mode()
    def generated(
        self,
        session: TransducerSession,
        state: MultiTrackState,
        *,
        completed_turn: bool = False,
        deadline: float | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> Lookup | None:
        """Adopt the state after one generated ordinary meridian; it counts like any other."""
        with session.lock:
            self._check(session)
            if int(state.meridians_seen[0]) != int(session.state.meridians_seen[0]) + 1:
                raise ValueError("A generated meridian must extend the session by exactly one")
            if any(track in RESERVED_TRACKS for track in state.frontier_tracks):
                raise ValueError("Internal events are not generated ordinary meridians")
            session.state = state
            return self._after_ordinary(session, completed_turn, deadline, cancelled)

    def _after_ordinary(self, session, real_boundary, deadline, cancelled) -> Lookup | None:
        if not session.scheduler.ordinary_commit(real_boundary=real_boundary):
            return None
        return self._run(session, deadline, cancelled)

    def _run(self, session, deadline, cancelled) -> Lookup | None:
        """Serve the outstanding lookup, then a deferred one if a boundary arrived meanwhile."""

        def stopped() -> bool:
            return (cancelled is not None and cancelled()) or (
                deadline is not None and time.monotonic() >= deadline
            )

        result = None
        while True:
            try:
                state, lookup = (
                    (session.state, None)
                    if stopped()
                    else self.native.boundary(
                        self.multi, session.state, session.agenda, cancelled=stopped
                    )
                )
            except BaseException:
                session.scheduler.finish(committed=False)
                raise
            if lookup is not None:
                session.state = state
                result = lookup
            if not session.scheduler.finish(committed=lookup is not None):
                return result

    def state_snapshot(self, session: TransducerSession) -> dict:
        """Tensor-only session state, independent of Python object identity."""
        with session.lock:
            self._check(session)
            return {
                "identity": _SESSION_IDENTITY,
                "session_id": session.session_id,
                "bank_id": self.bank.bank_id,
                "bank_version": self.bank.version,
                "scheduler": session.scheduler.state_dict(),
                "agenda_locations": [x.detach().cpu().clone() for x in session.agenda.locations],
                "agenda_state": session.agenda.state.detach().cpu().clone(),
                "agenda_lookups": session.agenda.lookups,
                "causal_state": snapshot_causal_state(session.state),
            }

    def restore_session(self, snapshot: dict) -> TransducerSession:
        if snapshot.get("identity") != _SESSION_IDENTITY:
            raise ValueError("Refusing an incompatible transducer session")
        if snapshot["bank_id"] != self.bank.bank_id:
            raise ValueError("Session refers to a different canonical v10 bank")
        state = restore_causal_state(self.multi.tracks, snapshot["causal_state"])
        scheduler = BoundaryScheduler(max_lookups=self.bank.geometry.max_lookups)
        scheduler.load_state_dict(snapshot["scheduler"])
        if not snapshot["agenda_locations"]:
            raise ValueError("Incomplete transducer session snapshot")
        agenda = self.bank.agenda_from(
            snapshot["agenda_locations"],
            snapshot["agenda_state"],
            int(snapshot["agenda_lookups"]),
        )
        restored = TransducerSession(
            str(snapshot["session_id"]), state, agenda, scheduler, id(self)
        )
        self._check(restored)
        return restored
