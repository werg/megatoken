"""Serving adapter for coherent v10 transducer agendas and native C history.

No teacher, old point/pair KB, or training hindsight is present on this path.
A query/knowledge pair is one cancelable transaction; only ordinary externally
observed meridians can schedule new retrievals.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

from megatoken.memory.transducer import Lookup, SearchAgenda
from megatoken.model.config import HINDSIGHT_TRACK, KNOWLEDGE_TRACK, RETRIEVAL_QUERY_TRACK
from megatoken.model.multitrack import MultiTrackState, ObservedEvent
from megatoken.model.transducer import NativeTransducer
from megatoken.model.transducer_state import restore_causal_state, snapshot_causal_state
from megatoken.training.transducer import BoundaryScheduler

_INTERNAL_TRACKS = frozenset((HINDSIGHT_TRACK, KNOWLEDGE_TRACK, RETRIEVAL_QUERY_TRACK))
_SESSION_IDENTITY = "v10-serving-session"


@dataclass
class TransducerSession:
    session_id: str
    state: MultiTrackState
    agenda: SearchAgenda
    scheduler: BoundaryScheduler
    model_id: int


class TransducerInferenceEngine:
    """An independently instantiated v10 native serving session owner."""

    def __init__(self, native: NativeTransducer) -> None:
        if native.core.initial_facets.device != native.bank.device:
            raise ValueError("The v10 model and packed bank must be on one device")
        self.native = native.eval()
        self.bank = native.bank

    @torch.inference_mode()
    def new_session(self, metadata: dict[str, object]) -> TransducerSession:
        state, agenda, _ = self.native.start(metadata)
        scheduler = BoundaryScheduler(max_lookups=self.bank.geometry.max_lookups)
        if not scheduler.first_metadata():
            raise RuntimeError("First metadata lookup was unexpectedly blocked")
        scheduler.internal_commit()
        scheduler.finish(committed=True)
        return TransducerSession(uuid.uuid4().hex, state, agenda, scheduler, id(self))

    def _check(self, session: TransducerSession) -> None:
        if session.model_id != id(self):
            raise ValueError("Transducer session belongs to a different serving owner")
        if session.scheduler.lookups != session.agenda.lookups:
            raise ValueError("Transducer scheduler and search agenda cursors diverged")
        self.native.tracks._validate(session.state)

    @torch.inference_mode()
    def observe(
        self,
        session: TransducerSession,
        events: Sequence[ObservedEvent],
        *,
        completed_turn: bool = False,
        source_arrived: bool = False,
        deadline: float | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> Lookup | None:
        """Commit one ordinary meridian, then perform any due lookup."""
        self._check(session)
        if not events or any(event.track in _INTERNAL_TRACKS for event in events):
            raise ValueError("Only external, nonempty observed meridians may schedule retrieval")
        committed = self.native.tracks.observe_group(session.state, events)
        session.state = committed
        if not session.scheduler.ordinary_commit(
            completed_turn=completed_turn, source_arrived=source_arrived
        ):
            return None
        return self.boundary(session, deadline=deadline, cancelled=cancelled)

    @torch.inference_mode()
    def boundary(
        self,
        session: TransducerSession,
        *,
        deadline: float | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> Lookup | None:
        """Timeout/cancellation leaves the agenda and native state unpublished."""
        self._check(session)
        if not session.scheduler.pending and not session.scheduler.begin():
            return None

        def stopped() -> bool:
            return (cancelled is not None and cancelled()) or (
                deadline is not None and time.monotonic() >= deadline
            )

        try:
            if stopped():
                result_state, lookup = session.state, None
            else:
                result_state, lookup = self.native.boundary(
                    session.state, session.agenda, cancelled=stopped
                )
        except BaseException:
            session.scheduler.finish(committed=False)
            raise
        if lookup is None:
            session.scheduler.finish(committed=False)
            return None
        session.state = result_state
        session.scheduler.internal_commit()
        session.scheduler.finish(committed=True)
        self._check(session)
        return lookup

    def state_snapshot(self, session: TransducerSession) -> dict:
        """Tensor-only session state, independent of local Python object identity."""
        self._check(session)
        if session.scheduler.pending:
            raise RuntimeError("Cannot checkpoint a half-completed retrieval boundary")
        return {
            "identity": _SESSION_IDENTITY,
            "session_id": session.session_id,
            "bank_id": self.bank.bank_id,
            "bank_version": self.bank.version,
            "scheduler": session.scheduler.state_dict(),
            "agenda_locations": [s.detach().cpu().clone() for s in session.agenda.locations],
            "agenda_lookups": session.agenda.lookups,
            "causal_state": snapshot_causal_state(session.state),
        }

    def restore_session(self, snapshot: dict) -> TransducerSession:
        if snapshot.get("identity") != _SESSION_IDENTITY:
            raise ValueError("Refusing an incompatible transducer session")
        if snapshot["bank_id"] != self.bank.bank_id:
            raise ValueError("Session refers to a different canonical v10 bank")
        if snapshot["bank_version"] > self.bank.version:
            raise ValueError("Cannot restore a session from a future bank snapshot")
        state = restore_causal_state(self.native.tracks, snapshot["causal_state"])
        scheduler = BoundaryScheduler(max_lookups=self.bank.geometry.max_lookups)
        scheduler.load_state_dict(snapshot["scheduler"])
        agenda = SearchAgenda(
            [x.to(self.bank.device) for x in snapshot["agenda_locations"]],
            lookups=int(snapshot["agenda_lookups"]),
        )
        if not agenda.locations or scheduler.lookups != agenda.lookups:
            raise ValueError("Incomplete transducer session snapshot")
        self.bank.refresh(agenda)
        restored = TransducerSession(
            str(snapshot["session_id"]), state, agenda, scheduler, id(self)
        )
        self._check(restored)
        return restored
