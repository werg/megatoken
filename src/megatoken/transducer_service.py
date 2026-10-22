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
from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from megatoken.data.knowledge import FilePreparer, PreparedFile, knowledge_items
from megatoken.kb import add_items, model_identity, new_base, open_base
from megatoken.memory.knowledge import KnowledgeBase
from megatoken.memory.transducer import Lookup, SearchAgenda
from megatoken.model.config import RESERVED_TRACKS
from megatoken.model.multitrack import MultiTrackGenerator, MultiTrackState, ObservedEvent
from megatoken.model.transducer import metadata_payload, real_boundary
from megatoken.retrieval.collection import SourceCollection
from megatoken.retrieval.ingest import FileIngest
from megatoken.retrieval.session import DEFAULT_MAX_SOURCES, SourceSession
from megatoken.training.transducer import BoundaryScheduler


@dataclass
class TransducerSession:
    session_id: str
    state: MultiTrackState
    agenda: SearchAgenda
    scheduler: BoundaryScheduler
    owner: int
    sources: SourceSession | None = None
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


@dataclass
class AddedFiles:
    """What ``add_files`` added: new collection items, and the entry ids written into a base."""

    collection: int = 0
    base: list[int] = field(default_factory=list)


class TransducerInferenceEngine:
    """Owns serving sessions for one multitrack generator and its transducer.

    ``collections`` maps names to source collection directories, mounted for the
    engine's life; a session searches the ones it names (default: all). Knowledge
    bases are opened or created by name (``open_base``, ``create_base``) and each
    session mounts the ones it names with blend weights. A ``preparer`` (called once,
    when first needed; ``megatoken.data.media_prepare.MediaPreparer``) lets the engine
    add files to its collections and writable bases; without one collections are
    mounted read-only.
    """

    def __init__(
        self,
        multi: MultiTrackGenerator,
        *,
        collections: Mapping[str, str | Path] | None = None,
        preparer: Callable[[], FilePreparer] | None = None,
    ) -> None:
        """``preparer`` (called once, when first needed) prepares files to add."""
        model = multi.generator.eval()
        self.native = model.transducer
        self.bank = self.native.bank
        if model.core.initial_facets.device != self.bank.device:
            raise ValueError("The v10 model and packed bank must be on one device")
        self.multi = multi
        self.bases: dict[str, KnowledgeBase] = {}
        self._preparer_factory = preparer
        self._preparer: FilePreparer | None = None
        self._identity: dict[str, Any] | None = None
        self.collections = {
            name: SourceCollection.open(
                path, writable=preparer is not None, teacher=model.config.media_extractor
            )
            for name, path in (collections or {}).items()
        }

    # ---------------------------------------------------------- knowledge bases

    @property
    def identity(self) -> dict[str, Any]:
        """What a base must share with this model (hashing the weights once, on first use)."""
        if self._identity is None:
            self._identity = model_identity(self.multi.generator)
        return self._identity

    def _register(self, name: str, base: KnowledgeBase) -> KnowledgeBase:
        if not name or name in self.bases:
            raise ValueError(
                f"A knowledge base named {name!r} is already open (or the name is empty)"
            )
        self.bases[name] = base
        return base

    def open_base(self, name: str, path: str | Path, *, writable: bool = False) -> KnowledgeBase:
        """Open a base file built for this model; a base from another model is refused."""
        return self._register(
            name, open_base(path, self.multi.generator, identity=self.identity, writable=writable)
        )

    def create_base(self, name: str) -> KnowledgeBase:
        """A new empty base that files can be added to while serving."""
        return self._register(
            name, new_base(self.multi.generator, name, identity=self.identity, writable=True)
        )

    def close_base(self, name: str) -> None:
        """Forget a base: no new session can mount it. Open sessions keep searching it."""
        del self.bases[name]

    @torch.inference_mode()
    def add_files(
        self,
        paths: Sequence[str | Path],
        *,
        collection: str | None = None,
        base: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> AddedFiles:
        """Prepare files once and add them to a source collection and/or a knowledge base.

        Each file is read by the teacher with the existing media preparation, and what that
        gives is shared by both destinations. Sessions that mount either find the new items
        from their next lookup; no weight changes. ``metadata`` files the base's entries
        (task, topic, language, date); a collection keeps only the original file.
        """
        if collection is None and base is None:
            raise ValueError("Adding files needs a collection, a knowledge base, or both")
        if collection is not None and collection not in self.collections:
            raise ValueError(f"No mounted collection named {collection!r}")
        if base is not None and base not in self.bases:
            raise ValueError(f"No open knowledge base named {base!r}")
        if self._preparer_factory is None:
            raise ValueError("Adding files needs the media preparer (serve with the pack manifest)")
        if base is not None and not self.bases[base].writable:
            raise RuntimeError(f"Knowledge base {base!r} is read-only")
        if self._preparer is None:
            self._preparer = self._preparer_factory()
        files = [PreparedFile(path, self._preparer) for path in paths]
        added = AddedFiles()
        if collection is not None:
            added.collection = FileIngest(self.collections[collection], self._preparer).add_files(
                files
            )
        if base is not None:
            target = self.bases[base]
            items = knowledge_items(files, metadata=metadata)
            with target.lock:
                added.base = add_items(target, self.multi.generator, self.multi, items)
        return added

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
        bases: Mapping[str, float] | None = None,
        trained: float = 1.0,
        effort: int = 6,
        deadline: float | None = None,
        cancelled: Callable[[], bool] | None = None,
        collections: Sequence[str] | None = None,
        max_sources: int = DEFAULT_MAX_SOURCES,
        threshold: float | None = None,
    ) -> TransducerSession:
        """Commit the metadata prefix, then the first lookup, before any answer.

        ``bases`` maps open base names to blend weights (0 leaves one out);
        ``trained`` is the trained base's weight. The session searches ``collections``
        (default: every mounted one) and observes up to ``max_sources`` items per
        lookup above ``threshold`` (default: each collection's own).
        """
        try:
            mounts = {name: (self.bases[name], weight) for name, weight in (bases or {}).items()}
        except KeyError as unknown:
            raise ValueError(f"No open knowledge base named {unknown}") from None
        state = self.multi.tracks.initial_state(1, device=self.bank.device)
        state, agenda = self.native.begin(
            self.multi,
            state,
            metadata_payload(metadata),
            mounted=self.bank.mount(mounts, trained=trained),
        )
        scheduler = BoundaryScheduler(max_lookups=self.bank.geometry.max_lookups)
        names = list(self.collections if collections is None else collections)
        if unknown := [n for n in names if n not in self.collections]:
            raise ValueError(f"No mounted collection named {unknown[0]!r}")
        sources = (
            SourceSession({n: self.collections[n] for n in names}, max_sources, threshold)
            if names
            else None
        )
        session = TransducerSession(uuid.uuid4().hex, state, agenda, scheduler, id(self), sources)
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
                # A base being filled reallocates its table: wait rather than read half a change.
                filling = {
                    id(m.base): m.base.lock for m in session.agenda.mounted if m.base.writable
                }
                with ExitStack() as locks:
                    for _, lock in sorted(filling.items()):
                        locks.enter_context(lock)
                    state, lookup = self.native.boundary(
                        self.multi,
                        state,
                        session.agenda,
                        effort=effort,
                        cancelled=stopped,
                        sources=session.sources,
                    )
        finally:
            session.scheduler.finish(committed=lookup is not None)
        return state, lookup
