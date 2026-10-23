"""Serving sessions for the v10 transducer: one agenda per session, native C history.

Scoped overlays remember completed conversation evidence without retraining. A
query/knowledge pair is one cancellable transaction: on timeout or
cancellation neither the agenda nor the native state changes. Only ordinary
meridians (observed or generated) schedule lookups, and the roles of a
meridian decide whether it is a real boundary, exactly as in training. A
meridian and the lookup it makes due commit together: an error leaves the
session as it was. Every call on a session is serialized by that session's lock.
"""

from __future__ import annotations

import copy
import hashlib
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, field, replace
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
from megatoken.serving_writes import ConversationWrites, WritePolicy
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
    writes: ConversationWrites | None = None
    closed: bool = False


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
        memory_root: str | Path | None = None,
    ) -> None:
        """``preparer`` (called once, when first needed) prepares files to add."""
        model = multi.generator.eval()
        self.native = model.transducer
        self.bank = self.native.bank
        if model.core.initial_facets.device != self.bank.device:
            raise ValueError("The v10 model and packed bank must be on one device")
        self.multi = multi
        self.bases: dict[str, KnowledgeBase] = {}
        self.sessions: dict[str, TransducerSession] = {}
        self.memory_root = None if memory_root is None else Path(memory_root).resolve()
        self._scope_lock = threading.RLock()
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

    def _scope_in_use(self, scope: str) -> bool:
        return any(b.writable and b.scope == scope for b in self.bases.values()) or any(
            s.writes is not None and s.writes.scope == scope for s in self.sessions.values()
        )

    def _register(self, name: str, base: KnowledgeBase) -> KnowledgeBase:
        if not name or name in self.bases:
            raise ValueError(
                f"A knowledge base named {name!r} is already open (or the name is empty)"
            )
        if base.writable and base.scope is not None and self._scope_in_use(base.scope):
            raise ValueError("This scope already has an open overlay or an active writing session")
        self.bases[name] = base
        return base

    def open_base(self, name: str, path: str | Path, *, writable: bool = False) -> KnowledgeBase:
        """Open a base file built for this model; a base from another model is refused."""
        return self._register(
            name, open_base(path, self.multi.generator, identity=self.identity, writable=writable)
        )

    @torch.inference_mode(False)
    def scope_overlay(self, scope: str) -> str:
        """Reuse, reopen or create the persistent private store for an authenticated scope."""
        with self._scope_lock:
            for name, base in self.bases.items():
                if base.writable and base.scope == scope:
                    return name
            # An unmounted base may still be used by an active session. Reuse its
            # lock and rows instead of opening a second independent disk writer.
            for session in self.sessions.values():
                if session.writes is not None and session.writes.scope == scope:
                    base = session.writes.base
                    if base.name in self.bases:
                        raise ValueError("The scope's active memory name is already occupied")
                    self.bases[base.name] = base
                    return base.name
            if self.memory_root is None:
                raise ValueError(
                    "Scoped memory needs a persistent memory_root (serve with --kb-root), "
                    "or an existing writable scope store; set memory_writes=False to disable"
                )
            digest = hashlib.sha256(scope.encode("utf-8")).hexdigest()
            name = f"scope-{digest}"
            path = (self.memory_root / "scopes" / f"{digest}.kb").resolve()
            if not path.is_relative_to(self.memory_root):
                raise ValueError("Scope memory path lies outside the memory root")
            if path.exists():
                base = open_base(path, self.multi.generator, identity=self.identity, writable=True)
                if base.scope != scope:
                    raise ValueError("Stored memory belongs to a different scope")
                base.name = name
                self._register(name, base)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                count = min(2048, self.bank.geometry.max_edges)
                self.create_overlay(
                    name,
                    scope,
                    path,
                    max_edges=count,
                    byte_budget=min(
                        self.bank.geometry.byte_budget, count * self.bank.geometry.row_bytes
                    ),
                )
            return name

    def create_base(self, name: str) -> KnowledgeBase:
        """A new empty base that files can be added to while serving."""
        return self._register(
            name, new_base(self.multi.generator, name, identity=self.identity, writable=True)
        )

    def close_base(self, name: str) -> None:
        """Forget a base: no new session can mount it. Open sessions keep searching it."""
        base = self.bases[name]
        if base.writable and base.path is not None:
            base.save(base.path)
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
            added.base = add_items(target, self.multi.generator, self.multi, items)
        return added

    def _check(self, session: TransducerSession) -> None:
        if session.closed:
            raise ValueError("Session is closed")
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
        overlay: str | None = None,
        scope: str | None = None,
        write_policy: WritePolicy | None = None,
        memory_writes: bool = True,
    ) -> TransducerSession:
        """Commit the metadata prefix, then the first lookup, before any answer.

        ``bases`` maps open base names to blend weights (0 leaves one out);
        ``trained`` is the trained base's weight. The session searches ``collections``
        (default: every mounted one) and observes up to ``max_sources`` items per
        lookup above ``threshold`` (default: each collection's own).
        """
        if scope is not None and (not isinstance(scope, str) or not scope):
            raise ValueError("Scope must be a nonempty string")
        if type(memory_writes) is not bool:
            raise ValueError("memory_writes must be a boolean")
        if overlay is None and scope is not None and memory_writes:
            # Validate explicit mounts before creating any persistent state.
            for name in bases or {}:
                mounted = self.bases.get(name)
                if mounted is None:
                    raise ValueError(f"No open knowledge base named {name!r}")
                if mounted.scope is not None and mounted.scope != scope:
                    raise ValueError("Private bases can only mount in their own scope")
            overlay = self.scope_overlay(scope)
        if overlay is not None:
            base = self.bases.get(overlay)
            if base is None or not base.writable or not scope or base.scope != scope:
                raise ValueError("Writing requires a writable overlay belonging to this scope")
            if base.path is None:
                raise ValueError("Writing requires a persistent overlay path")
            bases = {**(bases or {}), overlay: 1.0}
        try:
            mounts = {name: (self.bases[name], weight) for name, weight in (bases or {}).items()}
        except KeyError as unknown:
            raise ValueError(f"No open knowledge base named {unknown}") from None
        if any(m.scope is not None and m.scope != scope for m, _ in mounts.values()):
            raise ValueError("Private bases can only mount in their own scope")
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
        if overlay is not None and memory_writes:
            session.writes = ConversationWrites(
                scope, base, metadata_payload(metadata), write_policy or WritePolicy()
            )
        if scheduler.first_metadata():
            with session.lock:
                session.state, initial = self._run(session, state, effort, deadline, cancelled)
                if session.writes is not None:
                    session.writes.remember_lookup(initial, session.agenda.lookups - 1)
        self.sessions[session.session_id] = session
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
            scores = []
            if session.writes is not None and session.writes.policy.surprise:
                scores = [
                    (i, session.writes.meter.score(self.multi, session.state, e, effort=effort))
                    for i, e in enumerate(events)
                ]
            state = self.multi.tracks.observe_group(session.state, events)
            boundary = real_boundary([e.role_id for e in events], [e.terminal for e in events])
            return self._commit(session, state, boundary, effort, deadline, cancelled, scores)

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

    def _commit(
        self, session, state, boundary, effort, deadline, cancelled, scores=()
    ) -> Lookup | None:
        """Publish accepted history, committed lookup, and memory writes together."""
        scheduler = session.scheduler
        previous = dict(vars(scheduler))
        agenda = session.agenda.copy()
        ordinary = state
        lookup = None
        source_snapshot = self._source_snapshot(session)
        try:
            if scheduler.ordinary_commit(real_boundary=boundary):
                state, lookup = self._run(session, state, effort, deadline, cancelled)
            if session.writes is not None:
                session.writes.commit(session, self.native, self.multi, ordinary, lookup, scores)
        except BaseException:
            vars(scheduler).update(previous)
            session.agenda.publish(agenda)
            self._restore_sources(session, source_snapshot)
            raise
        session.state = state
        return lookup

    @staticmethod
    def _source_snapshot(session):
        if session.sources is None:
            return None
        sources = session.sources
        return (
            list(sources.log),
            {n: set(v) for n, v in sources.observed.items()},
            sources.last,
            copy.deepcopy(sources.search_snapshots),
        )

    @staticmethod
    def _restore_sources(session, snapshot):
        if snapshot is not None:
            sources = session.sources
            sources.log[:], sources.observed, sources.last, sources.search_snapshots = snapshot

    @torch.inference_mode()
    def generated_completion(
        self,
        session: TransducerSession,
        roles: Sequence[int],
        terminal: Sequence[bool],
        *,
        effort: int = 6,
    ) -> Lookup | None:
        """Finalize a rendered turn without counting its selection a second time.

        Incoming observations may have advanced shared history while rendering.
        A completion lookup reads that current history, never a stale render state.
        """
        with session.lock:
            self._check(session)
            previous = dict(vars(session.scheduler))
            if (
                real_boundary(roles, terminal)
                and session.scheduler.ordinary_since
                and session.scheduler.first_metadata()
            ):
                agenda = session.agenda.copy()
                source_snapshot = self._source_snapshot(session)
                try:
                    state, lookup = self._run(session, session.state, effort, None, None)
                    if session.writes is not None:
                        session.writes.commit(session, self.native, self.multi, lookup=lookup)
                except BaseException:
                    vars(session.scheduler).update(previous)
                    session.agenda.publish(agenda)
                    self._restore_sources(session, source_snapshot)
                    raise
                session.state = state
                return lookup
        return None

    def create_overlay(
        self,
        name: str,
        scope: str,
        path: str | Path,
        *,
        max_edges: int = 2048,
        byte_budget: int | None = None,
    ) -> KnowledgeBase:
        """Create one small persistent store for a caller-authenticated scope."""
        if not isinstance(scope, str) or not scope or self._scope_in_use(scope):
            raise ValueError("A scope needs one uniquely named overlay")
        if Path(path).exists():
            raise FileExistsError(path)
        geometry = replace(
            self.bank.geometry,
            max_edges=max_edges,
            byte_budget=(
                max_edges * self.bank.geometry.row_bytes if byte_budget is None else byte_budget
            ),
        )
        base = KnowledgeBase(
            geometry, self.identity, name=name, writable=True, device=self.bank.device
        )
        base.scope = scope
        if not name or name in self.bases:
            raise ValueError("Overlay name is empty or already open")
        base.save(path)
        return self._register(name, base)

    @torch.inference_mode()
    def flush(self, session: TransducerSession) -> None:
        with session.lock:
            self._check(session)
            if session.writes is not None:
                session.writes.commit(session, self.native, self.multi, flush=True)

    def close_session(self, session: TransducerSession) -> None:
        with session.lock:
            self._check(session)
            if session.writes is not None:
                session.writes.commit(session, self.native, self.multi, flush=True)
            session.closed = True
            if session.writes is not None:
                session.writes.events.clear()
                session.writes.pending.clear()
            self.sessions.pop(session.session_id, None)

    def close(self) -> None:
        for session in list(self.sessions.values()):
            self.close_session(session)
        for base in self.bases.values():
            if base.writable and base.path is not None:
                base.save(base.path)

    def forget(
        self, name: str, *, ids=None, session: str | None = None, scope: str | None = None
    ) -> list[int]:
        base = self.bases[name]
        if ids is None and session is None and scope is None:
            raise ValueError("Forget needs IDs, session, or scope")
        if base.path is None:
            raise ValueError("Forget requires a saved base")
        affected = sorted(
            (
                s
                for s in self.sessions.values()
                if s.writes is not None
                and s.writes.base is base
                and (session is None or s.session_id == session)
                and (scope is None or s.writes.scope == scope)
            ),
            key=lambda s: s.session_id,
        )
        with ExitStack() as locks:
            for active in affected:
                locks.enter_context(active.lock)
            with base.transaction():
                selected = set(base.edge_ids().tolist() if ids is None else map(int, ids))
                if session is not None:
                    selected &= {
                        e
                        for e, p in base.provenance.items()
                        if p.get("session") == session or session in p.get("sessions", [])
                    }
                if scope is not None:
                    selected &= {e for e, p in base.provenance.items() if p.get("scope") == scope}
                base.retire(sorted(selected))
                base.save(base.path)
            # Erase retained evidence too, so a later flush cannot recreate it.
            for active in affected:
                active.writes.events.clear()
                active.writes.pending.clear()
                active.writes.journal.clear()
        return sorted(selected)

    def promote(self, name: str, destination: str, ids: Sequence[int], *, reviewed: bool = False):
        if not reviewed:
            raise ValueError("Promotion requires explicit review")
        source, target = self.bases[name], self.bases[destination]
        ids = [int(edge) for edge in ids]
        if len(set(ids)) != len(ids):
            raise ValueError("Promote needs distinct edge IDs")
        if not ids:
            return []
        if source is target or not target.writable or target.scope is not None:
            raise ValueError("Promote into a separate writable shared base")
        from megatoken.memory.knowledge import check_identity

        check_identity(source.identity, target.identity)
        with source.lock:
            rows = source.table.state(list(ids))
            original_ids = rows["ids"].tolist()
            provenance = {e: copy.deepcopy(source.provenance.get(e, {})) for e in original_ids}
            labels = {e: source.sources[e] for e in original_ids if e in source.sources}
        # Copy packed columns exactly; assign fresh IDs in the destination.
        with target.transaction():
            if len(ids) > target.edge_room:
                raise ValueError("Promotion exceeds destination capacity")
            added = torch.arange(target.table.next_id, target.table.next_id + len(ids))
            rows["ids"] = added
            target.table.upsert(rows)
            target.table.next_id += len(ids)
            for old, new in zip(original_ids, added.tolist(), strict=True):
                target.provenance[new] = {
                    **provenance[old],
                    "promoted_from": name,
                    "promoted_edge": old,
                }
                if old in labels:
                    target.sources[new] = labels[old]
            target._changed()
            if target.path is not None:
                target.save(target.path)
        return added.tolist()

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
