"""One native retrieval boundary for literal source E/C and direct semantic KB.

The backend is a candidate locator. Native P creates the query event, R
realizes its selected raw facets, and source evidence is observed byte-exact
through the existing E/C runtime; semantic K/V is never an artificial event.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from hashlib import sha256

import numpy as np
import torch

from megatoken.memory.base import Record, SearchBatch
from megatoken.memory.bilateral import FrozenMemoryFields, selected_joint_groups
from megatoken.memory.context import admit_kb
from megatoken.memory.contracts import Collection, QueryBoundary, SemanticSpace, Visibility
from megatoken.memory.replay import QueryActionReceipt
from megatoken.memory.retrieval import (
    Interval,
    RetrievalService,
    RetrievalSession,
    SourceObligation,
    subtract,
)
from megatoken.memory.sqlite import SQLiteKB
from megatoken.model.core import ROLE_IDS
from megatoken.model.multitrack import MultiTrackGenerator, ObservedEvent
from megatoken.multitrack_runtime import MultiTrackRuntime, TrackSession


@dataclass(frozen=True)
class RetrievalPolicy:
    source_candidates: int = 32
    memory_candidates: int = 32
    source_threshold: float = 0.35
    memory_threshold: float = 0.25
    max_source_bytes: int = 16384
    max_source_events: int = 4
    max_compat_groups: int = 4
    deadline_meridians: int = 4

    def __post_init__(self):
        if min(self.source_candidates, self.memory_candidates, self.max_source_bytes,
               self.max_source_events, self.max_compat_groups,
               self.deadline_meridians) < 0:
            raise ValueError("Retrieval limits must be nonnegative")


@dataclass(frozen=True)
class BoundaryResult:
    query: QueryBoundary
    vector: np.ndarray
    source_candidates: SearchBatch
    memory_candidates: SearchBatch
    newly_qualified: tuple[SourceObligation, ...]
    admitted_groups: tuple[str, ...]
    source_shortfall: int
    memory_shortfall: int


class NativeRetrieval:
    def __init__(self, generator: MultiTrackGenerator, runtime: MultiTrackRuntime,
                 service: RetrievalService, space: SemanticSpace, policy: RetrievalPolicy):
        if generator.generator.embedding_realizer is None or (
            generator.generator.embedding_realizer.dimension != space.dimension
        ):
            raise ValueError("Model embedding R and KB learned semantic space disagree")
        if generator.generator.core.memory_projector is None:
            raise ValueError("Native K/V projection must be enabled for learned memory")
        if service.backend.store.space.fingerprint != space.fingerprint:
            raise ValueError("Retriever index and model query target use different spaces")
        self.generator, self.runtime = generator, runtime
        self.service, self.space, self.policy = service, space, policy
        self.sessions: dict[int, RetrievalSession] = {}

    def ledger(self, session: TrackSession) -> RetrievalSession:
        return self.sessions.setdefault(id(session), RetrievalSession())

    def _excluded_content(self, session: TrackSession, visibility: Visibility) -> set[int]:
        covered = self.ledger(session).injected
        records = self.service.backend.store.scan(Collection.SOURCE, visibility.scope)
        return {
            row.id for row in records
            if row.evidence_span is not None and not subtract(
                Interval(row.evidence_span.source_id, row.evidence_span.version,
                         row.evidence_span.start, row.evidence_span.end), covered,
            )
        }

    def boundary(self, session: TrackSession, *, request: bytes,
                 purpose: str, visibility: Visibility, request_id: str,
                 effort: int = 6, generator: torch.Generator | None = None,
                 enable_memory: bool = True) -> BoundaryResult:
        if not request_id or not isinstance(request, bytes):
            raise ValueError("A retrieval boundary requires an immutable request identity")
        clock = session.causal_clock
        native = int(session.state.meridians_seen.min())
        if visibility.committed_cursor > clock:
            raise ValueError("Future sources cannot be made available to a past query")
        vector, next_state, _raw = self.generator.retrieve_embedding(
            session.state, main_request=request, purpose=purpose,
            space_fingerprint=self.space.fingerprint, effort=effort,
            generator=generator,
        )
        # Stage native query and scheduling state until BOTH collections,
        # source qualification, and compatible KV admission succeed. A
        # backend failure must not consume the session's query meridian.
        scheduling = self.ledger(session).fork()
        query = vector.detach().float().cpu().numpy()[0]
        revisions = self.service.backend.store.revisions()
        query_model = f"model-v{self.generator.generator.core.model_version}"
        epoch = f"{revisions.scoring_identity(query_model, self.space)}:{revisions.policy}"
        receipt = QueryBoundary(
            event_id=request_id, meridian=clock, purpose=purpose,
            originating_track="__retrieval_query__",
            request_hash=sha256(request).hexdigest(),
            space_fingerprint=self.space.fingerprint, snapshot_id=epoch,
            deadline=clock + self.policy.deadline_meridians,
        )
        source, memory = self.service.search(
            query, visibility=visibility,
            source_limit=self.policy.source_candidates,
            memory_limit=self.policy.memory_candidates,
            excluded_source=self._excluded_content(session, visibility),
        )
        if source.revisions != memory.revisions:
            raise RuntimeError("Source and semantic candidates came from incompatible KB views")
        queued = scheduling.incremental(
            list(source.hits), boundary=receipt,
            threshold=self.policy.source_threshold,
        )
        groups = ()
        if enable_memory and memory.hits and self.policy.max_compat_groups:
            store = self.service.backend.store
            if not isinstance(store, SQLiteKB):
                raise TypeError("Point/pair serving requires the authoritative SQLite KB")
            chosen = [
                hit.record for hit in memory.hits
                if hit.similarity >= self.policy.memory_threshold
            ]
            if chosen:
                # Only sources already committed through real E/C admissions
                # may supply the independent second leg of a relation. ANN hits,
                # future observations and source coverage reservations do not.
                covered = {
                    (item.content_id, item.version)
                    for item in self.ledger(session).injected
                }
                observed_sources = sorted({
                    span.source_id
                    for row in store.scan(Collection.SOURCE, visibility.scope)
                    if row.active
                    and row.evidence_span is not None
                    and (row.evidence_span.source_id, row.evidence_span.version) in covered
                    for span in row.footprint.ranges
                })
                device = session.state.device
                with torch.no_grad():
                    groups, _ = selected_joint_groups(
                        bank=store, selected=FrozenMemoryFields(store, device),
                        query=torch.from_numpy(query).to(device)[None],
                        records=chosen, visibility=visibility,
                        anchor=int(next_state.meridians_seen.min()),
                        threshold=self.policy.memory_threshold,
                        max_groups=self.policy.max_compat_groups,
                        pair_support_units=tuple(observed_sources),
                        pair_context_complete=False,
                    )
                for group in groups:
                    next_state = admit_kb(next_state, group)
        # Complete all serialization/validation work BEFORE session commit.
        # Replay is positional: the tape pins the NATIVE meridian it starts from.
        tape = QueryActionReceipt.capture(
            event_id=request_id, meridian=native,
            model_version=self.generator.generator.core.model_version,
            request=request, purpose=purpose,
            space_fingerprint=self.space.fingerprint,
            selected=_raw, vector=vector, effort=effort,
            snapshot_id=receipt.snapshot_id,
        )
        query_receipt = {
            "kind": "query", "event_id": request_id,
            "action": tape.to_dict(),
            "meridian": clock, "native_meridian": native,
            "request_hash": receipt.request_hash,
            "snapshot_id": receipt.snapshot_id,
            "query_vector_sha256": sha256(
                query.astype("<f4", copy=False).tobytes()
            ).hexdigest(),
            "selected_raw_sha256": sha256(
                _raw.detach().float().cpu().contiguous().numpy().tobytes()
            ).hexdigest(),
            "source_ids": list(source.ids), "memory_ids": list(memory.ids),
            "coverage": scheduling.to_dict(),
        }
        # Publish the entire causal transaction only after every step succeeds.
        self.sessions[id(session)] = scheduling
        session.state = next_state
        session.internal_meridians += 1
        session.revision += 1
        session.pending = None
        session.events.append((session.revision, ("__retrieval_query__",)))
        session.retrieval_receipts.append(query_receipt)
        return BoundaryResult(
            receipt, query, source, memory, tuple(queued),
            tuple(group.group_id for group in groups),
            source.shortfall, memory.shortfall,
        )

    def ready_sources(self, session: TrackSession, *, request_id: str
                      ) -> tuple[SourceObligation, ...]:
        return self.ledger(session).reserve(
            max_bytes=self.policy.max_source_bytes,
            max_events=self.policy.max_source_events,
            now=session.causal_clock,
            request_id=request_id,
        )

    def observe_source(
        self, session: TrackSession, obligation: SourceObligation,
        *, request_id: str, fetch: Callable[[Record, Interval], bytes],
        visibility: Visibility,
    ) -> bool:
        """Source bytes become covered only after the E/C event is committed."""
        ledger = self.ledger(session)
        interval = obligation.interval
        if interval not in ledger.reserved or ledger.reserved[interval] != request_id:
            raise ValueError("Source admission requires a matching reservation")
        if obligation.epoch != ledger.scoring_epoch:
            ledger.release(interval, request_id)
            return False
        try:
            row = self.service.backend.store.get(Collection.SOURCE, obligation.record_id)
            if not row.active or not visibility.permits(row.footprint):
                ledger.release(interval, request_id)
                return False
            span = row.evidence_span
            if (
                span is None or span.source_id != interval.content_id
                or span.version != interval.version
                or interval.start < span.start or interval.end > span.end
            ):
                ledger.release(interval, request_id)
                raise ValueError("Source obligation no longer matches immutable versioned bytes")
            data = fetch(row, interval)
            if not isinstance(data, bytes) or len(data) != interval.end - interval.start:
                raise ValueError("A source asset must return the exact versioned byte slice")
            if obligation.deadline < session.causal_clock:
                ledger.release(interval, request_id)
                return False
            # E/C is a real selected observation. Construct successor off-state
            # and publish content coverage before making it visible to the session.
            observed = ObservedEvent(
                "retrieved_source", "text", data, role_id=ROLE_IDS["document"]
            )
            candidate = self.generator.tracks.observe_group(
                session.state, (observed,), encoder_grad=False,
            )
            receipt = (
                f"source:{interval.content_id}:{interval.version}:"
                f"{interval.start}:{interval.end}:{request_id}"
            )
            ledger.commit(interval, request_id=request_id, receipt=receipt)
            session.state = candidate
            session.internal_meridians += 1
            session.revision += 1
            session.pending = None
            session.events.append((session.revision, ("retrieved_source",)))
            session.retrieval_receipts.append({
                "kind": "source", "request_id": request_id,
                "source_interval": asdict(interval), "record_id": row.id,
                "source_asset_ref": row.asset_ref,
                "source_bytes_sha256": sha256(data).hexdigest(),
                "native_meridian_after": int(session.state.meridians_seen.min()),
                "commit_receipt": receipt,
                "coverage": ledger.to_dict(),
            })
            return True
        except Exception:
            if ledger.reserved.get(interval) == request_id:
                ledger.release(interval, request_id)
            raise
