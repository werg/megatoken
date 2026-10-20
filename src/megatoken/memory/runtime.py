"""One native retrieval boundary for literal source E/C and direct semantic KB.

The backend is a candidate locator. Native P creates the query event, R
realizes its selected raw facets, and source evidence is observed byte-exact
through the existing E/C runtime; semantic K/V is never an artificial event.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Callable

import numpy as np
import torch

from megatoken.memory.base import Record, SearchBatch
from megatoken.memory.contracts import Collection, QueryBoundary, SemanticSpace, Visibility
from megatoken.memory.reader import build_memory_groups
from megatoken.memory.retrieval import (
    Interval, RetrievalService, RetrievalSession, SourceObligation, subtract,
)
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
            row.id for row in records if row.footprint.ranges and all(
                not subtract(
                    Interval(row.content_id, part.version, part.start, part.end), covered
                )
                for part in row.footprint.ranges
            )
        }

    def boundary(self, session: TrackSession, *, request: bytes,
                 purpose: str, visibility: Visibility, request_id: str,
                 effort: int = 6, generator: torch.Generator | None = None,
                 enable_memory: bool = True) -> BoundaryResult:
        if not request_id or not isinstance(request, bytes):
            raise ValueError("A retrieval boundary requires an immutable request identity")
        clock = int(session.state.meridians_seen.min())
        if visibility.committed_cursor > clock:
            raise ValueError("Future sources cannot be made available to a past query")
        vector, next_state, _raw = self.generator.retrieve_embedding(
            session.state, main_request=request, purpose=purpose,
            space_fingerprint=self.space.fingerprint, effort=effort,
            generator=generator,
        )
        session.state = next_state
        session.revision += 1
        session.pending = None
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
        queued = self.ledger(session).incremental(
            list(source.hits), boundary=receipt,
            threshold=self.policy.source_threshold,
        )
        groups = ()
        if enable_memory and memory.hits and self.policy.max_compat_groups:
            valid_hits = [hit for hit in memory.hits
                          if hit.similarity >= self.policy.memory_threshold]
            # Bound incompatible groups before materialization, deterministic
            # rank-score order, no grouping solely by vector proximity.
            chosen: list = []
            named = set()
            for hit in valid_hits:
                row = hit.record
                key = (row.scope, row.group or f"point:{row.id}", row.payload_revision)
                if key not in named and len(named) >= self.policy.max_compat_groups:
                    continue
                chosen.append(row)
                named.add(key)
            if chosen:
                device = session.state.device
                with torch.no_grad():
                    keys = torch.from_numpy(np.stack([r.key for r in chosen])).to(device)[None]
                    values = torch.from_numpy(np.stack([r.value for r in chosen])).to(device)[None]
                    strengths = torch.tensor(
                        [[r.strength for r in chosen]], device=device
                    )
                    lifetimes = torch.tensor(
                        [[r.lifetime_logit for r in chosen]], device=device
                    )
                    queries = torch.from_numpy(query).to(device)[None]
                    groups, _ = build_memory_groups(
                        queries=queries, keys=keys, semantics=values,
                        strengths=strengths, lifetime_logits=lifetimes,
                        record_ids=[f"memory:{r.id}" for r in chosen],
                        scopes=[r.scope for r in chosen],
                        groups=[r.group or f"point:{r.id}" for r in chosen],
                        versions=[f"payload-v{r.payload_revision}" for r in chosen],
                        eligible=torch.ones(1, len(chosen), dtype=torch.bool, device=device),
                        anchor=int(session.state.meridians_seen.min()),
                        threshold=self.policy.memory_threshold,
                        max_groups=self.policy.max_compat_groups,
                    )
                for group in groups:
                    self.runtime.admit_memory(session, group)
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
            now=int(session.state.meridians_seen.min()),
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
            data = fetch(row, interval)
            if not isinstance(data, bytes) or len(data) != interval.end - interval.start:
                raise ValueError("A source asset must return the exact versioned byte slice")
            if obligation.deadline < int(session.state.meridians_seen.min()):
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
            session.revision += 1
            session.pending = None
            session.events.append((session.revision, ("retrieved_source",)))
            return True
        except Exception:
            if ledger.reserved.get(interval) == request_id:
                ledger.release(interval, request_id)
            raise
