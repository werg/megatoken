"""Incremental query MaxSim and durable, disjoint source byte coverage."""
from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock

import numpy as np

from .base import Hit, SearchBackend
from .contracts import Collection, QueryBoundary, Visibility


@dataclass(frozen=True, order=True)
class Interval:
    content_id: str
    version: str
    start: int
    end: int

    def __post_init__(self) -> None:
        if not self.content_id or not self.version or self.start < 0 or self.end <= self.start:
            raise ValueError("Coverage requires a versioned positive byte interval")


def subtract(interval: Interval, occupied: list[Interval]) -> list[Interval]:
    available = [(interval.start, interval.end)]
    for previous in occupied:
        if (previous.content_id, previous.version) != (interval.content_id, interval.version):
            continue
        upcoming = []
        for lo, hi in available:
            if previous.end <= lo or previous.start >= hi:
                upcoming.append((lo, hi))
            else:
                if lo < previous.start:
                    upcoming.append((lo, previous.start))
                if previous.end < hi:
                    upcoming.append((previous.end, hi))
        available = upcoming
    return [Interval(interval.content_id, interval.version, lo, hi) for lo, hi in available]


@dataclass(frozen=True)
class SourceObligation:
    interval: Interval
    record_id: int
    score: float
    event_id: str
    epoch: str
    deadline: int
    ordinal: int = 0


@dataclass
class RetrievalSession:
    """Pure scheduling state: never a second native event clock."""
    scoring_epoch: str = ""
    maxima: dict[int, float] = field(default_factory=dict)
    qualified: dict[Interval, SourceObligation] = field(default_factory=dict)
    reserved: dict[Interval, str] = field(default_factory=dict)
    injected: list[Interval] = field(default_factory=list)
    receipts: set[str] = field(default_factory=set)
    _lock: RLock = field(default_factory=RLock, repr=False)

    def new_epoch(self, epoch: str) -> None:
        with self._lock:
            if epoch != self.scoring_epoch:
                self.scoring_epoch = epoch
                self.maxima.clear()
                self.qualified.clear()
                self.reserved.clear()
                # Committed bytes remain covered across key/index changes.

    def incremental(
        self, hits: list[Hit], *, boundary: QueryBoundary, threshold: float,
    ) -> list[SourceObligation]:
        """Only fresh incoming scores can QUALIFY bytes; old high maxima cannot."""
        if not -1 <= threshold <= 1:
            raise ValueError("Threshold must be a cosine value")
        with self._lock:
            self.new_epoch(boundary.snapshot_id)
            new = []
            for hit in hits:
                row = hit.record
                if row.collection is not Collection.SOURCE:
                    raise ValueError("Only literal source chunks enter MaxSim")
                self.maxima[row.id] = max(self.maxima.get(row.id, -1.0), hit.similarity)
                if hit.similarity < threshold:
                    continue
                for part in row.footprint.ranges:
                    whole = Interval(row.content_id, part.version, part.start, part.end)
                    taken = self.injected + list(self.reserved) + list(self.qualified)
                    for missing in subtract(whole, taken):
                        if missing not in self.qualified:
                            obligation = SourceObligation(
                                missing, row.id, hit.similarity, boundary.event_id,
                                self.scoring_epoch, boundary.deadline
                            )
                            self.qualified[missing] = obligation
                            new.append(obligation)
            return new

    def reserve(self, *, max_bytes: int, max_events: int, now: int,
                request_id: str) -> tuple[SourceObligation, ...]:
        """Claim novel chunks before fetching; retry keeps the original score."""
        if max_bytes < 0 or max_events < 0 or not request_id:
            raise ValueError("Invalid source reservation budget/identity")
        with self._lock:
            selected = []
            remaining = max_bytes
            for interval, obligation in sorted(
                self.qualified.items(), key=lambda x: (-x[1].score, x[1].ordinal, x[0])
            ):
                if obligation.deadline < now or obligation.epoch != self.scoring_epoch:
                    continue
                if interval in self.reserved or interval.end - interval.start > remaining:
                    continue
                if len(selected) >= max_events:
                    break
                remainder = subtract(interval, self.injected + list(self.reserved))
                if len(remainder) != 1 or remainder[0] != interval:
                    continue
                self.reserved[interval] = request_id
                remaining -= interval.end - interval.start
                selected.append(obligation)
            return tuple(selected)

    def release(self, interval: Interval, request_id: str) -> None:
        with self._lock:
            if self.reserved.get(interval) != request_id:
                raise ValueError("Reservation is absent or owned by another fetch")
            del self.reserved[interval]

    def commit(self, interval: Interval, *, request_id: str, receipt: str) -> None:
        """Call atomically WITH successful native E/C ingestion, not on fetch."""
        with self._lock:
            if receipt in self.receipts:
                return
            if self.reserved.get(interval) != request_id:
                raise ValueError("Coverage commit requires a matching reservation")
            if subtract(interval, self.injected) != [interval]:
                raise ValueError("Overlapping source coverage cannot be committed twice")
            self.injected.append(interval)
            del self.reserved[interval]
            self.qualified.pop(interval, None)
            self.receipts.add(receipt)

    def expire(self, now: int) -> None:
        with self._lock:
            for interval, item in tuple(self.qualified.items()):
                if item.deadline < now:
                    self.qualified.pop(interval)
                    self.reserved.pop(interval, None)


class RetrievalService:
    def __init__(self, backend: SearchBackend):
        self.backend = backend

    def search(self, query: np.ndarray, *, visibility: Visibility,
               source_limit: int, memory_limit: int, excluded_source: set[int] | None = None):
        """Same learned space, independent budgets and eligibility-first scans."""
        options = {"scope": visibility.scope, "visibility": visibility}
        sources = self.backend.search(query, collection=Collection.SOURCE, limit=source_limit,
                                      excluded=excluded_source, **options)
        memories = self.backend.search(query, collection=Collection.MEMORY, limit=memory_limit,
                                       **options)
        return sources, memories
