"""Eligible-first exact and coarse IVF candidate search with live-key rescoring."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import Hit, RecordStore, SearchBatch
from .contracts import Collection, Visibility, unit_finite


class ExactIndex:
    """Reference search: scope/footprint/coverage exclusions precede local top-k."""

    def __init__(self, store: RecordStore):
        self.store = store

    def search(
        self,
        query,
        *,
        collection: Collection,
        scope: str,
        visibility: Visibility,
        limit: int,
        excluded: set[int] | None = None,
    ) -> SearchBatch:
        (result,) = self.search_many(
            [query],
            collection=collection,
            scope=scope,
            visibility=visibility,
            limit=limit,
            excluded=excluded,
        )
        return result

    def search_many(
        self,
        queries,
        *,
        collection: Collection,
        scope: str,
        visibility: Visibility,
        limit: int,
        excluded: set[int] | None = None,
    ) -> list[SearchBatch]:
        """One eligibility scan and one FP64 score matrix for every query."""
        if limit < 0:
            raise ValueError("Candidate count cannot be negative")
        dimension = self.store.space.dimension
        q = np.stack([unit_finite(x, dimension=dimension) for x in queries]).astype(np.float64)
        blocked = excluded or set()
        rows = [
            row
            for row in self.store.scan(collection, scope)
            if row.active and row.id not in blocked and visibility.permits(row.footprint)
        ]
        revisions = self.store.revisions()
        if not rows:
            return [SearchBatch((), limit, 0, revisions) for _ in range(len(q))]
        keys = np.stack([row.key for row in rows]).astype(np.float64)
        ids = np.asarray([row.id for row in rows])
        scores = q @ keys.T
        results = []
        for score in scores:
            # Rank by similarity, ties broken by stable record ID.
            order = np.lexsort((ids, -score))[:limit]
            results.append(
                SearchBatch(
                    tuple(Hit(rows[i], float(score[i])) for i in order),
                    max(0, limit - len(rows)),
                    len(rows),
                    revisions,
                )
            )
        return results


@dataclass
class CoarseIVF:
    """Deterministic coarse reference. Exact-rescore probed eligible cells.

    Disk persistence of postings is owned by SQLite; this mirror is disposable.
    """

    store: RecordStore
    centroids: np.ndarray
    probes: int = 4

    def __post_init__(self) -> None:
        c = np.asarray(self.centroids, dtype="<f4")
        if c.ndim != 2 or c.shape[1] != self.store.space.dimension or self.probes < 1:
            raise ValueError("IVF cells must share the learned address geometry")
        self.centroids = np.stack([unit_finite(x, dimension=c.shape[1]) for x in c])

    def posting_cell(self, key: np.ndarray) -> int:
        return int(np.argmax(self.centroids @ key))

    def search(
        self,
        query,
        *,
        collection: Collection,
        scope: str,
        visibility: Visibility,
        limit: int,
        excluded: set[int] | None = None,
    ) -> SearchBatch:
        if limit < 0:
            raise ValueError("Negative source/memory retrieval budget")
        q = unit_finite(query, dimension=self.store.space.dimension)
        probes = np.argsort(-(self.centroids @ q), kind="stable")[: self.probes]
        cells = set(probes.tolist())
        blocked = excluded or set()
        hits = []
        for row in self.store.scan(collection, scope):
            if not row.active or row.id in blocked or not visibility.permits(row.footprint):
                continue
            # Moved learned keys are assigned by their CURRENT normalized location.
            if self.posting_cell(row.key) in cells:
                hits.append(
                    Hit(row, float(np.dot(q.astype(np.float64), row.key.astype(np.float64))))
                )
        hits.sort(key=lambda x: (-x.similarity, x.record.id))
        return SearchBatch(
            tuple(hits[:limit]), max(0, limit - len(hits)), len(hits), self.store.revisions()
        )
