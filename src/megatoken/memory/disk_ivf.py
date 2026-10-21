"""Disk-backed IVF postings over canonical SQLite, with exact per-cell rescoring."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

import numpy as np

from .base import Hit, SearchBatch
from .contracts import Collection, Visibility, unit_finite
from .sqlite import SQLiteKB


@dataclass(frozen=True)
class IVFIdentity:
    centroid_sha256: str
    cells: int
    space_fingerprint: str


class SQLiteDiskIVF:
    """Keeps ONLY centroids in RAM, never an all-key index mirror.

    Postings are part of each key-commit transaction. Filter visibility before
    top-k, and return one coherent SQLite scoring snapshot per call.
    """

    def __init__(self, store: SQLiteKB, centroids, *, nprobe: int, batch_size: int = 1024) -> None:
        matrix = np.asarray(centroids, dtype="<f4")
        if (
            matrix.ndim != 2
            or matrix.shape[1] != store.space.dimension
            or not 1 <= nprobe <= matrix.shape[0]
            or batch_size < 1
        ):
            raise ValueError("Disk-IVF centroids require a compatible learned space")
        self.centroids = np.stack(
            [unit_finite(row, dimension=store.space.dimension) for row in matrix]
        )
        self.identity = IVFIdentity(
            sha256(self.centroids.tobytes()).hexdigest(),
            matrix.shape[0],
            store.space.fingerprint,
        )
        self.store, self.nprobe, self.batch_size = store, nprobe, batch_size
        # (collection, scope) -> store revisions at which postings were verified complete.
        self._verified: dict[tuple[str, str], object] = {}

    @classmethod
    def published(cls, store: SQLiteKB, centroids) -> SQLiteDiskIVF:
        """The index as published into ``store``, with its calibrated probe count."""
        receipt = store.growth_state("ivf_publication")
        index = cls(store, centroids, nprobe=1)
        if receipt is None or receipt["centroid_vectors_sha256"] != index.identity.centroid_sha256:
            raise RuntimeError(
                "The bank has no IVF publication for these centroids; run megatoken-memory-index"
            )
        if not 1 <= receipt["nprobe"] <= index.identity.cells:
            raise RuntimeError("Published IVF probe count is outside the centroid table")
        index.nprobe = int(receipt["nprobe"])
        return index

    def cell(self, key) -> int:
        return int(
            np.argmax(self.centroids @ unit_finite(key, dimension=self.store.space.dimension))
        )

    def rebuild(self, *, scope: str, collection: Collection) -> int:
        """Stream bounded SQLite key pages into atomic IVF postings.

        Neither a full key-bank tensor nor record metadata mirror is needed.
        """
        count = 0
        with self.store.transaction():
            cursor = self.store.db.execute(
                """SELECT id,key,key_revision FROM records
                   WHERE collection=? AND scope=? AND active=1 ORDER BY id""",
                (collection.value, scope),
            )
            while True:
                rows = cursor.fetchmany(self.batch_size)
                if not rows:
                    break
                self.store.db.executemany(
                    """INSERT INTO postings VALUES (?,?,?,?,?)
                       ON CONFLICT(collection,record_id) DO UPDATE SET
                       cell=excluded.cell,centroids_revision=excluded.centroids_revision,
                       key_revision=excluded.key_revision""",
                    [
                        (
                            collection.value,
                            int(row["id"]),
                            self.cell(np.frombuffer(row["key"], dtype="<f4")),
                            self.identity.centroid_sha256,
                            int(row["key_revision"]),
                        )
                        for row in rows
                    ],
                )
                count += len(rows)
            self.store.db.execute(
                """DELETE FROM postings WHERE collection=? AND record_id IN (
                     SELECT id FROM records WHERE collection=? AND scope=? AND active=0
                   )""",
                (collection.value, collection.value, scope),
            )
            self.store.db.execute(
                """INSERT INTO meta VALUES (?,?) ON CONFLICT(key)
                   DO UPDATE SET value=excluded.value""",
                (
                    f"ivf:{collection.value}:{scope}:{self.identity.centroid_sha256}",
                    str(self.store.revisions().key),
                ),
            )
            self.store._bump("index")
        return count

    def patch_key(
        self, collection: Collection, record_id: int, *, changed_key, expected_revision: int
    ) -> None:
        """Storage applies key/posting and revisions in ONE owned write txn."""
        with self.store.transaction():
            self._patch_key(collection, record_id, changed_key, expected_revision)

    def _patch_key(self, collection, record_id, changed_key, expected_revision) -> None:
        row = self.store.get(collection, record_id)
        self.store.update_record(
            collection,
            record_id,
            key=changed_key,
            posting_cell=self.cell(changed_key),
            centroids_revision=self.identity.centroid_sha256,
            expected_key_revision=expected_revision,
        )
        # One-row publication is current only if every previous posting was
        # complete. Do not claim that a partially built corpus is indexed.
        existing = self.store.db.execute(
            """SELECT COUNT(*) FROM records r LEFT JOIN postings p ON
               r.id=p.record_id AND p.collection=r.collection AND
               p.centroids_revision=? AND p.key_revision=r.key_revision
               WHERE r.collection=? AND r.scope=? AND r.active=1 AND p.record_id IS NULL""",
            (self.identity.centroid_sha256, collection.value, row.scope),
        ).fetchone()[0]
        if existing == 0:
            with self.store.transaction():
                self.store.db.execute(
                    """INSERT INTO meta VALUES (?,?) ON CONFLICT(key)
                       DO UPDATE SET value=excluded.value""",
                    (
                        f"ivf:{collection.value}:{row.scope}:{self.identity.centroid_sha256}",
                        str(self.store.revisions().key),
                    ),
                )

    def _check_complete(self, collection: Collection, scope: str, revisions) -> None:
        # Any record or key change advances a store revision, so a verified
        # (collection, scope) stays complete until the revisions move.
        if self._verified.get((collection.value, scope)) == revisions:
            return
        key = f"ivf:{collection.value}:{scope}:{self.identity.centroid_sha256}"
        entry = self.store.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        if entry is None:
            raise RuntimeError("Disk-IVF postings are not published for this collection/scope")
        # Cross-collection updates may advance global key generation: validate
        # actual per-row posting revisions instead of a global counter equality.
        gap = self.store.db.execute(
            """SELECT EXISTS(SELECT 1 FROM records r LEFT JOIN postings p
               ON r.id=p.record_id AND p.collection=r.collection AND
                  p.centroids_revision=? AND p.key_revision=r.key_revision
               WHERE r.collection=? AND r.scope=? AND r.active=1 AND p.record_id IS NULL)""",
            (self.identity.centroid_sha256, collection.value, scope),
        ).fetchone()[0]
        if gap:
            raise RuntimeError("Index cannot search unposted or stale learned keys")
        self._verified[(collection.value, scope)] = revisions

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
        q = unit_finite(query, dimension=self.store.space.dimension).astype(np.float64)
        cells = np.argsort(-(self.centroids @ q), kind="stable")[: self.nprobe]
        with self.store.read_snapshot():
            revisions = self.store.revisions()
            self._check_complete(collection, scope, revisions)
            ids: list[int] = []
            keys: list[bytes] = []
            for cell in cells:
                rows = self.store.db.execute(
                    """SELECT r.id, r.key FROM postings p JOIN records r
                       ON r.id=p.record_id AND r.collection=p.collection
                       WHERE p.collection=? AND r.scope=? AND p.cell=? AND
                         p.centroids_revision=? AND r.active=1""",
                    (collection.value, scope, int(cell), self.identity.centroid_sha256),
                ).fetchall()
                if rows:
                    cell_ids, cell_keys = zip(*rows, strict=True)
                    ids.extend(cell_ids)
                    keys.extend(cell_keys)
            candidates = np.asarray(ids, dtype=np.int64)
            keep = ~np.isin(candidates, list(excluded or ()))
            if not keep.any() or limit == 0:
                return SearchBatch((), limit, int(keep.sum()), revisions)
            # Score every candidate key at once; materialize records only in
            # score order, until ``limit`` of them pass visibility.
            matrix = np.frombuffer(b"".join(keys), dtype="<f4")
            if matrix.size != len(ids) * self.store.space.dimension:
                raise ValueError("Corrupt or incompatible IVF candidate keys")
            matrix = matrix.reshape(len(ids), -1)[keep]
            candidates = candidates[keep]
            scores = matrix.astype(np.float64) @ q
            if not np.isfinite(scores).all():
                raise ValueError("Stored IVF candidate keys contain nonfinite coordinates")
            order = np.lexsort((candidates, -scores))
            hits: list[Hit] = []
            start, size = 0, 2 * limit  # visibility rarely rejects; grow if it does
            while start < len(order):
                chunk = order[start : start + size]
                start, size = start + size, self.batch_size
                chosen = [int(i) for i in candidates[chunk]]
                rows = {
                    row["id"]: row
                    for row in self.store.db.execute(
                        f"SELECT * FROM records WHERE id IN ({','.join('?' * len(chosen))})",
                        chosen,
                    )
                }
                for index, record_id in zip(chunk, chosen, strict=True):
                    record = self.store._record(rows[record_id])
                    if visibility.permits(record.footprint):
                        hits.append(Hit(record, float(scores[index])))
                        if len(hits) == limit:
                            return SearchBatch(tuple(hits), 0, len(candidates), revisions)
            return SearchBatch(tuple(hits), limit - len(hits), len(candidates), revisions)
