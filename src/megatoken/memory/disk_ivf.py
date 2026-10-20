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

    def __init__(self, store: SQLiteKB, centroids, *,
                 nprobe: int = 8, batch_size: int = 1024):
        matrix = np.asarray(centroids, dtype="<f4")
        if matrix.ndim != 2 or matrix.shape[1] != store.space.dimension or (
            matrix.shape[0] < 1 or nprobe < 1 or batch_size < 1
        ):
            raise ValueError("Disk-IVF centroids require a compatible learned space")
        self.centroids = np.stack([
            unit_finite(row, dimension=store.space.dimension) for row in matrix
        ])
        self.identity = IVFIdentity(
            sha256(self.centroids.tobytes()).hexdigest(),
            matrix.shape[0], store.space.fingerprint,
        )
        self.store, self.nprobe, self.batch_size = store, nprobe, batch_size

    def cell(self, key) -> int:
        return int(np.argmax(self.centroids @ unit_finite(
            key, dimension=self.store.space.dimension
        )))

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
                            collection.value, int(row["id"]),
                            self.cell(np.frombuffer(row["key"], dtype="<f4")),
                            self.identity.centroid_sha256, int(row["key_revision"]),
                        ) for row in rows
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

    def patch_key(self, collection: Collection, record_id: int, *,
                  changed_key, expected_revision: int) -> None:
        """Storage applies key/posting and revisions in ONE owned write txn."""
        row = self.store.get(collection, record_id)
        self.store.update_record(
            collection, record_id, key=changed_key,
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
                    (f"ivf:{collection.value}:{row.scope}:{self.identity.centroid_sha256}",
                     str(self.store.revisions().key)),
                )

    def _check_complete(self, collection: Collection, scope: str):
        key = f"ivf:{collection.value}:{scope}:{self.identity.centroid_sha256}"
        entry = self.store.db.execute(
            "SELECT value FROM meta WHERE key=?", (key,)
        ).fetchone()
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

    def search(self, query, *, collection: Collection, scope: str,
               visibility: Visibility, limit: int,
               excluded: set[int] | None = None) -> SearchBatch:
        if limit < 0:
            raise ValueError("Negative source/memory retrieval budget")
        q = unit_finite(query, dimension=self.store.space.dimension)
        cells = np.argsort(-(self.centroids @ q), kind="stable")[:self.nprobe]
        with self.store._lock:
            self._check_complete(collection, scope)
            revisions = self.store.revisions()
            hits = []
            candidates_evaluated = 0
            blocked = excluded or set()
            for cell in cells:
                after = 0
                while True:
                    rows = self.store.db.execute(
                        """SELECT r.* FROM postings p JOIN records r
                           ON r.id=p.record_id AND r.collection=p.collection
                           WHERE p.collection=? AND r.scope=? AND p.cell=? AND
                             p.centroids_revision=? AND r.id>?
                           ORDER BY r.id LIMIT ?""",
                        (collection.value, scope, int(cell),
                         self.identity.centroid_sha256, after, self.batch_size),
                    ).fetchall()
                    if not rows:
                        break
                    after = rows[-1]["id"]
                    for entry in rows:
                        if entry["id"] in blocked:
                            continue
                        row = self.store._record(entry)
                        if not row.active or not visibility.permits(row.footprint):
                            continue
                        score = float(np.dot(q.astype(np.float64), row.key.astype(np.float64)))
                        hits.append(Hit(row, score))
                        candidates_evaluated += 1
            hits.sort(key=lambda x: (-x.similarity, x.record.id))
            return SearchBatch(tuple(hits[:limit]), max(0, limit - len(hits)),
                               candidates_evaluated, revisions)
