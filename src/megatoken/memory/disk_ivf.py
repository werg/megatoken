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
                keys = [np.frombuffer(row["key"], dtype="<f4") for row in rows]
                self.store.write_postings(
                    collection,
                    [
                        (int(row["id"]), self.cell(key), int(row["key_revision"]), key)
                        for row, key in zip(rows, keys, strict=True)
                    ],
                    scope=scope,
                    centroids_revision=self.identity.centroid_sha256,
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
        """Exact FP64 top-``limit`` over the probed cells' candidates.

        Candidates are ranked by their int8 codes; FP64 keys are read only in
        that order until the ``limit``-th visible exact score beats every
        unread candidate's code score plus its quantization error bound
        (``scale / 2 * |q|_1``), so the result equals exact rescoring of all
        candidates.
        """
        if limit < 0:
            raise ValueError("Negative source/memory retrieval budget")
        q = unit_finite(query, dimension=self.store.space.dimension).astype(np.float64)
        cells = np.argsort(-(self.centroids @ q), kind="stable")[: self.nprobe]
        with self.store.read_snapshot():
            revisions = self.store.revisions()
            self._check_complete(collection, scope, revisions)
            rows = self.store.db.execute(
                f"""SELECT record_id, scale, code FROM postings
                    WHERE collection=? AND centroids_revision=? AND scope=?
                      AND cell IN ({",".join("?" * len(cells))})""",
                (collection.value, self.identity.centroid_sha256, scope, *map(int, cells)),
            ).fetchall()
            if excluded:
                rows = [row for row in rows if row[0] not in excluded]
            if not rows or limit == 0:
                return SearchBatch((), limit, len(rows), revisions)
            ids, scales, codes = zip(*rows, strict=True)
            candidates = np.asarray(ids, dtype=np.int64)
            scales = np.asarray(scales, dtype=np.float64)
            matrix = np.frombuffer(b"".join(codes), dtype=np.int8)
            if matrix.size != len(rows) * self.store.space.dimension:
                raise ValueError("Corrupt or incompatible IVF posting codes")
            approx = (
                matrix.reshape(len(rows), -1).astype(np.float32) @ q.astype(np.float32)
            ) * scales
            bound = approx + scales * (0.5 * np.abs(q).sum() + 1e-6)
            order = np.argsort(-approx, kind="stable")
            # ceiling[i]: the best exact score any candidate from order[i:] can have.
            ceiling = np.append(np.maximum.accumulate(bound[order][::-1])[::-1], -np.inf)
            exact: dict[int, float] = {}
            records: dict[int, object] = {}
            read, window = 0, max(4 * limit, 64)
            while True:
                chosen = [int(i) for i in candidates[order[read : read + window]]]
                read, window = read + len(chosen), 2 * window
                for row in self.store.db.execute(
                    f"SELECT id, key FROM records WHERE id IN ({','.join('?' * len(chosen))})",
                    chosen,
                ):
                    key = np.frombuffer(row["key"], dtype="<f4").astype(np.float64)
                    exact[int(row["id"])] = float(key @ q)
                ranked = sorted(exact, key=lambda i: (-exact[i], i))
                missing = [i for i in ranked if i not in records]
                if missing:
                    for row in self.store.db.execute(
                        f"SELECT * FROM records WHERE id IN ({','.join('?' * len(missing))})",
                        missing,
                    ):
                        record = self.store._record(row)
                        records[record.id] = (
                            record if visibility.permits(record.footprint) else None
                        )
                hits = [Hit(records[i], exact[i]) for i in ranked if records[i] is not None]
                if read >= len(order) or (
                    len(hits) >= limit and hits[limit - 1].similarity > ceiling[read]
                ):
                    break
            if not np.isfinite([hit.similarity for hit in hits]).all():
                raise ValueError("Stored IVF candidate keys contain nonfinite coordinates")
            hits = hits[:limit]
            return SearchBatch(tuple(hits), limit - len(hits), len(candidates), revisions)
