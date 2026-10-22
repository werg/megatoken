"""A source collection: original items of any medium behind one search key each.

A collection is a directory: ``collection.sqlite`` (the catalog, the keys and the
index) and ``media/`` (the teacher features of the items a model observes as
features, in the same content-addressed form as a prepared dataset). Its items:

- ``text``: a short passage, kept as its exact bytes (at most ``TEXT_LIMIT``);
- ``compressed_text``: one piece of a long text, kept as EmbeddingGemma2 hidden states;
- ``audio``, ``image``, ``video``: a clip, kept as its EmbeddingGemma2 hidden states.

The key of every item is the teacher's own pooled 768-wide embedding of it, one
space for every medium, as stored with each prepared item. The index is the
quantized inverted file of ``megatoken.retrieval.index`` on disk: only the cell
centres are held in memory. A collection is written once (``reindex`` after the
first items) and can then be extended by ``add_*``; a mounted, read-only
collection is shared freely between processes.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import threading
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from megatoken.data.media_assets import MediaRecord, TensorAssetStore
from megatoken.io import canonical_json
from megatoken.teacher import EMBEDDING_WIDTH

from .index import (
    MAX_CELLS,
    MAX_FIT_KEYS,
    calibrate_nprobe,
    fit_centroids,
    noisy_queries,
    quantize,
    rank_exact,
    unit,
)

SCHEMA = 1
DATABASE = "collection.sqlite"
TEXT_LIMIT = 4096
MODALITIES = ("text", "compressed_text", "audio", "image", "video")
# Descriptor modality of a feature item -> the item modality (and event modality).
ITEM_MODALITY = {"text": "compressed_text", "audio": "audio", "image": "image", "video": "video"}
# A starting score floor on the raw cosine between a query and an item key.
DEFAULT_THRESHOLD = 0.5
_BATCH = 900

_TABLES = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value NOT NULL);
CREATE TABLE items (
    id INTEGER PRIMARY KEY,
    item_id TEXT NOT NULL UNIQUE,
    modality TEXT NOT NULL,
    origin TEXT NOT NULL,
    text BLOB,
    descriptor BLOB,
    key BLOB NOT NULL,
    CHECK ((text IS NULL) != (descriptor IS NULL))
);
CREATE TABLE postings (
    cell INTEGER NOT NULL,
    item INTEGER NOT NULL,
    scale REAL NOT NULL,
    code BLOB NOT NULL,
    PRIMARY KEY (cell, item)
) WITHOUT ROWID;
"""


def text_item_id(text: bytes) -> str:
    return hashlib.sha256(text).hexdigest()


@dataclass(frozen=True)
class Item:
    """One original item. ``text`` for a passage, ``record`` for teacher features."""

    row: int
    item_id: str
    modality: str
    origin: str
    text: bytes | None
    record: MediaRecord | None


@dataclass(frozen=True)
class Hit:
    row: int
    item_id: str
    score: float


class SourceCollection:
    """The catalog, keys and disk index of one collection directory."""

    def __init__(self, path: Path, db: sqlite3.Connection, *, writable: bool) -> None:
        self.path = path
        self.writable = writable
        self.assets = TensorAssetStore(path)
        self._db = db
        self._lock = threading.RLock()
        self._centroids: np.ndarray | None = None
        self.name = self._meta("name")
        self.teacher = self._meta("teacher")
        if self._meta("schema") != SCHEMA or self._meta("width") != EMBEDDING_WIDTH:
            raise ValueError(f"{path} is not a version {SCHEMA} source collection")

    # ------------------------------------------------------------------ storage

    @classmethod
    def create(cls, path: str | Path, *, name: str, teacher: str) -> SourceCollection:
        path = Path(path)
        if (path / DATABASE).exists():
            raise FileExistsError(f"A collection already exists at {path}")
        if not name or not teacher:
            raise ValueError("A collection needs a name and the teacher whose keys it holds")
        path.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(path / DATABASE, check_same_thread=False)
        with db:
            db.executescript(_TABLES)
            db.executemany(
                "INSERT INTO meta VALUES (?, ?)",
                [
                    ("schema", json.dumps(SCHEMA)),
                    ("width", json.dumps(EMBEDDING_WIDTH)),
                    ("name", json.dumps(name)),
                    ("teacher", json.dumps(teacher)),
                    ("threshold", json.dumps(DEFAULT_THRESHOLD)),
                ],
            )
        return cls(path, db, writable=True)

    @classmethod
    def open(
        cls, path: str | Path, *, writable: bool = False, teacher: str | None = None
    ) -> SourceCollection:
        """Open an existing collection; ``teacher`` must be the one its keys come from."""
        path = Path(path)
        database = path / DATABASE
        if not database.is_file():
            raise FileNotFoundError(f"No source collection at {path}")
        db = sqlite3.connect(
            f"file:{database}?mode={'rw' if writable else 'ro'}", uri=True, check_same_thread=False
        )
        collection = cls(path, db, writable=writable)
        if teacher is not None and collection.teacher != teacher:
            raise ValueError(
                f"Collection {collection.name!r} holds keys of teacher {collection.teacher!r}, "
                f"not {teacher!r}"
            )
        return collection

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> SourceCollection:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def _meta(self, key: str) -> Any:
        row = self._db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        if row is None:
            return None
        value = row[0]
        return value if isinstance(value, bytes) else json.loads(value)

    def _set_meta(self, key: str, value: Any) -> None:
        stored = value if isinstance(value, bytes) else json.dumps(value)
        self._db.execute(
            "INSERT INTO meta VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, stored),
        )

    def _write(self) -> None:
        if not self.writable:
            raise PermissionError(f"Collection {self.name!r} is mounted read-only")

    @property
    def count(self) -> int:
        with self._lock:
            return self._db.execute("SELECT COUNT(*) FROM items").fetchone()[0]

    @property
    def threshold(self) -> float:
        return self._meta("threshold")

    def set_threshold(self, value: float) -> None:
        self._write()
        if not -1 <= value <= 1:
            raise ValueError("A score threshold is a cosine")
        with self._lock, self._db:
            self._set_meta("threshold", value)

    @property
    def query_center(self) -> np.ndarray | None:
        """Mean unit teacher embedding of the queries this collection was built to answer."""
        value = self._meta("query_center")
        return None if value is None else np.asarray(value, dtype="<f4")

    def set_query_center(self, center: np.ndarray) -> None:
        self._write()
        center = np.asarray(center, dtype="<f4")
        if center.shape != (EMBEDDING_WIDTH,) or not np.isfinite(center).all():
            raise ValueError("A query center is one finite teacher-width vector")
        with self._lock, self._db:
            self._set_meta("query_center", center.tolist())

    # -------------------------------------------------------------------- items

    def _insert(
        self,
        item_id: str,
        modality: str,
        origin: str,
        key: np.ndarray,
        *,
        text: bytes | None = None,
        descriptor: bytes | None = None,
    ) -> int | None:
        """The new item's row, or None when an item with this identity is already held."""
        self._write()
        key = unit(key, width=EMBEDDING_WIDTH)
        with self._lock, self._db:
            if self._db.execute("SELECT 1 FROM items WHERE item_id=?", (item_id,)).fetchone():
                return None
            row = self._db.execute(
                "INSERT INTO items(item_id, modality, origin, text, descriptor, key) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (item_id, modality, origin, text, descriptor, key.tobytes()),
            ).lastrowid
            centroids = self._index()
            if centroids is not None:
                scale, code = quantize(key)
                self._db.execute(
                    "INSERT INTO postings VALUES (?, ?, ?, ?)",
                    (int(np.argmax(centroids @ key)), row, scale, code.tobytes()),
                )
            return row

    def add_text(self, text: bytes, key: np.ndarray, *, origin: str) -> int | None:
        """A short passage; ``key`` is the teacher's embedding of it."""
        if not text or len(text) > TEXT_LIMIT:
            raise ValueError(
                f"A text item is 1 to {TEXT_LIMIT} bytes; longer text is read compressed"
            )
        text.decode("utf-8")
        return self._insert(text_item_id(text), "text", origin, key, text=text)

    def add_record(
        self, record: MediaRecord, source: TensorAssetStore, *, origin: str
    ) -> int | None:
        """A prepared feature item (a clip, an image, or a piece of a long text).

        Its teacher features come into this collection's own asset store, so the
        collection stands alone; the realization target, if any, stays behind.
        """
        if record.modality not in ITEM_MODALITY:
            raise ValueError("Unsupported item modality")
        self._write()
        held = MediaRecord(
            record.modality, record.source, prompt=record.prompt, embedding=record.embedding
        )
        if source.root != self.assets.root:
            for asset in (held.source, held.embedding):
                stored = self.assets.put(
                    source.load(asset),
                    **{k: v for k, v in asdict(asset).items() if k not in ("sha256", "shape")},
                )
                if stored != asset:
                    raise ValueError("Imported tensor asset changed its identity")
        key = self.assets.load(held.embedding)
        return self._insert(
            held.source.source_family,
            ITEM_MODALITY[record.modality],
            origin,
            key,
            descriptor=held.to_bytes(),
        )

    def _item(self, row: Sequence[Any]) -> Item:
        record = None if row[5] is None else MediaRecord.from_bytes(bytes(row[5]))
        return Item(
            row[0], row[1], row[2], row[3], None if row[4] is None else bytes(row[4]), record
        )

    _COLUMNS = "id, item_id, modality, origin, text, descriptor"

    def item(self, row: int) -> Item:
        with self._lock:
            found = self._db.execute(
                f"SELECT {self._COLUMNS} FROM items WHERE id=?", (row,)
            ).fetchone()
        if found is None:
            raise KeyError(row)
        return self._item(found)

    def find(self, item_ids: Iterable[str]) -> dict[str, int]:
        """The rows of those of the named items this collection holds."""
        wanted = list(dict.fromkeys(item_ids))
        found: dict[str, int] = {}
        with self._lock:
            for start in range(0, len(wanted), _BATCH):
                batch = wanted[start : start + _BATCH]
                found.update(
                    self._db.execute(
                        "SELECT item_id, id FROM items "
                        f"WHERE item_id IN ({','.join('?' * len(batch))})",
                        batch,
                    )
                )
        return found

    def keys(self, rows: Sequence[int]) -> np.ndarray:
        """The float keys of ``rows``, in that order: [len(rows), width]."""
        found: dict[int, np.ndarray] = {}
        with self._lock:
            for start in range(0, len(rows), _BATCH):
                batch = [int(row) for row in rows[start : start + _BATCH]]
                for row, blob in self._db.execute(
                    f"SELECT id, key FROM items WHERE id IN ({','.join('?' * len(batch))})", batch
                ):
                    found[row] = np.frombuffer(blob, dtype="<f4")
        if len(found) != len(set(int(row) for row in rows)):
            raise KeyError("Unknown item row")
        if not rows:
            return np.empty((0, EMBEDDING_WIDTH), dtype="<f4")
        return np.stack([found[int(row)] for row in rows])

    def features(self, item: Item) -> np.ndarray:
        """The teacher features [tokens, 512] an item is observed as."""
        if item.record is None:
            raise ValueError("A text item is observed as its bytes")
        return self.assets.load(item.record.source)

    # ------------------------------------------------------------------- index

    def _index(self) -> np.ndarray | None:
        if self._centroids is None:
            blob = self._meta("centroids")
            if blob is not None:
                self._centroids = np.frombuffer(blob, dtype="<f4").reshape(-1, EMBEDDING_WIDTH)
        return self._centroids

    @property
    def indexed(self) -> bool:
        return self._index() is not None

    @property
    def nprobe(self) -> int:
        return self._meta("nprobe")

    def reindex(
        self,
        *,
        cells: int | None = None,
        target_recall: float = 0.97,
        k: int = 10,
        seed: int = 12345,
    ) -> dict[str, Any]:
        """Fit the cells, post every item and calibrate the probe count.

        The probe count is the smallest whose top-k overlaps the exact top-k at
        ``target_recall``, measured on a key sample with noisy copies of keys as
        queries (a stand-in for a student's imperfect queries).
        """
        self._write()
        total = self.count
        if not total:
            raise ValueError("Cannot index an empty collection")
        cells = cells or max(1, min(MAX_CELLS, round(math.sqrt(total))))
        cells = min(cells, total)
        rng = np.random.default_rng(seed)
        sample: list[np.ndarray] = []
        with self._lock:
            for seen, (blob,) in enumerate(self._db.execute("SELECT key FROM items ORDER BY id")):
                if len(sample) < MAX_FIT_KEYS:
                    sample.append(np.frombuffer(blob, dtype="<f4"))
                elif (slot := int(rng.integers(0, seen + 1))) < MAX_FIT_KEYS:
                    sample[slot] = np.frombuffer(blob, dtype="<f4")
        keys = np.stack(sample)
        centroids = fit_centroids(keys, cells, seed=seed)
        nprobe, curve = 1, {}
        if cells > 1:
            k = min(k, len(keys) - 1)
            nprobe, curve = calibrate_nprobe(
                keys, centroids, noisy_queries(keys, seed=seed), k=k, target_recall=target_recall
            )
        with self._lock, self._db:
            self._db.execute("DELETE FROM postings")
            cursor = self._db.execute("SELECT id, key FROM items ORDER BY id")
            while rows := cursor.fetchmany(4096):
                block = np.stack([np.frombuffer(blob, dtype="<f4") for _, blob in rows])
                cells_of = np.argmax(block @ centroids.T, axis=1)
                coded = [quantize(key) for key in block]
                self._db.executemany(
                    "INSERT INTO postings VALUES (?, ?, ?, ?)",
                    [
                        (int(cell), row, scale, code.tobytes())
                        for (row, _), cell, (scale, code) in zip(rows, cells_of, coded, strict=True)
                    ],
                )
            report = {
                "items": total,
                "cells": cells,
                "nprobe": nprobe,
                "target_recall": target_recall,
                "k": k,
                "overlap": {str(n): v for n, v in curve.items()},
            }
            self._set_meta("centroids", np.ascontiguousarray(centroids, dtype="<f4").tobytes())
            self._set_meta("nprobe", nprobe)
            self._set_meta("index", report)
        self._centroids = None
        return report

    def search(
        self,
        query: np.ndarray,
        limit: int,
        *,
        exclude: Iterable[int] = (),
        probes: int | None = None,
    ) -> list[Hit]:
        """The ``limit`` items of the probed cells with the highest exact cosine to ``query``.

        ``probes`` overrides the calibrated number of cells probed (all of them: exact).

        ``exclude`` rows are skipped before ranking, so the result is still ``limit``
        long when enough items remain. Deterministic: scores are float64, ties go
        to the lower row.
        """
        centroids = self._index()
        if centroids is None:
            if self.count:
                raise RuntimeError(f"Collection {self.name!r} has no index; run reindex first")
            return []
        q = unit(query, width=EMBEDDING_WIDTH).astype(np.float64)
        cells = [
            int(c) for c in np.argsort(-(centroids @ q), kind="stable")[: probes or self.nprobe]
        ]
        with self._lock:
            rows = self._db.execute(
                "SELECT item, scale, code FROM postings "
                f"WHERE cell IN ({','.join('?' * len(cells))})",
                cells,
            ).fetchall()
            blocked = set(exclude)
            if blocked:
                rows = [row for row in rows if row[0] not in blocked]
            if not rows:
                return []
            ids, scales, codes = zip(*rows, strict=True)
            matrix = np.frombuffer(b"".join(codes), dtype=np.int8).reshape(len(rows), -1)
            ranked = rank_exact(
                np.asarray(ids, dtype=np.int64),
                np.asarray(scales, dtype=np.float64),
                matrix,
                q,
                limit,
                lambda chosen: self.keys(chosen.tolist()),
            )
            names = {
                row: item_id
                for item_id, row in self._rows_to_ids([row for row, _ in ranked]).items()
            }
        return [Hit(row, names[row], score) for row, score in ranked]

    def _rows_to_ids(self, rows: list[int]) -> dict[str, int]:
        found: dict[str, int] = {}
        for start in range(0, len(rows), _BATCH):
            batch = rows[start : start + _BATCH]
            found.update(
                self._db.execute(
                    f"SELECT item_id, id FROM items WHERE id IN ({','.join('?' * len(batch))})",
                    batch,
                )
            )
        return found

    def search_exact(
        self, query: np.ndarray, limit: int, *, exclude: Iterable[int] = ()
    ) -> list[Hit]:
        """The same search without the index: every item scored (the reference)."""
        q = unit(query, width=EMBEDDING_WIDTH).astype(np.float64)
        blocked = set(exclude)
        scored: list[tuple[float, int]] = []
        with self._lock:
            cursor = self._db.execute("SELECT id, key FROM items ORDER BY id")
            while rows := cursor.fetchmany(4096):
                keys = np.stack([np.frombuffer(blob, dtype="<f4") for _, blob in rows])
                scored += [
                    (float(score), row)
                    for (row, _), score in zip(rows, keys.astype(np.float64) @ q, strict=True)
                    if row not in blocked
                ]
            best = sorted(scored, key=lambda pair: (-pair[0], pair[1]))[:limit]
            names = {
                row: item_id for item_id, row in self._rows_to_ids([r for _, r in best]).items()
            }
        return [Hit(row, names[row], score) for score, row in best]

    def identity(self) -> str:
        """A digest of what the collection holds (every item identity and key), for receipts."""
        digest = hashlib.sha256()
        with self._lock:
            for item_id, key in self._db.execute("SELECT item_id, key FROM items ORDER BY id"):
                digest.update(canonical_json(item_id))
                digest.update(key)
        return digest.hexdigest()

    def info(self) -> dict[str, Any]:
        counts = dict(
            self._db.execute("SELECT modality, COUNT(*) FROM items GROUP BY modality ORDER BY 1")
        )
        return {
            "name": self.name,
            "teacher": self.teacher,
            "items": self.count,
            "modalities": counts,
            "threshold": self.threshold,
            "index": self._meta("index"),
            "indexed_items": self._db.execute("SELECT COUNT(*) FROM postings").fetchone()[0],
            "has_query_center": self._meta("query_center") is not None,
        }
