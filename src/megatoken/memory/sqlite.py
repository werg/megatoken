"""Single-writer canonical SQLite bank with typed IDs and independent generations.

Adapted from the transactional storage patterns of hindsight-kb@4501647;
all vector coordinates, provenance and optimizer rows are owned here.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import sqlite3
from threading import RLock
import uuid

import numpy as np

from .base import Record, Relation
from .contracts import Collection, Footprint, Revisions, SemanticSpace, SourceRange, unit_finite


def _vector(value, count: int) -> bytes:
    a = np.asarray(value, dtype="<f4")
    if a.shape != (count,) or not np.isfinite(a).all():
        raise ValueError("Wrong dimensionality or nonfinite vector")
    return a.tobytes()


def _unpack(value: bytes, count: int) -> np.ndarray:
    if value is None or len(value) != 4 * count:
        raise ValueError("Corrupt or incompatible little-endian FP32 vector")
    array = np.frombuffer(value, dtype="<f4").copy()
    if not np.isfinite(array).all():
        raise ValueError("Stored vector contains nonfinite coordinates")
    return array


_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    collection TEXT NOT NULL CHECK(collection IN ('source','memory')),
    scope TEXT NOT NULL,
    key BLOB NOT NULL, value BLOB,
    footprint TEXT NOT NULL,
    content_id TEXT NOT NULL DEFAULT '', asset_ref TEXT NOT NULL DEFAULT '',
    strength REAL NOT NULL, lifetime_logit REAL NOT NULL,
    birth_step INTEGER NOT NULL DEFAULT 0,
    group_id TEXT NOT NULL DEFAULT '', assignment TEXT NOT NULL DEFAULT '',
    source_span TEXT NOT NULL DEFAULT '',
    active INTEGER NOT NULL DEFAULT 1,
    key_revision INTEGER NOT NULL DEFAULT 1, payload_revision INTEGER NOT NULL DEFAULT 1,
    UNIQUE(collection, scope, content_id, assignment)
);
CREATE INDEX IF NOT EXISTS record_scope ON records(collection, scope);
CREATE TABLE IF NOT EXISTS relations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    left_id INTEGER NOT NULL REFERENCES records(id) ON DELETE CASCADE,
    right_id INTEGER NOT NULL REFERENCES records(id) ON DELETE CASCADE,
    scope TEXT NOT NULL,
    prior REAL NOT NULL, value BLOB NOT NULL, dependence BLOB NOT NULL,
    footprint TEXT NOT NULL, strength REAL NOT NULL, lifetime_logit REAL NOT NULL,
    payload_revision INTEGER NOT NULL DEFAULT 1,
    active INTEGER NOT NULL DEFAULT 1,
    UNIQUE(left_id, right_id)
);
CREATE TABLE IF NOT EXISTS source_range_refs (
    collection TEXT NOT NULL, entity_kind TEXT NOT NULL,
    entity_id INTEGER NOT NULL, source_id TEXT NOT NULL, source_version TEXT NOT NULL,
    byte_start INTEGER NOT NULL, byte_end INTEGER NOT NULL,
    PRIMARY KEY (collection, entity_kind, entity_id, source_id,
                 source_version, byte_start, byte_end)
);
CREATE INDEX IF NOT EXISTS range_reverse ON source_range_refs(source_id,source_version);
CREATE TABLE IF NOT EXISTS source_windows (
    window_id TEXT PRIMARY KEY, scope TEXT NOT NULL,
    footprint TEXT NOT NULL, source_ids TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS contributions (
    assignment_id TEXT PRIMARY KEY,
    source_content_id TEXT NOT NULL, record_id INTEGER NOT NULL REFERENCES records(id),
    footprint TEXT NOT NULL, original_key BLOB NOT NULL, initial_value BLOB NOT NULL,
    assignment_distance REAL NOT NULL, optimization_lineage TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS postings (
    collection TEXT NOT NULL, record_id INTEGER NOT NULL REFERENCES records(id) ON DELETE CASCADE,
    cell INTEGER NOT NULL, centroids_revision TEXT NOT NULL,
    key_revision INTEGER NOT NULL, PRIMARY KEY (collection, record_id)
);
CREATE INDEX IF NOT EXISTS posting_cell ON postings(collection, centroids_revision, cell);
CREATE TABLE IF NOT EXISTS optimizer_rows (
    collection TEXT NOT NULL, kind TEXT NOT NULL, entity_id INTEGER NOT NULL,
    field TEXT NOT NULL, layout_generation TEXT NOT NULL,
    first BLOB NOT NULL, second BLOB NOT NULL,
    updates INTEGER NOT NULL, step INTEGER NOT NULL, row_revision INTEGER NOT NULL,
    PRIMARY KEY(collection,kind,entity_id,field,layout_generation)
);
CREATE TABLE IF NOT EXISTS demand_events (
    event_id TEXT PRIMARY KEY, partition_id TEXT NOT NULL, region TEXT NOT NULL,
    family TEXT NOT NULL, step INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS growth_state (
    name TEXT PRIMARY KEY, content TEXT NOT NULL
);
"""


class SQLiteKB:
    SCHEMA_VERSION = 1

    def __init__(self, path: str | Path, space: SemanticSpace, *, value_dim: int,
                 readonly: bool = False):
        if not 1 <= value_dim <= 16384:
            raise ValueError("Invalid semantic-value width")
        self.path = Path(path)
        self.space = space
        self.value_dim = value_dim
        self.readonly = readonly
        self._lock = RLock()
        self._depth = 0
        if not readonly:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        uri = f"file:{self.path.resolve().as_posix()}?mode=ro&immutable=1" if readonly else None
        self.db = sqlite3.connect(uri if readonly else str(self.path), uri=readonly,
                                  isolation_level=None, timeout=10, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.execute("PRAGMA busy_timeout=10000")
        if not readonly:
            # FULL durability; DELETE remains portable to older Python/SQLite builds.
            self.db.execute("PRAGMA synchronous=FULL")
            self.db.executescript(_SCHEMA)
            with self.transaction():
                self._set_once("schema_version", str(self.SCHEMA_VERSION))
                self._set_once("space", json.dumps(asdict(space), sort_keys=True))
                self._set_once("value_dim", str(value_dim))
                self._set_once("database_id", uuid.uuid4().hex)
                for name in ("key", "index", "payload", "policy", "graph", "growth"):
                    self._set_once("generation_" + name, "0")
        self._verify_manifest()

    def _set_once(self, key: str, value: str) -> None:
        self.db.execute("INSERT OR IGNORE INTO meta VALUES (?,?)", (key, value))

    def _meta(self, key: str) -> str:
        row = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        if row is None:
            raise ValueError(f"Missing KB metadata {key}")
        return str(row[0])

    def _verify_manifest(self) -> None:
        if int(self._meta("schema_version")) != self.SCHEMA_VERSION or (
            json.loads(self._meta("space")) != asdict(self.space)
        ) or int(self._meta("value_dim")) != self.value_dim:
            raise ValueError("Incompatible KB space, schema, or semantic-value layout")

    @property
    def database_id(self) -> str:
        return self._meta("database_id")

    @contextmanager
    def transaction(self):
        if self.readonly:
            raise PermissionError("An immutable read snapshot cannot be mutated")
        with self._lock:
            name = f"kb_savepoint_{self._depth}"
            if self._depth:
                self.db.execute(f"SAVEPOINT {name}")
            else:
                self.db.execute("BEGIN IMMEDIATE")
            self._depth += 1
            try:
                yield self
            except BaseException:
                self._depth -= 1
                if self._depth:
                    self.db.execute(f"ROLLBACK TO SAVEPOINT {name}")
                    self.db.execute(f"RELEASE SAVEPOINT {name}")
                else:
                    self.db.execute("ROLLBACK")
                raise
            else:
                self._depth -= 1
                if self._depth:
                    self.db.execute(f"RELEASE SAVEPOINT {name}")
                else:
                    self.db.execute("COMMIT")

    def _bump(self, *generations: str) -> None:
        for name in set(generations):
            self.db.execute(
                "UPDATE meta SET value=CAST(value AS INTEGER)+1 WHERE key=?",
                ("generation_" + name,),
            )

    def revisions(self) -> Revisions:
        with self._lock:
            values = {name: int(self._meta("generation_" + name))
                      for name in ("key", "index", "payload", "policy", "graph", "growth")}
            return Revisions(**values)

    def _record(self, row: sqlite3.Row) -> Record:
        collection = Collection(row["collection"])
        return Record(
            row["id"], collection, row["scope"],
            _unpack(row["key"], self.space.dimension),
            Footprint.from_json(row["footprint"]), row["key_revision"],
            row["payload_revision"], row["content_id"], row["asset_ref"],
            None if row["value"] is None else _unpack(row["value"], self.value_dim),
            row["strength"], row["lifetime_logit"], row["birth_step"],
            row["group_id"], row["assignment"], bool(row["active"]),
            None if not row["source_span"] else SourceRange(**{
                **json.loads(row["source_span"]),
                "ancestors": tuple(json.loads(row["source_span"]).get("ancestors", ())),
            }),
        )

    def scan(self, collection: Collection, scope: str) -> tuple[Record, ...]:
        with self._lock:
            rows = self.db.execute(
                "SELECT * FROM records WHERE collection=? AND scope=? ORDER BY id",
                (collection.value, scope),
            ).fetchall()
            return tuple(self._record(row) for row in rows)

    def get(self, collection: Collection, record_id: int) -> Record:
        with self._lock:
            row = self.db.execute(
                "SELECT * FROM records WHERE id=? AND collection=?",
                (record_id, collection.value),
            ).fetchone()
            if row is None:
                raise KeyError((collection.value, record_id))
            return self._record(row)

    def _insert(self, collection: Collection, scope: str, footprint: Footprint, key,
                *, value=None, content_id: str = "", asset_ref: str = "",
                strength: float = 1, lifetime_logit: float = -10, birth_step: int = 0,
                group: str = "", assignment: str = "",
                evidence_span: SourceRange | None = None) -> int:
        if not scope or any(r.scope != scope for r in footprint.ranges):
            raise ValueError("Every support range must belong to the record scope")
        if not all(math.isfinite(x) for x in (strength, lifetime_logit)) or strength < 0:
            raise ValueError("Invalid learned memory gate/lifetime")
        if birth_step < 0:
            raise ValueError("Invalid record birth step")
        if collection is Collection.SOURCE:
            if not content_id or not asset_ref or value is not None:
                raise ValueError("A source key requires immutable content identity/ref")
            if evidence_span is None:
                if len(footprint.ranges) != 1:
                    raise ValueError("Contextual source keys require an explicit byte evidence span")
                evidence_span = footprint.ranges[0]
            if evidence_span not in footprint.ranges:
                raise ValueError("Delivered source span must belong to full key provenance")
        elif value is None or not assignment:
            raise ValueError("A free memory point needs value and stable source assignment")
        k = unit_finite(key, dimension=self.space.dimension).tobytes()
        v = None if value is None else _vector(value, self.value_dim)
        with self.transaction():
            cursor = self.db.execute(
                """INSERT INTO records
                (collection,scope,key,value,footprint,content_id,asset_ref,strength,
                 lifetime_logit,birth_step,group_id,assignment,source_span)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (collection.value, scope, k, v, footprint.to_json(), content_id, asset_ref,
                 strength, lifetime_logit, birth_step, group, assignment,
                 "" if evidence_span is None else json.dumps(asdict(evidence_span), sort_keys=True)),
            )
            record_id = int(cursor.lastrowid)
            self._footprint_refs(collection.value, "record", record_id, footprint)
            self._bump("key", "policy", *(("payload", "growth") if value is not None else ()))
            return record_id

    def _footprint_refs(self, collection: str, kind: str, identity: int,
                        footprint: Footprint) -> None:
        self.db.executemany(
            """INSERT INTO source_range_refs VALUES (?,?,?,?,?,?,?)""",
            [
                (collection, kind, identity, r.source_id, r.version, r.start, r.end)
                for r in footprint.ranges
            ],
        )

    def tombstone_source_version(self, source_id: str, version: str) -> dict[str, int]:
        """Revoke every context-dependent source, free point and relation atomically."""
        if not source_id or not version:
            raise ValueError("Source invalidation requires both source and version")
        with self.transaction():
            rows = self.db.execute(
                """SELECT DISTINCT collection,entity_kind,entity_id
                   FROM source_range_refs WHERE source_id=? AND source_version=?""",
                (source_id, version),
            ).fetchall()
            counts = {"source": 0, "memory": 0, "relation": 0}
            revoked_records: set[int] = set()
            for row in rows:
                identity = int(row["entity_id"])
                if row["entity_kind"] == "record":
                    changed = self.db.execute(
                        "UPDATE records SET active=0 WHERE id=? AND active=1",
                        (identity,),
                    ).rowcount
                    counts[row["collection"]] += changed
                    if changed:
                        revoked_records.add(identity)
                        # IVF must never retain a live-looking posting for a
                        # retired key, even if a stale candidate cache survives.
                        self.db.execute(
                            "DELETE FROM postings WHERE record_id=?", (identity,)
                        )
                else:
                    counts["relation"] += self.db.execute(
                        "UPDATE relations SET active=0 WHERE id=? AND active=1",
                        (identity,),
                    ).rowcount
            # A relation is invalid whenever either member is retired, even
            # if its own source-window footprint omitted a member's contextual
            # source (older schema/partial migration). Preserve the tombstoned
            # row for audit; never offer it to an active graph reader.
            if revoked_records:
                counts["relation"] += self.db.execute(
                    """UPDATE relations SET active=0 WHERE active=1 AND
                    (left_id IN (SELECT id FROM records WHERE active=0) OR
                     right_id IN (SELECT id FROM records WHERE active=0))"""
                ).rowcount
            if any(counts.values()):
                self._bump("policy", "graph", "index")
            return counts

    def insert_source(self, *, scope: str, footprint: Footprint, key,
                      content_id: str, asset_ref: str,
                      evidence_span: SourceRange | None = None) -> int:
        return self._insert(Collection.SOURCE, scope, footprint, key,
                            content_id=content_id, asset_ref=asset_ref,
                            evidence_span=evidence_span)

    def insert_memory(self, *, scope: str, footprint: Footprint, key, value,
                      assignment: str, birth_step: int, group: str = "",
                      strength: float = 0.1, lifetime_logit: float = -10) -> int:
        return self._insert(Collection.MEMORY, scope, footprint, key, value=value,
                            assignment=assignment, birth_step=birth_step, group=group,
                            strength=strength, lifetime_logit=lifetime_logit)

    def update_record(self, collection: Collection, record_id: int, *,
                      key=None, value=None, strength=None, lifetime_logit=None,
                      expected_key_revision: int | None = None,
                      expected_payload_revision: int | None = None,
                      posting_cell: int | None = None, centroids_revision: str = "") -> Record:
        with self.transaction():
            old = self.get(collection, record_id)
            if expected_key_revision is not None and old.key_revision != expected_key_revision:
                raise RuntimeError("Stale KB key revision")
            if expected_payload_revision is not None and (
                old.payload_revision != expected_payload_revision
            ):
                raise RuntimeError("Stale KB payload revision")
            new_key = old.key if key is None else unit_finite(
                key, dimension=self.space.dimension
            )
            new_value = old.value if value is None else (
                _unpack(_vector(value, self.value_dim), self.value_dim)
            )
            if collection is Collection.SOURCE and value is not None:
                raise ValueError("A source value must remain exact immutable bytes")
            st = old.strength if strength is None else float(strength)
            lt = old.lifetime_logit if lifetime_logit is None else float(lifetime_logit)
            if not math.isfinite(st) or st < 0 or not math.isfinite(lt):
                raise ValueError("Invalid value controls")
            key_changed = not np.array_equal(new_key, old.key)
            payload_changed = (
                (new_value is not None and not np.array_equal(new_value, old.value))
                or st != old.strength or lt != old.lifetime_logit
            )
            if key_changed or payload_changed:
                self.db.execute(
                    """UPDATE records SET key=?,value=?,strength=?,lifetime_logit=?,
                    key_revision=?,payload_revision=? WHERE id=?""",
                    (new_key.tobytes(), None if new_value is None else new_value.tobytes(),
                     st, lt, old.key_revision + int(key_changed),
                     old.payload_revision + int(payload_changed), record_id),
                )
                self._bump(*(["key"] if key_changed else []),
                           *(["payload"] if payload_changed else []))
            if posting_cell is not None:
                if posting_cell < 0 or not centroids_revision:
                    raise ValueError("Postings require a pinned IVF centroid revision")
                self.db.execute(
                    """INSERT INTO postings VALUES (?,?,?,?,?)
                    ON CONFLICT(collection,record_id) DO UPDATE SET
                    cell=excluded.cell,centroids_revision=excluded.centroids_revision,
                    key_revision=excluded.key_revision""",
                    (collection.value, record_id, posting_cell, centroids_revision,
                     old.key_revision + int(key_changed)),
                )
                self._bump("index")
            elif key_changed:
                self.db.execute("DELETE FROM postings WHERE record_id=?", (record_id,))
                self._bump("index")
            return self.get(collection, record_id)

    def insert_relation(self, *, scope: str, left: int, right: int, prior: float,
                        value, dependence_logits: tuple[float, float],
                        footprint: Footprint, strength: float = 0.1,
                        lifetime_logit: float = -10) -> int:
        if not 0 <= prior <= 1 or len(dependence_logits) != 2 or (
            not all(math.isfinite(x) for x in (*dependence_logits, strength, lifetime_logit))
        ):
            raise ValueError("Invalid relation probability/control")
        with self.transaction():
            a, b = sorted((left, right))
            if a == b or any(
                self.get(Collection.MEMORY, x).scope != scope for x in (a, b)
            ) or any(r.scope != scope for r in footprint.ranges):
                raise ValueError("Pair requires distinct eligible memory members and support")
            result = self.db.execute(
                """INSERT INTO relations (left_id,right_id,scope,prior,value,dependence,
                footprint,strength,lifetime_logit) VALUES (?,?,?,?,?,?,?,?,?)""",
                (a, b, scope, prior, _vector(value, self.value_dim),
                 _vector(dependence_logits, 2), footprint.to_json(), strength, lifetime_logit),
            )
            relation_id = int(result.lastrowid)
            self._footprint_refs(Collection.MEMORY.value, "relation", relation_id, footprint)
            self._bump("graph", "payload", "growth")
            return relation_id

    def relations(self, scope: str) -> tuple[Relation, ...]:
        with self._lock:
            rows = self.db.execute(
                "SELECT * FROM relations WHERE scope=? AND active=1 ORDER BY id",
                (scope,),
            ).fetchall()
        return tuple(Relation(
            row["id"], row["left_id"], row["right_id"], row["scope"], row["prior"],
            _unpack(row["value"], self.value_dim),
            tuple(float(x) for x in _unpack(row["dependence"], 2)),
            Footprint.from_json(row["footprint"]), row["strength"], row["lifetime_logit"],
            row["payload_revision"],
        ) for row in rows)

    def get_relation(self, relation_id: int) -> Relation:
        with self._lock:
            row = self.db.execute("SELECT * FROM relations WHERE id=?", (relation_id,)
                                  ).fetchone()
        if row is None:
            raise KeyError(relation_id)
        return Relation(
            row["id"], row["left_id"], row["right_id"], row["scope"], row["prior"],
            _unpack(row["value"], self.value_dim),
            tuple(float(x) for x in _unpack(row["dependence"], 2)),
            Footprint.from_json(row["footprint"]), row["strength"], row["lifetime_logit"],
            row["payload_revision"],
        )

    def update_relation(self, relation_id: int, *, value=None, dependence_logits=None,
                        strength=None, lifetime_logit=None) -> None:
        with self.transaction():
            row = self.db.execute("SELECT * FROM relations WHERE id=?",
                                  (relation_id,)).fetchone()
            if row is None:
                raise KeyError(relation_id)
            v = row["value"] if value is None else _vector(value, self.value_dim)
            d = row["dependence"] if dependence_logits is None else _vector(dependence_logits, 2)
            s = row["strength"] if strength is None else float(strength)
            l = row["lifetime_logit"] if lifetime_logit is None else float(lifetime_logit)
            if not math.isfinite(s) or s < 0 or not math.isfinite(l):
                raise ValueError("Invalid relation strength or lifetime")
            self.db.execute(
                """UPDATE relations SET value=?,dependence=?,strength=?,lifetime_logit=?,
                payload_revision=payload_revision+1 WHERE id=?""",
                (v, d, s, l, relation_id),
            )
            self._bump("graph", "payload")

    def write_moments(self, *, collection: Collection, kind: str, entity_id: int,
                      field: str, layout_generation: str, first, second,
                      updates: int, step: int, row_revision: int) -> None:
        if kind not in {"record", "relation"} or not field or not layout_generation:
            raise ValueError("Sparse moments require an unambiguous typed field identity")
        if min(updates, step, row_revision) < 0:
            raise ValueError("Invalid optimizer chronology")
        a = np.asarray(first, dtype="<f4").reshape(-1)
        b = np.asarray(second, dtype="<f4").reshape(-1)
        if a.shape != b.shape:
            raise ValueError("Sparse optimizer moments must match")
        with self.transaction():
            self.db.execute(
                """INSERT INTO optimizer_rows VALUES (?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(collection,kind,entity_id,field,layout_generation) DO UPDATE SET
                first=excluded.first,second=excluded.second,updates=excluded.updates,
                step=excluded.step,row_revision=excluded.row_revision""",
                (collection.value, kind, entity_id, field, layout_generation,
                 _vector(a, a.size), _vector(b, b.size), updates, step, row_revision),
            )

    def moments(self, *, collection: Collection, kind: str, entity_id: int,
                field: str, layout_generation: str):
        with self._lock:
            row = self.db.execute(
                """SELECT * FROM optimizer_rows WHERE collection=? AND kind=? AND
                   entity_id=? AND field=? AND layout_generation=?""",
                (collection.value, kind, entity_id, field, layout_generation),
            ).fetchone()
            if row is None:
                return None
            n = len(row["first"]) // 4
            return (_unpack(row["first"], n), _unpack(row["second"], n),
                    row["updates"], row["step"], row["row_revision"])

    def record_demand(self, event_id: str, partition: str, region: str,
                      family: str, step: int) -> bool:
        with self.transaction():
            cursor = self.db.execute(
                "INSERT OR IGNORE INTO demand_events VALUES (?,?,?,?,?)",
                (event_id, partition, region, family, step),
            )
            return cursor.rowcount == 1

    def growth_state(self, name: str, default=None):
        with self._lock:
            row = self.db.execute("SELECT content FROM growth_state WHERE name=?", (name,)
                                  ).fetchone()
        return default if row is None else json.loads(row[0])

    def put_growth_state(self, name: str, value) -> None:
        with self.transaction():
            self.db.execute(
                """INSERT INTO growth_state VALUES (?,?) ON CONFLICT(name)
                   DO UPDATE SET content=excluded.content""",
                (name, json.dumps(value, sort_keys=True)),
            )
            self._bump("growth")

    def backup(self, destination: str | Path) -> Path:
        """SQLite online backup includes all committed WAL/journal state."""
        destination = Path(destination)
        if destination.exists():
            raise FileExistsError("Never overwrite a published KB snapshot")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + "." + uuid.uuid4().hex + ".tmp")
        try:
            with self._lock, sqlite3.connect(temporary) as out:
                self.db.backup(out)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        return destination

    def close(self) -> None:
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
