"""DiLoCo row-sync protocol for the free semantic KB.

Neural parameters synchronize as dense fragments. Free-KB rows are sparse,
dynamically allocated SQLite records, so they need their own protocol:

* **Ownership.** The coordinator holds the ONE canonical bank. Only it allocates
  rows (growth) and assigns IDs; learners never grow locally. Learners forward
  query demand, which the coordinator records and grows from at merge time.
* **Pseudo-gradients.** Each learner trains a local copy of the published bank
  with its own sparse Adam moments. Every ``local_steps`` it uploads, for each
  touched row field, ``baseline - current`` where the baseline is the value at
  the canonical version it last installed.
* **Merge.** Per row field, contributions from the learners that touched it are
  merged with the same token/staleness weighting as fragments, then applied with
  per-row outer Nesterov momentum. Rows retired or missing on the canonical
  bank are skipped (their updates are stale by definition).
* **Staleness and duplicates.** Submissions carry (worker, session, sequence)
  and a base version. A committed per-worker sequence ledger rejects replays;
  bases older than ``max_staleness`` versions are refused with a 409 so the
  learner reinstalls the current bank.
* **Publication.** Row updates, demand, growth, the version and the sequence
  ledger commit in one SQLite transaction. Triggers on the canonical bank log
  every changed row's primary key under the version being produced, so a
  learner at version ``v`` pulls only rows changed since ``v`` (cost scales
  with the merge, not the bank). Learners apply them between optimizer steps,
  keeping their own sparse Adam moments. A full stripped snapshot is built
  lazily for bootstrap and for learners older than the retained history.
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from megatoken.io import sha256_file, write_json
from megatoken.memory.contracts import Collection
from megatoken.memory.optimizer import RowID
from megatoken.memory.sqlite import SQLiteKB
from megatoken.training.diloco_protocol import (
    load_tensors,
    merge_pseudogradients,
    outer_nesterov,
    save_tensors,
)

RECORD_FIELDS = ("key", "value", "strength", "lifetime_logit")
# Canonical tables mirrored to learners, parents before children. Learner-local
# optimizer moments and coordinator-only outer state are never synchronized.
SYNCED_TABLES = (
    "meta",
    "records",
    "relations",
    "source_range_refs",
    "source_windows",
    "contributions",
    "postings",
    "retired_source_versions",
    "demand_events",
    "growth_state",
)
COORDINATOR_TABLES = ("diloco_kb_outer", "diloco_kb_meta", "diloco_kb_changes")


class KBChangesExpired(LookupError):
    """The requested base version predates the retained change history."""


RELATION_FIELDS = ("value", "dependence_logits", "strength", "lifetime_logit")
MAX_DEMAND = 1 << 20


def row_key(identity: RowID) -> str:
    return "|".join(
        (
            identity.collection.value,
            identity.kind,
            str(identity.id),
            identity.field,
            identity.layout,
        )
    )


def parse_row_key(key: str) -> RowID:
    parts = key.split("|")
    if len(parts) != 5:
        raise ValueError("Malformed KB row key")
    collection, kind, identity, field, layout = parts
    if (
        kind == "record"
        and field not in RECORD_FIELDS
        or (kind == "relation" and field not in RELATION_FIELDS)
        or kind not in {"record", "relation"}
        or not identity.isdigit()
    ):
        raise ValueError("Unsupported KB row field")
    return RowID(Collection(collection), kind, int(identity), field, layout)


def read_field(bank: SQLiteKB, identity: RowID) -> np.ndarray | None:
    """Current FP32 field value, or None when the row is missing or retired."""
    try:
        row = (
            bank.get(identity.collection, identity.id)
            if identity.kind == "record"
            else bank.get_relation(identity.id)
        )
    except KeyError:
        return None
    if not row.active:
        return None
    value = getattr(row, identity.field)
    if value is None:
        return None
    return np.asarray(value, dtype=np.float32).reshape(-1).copy()


def write_field(bank: SQLiteKB, identity: RowID, values: np.ndarray, ivf: Any = None) -> None:
    if identity.field == "strength":
        # Outer momentum may overshoot; strength is a nonnegative mass.
        values = np.maximum(values, 0)
    replacement = values if values.size > 1 else float(values[0])
    if identity.kind == "record":
        index = {}
        if ivf is not None and identity.field == "key":
            index = {
                "posting_cell": ivf.cell(values),
                "centroids_revision": ivf.identity.centroid_sha256,
            }
        bank.update_record(
            identity.collection, identity.id, **{identity.field: replacement}, **index
        )
    else:
        bank.update_relation(identity.id, **{identity.field: replacement})


def _metadata(path: Path) -> dict[str, str]:
    from safetensors import safe_open

    with safe_open(str(path), framework="pt", device="cpu") as file:
        return file.metadata() or {}


def save_kb_update(
    path: Path,
    rows: dict[str, np.ndarray],
    demand: list[tuple[str, str, str, int]],
    *,
    wire_dtype: str,
    metadata: dict[str, str] | None = None,
) -> None:
    keys = sorted(rows)
    flat = torch.from_numpy(np.concatenate([rows[k] for k in keys])) if keys else torch.zeros(0)
    save_tensors(
        path,
        {"delta": flat.to(torch.bfloat16 if wire_dtype == "bf16" else torch.float32)},
        metadata={
            "rows": json.dumps([[k, int(rows[k].size)] for k in keys]),
            "demand": json.dumps([list(item) for item in demand]),
            **(metadata or {}),
        },
    )


@dataclass(frozen=True)
class KBUpdate:
    rows: dict[str, torch.Tensor]
    demand: tuple[tuple[str, str, str, int], ...]


def load_kb_update(path: Path) -> KBUpdate:
    meta = _metadata(path)
    tensors = load_tensors(path)
    if tensors.keys() != {"delta"}:
        raise ValueError("Expected exactly one KB delta tensor")
    flat = tensors["delta"]
    if (
        flat.dtype not in (torch.float32, torch.bfloat16)
        or flat.ndim != 1
        or (not torch.isfinite(flat).all())
    ):
        raise ValueError("KB delta dtype, shape, or finiteness invalid")
    layout = json.loads(meta.get("rows", "[]"))
    demand = json.loads(meta.get("demand", "[]"))
    if not isinstance(layout, list) or not isinstance(demand, list) or len(demand) > MAX_DEMAND:
        raise ValueError("Malformed KB update metadata")
    rows: dict[str, torch.Tensor] = {}
    cursor = 0
    for key, size in layout:
        parse_row_key(key)
        if type(size) is not int or size < 1 or key in rows:
            raise ValueError("Malformed KB update row layout")
        rows[key] = flat[cursor : cursor + size].float()
        cursor += size
    if cursor != flat.numel():
        raise ValueError("KB update row layout does not cover the delta")
    parsed = []
    for item in demand:
        if (
            not isinstance(item, list)
            or len(item) != 4
            or not all(isinstance(x, str) for x in item[:3])
            or type(item[3]) is not int
        ):
            raise ValueError("Malformed KB query demand entry")
        parsed.append((item[0], item[1], item[2], item[3]))
    return KBUpdate(rows, tuple(parsed))


def _primary_key(db: sqlite3.Connection, table: str, schema: str = "main") -> list[str]:
    info = db.execute(f"PRAGMA {schema}.table_info({table})").fetchall()
    return [row[1] for row in sorted(info, key=lambda r: r[5]) if row[5] > 0]


def _columns(db: sqlite3.Connection, table: str, schema: str = "main") -> list[str]:
    return [row[1] for row in db.execute(f"PRAGMA {schema}.table_info({table})")]


def _install_change_log(db: sqlite3.Connection) -> None:
    db.execute(
        """CREATE TABLE IF NOT EXISTS diloco_kb_changes (
               version INTEGER NOT NULL, tbl TEXT NOT NULL, pk TEXT NOT NULL,
               deleted INTEGER NOT NULL, PRIMARY KEY (version, tbl, pk))"""
    )
    db.execute("CREATE INDEX IF NOT EXISTS diloco_kb_change_version ON diloco_kb_changes(version)")
    producing = (
        "COALESCE((SELECT CAST(content AS INTEGER) FROM diloco_kb_meta WHERE name='version'),0)+1"
    )
    for table in SYNCED_TABLES:
        key = _primary_key(db, table)
        if not key:
            raise RuntimeError(f"Synchronized KB table {table} lacks a primary key")
        for op, alias, deleted in (
            ("INSERT", "NEW", 0),
            ("UPDATE", "NEW", 0),
            ("DELETE", "OLD", 1),
        ):
            values = ",".join(f"{alias}.{column}" for column in key)
            db.execute(
                f"""CREATE TRIGGER IF NOT EXISTS diloco_kb_{op.lower()}_{table}
                    AFTER {op} ON {table} BEGIN
                    INSERT OR REPLACE INTO diloco_kb_changes VALUES (
                        {producing}, '{table}', json_array({values}), {deleted});
                    END"""
            )


def strip_snapshot(path: Path) -> None:
    """Published snapshots carry no coordinator state or learner optimizer moments."""
    connection = sqlite3.connect(path, isolation_level=None)
    try:
        for (trigger,) in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='trigger' AND name LIKE 'diloco_kb_%'"
        ).fetchall():
            connection.execute(f"DROP TRIGGER {trigger}")
        for table in COORDINATOR_TABLES:
            connection.execute(f"DROP TABLE IF EXISTS {table}")
        connection.execute("DELETE FROM optimizer_rows")
        connection.execute("VACUUM")
    finally:
        connection.close()


def install_snapshot(downloaded: Path, local: Path) -> None:
    """Replace the learner's bank, keeping its own sparse optimizer moments.

    The caller must have closed every connection to ``local``.
    """
    connection = sqlite3.connect(downloaded, isolation_level=None)
    try:
        if local.exists():
            connection.execute("ATTACH DATABASE ? AS previous", (str(local),))
            connection.execute("BEGIN")
            connection.execute("DELETE FROM optimizer_rows")
            connection.execute("INSERT INTO optimizer_rows SELECT * FROM previous.optimizer_rows")
            connection.execute("COMMIT")
            connection.execute("DETACH DATABASE previous")
    finally:
        connection.close()
    os.replace(downloaded, local)


def apply_changes(bank: SQLiteKB, path: Path) -> tuple[int, frozenset[tuple[str, int]]]:
    """Bring a learner bank to the exported canonical version in one transaction.

    Returns the version and the (kind, id) rows the export overwrote.

    The export holds every row changed since the learner's base plus every row
    the learner trained, so afterwards all synchronized tables equal a full
    snapshot, while learner-local optimizer moments are kept. Rows are
    upserted by primary key, never REPLACEd: that would cascade deletes
    through foreign keys.
    """
    db = bank.db
    with bank._lock:
        db.execute("ATTACH DATABASE ? AS delta", (str(path),))
        try:
            version = int(db.execute("SELECT version FROM delta._version").fetchone()[0])
            present = {
                row[0]
                for row in db.execute("SELECT name FROM delta.sqlite_master WHERE type='table'")
            }
            overwritten = frozenset(
                (kind, int(row[0]))
                for kind, table in (("record", "records"), ("relation", "relations"))
                if table in present
                for row in db.execute(f"SELECT id FROM delta.{table}")
            )
            with bank.transaction():
                db.execute("PRAGMA defer_foreign_keys=ON")
                for table in SYNCED_TABLES:
                    if table not in present:
                        continue
                    columns = _columns(db, table)
                    key = _primary_key(db, table)
                    names = ",".join(columns)
                    rest = [c for c in columns if c not in key]
                    conflict = (
                        "DO UPDATE SET " + ",".join(f"{c}=excluded.{c}" for c in rest)
                        if rest
                        else "DO NOTHING"
                    )
                    db.execute(
                        f"""INSERT INTO main.{table}({names}) SELECT {names}
                            FROM delta.{table} WHERE true
                            ON CONFLICT({",".join(key)}) {conflict}"""
                    )
                for table, pk in db.execute("SELECT tbl, pk FROM delta._deleted").fetchall():
                    if table not in SYNCED_TABLES:
                        raise ValueError("KB change set deletes an unsynchronized table")
                    db.execute(
                        f"DELETE FROM main.{table} WHERE "
                        + " AND ".join(f"{c}=?" for c in _primary_key(db, table)),
                        json.loads(pk),
                    )
        finally:
            db.execute("DETACH DATABASE delta")
    return version, overwritten


def initialize_kb(
    root: Path,
    *,
    recipe_path: str,
    bank_path: str,
    centroids_path: str | None,
    centroids_sha256: str | None,
    nprobe: int,
    scopes: tuple[str, ...],
    growth_budget: int,
) -> dict:
    """Copy the gate-ready source-populated bank into coordinator ownership."""
    from megatoken.memory.recipe import RetrievalRecipe, open_bank_for_recipe

    recipe = RetrievalRecipe.load(recipe_path)
    if recipe.phase not in {"progressive_warmup", "free_kb_pretraining"}:
        raise ValueError("A coordinator KB is only for live free-KB phases")
    if not scopes or growth_budget < 0:
        raise ValueError("Coordinator KB needs retrieval scopes and a bounded growth budget")
    directory = root / "kb"
    directory.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(recipe_path, directory / "recipe.json")
    if centroids_path:
        if sha256_file(Path(centroids_path)) != centroids_sha256:
            raise ValueError("Learned IVF centroid artifact SHA256 mismatch")
        shutil.copyfile(centroids_path, directory / "centroids.npy")
    with open_bank_for_recipe(recipe, bank_path, readonly=True) as source:
        ready = source.growth_state("readiness", {})
        if not (ready.get("coherence") and ready.get("embeddings")):
            raise RuntimeError("Free-KB training requires verified coherence/embedding readiness")
        database_id = source.database_id
        source.backup(directory / "canonical.sqlite")
    settings = {
        "database_id": database_id,
        "recipe_fingerprint": recipe.fingerprint,
        "centroids_sha256": centroids_sha256 if centroids_path else None,
        "nprobe": nprobe,
        "scopes": sorted(set(scopes)),
        "growth_budget": growth_budget,
    }
    # Opening once creates the coordinator tables and publishes version 0.
    KBCoordinator(
        root,
        settings,
        outer_lr=1.0,
        outer_momentum=0.0,
        max_staleness=0,
        min_quorum=1,
        workers=1,
        grace_seconds=0,
    ).close()
    return settings


class KBCoordinator:
    """Single canonical bank writer behind the fragment coordinator's lock."""

    def __init__(
        self,
        root: Path,
        settings: dict,
        *,
        outer_lr: float,
        outer_momentum: float,
        max_staleness: int,
        min_quorum: int,
        workers: int,
        grace_seconds: float,
    ) -> None:
        from megatoken.memory.growth import GrowthAllocator
        from megatoken.memory.recipe import RetrievalRecipe, open_bank_for_recipe

        self.root = Path(root) / "kb"
        self.settings = settings
        self.outer_lr, self.outer_momentum = outer_lr, outer_momentum
        self.max_staleness, self.min_quorum = max_staleness, min_quorum
        self.workers, self.grace_seconds = workers, grace_seconds
        self.recipe = RetrievalRecipe.load(self.root / "recipe.json")
        if self.recipe.fingerprint != settings["recipe_fingerprint"]:
            raise ValueError("Coordinator KB recipe changed")
        self.bank = open_bank_for_recipe(self.recipe, self.root / "canonical.sqlite")
        if self.bank.database_id != settings["database_id"]:
            raise ValueError("Coordinator KB database identity changed")
        self.bank.db.executescript(
            """CREATE TABLE IF NOT EXISTS diloco_kb_meta (
                   name TEXT PRIMARY KEY, content TEXT NOT NULL);
               CREATE TABLE IF NOT EXISTS diloco_kb_outer (
                   row TEXT PRIMARY KEY, momentum BLOB NOT NULL);"""
        )
        _install_change_log(self.bank.db)
        self.retained_versions = int(settings.get("retained_versions", 1024))
        self.growth = GrowthAllocator(self.bank, self.recipe.growth)
        self.ivf = None
        if settings.get("centroids_sha256"):
            from megatoken.memory.disk_ivf import SQLiteDiskIVF

            centroids = self.root / "centroids.npy"
            if sha256_file(centroids) != settings["centroids_sha256"]:
                raise ValueError("Coordinator IVF centroids changed")
            self.ivf = SQLiteDiskIVF(
                self.bank, np.load(centroids, allow_pickle=False), nprobe=settings["nprobe"]
            )
        self.quorum_since: float | None = None
        self.merges = 0

    def close(self) -> None:
        self.bank.close()

    def _meta(self, name: str, default: Any) -> Any:
        row = self.bank.db.execute(
            "SELECT content FROM diloco_kb_meta WHERE name=?", (name,)
        ).fetchone()
        return default if row is None else json.loads(row[0])

    def _put_meta(self, name: str, value: Any) -> None:
        self.bank.db.execute(
            """INSERT INTO diloco_kb_meta VALUES (?,?) ON CONFLICT(name)
               DO UPDATE SET content=excluded.content""",
            (name, json.dumps(value, sort_keys=True)),
        )

    def version(self) -> int:
        return int(self._meta("version", 0))

    @property
    def pending(self) -> Path:
        return self.root / "pending"

    def published(self) -> tuple[int, Path]:
        """Full snapshot of the current version, built lazily (bootstrap/fallback)."""
        self.sync_public()
        pointer = json.loads((self.root / "public.json").read_text())
        return int(pointer["version"]), self.root / pointer["file"]

    def export_changes(
        self,
        after: int,
        destination: Path,
        *,
        records: tuple[int, ...] = (),
        relations: tuple[int, ...] = (),
    ) -> int | None:
        """Write rows changed since ``after`` plus the learner's trained rows.

        ``records``/``relations`` are the IDs the learner modified locally; their
        canonical contents (and postings) overwrite the local values. Returns
        None when already current and nothing was trained.
        """
        version = self.version()
        if type(after) is not int or after > version:
            raise ValueError("Unknown KB base version")
        if after == version and not (records or relations):
            return None
        if after < int(self._meta("changes_floor", 0)):
            raise KBChangesExpired("KB base predates retained changes; fetch the snapshot")
        latest: dict[tuple[str, str], int] = {}
        for table, pk, deleted in self.bank.db.execute(
            """SELECT tbl, pk, deleted FROM diloco_kb_changes WHERE version > ?
               ORDER BY version""",
            (after,),
        ):
            latest[(table, pk)] = deleted
        db = self.bank.db
        for table, ids in (("records", records), ("relations", relations)):
            for identity in ids:
                if type(identity) is not int:
                    raise ValueError("Trained KB row IDs must be integers")
                latest.setdefault((table, json.dumps([identity])), 0)
        for identity in records:
            for (collection,) in db.execute(
                "SELECT collection FROM postings WHERE record_id=?", (identity,)
            ):
                latest.setdefault(("postings", json.dumps([collection, identity])), 0)
        # meta holds tiny generation counters that learners bump locally.
        for (name,) in db.execute("SELECT key FROM meta"):
            latest.setdefault(("meta", json.dumps([name])), 0)
        destination.unlink(missing_ok=True)
        db.execute("ATTACH DATABASE ? AS delta", (str(destination),))
        try:
            db.execute("BEGIN")
            db.execute("CREATE TABLE delta._version (version INTEGER NOT NULL)")
            db.execute("INSERT INTO delta._version VALUES (?)", (version,))
            db.execute("CREATE TABLE delta._deleted (tbl TEXT NOT NULL, pk TEXT NOT NULL)")
            by_table: dict[str, list[list]] = {}
            for (table, pk), deleted in latest.items():
                if deleted:
                    db.execute("INSERT INTO delta._deleted VALUES (?,?)", (table, pk))
                else:
                    by_table.setdefault(table, []).append(json.loads(pk))
            for table in SYNCED_TABLES:
                if table not in by_table:
                    continue
                key = _primary_key(db, table)
                db.execute(f"CREATE TABLE delta.{table} AS SELECT * FROM main.{table} WHERE 0")
                db.executemany(
                    f"INSERT INTO delta.{table} SELECT * FROM main.{table} WHERE "
                    + " AND ".join(f"{c}=?" for c in key),
                    by_table[table],
                )
                if table in {"records", "relations"}:
                    # A trained row retired or never present canonically: drop it.
                    found = {row[0] for row in db.execute(f"SELECT id FROM delta.{table}")}
                    for (identity,) in by_table[table]:
                        if identity not in found:
                            db.execute(
                                "INSERT INTO delta._deleted VALUES (?,?)",
                                (table, json.dumps([identity])),
                            )
            db.execute("COMMIT")
        except BaseException:
            db.execute("ROLLBACK")
            raise
        finally:
            db.execute("DETACH DATABASE delta")
        return version

    def sync_public(self) -> None:
        """Build the full snapshot for the current version if it is stale."""
        version = self.version()
        pointer = self.root / "public.json"
        if pointer.exists() and json.loads(pointer.read_text())["version"] == version:
            return
        name = f"public-{version:09d}.sqlite"
        target = self.root / name
        target.unlink(missing_ok=True)
        self.bank.backup(target)
        strip_snapshot(target)
        write_json(pointer, {"version": version, "file": name, "sha256": sha256_file(target)})
        for old in self.root.glob("public-*.sqlite"):
            if old.name != name:
                old.unlink(missing_ok=True)

    def submit(
        self,
        worker: int,
        sequence: int,
        base_version: int,
        steps: int,
        tokens: int,
        incoming: Path,
        conflict: type[Exception],
    ) -> dict:
        if (
            type(sequence) is not int
            or sequence < 1
            or type(base_version) is not int
            or base_version < 0
            or type(steps) is not int
            or steps <= 0
            or type(tokens) is not int
            or tokens <= 0
        ):
            raise ValueError("Invalid KB update metadata")
        version = self.version()
        last_seq = self._meta("last_seq", {})
        if sequence <= last_seq.get(str(worker), 0):
            return {"status": "already_merged", "version": version}
        if base_version > version or version - base_version > self.max_staleness:
            raise conflict("Stale KB baseline; fetch the current canonical bank")
        update = load_kb_update(incoming)
        target = self.pending / f"{worker:03d}.safetensors"
        if target.exists():
            if int(_metadata(target)["sequence"]) == sequence:
                return {"status": "already_queued", "version": version}
            raise conflict("One KB update per worker may be in flight")
        save_kb_update(
            target,
            {k: v.numpy() for k, v in update.rows.items()},
            list(update.demand),
            wire_dtype="fp32",
            metadata={
                "worker": str(worker),
                "sequence": str(sequence),
                "base_version": str(base_version),
                "steps": str(steps),
                "tokens": str(tokens),
            },
        )
        self.try_merge()
        return {"status": "queued", "version": self.version()}

    def try_merge(self) -> bool:
        from megatoken.training.bank import grow_bank

        version = self.version()
        last_seq = self._meta("last_seq", {})
        proposals = []
        for file in sorted(self.pending.glob("*.safetensors")):
            item = _metadata(file)
            worker, sequence = int(item["worker"]), int(item["sequence"])
            age = version - int(item["base_version"])
            if sequence <= last_seq.get(str(worker), 0) or not 0 <= age <= self.max_staleness:
                file.unlink()
                continue
            proposals.append((file, item, age))
        if len(proposals) < self.min_quorum:
            self.quorum_since = None
            return False
        if self.quorum_since is None:
            self.quorum_since = time.monotonic()
        if (
            len(proposals) < self.workers
            and time.monotonic() - self.quorum_since < self.grace_seconds
        ):
            return False
        self.quorum_since = None
        updates = [(load_kb_update(file), item, age) for file, item, age in proposals]
        contributions: dict[str, list[tuple[torch.Tensor, int, int, int]]] = {}
        for update, item, age in updates:
            for key, delta in update.rows.items():
                contributions.setdefault(key, []).append(
                    (delta, int(item["tokens"]), int(item["steps"]), age)
                )
        with self.bank.transaction():
            for key, parts in sorted(contributions.items()):
                identity = parse_row_key(key)
                current = read_field(self.bank, identity)
                if current is None:
                    continue  # Retired or unknown on the canonical bank: stale update.
                parts = [p for p in parts if p[0].numel() == current.size]
                if not parts:
                    continue
                delta = merge_pseudogradients(
                    [p[0] for p in parts],
                    [p[1] for p in parts],
                    [p[2] for p in parts],
                    [p[3] for p in parts],
                )
                stored = self.bank.db.execute(
                    "SELECT momentum FROM diloco_kb_outer WHERE row=?", (key,)
                ).fetchone()
                velocity = (
                    torch.from_numpy(np.frombuffer(stored[0], dtype="<f4").copy())
                    if stored is not None and len(stored[0]) == 4 * current.size
                    else torch.zeros(current.size)
                )
                theta, velocity = outer_nesterov(
                    torch.from_numpy(current),
                    velocity,
                    delta,
                    lr=self.outer_lr,
                    momentum=self.outer_momentum,
                )
                write_field(self.bank, identity, theta.numpy(), self.ivf)
                self.bank.db.execute(
                    """INSERT INTO diloco_kb_outer VALUES (?,?) ON CONFLICT(row)
                       DO UPDATE SET momentum=excluded.momentum""",
                    (key, velocity.numpy().astype("<f4").tobytes()),
                )
            for update, _, _ in updates:
                for event_id, region, family, cursor in update.demand:
                    self.growth.demand.observe(
                        event_id=event_id,
                        region=region,
                        family=family,
                        step=cursor,
                    )
            if self.settings["growth_budget"]:
                grow_bank(
                    self.bank,
                    self.recipe,
                    self.growth,
                    self.ivf,
                    scopes=tuple(self.settings["scopes"]),
                    step=version + 1,
                    budget=self.settings["growth_budget"],
                )
            for _, item, _ in updates:
                last_seq[item["worker"]] = int(item["sequence"])
            # The version and sequence ledger commit with the rows: replayed
            # queued proposals are discarded after a coordinator restart.
            self._put_meta("last_seq", last_seq)
            self._put_meta("version", version + 1)
            floor = version + 1 - self.retained_versions
            if floor > int(self._meta("changes_floor", 0)):
                self.bank.db.execute("DELETE FROM diloco_kb_changes WHERE version <= ?", (floor,))
                self._put_meta("changes_floor", floor)
        for file, _, _ in proposals:
            file.unlink(missing_ok=True)
        self.merges += 1
        return True


class KBSync:
    """Learner side: baselines, outgoing row deltas and canonical installs.

    Network I/O runs on the owning DecoupledWorker's pool. Bank reads and
    installs happen only on the training thread between optimizer steps.
    """

    def __init__(self, *, version: int, wire_dtype: str, local_steps: int) -> None:
        self.version = version
        self.wire_dtype = wire_dtype
        self.local_steps = local_steps
        self.baselines: dict[str, np.ndarray] = {}
        self.demand: list[tuple[str, str, str, int]] = []
        self.steps_since = 0
        self.tokens_since = 0
        self.sequence = 0
        self.pending: dict | None = None
        self.awaiting_global = False
        self.download = None

    def capture(self, leaves: dict[RowID, torch.Tensor]) -> None:
        """Record canonical values of rows touched for the first time since install."""
        for identity, leaf in leaves.items():
            key = row_key(identity)
            if key not in self.baselines:
                self.baselines[key] = leaf.detach().float().cpu().numpy().reshape(-1).copy()

    def add_demand(self, demand: list[tuple[str, str, str, int]]) -> None:
        self.demand.extend(demand)

    def deltas(self, bank: SQLiteKB) -> dict[str, np.ndarray]:
        rows = {}
        for key, baseline in self.baselines.items():
            current = read_field(bank, parse_row_key(key))
            if current is None or current.shape != baseline.shape:
                continue
            delta = baseline - current
            if not np.isfinite(delta).all():
                raise ValueError("Nonfinite local KB pseudogradient")
            if np.any(delta):
                rows[key] = delta.astype(np.float32)
        return rows

    def trained_rows(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Record and relation IDs modified locally since the last install."""
        identities = [parse_row_key(key) for key in self.baselines]
        return (
            tuple(sorted({i.id for i in identities if i.kind == "record"})),
            tuple(sorted({i.id for i in identities if i.kind == "relation"})),
        )

    def uploaded(self, bank: SQLiteKB) -> None:
        """Re-anchor every touched row at its upload point (the value just sent)."""
        for key in list(self.baselines):
            current = read_field(bank, parse_row_key(key))
            if current is None:
                del self.baselines[key]
            else:
                self.baselines[key] = current
        self.steps_since = 0
        self.tokens_since = 0

    def residuals(self, bank: SQLiteKB) -> dict[str, np.ndarray]:
        """Local row progress not yet contained in any upload."""
        return {key: -delta for key, delta in self.deltas(bank).items()}

    def installed(
        self,
        version: int,
        bank: SQLiteKB,
        residuals: dict[str, np.ndarray],
        overwritten: frozenset[tuple[str, int]] | None,
        ivf: Any = None,
    ) -> None:
        """Adopt canonical rows, then carry unsent local progress onto them.

        Mirrors the fragment install: steps trained while a round is in flight
        are kept and become the next upload, never overwritten. ``overwritten``
        names the rows the install replaced (None: a full snapshot). Rows it
        left alone still hold their local progress against a valid baseline.
        """
        self.version = version
        self.awaiting_global = False
        for key, residual in residuals.items():
            identity = parse_row_key(key)
            if overwritten is not None and (identity.kind, identity.id) not in overwritten:
                continue
            del self.baselines[key]
            canonical = read_field(bank, identity)
            if canonical is None or canonical.shape != residual.shape:
                continue
            write_field(bank, identity, canonical + residual, ivf)
            self.baselines[key] = canonical
        for key in [k for k in self.baselines if k not in residuals]:
            identity = parse_row_key(key)
            if overwritten is None or (identity.kind, identity.id) in overwritten:
                # No unsent progress: the row now simply equals canonical.
                del self.baselines[key]

    def state_dict(self) -> dict:
        return {
            "version": self.version,
            "baselines": {k: torch.from_numpy(v.copy()) for k, v in self.baselines.items()},
            "demand": [list(x) for x in self.demand],
            "steps_since": self.steps_since,
            "tokens_since": self.tokens_since,
            "sequence": self.sequence,
            "pending": None
            if self.pending is None
            else {k: v for k, v in self.pending.items() if k != "future"},
            "pending_payload": (
                None if self.pending is None else load_tensors(self.pending["filename"])["delta"]
            ),
            "pending_meta": (
                None if self.pending is None else _metadata(Path(self.pending["filename"]))
            ),
            "awaiting_global": self.awaiting_global,
        }

    def load_state_dict(self, value: dict, outbox: Path) -> None:
        self.version = int(value["version"])
        self.baselines = {k: v.float().numpy().copy() for k, v in value["baselines"].items()}
        self.demand = [tuple(x) for x in value["demand"]]
        self.steps_since = int(value["steps_since"])
        self.tokens_since = int(value["tokens_since"])
        self.sequence = int(value["sequence"])
        self.awaiting_global = bool(value["awaiting_global"])
        self.pending = None
        if value["pending"] is not None:
            entry = dict(value["pending"])
            filename = outbox / f"kb-{entry['sequence']:09d}.safetensors"
            if not filename.exists():
                save_tensors(
                    filename, {"delta": value["pending_payload"]}, metadata=value["pending_meta"]
                )
            entry["filename"] = str(filename)
            entry["future"] = None
            entry["retry_after"] = 0
            self.pending = entry
