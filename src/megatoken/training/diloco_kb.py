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
  ledger commit in one SQLite transaction. A stripped snapshot (no outer
  momentum, no learner optimizer moments) is then published atomically.
  Learners install it between optimizer steps, keeping their own moments.
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
RELATION_FIELDS = ("value", "dependence_logits", "strength", "lifetime_logit")
MAX_DEMAND = 1 << 20


def row_key(identity: RowID) -> str:
    return "|".join((
        identity.collection.value, identity.kind, str(identity.id),
        identity.field, identity.layout,
    ))


def parse_row_key(key: str) -> RowID:
    parts = key.split("|")
    if len(parts) != 5:
        raise ValueError("Malformed KB row key")
    collection, kind, identity, field, layout = parts
    if kind == "record" and field not in RECORD_FIELDS or (
        kind == "relation" and field not in RELATION_FIELDS
    ) or kind not in {"record", "relation"} or not identity.isdigit():
        raise ValueError("Unsupported KB row field")
    return RowID(Collection(collection), kind, int(identity), field, layout)


def read_field(bank: SQLiteKB, identity: RowID) -> np.ndarray | None:
    """Current FP32 field value, or None when the row is missing or retired."""
    try:
        row = (bank.get(identity.collection, identity.id) if identity.kind == "record"
               else bank.get_relation(identity.id))
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
            index = {"posting_cell": ivf.cell(values),
                     "centroids_revision": ivf.identity.centroid_sha256}
        bank.update_record(identity.collection, identity.id,
                           **{identity.field: replacement}, **index)
    else:
        bank.update_relation(identity.id, **{identity.field: replacement})


def _metadata(path: Path) -> dict[str, str]:
    from safetensors import safe_open

    with safe_open(str(path), framework="pt", device="cpu") as file:
        return file.metadata() or {}


def save_kb_update(path: Path, rows: dict[str, np.ndarray],
                   demand: list[tuple[str, str, str, int]], *, wire_dtype: str,
                   metadata: dict[str, str] | None = None) -> None:
    keys = sorted(rows)
    flat = (torch.from_numpy(np.concatenate([rows[k] for k in keys]))
            if keys else torch.zeros(0))
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
    if flat.dtype not in (torch.float32, torch.bfloat16) or flat.ndim != 1 or (
        not torch.isfinite(flat).all()
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
        rows[key] = flat[cursor:cursor + size].float()
        cursor += size
    if cursor != flat.numel():
        raise ValueError("KB update row layout does not cover the delta")
    parsed = []
    for item in demand:
        if (not isinstance(item, list) or len(item) != 4
                or not all(isinstance(x, str) for x in item[:3]) or type(item[3]) is not int):
            raise ValueError("Malformed KB query demand entry")
        parsed.append((item[0], item[1], item[2], item[3]))
    return KBUpdate(rows, tuple(parsed))


def strip_snapshot(path: Path) -> None:
    """Published snapshots carry no outer momentum or learner optimizer moments."""
    connection = sqlite3.connect(path, isolation_level=None)
    try:
        connection.execute("DROP TABLE IF EXISTS diloco_kb_outer")
        connection.execute("DROP TABLE IF EXISTS diloco_kb_meta")
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
            connection.execute(
                "INSERT INTO optimizer_rows SELECT * FROM previous.optimizer_rows"
            )
            connection.execute("COMMIT")
            connection.execute("DETACH DATABASE previous")
    finally:
        connection.close()
    os.replace(downloaded, local)


def initialize_kb(root: Path, *, recipe_path: str, bank_path: str,
                  centroids_path: str | None, centroids_sha256: str | None,
                  nprobe: int, scopes: tuple[str, ...], growth_budget: int) -> dict:
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
        "database_id": database_id, "recipe_fingerprint": recipe.fingerprint,
        "centroids_sha256": centroids_sha256 if centroids_path else None,
        "nprobe": nprobe, "scopes": sorted(set(scopes)), "growth_budget": growth_budget,
    }
    # Opening once creates the coordinator tables and publishes version 0.
    KBCoordinator(root, settings, outer_lr=1.0, outer_momentum=0.0, max_staleness=0,
                  min_quorum=1, workers=1, grace_seconds=0).close()
    return settings


class KBCoordinator:
    """Single canonical bank writer behind the fragment coordinator's lock."""

    def __init__(self, root: Path, settings: dict, *, outer_lr: float,
                 outer_momentum: float, max_staleness: int, min_quorum: int,
                 workers: int, grace_seconds: float) -> None:
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
        self.growth = GrowthAllocator(self.bank, self.recipe.growth)
        self.ivf = None
        if settings.get("centroids_sha256"):
            from megatoken.memory.disk_ivf import SQLiteDiskIVF

            centroids = self.root / "centroids.npy"
            if sha256_file(centroids) != settings["centroids_sha256"]:
                raise ValueError("Coordinator IVF centroids changed")
            self.ivf = SQLiteDiskIVF(self.bank, np.load(centroids, allow_pickle=False),
                                     nprobe=settings["nprobe"])
        self.quorum_since: float | None = None
        self.merges = 0
        self.sync_public()

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
        pointer = json.loads((self.root / "public.json").read_text())
        return int(pointer["version"]), self.root / pointer["file"]

    def sync_public(self) -> None:
        """Republish after a crash between the commit and snapshot publication."""
        version = self.version()
        pointer = self.root / "public.json"
        if pointer.exists() and json.loads(pointer.read_text())["version"] == version:
            return
        name = f"public-{version:09d}.sqlite"
        target = self.root / name
        target.unlink(missing_ok=True)
        self.bank.backup(target)
        strip_snapshot(target)
        write_json(pointer, {"version": version, "file": name,
                             "sha256": sha256_file(target)})
        for old in self.root.glob("public-*.sqlite"):
            if old.name != name:
                old.unlink(missing_ok=True)

    def submit(self, worker: int, sequence: int, base_version: int, steps: int,
               tokens: int, incoming: Path, conflict: type[Exception]) -> dict:
        if (type(sequence) is not int or sequence < 1
                or type(base_version) is not int or base_version < 0
                or type(steps) is not int or steps <= 0
                or type(tokens) is not int or tokens <= 0):
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
            target, {k: v.numpy() for k, v in update.rows.items()},
            list(update.demand), wire_dtype="fp32",
            metadata={"worker": str(worker), "sequence": str(sequence),
                      "base_version": str(base_version), "steps": str(steps),
                      "tokens": str(tokens)},
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
        if (len(proposals) < self.workers
                and time.monotonic() - self.quorum_since < self.grace_seconds):
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
                    [p[0] for p in parts], [p[1] for p in parts],
                    [p[2] for p in parts], [p[3] for p in parts],
                )
                stored = self.bank.db.execute(
                    "SELECT momentum FROM diloco_kb_outer WHERE row=?", (key,)
                ).fetchone()
                velocity = (torch.from_numpy(np.frombuffer(stored[0], dtype="<f4").copy())
                            if stored is not None and len(stored[0]) == 4 * current.size
                            else torch.zeros(current.size))
                theta, velocity = outer_nesterov(
                    torch.from_numpy(current), velocity, delta,
                    lr=self.outer_lr, momentum=self.outer_momentum,
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
                        event_id=event_id, region=region, family=family, step=cursor,
                    )
            if self.settings["growth_budget"]:
                grow_bank(
                    self.bank, self.recipe, self.growth, self.ivf,
                    scopes=tuple(self.settings["scopes"]), step=version + 1,
                    budget=self.settings["growth_budget"],
                )
            for _, item, _ in updates:
                last_seq[item["worker"]] = int(item["sequence"])
            # The version and sequence ledger commit with the rows: replayed
            # queued proposals are discarded after a coordinator restart.
            self._put_meta("last_seq", last_seq)
            self._put_meta("version", version + 1)
        self.sync_public()
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

    def installed(self, version: int) -> None:
        self.version = version
        self.baselines.clear()
        self.steps_since = 0
        self.tokens_since = 0
        self.awaiting_global = False

    def state_dict(self) -> dict:
        return {
            "version": self.version,
            "baselines": {k: torch.from_numpy(v.copy()) for k, v in self.baselines.items()},
            "demand": [list(x) for x in self.demand],
            "steps_since": self.steps_since,
            "tokens_since": self.tokens_since,
            "sequence": self.sequence,
            "pending": None if self.pending is None else {
                k: v for k, v in self.pending.items() if k != "future"
            },
            "pending_payload": (
                None if self.pending is None
                else load_tensors(self.pending["filename"])["delta"]
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
                save_tensors(filename, {"delta": value["pending_payload"]},
                             metadata=value["pending_meta"])
            entry["filename"] = str(filename)
            entry["future"] = None
            entry["retry_after"] = 0
            self.pending = entry

