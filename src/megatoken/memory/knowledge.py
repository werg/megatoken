"""Packed knowledge bases and bounded writable conversation overlays.

An ingested base shares the trained bank's geometry and is searched with it
(``Mounted`` in ``transducer.py``), but it has no gradient masters, optimizer,
growth or coordinator. Writable bases support erasure and weighted value updates.
 The trained base alone is checkpointed and
synced by DiLoCo; ingested bases ship as their own files.

A base file holds the packed table and the identity of what wrote it. Entries
are written and read in the language of one trained model, so a base mounts
only on the model that wrote it (same weights), and refuses anything else.
"""

from __future__ import annotations

import copy
import threading
import uuid
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from megatoken.io import atomic_binary_writer

from .transducer import EdgeStore, EdgeTable, TransducerGeometry
from .transducer_packing import pack_fp4, pack_fp8, row_seeds

FORMAT = "megatoken-knowledge-base-v1"  # Stable model identity.
FILE_FORMAT = "megatoken-knowledge-base-v2"
WRITE_HALF_LIFE = 16384
# What decides whether a base and a model speak the same language.
IDENTITY_KEYS = ("format", "geometry", "teacher", "architecture", "weights")
# The geometry fields that fix the stored values, addresses and the content projection.
GEOMETRY_KEYS = ("location_dim", "query_dim", "value_dim", "seed")


def geometry_identity(geometry: TransducerGeometry) -> dict[str, int]:
    return {name: getattr(geometry, name) for name in GEOMETRY_KEYS}


def check_identity(found: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    """Refuse a base written by a different model, teacher or geometry, naming the difference."""
    for key in IDENTITY_KEYS:
        if found.get(key) != expected.get(key):
            raise ValueError(
                f"This knowledge base was written for a different {key} "
                f"({found.get(key)!r}, this model has {expected.get(key)!r}); build it again "
                "with this model"
            )


class KnowledgeBase(EdgeStore):
    """A packed edge table of ingested entries, mountable beside the trained bank."""

    def __init__(
        self,
        geometry: TransducerGeometry,
        identity: Mapping[str, Any],
        *,
        name: str = "",
        writable: bool = False,
        device: torch.device | str = "cpu",
    ) -> None:
        self.geometry = geometry
        self.identity = dict(identity)
        self.name = name
        self.writable = writable
        self.table = EdgeTable(geometry, torch.device(device))
        self.version = 0
        self.base_id = uuid.uuid4().hex
        self.sources: dict[int, str] = {}  # edge ID -> where the item came from
        # Held while entries are appended, and by lookups of a writable base (serving).
        self.lock = threading.RLock()
        self.provenance: dict[int, dict[str, Any]] = {}
        self.masters: dict[int, Tensor] = {}
        self.use_counts: dict[int, float] = {}
        self.scope: str | None = None
        self.clock = 0
        self.path: Path | None = None

    @contextmanager
    def transaction(self):
        """Restore rows and bookkeeping if any publication step fails."""
        with self.lock:
            rows, next_id, version = self.table.state(), self.table.next_id, self.version
            metadata = copy.deepcopy(
                (self.sources, self.provenance, self.masters, self.use_counts, self.clock)
            )
            try:
                yield
            except BaseException:
                self.table.load(rows)
                self.table.next_id, self.version = next_id, version
                (self.sources, self.provenance, self.masters, self.use_counts, self.clock) = (
                    metadata
                )
                raise

    @torch.inference_mode()
    def retire(self, ids) -> None:
        """Zero every stored column before releasing rows for reuse."""
        if not self.writable:
            raise RuntimeError("This knowledge base is read-only")
        with self.lock:
            edges = list(dict.fromkeys(int(i) for i in ids))
            slots = self.table.slots(edges)
            self.table.release(slots)
            for name in self.table.spec:
                if name != "ids":
                    getattr(self.table, name)[slots] = 0
            for edge in edges:
                for mapping in (self.sources, self.provenance, self.masters, self.use_counts):
                    mapping.pop(edge, None)
            if edges:
                self._changed()

    @torch.inference_mode()
    def blend(self, ids, values: Tensor, mass, clock: int, *, destination=None) -> None:
        """Weighted means with FP32 masters; clock counts committed ordinary groups."""
        if not self.writable:
            raise RuntimeError("This knowledge base is read-only")
        edges = [int(i) for i in ids]
        with self.lock:
            if not edges or len(set(edges)) != len(edges) or clock < self.clock:
                raise ValueError("Blend needs distinct IDs and a monotonic clock")
            slots = self.table.slots(edges)
            values = values.detach().to(self.device).float()
            weights = torch.as_tensor(mass, device=self.device).float().expand(len(edges))
            if (
                values.shape != (len(edges), self.geometry.value_dim)
                or not bool(torch.isfinite(values).all())
                or not bool(torch.isfinite(weights).all())
                or bool((weights <= 0).any())
            ):
                raise ValueError("Blend needs finite values and positive mass")
            age = (clock - self.table.last_write[slots]).float().clamp_min(0)
            old_mass = self.table.write_mass[slots] * torch.exp2(-age / WRITE_HALF_LIFE)
            old = torch.stack([self.masters.get(e, self.decode("value", [e])[0]) for e in edges])
            total = old_mass + weights
            mean = (old * old_mass[:, None] + values * weights[:, None]) / total[:, None]
            device_ids = torch.tensor(edges, device=self.device)
            packed = pack_fp4(mean, row_seeds(device_ids, "value", clock, 0))
            destinations = (
                None
                if destination is None
                else pack_fp8(destination(mean), row_seeds(device_ids, "destination", clock, 0))
            )
            self.table.write_values(slots, packed)
            if destinations is not None:
                self.table.write_address("destination", slots, destinations)
            self.table.write_mass[slots] = total
            self.table.last_write[slots] = clock
            self.masters.update(zip(edges, mean.unbind(), strict=True))
            self.clock = clock
            self._changed()

    def make_room(self, count: int) -> None:
        """Evict memories with lowest write mass times use count."""
        limit = min(self.geometry.max_edges, self.geometry.byte_budget // self.geometry.row_bytes)
        if count > limit:
            raise ValueError("Write exceeds overlay capacity")
        needed = max(0, count - self.edge_room)
        if needed:
            edges = self.edge_ids().tolist()
            masses = self.table.write_mass[self.table.slots(edges)].tolist()
            ranked = sorted(
                zip(edges, masses, strict=True),
                key=lambda pair: (pair[1] * self.use_counts.get(pair[0], 0), pair[0]),
            )
            self.retire([edge for edge, _ in ranked[:needed]])

    def to(self, device: torch.device | str) -> KnowledgeBase:
        self.table.to(torch.device(device))
        self.masters = {e: v.to(device) for e, v in self.masters.items()}
        return self

    @torch.inference_mode()
    def add_edges(self, **fields) -> Tensor:
        if not self.writable:
            raise RuntimeError("This knowledge base is read-only")
        with self.lock:
            return super().add_edges(**fields)

    def add(
        self,
        *,
        departure: Tensor,
        key: Tensor,
        value: Tensor,
        destination: Tensor,
        sources: Sequence[str],
    ) -> Tensor:
        """Append one entry per row (bias zero), remembering each item's source; new edge IDs."""
        if len(sources) != departure.shape[0]:
            raise ValueError("Every ingested entry needs a source label")
        with self.lock:
            count = departure.shape[0]
            ids = self.add_edges(
                departure=departure,
                key=key,
                value=value,
                destination=destination,
                bias=torch.zeros(count),
                write_mass=torch.zeros(count),
                step=0,
            )
            self.sources.update(zip(ids.tolist(), sources, strict=True))
            return ids

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "edges": self.edge_count,
            "packed_bytes": self.packed_bytes,
            "writable": self.writable,
            "sources": len(set(self.sources.values())),
            "identity": self.identity,
        }

    # ------------------------------------------------------------------ files

    def save(self, path: str | Path) -> None:
        """Write the base as one self-contained file (tensors and plain values only)."""
        with self.lock, atomic_binary_writer(Path(path)) as handle:
            torch.save(
                {
                    "format": FILE_FORMAT,
                    "name": self.name,
                    "identity": self.identity,
                    "next_edge_id": self.table.next_id,
                    "rows": self.table.state(),
                    "sources": self.sources,
                    "provenance": self.provenance,
                    "masters": {e: v.cpu() for e, v in self.masters.items()},
                    "use_counts": self.use_counts,
                    "clock": self.clock,
                    "scope": self.scope,
                    "geometry": self.geometry.__dict__,
                },
                handle,
            )
            self.path = Path(path)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        geometry: TransducerGeometry,
        identity: Mapping[str, Any],
        device: torch.device | str = "cpu",
        writable: bool = False,
    ) -> KnowledgeBase:
        """Open a base file for a model; ``identity`` is that model's (``kb.model_identity``)."""
        data = read_file(path)
        check_identity(data["identity"], identity)
        rows = data["rows"]
        if int(rows["ids"].numel()) > geometry.max_edges:
            raise ValueError("Knowledge base exceeds this model's edge ceiling")
        base = cls(geometry, data["identity"], name=data["name"], writable=writable, device=device)
        if "geometry" in data:
            stored = TransducerGeometry(**data["geometry"])
            if geometry_identity(stored) != geometry_identity(geometry):
                raise ValueError("Stored geometry differs from model")
            if int(rows["ids"].numel()) > min(
                stored.max_edges, stored.byte_budget // stored.row_bytes
            ):
                raise ValueError("Stored rows exceed knowledge base capacity")
            base.geometry = stored
            base.table = EdgeTable(stored, torch.device(device))
        base.table.load(rows)
        base.table.next_id = int(data["next_edge_id"])
        base.sources = {int(edge): str(label) for edge, label in data["sources"].items()}
        base.provenance = data.get("provenance", {})
        base.masters = {int(e): v.to(device) for e, v in data.get("masters", {}).items()}
        base.use_counts = data.get("use_counts", {})
        base.clock = int(data.get("clock", 0))
        base.scope = data.get("scope")
        base.path = Path(path)
        base.version = 1
        return base


def read_file(path: str | Path) -> dict[str, Any]:
    """A base file's contents, checked to be one (it is read with ``weights_only``)."""
    with Path(path).open("rb") as handle:
        data = torch.load(handle, map_location="cpu", weights_only=True)
    if not isinstance(data, dict) or data.get("format") not in (FORMAT, FILE_FORMAT):
        raise ValueError("Not a megatoken knowledge base file")
    return data
