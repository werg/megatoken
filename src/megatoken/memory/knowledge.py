"""Ingested knowledge bases: fixed packed edge tables written by the ingest heads.

An ingested base shares the trained bank's geometry and is searched with it
(``Mounted`` in ``transducer.py``), but it has no gradient masters, optimizer,
growth or coordinator: entries are only ever appended (and only while the base
is writable, e.g. a user's base filled while serving), never changed, split or
removed. It is deleted as a whole. The trained base alone is checkpointed and
synced by DiLoCo; ingested bases ship as their own files.

A base file holds the packed table and the identity of what wrote it. Entries
are written and read in the language of one trained model, so a base mounts
only on the model that wrote it (same weights), and refuses anything else.
"""

from __future__ import annotations

import threading
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from megatoken.io import atomic_binary_writer

from .transducer import EdgeStore, EdgeTable, TransducerGeometry

FORMAT = "megatoken-knowledge-base-v1"
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

    def to(self, device: torch.device | str) -> KnowledgeBase:
        self.table.to(torch.device(device))
        return self

    def add_edges(self, **fields) -> Tensor:
        if not self.writable:
            raise RuntimeError("This knowledge base is read-only")
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
        with atomic_binary_writer(Path(path)) as handle:
            torch.save(
                {
                    "format": FORMAT,
                    "name": self.name,
                    "identity": self.identity,
                    "next_edge_id": self.table.next_id,
                    "rows": self.table.state(),
                    "sources": self.sources,
                },
                handle,
            )

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
        base.table.load(rows)
        base.table.next_id = int(data["next_edge_id"])
        base.sources = {int(edge): str(label) for edge, label in data["sources"].items()}
        base.version = 1
        return base


def read_file(path: str | Path) -> dict[str, Any]:
    """A base file's contents, checked to be one (it is read with ``weights_only``)."""
    with Path(path).open("rb") as handle:
        data = torch.load(handle, map_location="cpu", weights_only=True)
    if not isinstance(data, dict) or data.get("format") != FORMAT:
        raise ValueError("Not a megatoken knowledge base file")
    return data
