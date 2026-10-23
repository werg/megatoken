"""v10 agenda-wide continuous-state transducer bank.

Edges live in a columnar table on the model's device: one tensor per packed
field, indexed by slot. Edge IDs are stable; slots are an internal detail and
can be reused after retirement. There is no query-key index, global candidate
search or semantic-KV materialization: every lookup scores the departure
shortlists of its session's agenda locations and keeps the best pairs.

All structural and parameter changes happen at window boundaries (optimizer
steps, coordinator merges, write-out after backward). Each change bumps the
bank version; agendas recompute their cached shortlists when it moves.

A session searches several knowledge bases at once (``Mounted``): the trained
``TransducerBank`` and any number of ingested, read-only ``KnowledgeBase``s
(``memory/knowledge.py``), which share its geometry. A lookup scans each base's
departures, scores every pair together with the base's log blend weight added,
and keeps the best pairs across all of them in one softmax with the one sink.
"""

from __future__ import annotations

import math
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor, nn

from .transducer_packing import (
    FP4Rows,
    FP8Rows,
    fp4_nbytes,
    fp8_nbytes,
    pack_fp4,
    pack_fp8,
    packed_unit,
    row_seeds,
    straight_through,
    unit,
)
from .transducer_rows import RowStore

ADDRESS_FIELDS = ("key", "departure", "destination")
FIELDS = (*ADDRESS_FIELDS, "bias", "value")
FIELD_COLUMNS = {
    **{name: (f"{name}_codes", f"{name}_scales") for name in ADDRESS_FIELDS},
    "value": ("value_codes", "value_groups", "value_rows"),
    "bias": ("bias",),
}


# Provisional (learner-local) edge IDs live far above any canonical ID.
PROVISIONAL_BASE = 1 << 62
# Distinct shortlisted edges decoded together when scoring an agenda's pairs.
PAIR_CHUNK = 16384


class BankFull(RuntimeError):
    """The hard edge ceiling or the packed-byte budget would be exceeded."""


@dataclass(frozen=True)
class TransducerGeometry:
    location_dim: int = 256
    query_dim: int = 1024
    value_dim: int = 32768
    neighbors: int = 1024
    selected: int = 32
    exploration: int = 2
    max_lookups: int = 256
    max_edges: int = 1_000_000
    byte_budget: int = 24_000_000_000
    seed: int = 17

    def __post_init__(self) -> None:
        if (
            min(
                self.location_dim,
                self.query_dim,
                self.value_dim,
                self.neighbors,
                self.selected,
                self.max_lookups,
                self.max_edges,
                self.byte_budget,
            )
            < 1
        ):
            raise ValueError("Transducer geometry must be positive")
        if not 0 <= self.exploration < self.selected:
            raise ValueError("Exploration reserves slots inside the selected pair budget")

    @property
    def row_bytes(self) -> int:
        """Packed bytes of one edge: three FP8 addresses, one FP4 value and an FP32 bias."""
        return (
            2 * fp8_nbytes(self.location_dim)
            + fp8_nbytes(self.query_dim)
            + fp4_nbytes(self.value_dim)
            + 4
        )

    def dims(self, name: str) -> int:
        return {
            "departure": self.location_dim,
            "destination": self.location_dim,
            "key": self.query_dim,
            "value": self.value_dim,
            "bias": 1,
        }[name]


# Columns recomputed from the packed ones, never saved or published.
DERIVED_COLUMNS = ("departure_unit", "key_unit")
# Rows decoded, gathered or copied per call by whole-table passes (load, snapshot).
TABLE_CHUNK = 65536


class EdgeTable:
    """Columnar packed storage. Not an nn.Module: rows are trained by the sparse optimizer.

    ``departure_unit`` caches each row's decoded unit departure (FP32) so the
    exact neighbourhood scan is one matrix product; ``key_unit`` caches its unit
    key in BF16 (an E4M3 code is exact in BF16; only its scale rounds) for
    scoring shortlisted pairs. Both are rewritten whenever their field is.
    """

    def __init__(self, geometry: TransducerGeometry, device: torch.device, capacity: int = 256):
        self.geometry = geometry
        self.device = torch.device(device)
        self.capacity = 0
        self.slot_of: dict[int, int] = {}
        self.free: list[int] = []
        self.high_water = 0
        self.next_id = 1
        self._allocate(capacity)

    @property
    def spec(self) -> dict[str, tuple[tuple[int, ...], torch.dtype, float]]:
        """Each column's per-row shape, dtype and fill value."""
        g, fp8 = self.geometry, torch.float8_e4m3fn
        location, query = (g.location_dim,), (g.query_dim,)
        return {
            "ids": ((), torch.int64, -1),
            "departure_codes": (location, fp8, 0),
            "departure_scales": ((-(-g.location_dim // 128),), torch.float32, 1),
            "departure_unit": (location, torch.float32, 0),
            "key_unit": (query, torch.bfloat16, 0),
            "key_codes": (query, fp8, 0),
            "key_scales": ((-(-g.query_dim // 128),), torch.float32, 1),
            "destination_codes": (location, fp8, 0),
            "destination_scales": ((-(-g.location_dim // 128),), torch.float32, 1),
            "value_codes": ((-(-g.value_dim // 2),), torch.uint8, 0),
            "value_groups": ((-(-g.value_dim // 16),), fp8, 1),
            "value_rows": ((), torch.float32, 1),
            "value_norm2": ((), torch.float32, 0),
            "bias": ((), torch.float32, 0),
            "write_mass": ((), torch.float32, 0),
            "birth": ((), torch.int64, 0),
            "last_write": ((), torch.int64, 0),
        }

    def _allocate(self, capacity: int) -> None:
        """Resize one column at a time, so growth never holds two whole tables."""
        for name, (tail, dtype, fill) in self.spec.items():
            fresh = torch.full((capacity, *tail), fill, dtype=dtype, device=self.device)
            if self.capacity:
                fresh[: self.capacity] = getattr(self, name)
            setattr(self, name, fresh)
        self.capacity = capacity

    def to(self, device: torch.device) -> None:
        if torch.device(device) == self.device:
            return
        for name in self.spec:
            setattr(self, name, getattr(self, name).to(device))
        self.device = torch.device(device)

    @property
    def count(self) -> int:
        return len(self.slot_of)

    def slots(self, ids: Tensor | list[int]) -> Tensor:
        try:
            values = [
                self.slot_of[int(i)] for i in (ids.tolist() if isinstance(ids, Tensor) else ids)
            ]
        except KeyError as missing:
            raise KeyError(f"Edge {missing} is not in the bank") from None
        return torch.tensor(values, dtype=torch.int64, device=self.device)

    def alive_slots(self) -> Tensor:
        return torch.nonzero(self.ids[: self.high_water] >= 0).flatten()

    def ids_of(self, slots: Tensor) -> Tensor:
        return self.ids[slots]

    def claim(self, count: int) -> Tensor:
        reused = [self.free.pop() for _ in range(min(count, len(self.free)))]
        needed = count - len(reused)
        if self.high_water + needed > self.capacity:
            ceiling = max(self.geometry.max_edges, self.high_water + needed)
            self._allocate(min(max(self.capacity * 2, self.high_water + needed), ceiling))
        fresh = list(range(self.high_water, self.high_water + needed))
        self.high_water += needed
        return torch.tensor(reused + fresh, dtype=torch.int64, device=self.device)

    def release(self, slots: Tensor) -> None:
        for slot, edge in zip(slots.tolist(), self.ids[slots].tolist(), strict=True):
            del self.slot_of[edge]
            self.free.append(slot)
        self.ids[slots] = -1

    def write_address(self, name: str, slots: Tensor, rows: FP8Rows) -> None:
        getattr(self, f"{name}_codes")[slots] = rows.codes
        getattr(self, f"{name}_scales")[slots] = rows.scales
        if name == "departure":
            self.departure_unit[slots] = packed_unit(rows.decode())
        elif name == "key":
            self.key_unit[slots] = packed_unit(rows.decode()).to(torch.bfloat16)

    def _derive(self, slots: Tensor) -> None:
        for chunk in slots.split(TABLE_CHUNK):
            self.departure_unit[chunk] = packed_unit(self.decode("departure", chunk))
            self.key_unit[chunk] = packed_unit(self.decode("key", chunk)).to(torch.bfloat16)

    def write_values(self, slots: Tensor, rows: FP4Rows) -> None:
        self.value_codes[slots] = rows.codes
        self.value_groups[slots] = rows.group_scales
        self.value_rows[slots] = rows.row_scales
        self.value_norm2[slots] = rows.decode(self.geometry.value_dim).square().sum(dim=1)

    def columns(self, name: str, slots: Tensor) -> dict[str, Tensor]:
        """A copy of one field's packed columns for these slots."""
        return {column: getattr(self, column)[slots].clone() for column in FIELD_COLUMNS[name]}

    def decode_columns(self, name: str, columns: dict[str, Tensor]) -> Tensor:
        if name == "bias":
            return columns["bias"]
        if name == "value":
            return FP4Rows(
                columns["value_codes"], columns["value_groups"], columns["value_rows"]
            ).decode(self.geometry.value_dim)
        return FP8Rows(columns[f"{name}_codes"], columns[f"{name}_scales"]).decode()

    def decode(self, name: str, slots: Tensor) -> Tensor:
        return self.decode_columns(
            name, {column: getattr(self, column)[slots] for column in FIELD_COLUMNS[name]}
        )

    def state(self, ids: list[int] | None = None) -> dict[str, Tensor]:
        """Compact CPU copy of the live rows (or the given ones), ordered by edge ID."""
        order = sorted(self.slot_of) if ids is None else sorted(ids)
        slots = self.slots(order)
        state = {}
        for name, (tail, dtype, _) in self.spec.items():
            if name in DERIVED_COLUMNS:
                continue
            column = getattr(self, name)
            state[name] = torch.empty((len(order), *tail), dtype=dtype)
            for start in range(0, len(order), TABLE_CHUNK):
                chunk = slots[start : start + TABLE_CHUNK]
                state[name][start : start + chunk.numel()] = column[chunk].cpu()
        return state

    def upsert(self, rows: dict[str, Tensor]) -> None:
        """Overwrite existing edges and add new ones, columns as produced by ``state``."""
        ids = rows["ids"].tolist()
        if len(set(ids)) != len(ids) or any(edge <= 0 for edge in ids):
            raise ValueError("Invalid or duplicate edge IDs in published rows")
        fresh = [edge for edge in ids if edge not in self.slot_of]
        claimed = self.claim(len(fresh))
        self.slot_of.update(zip(fresh, claimed.tolist(), strict=True))
        slots = self.slots(ids)
        for name, tensor in rows.items():
            getattr(self, name)[slots] = tensor.to(self.device)
        self._derive(slots)

    def load(self, state: dict[str, Tensor]) -> None:
        count = int(state["ids"].numel())
        slot_of = {int(edge): slot for slot, edge in enumerate(state["ids"].tolist())}
        if len(slot_of) != count or any(edge <= 0 for edge in slot_of):
            raise ValueError("Invalid or duplicate edge IDs in a transducer snapshot")
        # Free the old columns before allocating, so a load never holds two tables.
        for name in self.spec:
            delattr(self, name)
        self.capacity = 0
        self._allocate(max(count, 256))
        for name, tensor in state.items():
            column = getattr(self, name)
            for start in range(0, count, TABLE_CHUNK):
                piece = tensor[start : start + TABLE_CHUNK]
                column[start : start + piece.shape[0]] = piece.to(self.device)
        self._derive(torch.arange(count, device=self.device))
        self.slot_of, self.free, self.high_water = slot_of, [], count

    def nearest(self, locations: Tensor, count: int) -> Tensor:
        """Exact scan of the packed departures: [rows, min(count, edges)] nearest slots."""
        rows = locations.shape[0]
        count = min(count, self.count)
        if not count:
            return torch.full((rows, 0), -1, dtype=torch.int64, device=self.device)
        query = packed_unit(locations.detach().float())
        best_scores: Tensor | None = None
        best_slots: Tensor | None = None
        for start in range(0, self.high_water, TABLE_CHUNK):
            stop = min(start + TABLE_CHUNK, self.high_water)
            scores = query @ self.departure_unit[start:stop].T
            scores = scores.masked_fill(self.ids[start:stop] < 0, -math.inf)
            chunk = torch.arange(start, stop, device=self.device).expand(rows, -1)
            if best_scores is not None:
                scores = torch.cat((best_scores, scores), dim=1)
                chunk = torch.cat((best_slots, chunk), dim=1)
            top = torch.topk(scores, min(count, scores.shape[1]), dim=1)
            best_scores = top.values
            best_slots = torch.gather(chunk, 1, top.indices)
        return best_slots

    def copy_fields(self, names: tuple[str, ...], sources: Tensor, targets: Tensor) -> None:
        """Copy packed fields row for row, codes and all (no requantization)."""
        for name in names:
            for column in FIELD_COLUMNS[name]:
                getattr(self, column)[targets] = getattr(self, column)[sources]
            if name == "value":
                self.value_norm2[targets] = self.value_norm2[sources]


TRAINED = "trained"  # the name of the bank training grows, in every mount


@dataclass(frozen=True)
class Mounted:
    """One knowledge base in a session's search, with the blend weight it was mounted at.

    The weight enters every one of the base's pair scores as ``log(weight)``: 1
    changes nothing, a small weight lets the base contribute only where it fits
    much better. A weight of 0 means not mounted at all (``mount`` drops it).
    """

    name: str
    base: EdgeStore
    weight: float

    def __post_init__(self) -> None:
        if not self.name or not math.isfinite(self.weight) or self.weight <= 0:
            raise ValueError("A mounted base needs a name and a positive finite weight")

    @property
    def log_weight(self) -> float:
        return math.log(self.weight)


@dataclass
class SearchAgenda:
    """One session's reached locations and their cached departure shortlists.

    ``locations`` are the distinct appended locations; ``state`` is where the
    search actually is now (the last landing), which the sink holds still.
    ``mounted`` are the bases searched together; ``shortlists[b][j]`` is base
    ``b``'s cached neighbourhood of location ``j``, valid while ``versions[b]``
    is that base's version (one base changing rescans only that base).
    """

    locations: list[Tensor]
    state: Tensor
    mounted: tuple[Mounted, ...]
    shortlists: list[list[Tensor]] = field(default_factory=list)
    bands: list[list[Tensor]] = field(default_factory=list)
    versions: list[int] = field(default_factory=list)
    lookups: int = 0
    births: list[int] = field(default_factory=list)  # lookup count when each location joined

    def __post_init__(self) -> None:
        if not self.births:
            self.births = [0] * len(self.locations)
        if not self.versions:
            self.versions = [-1] * len(self.mounted)
            self.shortlists = [[] for _ in self.mounted]
            self.bands = [[] for _ in self.mounted]

    def copy(self) -> SearchAgenda:
        return SearchAgenda(
            list(self.locations),
            self.state,
            self.mounted,
            [list(x) for x in self.shortlists],
            [list(x) for x in self.bands],
            list(self.versions),
            self.lookups,
            list(self.births),
        )

    def publish(self, other: SearchAgenda) -> None:
        self.locations, self.state = other.locations, other.state
        self.shortlists, self.bands = other.shortlists, other.bands
        self.versions, self.lookups = other.versions, other.lookups
        self.births = other.births

    def truncate_credit(self, groups: int) -> None:
        """Keep gradients through the last ``groups`` lookups only.

        Locations that joined earlier are detached; the search state's own
        recurrence (the sink holds it still) is cut every ``groups`` lookups.
        """
        if groups < 1:
            raise ValueError("At least one differentiable retrieval group is required")
        oldest = self.lookups - groups
        self.locations = [
            x.detach() if birth <= oldest else x
            for x, birth in zip(self.locations, self.births, strict=True)
        ]
        if self.lookups % groups == 0:
            self.state = self.state.detach()


@dataclass(frozen=True)
class Lookup:
    """An uncommitted, differentiable retrieval. Cancelling it changes nothing."""

    value: Tensor  # what the readout receives (after the hindsight overlay)
    landing: Tensor  # the new location (after hindsight steering)
    bank_value: Tensor  # the blend over every mounted base, sink included
    bank_landing_raw: Tensor  # the bank's unnormalized landing, lambda_t
    weights: Tensor  # pair weights followed by the sink weight
    pair_logits: Tensor
    sink_weight: Tensor
    pairs: tuple[tuple[int, str, int], ...]  # (agenda location index, base name, edge ID)
    # The trained bank's distinct selected edges and their total weights omega_i: the
    # only rows that train, write or evolve. Ingested bases never appear here.
    edge_ids: Tensor
    edge_weights: Tensor
    start: Tensor  # where the search was (the sink's hold-still destination)
    start_key: Tensor  # the query read through the start location
    query: Tensor  # the unit query asked
    versions: tuple[int, ...]  # each mounted base's version when the lookup ran
    base_edges: dict[str, tuple[Tensor, Tensor]] = field(default_factory=dict)


class MasterCache:
    """FP32 working masters of trained fields: one LRU-bounded row store per field.

    A training lookup gathers its selected rows into one fresh leaf per field;
    ``collect`` sums each window's leaf gradients per row for the sparse step.
    Rows are only created, evicted or dropped between windows.
    """

    def __init__(self, geometry: TransducerGeometry, device: torch.device) -> None:
        self.geometry = geometry
        self.stores = {
            name: RowStore(
                device,
                {
                    "data": ((geometry.dims(name),), torch.float32),
                    "edge": ((), torch.int64),
                    "used": ((), torch.int64),
                },
            )
            for name in FIELDS
        }
        self.clock = 0
        self.pending: dict[str, list[tuple[Tensor, Tensor]]] = {name: [] for name in FIELDS}
        self.collected: dict[str, tuple[Tensor, Tensor]] = {}

    def to(self, device: torch.device) -> None:
        if not self.stores["key"].device == torch.device(device):
            states = {name: store.state_dict() for name, store in self.stores.items()}
            for name, store in self.stores.items():
                store.device = torch.device(device)
                store.load_state_dict(states[name])
            self.pending = {name: [] for name in FIELDS}
            self.collected = {}

    def leaf(self, name: str, ids: list[int], decoded: Tensor) -> Tensor:
        """A trainable ``[edges, width]`` copy of these edges' masters.

        Masters not yet cached start from their ``decoded`` packed rows.
        """
        store = self.stores[name]
        rows, created = store.rows(ids)
        if created:
            at = torch.tensor(created, device=rows.device)
            store.data[rows[at]] = decoded.reshape(len(ids), -1)[at].detach().float()
            store.edge[rows[at]] = torch.tensor([ids[k] for k in created], device=rows.device)
        self.clock += 1
        store.used[rows] = self.clock
        leaf = store.data[rows].requires_grad_()
        self.pending[name].append((rows, leaf))
        return leaf

    def collect(self) -> dict[str, tuple[Tensor, Tensor]]:
        """Per field: (rows, summed gradients) of every leaf that received one."""
        for name, entries in self.pending.items():
            graded = [(rows, leaf.grad) for rows, leaf in entries if leaf.grad is not None]
            entries.clear()
            if not graded:
                continue
            if name in self.collected:
                graded.append(self.collected[name])
            rows = torch.cat([r for r, _ in graded])
            grads = torch.cat([g.reshape(r.numel(), -1).float() for r, g in graded])
            unique, inverse = torch.unique(rows, return_inverse=True)
            summed = torch.zeros(unique.numel(), grads.shape[1], device=grads.device)
            self.collected[name] = unique, summed.index_add_(0, inverse, grads)
        return self.collected

    def take(self) -> dict[str, tuple[Tensor, Tensor]]:
        collected = self.collect()
        self.collected = {}
        return collected

    def evict(self, capacity: int) -> None:
        """Drop each field's least recently used rows beyond ``capacity`` edges."""
        for store in self.stores.values():
            excess = len(store) - capacity
            if excess <= 0:
                continue
            ids = store.ids()
            used = store.used[store.find(ids)]
            oldest = torch.topk(used, excess, largest=False).indices.tolist()
            store.drop([ids[k] for k in oldest])

    def drop(self, ids, names=FIELDS) -> None:
        ids = [int(edge) for edge in ids]
        for name in names:
            self.stores[name].drop(ids)
        self._forget_windows()

    def clear(self, names=FIELDS) -> None:
        for name in names:
            self.stores[name].clear()
        self._forget_windows()

    def _forget_windows(self) -> None:
        # Rows can be reused after a drop: an unfinished window's leaves are void.
        self.pending = {name: [] for name in FIELDS}
        self.collected = {}

    def state_dict(self) -> dict:
        """Masters with their recency (``used`` and the clock), so eviction resumes exactly."""
        return {
            "clock": self.clock,
            "fields": {name: store.state_dict() for name, store in self.stores.items()},
        }

    def load_state_dict(self, state: dict) -> None:
        self._forget_windows()
        self.clock = int(state["clock"])
        for name, saved in state["fields"].items():
            self.stores[name].load_state_dict(saved)


class EdgeStore:
    """What every knowledge base is: a packed edge table in the shared spaces, a version, a search.

    The trained bank and an ingested base are both stores; a lookup treats them
    alike, and only the bank carries gradient masters, an optimizer and growth.
    """

    geometry: TransducerGeometry
    table: EdgeTable
    version: int
    writable = False  # whether entries are appended while sessions search it (a user's base)

    @property
    def device(self) -> torch.device:
        return self.table.device

    @property
    def edge_count(self) -> int:
        return self.table.count

    @property
    def packed_bytes(self) -> int:
        return self.table.count * self.geometry.row_bytes

    @property
    def edge_room(self) -> int:
        """How many more edges fit under both the edge ceiling and the packed-byte budget."""
        g = self.geometry
        return max(
            0,
            min(g.max_edges - self.edge_count, (g.byte_budget - self.packed_bytes) // g.row_bytes),
        )

    def edge_ids(self) -> Tensor:
        """Live edge IDs in increasing order."""
        return self.table.ids_of(self.table.alive_slots()).sort().values

    def __contains__(self, edge_id: int) -> bool:
        return int(edge_id) in self.table.slot_of

    def _changed(self) -> None:
        self.version += 1

    def add_edges(
        self,
        *,
        departure: Tensor,
        key: Tensor,
        value: Tensor,
        destination: Tensor,
        bias: Tensor,
        write_mass: Tensor,
        step: int,
        worker: int = 0,
        ids: Tensor | None = None,
        provisional: bool = False,
    ) -> Tensor:
        """Append edges with fresh canonical IDs (or the given ones). Raises BankFull.

        Provisional edges (a learner's own write-outs) carry learner-local IDs and
        leave the canonical ID high-water mark alone.
        """
        g = self.geometry
        count = departure.shape[0]
        if count == 0:
            return torch.empty(0, dtype=torch.int64)
        shapes = (departure.shape, key.shape, value.shape, destination.shape, bias.shape)
        if shapes != (
            (count, g.location_dim),
            (count, g.query_dim),
            (count, g.value_dim),
            (count, g.location_dim),
            (count,),
        ) or write_mass.shape != (count,):
            raise ValueError("Edge field dimensions do not match v10 geometry")
        if step < 0 or not bool(torch.isfinite(bias).all()) or bool((write_mass < 0).any()):
            raise ValueError("Invalid edge statistics or bias")
        if count > self.edge_room:
            raise BankFull("Transducer has reached its hard edge ceiling or packed-byte budget")
        if ids is None:
            ids = torch.arange(self.table.next_id, self.table.next_id + count)
        ids = ids.to(torch.int64).cpu()
        id_list = ids.tolist()
        if (
            min(id_list) < self.table.next_id
            or len(set(id_list)) != count
            or any(i in self.table.slot_of for i in id_list)
        ):
            raise ValueError("Canonical edge IDs must increase and never be reused")
        # Pack (and so validate) every field before claiming slots: a failure adds nothing.
        device_ids = ids.to(self.device)
        addresses = {
            name: pack_fp8(
                unit(values.to(self.device).float()), row_seeds(device_ids, name, step, worker)
            )
            for name, values in (
                ("departure", departure),
                ("key", key),
                ("destination", destination),
            )
        }
        values = pack_fp4(
            value.to(self.device).float(), row_seeds(device_ids, "value", step, worker)
        )
        slots = self.table.claim(count)
        self.table.ids[slots] = device_ids
        self.table.slot_of.update(zip(id_list, slots.tolist(), strict=True))
        if not provisional:
            self.table.next_id = int(ids.max()) + 1
        for name, rows in addresses.items():
            self.table.write_address(name, slots, rows)
        self.table.write_values(slots, values)
        self.table.bias[slots] = bias.to(self.device).float()
        self.table.write_mass[slots] = write_mass.to(self.device).float()
        self.table.birth[slots] = step
        self.table.last_write[slots] = step
        self._changed()
        return ids

    def decode(self, name: str, ids: Tensor | list[int]) -> Tensor:
        return self.table.decode(name, self.table.slots(ids))

    def shortlists(self, locations: list[Tensor]) -> tuple[list[Tensor], list[Tensor]]:
        """Per location, its ``neighbors`` nearest departures and the band just past them."""
        g = self.geometry
        with torch.no_grad():
            ranked = self.table.nearest(torch.stack(locations), 2 * g.neighbors)
        return (
            [row[: g.neighbors] for row in ranked],
            [row[g.neighbors :] for row in ranked],
        )

    def pair_scores(
        self,
        slots: Tensor,
        where: Tensor,
        modulated: Tensor,
        locations: Tensor,
        beta_q: Tensor,
        beta_s: Tensor,
    ) -> Tensor:
        """Selection scores of every (location, edge) pair from the cached unit rows.

        Each distinct edge is read once; key products run in BF16 with FP32 output.
        The selected pairs' logits are recomputed exactly by ``lookup``.
        """
        unique, inverse = torch.unique(slots, return_inverse=True)
        modulated = modulated.to(torch.bfloat16)
        # Pairs grouped by edge, so every chunk of edges owns one contiguous run of pairs.
        order = torch.argsort(inverse)
        inverse, where, slots = inverse[order], where[order], slots[order]
        starts = torch.arange(0, unique.numel(), PAIR_CHUNK, device=slots.device)
        bounds = torch.searchsorted(inverse, starts).tolist() + [inverse.numel()]
        scores = torch.empty(slots.shape[0], device=self.device)
        for index, start in enumerate(starts.tolist()):
            low, high = bounds[index], bounds[index + 1]
            chunk = unique[start : start + PAIR_CHUNK]
            keys = (self.table.key_unit[chunk] @ modulated.T).float()
            near = self.table.departure_unit[chunk] @ locations.T
            row, column = inverse[low:high] - start, where[low:high]
            scores[low:high] = (
                beta_q * keys[row, column]
                + beta_s * near[row, column]
                + self.table.bias[slots[low:high]]
            )
        return torch.empty_like(scores).scatter_(0, order, scores)


class TransducerBank(EdgeStore, nn.Module):
    """Dense lookup controls (trained with the model) plus the packed sparse edge table.

    This is the trained base: the one knowledge base that learns, grows and
    syncs. The lookup itself (scoring, the sink, the blend) lives here because
    its controls are model parameters; it searches whatever the session mounted.
    """

    identity = "continuous-state-transducer-v10"

    def __init__(self, geometry: TransducerGeometry | None = None) -> None:
        super().__init__()
        self.geometry = geometry or TransducerGeometry()
        g = self.geometry
        self.modulation = nn.Parameter(torch.zeros(g.query_dim, g.location_dim))
        self.log_beta_location = nn.Parameter(torch.zeros(()))
        self.log_beta_query = nn.Parameter(torch.zeros(()))
        self.sink_direction = nn.Parameter(torch.zeros(g.query_dim))
        # The sink bias starts at the mean pair score of a fresh bank (zero).
        self.sink_bias = nn.Parameter(torch.zeros(()))
        self.sink_value = nn.Parameter(torch.zeros(g.value_dim))
        self.base_location = nn.Parameter(torch.randn(g.location_dim) * 0.01)
        self.metadata_location = nn.Linear(g.query_dim, g.location_dim, bias=False)
        self.table = EdgeTable(g, self.sink_value.device)
        self.bank_id = uuid.uuid4().hex
        self.version = 0
        self.learned_values = False
        self.masters = MasterCache(g, self.sink_value.device)
        self._projection: Tensor | None = None

    def _apply(self, fn, recurse: bool = True):
        result = super()._apply(fn, recurse)
        self.table.to(self.sink_value.device)
        self.masters.to(self.sink_value.device)
        if self._projection is not None:
            self._projection = self._projection.to(self.sink_value.device)
        return result

    @property
    def device(self) -> torch.device:
        return self.sink_value.device

    def switch_to_learned(self) -> None:
        """Values become trained parameters; written-mass bookkeeping ends."""
        if self.learned_values:
            raise RuntimeError("Value optimizer was already activated")
        self.learned_values = True
        self.table.write_mass.zero_()
        self.masters.clear(("value",))
        self._changed()

    # ------------------------------------------------------------------ rows

    def retire(self, ids: Tensor | list[int]) -> None:
        ids = torch.as_tensor(ids, dtype=torch.int64)
        if not ids.numel():
            return
        self.table.release(self.table.slots(ids))
        self.masters.drop(ids.tolist())
        self._changed()

    def publish_addresses(
        self, name: str, ids: Tensor, values: Tensor, *, step: int, worker: int
    ) -> None:
        if name not in ADDRESS_FIELDS:
            raise KeyError(name)
        device_ids = ids.to(self.device)
        self.table.write_address(
            name,
            self.table.slots(ids),
            pack_fp8(
                unit(values.to(self.device).float()), row_seeds(device_ids, name, step, worker)
            ),
        )
        self._changed()

    def publish_values(self, ids: Tensor, values: Tensor, *, step: int, worker: int) -> None:
        device_ids = ids.to(self.device)
        self.table.write_values(
            self.table.slots(ids),
            pack_fp4(values.to(self.device).float(), row_seeds(device_ids, "value", step, worker)),
        )
        self._changed()

    def publish_bias(self, ids: Tensor, values: Tensor) -> None:
        if not bool(torch.isfinite(values).all()):
            raise FloatingPointError("Nonfinite bias publication")
        self.table.bias[self.table.slots(ids)] = values.to(self.device).float()
        self._changed()

    def set_write_mass(self, ids: Tensor, mass: Tensor, *, step: int) -> None:
        slots = self.table.slots(ids)
        self.table.write_mass[slots] = mass.to(self.device).float()
        self.table.last_write[slots] = step

    def _fields(self, name: str, slots: Tensor, ids: list[int] | None) -> Tensor:
        """Decoded rows; with ``ids`` (trained rows), straight-through to the FP32 masters."""
        decoded = self.table.decode(name, slots)
        if ids is None or (name == "value" and not self.learned_values):
            return decoded
        leaf = self.masters.leaf(name, ids, decoded)
        return straight_through(leaf.reshape(decoded.shape), decoded)

    # --------------------------------------------------------------- search

    def mount(
        self,
        bases: Mapping[str, tuple[EdgeStore, float]] | None = None,
        *,
        trained: float = 1.0,
    ) -> tuple[Mounted, ...]:
        """The bases one session searches: this bank (at ``trained``) and ``bases`` by name.

        A weight of 0 leaves a base out. Mounted bases must share this bank's
        geometry (locations, queries, values and the content projection's seed).
        """
        g = self.geometry
        if bases and TRAINED in bases:
            raise ValueError(f"{TRAINED!r} names the trained base")
        mounted = []
        for name, (base, weight) in {TRAINED: (self, trained), **(bases or {})}.items():
            if not math.isfinite(weight) or weight < 0:
                raise ValueError("Blend weights are finite and nonnegative; 0 unmounts a base")
            other = base.geometry
            if (other.location_dim, other.query_dim, other.value_dim, other.seed) != (
                g.location_dim,
                g.query_dim,
                g.value_dim,
                g.seed,
            ):
                raise ValueError(f"Base {name!r} has a different geometry than this model's bank")
            if base.device != self.device:
                raise ValueError(f"Base {name!r} is not on the model's device")
            if weight > 0:
                mounted.append(Mounted(name, base, float(weight)))
        return tuple(mounted)

    def initial_agenda(
        self, metadata: Tensor, mounted: tuple[Mounted, ...] | None = None
    ) -> SearchAgenda:
        if metadata.shape != (self.geometry.query_dim,):
            raise ValueError("Metadata prefix must supply a query-width vector")
        start = self.metadata_start(metadata)
        return SearchAgenda(
            locations=[start], state=start, mounted=self.mount() if mounted is None else mounted
        )

    def metadata_start(self, metadata: Tensor) -> Tensor:
        """Where a metadata prefix places a search, or an ingested item: the one metadata map."""
        return unit(
            self.base_location.float()
            + nn.functional.linear(metadata.float(), self.metadata_location.weight.float())
        )

    def refresh(self, agenda: SearchAgenda) -> None:
        """Rescan the departures of every mounted base that changed since its cache was built."""
        for b, mounted in enumerate(agenda.mounted):
            if agenda.versions[b] != mounted.base.version:
                agenda.shortlists[b], agenda.bands[b] = mounted.base.shortlists(agenda.locations)
                agenda.versions[b] = mounted.base.version

    def content_location(self, values: Tensor) -> Tensor:
        """normalize(Pi h): the fixed content-derived location of a value (rows or one)."""
        g = self.geometry
        if values.shape[-1] != g.value_dim or not bool(torch.isfinite(values).all()):
            raise ValueError("Content locations need finite full-width values")
        if self._projection is None:
            generator = torch.Generator(device="cpu").manual_seed(g.seed)
            self._projection = (
                torch.randn(g.location_dim, g.value_dim, generator=generator)
                / math.sqrt(g.value_dim)
            ).to(self.device)
        return packed_unit(values.float() @ self._projection.T)

    def modulated(self, query: Tensor, locations: Tensor) -> Tensor:
        """The query read through each location: normalize(q * (1 + tanh(M s)))."""
        return packed_unit(query[None] * (1 + torch.tanh(locations @ self.modulation.float().T)))

    def _select(
        self,
        agenda: SearchAgenda,
        modulated: Tensor,
        locations: Tensor,
        beta_q: Tensor,
        beta_s: Tensor,
        limit: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """The best ``limit`` pairs over every mounted base: slots, locations and owning bases.

        Every base's shortlisted pairs are scored together, each base's scores
        shifted by its log blend weight, and ranked as one list.
        """
        scored, slots, where, owner = [], [], [], []
        for b, mounted in enumerate(agenda.mounted):
            lists = agenda.shortlists[b]
            every_slot = torch.cat(lists)
            if not every_slot.numel():
                continue
            every_location = torch.arange(len(lists), device=self.device)
            every_location = every_location.repeat_interleave(lists[0].numel())
            scores = mounted.base.pair_scores(
                every_slot, every_location, modulated, locations, beta_q, beta_s
            )
            scored.append(scores + mounted.log_weight)
            slots.append(every_slot)
            where.append(every_location)
            owner.append(torch.full_like(every_slot, b))
        if not scored:
            empty = torch.empty(0, dtype=torch.int64, device=self.device)
            return empty, empty, empty
        scores = torch.cat(scored)
        top = torch.topk(scores, min(limit, scores.numel())).indices
        return torch.cat(slots)[top], torch.cat(where)[top], torch.cat(owner)[top]

    def lookup(
        self,
        query: Tensor,
        agenda: SearchAgenda,
        *,
        training: bool = False,
        hindsight: Tensor | None = None,
        overlay: float = 0.0,
        generator: torch.Generator | None = None,
        train_rows: bool | None = None,
    ) -> Lookup | None:
        """Score every shortlisted (agenda location, edge) pair, keep the best, blend.

        The pairs of every mounted base compete in one softmax with the one
        sink; each base's scores carry its log blend weight. ``train_rows``
        (default: ``training``) gives the selected trained-base rows gradient
        masters (ingested rows never have any); hindsight-only lookups pass
        False so the unused bank trains nothing.
        """
        g = self.geometry
        if agenda.lookups >= g.max_lookups:
            return None
        if query.shape != (g.query_dim,) or not bool(torch.isfinite(query).all()):
            raise ValueError("Lookup requires one finite query-width vector")
        if not 0 <= overlay <= 1 or (overlay > 0 and hindsight is None):
            raise ValueError("Overlay requires a hindsight value and a bounded weight")
        train_rows = training if train_rows is None else train_rows
        self.refresh(agenda)
        q = unit(query.float())
        locations = torch.stack(agenda.locations).float()  # unit on entry to the agenda
        modulated = self.modulated(q, locations)
        # Scoring controls are FP32 whatever the model's storage precision.
        beta_q, beta_s = self.log_beta_query.float().exp(), self.log_beta_location.float().exp()
        own = next((b for b, m in enumerate(agenda.mounted) if m.base is self), None)
        # Exploration reaches only the trained base: ingested bases are fixed.
        explore = bool(
            training
            and g.exploration
            and own is not None
            and any(band.numel() for band in agenda.bands[own])
        )
        limit = g.selected - (g.exploration if explore else 0)
        with torch.no_grad():
            slots, where, owner = self._select(agenda, modulated, locations, beta_q, beta_s, limit)
            if explore:
                slots, where, owner = self._explore(agenda, own, slots, where, owner, generator)
            # Grouped by base, so each base's selected rows are one contiguous run.
            order = torch.argsort(owner, stable=True)
            slots, where, owner = slots[order], where[order], owner[order]
        sink_logit = self.sink_bias.float() + self.sink_direction.float() @ q
        start = agenda.state.float()  # unit: a landing or the initial location
        pairs: tuple[tuple[int, str, int], ...] = ()
        edge_ids = torch.empty(0, dtype=torch.int64, device=self.device)
        edge_weights = torch.empty(0, device=self.device)
        base_edges = {}
        if slots.numel():
            runs = torch.bincount(owner, minlength=len(agenda.mounted)).tolist()
            ids, keys, departures, biases, values, destinations = [], [], [], [], [], []
            low = 0
            for b, run in enumerate(runs):
                if not run:
                    continue
                base, chosen = agenda.mounted[b].base, slots[low : low + run]
                low += run
                ids.append(base.table.ids_of(chosen))
                # Only the trained base's rows have gradient masters.
                trained = ids[-1].tolist() if base is self and train_rows else None

                def read(name: str, base=base, chosen=chosen, trained=trained) -> Tensor:
                    if base is self:
                        return self._fields(name, chosen, trained)
                    return base.table.decode(name, chosen)

                keys.append(read("key"))
                departures.append(read("departure"))
                biases.append(read("bias").reshape(-1))
                values.append(read("value"))
                destinations.append(read("destination"))
            ids = torch.cat(ids)
            names = [agenda.mounted[b].name for b in owner.tolist()]
            pairs = tuple(zip(where.tolist(), names, ids.tolist(), strict=True))
            keys = packed_unit(torch.cat(keys))
            departures = packed_unit(torch.cat(departures))
            shifts = torch.tensor([m.log_weight for m in agenda.mounted], device=self.device)
            logits = (
                beta_q * (keys * modulated[where]).sum(-1)
                + beta_s * (departures * locations[where]).sum(-1)
                + torch.cat(biases)
                + shifts[owner]
            )
            weights = torch.softmax(torch.cat((logits, sink_logit[None])).float(), dim=0)
            pair_weights = weights[:-1]
            values = torch.cat(values)
            destinations = packed_unit(torch.cat(destinations))
            bank_value = weights[-1] * self.sink_value.float() + pair_weights @ values
            landing_raw = weights[-1] * start + pair_weights @ destinations
            for b, mount in enumerate(agenda.mounted):
                selected = owner == b
                distinct, inverse = torch.unique(ids[selected], return_inverse=True)
                total = torch.zeros(distinct.numel(), device=self.device).index_add(
                    0, inverse, pair_weights[selected])
                base_edges[mount.name] = (distinct, total)
            if own is not None and bool((owner == own).any()):
                mine = owner == own
                edge_ids, inverse = torch.unique(ids[mine], return_inverse=True)
                edge_weights = torch.zeros(edge_ids.numel(), device=self.device).index_add(
                    0, inverse, pair_weights[mine]
                )
        else:
            logits = sink_logit.new_zeros(0)
            weights = torch.ones(1, device=self.device) + 0 * sink_logit
            bank_value = weights[-1] * self.sink_value.float()
            landing_raw = weights[-1] * start
        value, landing_mix = bank_value, landing_raw
        destination = None
        if hindsight is not None:
            destination = self.content_location(hindsight.detach())
        if overlay > 0:
            value = (1 - overlay) * bank_value + overlay * hindsight.float()
            landing_mix = (1 - overlay) * landing_raw + overlay * destination
        return Lookup(
            value=value,
            landing=packed_unit(landing_mix),
            bank_value=bank_value,
            bank_landing_raw=landing_raw,
            weights=weights,
            pair_logits=logits,
            sink_weight=weights[-1],
            pairs=pairs,
            edge_ids=edge_ids,
            edge_weights=edge_weights,
            start=start,
            start_key=self.modulated(q, start[None])[0],
            query=q,
            versions=tuple(m.base.version for m in agenda.mounted),
            base_edges=base_edges,
        )

    def _explore(
        self,
        agenda: SearchAgenda,
        own: int,
        slots: Tensor,
        where: Tensor,
        owner: Tensor,
        generator: torch.Generator | None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Add random pairs from the rank band just past the trained base's shortlists."""
        bands = agenda.bands[own]
        chosen = set(zip(where.tolist(), slots.tolist(), owner.tolist(), strict=True))
        added_slots, added_where = [], []
        for _ in range(self.geometry.exploration):
            index = int(torch.randint(len(bands), (1,), generator=generator))
            band = bands[index].tolist()
            band = [slot for slot in band if (index, slot, own) not in chosen]
            if not band:
                continue
            slot = band[int(torch.randint(len(band), (1,), generator=generator))]
            chosen.add((index, slot, own))
            added_slots.append(slot)
            added_where.append(index)
        if not added_slots:
            return slots, where, owner
        device, count = slots.device, len(added_slots)
        return (
            torch.cat((slots, torch.tensor(added_slots, dtype=slots.dtype, device=device))),
            torch.cat((where, torch.tensor(added_where, dtype=where.dtype, device=device))),
            torch.cat((owner, torch.full((count,), own, dtype=owner.dtype, device=device))),
        )

    def commit(self, agenda: SearchAgenda, lookup: Lookup) -> bool:
        """Publish a lookup that C accepted: move the search; append a distinct location."""
        if agenda.lookups >= self.geometry.max_lookups:
            raise RuntimeError("Lookup cap reached before publication")
        current = tuple(m.base.version for m in agenda.mounted)
        if lookup.versions != current or tuple(agenda.versions) != current:
            raise RuntimeError("Stale transducer lookup cannot commit across a bank publication")
        agenda.lookups += 1
        agenda.state = lookup.landing
        with torch.no_grad():
            known = torch.stack(agenda.locations).float()
            if float((known @ lookup.landing.float()).max()) >= 0.99:
                return False
        agenda.locations.append(lookup.landing)
        agenda.births.append(agenda.lookups)
        for b, mounted in enumerate(agenda.mounted):
            shortlist, band = mounted.base.shortlists([lookup.landing])
            agenda.shortlists[b].append(shortlist[0])
            agenda.bands[b].append(band[0])
        return True

    # ------------------------------------------------------------- snapshots

    def export_rows(self, ids: list[int]) -> dict[str, Tensor]:
        return self.table.state(ids)

    def import_rows(self, rows: dict[str, Tensor], *, retired: list[int], next_id: int) -> None:
        """Install a published change set: retire, overwrite and add packed rows."""
        gone = [edge for edge in retired if edge in self.table.slot_of]
        if gone:
            self.retire(gone)
        if rows["ids"].numel():
            count = self.table.count + sum(
                int(edge) not in self.table.slot_of for edge in rows["ids"].tolist()
            )
            if count > self.geometry.max_edges or count * self.geometry.row_bytes > (
                self.geometry.byte_budget
            ):
                raise BankFull("Published rows exceed the canonical edge or byte budget")
            self.table.upsert(rows)
            self.masters.drop(rows["ids"].tolist())
        self.table.next_id = max(self.table.next_id, next_id)
        self._changed()

    def state_snapshot(self) -> dict[str, Any]:
        """Compact, tensor-only snapshot of the live rows (loadable with weights_only)."""
        return {
            "identity": self.identity,
            "bank_id": self.bank_id,
            "geometry": dict(vars(self.geometry)),
            "version": self.version,
            "next_edge_id": self.table.next_id,
            "learned_values": self.learned_values,
            "rows": self.table.state(),
        }

    def load_snapshot(self, data: dict[str, Any]) -> None:
        if data.get("identity") != self.identity or data["geometry"] != vars(self.geometry):
            raise ValueError("Only exact v10 packed-edge snapshot identities are accepted")
        rows = data["rows"]
        count = int(rows["ids"].numel())
        if count > self.geometry.max_edges or count * self.geometry.row_bytes > (
            self.geometry.byte_budget
        ):
            raise ValueError("Snapshot exceeds canonical edge or byte budget")
        self.table.load(rows)
        next_id = int(data["next_edge_id"])
        canonical = rows["ids"][rows["ids"] < PROVISIONAL_BASE]
        if canonical.numel() and next_id <= int(canonical.max()):
            raise ValueError("Invalid canonical edge ID high-water mark")
        self.table.next_id = next_id
        self.bank_id = str(data["bank_id"])
        self.learned_values = bool(data["learned_values"])
        # Past every version this bank has had, so no cached shortlist survives the load.
        self.version = max(self.version, int(data["version"])) + 1
        self.masters.clear()
