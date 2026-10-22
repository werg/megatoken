"""v10 agenda-wide continuous-state transducer bank.

Edges live in a columnar table on the model's device: one tensor per packed
field, indexed by slot. Edge IDs are stable; slots are an internal detail and
can be reused after retirement. There is no query-key index, global candidate
search or semantic-KV materialization: every lookup scores the departure
shortlists of its session's agenda locations and keeps the best pairs.

All structural and parameter changes happen at window boundaries (optimizer
steps, coordinator merges, write-out after backward). Each change bumps the
bank version; agendas recompute their cached shortlists when it moves.
"""

from __future__ import annotations

import math
import uuid
from collections import OrderedDict
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
    row_seeds,
    straight_through,
    unit,
)

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


class EdgeTable:
    """Columnar packed storage. Not an nn.Module: rows are trained by the sparse optimizer."""

    def __init__(self, geometry: TransducerGeometry, device: torch.device, capacity: int = 256):
        self.geometry = geometry
        self.device = torch.device(device)
        self.capacity = 0
        self.slot_of: dict[int, int] = {}
        self.free: list[int] = []
        self.high_water = 0
        self.next_id = 1
        self._allocate(capacity)

    def _empty(self, capacity: int) -> dict[str, Tensor]:
        g, d = self.geometry, self.device
        fp8 = torch.float8_e4m3fn

        def scales(dim: int) -> Tensor:
            return torch.ones(capacity, -(-dim // 128), device=d)

        return {
            "ids": torch.full((capacity,), -1, dtype=torch.int64, device=d),
            "departure_codes": torch.zeros(capacity, g.location_dim, device=d).to(fp8),
            "departure_scales": scales(g.location_dim),
            "key_codes": torch.zeros(capacity, g.query_dim, device=d).to(fp8),
            "key_scales": scales(g.query_dim),
            "destination_codes": torch.zeros(capacity, g.location_dim, device=d).to(fp8),
            "destination_scales": scales(g.location_dim),
            "value_codes": torch.zeros(capacity, -(-g.value_dim // 2), dtype=torch.uint8, device=d),
            "value_groups": torch.ones(capacity, -(-g.value_dim // 16), device=d).to(fp8),
            "value_rows": torch.ones(capacity, device=d),
            "value_norm2": torch.zeros(capacity, device=d),
            "bias": torch.zeros(capacity, device=d),
            "write_mass": torch.zeros(capacity, device=d),
            "birth": torch.zeros(capacity, dtype=torch.int64, device=d),
            "last_write": torch.zeros(capacity, dtype=torch.int64, device=d),
        }

    def _allocate(self, capacity: int) -> None:
        fresh = self._empty(capacity)
        if self.capacity:
            for name, tensor in fresh.items():
                tensor[: self.capacity] = getattr(self, name)
        for name, tensor in fresh.items():
            setattr(self, name, tensor)
        self.capacity = capacity

    def to(self, device: torch.device) -> None:
        if torch.device(device) == self.device:
            return
        for name in self._empty(0):
            setattr(self, name, getattr(self, name).to(device))
        self.device = torch.device(device)

    @property
    def count(self) -> int:
        return len(self.slot_of)

    def slots(self, ids: Tensor | list[int]) -> Tensor:
        try:
            values = [self.slot_of[int(i)] for i in torch.as_tensor(ids).tolist()]
        except KeyError as missing:
            raise KeyError(f"Edge {missing} is not in the bank") from None
        return torch.tensor(values, dtype=torch.int64, device=self.device)

    def alive_slots(self) -> Tensor:
        return torch.tensor(sorted(self.slot_of.values()), dtype=torch.int64, device=self.device)

    def ids_of(self, slots: Tensor) -> Tensor:
        return self.ids[slots]

    def claim(self, count: int) -> Tensor:
        reused = [self.free.pop() for _ in range(min(count, len(self.free)))]
        needed = count - len(reused)
        if self.high_water + needed > self.capacity:
            self._allocate(max(self.capacity * 2, self.high_water + needed))
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
        return {name: getattr(self, name)[slots].cpu() for name in self._empty(0)}

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

    def load(self, state: dict[str, Tensor]) -> None:
        count = int(state["ids"].numel())
        self.capacity = 0
        self.free = []
        self.slot_of = {}
        self._allocate(max(count, 256))
        for name, tensor in state.items():
            getattr(self, name)[:count] = tensor.to(self.device)
        self.slot_of = {int(edge): slot for slot, edge in enumerate(state["ids"].tolist())}
        if len(self.slot_of) != count or any(edge <= 0 for edge in self.slot_of):
            raise ValueError("Invalid or duplicate edge IDs in a transducer snapshot")
        self.high_water = count


@dataclass
class SearchAgenda:
    """One session's reached locations and their cached departure shortlists.

    ``locations`` are the distinct appended locations; ``state`` is where the
    search actually is now (the last landing), which the sink holds still.
    """

    locations: list[Tensor]
    state: Tensor
    shortlists: list[Tensor] = field(default_factory=list)
    bands: list[Tensor] = field(default_factory=list)
    version: int = -1
    lookups: int = 0
    births: list[int] = field(default_factory=list)  # lookup count when each location joined

    def __post_init__(self) -> None:
        if not self.births:
            self.births = [0] * len(self.locations)

    def copy(self) -> SearchAgenda:
        return SearchAgenda(
            list(self.locations),
            self.state,
            list(self.shortlists),
            list(self.bands),
            self.version,
            self.lookups,
            list(self.births),
        )

    def publish(self, other: SearchAgenda) -> None:
        self.locations, self.state = other.locations, other.state
        self.shortlists, self.bands = other.shortlists, other.bands
        self.version, self.lookups = other.version, other.lookups
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
    bank_value: Tensor  # the bank's own blend, sink included
    bank_landing_raw: Tensor  # the bank's unnormalized landing, lambda_t
    weights: Tensor  # pair weights followed by the sink weight
    pair_logits: Tensor
    sink_weight: Tensor
    pairs: tuple[tuple[int, int], ...]  # (agenda location index, edge ID)
    edge_ids: Tensor  # distinct selected edges
    edge_weights: Tensor  # each edge's total weight over its pairs, omega_i
    query: Tensor  # unit query
    start: Tensor  # where the search was (the sink's hold-still destination)
    start_key: Tensor  # the query read through the start location
    hindsight_destination: Tensor | None
    overlay: float
    bank_version: int

    @property
    def candidates(self) -> tuple[int, ...]:
        return tuple(edge for _, edge in self.pairs)


class TransducerBank(nn.Module):
    """Dense lookup controls (trained with the model) plus the packed sparse edge table."""

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
        # FP32 working masters of trained fields, keyed by (edge ID, field).
        self.masters: OrderedDict[tuple[int, str], Tensor] = OrderedDict()
        self._projection: Tensor | None = None

    def _apply(self, fn, recurse: bool = True):
        result = super()._apply(fn, recurse)
        self.table.to(self.sink_value.device)
        if self._projection is not None:
            self._projection = self._projection.to(self.sink_value.device)
        return result

    @property
    def device(self) -> torch.device:
        return self.sink_value.device

    @property
    def edge_count(self) -> int:
        return self.table.count

    @property
    def packed_bytes(self) -> int:
        return self.table.count * self.geometry.row_bytes

    def edge_ids(self) -> Tensor:
        """Live edge IDs in increasing order."""
        return self.table.ids_of(self.table.alive_slots()).sort().values

    def __contains__(self, edge_id: int) -> bool:
        return int(edge_id) in self.table.slot_of

    def _changed(self) -> None:
        self.version += 1

    # ------------------------------------------------------------------ rows

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
        if self.table.count + count > g.max_edges:
            raise BankFull("Transducer has reached its hard edge ceiling")
        if self.packed_bytes + count * g.row_bytes > g.byte_budget:
            raise BankFull("Packed-byte budget exceeded")
        if ids is None:
            ids = torch.arange(self.table.next_id, self.table.next_id + count)
        ids = ids.to(torch.int64).cpu()
        if (
            int(ids.min()) < self.table.next_id
            or len(set(ids.tolist())) != count
            or any(int(i) in self.table.slot_of for i in ids)
        ):
            raise ValueError("Canonical edge IDs must increase and never be reused")
        slots = self.table.claim(count)
        device_ids = ids.to(self.device)
        self.table.ids[slots] = device_ids
        self.table.slot_of.update(zip(ids.tolist(), slots.tolist(), strict=True))
        if not provisional:
            self.table.next_id = int(ids.max()) + 1
        for name, values in (
            ("departure", departure),
            ("key", key),
            ("destination", destination),
        ):
            self.table.write_address(
                name,
                slots,
                pack_fp8(
                    unit(values.to(self.device).float()), row_seeds(device_ids, name, step, worker)
                ),
            )
        self.table.write_values(
            slots,
            pack_fp4(value.to(self.device).float(), row_seeds(device_ids, "value", step, worker)),
        )
        self.table.bias[slots] = bias.to(self.device).float()
        self.table.write_mass[slots] = write_mass.to(self.device).float()
        self.table.birth[slots] = step
        self.table.last_write[slots] = step
        self._changed()
        return ids

    def retire(self, ids: Tensor | list[int]) -> None:
        ids = torch.as_tensor(ids, dtype=torch.int64)
        if not ids.numel():
            return
        gone = set(ids.tolist())
        self.table.release(self.table.slots(ids))
        for identity in [k for k in self.masters if k[0] in gone]:
            del self.masters[identity]
        self._changed()

    def decode(self, name: str, ids: Tensor | list[int]) -> Tensor:
        return self.table.decode(name, self.table.slots(ids))

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

    def publish_values(self, ids: Tensor, values: Tensor, *, step: int, worker: int) -> None:
        device_ids = ids.to(self.device)
        self.table.write_values(
            self.table.slots(ids),
            pack_fp4(values.to(self.device).float(), row_seeds(device_ids, "value", step, worker)),
        )

    def publish_bias(self, ids: Tensor, values: Tensor) -> None:
        if not bool(torch.isfinite(values).all()):
            raise FloatingPointError("Nonfinite bias publication")
        self.table.bias[self.table.slots(ids)] = values.to(self.device).float()

    def set_write_mass(self, ids: Tensor, mass: Tensor, *, step: int) -> None:
        slots = self.table.slots(ids)
        self.table.write_mass[slots] = mass.to(self.device).float()
        self.table.last_write[slots] = step

    def master(self, edge_id: int, name: str) -> Tensor:
        """The FP32 trained master of one field, created from its packed row on first use."""
        identity = edge_id, name
        leaf = self.masters.get(identity)
        if leaf is None:
            decoded = self.decode(name, [edge_id])[0]
            leaf = decoded.detach().float().clone().requires_grad_()
            self.masters[identity] = leaf
        else:
            self.masters.move_to_end(identity)
        return leaf

    def _fields(self, name: str, slots: Tensor, *, train: bool) -> Tensor:
        """Decoded rows; when trained, straight-through to the FP32 masters."""
        decoded = self.table.decode(name, slots)
        trained = train and (name != "value" or self.learned_values)
        if not trained:
            return decoded
        ids = self.table.ids_of(slots).tolist()
        masters = torch.stack([self.master(edge, name) for edge in ids])
        return straight_through(masters, decoded)

    # --------------------------------------------------------------- search

    def initial_agenda(self, metadata: Tensor) -> SearchAgenda:
        if metadata.shape != (self.geometry.query_dim,):
            raise ValueError("Metadata prefix must supply a query-width vector")
        start = unit(
            self.base_location.float()
            + nn.functional.linear(metadata.float(), self.metadata_location.weight.float())
        )
        return SearchAgenda(locations=[start], state=start)

    def agenda_from(self, locations: list[Tensor], state: Tensor, lookups: int) -> SearchAgenda:
        agenda = SearchAgenda(
            [unit(x.to(self.device).float()) for x in locations],
            unit(state.to(self.device).float()),
            lookups=lookups,
        )
        self.refresh(agenda)
        return agenda

    def nearest(self, locations: Tensor, count: int) -> Tensor:
        """Exact scan of the packed departures: [rows, count] slots, -1 where absent."""
        alive = self.table.alive_slots()
        rows = locations.shape[0]
        if not alive.numel():
            return torch.full((rows, 0), -1, dtype=torch.int64, device=self.device)
        query = unit(locations.detach().float())
        best_scores: Tensor | None = None
        best_slots: Tensor | None = None
        for chunk in alive.split(65536):
            departures = unit(self.table.decode("departure", chunk))
            scores = query @ departures.T
            if best_scores is not None:
                scores = torch.cat((best_scores, scores), dim=1)
                chunk = torch.cat((best_slots, chunk[None].expand(rows, -1)), dim=1)
            else:
                chunk = chunk[None].expand(rows, -1)
            keep = min(count, scores.shape[1])
            top = torch.topk(scores, keep, dim=1)
            best_scores = top.values
            best_slots = torch.gather(chunk, 1, top.indices)
        return best_slots

    def _shortlists(self, locations: list[Tensor]) -> tuple[list[Tensor], list[Tensor]]:
        g = self.geometry
        with torch.no_grad():
            ranked = self.nearest(torch.stack(locations), 2 * g.neighbors)
        return (
            [row[: g.neighbors] for row in ranked],
            [row[g.neighbors :] for row in ranked],
        )

    def refresh(self, agenda: SearchAgenda) -> None:
        if agenda.version != self.version:
            agenda.shortlists, agenda.bands = self._shortlists(agenda.locations)
            agenda.version = self.version

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
        return unit(values.float() @ self._projection.T)

    def modulated(self, query: Tensor, locations: Tensor) -> Tensor:
        """The query read through each location: normalize(q * (1 + tanh(M s)))."""
        return unit(query[None] * (1 + torch.tanh(locations @ self.modulation.float().T)))

    def _pair_scores(
        self,
        slots: Tensor,
        where: Tensor,
        modulated: Tensor,
        locations: Tensor,
        beta_q: Tensor,
        beta_s: Tensor,
    ) -> Tensor:
        """Scores of every (location, edge) pair; each distinct edge is decoded once."""
        unique, inverse = torch.unique(slots, return_inverse=True)
        scores = torch.empty(slots.shape[0], device=self.device)
        for start in range(0, unique.numel(), PAIR_CHUNK):
            chunk = unique[start : start + PAIR_CHUNK]
            keys = unit(self.table.decode("key", chunk)) @ modulated.T  # [edges, locations]
            near = unit(self.table.decode("departure", chunk)) @ locations.T
            mine = (inverse >= start) & (inverse < start + chunk.numel())
            row, column = inverse[mine] - start, where[mine]
            scores[mine] = (
                beta_q * keys[row, column]
                + beta_s * near[row, column]
                + self.table.bias[slots[mine]]
            )
        return scores

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

        ``train_rows`` (default: ``training``) gives the selected rows gradient
        masters; hindsight-only lookups pass False so the unused bank trains nothing.
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
        locations = unit(torch.stack(agenda.locations).float())
        modulated = self.modulated(q, locations)
        # Scoring controls are FP32 whatever the model's storage precision.
        beta_q, beta_s = self.log_beta_query.float().exp(), self.log_beta_location.float().exp()
        limit = g.selected - (g.exploration if training else 0)
        pair_locations: list[int] = []
        pair_slots: list[int] = []
        with torch.no_grad():
            every_slot = torch.cat(agenda.shortlists)
            every_location = torch.cat(
                [torch.full_like(row, index) for index, row in enumerate(agenda.shortlists)]
            )
            present = every_slot >= 0
            every_slot, every_location = every_slot[present], every_location[present]
            if every_slot.numel():
                scores = self._pair_scores(
                    every_slot, every_location, modulated, locations, beta_q, beta_s
                )
                top = torch.topk(scores, min(limit, scores.numel())).indices
                pair_locations = every_location[top].tolist()
                pair_slots = every_slot[top].tolist()
            if training and g.exploration and agenda.bands:
                chosen = set(zip(pair_locations, pair_slots, strict=True))
                for _ in range(g.exploration):
                    index = int(torch.randint(len(agenda.bands), (1,), generator=generator))
                    band = agenda.bands[index]
                    band = [s for s in band[band >= 0].tolist() if (index, s) not in chosen]
                    if not band:
                        continue
                    slot = band[int(torch.randint(len(band), (1,), generator=generator))]
                    chosen.add((index, slot))
                    pair_locations.append(index)
                    pair_slots.append(slot)
        sink_logit = self.sink_bias.float() + self.sink_direction.float() @ q
        start = unit(agenda.state.float())
        if pair_slots:
            slots = torch.tensor(pair_slots, dtype=torch.int64, device=self.device)
            where = torch.tensor(pair_locations, dtype=torch.int64, device=self.device)
            keys = unit(self._fields("key", slots, train=train_rows))
            departures = unit(self._fields("departure", slots, train=train_rows))
            biases = self._fields("bias", slots, train=train_rows).reshape(-1)
            logits = (
                beta_q * (keys * modulated[where]).sum(-1)
                + beta_s * (departures * locations[where]).sum(-1)
                + biases
            )
            weights = torch.softmax(torch.cat((logits, sink_logit[None])).float(), dim=0)
            pair_weights = weights[:-1]
            values = self._fields("value", slots, train=train_rows)
            destinations = unit(self._fields("destination", slots, train=train_rows))
            bank_value = weights[-1] * self.sink_value.float() + pair_weights @ values
            landing_raw = weights[-1] * start + pair_weights @ destinations
            edge_ids, inverse = torch.unique(self.table.ids_of(slots), return_inverse=True)
            edge_weights = torch.zeros(edge_ids.numel(), device=self.device).index_add(
                0, inverse, pair_weights
            )
        else:
            logits = sink_logit.new_zeros(0)
            weights = torch.ones(1, device=self.device) + 0 * sink_logit
            bank_value = weights[-1] * self.sink_value.float()
            landing_raw = weights[-1] * start
            edge_ids = torch.empty(0, dtype=torch.int64, device=self.device)
            edge_weights = torch.empty(0, device=self.device)
        value, landing_mix = bank_value, landing_raw
        destination = None
        if hindsight is not None:
            destination = self.content_location(hindsight.detach())
        if overlay > 0:
            value = (1 - overlay) * bank_value + overlay * hindsight.float()
            landing_mix = (1 - overlay) * landing_raw + overlay * destination
        ids = self.table.ids_of(torch.tensor(pair_slots, dtype=torch.int64, device=self.device))
        return Lookup(
            value=value,
            landing=unit(landing_mix),
            bank_value=bank_value,
            bank_landing_raw=landing_raw,
            weights=weights,
            pair_logits=logits,
            sink_weight=weights[-1],
            pairs=tuple(zip(pair_locations, ids.tolist(), strict=True)),
            edge_ids=edge_ids,
            edge_weights=edge_weights,
            query=q,
            start=start,
            start_key=self.modulated(q, start[None])[0],
            hindsight_destination=destination,
            overlay=overlay,
            bank_version=self.version,
        )

    def commit(self, agenda: SearchAgenda, lookup: Lookup) -> bool:
        """Publish a lookup that C accepted: move the search; append a distinct location."""
        if agenda.lookups >= self.geometry.max_lookups:
            raise RuntimeError("Lookup cap reached before publication")
        if lookup.bank_version != self.version or agenda.version != self.version:
            raise RuntimeError("Stale transducer lookup cannot commit across a bank publication")
        agenda.lookups += 1
        agenda.state = lookup.landing
        with torch.no_grad():
            known = unit(torch.stack(agenda.locations).float())
            if float((known @ unit(lookup.landing.float())).max()) >= 0.99:
                return False
        agenda.locations.append(lookup.landing)
        agenda.births.append(agenda.lookups)
        shortlist, band = self._shortlists([lookup.landing])
        agenda.shortlists.append(shortlist[0])
        agenda.bands.append(band[0])
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
            for edge in rows["ids"].tolist():
                for name in FIELDS:
                    self.masters.pop((edge, name), None)
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
        self.version = int(data["version"])
        self.masters.clear()
