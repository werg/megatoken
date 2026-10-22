"""v10 agenda-wide continuous-state transducer (executable exact reference).

The bank is deliberately distinct from the v9 point/pair and IVF stores. It
has no query-key index, global candidate search, or semantic-KV materialization.
The packed departure scan is the reference for a device-specific kernel.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor, nn

from .transducer_packing import (
    PackedField,
    pack_fp4,
    pack_fp8,
    rounding_seed,
    straight_through,
    unit,
)


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
        if self.value_dim % 512 and self.value_dim == 32768:
            raise ValueError("Wide bank values must be divisible into 512-wide rows")
        if not 0 <= self.exploration < self.selected:
            raise ValueError("Exploration reserves slots inside the selected pair budget")


@dataclass
class PackedEdge:
    id: int
    departure: PackedField
    key: PackedField
    value: PackedField
    destination: PackedField
    bias: float
    birth_step: int = 0
    write_mass: float = 0.0

    @property
    def nbytes(self) -> int:
        return (
            self.departure.nbytes
            + self.key.nbytes
            + self.value.nbytes
            + self.destination.nbytes
            + 12
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "departure": self.departure.to_dict(),
            "key": self.key.to_dict(),
            "value": self.value.to_dict(),
            "destination": self.destination.to_dict(),
            "bias": self.bias,
            "birth_step": self.birth_step,
            "write_mass": self.write_mass,
        }

    @classmethod
    def from_dict(cls, raw: dict) -> PackedEdge:
        return cls(
            id=int(raw["id"]),
            departure=PackedField.from_dict(raw["departure"]),
            key=PackedField.from_dict(raw["key"]),
            value=PackedField.from_dict(raw["value"]),
            destination=PackedField.from_dict(raw["destination"]),
            bias=float(raw["bias"]),
            birth_step=int(raw["birth_step"]),
            write_mass=float(raw["write_mass"]),
        )


@dataclass
class SearchAgenda:
    """One serialized search path; cached shortlists are invalidated at row barriers."""

    locations: list[Tensor]
    neighborhoods: list[tuple[int, ...]] = field(default_factory=list)
    version: int = -1
    lookups: int = 0

    @property
    def current(self) -> Tensor:
        return self.locations[-1]

    def detach_older_than(self, retained: int = 4) -> None:
        if retained < 1:
            raise ValueError("At least one differentiable location is required")
        self.locations[:-retained] = [x.detach() for x in self.locations[:-retained]]


@dataclass(frozen=True)
class Lookup:
    """Uncommitted, differentiable retrieval; cancellation cannot mutate an agenda."""

    value: Tensor
    landing: Tensor
    bank_value: Tensor
    bank_landing: Tensor
    bank_landing_raw: Tensor
    weights: Tensor
    sink_weight: Tensor
    pairs: tuple[tuple[int, int], ...]
    edge_weights: dict[int, Tensor]
    query: Tensor
    start: Tensor
    hindsight_destination: Tensor | None
    bank_version: int

    @property
    def candidates(self) -> tuple[int, ...]:
        return tuple(edge for _, edge in self.pairs)


class TransducerBank(nn.Module):
    """Model-owned controls plus canonical, packed, separately optimized sparse rows."""

    identity = "continuous-state-transducer-v10"

    def __init__(self, geometry: TransducerGeometry | None = None) -> None:
        super().__init__()
        self.geometry = geometry or TransducerGeometry()
        g = self.geometry
        self.modulation = nn.Parameter(torch.zeros(g.query_dim, g.location_dim))
        self.log_beta_location = nn.Parameter(torch.zeros(()))
        self.log_beta_query = nn.Parameter(torch.zeros(()))
        self.sink_direction = nn.Parameter(torch.zeros(g.query_dim))
        self.sink_bias = nn.Parameter(torch.zeros(()))
        self.sink_value = nn.Parameter(torch.zeros(g.value_dim))
        self.base_location = nn.Parameter(torch.randn(g.location_dim) * 0.01)
        self.metadata_location = nn.Linear(g.query_dim, g.location_dim, bias=False)
        self.rows: dict[int, PackedEdge] = {}
        self.next_edge_id = 1
        self.version = 0
        self.learned_values = False
        self._masters: dict[tuple[int, str], Tensor] = {}
        self._departure_ids: tuple[int, ...] = ()
        self._departures: Tensor | None = None
        self._table_version = -1
        self._projection: Tensor | None = None

    @property
    def device(self) -> torch.device:
        return self.sink_value.device

    @property
    def packed_bytes(self) -> int:
        return sum(row.nbytes for row in self.rows.values())

    def _invalidate(self) -> None:
        self.version += 1
        self._table_version = -1

    def initial_agenda(self, metadata: Tensor) -> SearchAgenda:
        if metadata.shape != (self.geometry.query_dim,):
            raise ValueError("Metadata prefix must supply a query-width vector")
        state = unit(self.base_location + self.metadata_location(metadata.float()))
        return SearchAgenda(locations=[state])

    def agenda_from_location(self, location: Tensor) -> SearchAgenda:
        if location.shape != (self.geometry.location_dim,):
            raise ValueError("Initial location has the wrong geometry")
        return SearchAgenda([unit(location.to(self.device).float())])

    def add_edge(
        self,
        *,
        departure: Tensor,
        key: Tensor,
        value: Tensor,
        destination: Tensor,
        bias: float = 0.0,
        id: int | None = None,
        step: int = 0,
        write_mass: float = 0.0,
        worker: int = 0,
    ) -> int:
        if step < 0 or not math.isfinite(bias) or write_mass < 0:
            raise ValueError("Invalid edge statistics or bias")
        if len(self.rows) >= self.geometry.max_edges:
            raise RuntimeError("Transducer has reached its hard edge ceiling")
        g = self.geometry
        if tuple(
            map(int, (departure.numel(), key.numel(), value.numel(), destination.numel()))
        ) != (g.location_dim, g.query_dim, g.value_dim, g.location_dim):
            raise ValueError("Edge field dimensions do not match v10 geometry")
        edge_id = self.next_edge_id if id is None else id
        if edge_id < self.next_edge_id or edge_id in self.rows:
            raise ValueError("Canonical edge IDs must increase and never be reused")
        row = PackedEdge(
            edge_id,
            pack_fp8(
                unit(departure.float()),
                stochastic_seed=rounding_seed(edge_id, "departure", step, worker),
            ),
            pack_fp8(
                unit(key.float()), stochastic_seed=rounding_seed(edge_id, "key", step, worker)
            ),
            pack_fp4(value, stochastic_seed=rounding_seed(edge_id, "value", step, worker)),
            pack_fp8(
                unit(destination.float()),
                stochastic_seed=rounding_seed(edge_id, "destination", step, worker),
            ),
            float(bias),
            step,
            float(write_mass),
        )
        if self.packed_bytes + row.nbytes > g.byte_budget:
            raise RuntimeError("Packed-byte budget exceeded")
        self.rows[edge_id] = row
        self.next_edge_id = edge_id + 1
        self._invalidate()
        return edge_id

    def retire(self, edge_id: int) -> None:
        del self.rows[edge_id]
        for key in list(self._masters):
            if key[0] == edge_id:
                del self._masters[key]
        self._invalidate()

    def field(self, edge_id: int, name: str, *, train: bool = False) -> Tensor:
        if name not in {"departure", "key", "value", "destination", "bias"}:
            raise KeyError(name)
        row = self.rows[edge_id]
        packed = getattr(row, name)
        if name == "bias":
            decoded = torch.tensor([packed], dtype=torch.float32, device=self.device)
        else:
            decoded = packed.decode(device=self.device)
        if not train or (name == "value" and not self.learned_values):
            return decoded
        identity = edge_id, name
        if identity not in self._masters:
            self._masters[identity] = decoded.detach().requires_grad_()
        master = self._masters[identity]
        if name == "bias":
            return master
        return straight_through(master, packed)

    def _table(self) -> tuple[tuple[int, ...], Tensor]:
        if self._table_version != self.version or self._departures is None:
            self._departure_ids = tuple(sorted(self.rows))
            g = self.geometry
            self._departures = (
                torch.stack(
                    [
                        unit(self.rows[i].departure.decode(device=self.device))
                        for i in self._departure_ids
                    ]
                )
                if self._departure_ids
                else torch.empty(0, g.location_dim, device=self.device)
            )
            self._table_version = self.version
        return self._departure_ids, self._departures

    def nearest(self, location: Tensor, count: int | None = None) -> tuple[int, ...]:
        """Exact scan of decoded FP8 departures; independent of query keys."""
        ids, departures = self._table()
        if not ids:
            return ()
        requested = self.geometry.neighbors if count is None else count
        if requested < 1:
            raise ValueError("Nearest-neighbour count must be positive")
        similarities = departures.float() @ unit(location.float())
        indices = torch.argsort(similarities, descending=True, stable=True)[:requested]
        return tuple(ids[i] for i in indices.tolist())

    def refresh(self, agenda: SearchAgenda) -> None:
        if agenda.version != self.version:
            agenda.neighborhoods = [
                self.nearest(location, self.geometry.neighbors) for location in agenda.locations
            ]
            agenda.version = self.version

    def _content_location(self, h: Tensor) -> Tensor:
        g = self.geometry
        if h.shape != (g.value_dim,) or not bool(torch.isfinite(h).all()):
            raise ValueError("Hindsight must be one finite full-width bank value")
        if self._projection is None or self._projection.device != h.device:
            generator = torch.Generator(device="cpu").manual_seed(g.seed)
            self._projection = (
                torch.randn(g.location_dim, g.value_dim, generator=generator)
                / math.sqrt(g.value_dim)
            ).to(h.device)
        return unit(self._projection @ h.float())

    def lookup(
        self,
        query: Tensor,
        agenda: SearchAgenda,
        *,
        training: bool = False,
        hindsight: Tensor | None = None,
        overlay: float = 0.0,
        exploration_seed: int = 0,
    ) -> Lookup | None:
        """Score every (agenda location, departure-neighbour) pair before top-k."""
        g = self.geometry
        if agenda.lookups >= g.max_lookups:
            return None
        if query.shape != (g.query_dim,) or not bool(torch.isfinite(query).all()):
            raise ValueError("Lookup requires one finite query-width vector")
        if not 0 <= overlay <= 1 or (overlay and hindsight is None):
            raise ValueError("Overlay requires a hindsight value and a bounded weight")
        self.refresh(agenda)
        q = unit(query.float())
        all_pairs: list[tuple[int, int]] = []
        all_scores: list[Tensor] = []
        modulated: list[Tensor] = []
        for location_id, (location, neighborhood) in enumerate(
            zip(agenda.locations, agenda.neighborhoods, strict=True)
        ):
            modq = unit(q * (1 + torch.tanh(self.modulation @ unit(location))))
            modulated.append(modq)
            if not neighborhood:
                continue
            keys = torch.stack([unit(self.field(i, "key", train=False)) for i in neighborhood])
            departures = torch.stack(
                [unit(self.field(i, "departure", train=False)) for i in neighborhood]
            )
            biases = torch.cat([self.field(i, "bias", train=False) for i in neighborhood])
            scores = (
                self.log_beta_query.exp() * (keys @ modq)
                + self.log_beta_location.exp() * (departures @ unit(location))
                + biases
            )
            all_pairs.extend((location_id, edge_id) for edge_id in neighborhood)
            all_scores.extend(scores.unbind())
        limit = g.selected - (g.exploration if training else 0)
        if all_scores:
            scores = torch.stack(all_scores)
            top_indices = torch.topk(scores, min(limit, len(all_pairs))).indices.tolist()
            pairs = [all_pairs[i] for i in top_indices]
            # Top-k and neighbourhood membership are discrete. Only the chosen
            # edge masters receive sparse gradient leaves.
            selected_scores = [
                self.log_beta_query.exp()
                * (unit(self.field(edge_id, "key", train=training)) @ modulated[j])
                + self.log_beta_location.exp()
                * (
                    unit(self.field(edge_id, "departure", train=training))
                    @ unit(agenda.locations[j])
                )
                + self.field(edge_id, "bias", train=training).squeeze()
                for j, edge_id in pairs
            ]
        else:
            pairs, selected_scores = [], []
        if training and self.rows and g.exploration:
            rng = torch.Generator().manual_seed(exploration_seed)
            for _ in range(g.exploration):
                location_id = int(torch.randint(len(agenda.locations), (1,), generator=rng))
                band = self.nearest(
                    agenda.locations[location_id],
                    min(len(self.rows), 2 * g.neighbors),
                )[g.neighbors : 2 * g.neighbors]
                if not band:
                    continue
                edge_id = band[int(torch.randint(len(band), (1,), generator=rng))]
                location = unit(agenda.locations[location_id])
                score = (
                    self.log_beta_query.exp()
                    * (unit(self.field(edge_id, "key", train=True)) @ modulated[location_id])
                    + self.log_beta_location.exp()
                    * (unit(self.field(edge_id, "departure", train=True)) @ location)
                    + self.field(edge_id, "bias", train=True).squeeze()
                )
                pairs.append((location_id, edge_id))
                selected_scores.append(score)
        logits = torch.stack([*selected_scores, self.sink_bias + self.sink_direction @ q])
        weights = torch.softmax(logits.float(), dim=0)
        w_sink = weights[-1]
        bank_value = w_sink * self.sink_value.float()
        bank_landing = w_sink * unit(agenda.current.float())
        edge_weights: dict[int, Tensor] = {}
        for idx, (_, edge_id) in enumerate(pairs):
            w = weights[idx]
            bank_value = bank_value + w * self.field(edge_id, "value", train=training)
            bank_landing = bank_landing + w * unit(
                self.field(edge_id, "destination", train=training)
            )
            edge_weights[edge_id] = edge_weights.get(edge_id, w.new_zeros(())) + w
        landed = unit(bank_landing)
        value = bank_value
        h_destination = None
        if hindsight is not None:
            h_destination = self._content_location(hindsight)
        if overlay:
            value = (1 - overlay) * bank_value + overlay * hindsight.float()
            landed = unit((1 - overlay) * bank_landing + overlay * h_destination)
        return Lookup(
            value,
            landed,
            bank_value,
            unit(bank_landing),
            bank_landing,
            weights,
            w_sink,
            tuple(pairs),
            edge_weights,
            q,
            agenda.current,
            h_destination,
            self.version,
        )

    def commit(self, agenda: SearchAgenda, lookup: Lookup) -> bool:
        """Commit atomically after C accepted the knowledge event."""
        if agenda.lookups >= self.geometry.max_lookups:
            raise RuntimeError("Lookup cap reached before publication")
        if lookup.bank_version != self.version or agenda.version != self.version:
            raise RuntimeError("Stale transducer lookup cannot commit across a bank publication")
        agenda.lookups += 1
        location = lookup.landing
        if max(float(unit(s.detach()) @ unit(location.detach())) for s in agenda.locations) >= 0.99:
            return False
        agenda.locations.append(location)
        agenda.neighborhoods.append(self.nearest(location))
        agenda.version = self.version
        return True

    def state_snapshot(self) -> dict[str, Any]:
        return {
            "identity": self.identity,
            "geometry": vars(self.geometry),
            "version": self.version,
            "next_edge_id": self.next_edge_id,
            "learned_values": self.learned_values,
            "rows": [self.rows[k].to_dict() for k in sorted(self.rows)],
        }

    def load_snapshot(self, data: dict[str, Any]) -> None:
        if data.get("identity") != self.identity or data["geometry"] != vars(self.geometry):
            raise ValueError("Only exact v10 packed-edge snapshot identities are accepted")
        loaded = {row.id: row for row in map(PackedEdge.from_dict, data["rows"])}
        if len(loaded) != len(data["rows"]):
            raise ValueError("Duplicate edge ID in transducer checkpoint")
        if any(
            (row.departure.size, row.key.size, row.value.size, row.destination.size)
            != (
                self.geometry.location_dim,
                self.geometry.query_dim,
                self.geometry.value_dim,
                self.geometry.location_dim,
            )
            for row in loaded.values()
        ):
            raise ValueError("Incompatible packed-edge geometry")
        if len(loaded) > self.geometry.max_edges or sum(r.nbytes for r in loaded.values()) > (
            self.geometry.byte_budget
        ):
            raise ValueError("Snapshot exceeds canonical edge or byte budget")
        next_id = int(data.get("next_edge_id", max(loaded, default=0) + 1))
        if min(loaded, default=1) <= 0 or next_id <= max(loaded, default=0):
            raise ValueError("Invalid canonical edge ID high-water mark")
        self.rows = loaded
        self.next_edge_id = next_id
        self.learned_values = bool(data["learned_values"])
        self.version = int(data["version"])
        self._masters.clear()
        self._table_version = -1
        self._departures = None
