"""Hindsight write-out, structural evidence and bounded evolution of a v10 bank.

Evidence is kept per edge ID in flat tensors (one row per edge in the
``Evidence`` table), decayed lazily to the step it is read at. Everything the
rules need comes from the ordinary forward/backward pass: there is no replay,
candidate search or extra forward.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from .transducer import ADDRESS_FIELDS, PROVISIONAL_BASE, BankFull, Lookup, TransducerBank
from .transducer_optimizer import WRITE_HALF_LIFE, BlockScaleDescent
from .transducer_packing import rounding_seed, unit

EVIDENCE_HALF_LIFE = 16384
REMOVE_AGE = 4096
SPLIT_VISITS = 64
SKETCH = 64


@dataclass
class EvidenceRows:
    """Structural evidence columns for a set of edges (also the DiLoCo wire format)."""

    ids: Tensor
    utility: Tensor
    visits: Tensor
    dispersion: Tensor
    sketch: Tensor  # [n, 64]
    sketch_count: Tensor
    energy: Tensor
    last_step: Tensor  # int64


class Evidence:
    """Per-edge evidence keyed by edge ID, decayed lazily with a 16,384-step half-life."""

    names = ("utility", "visits", "dispersion", "sketch_count", "energy")

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.row_of: dict[int, int] = {}
        self.free: list[int] = []
        self.size = 0
        self._allocate(256)

    def _allocate(self, capacity: int) -> None:
        old = {name: getattr(self, name, None) for name in (*self.names, "sketch", "last_step")}
        self.capacity = capacity
        for name in self.names:
            setattr(self, name, torch.zeros(capacity, device=self.device))
        self.sketch = torch.zeros(capacity, SKETCH, device=self.device)
        self.last_step = torch.zeros(capacity, dtype=torch.int64, device=self.device)
        for name, tensor in old.items():
            if tensor is not None:
                getattr(self, name)[: tensor.shape[0]] = tensor

    def rows(self, ids: list[int], step: int) -> Tensor:
        """Rows for these edges (created empty if new), decayed to ``step``."""
        result = []
        for edge in ids:
            row = self.row_of.get(edge)
            if row is None:
                if self.free:
                    row = self.free.pop()
                else:
                    if self.size == self.capacity:
                        self._allocate(self.capacity * 2)
                    row = self.size
                    self.size += 1
                self.row_of[edge] = row
                for name in self.names:
                    getattr(self, name)[row] = 0
                self.sketch[row] = 0
                self.last_step[row] = step
            result.append(row)
        rows = torch.tensor(result, dtype=torch.int64, device=self.device)
        self.decay(rows, step)
        return rows

    def decay(self, rows: Tensor, step: int) -> None:
        # Learner and coordinator clocks differ slightly; never decay backwards.
        age = (step - self.last_step[rows]).float().clamp_min(0)
        factor = torch.exp2(-age / EVIDENCE_HALF_LIFE)
        for name in self.names:
            getattr(self, name)[rows] *= factor
        self.sketch[rows] *= factor[:, None]
        self.last_step[rows] = torch.maximum(self.last_step[rows], torch.full_like(rows, step))

    def drop(self, ids: list[int]) -> None:
        for edge in ids:
            row = self.row_of.pop(edge, None)
            if row is not None:
                self.free.append(row)

    def export(self, ids: list[int], step: int) -> EvidenceRows:
        rows = self.rows(ids, step)
        return EvidenceRows(
            torch.tensor(ids, dtype=torch.int64),
            *(getattr(self, name)[rows].cpu() for name in ("utility", "visits", "dispersion")),
            self.sketch[rows].cpu(),
            self.sketch_count[rows].cpu(),
            self.energy[rows].cpu(),
            self.last_step[rows].cpu(),
        )

    def add(self, incoming: EvidenceRows, step: int, scale: float = 1.0) -> None:
        """Fold another learner's increments in, aged from their step to ``step``."""
        if not incoming.ids.numel():
            return
        rows = self.rows(incoming.ids.tolist(), step)
        age = (step - incoming.last_step.to(self.device)).float().clamp_min(0)
        factor = torch.exp2(-age / EVIDENCE_HALF_LIFE) * scale
        for name in ("utility", "visits", "dispersion", "sketch_count", "energy"):
            getattr(self, name)[rows] += getattr(incoming, name).to(self.device) * factor
        self.sketch[rows] += incoming.sketch.to(self.device) * factor[:, None]

    def halve_into(self, source: int, target: int, step: int) -> None:
        rows = self.rows([source, target], step)
        for name in self.names:
            column = getattr(self, name)
            column[rows[0]] /= 2
            column[rows[1]] = column[rows[0]]
        self.sketch[rows[0]] /= 2
        self.sketch[rows[1]] = self.sketch[rows[0]]

    def state_dict(self) -> dict:
        ids = sorted(self.row_of)
        rows = torch.tensor([self.row_of[i] for i in ids], dtype=torch.int64, device=self.device)
        return {
            "ids": torch.tensor(ids, dtype=torch.int64),
            **{name: getattr(self, name)[rows].cpu() for name in self.names},
            "sketch": self.sketch[rows].cpu(),
            "last_step": self.last_step[rows].cpu(),
        }

    def load_state_dict(self, state: dict) -> None:
        count = int(state["ids"].numel())
        self.row_of, self.free, self.size = {}, [], 0
        for name in self.names:
            delattr(self, name)
        del self.sketch, self.last_step
        self._allocate(max(256, count))
        for name in (*self.names, "sketch", "last_step"):
            getattr(self, name)[:count] = state[name].to(self.device)
        self.row_of = {int(edge): row for row, edge in enumerate(state["ids"].tolist())}
        self.size = count


@dataclass
class WriteOut:
    """A hindsight entry chosen for write-out, waiting to become an edge."""

    departure: Tensor
    key: Tensor
    value: Tensor
    destination: Tensor
    bias: float


class TransducerGrowth:
    """Write-out, structural evidence and the split/remove/budget rules."""

    def __init__(
        self,
        bank: TransducerBank,
        *,
        max_writeouts: int = 512,
        optimizer: BlockScaleDescent | None = None,
        provisional: bool = False,
    ) -> None:
        if optimizer is not None and optimizer.bank is not bank:
            raise ValueError("Growth optimizer must own the same canonical bank")
        self.bank = bank
        self.optimizer = optimizer
        self.max_writeouts = max_writeouts
        # Learners under a coordinator create provisional rows with local IDs.
        self.provisional = provisional
        self.next_provisional = PROVISIONAL_BASE
        self.provisional_ids: list[int] = []  # not yet submitted
        self.submitted_ids: list[int] = []  # submitted, awaiting the coordinator's verdict
        self.writeouts = 0
        self.evidence = Evidence(bank.device)
        self.round_evidence = Evidence(bank.device)
        self._sketch_maps: dict[str, Tensor] = {}

    def begin_round(self) -> None:
        self.writeouts = 0

    def take_round(self, step: int) -> EvidenceRows:
        """Structural increments since the last round, for a DiLoCo submission."""
        ids = sorted(self.round_evidence.row_of)
        rows = self.round_evidence.export(ids, step)
        self.round_evidence = Evidence(self.bank.device)
        return rows

    def switch_to_learned(self) -> None:
        """Write dispersion is dropped; the gradient sketch and energy start from zero."""
        for evidence in (self.evidence, self.round_evidence):
            evidence.dispersion.zero_()
            evidence.sketch.zero_()
            evidence.sketch_count.zero_()
            evidence.energy.zero_()

    def _sketch_map(self, name: str, width: int) -> Tensor:
        projection = self._sketch_maps.get(name)
        if projection is None:
            generator = torch.Generator().manual_seed(
                rounding_seed(self.bank.geometry.seed, "gradient-sketch", name)
            )
            projection = (torch.randn(SKETCH, width, generator=generator) / math.sqrt(SKETCH)).to(
                self.bank.device
            )
            self._sketch_maps[name] = projection
        return projection

    # ---------------------------------------------------------- observation

    def observe(
        self,
        lookup: Lookup,
        *,
        gradient_value: Tensor,
        gradient_location: Tensor,
        step: int,
        hindsight: Tensor | None = None,
        sink_only: bool = False,
    ) -> None:
        """First-order removal utility of every selected edge; written-phase dispersion.

        ``gradient_value``/``gradient_location`` are the future-loss gradients with
        respect to the bank's own blend and unnormalized landing (not the overlay mix).
        """
        if not lookup.edge_ids.numel():
            return
        g = self.bank.geometry
        if gradient_value.shape != (g.value_dim,) or gradient_location.shape != (g.location_dim,):
            raise ValueError("Invalid structural observation")
        if not bool(torch.isfinite(gradient_value).all() & torch.isfinite(gradient_location).all()):
            raise FloatingPointError("Nonfinite structural gradient")
        ids = lookup.edge_ids.tolist()
        present = [i for i, edge in enumerate(ids) if edge in self.bank]
        if not present:
            return
        ids = [ids[i] for i in present]
        weights = lookup.edge_weights.detach()[present].float()
        values = self.bank.decode("value", ids)
        destinations = unit(self.bank.decode("destination", ids))
        # Removing edge i renormalizes the mixture to (F - w u_i) / (1 - w).
        removal = (lookup.bank_value.detach()[None] - values) @ gradient_value.float() + (
            lookup.bank_landing_raw.detach()[None] - destinations
        ) @ gradient_location.float()
        delta = weights / (1 - weights).clamp_min(1e-6) * removal
        written = not self.bank.learned_values and hindsight is not None and not sink_only
        if written:
            spread = weights * (hindsight.detach().float()[None] - values).square().sum(-1)
        for evidence in (self.evidence, self.round_evidence):
            rows = evidence.rows(ids, step)
            evidence.utility[rows] += delta
            evidence.visits[rows] += 1
            if written:
                evidence.dispersion[rows] += spread

    def observe_gradients(self, step: int) -> None:
        """Learned phase: one gradient sketch per edge per backward window, from its masters."""
        if not self.bank.learned_values:
            return
        per_edge: dict[int, Tensor] = {}
        for (edge, name), leaf in self.bank.masters.items():
            if leaf.grad is None or edge not in self.bank:
                continue
            grad = leaf.grad.detach().float().reshape(-1)
            sketch = self._sketch_map(name, grad.numel()) @ grad
            per_edge[edge] = per_edge.get(edge, 0) + sketch
        if not per_edge:
            return
        ids = list(per_edge)
        sketches = torch.stack([per_edge[edge] for edge in ids])
        for evidence in (self.evidence, self.round_evidence):
            rows = evidence.rows(ids, step)
            evidence.sketch[rows] += sketches
            evidence.sketch_count[rows] += 1
            evidence.energy[rows] += sketches.square().sum(-1)

    # ------------------------------------------------------------ write-out

    def candidate(
        self, lookup: Lookup, hindsight: Tensor, gradient_value: Tensor
    ) -> WriteOut | None:
        """A hindsight entry worth writing out: it helped, and the bank does not supply it."""
        if hindsight.shape != (self.bank.geometry.value_dim,) or gradient_value.shape != (
            hindsight.shape
        ):
            raise ValueError("Write-out needs a full-width value and future-loss gradient")
        h = hindsight.detach().float()
        if float(gradient_value.float() @ h) >= 0:
            return None
        residual = float((h - lookup.bank_value.detach()).square().sum())
        if lookup.edge_ids.numel() and residual <= 0.25 * float(h.square().sum()):
            return None
        beta = float((self.bank.log_beta_location.exp() + self.bank.log_beta_query.exp()).detach())
        best = float(lookup.pair_logits.detach().max()) if lookup.pair_logits.numel() else 0.0
        return WriteOut(
            departure=lookup.start.detach(),
            key=lookup.start_key.detach(),
            value=h,
            destination=self.bank.content_location(h),
            bias=best - beta,
        )

    def write_out(self, entries: list[WriteOut], *, step: int) -> Tensor:
        """Add the chosen entries as edges, within the per-round cap and the budget."""
        entries = entries[: max(0, self.max_writeouts - self.writeouts)]
        if not entries:
            return torch.empty(0, dtype=torch.int64)
        ids = None
        if self.provisional:
            ids = torch.arange(self.next_provisional, self.next_provisional + len(entries))
        try:
            added = self.bank.add_edges(
                departure=torch.stack([e.departure for e in entries]),
                key=torch.stack([e.key for e in entries]),
                value=torch.stack([e.value for e in entries]),
                destination=torch.stack([e.destination for e in entries]),
                bias=torch.tensor([e.bias for e in entries]),
                write_mass=torch.full((len(entries),), 0.0 if self.bank.learned_values else 1.0),
                step=step,
                ids=ids,
                provisional=self.provisional,
            )
        except BankFull:
            return torch.empty(0, dtype=torch.int64)
        if self.provisional:
            self.next_provisional += len(entries)
            self.provisional_ids.extend(added.tolist())
        self.writeouts += len(entries)
        return added

    # ------------------------------------------------------------ evolution

    def _retire(self, ids: list[int]) -> None:
        if not ids:
            return
        self.bank.retire(ids)
        self.evidence.drop(ids)
        self.round_evidence.drop(ids)
        if self.optimizer is not None:
            self.optimizer.forget(ids)

    def _split(self, ids: list[int], step: int) -> int:
        bank = self.bank
        room = min(
            bank.geometry.max_edges - bank.edge_count,
            (bank.geometry.byte_budget - bank.packed_bytes) // bank.geometry.row_bytes,
        )
        ids = ids[: max(0, room)]
        if not ids:
            return 0
        slots = bank.table.slots(ids)
        perturbed = {}
        for name in ("key", "departure"):
            base = unit(bank.table.decode(name, slots))
            noise = torch.stack(
                [
                    torch.randn(
                        base.shape[1],
                        generator=torch.Generator().manual_seed(
                            rounding_seed(bank.geometry.seed, "split", name, edge, step)
                        ),
                    )
                    for edge in ids
                ]
            ).to(bank.device)
            noise = unit(noise - (noise * base).sum(-1, keepdim=True) * base)
            perturbed[name] = (unit(base + 0.01 * noise), unit(base - 0.01 * noise))
        bias = bank.table.bias[slots] - math.log(2)
        mass = bank.table.write_mass[slots] / 2
        children = bank.add_edges(
            departure=perturbed["departure"][1],
            key=perturbed["key"][1],
            value=bank.table.decode("value", slots),
            destination=bank.table.decode("destination", slots),
            bias=bias,
            write_mass=mass,
            step=step,
        )
        tensor_ids = torch.tensor(ids, dtype=torch.int64)
        for name in ("key", "departure"):
            bank.publish_addresses(name, tensor_ids, perturbed[name][0], step=step, worker=-1)
        bank.publish_bias(tensor_ids, bias)
        bank.set_write_mass(tensor_ids, mass, step=step)
        for parent, child in zip(ids, children.tolist(), strict=True):
            self.evidence.halve_into(parent, child, step)
            if self.optimizer is not None:
                self.optimizer.copy_statistics(parent, child)
            for name in (*ADDRESS_FIELDS, "bias", "value"):
                bank.masters.pop((parent, name), None)
        return len(ids)

    def evolve(self, step: int, *, evidence_from: int = 0) -> dict[str, int]:
        """Coordinator-only structure changes after a complete round.

        Removal age counts from ``evidence_from`` at the earliest: the step the
        bank's own output first reaches the loss (the overlay's start). Before
        it, utility cannot be observed, so it cannot condemn an edge.
        """
        bank = self.bank
        ids = bank.edge_ids().tolist()
        if not ids:
            return {"removed": 0, "split": 0, "touched": []}
        rows = self.evidence.rows(ids, step)
        slots = bank.table.slots(ids)
        age = step - bank.table.birth[slots].clamp_min(evidence_from)
        utility = self.evidence.utility[rows]
        old = age >= REMOVE_AGE
        doomed = set(torch.tensor(ids)[(old & (utility <= 0)).cpu()].tolist())
        visits = self.evidence.visits[rows]
        if bank.learned_values:
            count = self.evidence.sketch_count[rows]
            sampled = count >= SPLIT_VISITS
            coherence = self.evidence.sketch[rows].square().sum(-1) / (
                count * self.evidence.energy[rows]
            ).clamp_min(1e-30)
            energy = self.evidence.energy[rows] / count.clamp_min(1)
            threshold = torch.quantile(energy[sampled], 0.9) if bool(sampled.any()) else math.inf
            split = sampled & (coherence < 0.1) & (energy >= threshold)
        else:
            elapsed = (step - bank.table.last_write[slots]).float().clamp_min(0)
            mass = bank.table.write_mass[slots] * torch.exp2(-elapsed / WRITE_HALF_LIFE)
            split = (
                (visits >= SPLIT_VISITS)
                & (mass > 0)
                & (
                    self.evidence.dispersion[rows] / mass.clamp_min(1e-30)
                    > 0.5 * bank.table.value_norm2[slots]
                )
            )
        splitting = [e for e in torch.tensor(ids)[split.cpu()].tolist() if e not in doomed]
        self._retire(sorted(doomed))
        removed = len(doomed)
        splits = self._split(splitting, step)
        touched = splitting[:splits]
        budget = bank.geometry.byte_budget
        if bank.packed_bytes > 0.95 * budget:
            ids = bank.edge_ids().tolist()
            rows = self.evidence.rows(ids, step)
            slots = bank.table.slots(ids)
            eligible = (step - bank.table.birth[slots].clamp_min(evidence_from)) >= REMOVE_AGE
            order = torch.argsort(self.evidence.utility[rows].masked_fill(~eligible, math.inf))
            excess = bank.packed_bytes - int(0.9 * budget)
            count = min(int(eligible.sum()), -(-excess // bank.geometry.row_bytes))
            victims = torch.tensor(ids)[order[:count].cpu()].tolist()
            self._retire(victims)
            removed += len(victims)
        return {"removed": removed, "split": splits, "touched": touched}

    def state_dict(self) -> dict:
        return {
            "writeouts": self.writeouts,
            "next_provisional": self.next_provisional,
            "provisional_ids": torch.tensor(self.provisional_ids, dtype=torch.int64),
            "submitted_ids": torch.tensor(self.submitted_ids, dtype=torch.int64),
            "evidence": self.evidence.state_dict(),
            "round_evidence": self.round_evidence.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.writeouts = int(state["writeouts"])
        self.next_provisional = int(state["next_provisional"])
        self.provisional_ids = state["provisional_ids"].tolist()
        self.submitted_ids = state["submitted_ids"].tolist()
        self.evidence.load_state_dict(state["evidence"])
        self.round_evidence.load_state_dict(state["round_evidence"])
