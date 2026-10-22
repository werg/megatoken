"""Momentum-free, discounted block-scale optimization of the packed v10 edges.

Per trained field of an edge it keeps only one squared-gradient sum per
256-coordinate block, an effective observation count and a timestamp, in
per-field row stores. Updates of all edges touched in a window are computed
together, one field at a time.

Until the learned-value switch, values are not stepped: they are written as
decayed weighted means of hindsight values (``add_write``/``_apply_writes``).
"""

from __future__ import annotations

import torch
from torch import Tensor

from .transducer import ADDRESS_FIELDS, FIELD_COLUMNS, FIELDS, TransducerBank
from .transducer_packing import unit
from .transducer_rows import RowStore

BLOCK = 256
STAT_HALF_LIFE = 4096
PRIOR_HALF_LIFE = 256
WRITE_HALF_LIFE = 16384


def _blocks(values: Tensor) -> Tensor:
    """Mean square of each 256-coordinate block of every row: [rows, blocks]."""
    rows, dim = values.shape
    padded = torch.nn.functional.pad(values.float(), (0, -dim % BLOCK))
    sizes = torch.full((padded.shape[1] // BLOCK,), BLOCK, device=values.device)
    sizes[-1] = dim - BLOCK * (sizes.numel() - 1)
    return padded.reshape(rows, -1, BLOCK).square().sum(-1) / sizes


def _per_block(values: Tensor, blocks: Tensor) -> Tensor:
    """Broadcast [rows, blocks] back to [rows, dim]."""
    return blocks.repeat_interleave(BLOCK, dim=1)[:, : values.shape[1]]


def statistics_store(device: torch.device, blocks: int) -> RowStore:
    return RowStore(
        device,
        {
            "squared_sums": ((blocks,), torch.float32),
            "count": ((), torch.float32),
            "last_step": ((), torch.int64),
        },
    )


def writes_store(device: torch.device, width: int) -> RowStore:
    return RowStore(device, {"mass": ((), torch.float32), "sums": ((width,), torch.float32)})


class BlockScaleDescent:
    """No momentum, no idle updates, block-sized second moments, bounded FP32 masters."""

    def __init__(
        self,
        bank: TransducerBank,
        *,
        cache_edges: int = 8192,
        worker: int = 0,
        address_lr: float = 1e-4,
        value_lr: float = 1e-3,
    ) -> None:
        if cache_edges < 1 or address_lr <= 0 or value_lr <= 0:
            raise ValueError("Invalid KB optimizer capacity or learning rate")
        self.bank = bank
        self.worker = worker
        self.capacity = cache_edges
        self.address_lr = address_lr
        self.value_lr = value_lr
        g, device = bank.geometry, bank.device
        self.blocks = {name: -(-g.dims(name) // BLOCK) for name in FIELDS}
        self.fields = {name: statistics_store(device, self.blocks[name]) for name in FIELDS}
        self.priors: dict[str, Tensor] = {}
        self.writes = writes_store(device, g.value_dim)
        # What this learner added since its last DiLoCo submission.
        self.round_stats = {name: statistics_store(device, self.blocks[name]) for name in FIELDS}
        self.round_writes = writes_store(device, g.value_dim)
        # Packed columns of each field as it was when first stepped this round.
        self.round_origin = self._origin_stores()
        self.step_number = 0
        self.prior_step = 0

    def _origin_stores(self) -> dict[str, RowStore]:
        table = self.bank.table
        return {
            name: RowStore(
                self.bank.device,
                {
                    column: (tuple(getattr(table, column).shape[1:]), getattr(table, column).dtype)
                    for column in FIELD_COLUMNS[name]
                },
            )
            for name in FIELDS
        }

    # --------------------------------------------------------------- writes

    def add_write(self, edge_ids: Tensor, weights: Tensor, hindsight: Tensor) -> None:
        """Accumulate one lookup's hindsight value into the edges it used."""
        if self.bank.learned_values:
            raise RuntimeError("Hindsight writes stop at the learned-value switch")
        if hindsight.shape != (self.bank.geometry.value_dim,):
            raise ValueError("Written knowledge has the wrong width")
        h = hindsight.detach().float().to(self.bank.device)
        w = weights.detach().float().to(self.bank.device)
        if not bool(torch.isfinite(h).all() & torch.isfinite(w).all()):
            raise ValueError("Written knowledge or its weights are nonfinite")
        positive = w > 0
        ids = edge_ids[positive.to(edge_ids.device)].tolist()
        if not ids:
            return
        w = w[positive]
        for store in (self.writes, self.round_writes):
            rows, _ = store.rows(ids)
            store.mass.index_add_(0, rows, w)
            store.sums.index_add_(0, rows, w[:, None] * h[None])

    def _apply_writes(self, step: int) -> int:
        edges = [e for e in self.writes.ids() if e in self.bank]
        if edges:
            rows = self.writes.find(edges)
            keep = self.writes.mass[rows] > 0
            rows = rows[keep]
            edges = torch.tensor(edges, dtype=torch.int64)[keep.cpu()].tolist()
        if not edges:
            self.writes.clear()
            return 0
        ids = torch.tensor(edges, dtype=torch.int64)
        table = self.bank.table
        slots = table.slots(ids)
        # A row the coordinator stamped slightly ahead of this learner's clock is not aged.
        age = (step - table.last_write[slots]).float().clamp_min(0)
        old_mass = table.write_mass[slots] * torch.exp2(-age / WRITE_HALF_LIFE)
        new_mass = old_mass + self.writes.mass[rows]
        blended = (
            table.decode("value", slots) * old_mass[:, None] + self.writes.sums[rows]
        ) / new_mass[:, None]
        self.bank.publish_values(ids, blended, step=step, worker=self.worker)
        self.bank.set_write_mass(ids, new_mass, step=step)
        self.writes.clear()
        return len(edges)

    # ----------------------------------------------------------------- step

    def _prior(self, name: str) -> Tensor:
        prior = self.priors.get(name)
        if prior is None:
            prior = torch.full((self.blocks[name],), 1e-6, device=self.bank.device)
            self.priors[name] = prior
        return prior

    def fold(self, store: RowStore, ids: list[int], squares: Tensor, counts: Tensor, step: int):
        """Decay these edges' statistics to ``step`` and add the new observations."""
        rows, created = store.rows(ids)
        if created:
            store.last_step[rows[torch.tensor(created, device=rows.device)]] = step
        elapsed = step - store.last_step[rows]
        if bool((elapsed < 0).any()):
            raise ValueError("Per-field optimizer clocks cannot go backwards")
        decay = torch.exp2(-elapsed.float() / STAT_HALF_LIFE)
        store.squared_sums[rows] = store.squared_sums[rows] * decay[:, None] + squares
        store.count[rows] = store.count[rows] * decay + counts
        store.last_step[rows] = step
        return rows

    def step(self, step: int) -> dict[str, int]:
        """One optimizer barrier: update touched masters, publish packed rows, apply writes."""
        if step <= self.step_number:
            raise ValueError("KB optimizer steps must increase strictly")
        masters = self.bank.masters
        updated = 0
        observations: dict[str, Tensor] = {}
        for name, (rows, grad) in masters.take().items():
            store = masters.stores[name]
            edges = store.edge[rows].tolist()
            alive = torch.tensor([edge in self.bank for edge in edges], device=rows.device)
            if not bool(alive.all()):
                rows, grad = rows[alive], grad[alive]
                edges = [edge for edge, keep in zip(edges, alive.tolist(), strict=True) if keep]
            if not edges:
                continue
            if name == "value" and not self.bank.learned_values:
                raise RuntimeError("Value gradients are forbidden before the switch")
            if not bool(torch.isfinite(grad).all()):
                raise FloatingPointError("Nonfinite sparse transducer gradient")
            old = store.data[rows]
            address = name in ADDRESS_FIELDS
            if address:
                # Decoded FP8 addresses are only nearly unit: move on the sphere itself.
                old = tangent = unit(old)
                grad = grad - tangent * (grad * tangent).sum(-1, keepdim=True)
            squares = _blocks(grad)
            ones = torch.ones(len(edges), device=squares.device)
            self.fold(self.round_stats[name], edges, squares, ones, step)
            stats = self.fields[name]
            at = self.fold(stats, edges, squares, ones, step)
            variance = (stats.squared_sums[at] + 4 * self._prior(name)) / (
                stats.count[at][:, None] + 4
            )
            rate = self.value_lr if name == "value" else self.address_lr
            delta = -rate * grad / (_per_block(grad, variance.clamp_min(1e-12).sqrt()) + 1e-8)
            if address:
                delta = delta - tangent * (delta * tangent).sum(-1, keepdim=True)
                norm = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
                delta = delta * torch.clamp(0.01 / norm.clamp_min(1e-30), max=1.0)
                replacement = unit(old + delta)
            else:
                if name == "value":
                    limit = 0.01 * _blocks(old).sqrt().clamp_min(1e-3)
                    rms = _blocks(delta).sqrt()
                    delta = delta * _per_block(delta, torch.clamp(limit / (rms + 1e-30), max=1.0))
                replacement = old + delta
            if not bool(torch.isfinite(replacement).all()):
                raise FloatingPointError("Nonfinite sparse KB parameter update")
            store.data[rows] = replacement
            ids = torch.tensor(edges, dtype=torch.int64)
            self._remember_origin(name, edges)
            if address:
                self.bank.publish_addresses(name, ids, replacement, step=step, worker=self.worker)
            elif name == "value":
                self.bank.publish_values(ids, replacement, step=step, worker=self.worker)
            else:
                self.bank.publish_bias(ids, replacement.reshape(-1))
            observations[name] = squares.mean(0)
            updated += len(edges)
        written = 0 if self.bank.learned_values else self._apply_writes(step)
        prior_decay = 2.0 ** (-(step - self.prior_step) / PRIOR_HALF_LIFE)
        for name, mean in observations.items():
            self.priors[name] = prior_decay * self._prior(name) + (1 - prior_decay) * mean
        self.prior_step = step
        self.step_number = step
        masters.evict(self.capacity)
        if updated or written:
            self.bank._changed()
        return {"updated_fields": updated, "written_values": written}

    def _remember_origin(self, name: str, edges: list[int]) -> None:
        store = self.round_origin[name]
        rows, created = store.rows(edges)
        if not created:
            return
        at = torch.tensor(created, device=rows.device)
        slots = self.bank.table.slots([edges[k] for k in created])
        for column, values in self.bank.table.columns(name, slots).items():
            getattr(store, column)[rows[at]] = values

    def round_deltas(self) -> dict[str, tuple[Tensor, Tensor]]:
        """Per field: (edge IDs, published rows minus the round's base rows)."""
        result = {}
        table = self.bank.table
        for name, store in self.round_origin.items():
            edges = [edge for edge in store.ids() if edge in self.bank]
            if not edges:
                continue
            base = table.decode_columns(name, store.columns(edges))
            current = table.decode(name, table.slots(edges))
            delta = (current - base).reshape(len(edges), -1).float().cpu()
            result[name] = (torch.tensor(edges, dtype=torch.int64), delta)
        return result

    def forget(self, ids) -> None:
        """Retired edges lose their statistics, pending writes and round origins."""
        ids = [int(edge) for edge in torch.as_tensor(ids).tolist()]
        for store in (
            *self.fields.values(),
            *self.round_stats.values(),
            *self.round_origin.values(),
            self.writes,
            self.round_writes,
        ):
            store.drop(ids)

    def forget_origins(self, ids) -> None:
        for store in self.round_origin.values():
            store.drop(ids)

    def reset_origins(self) -> None:
        self.round_origin = self._origin_stores()

    def copy_statistics(self, sources: list[int], targets: list[int]) -> None:
        """Split children inherit half of their parents' statistics (the parents keep half)."""
        for store in self.fields.values():
            present = [k for k, edge in enumerate(sources) if edge in store]
            if not present:
                continue
            parents = store.find([sources[k] for k in present])
            children, _ = store.rows([targets[k] for k in present])
            store.squared_sums[parents] /= 2
            store.count[parents] /= 2
            for column in ("squared_sums", "count", "last_step"):
                getattr(store, column)[children] = getattr(store, column)[parents]

    def switch_to_learned(self) -> None:
        if len(self.writes):
            raise RuntimeError("Flush written value evidence before switching")
        if self.bank.learned_values:
            raise RuntimeError("Value optimizer was already activated")
        self.bank.learned_values = True
        self.bank.table.write_mass.zero_()
        self.round_writes = writes_store(self.bank.device, self.bank.geometry.value_dim)
        self.bank.masters.clear(("value",))
        self.bank._changed()

    def take_round(self) -> tuple[dict, dict, RowStore]:
        """Deltas, statistics and writes since the last call, for a DiLoCo submission.

        Statistics come back per field as (edge IDs, block squared sums, counts).
        The FP32 masters restart from the next installed bank.
        """
        deltas = self.round_deltas()
        statistics = {}
        for name, store in self.round_stats.items():
            edges = [edge for edge in store.ids() if edge in self.bank]
            if edges:
                rows = store.find(edges)
                statistics[name] = (
                    torch.tensor(edges, dtype=torch.int64),
                    store.squared_sums[rows].cpu(),
                    store.count[rows].cpu(),
                )
        writes = self.round_writes
        device, g = self.bank.device, self.bank.geometry
        self.round_stats = {name: statistics_store(device, self.blocks[name]) for name in FIELDS}
        self.round_writes = writes_store(device, g.value_dim)
        self.reset_origins()
        return deltas, statistics, writes

    # ------------------------------------------------------------ checkpoint

    def state_dict(self) -> dict:
        """Tensor state for an exact interrupted-run resume, masters included."""
        return {
            "step": self.step_number,
            "prior_step": self.prior_step,
            "worker": self.worker,
            "priors": {k: v.cpu() for k, v in self.priors.items()},
            "fields": {name: store.state_dict() for name, store in self.fields.items()},
            "round_stats": {name: store.state_dict() for name, store in self.round_stats.items()},
            "masters": self.bank.masters.state_dict(),
            "writes": self.writes.state_dict(),
            "round_writes": self.round_writes.state_dict(),
            "round_origin": {name: store.state_dict() for name, store in self.round_origin.items()},
        }

    def load_state_dict(self, state: dict) -> None:
        if int(state["worker"]) != self.worker:
            raise ValueError("Cannot resume a different worker's fractional progress")
        if any(int(edge) not in self.bank for edge in self._saved_masters(state)):
            raise ValueError("Checkpoint master references a retired edge")
        device = self.bank.device
        self.step_number = int(state["step"])
        self.prior_step = int(state["prior_step"])
        self.priors = {k: v.to(device) for k, v in state["priors"].items()}
        for target, saved in (
            (self.fields, state["fields"]),
            (self.round_stats, state["round_stats"]),
            (self.round_origin, state["round_origin"]),
        ):
            for name, store in target.items():
                store.load_state_dict(saved[name])
        self.bank.masters.load_state_dict(state["masters"])
        self.writes.load_state_dict(state["writes"])
        self.round_writes.load_state_dict(state["round_writes"])

    @staticmethod
    def _saved_masters(state: dict) -> list[int]:
        fields = state["masters"]["fields"].values()
        return [edge for saved in fields for edge in saved["ids"].tolist()]
