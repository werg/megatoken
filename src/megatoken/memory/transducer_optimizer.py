"""Momentum-free, discounted block-scale optimization of the packed v10 edges.

Per trained field of an edge it keeps only one squared-gradient sum per
256-coordinate block, an effective observation count and a timestamp. Updates
of all edges touched in a window are computed together, one field at a time.

Until the learned-value switch, values are not stepped: they are written as
decayed weighted means of hindsight values (``add_write``/``_apply_writes``).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .transducer import ADDRESS_FIELDS, TransducerBank
from .transducer_packing import unit

BLOCK = 256
STAT_HALF_LIFE = 4096
PRIOR_HALF_LIFE = 256
WRITE_HALF_LIFE = 16384


@dataclass
class FieldStatistics:
    squared_sums: Tensor  # [blocks]
    count: float = 0.0
    last_step: int = 0


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
        self.fields: dict[tuple[int, str], FieldStatistics] = {}
        self.priors: dict[str, Tensor] = {}
        self.write_mass: dict[int, float] = {}
        self.write_sums: dict[int, Tensor] = {}
        # What this learner added since its last DiLoCo submission.
        self.round_stats: dict[tuple[int, str], FieldStatistics] = {}
        self.round_write_mass: dict[int, float] = {}
        self.round_write_sums: dict[int, Tensor] = {}
        # Packed columns of each field as it was when first stepped this round.
        self.round_origin: dict[str, dict[int, dict[str, Tensor]]] = {}
        self.step_number = 0
        self.prior_step = 0

    # --------------------------------------------------------------- writes

    def add_write(self, edge_ids: Tensor, weights: Tensor, hindsight: Tensor) -> None:
        """Accumulate one lookup's hindsight value into the edges it used."""
        if self.bank.learned_values:
            raise RuntimeError("Hindsight writes stop at the learned-value switch")
        if hindsight.shape != (self.bank.geometry.value_dim,) or not bool(
            torch.isfinite(hindsight).all()
        ):
            raise ValueError("Written knowledge has the wrong width or nonfinite values")
        h = hindsight.detach().float().to(self.bank.device)
        for edge, weight in zip(edge_ids.tolist(), weights.detach().float().tolist(), strict=True):
            if weight <= 0:
                continue
            if not weight < float("inf"):
                raise ValueError("Invalid written KB evidence")
            for masses, sums in (
                (self.write_mass, self.write_sums),
                (self.round_write_mass, self.round_write_sums),
            ):
                masses[edge] = masses.get(edge, 0.0) + weight
                if edge in sums:
                    sums[edge].add_(h, alpha=weight)
                else:
                    sums[edge] = h * weight

    def _apply_writes(self, step: int) -> int:
        edges = [e for e in self.write_mass if e in self.bank and self.write_mass[e] > 0]
        if not edges:
            self.write_mass.clear()
            self.write_sums.clear()
            return 0
        ids = torch.tensor(edges, dtype=torch.int64)
        slots = self.bank.table.slots(ids)
        table = self.bank.table
        # A row the coordinator stamped slightly ahead of this learner's clock is not aged.
        age = (step - table.last_write[slots]).float().clamp_min(0)
        old_mass = table.write_mass[slots] * torch.exp2(-age / WRITE_HALF_LIFE)
        added = torch.tensor([self.write_mass[e] for e in edges], device=self.bank.device)
        sums = torch.stack([self.write_sums[e] for e in edges])
        new_mass = old_mass + added
        blended = (table.decode("value", slots) * old_mass[:, None] + sums) / new_mass[:, None]
        self.bank.publish_values(ids, blended, step=step, worker=self.worker)
        self.bank.set_write_mass(ids, new_mass, step=step)
        self.write_mass.clear()
        self.write_sums.clear()
        return len(edges)

    # ----------------------------------------------------------------- step

    def _prior(self, name: str, blocks: int) -> Tensor:
        prior = self.priors.get(name)
        if prior is None:
            prior = torch.full((blocks,), 1e-6, device=self.bank.device)
            self.priors[name] = prior
        if prior.numel() != blocks:
            raise ValueError("Optimizer block geometry changed")
        return prior

    def _statistics(self, identities: list[tuple[int, str]], squares: Tensor, step: int) -> Tensor:
        """Fold this step's block squares into the per-field sums; return the variances."""
        name = identities[0][1]
        prior = self._prior(name, squares.shape[1])
        sums, counts = [], []
        for row, identity in enumerate(identities):
            for table in (self.fields, self.round_stats):
                state = table.get(identity)
                if state is None:
                    state = FieldStatistics(torch.zeros_like(squares[row]), 0.0, step)
                    table[identity] = state
                elapsed = step - state.last_step
                if elapsed < 0:
                    raise ValueError("Per-field optimizer clocks cannot go backwards")
                decay = 2.0 ** (-elapsed / STAT_HALF_LIFE)
                state.squared_sums = state.squared_sums * decay + squares[row]
                state.count = state.count * decay + 1
                state.last_step = step
            canonical = self.fields[identity]
            sums.append(canonical.squared_sums)
            counts.append(canonical.count)
        sums_t = torch.stack(sums)
        counts_t = torch.tensor(counts, device=sums_t.device)[:, None]
        return (sums_t + 4 * prior[None]) / (counts_t + 4)

    def step(self, step: int) -> dict[str, int]:
        """One optimizer barrier: update touched masters, publish packed rows, apply writes."""
        if step <= self.step_number:
            raise ValueError("KB optimizer steps must increase strictly")
        grouped: dict[str, list[tuple[int, Tensor]]] = {}
        for (edge, name), leaf in self.bank.masters.items():
            if leaf.grad is None or edge not in self.bank:
                continue
            if name == "value" and not self.bank.learned_values:
                raise RuntimeError("Value gradients are forbidden before the switch")
            grouped.setdefault(name, []).append((edge, leaf))
        updated = 0
        observations: dict[str, Tensor] = {}
        for name, items in grouped.items():
            edges = [edge for edge, _ in items]
            leaves = [leaf for _, leaf in items]
            old = torch.stack([leaf.detach().float().reshape(-1) for leaf in leaves])
            grad = torch.stack([leaf.grad.detach().float().reshape(-1) for leaf in leaves])
            if not bool(torch.isfinite(grad).all()):
                raise FloatingPointError("Nonfinite sparse transducer gradient")
            address = name in ADDRESS_FIELDS
            if address:
                # Decoded FP8 addresses are only nearly unit: move on the sphere itself.
                old = tangent = unit(old)
                grad = grad - tangent * (grad * tangent).sum(-1, keepdim=True)
            squares = _blocks(grad)
            variance = self._statistics([(e, name) for e in edges], squares, step)
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
            with torch.no_grad():
                for leaf, row in zip(leaves, replacement, strict=True):
                    leaf.copy_(row.reshape(leaf.shape))
                    leaf.grad = None
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
            self.priors[name] = (
                prior_decay * self._prior(name, mean.numel()) + (1 - prior_decay) * mean
            )
        self.prior_step = step
        self.step_number = step
        self._evict()
        if updated or written:
            self.bank._changed()
        return {"updated_fields": updated, "written_values": written}

    def _remember_origin(self, name: str, edges: list[int]) -> None:
        origins = self.round_origin.setdefault(name, {})
        new = [edge for edge in edges if edge not in origins]
        if not new:
            return
        columns = self.bank.table.columns(name, self.bank.table.slots(new))
        for index, edge in enumerate(new):
            origins[edge] = {column: rows[index] for column, rows in columns.items()}

    def round_deltas(self) -> dict[str, tuple[Tensor, Tensor]]:
        """Per field: (edge IDs, published rows minus the round's base rows)."""
        result = {}
        table = self.bank.table
        for name, origins in self.round_origin.items():
            edges = [edge for edge in origins if edge in self.bank]
            if not edges:
                continue
            base = table.decode_columns(
                name,
                {
                    column: torch.stack([origins[edge][column] for edge in edges])
                    for column in next(iter(origins.values()))
                },
            )
            current = table.decode(name, table.slots(edges))
            delta = (current - base).reshape(len(edges), -1).float().cpu()
            result[name] = (torch.tensor(edges, dtype=torch.int64), delta)
        return result

    def _evict(self) -> None:
        """Drop the least recently used edges' masters beyond the cache capacity."""
        masters = self.bank.masters
        edges = list(dict.fromkeys(edge for edge, _ in masters))
        for edge in edges[: max(0, len(edges) - self.capacity)]:
            for name in ("key", "departure", "destination", "bias", "value"):
                masters.pop((edge, name), None)

    def forget(self, ids: Tensor | list[int]) -> None:
        """Retired edges lose their statistics and pending writes."""
        gone = set(torch.as_tensor(ids).tolist())
        for table in (self.fields, self.round_stats):
            for identity in [k for k in table if k[0] in gone]:
                del table[identity]
        for origins in self.round_origin.values():
            for edge in gone & set(origins):
                del origins[edge]
        for table in (
            self.write_mass,
            self.write_sums,
            self.round_write_mass,
            self.round_write_sums,
        ):
            for edge in gone & set(table):
                del table[edge]

    def copy_statistics(self, source: int, target: int) -> None:
        """A split child inherits half of its parent's statistics (the parent keeps half)."""
        for (edge, name), state in list(self.fields.items()):
            if edge == source:
                state.squared_sums = state.squared_sums / 2
                state.count /= 2
                self.fields[target, name] = FieldStatistics(
                    state.squared_sums.clone(), state.count, state.last_step
                )

    def switch_to_learned(self) -> None:
        if self.write_mass:
            raise RuntimeError("Flush written value evidence before switching")
        if self.bank.learned_values:
            raise RuntimeError("Value optimizer was already activated")
        self.bank.learned_values = True
        self.bank.table.write_mass.zero_()
        self.round_write_mass.clear()
        self.round_write_sums.clear()
        for identity in [k for k in self.bank.masters if k[1] == "value"]:
            del self.bank.masters[identity]
        self.bank._changed()

    def take_round(self) -> tuple[dict, dict, dict, dict]:
        """Deltas, statistics and writes since the last call, for a DiLoCo submission.

        The FP32 masters restart from the next installed bank.
        """
        deltas = self.round_deltas()
        stats, masses, sums = self.round_stats, self.round_write_mass, self.round_write_sums
        self.round_stats, self.round_write_mass, self.round_write_sums = {}, {}, {}
        self.round_origin = {}
        return deltas, stats, masses, sums

    # ------------------------------------------------------------ checkpoint

    def state_dict(self) -> dict:
        """Tensor state for an exact interrupted-run resume, masters included."""

        def stats(table: dict[tuple[int, str], FieldStatistics]) -> dict:
            return {
                f"{edge}/{name}": {
                    "squared_sums": s.squared_sums.cpu(),
                    "count": s.count,
                    "last_step": s.last_step,
                }
                for (edge, name), s in table.items()
            }

        return {
            "step": self.step_number,
            "prior_step": self.prior_step,
            "worker": self.worker,
            "priors": {k: v.cpu() for k, v in self.priors.items()},
            "fields": stats(self.fields),
            "round_stats": stats(self.round_stats),
            "masters": {
                f"{edge}/{name}": leaf.detach().cpu()
                for (edge, name), leaf in self.bank.masters.items()
            },
            "write_mass": {str(k): v for k, v in self.write_mass.items()},
            "write_sums": {str(k): v.cpu() for k, v in self.write_sums.items()},
            "round_write_mass": {str(k): v for k, v in self.round_write_mass.items()},
            "round_write_sums": {str(k): v.cpu() for k, v in self.round_write_sums.items()},
            "round_origin": {
                f"{edge}/{name}": {column: row.cpu() for column, row in columns.items()}
                for name, origins in self.round_origin.items()
                for edge, columns in origins.items()
            },
        }

    def load_state_dict(self, state: dict) -> None:
        if int(state["worker"]) != self.worker:
            raise ValueError("Cannot resume a different worker's fractional progress")
        device = self.bank.device

        def stats(raw: dict) -> dict[tuple[int, str], FieldStatistics]:
            result = {}
            for key, value in raw.items():
                edge, name = key.split("/")
                result[int(edge), name] = FieldStatistics(
                    value["squared_sums"].to(device), float(value["count"]), int(value["last_step"])
                )
            return result

        self.step_number = int(state["step"])
        self.prior_step = int(state["prior_step"])
        self.priors = {k: v.to(device) for k, v in state["priors"].items()}
        self.fields = stats(state["fields"])
        self.round_stats = stats(state["round_stats"])
        self.bank.masters.clear()
        for key, value in state["masters"].items():
            edge, name = key.split("/")
            if int(edge) not in self.bank:
                raise ValueError("Checkpoint master references a retired edge")
            self.bank.masters[int(edge), name] = value.to(device).detach().requires_grad_()
        self.write_mass = {int(k): float(v) for k, v in state["write_mass"].items()}
        self.write_sums = {int(k): v.to(device) for k, v in state["write_sums"].items()}
        self.round_write_mass = {int(k): float(v) for k, v in state["round_write_mass"].items()}
        self.round_write_sums = {int(k): v.to(device) for k, v in state["round_write_sums"].items()}
        self.round_origin = {}
        for key, columns in state["round_origin"].items():
            edge, name = key.split("/")
            self.round_origin.setdefault(name, {})[int(edge)] = {
                column: row.to(device) for column, row in columns.items()
            }
