"""Momentum-free, discounted block-scale optimization for v10 packed KB rows."""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass, replace

import torch
from torch import Tensor

from .transducer import TransducerBank
from .transducer_packing import pack_fp4, pack_fp8, rounding_seed, unit

ADDRESS_FIELDS = ("key", "departure", "destination")
TRAINED_FIELDS = (*ADDRESS_FIELDS, "bias", "value")


@dataclass
class FieldStatistics:
    squared_sums: Tensor
    count: float = 0.0
    last_step: int = 0


@dataclass
class WrittenValue:
    mass: float = 0.0
    weighted_sum: Tensor | None = None


class BlockScaleDescent:
    """No momentum vector, no idle updates, and only block-sized second moments.

    FP32 masters survive across windows in a bounded LRU. Each forward reads
    packed codes with straight-through gradients to the selected masters.
    Publication happens only at a completed optimizer/DiLoCo boundary.
    """

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
        self.lru: OrderedDict[int, None] = OrderedDict()
        self.write_accumulators: dict[int, WrittenValue] = {}
        self.last_write: dict[int, int] = {}
        self.step_number = 0
        self.prior_step = 0

    @staticmethod
    def _blocks(values: Tensor) -> Tensor:
        return torch.stack(
            [piece.square().mean() for piece in values.float().split(256)]
        )

    def _prior(self, field: str, blocks: int) -> Tensor:
        if field not in self.priors:
            self.priors[field] = torch.full((blocks,), 1e-6, dtype=torch.float32)
        if self.priors[field].numel() != blocks:
            raise ValueError("Optimizer block geometry changed")
        return self.priors[field]

    def add_write(
        self, edge_id: int, hindsight: Tensor, weight: float, *, sink_only: bool = False
    ) -> None:
        if sink_only or weight == 0:
            return
        if self.bank.learned_values:
            raise RuntimeError("Hindsight writes stop at the learned-value switch")
        if edge_id not in self.bank.rows or weight < 0 or not math.isfinite(weight):
            raise ValueError("Invalid written KB evidence")
        if hindsight.numel() != self.bank.geometry.value_dim or not bool(
            torch.isfinite(hindsight).all()
        ):
            raise ValueError("Written knowledge has the wrong width or nonfinite values")
        accumulator = self.write_accumulators.setdefault(edge_id, WrittenValue())
        accumulator.mass += weight
        contribution = hindsight.detach().float().cpu() * weight
        if accumulator.weighted_sum is None:
            accumulator.weighted_sum = contribution
        else:
            accumulator.weighted_sum += contribution

    def _publish(self, edge_id: int, name: str, value: Tensor, step: int) -> None:
        row = self.bank.rows[edge_id]
        if name == "bias":
            if value.numel() != 1 or not math.isfinite(float(value)):
                raise FloatingPointError("Nonfinite bias publication")
            row = replace(row, bias=float(value))
        elif name == "value":
            row = replace(
                row,
                value=pack_fp4(
                    value, stochastic_seed=rounding_seed(edge_id, name, step, self.worker)
                ),
            )
        else:
            row = replace(
                row,
                **{
                    name: pack_fp8(
                        unit(value.float()),
                        stochastic_seed=rounding_seed(edge_id, name, step, self.worker),
                    )
                },
            )
        self.bank.rows[edge_id] = row

    def _apply_writes(self, step: int) -> int:
        written = 0
        for edge_id, accumulator in self.write_accumulators.items():
            if edge_id not in self.bank.rows or accumulator.mass <= 0:
                continue
            row = self.bank.rows[edge_id]
            age = step - self.last_write.get(edge_id, row.birth_step)
            if age < 0:
                raise ValueError("Write clocks must be monotonic")
            old_mass = row.write_mass * 2 ** (-age / 16384)
            new_mass = old_mass + accumulator.mass
            updated = (
                row.value.decode() * old_mass + accumulator.weighted_sum
            ) / new_mass
            self._publish(edge_id, "value", updated, step)
            self.bank.rows[edge_id] = replace(
                self.bank.rows[edge_id], write_mass=new_mass
            )
            self.last_write[edge_id] = step
            written += 1
        self.write_accumulators.clear()
        return written

    def step(self, step: int) -> dict[str, int]:
        """One optimizer barrier: train touched leaves, publish packed rows, then writes."""
        if step <= self.step_number:
            raise ValueError("KB optimizer steps must increase strictly")
        touched: dict[str, list[Tensor]] = {}
        updated = 0
        selected = {
            i for (i, _), leaf in self.bank._masters.items() if leaf.grad is not None
        }
        if len(selected) > self.capacity:
            raise RuntimeError("Active sparse backward window exceeds FP32 cache capacity")
        for (edge_id, name), leaf in list(self.bank._masters.items()):
            if leaf.grad is None or edge_id not in self.bank.rows:
                continue
            if name == "value" and not self.bank.learned_values:
                raise RuntimeError("Value gradients are forbidden before the switch")
            grad = leaf.grad.detach().float()
            if not bool(torch.isfinite(grad).all()):
                raise FloatingPointError("Nonfinite sparse transducer gradient")
            address = name in ADDRESS_FIELDS
            old = leaf.detach().float()
            if address:
                tangent = unit(old)
                grad = grad - tangent * torch.dot(grad, tangent)
            squares = self._blocks(grad)
            identity = edge_id, name
            state = self.fields.get(identity)
            if state is None:
                state = FieldStatistics(torch.zeros_like(squares), last_step=step)
                self.fields[identity] = state
            elapsed = step - state.last_step
            if elapsed < 0:
                raise ValueError("Per-field optimizer clocks cannot go backwards")
            decay = 2 ** (-elapsed / 4096)
            state.squared_sums = state.squared_sums * decay + squares.cpu()
            state.count = state.count * decay + 1
            state.last_step = step
            prior = self._prior(name, squares.numel())
            variance = (state.squared_sums + 4 * prior) / (state.count + 4)
            chunks = []
            for piece, var in zip(grad.split(256), variance, strict=True):
                chunks.append(piece / (torch.sqrt(var.to(piece.device).clamp_min(1e-12)) + 1e-8))
            delta = torch.cat(chunks) * -(self.value_lr if name == "value" else self.address_lr)
            if address:
                delta = delta - tangent * torch.dot(delta, tangent)
                angle = float(torch.linalg.vector_norm(delta))
                if angle > 0.01:
                    delta *= 0.01 / angle
                replacement = unit(old + delta)
            else:
                if name == "value":
                    clipped = []
                    for before, move in zip(old.split(256), delta.split(256), strict=True):
                        limit = 0.01 * max(float(before.square().mean().sqrt()), 1e-3)
                        rms = float(move.square().mean().sqrt())
                        clipped.append(move * min(1.0, limit / (rms + 1e-30)))
                    delta = torch.cat(clipped)
                replacement = old + delta
            if not bool(torch.isfinite(replacement).all()):
                raise FloatingPointError("Nonfinite sparse KB parameter update")
            with torch.no_grad():
                leaf.copy_(replacement)
                leaf.grad = None
            self._publish(edge_id, name, replacement.cpu(), step)
            touched.setdefault(name, []).append(squares.cpu())
            self.lru.pop(edge_id, None)
            self.lru[edge_id] = None
            updated += 1
        written = 0 if self.bank.learned_values else self._apply_writes(step)
        elapsed = step - self.prior_step
        prior_decay = 2 ** (-elapsed / 256)
        for name, observations in touched.items():
            mean = torch.stack(observations).mean(0)
            self.priors[name] = (
                prior_decay * self._prior(name, mean.numel()) + (1 - prior_decay) * mean
            )
        self.prior_step = step
        self.step_number = step
        while len(self.lru) > self.capacity:
            evicted, _ = self.lru.popitem(last=False)
            for key in list(self.bank._masters):
                if key[0] == evicted:
                    del self.bank._masters[key]
        if updated or written:
            self.bank._invalidate()
        return {"updated_fields": updated, "written_values": written}

    def switch_to_learned(self) -> None:
        if self.write_accumulators:
            raise RuntimeError("Flush written value evidence before switching")
        if self.bank.learned_values:
            raise RuntimeError("Value optimizer was already activated")
        self.bank.learned_values = True
        for identity in list(self.bank._masters):
            if identity[1] == "value":
                del self.bank._masters[identity]
        self.bank._invalidate()

    def state_dict(self) -> dict:
        """Includes resident fractions and writes for exact interrupted-run continuation."""
        return {
            "step": self.step_number,
            "prior_step": self.prior_step,
            "worker": self.worker,
            "lru": list(self.lru),
            "priors": {key: value.tolist() for key, value in self.priors.items()},
            "fields": {
                f"{edge}/{name}": {
                    "squared_sums": state.squared_sums.tolist(),
                    "count": state.count,
                    "last_step": state.last_step,
                }
                for (edge, name), state in self.fields.items()
            },
            "masters": {
                f"{edge}/{name}": master.detach().cpu().tolist()
                for (edge, name), master in self.bank._masters.items()
            },
            "writes": {
                str(edge): {
                    "mass": value.mass,
                    "weighted_sum": (
                        None if value.weighted_sum is None else value.weighted_sum.tolist()
                    ),
                }
                for edge, value in self.write_accumulators.items()
            },
            "last_write": self.last_write.copy(),
        }

    def load_state_dict(self, state: dict) -> None:
        if int(state["worker"]) != self.worker:
            raise ValueError("Cannot resume a different worker's fractional progress")
        self.step_number = int(state["step"])
        self.prior_step = int(state["prior_step"])
        self.priors = {
            key: torch.tensor(value, dtype=torch.float32)
            for key, value in state["priors"].items()
        }
        self.fields = {}
        for key, value in state["fields"].items():
            edge, name = key.split("/")
            self.fields[int(edge), name] = FieldStatistics(
                torch.tensor(value["squared_sums"]), value["count"], value["last_step"]
            )
        self.bank._masters = {}
        for key, values in state["masters"].items():
            edge, name = key.split("/")
            if int(edge) not in self.bank.rows:
                raise ValueError("Checkpoint master references a retired edge")
            self.bank._masters[int(edge), name] = (
                torch.tensor(values, dtype=torch.float32, device=self.bank.device)
                .detach().requires_grad_()
            )
        self.lru = OrderedDict((int(edge), None) for edge in state["lru"])
        if len(self.lru) > self.capacity:
            raise ValueError("Checkpoint exceeds configured FP32 master capacity")
        self.write_accumulators = {
            int(edge): WrittenValue(
                value["mass"],
                None if value["weighted_sum"] is None else torch.tensor(value["weighted_sum"]),
            )
            for edge, value in state["writes"].items()
        }
        self.last_write = {int(k): int(v) for k, v in state["last_write"].items()}
