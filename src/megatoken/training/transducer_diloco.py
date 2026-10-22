"""v10 DiLoCo KB coordinator: age-weighted sparse deltas, writes and new edges.

There is deliberately no outer KB optimizer, momentum or demand-forwarding path.
The coordinator alone allocates canonical edge IDs and evolves structure.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import torch
from torch import Tensor

from megatoken.memory.transducer import PackedEdge, TransducerBank
from megatoken.memory.transducer_growth import EdgeEvidence, TransducerGrowth
from megatoken.memory.transducer_optimizer import BlockScaleDescent
from megatoken.memory.transducer_packing import pack_fp4, pack_fp8, rounding_seed, unit


@dataclass
class ValueWrite:
    mass: float
    mean: object  # PackedField; encoded FP4 on the wire


@dataclass
class WorkerRound:
    worker: int
    sequence: int
    base_version: int
    examples: int
    deltas: dict[tuple[int, str], Tensor] = field(default_factory=dict)
    writes: dict[int, ValueWrite] = field(default_factory=dict)
    optimizer_increments: dict[tuple[int, str], tuple[Tensor, float]] = field(
        default_factory=dict
    )
    structural_increments: dict[int, EdgeEvidence] = field(default_factory=dict)
    provisional: list[PackedEdge] = field(default_factory=list)

    def __post_init__(self) -> None:
        if min(self.worker, self.sequence, self.base_version) < 0 or self.examples < 1:
            raise ValueError("Worker round identity and processed-example weight are required")


class TransducerCoordinator:
    """One canonical owner. Submit in any order and merge once per 32-step round."""

    local_steps = 32

    def __init__(
        self,
        bank: TransducerBank,
        optimizer: BlockScaleDescent,
        growth: TransducerGrowth,
    ) -> None:
        if optimizer.bank is not bank or growth.bank is not bank:
            raise ValueError("Coordinator components must share the exact bank")
        self.bank = bank
        self.optimizer = optimizer
        self.growth = growth
        self.version = 0
        self.ledger: dict[int, int] = {}
        self.pending: list[WorkerRound] = []
        self.learned = bank.learned_values
        self.validation: list[tuple[int, float]] = []
        self.switch_step: int | None = None

    def submit(self, round_: WorkerRound) -> bool:
        if round_.sequence <= self.ledger.get(round_.worker, -1) or any(
            (old.worker, old.sequence) == (round_.worker, round_.sequence)
            for old in self.pending
        ):
            return False
        if round_.base_version > self.version:
            raise ValueError("Worker cannot train from an unpublished future bank")
        self.pending.append(round_)
        return True

    def maybe_switch(self, step: int, bank_only_loss: float) -> bool:
        if step <= 0 or step % 1024:
            raise ValueError("Switch validation occurs on exactly 1024-step boundaries")
        if not torch.isfinite(torch.tensor(bank_only_loss)):
            raise FloatingPointError("Invalid bank-only continuation loss")
        if self.validation and step <= self.validation[-1][0]:
            raise ValueError("Validation steps must increase")
        self.validation.append((step, bank_only_loss))
        if self.learned or len(self.bank.rows) < 262144 or len(self.validation) < 16:
            return False
        baseline = self.validation[-16][1]
        improvement = (baseline - bank_only_loss) / max(abs(baseline), 1e-12)
        if improvement >= 0.005:
            return False
        if self.optimizer.write_accumulators:
            raise RuntimeError("Cannot switch before flushing the written-value window")
        self.optimizer.switch_to_learned()
        self.learned = True
        self.switch_step = step
        self.growth.evidence = {
            edge_id: replace(evidence, dispersion=0.0, sketch=torch.zeros(64), energy=0.0)
            for edge_id, evidence in self.growth.evidence.items()
        }
        self.optimizer.fields = {
            identity: state
            for identity, state in self.optimizer.fields.items()
            if identity[1] != "value"
        }
        return True

    def _merge_provisional(self, submissions: list[WorkerRound]) -> int:
        incoming: list[tuple[PackedEdge, int, int]] = [
            (row, submission.worker, submission.examples)
            for submission in submissions
            for row in submission.provisional
        ]
        adopted = 0
        # Combine same-location, same-query NEW rows from distinct learners.
        clusters: list[list[tuple[PackedEdge, int, int]]] = []
        for row, worker, examples in incoming:
            d, k = unit(row.departure.decode()), unit(row.key.decode())
            matched = False
            for cluster in clusters:
                lead, original_worker, _ = cluster[0]
                if original_worker == worker:
                    continue
                if float(d @ unit(lead.departure.decode())) >= 0.9 and float(
                    k @ unit(lead.key.decode())
                ) >= 0.9:
                    cluster.append((row, worker, examples))
                    matched = True
                    break
            if not matched:
                clusters.append([(row, worker, examples)])
        for cluster in clusters:
            if len(self.bank.rows) >= self.bank.geometry.max_edges:
                break
            total_examples = sum(examples for _, _, examples in cluster)
            masses = [max(row.write_mass, 0.0) for row, _, _ in cluster]
            denominator = sum(masses) or len(cluster)
            value = sum(
                (m if sum(masses) else 1.0) * row.value.decode()
                for (row, _, _), m in zip(cluster, masses, strict=True)
            ) / denominator
            key = unit(sum(
                examples * row.key.decode() for row, _, examples in cluster
            ) / total_examples)
            departure = unit(sum(
                examples * row.departure.decode() for row, _, examples in cluster
            ) / total_examples)
            destination = unit(sum(
                examples * row.destination.decode() for row, _, examples in cluster
            ) / total_examples)
            bias = sum(
                examples * row.bias for row, _, examples in cluster
            ) / total_examples
            try:
                edge_id = self.bank.add_edge(
                    departure=departure,
                    key=key,
                    value=value,
                    destination=destination,
                    bias=bias,
                    step=min(row.birth_step for row, _, _ in cluster),
                    write_mass=sum(masses),
                )
            except RuntimeError:
                break
            self.growth.evidence[edge_id] = EdgeEvidence()
            adopted += 1
        return adopted

    def merge(self, *, step: int) -> dict[str, int]:
        if step < 0 or step % self.local_steps:
            raise ValueError("Coordinator publishes only complete 32-step KB rounds")
        if not self.pending:
            return {"submitted": 0, "fields": 0, "written": 0, "adopted": 0}
        workers = self.pending
        # All participants count in the denominator even for untouched rows.
        weights = [
            float(submission.examples) * 2 ** (-(self.version - submission.base_version) / 8)
            for submission in workers
        ]
        denominator = sum(weights)
        if denominator <= 0:
            raise ValueError("No positive merge evidence")
        deltas: dict[tuple[int, str], Tensor] = {}
        written: dict[int, list[tuple[float, Tensor]]] = {}
        for submission, weight in zip(workers, weights, strict=True):
            for identity, delta in submission.deltas.items():
                if identity[0] not in self.bank.rows:
                    continue  # Retired since this learner's base.
                if identity[1] == "value" and not self.bank.learned_values:
                    raise ValueError("Written-phase values must merge as mass-weighted means")
                contribution = delta.detach().float().cpu() * weight
                if not bool(torch.isfinite(contribution).all()):
                    raise FloatingPointError("Nonfinite submitted KB row delta")
                if identity in deltas:
                    deltas[identity] += contribution
                else:
                    deltas[identity] = contribution
            if not self.bank.learned_values:
                for edge_id, write in submission.writes.items():
                    if edge_id in self.bank.rows and write.mass > 0:
                        aged = write.mass * weight / submission.examples
                        written.setdefault(edge_id, []).append(
                            (aged, write.mean.decode())
                        )
            for identity, (sums, count) in submission.optimizer_increments.items():
                if identity not in self.optimizer.fields or identity[0] not in self.bank.rows:
                    continue
                slot = self.optimizer.fields[identity]
                discount = weight / submission.examples
                slot.squared_sums += sums.detach().cpu() * discount
                slot.count += count * discount
            for edge_id, incoming in submission.structural_increments.items():
                if edge_id not in self.bank.rows:
                    continue
                slot = self.growth.evidence.setdefault(edge_id, EdgeEvidence())
                discount = weight / submission.examples
                slot.utility += incoming.utility * discount
                slot.visits += incoming.visits * discount
                slot.dispersion += incoming.dispersion * discount
                slot.sketch += incoming.sketch.cpu() * discount
                slot.energy += incoming.energy * discount
        for (edge_id, name), delta in deltas.items():
            previous = self.bank.field(edge_id, name)
            target = previous.cpu() + delta / denominator
            if name in {"key", "departure", "destination"}:
                target = unit(target)
                packed = pack_fp8(
                    target,
                    stochastic_seed=rounding_seed(edge_id, name, step, "coordinator"),
                )
                self.bank.rows[edge_id] = replace(self.bank.rows[edge_id], **{name: packed})
            elif name == "value":
                packed = pack_fp4(
                    target,
                    stochastic_seed=rounding_seed(edge_id, name, step, "coordinator"),
                )
                self.bank.rows[edge_id] = replace(self.bank.rows[edge_id], value=packed)
            else:
                self.bank.rows[edge_id] = replace(self.bank.rows[edge_id], bias=float(target))
        for edge_id, updates in written.items():
            row = self.bank.rows[edge_id]
            last = self.optimizer.last_write.get(edge_id, row.birth_step)
            old_mass = row.write_mass * 2 ** (-(step - last) / 16384)
            new_mass = old_mass + sum(mass for mass, _ in updates)
            merged = (
                old_mass * row.value.decode()
                + sum(mass * mean for mass, mean in updates)
            ) / new_mass
            self.bank.rows[edge_id] = replace(
                row,
                value=pack_fp4(
                    merged,
                    stochastic_seed=rounding_seed(edge_id, "value", step, "coordinator"),
                ),
                write_mass=new_mass,
            )
            self.optimizer.last_write[edge_id] = step
        adopted = self._merge_provisional(workers)
        for submission in workers:
            self.ledger[submission.worker] = submission.sequence
        self.pending = []
        self.version += 1
        self.bank._masters.clear()
        self.bank._invalidate()
        return {
            "submitted": len(workers),
            "fields": len(deltas),
            "written": len(written),
            "adopted": adopted,
        }

    def state_dict(self) -> dict:
        if self.pending:
            raise RuntimeError("Checkpoint only after a fully merged coordinator round")
        return {
            "version": self.version,
            "ledger": self.ledger.copy(),
            "learned": self.learned,
            "validation": self.validation.copy(),
            "switch_step": self.switch_step,
            "identity": self.bank.identity,
        }

    def load_state_dict(self, snapshot: dict) -> None:
        if snapshot["identity"] != self.bank.identity:
            raise ValueError("v9 KB state cannot be restored into a v10 coordinator")
        if bool(snapshot["learned"]) != self.bank.learned_values:
            raise ValueError("Coordinator and bank value phases disagree")
        self.version = int(snapshot["version"])
        self.ledger = {int(k): int(v) for k, v in snapshot["ledger"].items()}
        self.learned = bool(snapshot["learned"])
        self.validation = [(int(s), float(v)) for s, v in snapshot["validation"]]
        self.switch_step = snapshot["switch_step"]
        self.pending = []
