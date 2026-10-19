"""Stable-ID sparse AdamW leaves; existing model parameters stay in MasterAdamW."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .contracts import Collection
from .sqlite import SQLiteKB


@dataclass(frozen=True)
class RowID:
    collection: Collection
    kind: str
    id: int
    field: str
    layout: str


class SparseRowAdam:
    """One FP32 leaf per selected row/field, shared by repeated hits in a step.

    Caller synchronizes gradients and commits bank rows at the model optimizer
    barrier. This object stores only touched optimizer slots, not dense bank
    moments or another network optimizer.
    """

    def __init__(self, bank: SQLiteKB, *, lr: float = 1e-3, beta1: float = 0.9,
                 beta2: float = 0.999, eps: float = 1e-8, weight_decay: float = 0):
        if lr <= 0 or not (0 < beta1 < 1 and 0 < beta2 < 1) or eps <= 0:
            raise ValueError("Invalid sparse Adam hyperparameters")
        self.bank, self.lr, self.beta1, self.beta2 = bank, lr, beta1, beta2
        self.eps, self.weight_decay = eps, weight_decay
        self.leaves: dict[RowID, Tensor] = {}

    def select(self, identity: RowID) -> Tensor:
        if identity not in self.leaves:
            if identity.kind == "record":
                record = self.bank.get(identity.collection, identity.id)
                fields = {"key": record.key, "value": record.value,
                          "strength": [record.strength],
                          "lifetime_logit": [record.lifetime_logit]}
            elif identity.kind == "relation":
                record = next((x for x in self.bank.relations_by_id([identity.id])),
                              None)
                if record is None:
                    raise KeyError(identity)
                fields = {"value": record.value,
                          "dependence_logits": record.dependence_logits,
                          "strength": [record.strength],
                          "lifetime_logit": [record.lifetime_logit]}
            else:
                raise ValueError("Unsupported sparse row kind")
            if identity.field not in fields or fields[identity.field] is None:
                raise KeyError(identity)
            self.leaves[identity] = torch.tensor(
                fields[identity.field], dtype=torch.float32, requires_grad=True
            )
        return self.leaves[identity]

    def step(self, *, step: int, max_norm: float = 1.0) -> int:
        if max_norm <= 0:
            raise ValueError("Invalid sparse clipping threshold")
        updated = 0
        with self.bank.transaction():
            for identity, leaf in self.leaves.items():
                if leaf.grad is None:
                    continue
                grad = leaf.grad.detach()
                if not bool(torch.isfinite(grad).all()):
                    raise FloatingPointError("Nonfinite sparse KB gradient")
                grad = grad * min(1.0, max_norm / (float(grad.norm()) + 1e-12))
                prior = self.bank.moments(
                    collection=identity.collection, kind=identity.kind,
                    entity_id=identity.id, field=identity.field,
                    layout_generation=identity.layout,
                )
                if prior is None:
                    m, v, n = torch.zeros_like(grad), torch.zeros_like(grad), 0
                else:
                    m, v, n = (torch.from_numpy(prior[0]).reshape_as(grad),
                               torch.from_numpy(prior[1]).reshape_as(grad), prior[2])
                n += 1
                m = self.beta1 * m + (1 - self.beta1) * grad
                v = self.beta2 * v + (1 - self.beta2) * grad.square()
                delta = m / (1 - self.beta1**n) / (
                    (v / (1 - self.beta2**n)).sqrt() + self.eps
                )
                target = leaf.detach() * (1 - self.lr * self.weight_decay) - self.lr * delta
                values = target.cpu().numpy()
                if identity.kind == "record":
                    self.bank.update_record(identity.collection, identity.id,
                                            **{identity.field: values if values.size > 1
                                               else float(values[0])})
                else:
                    self.bank.update_relation(identity.id,
                                              **{identity.field: values if values.size > 1
                                                 else float(values[0])})
                self.bank.write_moments(
                    collection=identity.collection, kind=identity.kind, entity_id=identity.id,
                    field=identity.field, layout_generation=identity.layout,
                    first=m.numpy(), second=v.numpy(), updates=n, step=step, row_revision=n,
                )
                updated += 1
        self.leaves.clear()
        return updated
