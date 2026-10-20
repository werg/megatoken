"""Stable-ID sparse AdamW leaves; existing model parameters stay in MasterAdamW."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .contracts import Collection
from .disk_ivf import SQLiteDiskIVF
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
                 beta2: float = 0.999, eps: float = 1e-8, weight_decay: float = 0,
                 index: SQLiteDiskIVF | None = None):
        if lr <= 0 or not (0 < beta1 < 1 and 0 < beta2 < 1) or eps <= 0:
            raise ValueError("Invalid sparse Adam hyperparameters")
        if index is not None and index.store is not bank:
            raise ValueError("Sparse postings must share the canonical SQLite transaction")
        self.bank, self.lr, self.beta1, self.beta2 = bank, lr, beta1, beta2
        self.index = index
        self.eps, self.weight_decay = eps, weight_decay
        self.leaves: dict[RowID, Tensor] = {}
        self._selected_revisions: dict[tuple[Collection, str, int], tuple[int, ...]] = {}

    def select(self, identity: RowID) -> Tensor:
        if identity not in self.leaves:
            if identity.kind == "record":
                record = self.bank.get(identity.collection, identity.id)
                fields = {"key": record.key, "value": record.value,
                          "strength": [record.strength],
                          "lifetime_logit": [record.lifetime_logit]}
            elif identity.kind == "relation":
                record = self.bank.get_relation(identity.id)
                fields = {"value": record.value,
                          "dependence_logits": record.dependence_logits,
                          "strength": [record.strength],
                          "lifetime_logit": [record.lifetime_logit]}
            else:
                raise ValueError("Unsupported sparse row kind")
            if identity.field not in fields or fields[identity.field] is None:
                raise KeyError(identity)
            row = (identity.collection, identity.kind, identity.id)
            revision = ((record.key_revision, record.payload_revision)
                        if identity.kind == "record" else (record.payload_revision,))
            if row in self._selected_revisions and self._selected_revisions[row] != revision:
                raise RuntimeError("KB row changed during a selected training window")
            self._selected_revisions[row] = revision
            self.leaves[identity] = torch.tensor(
                fields[identity.field], dtype=torch.float32, requires_grad=True
            )
        return self.leaves[identity]

    def step(self, *, step: int, max_norm: float = 1.0) -> int:
        if max_norm <= 0:
            raise ValueError("Invalid sparse clipping threshold")
        updated = 0
        with self.bank.transaction():
            # Check all selected owners at one barrier BEFORE writing any row.
            # Per-field compare-and-swap during sequential updates would reject
            # our own updates to another field of the same record.
            for (collection, kind, entity_id), selected in self._selected_revisions.items():
                record = (self.bank.get(collection, entity_id) if kind == "record"
                          else self.bank.get_relation(entity_id))
                current = ((record.key_revision, record.payload_revision)
                           if kind == "record" else (record.payload_revision,))
                if selected != current:
                    raise RuntimeError("Stale selected KB row at sparse optimizer barrier")
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
                    if prior[3] >= step:
                        raise RuntimeError("Sparse KB optimizer step is not strictly increasing")
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
                    replacement = values if values.size > 1 else float(values[0])
                    index_update = {}
                    if self.index is not None and identity.field == "key":
                        index_update = {
                            "posting_cell": self.index.cell(values),
                            "centroids_revision": self.index.identity.centroid_sha256,
                        }
                    self.bank.update_record(
                        identity.collection, identity.id,
                        **{identity.field: replacement}, **index_update,
                    )
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
        self._selected_revisions.clear()
        return updated
