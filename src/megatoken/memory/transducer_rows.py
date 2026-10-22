"""Dense per-edge columns keyed by stable edge IDs.

Every per-edge record outside the packed table (optimizer statistics, pending
writes, structural evidence, round origins, FP32 masters) lives in one of
these: tensors with one row per edge, rows allocated, zeroed and reused in
batches, so per-edge bookkeeping is a handful of tensor operations per call
rather than one per edge.
"""

from __future__ import annotations

import torch
from torch import Tensor


class RowStore:
    """Named columns of shape ``[rows, *tail]``; ``row_of`` maps edge IDs to rows."""

    def __init__(
        self,
        device: torch.device | str,
        columns: dict[str, tuple[tuple[int, ...], torch.dtype]],
        capacity: int = 256,
    ) -> None:
        self.device = torch.device(device)
        self.spec = dict(columns)
        self.row_of: dict[int, int] = {}
        self.free: list[int] = []
        self.size = 0
        self.capacity = 0
        self._grow(capacity)

    def _grow(self, capacity: int) -> None:
        for name, (tail, dtype) in self.spec.items():
            fresh = torch.zeros((capacity, *tail), device=self.device).to(dtype)
            if self.capacity:
                fresh[: self.capacity] = getattr(self, name)
            setattr(self, name, fresh)
        self.capacity = capacity

    def __len__(self) -> int:
        return len(self.row_of)

    def __contains__(self, edge: int) -> bool:
        return edge in self.row_of

    def ids(self) -> list[int]:
        return sorted(self.row_of)

    def find(self, ids: list[int]) -> Tensor:
        """Rows of edges that must already be present."""
        return torch.tensor([self.row_of[i] for i in ids], dtype=torch.int64, device=self.device)

    def rows(self, ids: list[int]) -> tuple[Tensor, list[int]]:
        """Rows for ``ids`` (repeats share a row) and the positions whose row was created.

        Created rows are zero in every column.
        """
        get = self.row_of.get
        found = [get(edge) for edge in ids]
        created = [k for k, row in enumerate(found) if row is None]
        if created:
            fresh: list[int] = []
            for k in created:
                row = self.row_of.get(ids[k])
                if row is None:
                    row = self.free.pop() if self.free else self._claim()
                    self.row_of[ids[k]] = row
                    fresh.append(row)
                found[k] = row
            index = torch.tensor(fresh, dtype=torch.int64, device=self.device)
            for name in self.spec:
                getattr(self, name)[index] = 0
        return torch.tensor(found, dtype=torch.int64, device=self.device), created

    def _claim(self) -> int:
        if self.size == self.capacity:
            self._grow(self.capacity * 2)
        self.size += 1
        return self.size - 1

    def drop(self, ids) -> None:
        for edge in ids:
            row = self.row_of.pop(int(edge), None)
            if row is not None:
                self.free.append(row)

    def clear(self) -> None:
        self.row_of, self.free, self.size = {}, [], 0

    def columns(self, ids: list[int], names=None) -> dict[str, Tensor]:
        rows = self.find(ids)
        return {name: getattr(self, name)[rows] for name in names or self.spec}

    def state_dict(self) -> dict[str, Tensor]:
        ids = self.ids()
        return {
            "ids": torch.tensor(ids, dtype=torch.int64),
            **{name: value.cpu() for name, value in self.columns(ids).items()},
        }

    def load_state_dict(self, state: dict[str, Tensor]) -> None:
        count = int(state["ids"].numel())
        self.clear()
        self.capacity = 0
        self._grow(max(256, count))
        for name in self.spec:
            getattr(self, name)[:count] = state[name].to(self.device)
        self.row_of = {int(edge): row for row, edge in enumerate(state["ids"].tolist())}
        self.size = count
