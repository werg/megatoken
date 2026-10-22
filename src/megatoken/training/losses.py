"""Global-denominator losses with bounded activation memory.

Prepare all targets before the optimizer transaction, reduce their denominators,
then backward each microbatch immediately. This is mathematically equivalent to
one concatenated global batch even when ranks have different free-position masks.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import torch

from megatoken.training.metrics import materialize_scalars


@dataclass
class LossLedger:
    expected_local: dict[str, int]
    global_counts: dict[str, int]
    weights: dict[str, float]
    seen: dict[str, int] = field(default_factory=dict)
    _sums: dict[str, torch.Tensor] = field(default_factory=dict, repr=False)

    @property
    def sums(self) -> dict[str, float]:
        """Materialize logging values only after all local backwards finish."""
        return materialize_scalars(self._sums)

    def backward_loss(self, terms: Mapping[str, tuple[torch.Tensor, int]]) -> torch.Tensor:
        total: torch.Tensor | None = None
        for name, (numerator, denominator) in terms.items():
            if name not in self.weights or name not in self.global_counts:
                raise ValueError(f"Unplanned loss term {name!r}")
            if numerator.numel() != 1 or denominator < 0:
                raise ValueError(f"{name}: loss must be a scalar sum with a nonnegative count")
            self.seen[name] = self.seen.get(name, 0) + int(denominator)
            detached = numerator.detach().float()
            self._sums[name] = self._sums.get(name, 0.0) + detached
            count = self.global_counts[name]
            if denominator and count == 0:
                raise ValueError(f"{name}: nonempty local loss has zero global denominator")
            term = numerator * (self.weights[name] / count if count else 0.0)
            total = term if total is None else total + term
        if total is None:
            raise ValueError("A supervised step returned no losses")
        return total

    def finish(self) -> None:
        for name, expected in self.expected_local.items():
            actual = self.seen.get(name, 0)
            if actual != expected:
                raise RuntimeError(
                    f"Loss plan drift for {name}: expected {expected}, scored {actual}"
                )
