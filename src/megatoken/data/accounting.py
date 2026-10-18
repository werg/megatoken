"""Independent U/D/I/P/T meters; no ambiguous aggregate token counter."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .packing import TrainingWindow


@dataclass
class TokenMeters:
    U: int = 0
    D: int = 0
    I: int = 0  # noqa: E741 - the plan defines I as actual transcript ingestion.
    P: int = 0
    T: int = 0
    teacher_input: int = 0
    teacher_cache_read: int = 0
    teacher_cache_write: int = 0
    posterior_encodes: int = 0
    control_encodes: int = 0
    record_ingestions: int = 0
    reconstruction_ingestions: int = 0

    def __post_init__(self) -> None:
        if any(type(value) is not int or value < 0 for value in asdict(self).values()):
            raise ValueError("Token/work meters must be nonnegative integers")
        if self.I < self.D:
            raise ValueError("I includes D and cannot be smaller")

    def observe_window(
        self,
        window: TrainingWindow,
        *,
        actual_scored_positions: int | None = None,
        repair_positions: int = 0,
        identifier_positions: int = 0,
    ) -> None:
        scored = window.P if actual_scored_positions is None else actual_scored_positions
        if min(scored, repair_positions, identifier_positions) < 0:
            raise ValueError("Actual lexical work cannot be negative")
        self.D += window.D
        self.I += window.I
        self.P += scored + repair_positions + identifier_positions
        self.record_ingestions += window.record_ingestions
        self.reconstruction_ingestions += window.prefix_count

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Budget:
    meter: str
    limit: int

    def __post_init__(self) -> None:
        if self.meter not in {"U", "D", "I", "P", "T"} or self.limit <= 0:
            raise ValueError("A budget requires one explicit meter and a positive limit")

    def exhausted(self, meters: TokenMeters) -> bool:
        return getattr(meters, self.meter) >= self.limit

    def remaining(self, meters: TokenMeters) -> int:
        return max(0, self.limit - getattr(meters, self.meter))
