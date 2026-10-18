"""Independent U/D/I/P/T/Z meters; latent coordinates are not lexical tokens."""

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
    Z: int = 0
    T: int = 0
    teacher_input: int = 0
    teacher_cache_read: int = 0
    teacher_cache_write: int = 0
    facet_encodes: int = 0
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
        actual_scored_positions: int,
        latent_coordinates: int = 0,
    ) -> None:
        if min(actual_scored_positions, latent_coordinates) < 0:
            raise ValueError("Actual lexical work cannot be negative")
        self.D += window.D
        self.I += window.I
        self.P += actual_scored_positions
        self.Z += latent_coordinates
        self.record_ingestions += window.record_ingestions
        self.reconstruction_ingestions += window.prefix_count

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Budget:
    meter: str
    limit: int

    def __post_init__(self) -> None:
        if self.meter not in {"U", "D", "I", "P", "T", "Z"} or self.limit <= 0:
            raise ValueError("A budget requires one explicit meter and a positive limit")

    def exhausted(self, meters: TokenMeters) -> bool:
        return getattr(meters, self.meter) >= self.limit

    def remaining(self, meters: TokenMeters) -> int:
        return max(0, self.limit - getattr(meters, self.meter))
