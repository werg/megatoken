"""Explicit, auditable source and episode filters without model-side labels."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .schema import EncodedRecord, Episode


class QualityReject(ValueError):
    pass


class RowFilterReject(QualityReject):
    """A source's own declared row filter excludes this row; selection, not an error."""


@dataclass(frozen=True, slots=True)
class QualityPolicy:
    min_tokens: int = 1
    max_tokens: int = 262144
    min_scored_positions: int = 0
    required_verified_success: bool = False
    required_executed_observation: bool = False
    allowed_outcomes: tuple[str, ...] = ()
    max_repeated_line_fraction: float = 1.0

    def __post_init__(self) -> None:
        if (
            self.min_tokens < 0
            or self.max_tokens < self.min_tokens
            or self.min_scored_positions < 0
        ):
            raise ValueError("Invalid quality length/target bounds")
        if not 0 <= self.max_repeated_line_fraction <= 1:
            raise ValueError("Repetition limit must be a fraction")


def validate_quality(
    episode: Episode, encoded: Sequence[EncodedRecord], policy: QualityPolicy | None = None
) -> None:
    policy = policy or QualityPolicy()
    tokens = sum(record.tokens for record in encoded)
    if not policy.min_tokens <= tokens <= policy.max_tokens:
        raise QualityReject(
            f"quality:length:{tokens} outside [{policy.min_tokens},{policy.max_tokens}]"
        )
    positions = sum(record.P for record in encoded)
    if positions < policy.min_scored_positions:
        raise QualityReject(
            f"quality:scored_positions:{positions} below {policy.min_scored_positions}"
        )
    if policy.required_verified_success and not episode.supervision.verified_success:
        raise QualityReject("quality:requires_verified_success")
    if policy.required_executed_observation and not any(
        record.result_of and record.provenance == "executed" for record in episode.records
    ):
        raise QualityReject("quality:requires_actual_executed_observation")
    if policy.allowed_outcomes and episode.supervision.outcome not in policy.allowed_outcomes:
        raise QualityReject(f"quality:outcome:{episode.supervision.outcome}")
    if policy.max_repeated_line_fraction < 1:
        lines = [
            line.strip()
            for record in episode.records
            for line in record.payload.splitlines()
            if line.strip()
        ]
        repeated = sum(count - 1 for count in Counter(lines).values())
        if repeated / max(1, len(lines)) > policy.max_repeated_line_fraction:
            raise QualityReject("quality:excessive_repeated_lines")


def validate_row_filters(row: Mapping[str, Any], filters: list[dict[str, Any]]) -> None:
    """Only declared field comparisons are supported; no executable expressions."""
    import json

    for rule in filters:
        value: Any = row
        for key in rule["path"].split("."):
            if isinstance(value, str):
                try:
                    value = json.loads(value)
                except json.JSONDecodeError:
                    value = None
            value = value.get(key) if isinstance(value, Mapping) else None
        operation, expected = rule["op"], rule.get("value")
        if operation == "exists":
            accepted = value is not None
        elif operation == "nonempty":
            accepted = isinstance(value, (str, list, tuple, Mapping)) and len(value) > 0
        elif operation == "eq":
            accepted = value == expected
        elif operation == "in":
            accepted = value in expected
        elif operation in {"gte", "lte"}:
            accepted = isinstance(value, (int, float)) and not isinstance(value, bool)
            accepted = accepted and (value >= expected if operation == "gte" else value <= expected)
        else:
            raise ValueError(f"Unsupported source filter operation: {operation}")
        if not accepted:
            raise RowFilterReject(f"source_filter:{rule['path']}:{operation}")
