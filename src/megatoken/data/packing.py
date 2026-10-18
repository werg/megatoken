"""Causal training windows with explicit reconstruction and credit assignment."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from .schema import Codec, Dependency, EncodedRecord, Episode, encode_record


@dataclass(frozen=True, slots=True)
class WindowConfig:
    max_window_tokens: int = 8192
    max_decisions: int = 4
    respect_dependencies: bool = False
    max_dependency_tokens: int = 262144
    max_records: int | None = None

    def __post_init__(self) -> None:
        if min(self.max_window_tokens, self.max_decisions, self.max_dependency_tokens) < 1:
            raise ValueError("Window sizes and decision limits must be positive")
        if self.max_records is not None and self.max_records < 1:
            raise ValueError("max_records must be positive when supplied")


@dataclass(frozen=True, slots=True)
class TrainingWindow:
    episode: Episode
    prefix: tuple[EncodedRecord, ...]
    records: tuple[EncodedRecord, ...]
    start: int
    end: int
    index: int
    writer_reader_pairs: tuple[Dependency, ...] = ()
    token_overshoot: int = 0

    @property
    def episode_id(self) -> str:
        return self.episode.episode_id

    @property
    def reset_id(self) -> str:
        return self.episode.identity

    @property
    def stream(self) -> str:
        return self.episode.stream

    @property
    def source_id(self) -> str:
        return self.episode.lineage.source_id

    @property
    def D(self) -> int:
        return sum(record.tokens for record in self.records)

    @property
    def I(self) -> int:  # noqa: E743 - public v13 meter name, distinct from D and P.
        return self.D + sum(record.tokens for record in self.prefix)

    @property
    def P(self) -> int:
        return sum(record.P for record in self.records)

    @property
    def prefix_count(self) -> int:
        return len(self.prefix)

    @property
    def record_ingestions(self) -> int:
        return len(self.prefix) + len(self.records)

    @property
    def signature(self) -> tuple[tuple[str, bool, int, int, int, bool, bool], ...]:
        """Compatible event schedules can batch without altering any trajectory."""
        return tuple(
            (
                entry.record.role,
                entry.record.policy_mask,
                entry.positions,
                entry.record.part_index,
                entry.record.part_count,
                entry.record.action_id is not None,
                entry.record.result_of is not None,
            )
            for entry in self.records
        )


def iter_windows(
    episode: Episode, codec: Codec, config: WindowConfig | None = None
) -> Iterator[TrainingWindow]:
    """Visit each physical record once as D; earlier actual history is I-only.

    Windows never cross an episode or split a logical action. The complete earlier
    prefix must be reconstructed under current weights after every update. A writer
    only receives gradient from a later reader when both occur in ``records``.
    ``respect_dependencies`` extends selected windows to include their annotated
    readers; the explicit hard limit prevents accidental unbounded graphs.
    """
    config = config or WindowConfig()
    encoded = tuple(encode_record(record, codec) for record in episode.records)
    positions = {record.record_id: index for index, record in enumerate(episode.records)}
    dependencies = [
        (positions[dep.writer_id], positions[dep.reader_id], dep) for dep in episode.dependencies
    ]
    # Semantic prose/observation parts are independently schedulable. A physical
    # chunk boundary is still not permission to truncate a logical tool action.
    groups: list[tuple[int, int]] = []
    start = 0
    for index, record in enumerate(episode.records):
        if not record.action_id or record.part_index + 1 == record.part_count:
            groups.append((start, index + 1))
            start = index + 1
    group_index, window_index = 0, 0
    while group_index < len(groups):
        start = groups[group_index][0]
        end = start
        tokens = decisions = 0
        must_reach = start
        while group_index < len(groups):
            left, right = groups[group_index]
            group_tokens = sum(record.tokens for record in encoded[left:right])
            group_decisions = int(any(record.P for record in encoded[left:right]))
            if (
                end > start
                and end > must_reach
                and (
                    tokens + group_tokens > config.max_window_tokens
                    or decisions + group_decisions > config.max_decisions
                    or config.max_records is not None
                    and right - start > config.max_records
                )
            ):
                break
            tokens += group_tokens
            decisions += group_decisions
            end = right
            group_index += 1
            if config.respect_dependencies:
                for writer, reader, _ in dependencies:
                    if start <= writer < end:
                        must_reach = max(must_reach, reader)
                if tokens > config.max_dependency_tokens:
                    raise ValueError(
                        "A complete dependency/logical record exceeds max_dependency_tokens; choose a larger explicit graph budget"
                    )
        pairs = tuple(dep for writer, reader, dep in dependencies if start <= writer < reader < end)
        yield TrainingWindow(
            episode=episode,
            prefix=encoded[:start],
            records=encoded[start:end],
            start=start,
            end=end,
            index=window_index,
            writer_reader_pairs=pairs,
            token_overshoot=max(0, tokens - config.max_window_tokens),
        )
        window_index += 1


@dataclass
class WindowAudit:
    """Audit long-context requirements using observed windows, not source labels."""

    windows: int = 0
    scored_windows: int = 0
    writer_reader_windows: int = 0
    D: int = 0
    long_dependency_D: int = 0
    context_I: int = 0
    long_context_I: int = 0

    def observe(self, window: TrainingWindow, *, long_threshold: int = 131072) -> None:
        self.windows += 1
        self.D += window.D
        self.context_I += window.I
        if window.P:
            self.scored_windows += 1
            self.writer_reader_windows += bool(window.writer_reader_pairs)
        episode_tokens = sum(record.tokens for record in window.prefix + window.records)
        if episode_tokens >= long_threshold:
            self.long_context_I += window.I
            reader_ids = {record.record.record_id for record in window.records if record.P}
            if any(dep.reader_id in reader_ids for dep in window.episode.dependencies):
                self.long_dependency_D += window.D

    def assert_requirements(
        self,
        *,
        min_writer_reader_fraction: float = 0.0,
        min_long_dependency_D_fraction: float = 0.0,
        min_long_context_fraction: float = 0.0,
    ) -> None:
        if (
            min_writer_reader_fraction
            and self.writer_reader_windows
            < min_writer_reader_fraction * max(1, self.scored_windows)
        ):
            raise ValueError("Insufficient differentiable writer-to-reader windows")
        if (
            min_long_dependency_D_fraction
            and self.long_dependency_D < min_long_dependency_D_fraction * max(1, self.D)
        ):
            raise ValueError("Insufficient long-history D with a scored late dependency")
        if min_long_context_fraction and self.long_context_I < min_long_context_fraction * max(
            1, self.context_I
        ):
            raise ValueError("Insufficient long-history context exposure")
