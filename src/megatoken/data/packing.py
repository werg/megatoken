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
    def G(self) -> int:
        """Actual physical megatokens, independently of lexical token count."""
        return len(self.records)

    @property
    def Q(self) -> int:
        """Whole supervised groups and verified outcome-only selections."""
        import json

        selected_ids = {r.record.record_id for r in self.records if r.record.policy_mask}
        selected = (
            sum(
                any(e.record_id in selected_ids for e in group.events)
                for group in self.episode.meridians
            )
            if self.episode.meridians
            else len(selected_ids)
        )
        annotations = json.loads(self.episode.supervision.annotations_json)
        for entry in annotations.get("native_action_tapes", ()):
            cursor = entry["record_cursor"]
            if entry.get("outcome_credit") is None or not (
                self.start <= cursor < self.end or self.start == self.end == cursor
            ):
                continue
            bindings = entry.get("record_ids")
            if bindings is None:
                bindings = (
                    [self.episode.records[cursor].record_id]
                    if cursor < len(self.episode.records)
                    else []
                )
            if not selected_ids.intersection(bindings):
                selected += 1
        return selected


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
    # Whole meridians must not be split across causal optimizer windows.
    # Multichunk tool records may span several meridians, so merge the
    # corresponding window groups while retaining individual C transitions.
    groups: list[tuple[int, int]] = []
    start = 0
    boundaries = (
        [positions[meridian.events[-1].record_id] + 1 for meridian in episode.meridians]
        if episode.meridians
        else list(range(1, len(episode.records) + 1))
    )
    for end in boundaries:
        last = episode.records[end - 1]
        if last.action_id and last.part_index + 1 != last.part_count:
            continue
        groups.append((start, end))
        start = end
    if start != len(episode.records):
        raise ValueError("Window boundary would truncate a logical action or meridian")
    group_index, window_index = 0, 0
    while group_index < len(groups):
        start = groups[group_index][0]
        end = start
        tokens = decisions = 0
        must_reach = start
        while group_index < len(groups):
            left, right = groups[group_index]
            group_tokens = sum(record.tokens for record in encoded[left:right])
            group_decisions = int(
                any(
                    record.P
                    or (
                        record.record.policy_mask
                        and record.record.media_type
                        == "application/vnd.megatoken.media-reference+json"
                    )
                    for record in encoded[left:right]
                )
            )
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
        )
        window_index += 1
    # A failed or empty final selection can have verified policy credit but no
    # physical output. Give it an explicit transaction instead of losing it at
    # the final record cursor (or charging the previous payload twice).
    import json

    annotations = json.loads(episode.supervision.annotations_json)
    if any(
        entry.get("outcome_credit") is not None and entry.get("record_cursor") == len(encoded)
        for entry in annotations.get("native_action_tapes", ())
    ):
        yield TrainingWindow(
            episode=episode,
            prefix=encoded,
            records=(),
            start=len(encoded),
            end=len(encoded),
            index=window_index,
        )
