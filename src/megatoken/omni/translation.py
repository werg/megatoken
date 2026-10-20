"""Causal source-prefix translation clock and verified wait/latency accounting.

Speech segments, output text and Pocket speech are still ordinary Megatoken
events. This helper enforces physical availability separately from logical
meridian indices; it neither predicts nor schedules hidden semantic tokens.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SourceChunk:
    chunk_id: str
    start_seconds: float
    end_seconds: float
    available_at: float
    sample_rate: int
    sample_count: int
    language: str
    def __post_init__(self):
        if not self.chunk_id or not self.language or self.sample_rate <= 0:
            raise ValueError("Source chunk requires stable identity/language/clock")
        if not 0 <= self.start_seconds < self.end_seconds <= self.available_at:
            raise ValueError("Chunk crosses a future source decision boundary")
        if self.sample_count <= 0:
            raise ValueError("Available source chunk contains no samples")
        if abs(self.sample_count / self.sample_rate -
               (self.end_seconds - self.start_seconds)) > 1 / self.sample_rate:
            raise ValueError("Source chunk sample count differs from its actual clock")


@dataclass(frozen=True)
class TranslationCommit:
    decision_seconds: float
    source_chunk_ids: tuple[str, ...]
    target_text_id: str
    target_speech_id: str
    target_speech_start: float
    target_speech_end: float
    language: str
    delivery_mode: str = "simultaneous"
    def __post_init__(self):
        if not self.target_text_id or not self.target_speech_id or not self.language:
            raise ValueError("Translation must bind both text and speech output peers")
        if self.delivery_mode not in {"simultaneous", "dubbing"}:
            raise ValueError("Unknown target timing semantics")
        if not 0 <= self.decision_seconds <= self.target_speech_start <= self.target_speech_end:
            raise ValueError("Target speech has an impossible acquisition/delivery clock")


@dataclass(frozen=True)
class WaitRecord:
    start: float
    end: float
    reason: str
    def __post_init__(self):
        if not 0 <= self.start <= self.end or not self.reason:
            raise ValueError("Waiting requires real elapsed availability time")


class TranslationClock:
    def __init__(self):
        self.sources: list[SourceChunk] = []
        self.translations: list[TranslationCommit] = []
        self.waits: list[WaitRecord] = []

    def append_source(self, chunk: SourceChunk):
        if any(existing.chunk_id == chunk.chunk_id for existing in self.sources):
            raise ValueError("Source clip identities cannot be reused")
        if self.sources:
            last = self.sources[-1]
            if chunk.start_seconds < last.end_seconds or chunk.available_at < last.available_at:
                raise ValueError("Source windows must arrive monotonically, without overlap")
        self.sources.append(chunk)

    def commit(self, result: TranslationCommit):
        if not result.source_chunk_ids or len(set(result.source_chunk_ids)) != len(result.source_chunk_ids):
            raise ValueError("Translation must consume distinct available source chunks")
        by_id = {source.chunk_id: source for source in self.sources}
        if any(sid not in by_id for sid in result.source_chunk_ids):
            raise ValueError("Translation depends on absent or future source audio")
        if any(by_id[sid].available_at > result.decision_seconds for sid in result.source_chunk_ids):
            raise ValueError("Prefix-limited translation cannot read later source chunks")
        if self.translations and result.decision_seconds < self.translations[-1].decision_seconds:
            raise ValueError("Translation decisions must follow actual causal time")
        used = {sid for item in self.translations for sid in item.source_chunk_ids}
        if any(sid in used for sid in result.source_chunk_ids):
            raise ValueError("Source chunk already translated; record an explicit revision instead")
        # Translated speech's duration is its OWN clock. No source duration,
        # word-order or sample-rate copy is inferred from alignment metadata.
        self.translations.append(result)

    def wait(self, start: float, end: float, reason: str):
        if self.waits and start < self.waits[-1].end:
            raise ValueError("Wait intervals overlap")
        self.waits.append(WaitRecord(start, end, reason))

    def backlog(self, now: float) -> tuple[str, ...]:
        consumed = {cid for output in self.translations for cid in output.source_chunk_ids}
        return tuple(
            src.chunk_id for src in self.sources
            if src.available_at <= now and src.chunk_id not in consumed
        )

    def latency(self) -> dict[str, float]:
        lookup = {chunk.chunk_id: chunk for chunk in self.sources}
        return {
            str(index): result.target_speech_start - max(
                lookup[sid].end_seconds for sid in result.source_chunk_ids
            )
            for index, result in enumerate(self.translations)
        }
