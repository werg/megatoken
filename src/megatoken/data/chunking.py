"""Deterministic semantic records over one byte-exact reference tokenization.

Parsers identify offsets only. They never render, normalize, strip, overlap, or
re-tokenize source text. The same entry point is used for data and observations.
"""

from __future__ import annotations

import ast
import bisect
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field, replace
from typing import TYPE_CHECKING, Any

from megatoken.io import content_hash

if TYPE_CHECKING:
    from .schema import Codec


CHUNKING_VERSION = "semantic-v1"
_LINES = re.compile(rb"[^\r\n]*(?:\r\n|\r|\n)|[^\r\n]+\Z")
_FENCE = re.compile(rb" {0,3}(`{3,}|~{3,})(.*)")
_LIST = re.compile(rb"([ \t]*)(?:[-+*]|[0-9]{1,9}[.)])[ \t]+")
_SENTENCE = re.compile(rb"[.!?](?:[\"')\]]*)(?=[ \t\r\n]|$)|\xe3\x80[\x82\x81]|\xef\xbc[\x81\x9f]")
_PRIORITY = {
    "token": 0,
    "whitespace": 1,
    "line": 2,
    "sentence": 3,
    "statement": 4,
    "paragraph": 5,
    "list_item": 5,
    "function": 6,
    "code_block": 7,
    "json": 7,
    "end": 5,
}
_HISTOGRAM_BOUNDS = (0, 32, 64, 96, 128, 192, 256, 384, 512, 768, 1024)


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    target_tokens: int = 256
    soft_max_tokens: int = 512
    hard_max_tokens: int = 1024
    max_bytes: int = 16384
    version: str = CHUNKING_VERSION

    def __post_init__(self) -> None:
        if any(
            type(value) is not int
            for value in (
                self.target_tokens,
                self.soft_max_tokens,
                self.hard_max_tokens,
                self.max_bytes,
            )
        ):
            raise ValueError("Chunk sizes must be integer counts")
        if not 1 <= self.target_tokens <= self.soft_max_tokens <= self.hard_max_tokens:
            raise ValueError("Chunk sizes must satisfy 1 <= target <= soft <= hard")
        if self.max_bytes < 1:
            raise ValueError("Chunk max_bytes must be positive")
        if self.version != CHUNKING_VERSION:
            raise ValueError(f"Unsupported chunking version: {self.version}")

    def with_hard_limit(self, limit: int, *, max_bytes: int | None = None) -> ChunkingConfig:
        """Respect a smaller stage capacity without silently raising this policy."""
        if type(limit) is not int or limit < 1:
            raise ValueError("The stage hard limit must be a positive integer")
        hard = min(self.hard_max_tokens, limit)
        return replace(
            self,
            target_tokens=min(self.target_tokens, hard),
            soft_max_tokens=min(self.soft_max_tokens, hard),
            hard_max_tokens=hard,
            max_bytes=self.max_bytes if max_bytes is None else min(self.max_bytes, max_bytes),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def fingerprint(self) -> str:
        # Python syntax support can change between runtimes. Persist its identity
        # even when a particular batch happens to contain no Python source.
        return content_hash({"config": self.to_dict(), "python_ast": list(sys.version_info[:2])})

    def identity(self, codec: Codec) -> dict[str, Any]:
        value = {
            "config": self.to_dict(),
            "config_fingerprint": self.fingerprint,
            "codec_fingerprint": codec.fingerprint,
            "python_ast": list(sys.version_info[:2]),
        }
        return value | {"fingerprint": content_hash(value)}

    @classmethod
    def from_identity(cls, identity: dict[str, Any], codec: Codec) -> ChunkingConfig:
        """Restore only an exact matching chunker, codec, and parser runtime."""
        try:
            config = cls(**identity["config"])
            if identity != config.identity(codec):
                raise ValueError("identity differs")
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "Incompatible chunker, codec, or Python parser runtime; regenerate/replay with the declared preparation policy"
            ) from error
        return config


@dataclass(frozen=True, slots=True)
class SemanticChunk:
    payload: bytes
    token_ids: tuple[int, ...]
    byte_start: int
    byte_end: int
    token_start: int
    token_end: int
    boundary: str
    forced: bool = False


@dataclass(frozen=True, slots=True)
class _Unit:
    start: int
    end: int
    kind: str


def _line_offsets(payload: bytes) -> list[tuple[int, int, bytes]]:
    return [
        (match.start(), match.end(), match[0].rstrip(b"\r\n")) for match in _LINES.finditer(payload)
    ]


def _python_units(payload: bytes, offset: int = 0) -> list[_Unit]:
    """Use AST source coordinates, never ast.unparse or a reformatted source."""
    try:
        tree = ast.parse(payload.decode("utf-8"))
    except (SyntaxError, UnicodeError, ValueError, RecursionError):
        return []
    lines = _line_offsets(payload)
    units = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.stmt) or node.end_lineno is None:
            continue
        first = node.lineno
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            first = min([first, *(item.lineno for item in node.decorator_list)])
            kind = "function"
        else:
            kind = "statement"
        units.append(
            _Unit(offset + lines[first - 1][0], offset + lines[node.end_lineno - 1][1], kind)
        )
    return units


def _code_units(payload: bytes, offset: int = 0, language: str | None = None) -> list[_Unit]:
    if language in {"py", "python", "python3"}:
        parsed = _python_units(payload, offset)
        if parsed:
            return parsed
    # A conservative C/JS-style lexical scanner provides block/statement cuts
    # without mistaking braces inside strings or comments for syntax. Unknown
    # languages still have line/paragraph fallbacks; this is not a compiler.
    units: list[_Unit] = []
    braces: list[int] = []
    quote = 0
    escaped = False
    comment = ""
    parentheses = 0
    line_start = statement_start = 0
    index = 0
    while index < len(payload):
        value = payload[index]
        pair = payload[index : index + 2]
        if value in (10, 13):
            line_start = index + 1
            if comment == "line":
                comment = ""
        if comment == "block":
            if pair == b"*/":
                comment = ""
                index += 2
                continue
        elif comment == "line":
            pass
        elif quote:
            if escaped:
                escaped = False
            elif value == 92:
                escaped = True
            elif value == quote:
                quote = 0
        elif pair in (b"//", b"/*"):
            comment = "line" if pair == b"//" else "block"
            index += 2
            continue
        elif value in (34, 39, 96):
            quote = value
        elif value in (40, 91):
            parentheses += 1
        elif value in (41, 93):
            parentheses = max(0, parentheses - 1)
        elif value == 123:
            braces.append(line_start)
            statement_start = index + 1
        elif value == 125 and braces:
            start = braces.pop()
            end = index + 1
            while end < len(payload) and payload[end] in b"; \t\r\n":
                end += 1
            kind = "function" if not braces else "statement"
            units.append(_Unit(offset + start, offset + end, kind))
            statement_start = index + 1
        elif value == 59 and not parentheses:
            units.append(_Unit(offset + statement_start, offset + index + 1, "statement"))
            statement_start = index + 1
        index += 1
    return units


def _boundaries(
    payload: bytes, content_kind: str, language: str | None
) -> tuple[dict[int, str], list[_Unit]]:
    boundaries: dict[int, str] = {len(payload): "end"}
    units: list[_Unit] = []

    def add(end: int, kind: str) -> None:
        if end > 0 and _PRIORITY[kind] >= _PRIORITY.get(boundaries.get(end, "token"), 0):
            boundaries[end] = kind

    if content_kind == "binary":
        return boundaries, units
    lines = _line_offsets(payload)
    paragraph_start = 0
    fence: tuple[int, bytes, int, bytes] | None = None
    list_start: int | None = None
    list_indent = 0
    for start, end, body in lines:
        add(end, "line")
        marker = _FENCE.fullmatch(body) if content_kind != "code" else None
        if fence is not None:
            opening, delimiter, body_start, tag = fence
            if (
                marker
                and marker[1][:1] == delimiter[:1]
                and len(marker[1]) >= len(delimiter)
                and not marker[2].strip()
            ):
                units.append(_Unit(opening, end, "code_block"))
                add(opening, "paragraph")
                add(end, "code_block")
                units.extend(
                    _code_units(
                        payload[body_start:start], body_start, tag.decode("ascii", errors="ignore")
                    )
                )
                fence = None
                paragraph_start = end
            continue
        if marker:
            if list_start is not None:
                units.append(_Unit(list_start, start, "list_item"))
                add(start, "list_item")
                list_start = None
            fence = (start, marker[1], end, marker[2].strip().split(b" ", 1)[0].lower())
            continue
        if not body.strip(b" \t"):
            add(end, "paragraph")
            if paragraph_start < start:
                units.append(_Unit(paragraph_start, end, "paragraph"))
            paragraph_start = end
            continue
        item = _LIST.match(body) if content_kind != "code" else None
        if item:
            indent = len(item[1].expandtabs(4))
            if list_start is not None and indent <= list_indent:
                units.append(_Unit(list_start, start, "list_item"))
                add(start, "list_item")
                list_start = None
            if list_start is None:
                list_start, list_indent = start, indent
        elif list_start is not None and len(body) - len(body.lstrip(b" \t")) <= list_indent:
            units.append(_Unit(list_start, start, "list_item"))
            add(start, "list_item")
            list_start = None
    if fence is not None:
        opening, _, body_start, tag = fence
        units.append(_Unit(opening, len(payload), "code_block"))
        add(opening, "paragraph")
        add(len(payload), "code_block")
        units.extend(
            _code_units(payload[body_start:], body_start, tag.decode("ascii", errors="ignore"))
        )
    if list_start is not None:
        units.append(_Unit(list_start, len(payload), "list_item"))
        add(len(payload), "list_item")
    if content_kind == "code":
        units.extend(_code_units(payload, language=language))
    if content_kind == "json":
        # Structured actions remain complete while they fit their hard capacity.
        # Oversized legacy actions can still become ordered physical parts.
        units.append(_Unit(0, len(payload), "json"))
        add(len(payload), "json")
    for unit in units:
        if unit.kind in {"function", "statement"}:
            add(unit.start, unit.kind)
            add(unit.end, unit.kind)
    for match in _SENTENCE.finditer(payload):
        add(match.end(), "sentence")
    for match in re.finditer(rb"[ \t]+", payload):
        # Both sides are candidates: a BPE may absorb leading whitespace into
        # the next word. No source token is split just to hit a text offset.
        add(match.start(), "whitespace")
        add(match.end(), "whitespace")
    return boundaries, units


def segment_bytes(
    payload: bytes,
    codec: Codec,
    config: ChunkingConfig | None = None,
    *,
    content_kind: str = "auto",
    language: str | None = None,
) -> tuple[SemanticChunk, ...]:
    """Partition one logical event, preserving bytes and original reference IDs.

    Paragraph/list/code units fitting the soft cap remain intact. Larger units
    use statement, sentence, line, then whitespace boundaries near the target.
    An indivisible code/list/JSON unit may exceed the soft cap up to the hard
    cap. Last-resort token cuts are reported, including binary input.
    """
    config = config or ChunkingConfig()
    if not isinstance(payload, bytes):
        raise TypeError("Semantic chunking requires immutable bytes")
    if content_kind not in {"auto", "text", "markdown", "code", "json", "binary"}:
        raise ValueError(f"Unsupported semantic content kind: {content_kind}")
    if language is not None and not isinstance(language, str):
        raise TypeError("The source language must be a string or null")
    language = language.casefold() if language else None
    ids = tuple(codec.encode(payload))
    pieces = [codec.decode([token]) for token in ids]
    if b"".join(pieces) != payload or any(not piece for piece in pieces):
        raise ValueError("The raw codec must have nonempty, byte-exact compositional tokens")
    offsets = [0]
    for piece in pieces:
        offsets.append(offsets[-1] + len(piece))
    if not ids:
        return (SemanticChunk(b"", (), 0, 0, 0, 0, "end"),)
    valid_utf8 = True
    try:
        payload.decode("utf-8")
    except UnicodeError:
        valid_utf8 = False
        content_kind = "binary"
    raw_boundaries, units = _boundaries(payload, content_kind, language)
    boundary_ids: dict[int, str] = {}
    for offset, kind in raw_boundaries.items():
        index = bisect.bisect_left(offsets, offset)
        if offsets[index] != offset:
            # Moving within a separator run preserves its block boundary. A
            # token containing real text on both sides makes this cut ineligible.
            if payload[offsets[index - 1] : offsets[index]].strip(b" \t\r\n"):
                continue
        if index and _PRIORITY[kind] >= _PRIORITY.get(boundary_ids.get(index, "token"), 0):
            boundary_ids[index] = kind
    protected: list[tuple[int, int]] = []
    atomic: list[tuple[int, int, str]] = []
    for unit in units:
        left = bisect.bisect_left(offsets, unit.start)
        right = bisect.bisect_left(offsets, unit.end)
        if unit.kind in {"function", "code_block", "list_item", "json"}:
            if right - left <= config.soft_max_tokens and unit.end - unit.start <= config.max_bytes:
                protected.append((left, right))
            atomic.append((left, right, unit.kind))
    # Merge protected intervals once; boundary selection must not compare every
    # candidate with every function/list in a long source file.
    merged: list[list[int]] = []
    for left, right in sorted(protected):
        if merged and left < merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], right)
        else:
            merged.append([left, right])
    protected_starts = [left for left, _ in merged]

    def blocked(index: int) -> bool:
        position = bisect.bisect_left(protected_starts, index) - 1
        return position >= 0 and index < merged[position][1]

    candidates = sorted(boundary_ids)
    atomic.sort()
    atomic_index = 0
    active_units: list[tuple[int, int, str]] = []
    chunks: list[SemanticChunk] = []
    start = 0
    while start < len(ids):
        while atomic_index < len(atomic) and atomic[atomic_index][0] <= start:
            active_units.append(atomic[atomic_index])
            atomic_index += 1
        active_units = [unit for unit in active_units if unit[1] > start]
        hard = min(
            len(ids),
            start + config.hard_max_tokens,
            bisect.bisect_right(offsets, offsets[start] + config.max_bytes) - 1,
        )
        if hard <= start:
            raise ValueError("One indivisible codec token exceeds max_record_bytes")
        soft = min(hard, start + config.soft_max_tokens)
        target = min(soft, start + config.target_tokens)

        def safe(index: int) -> bool:
            byte = offsets[index]
            return payload[max(0, byte - 1) : byte + 1] != b"\r\n" and (
                not valid_utf8 or byte == len(payload) or payload[byte] & 0xC0 != 0x80
            )

        choices = [
            index
            for index in candidates[
                bisect.bisect_right(candidates, start) : bisect.bisect_right(candidates, soft)
            ]
            if safe(index) and not blocked(index)
        ]
        # Complete JSON is one action, even when larger than the prose soft cap.
        whole_json = next(
            (
                right
                for left, right, kind in active_units
                if kind == "json" and left == start and right <= hard
            ),
            None,
        )
        if whole_json is not None:
            end, kind = whole_json, "json"
        elif choices:
            primary = [index for index in choices if _PRIORITY[boundary_ids[index]] >= 5]
            semantic = [index for index in choices if _PRIORITY[boundary_ids[index]] >= 3]
            selected = primary or semantic or choices
            best_priority = max(_PRIORITY[boundary_ids[index]] for index in selected)
            if not primary:
                selected = [
                    index for index in selected if _PRIORITY[boundary_ids[index]] == best_priority
                ]
            end = min(selected, key=lambda index: (abs(index - target), -index))
            kind = boundary_ids[end]
        else:
            complete = [
                (right, kind)
                for left, right, kind in active_units
                if left <= start < right <= hard and safe(right)
            ]
            if complete:
                end, kind = min(complete, key=lambda pair: pair[0])
            else:
                end, kind = target, "token"
                while end > start and not safe(end):
                    end -= 1
                # Tiny explicit byte/token limits can be smaller than one UTF-8
                # scalar. First expand up to the hard cap to finish the scalar
                # or CRLF when possible; otherwise preserve bytes and progress.
                if end == start:
                    end = next(
                        (index for index in range(target + 1, hard + 1) if safe(index)), target
                    )
        if end == len(ids):
            kind = "end"
        chunks.append(
            SemanticChunk(
                payload[offsets[start] : offsets[end]],
                ids[start:end],
                offsets[start],
                offsets[end],
                start,
                end,
                kind,
                kind == "token",
            )
        )
        start = end
    return tuple(chunks)


@dataclass
class ChunkingStats:
    """Bounded-memory counters over accepted physical records, not exposures."""

    records: int = 0
    bytes: int = 0
    reference_tokens: int = 0
    positions: int = 0
    max_positions: int = 0
    over_soft_max: int = 0
    forced_splits: int = 0
    unversioned_records: int = 0
    boundaries: Counter = field(default_factory=Counter)
    histogram: Counter = field(default_factory=Counter)

    def observe(
        self,
        *,
        positions: int,
        reference_tokens: int,
        byte_count: int,
        boundary: str | None,
        forced: bool,
        soft_max: int,
    ) -> None:
        self.records += 1
        self.bytes += byte_count
        self.reference_tokens += reference_tokens
        self.positions += positions
        self.max_positions = max(self.max_positions, positions)
        self.over_soft_max += positions > soft_max
        self.forced_splits += forced
        self.unversioned_records += boundary is None
        self.boundaries[boundary or "native_preserved"] += 1
        bucket = next((str(bound) for bound in _HISTOGRAM_BOUNDS if positions <= bound), ">1024")
        self.histogram[bucket] += 1

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["boundaries"] = dict(self.boundaries)
        result["mean_positions"] = self.positions / max(1, self.records)
        result["histogram"] = {
            str(bound): self.histogram[str(bound)] for bound in _HISTOGRAM_BOUNDS
        } | {">1024": self.histogram[">1024"]}
        result["histogram_kind"] = "disjoint buckets, inclusive upper bounds"
        return result
