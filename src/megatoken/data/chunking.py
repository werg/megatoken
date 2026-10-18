"""Chat framing, recursive content parsing, and lossless tree-aware planning.

The same entry point serves preparation and observed runtime inputs. A logical
event is tokenized once; physical chunks only slice that canonical token stream.
Serialized chat is framed before tokenization so latency boundaries are exact,
even when a codec could otherwise merge bytes from adjacent messages.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field, replace
from typing import TYPE_CHECKING, Any

from .parsing.config import CHUNKING_VERSION, ChunkingConfig
from .parsing.planner import TokenIndex, plan_chunks
from .parsing.types import Boundary, Diagnostic, ParseResult, Region

if TYPE_CHECKING:
    from .schema import Codec

__all__ = [
    "CHUNKING_VERSION",
    "ChunkingConfig",
    "ChunkingReport",
    "ChunkingStats",
    "SemanticChunk",
    "segment_bytes",
    "segment_with_report",
]
_HISTOGRAM_BOUNDS = (0, 32, 64, 96, 128, 192, 256, 384, 512, 768, 1024)


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
    context: tuple[str, ...] = ()
    languages: tuple[str, ...] = ()
    diagnostics: tuple[str, ...] = ()
    tree_distance: float = 0.0


@dataclass(frozen=True)
class ChunkingReport:
    chunks: tuple[SemanticChunk, ...]
    parse: ParseResult
    chat_template: str | None
    canonical_events: tuple[tuple[int, int], ...]
    planner_cost: float
    planner_candidates: int
    planner_edges: int

    def to_dict(self, *, include_payload: bool = False) -> dict[str, Any]:
        chunks = []
        for chunk in self.chunks:
            item = {
                "byte_start": chunk.byte_start,
                "byte_end": chunk.byte_end,
                "token_start": chunk.token_start,
                "token_end": chunk.token_end,
                "tokens": len(chunk.token_ids),
                "boundary": chunk.boundary,
                "forced": chunk.forced,
                "context": list(chunk.context),
                "languages": list(chunk.languages),
                "diagnostics": list(chunk.diagnostics),
                "tree_distance": chunk.tree_distance,
            }
            if include_payload:
                item["payload_hex"] = chunk.payload.hex()
                item["token_ids"] = list(chunk.token_ids)
            chunks.append(item)
        return {
            "version": CHUNKING_VERSION,
            "chat_template": self.chat_template,
            "canonical_events": self.canonical_events,
            "chunks": chunks,
            "parsers": sorted(self.parse.parsers),
            "languages": sorted(self.parse.languages),
            "diagnostics": [asdict(item) for item in self.parse.diagnostics],
            "regions": [asdict(item) for item in self.parse.regions],
            "boundaries": [asdict(item) for item in self.parse.boundaries],
            "planner": {
                "cost": self.planner_cost,
                "candidates": self.planner_candidates,
                "evaluated_edges": self.planner_edges,
            },
        }


def _materialize(index: TokenIndex, parsed: ParseResult, config: ChunkingConfig):
    plan = plan_chunks(index, parsed, config)
    regions = sorted(parsed.regions, key=lambda r: (r.start, -r.end, r.depth, r.kind))
    diagnostics = sorted(parsed.diagnostics, key=lambda d: (d.start, d.end, d.code))
    region_cursor = diagnostic_cursor = 0
    active_regions: list[Region] = []
    active_diagnostics: list[Diagnostic] = []
    chunks = []
    for part in plan.chunks:
        start, end = index.offsets[part.start], index.offsets[part.end]
        while region_cursor < len(regions) and regions[region_cursor].start < end:
            active_regions.append(regions[region_cursor])
            region_cursor += 1
        active_regions = [item for item in active_regions if item.end > start]
        while diagnostic_cursor < len(diagnostics) and diagnostics[diagnostic_cursor].start <= end:
            active_diagnostics.append(diagnostics[diagnostic_cursor])
            diagnostic_cursor += 1
        active_diagnostics = [item for item in active_diagnostics if item.end >= start]
        context = tuple(
            dict.fromkeys(
                f"{item.language or 'content'}:{item.kind}"
                for item in sorted(active_regions, key=lambda r: (r.depth, r.start, -r.end))
            )
        )
        languages = tuple(sorted({item.language for item in active_regions if item.language}))
        codes = tuple(sorted({item.code for item in active_diagnostics}))
        chunks.append(
            SemanticChunk(
                index.payload[start:end],
                index.ids[part.start : part.end],
                start,
                end,
                part.start,
                part.end,
                part.boundary,
                part.forced,
                context,
                languages,
                codes,
                part.tree_distance,
            )
        )
    return tuple(chunks), plan


def segment_with_report(
    payload: bytes,
    codec: Codec,
    config: ChunkingConfig | None = None,
    *,
    content_kind: str = "auto",
    language: str | None = None,
    filename: str | None = None,
    mime_type: str | None = None,
    locale: str | None = None,
    chat_template: str | None = None,
) -> ChunkingReport:
    """Parse arbitrary input and expose the evidence behind every physical cut.

    Structured adapters pass chat_template='none' because their message and tool
    boundaries are authoritative. Raw templates are parsed only at this outermost
    level, never recursively inside code, strings, or message bodies.
    """
    from .parsing.chat import parse_chat
    from .parsing.engine import parse_content

    config = config or ChunkingConfig()
    if not isinstance(payload, bytes):
        raise TypeError("Semantic chunking requires immutable bytes")
    for name, value in (
        ("language", language),
        ("filename", filename),
        ("mime_type", mime_type),
        ("locale", locale),
        ("chat_template", chat_template),
    ):
        if value is not None and not isinstance(value, str):
            raise TypeError(f"{name} must be a string or null")
    if content_kind not in {
        "auto",
        "text",
        "prose",
        "markdown",
        "code",
        "json",
        "html",
        "xml",
        "binary",
    }:
        raise ValueError(f"Unsupported semantic content kind: {content_kind}")
    selected_template = chat_template if chat_template is not None else config.chat_template
    if content_kind == "binary":
        selected_template = "none"
    chat = parse_chat(payload, selected_template, max_frames=config.limits.max_regions)
    parsed_all = ParseResult(diagnostics=list(chat.diagnostics))
    if chat.detected:
        chat.validate(len(payload))
        frames = [(f.start, f.end, f.header_end, f.body_end, f.role) for f in chat.frames]
        parsed_all.parsers.add(f"chat:{chat.template}")
    else:
        frames = [(0, len(payload), 0, len(payload), None)]
    chunks: list[SemanticChunk] = []
    events: list[tuple[int, int]] = []
    token_base = candidates = edges = 0
    objective = 0.0
    for frame_start, frame_end, body_start, body_end, role in frames:
        source = payload[frame_start:frame_end]
        index = TokenIndex.encode(source, codec)
        local_body = body_start - frame_start

        def needs_refinement(
            start: int, end: int, *, token_index=index, body_offset=local_body
        ) -> bool:
            return (
                token_index.count(body_offset + start, body_offset + end) > config.soft_max_tokens
                or end - start > config.max_bytes
            )

        parsed = ParseResult()
        if content_kind != "binary":
            inner = parse_content(
                payload[body_start:body_end],
                content_kind=content_kind,
                language=language,
                filename=filename,
                mime_type=mime_type,
                locale=locale,
                limits=config.limits,
                backend=config.parser_backend,
                use_icu=config.use_icu,
                needs_refinement=needs_refinement,
            )
            inner.validate(body_end - body_start)
            parsed.extend(inner, offset=local_body, depth=1 if chat.detected else 0)
        else:
            parsed.parsers.add("binary")
        if chat.detected:
            parsed.regions.append(
                Region(0, len(source), "chat_message", role, 0, False, f"chat:{chat.template}")
            )
            for start, end in ((0, local_body), (body_end - frame_start, len(source))):
                if end > start:
                    parsed.regions.append(
                        Region(
                            start, end, "template_marker", None, 1, True, f"chat:{chat.template}"
                        )
                    )
            parsed.boundaries.append(
                Boundary(len(source), "chat", source=f"chat:{chat.template}", mandatory=True)
            )
        parsed.boundaries.append(Boundary(len(source), "end"))
        parsed.diagnostics.extend(
            replace(
                d, start=max(0, d.start - frame_start), end=min(len(source), d.end - frame_start)
            )
            for d in chat.diagnostics
            if d.start <= frame_end and d.end >= frame_start
        )
        parsed.validate(len(source))
        physical, plan = _materialize(index, parsed, config)
        chunks.extend(
            replace(
                part,
                byte_start=part.byte_start + frame_start,
                byte_end=part.byte_end + frame_start,
                token_start=part.token_start + token_base,
                token_end=part.token_end + token_base,
                boundary="chat"
                if chat.detected and part.byte_end == len(source)
                else part.boundary,
            )
            for part in physical
        )
        parsed_all.extend(parsed, offset=frame_start)
        events.append((frame_start, frame_end))
        token_base += len(index.ids)
        candidates += plan.candidate_count
        edges += plan.evaluated_edges
        objective += plan.cost
    parsed_all.diagnostics[:] = list(dict.fromkeys(parsed_all.diagnostics))
    return ChunkingReport(
        tuple(chunks),
        parsed_all,
        chat.template if chat.detected else None,
        tuple(events),
        objective,
        candidates,
        edges,
    )


def segment_bytes(
    payload: bytes,
    codec: Codec,
    config: ChunkingConfig | None = None,
    *,
    content_kind: str = "auto",
    language: str | None = None,
    filename: str | None = None,
    mime_type: str | None = None,
    locale: str | None = None,
    chat_template: str | None = None,
) -> tuple[SemanticChunk, ...]:
    return segment_with_report(
        payload,
        codec,
        config,
        content_kind=content_kind,
        language=language,
        filename=filename,
        mime_type=mime_type,
        locale=locale,
        chat_template=chat_template,
    ).chunks


@dataclass
class ChunkingStats:
    """Bounded counters over accepted physical records, not training exposures."""

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
    languages: Counter = field(default_factory=Counter)
    diagnostics: Counter = field(default_factory=Counter)

    def observe(
        self,
        *,
        positions: int,
        reference_tokens: int,
        byte_count: int,
        boundary: str | None,
        forced: bool,
        soft_max: int,
        languages: tuple[str, ...] = (),
        diagnostics: tuple[str, ...] = (),
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
        self.languages.update(set(languages))
        self.diagnostics.update(set(diagnostics))
        bucket = next((str(bound) for bound in _HISTOGRAM_BOUNDS if positions <= bound), ">1024")
        self.histogram[bucket] += 1

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        for key in ("boundaries", "languages", "diagnostics"):
            result[key] = dict(getattr(self, key))
        result["mean_positions"] = self.positions / max(1, self.records)
        result["histogram"] = {
            str(bound): self.histogram[str(bound)] for bound in _HISTOGRAM_BOUNDS
        } | {">1024": self.histogram[">1024"]}
        result["histogram_kind"] = "disjoint buckets, inclusive upper bounds"
        return result
