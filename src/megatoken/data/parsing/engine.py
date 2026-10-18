"""Recursive chat-body/content parsing with shared budgets and exact source maps.

Chat framing is applied by the caller before this engine. This layer recursively
routes document containers, native syntax, embedded languages and Unicode prose.
The source is immutable: even disjoint analysis views point back into the original
bytes rather than materialising decoded string literals or rendered markup.
"""

from __future__ import annotations

import bisect
import locale as system_locale
import re
from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import cached_property

from .documents import (
    DOCUMENT_LANGUAGES,
    MARKUP_LANGUAGES,
    scan_document_containers,
    scan_markdown,
    scan_markup_fallback,
)
from .prose import analyze_prose
from .registry import bundled_languages, canonical_language, parser_provider_specs, resolve_language
from .syntax import Injection, Scan, scan_generic, scan_syntax
from .types import Diagnostic, ParseLimits, ParseResult, Region

ENGINE_VERSION = "recursive-content-v1"
_DEFAULT_LIMITS = ParseLimits()
_MARKDOWN_SIGNATURE = re.compile(
    rb"(?m)^[ \t]{0,3}(?:`{3,}|~{3,}|#{1,6}[ \t]+|>[ \t]+|[-+*][ \t]+|[0-9]{1,9}[.)][ \t]+)"
)
_MARKUP_SIGNATURE = re.compile(
    rb"(?i)<(?:!\[CDATA\[|!--|/?(?:html|div|p|span|a|section|article|script|style|body|head|main|h[1-6]|pre|code|ul|ol|li|table)(?:[ \t\r\n/>]))"
)


@dataclass(frozen=True, slots=True)
class _Segment:
    offset: int
    start: int
    end: int


@dataclass(frozen=True)
class SourceView:
    """A byte view with a piecewise identity map into the original document."""

    data: bytes
    segments: tuple[_Segment, ...]

    @classmethod
    def original(cls, payload: bytes) -> SourceView:
        return cls(payload, (_Segment(0, 0, len(payload)),))

    @cached_property
    def offsets(self) -> tuple[int, ...]:
        return tuple(segment.offset for segment in self.segments)

    def position(self, offset: int, *, end: bool = False) -> int:
        if not self.segments:
            return 0
        if not 0 <= offset <= len(self.data):
            raise ValueError("Analysis offset outside source view")
        index = (
            bisect.bisect_left(self.offsets, offset)
            if end
            else bisect.bisect_right(self.offsets, offset)
        ) - 1
        index = max(0, index)
        segment = self.segments[index]
        return min(segment.end, segment.start + offset - segment.offset)

    def subview(self, ranges: tuple[tuple[int, int], ...]) -> SourceView:
        parts, segments, total, previous_end = [], [], 0, -1
        for start, end in ranges:
            if not 0 <= start < end <= len(self.data) or start < previous_end:
                raise ValueError(
                    "Nested source ranges must be sorted, disjoint, and inside their parent"
                )
            previous_end = end
            first = max(0, bisect.bisect_right(self.offsets, start) - 1)
            for segment in self.segments[first:]:
                if segment.offset >= end:
                    break
                left, right = (
                    max(start, segment.offset),
                    min(end, segment.offset + segment.end - segment.start),
                )
                if left < right:
                    parts.append(self.data[left:right])
                    segments.append(
                        _Segment(
                            total,
                            segment.start + left - segment.offset,
                            segment.start + right - segment.offset,
                        )
                    )
                    total += right - left
        return SourceView(b"".join(parts), tuple(segments))

    def identity(self) -> tuple[tuple[int, int], ...]:
        return tuple((segment.start, segment.end) for segment in self.segments)


class _Engine:
    def __init__(
        self,
        payload: bytes,
        limits: ParseLimits,
        backend: str,
        locale: str | None,
        use_icu: bool,
        needs_refinement: Callable[[int, int], bool] | None,
        template_language: str | None = None,
    ) -> None:
        self.payload = payload
        self.limits = limits
        self.backend = backend
        self.locale = locale
        self.use_icu = use_icu
        self.needs_refinement = needs_refinement
        self.template_language = template_language
        self.result = ParseResult()
        self.parse_bytes = 0
        self.nodes = 0
        self.injections = 0
        self.seen: set[tuple[tuple[tuple[int, int], ...], str, str]] = set()
        self.limit_diagnostics: set[tuple[str, int, int]] = set()

    def diagnostic(self, code: str, view: SourceView, detail: str = "") -> None:
        if len(self.result.diagnostics) >= 256:
            return
        start, end = view.position(0), view.position(len(view.data), end=True)
        key = (code, start, end)
        if key not in self.limit_diagnostics:
            self.limit_diagnostics.add(key)
            self.result.diagnostics.append(Diagnostic(code, start, end, ENGINE_VERSION, detail))

    def remaining_limits(self) -> ParseLimits:
        return replace(
            self.limits,
            max_nodes=max(1, self.limits.max_nodes - self.nodes),
            max_regions=max(1, self.limits.max_regions - len(self.result.regions)),
            max_injections=max(1, self.limits.max_injections - self.injections),
        )

    def merge(self, scan: Scan, view: SourceView, depth: int) -> None:
        try:
            scan.result.validate(len(view.data))
        except ValueError:
            self.diagnostic(
                "invalid_parser_evidence",
                view,
                "Parser evidence contained out-of-range source coordinates",
            )
            return
        self.nodes += scan.nodes
        self.result.parsers.update(scan.result.parsers)
        self.result.languages.update(scan.result.languages)
        for boundary in scan.result.boundaries:
            if len(self.result.boundaries) >= self.limits.max_regions * 4:
                self.diagnostic("boundary_budget_exhausted", view)
                break
            self.result.boundaries.append(
                replace(
                    boundary, offset=view.position(boundary.offset), depth=depth + boundary.depth
                )
            )
        for region in scan.result.regions:
            if len(self.result.regions) >= self.limits.max_regions:
                self.diagnostic("region_budget_exhausted", view)
                break
            changes = {
                "start": view.position(region.start),
                "end": view.position(region.end, end=True),
                "depth": depth + region.depth,
            }
            if hasattr(region, "path"):
                prefix = f"{view.position(0)}:{view.position(len(view.data), end=True)}:{depth}"
                changes["path"] = (prefix, *region.path)
            self.result.regions.append(replace(region, **changes))
        for diagnostic in scan.result.diagnostics:
            if len(self.result.diagnostics) < 256:
                self.result.diagnostics.append(
                    replace(
                        diagnostic,
                        start=view.position(diagnostic.start),
                        end=view.position(diagnostic.end, end=True),
                    )
                )

    def prose(self, view: SourceView, depth: int) -> None:
        if not view.data:
            return
        if (
            self.nodes >= self.limits.max_nodes
            or len(self.result.regions) >= self.limits.max_regions
        ):
            self.diagnostic(
                "node_budget_exhausted"
                if self.nodes >= self.limits.max_nodes
                else "region_budget_exhausted",
                view,
            )
            return

        def refine(start: int, end: int) -> bool:
            return bool(
                self.needs_refinement
                and self.needs_refinement(view.position(start), view.position(end, end=True))
            )

        evidence = analyze_prose(
            view.data,
            locale=self.locale,
            refine=refine,
            use_icu=self.use_icu,
            limits=self.remaining_limits(),
        )
        self.merge(Scan(evidence, nodes=len(evidence.regions)), view, depth)

    def native(self, view: SourceView, language: str) -> Scan:
        limits = self.remaining_limits()
        if self.nodes >= self.limits.max_nodes:
            self.diagnostic("node_budget_exhausted", view)
            return Scan()
        if self.backend in {"fallback", "lexical", "builtin"}:
            return scan_generic(view.data, language, limits)
        if getattr(limits, "isolate_native", False):
            from .workers import scan_syntax_isolated

            result = scan_syntax_isolated(view.data, language, limits, self.backend)
        else:
            try:
                result = scan_syntax(view.data, language, limits, self.backend)
            except Exception as exc:
                result = Scan(
                    ParseResult(
                        diagnostics=[
                            Diagnostic(
                                "native_parser_exception",
                                0,
                                len(view.data),
                                f"tree_sitter:{language}",
                                type(exc).__name__,
                            )
                        ]
                    )
                )
        if not result.result.parsers:
            fallback = scan_generic(view.data, language, limits)
            fallback.result.extend(result.result)
            return fallback
        return result

    def prose_outside(self, view: SourceView, depth: int, excluded: list[Region]) -> None:
        cursor = 0
        for region in sorted(excluded, key=lambda item: (item.start, -item.end)):
            if cursor < region.start:
                self.prose(view.subview(((cursor, region.start),)), depth)
            cursor = max(cursor, region.end)
        if cursor < len(view.data):
            self.prose(view.subview(((cursor, len(view.data)),)), depth)

    def parse(self, view: SourceView, kind: str, language: str | None, depth: int) -> None:
        if not view.data:
            return
        canonical = canonical_language(language)
        if canonical in {None, "auto", "code", "unknown"}:
            canonical = resolve_language(payload=view.data)
        resolved_kind = kind.lower().replace("-", "_")
        if canonical is None:
            canonical = (
                "unknown"
                if resolved_kind in {"code", "program", "source", "programming"}
                else "prose"
            )
        if canonical in {"text", "plain", "plaintext"}:
            canonical = "prose"
        identity = (view.identity(), canonical, "content")
        if identity in self.seen:
            return
        self.seen.add(identity)
        if depth > self.limits.max_depth:
            self.diagnostic("depth_budget_exhausted", view)
            return
        if self.nodes >= self.limits.max_nodes:
            self.diagnostic("node_budget_exhausted", view)
            return
        if len(self.result.regions) >= self.limits.max_regions:
            self.diagnostic("region_budget_exhausted", view)
            return
        remaining = self.limits.max_parse_bytes - self.parse_bytes
        if remaining <= 0:
            self.diagnostic("parse_byte_budget_exhausted", view)
            return
        if len(view.data) > remaining:
            self.diagnostic(
                "parse_byte_budget_exhausted",
                view,
                f"Only {remaining} bytes remain in the shared recursive analysis budget",
            )
            view = view.subview(((0, remaining),))
        self.parse_bytes += len(view.data)
        self.result.languages.add(canonical)
        queued: list[Injection] = []

        def consume(scan: Scan) -> None:
            self.merge(scan, view, depth)
            queued.extend(scan.injections)

        if canonical == "prose":
            self.prose(view, depth)
            if self.nodes < self.limits.max_nodes and _MARKDOWN_SIGNATURE.search(view.data):
                consume(scan_markdown(view.data, self.remaining_limits()))
            if self.nodes < self.limits.max_nodes and _MARKUP_SIGNATURE.search(view.data):
                consume(scan_markup_fallback(view.data, "html", self.remaining_limits()))
        elif canonical == "markdown":
            native = self.native(view, canonical)
            consume(native)
            if self.nodes < self.limits.max_nodes and not any(
                parser.startswith("tree_sitter:") for parser in native.result.parsers
            ):
                consume(scan_markdown(view.data, self.remaining_limits()))
        elif canonical == "markdown_inline":
            native = self.native(view, canonical)
            consume(native)
            self.prose_outside(
                view,
                depth,
                [region for region in native.result.regions if region.kind == "inline_code"],
            )
            if self.nodes < self.limits.max_nodes and _MARKUP_SIGNATURE.search(view.data):
                consume(scan_markup_fallback(view.data, "html", self.remaining_limits()))
        elif canonical in DOCUMENT_LANGUAGES:
            containers = scan_document_containers(view.data, canonical, self.remaining_limits())
            consume(containers)
            if canonical != "asciidoc":
                consume(self.native(view, canonical))
            # Paragraph/sentence refinement is useful in document text even when
            # its grammar does not label prose leaves (RST and Org vary here).
            self.prose_outside(
                view,
                depth,
                [region for region in containers.result.regions if region.kind == "code_block"],
            )
        elif canonical in MARKUP_LANGUAGES:
            native = self.native(view, canonical)
            consume(native)
            if self.nodes < self.limits.max_nodes and not any(
                parser.startswith("tree_sitter:") for parser in native.result.parsers
            ):
                consume(scan_markup_fallback(view.data, canonical, self.remaining_limits()))
        else:
            consume(self.native(view, canonical))
        # Parent attachments are established before nested evidence is collected.
        # Processing children in byte order makes shared-budget decisions stable.
        queued.sort(key=lambda item: (item.start, -item.end, item.language or "", item.source))
        for injection in queued:
            if self.injections >= self.limits.max_injections:
                self.diagnostic("injection_budget_exhausted", view)
                break
            ranges = injection.ranges or ((injection.start, injection.end),)
            try:
                child = view.subview(ranges)
            except ValueError as exc:
                self.diagnostic("invalid_injection_range", view, str(exc))
                continue
            child_language = injection.language or resolve_language(payload=child.data)
            if injection.source == "template_code" and self.template_language:
                child_language = self.template_language
            child_kind = injection.kind
            if child_language is None and child_kind == "auto":
                child_language = "prose"
            child_id = (
                child.identity(),
                canonical_language(child_language) or "unknown",
                "content",
            )
            if child_id in self.seen:
                continue
            self.injections += 1
            if len(self.result.regions) < self.limits.max_regions:
                self.result.regions.append(
                    Region(
                        child.position(0),
                        child.position(len(child.data), end=True),
                        "embedded_content",
                        child_language,
                        depth + 1,
                        False,
                        injection.source,
                    )
                )
            self.parse(child, child_kind, child_language, depth + 1)


def parse_content(
    payload: bytes,
    *,
    content_kind: str = "auto",
    language: str | None = None,
    filename: str | None = None,
    mime_type: str | None = None,
    locale: str | None = None,
    limits: ParseLimits = _DEFAULT_LIMITS,
    backend: str = "tree_sitter",
    use_icu: bool = False,
    needs_refinement: Callable[[int, int], bool] | None = None,
) -> ParseResult:
    """Analyse arbitrary bytes into bounded, recursive structural evidence.

    ``needs_refinement(start, end)`` lets the exact codec-aware planner request
    sentence/word processing only for paragraphs too large to keep intact. All
    returned coordinates, including those from nested source views, address the
    original ``payload``. Missing grammars and recovery are observable diagnostics.
    """
    if not isinstance(payload, bytes):
        raise TypeError("Content parsing requires bytes to preserve exact source coordinates")
    if backend not in {"tree_sitter", "fallback", "lexical", "builtin"}:
        raise ValueError(f"Unknown parser backend: {backend!r}")
    explicit_kind = content_kind.lower().replace("-", "_")
    hint = language
    if explicit_kind in {"prose", "text", "plain", "plaintext", "natural_language"} and language:
        base = language.lower().replace("_", "-").split("-", 1)[0]
        known_locales = {
            name.split("_", 1)[0].split(".", 1)[0] for name in system_locale.locale_alias
        }
        grammars = set(bundled_languages()) | {
            item for spec in parser_provider_specs() for item in spec.languages
        }
        if (
            canonical_language(language) not in grammars
            and base in known_locales
            and re.fullmatch(r"[A-Za-z]{2,3}(?:[-_][A-Za-z0-9]{2,8})*", language)
        ):
            locale = locale or language.replace("_", "-")
            hint = "prose"
    if hint is None and explicit_kind in DOCUMENT_LANGUAGES | MARKUP_LANGUAGES | {
        "json",
        "yaml",
        "toml",
        "csv",
        "tsv",
        "prose",
        "text",
        "plain",
        "plaintext",
    }:
        hint = explicit_kind
    resolved = resolve_language(hint, filename, mime_type, payload)
    dialect = (language or "").lower()
    if filename:
        dialect = filename.rsplit(".", 1)[-1].lower()
    template_language = {"erb": "ruby", "rhtml": "ruby", "ejs": "javascript"}.get(dialect)
    engine = _Engine(payload, limits, backend, locale, use_icu, needs_refinement, template_language)
    engine.result.parsers.add(ENGINE_VERSION)
    engine.parse(SourceView.original(payload), explicit_kind, resolved, 0)
    # Exact duplicate spans are common where CommonMark and a syntax tree agree.
    # Preserve distinct provenance, but remove repeats caused by overlapping views.
    engine.result.boundaries = list(dict.fromkeys(engine.result.boundaries))
    engine.result.regions = list(dict.fromkeys(engine.result.regions))
    engine.result.diagnostics = list(dict.fromkeys(engine.result.diagnostics))
    engine.result.boundaries.sort(
        key=lambda item: (item.offset, item.depth, item.kind, item.source)
    )
    engine.result.regions.sort(
        key=lambda item: (item.start, -item.end, item.depth, item.kind, item.source)
    )
    engine.result.diagnostics.sort(key=lambda item: (item.start, item.end, item.code, item.parser))
    engine.result.validate(len(payload))
    return engine.result
