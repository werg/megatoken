"""Lossless, lazy Unicode prose analysis for standalone and embedded text.

Paragraphs are inexpensive structural evidence. UAX #29 sentence and word
analysis runs only when the caller asks to refine an oversized span. Optional
ICU refinement is selected explicitly and recorded in the runtime identity;
the set of installed packages never chooses a segmentation policy implicitly.

Every reported position addresses the original bytes. In particular, string
escapes, CRLF, combining sequences, and malformed UTF-8 are never rewritten.
"""

from __future__ import annotations

import importlib
import re
from array import array
from bisect import bisect_right
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from heapq import merge
from importlib.metadata import version
from itertools import chain, islice
from types import ModuleType
from typing import Literal

import numpy as np
from uniseg import unidata_version
from uniseg.sentencebreak import SentenceBreak, sentence_break

from .segmentation import grapheme_unsafe, sentence_boundaries, word_boundaries
from .types import Boundary, Diagnostic, ParseLimits, ParseResult, Region

PROSE_RULES_VERSION = "unicode-prose-v1"
UNICODE_VERSION = unidata_version

# The CR/LF alternatives cannot backtrack into the middle of one CRLF pair.
_LINE_BREAK = r"(?:\r\n|\r(?!\n)|(?<!\r)\n|[\v\x85\u2028])"
_HORIZONTAL_SPACE = r"[^\S\r\n\v\f\x85\u2028\u2029]"
_PARAGRAPH_SEPARATOR = re.compile(
    rf"{_LINE_BREAK}(?:{_HORIZONTAL_SPACE}*{_LINE_BREAK})+|[\f\u2029]+"
)
_LINE_SEPARATOR = re.compile(r"\r\n|[\r\n\v\f\x85\u2028\u2029]")
_IDEOGRAPHIC_INDENT = re.compile(rf"{_LINE_BREAK}(?=\u3000+\S)")
_INVALID_UTF8 = re.compile("[\udc80-\udcff]+")
_ASCII_ATOM = re.compile(r"[A-Za-z0-9_]+\Z")
_ABBREVIATION_SUFFIX = re.compile(r"(?:[^\W\d_]+[ \t]*\.[ \t]*){1,6}\Z")
_ABBREVIATION_CONTINUATION = re.compile(r"[ \t]*(?:[^\W\d_]+[ \t]*\.[ \t]*){1,5}")
_INITIAL_SUFFIX = re.compile(r"(?<!\w)(?:[^\W\d_]\.[ \t]*){1,4}\Z")

# This is a high-confidence punctuation supplement, not language detection.
# UAX #29 already recognizes most of these; including them explicitly also
# permits sentences without inter-sentence spaces in their writing systems.
_CULTURAL_TERMINALS = frozenset("。！？｡؟۔।॥።᙮꧉")
_ASCII_CLOSERS = frozenset("\"')]}>")
_WORD_TRAILERS = frozenset(",;:،؛、")
_COMMON_ABBREVIATIONS = frozenset({"dr", "prof", "e.g", "i.e", "cf"})

# Small conservative locale profiles filter obvious abbreviation boundaries.
# They are refinements of UAX #29, not claims to full linguistic parsing.
# Additional language-specific segmenters can be provided by explicit ICU.
_LOCALE_ABBREVIATIONS = {
    "en": frozenset("mr mrs ms dr prof sr jr st mt vs fig no nos vol pp cf e.g i.e".split()),
    "de": frozenset("dr prof hr fr bzw z.b d.h u.a u.ä usw ggf ca nr abs bsp vgl".split()),
    "fr": frozenset("m mme mmes mlle mlles dr pr st ste pp ex fig env av apr".split()),
    "es": frozenset("sr sra srta dr dra ud uds p.ej núm aprox pág págs d dña".split()),
    "pt": frozenset("sr sra srta dr dra prof profa ex p.ex pág págs aprox".split()),
    "it": frozenset("sig sig.ra sig.na dott dott.ssa prof prof.ssa es pag pagg art".split()),
    "nl": frozenset("dhr mevr dr prof bijv m.a.w d.w.z nr".split()),
    "sv": frozenset("dr prof bl.a d.v.s t.ex osv nr".split()),
    "da": frozenset("dr prof bl.a f.eks dvs nr".split()),
    "nb": frozenset("dr prof bl.a f.eks dvs nr".split()),
    "nn": frozenset("dr prof bl.a t.d dvs nr".split()),
    "fi": frozenset("esim mm ns jne prof tri".split()),
    "pl": frozenset("dr prof mgr np tzn itd itp tj ul nr godz".split()),
    "cs": frozenset("dr prof např tj atd č str ing".split()),
    "sk": frozenset("dr prof napr t.j atď č ing".split()),
    "hu": frozenset("dr prof pl stb ún ill".split()),
    "ro": frozenset("dr prof dl dna dvs nr".split()),
    "ru": frozenset("г гг ул д кв т.е т.д т.п напр им см стр рис проф тов".split()),
    "uk": frozenset("р рр вул ім т.д т.п напр див стор рис".split()),
    "tr": frozenset("dr prof doç sn say örn vb vs no".split()),
}

_LINK = re.compile(
    r"(?i)(?<![\w@])(?:"
    r"(?:[a-z][a-z0-9+.-]{1,31}://|www\.)[^\s<>\"'`]+"
    r"|[\w.+-]{1,256}@[\w-]+(?:\.[\w-]+)+)"
)
_ESCAPE = re.compile(
    r"\\(?:u\{[0-9a-fA-F]{1,8}\}|u[0-9a-fA-F]{4}|U[0-9a-fA-F]{8}"
    r"|x[0-9a-fA-F]{2}|[0-7]{1,3}|N\{[^{}\r\n]{1,128}\}"
    r"|[\\\"'/abfnrtv]|\r\n|[\r\n])"
)


_OFFSET_DTYPE = {"I": np.uint32, "Q": np.uint64}


@dataclass(frozen=True, slots=True)
class _TextView:
    text: str
    byte_offsets: Sequence[int]

    @classmethod
    def from_bytes(cls, payload: bytes) -> _TextView:
        text = payload.decode("utf-8", errors="surrogateescape")
        if payload.isascii():
            return cls(text, range(len(payload) + 1))
        # Each undecodable byte became one escaped surrogate and stays one byte long.
        codepoints = np.frombuffer(text.encode("utf-32-le", "surrogatepass"), dtype="<u4")
        widths = np.where(
            (codepoints >= 0xDC80) & (codepoints <= 0xDCFF),
            1,
            1 + (codepoints >= 0x80) + (codepoints >= 0x800) + (codepoints >= 0x10000),
        )
        offsets = array("I" if len(payload) < 2**32 else "Q", [0])
        offsets.frombytes(np.cumsum(widths, dtype=_OFFSET_DTYPE[offsets.typecode]).tobytes())
        return cls(text, offsets)


def _valid_runs(text: str) -> Iterable[tuple[int, int]]:
    cursor = 0
    for match in _INVALID_UTF8.finditer(text):
        if cursor < match.start():
            yield cursor, match.start()
        cursor = match.end()
    if cursor < len(text):
        yield cursor, len(text)


def _grapheme_unsafe(text: str) -> set[int]:
    """The indices strictly inside a grapheme cluster, which no cut may use.

    Every other index is a boundary. Invalid bytes are independent atoms, while
    valid neighboring runs retain the full Unicode grapheme guarantees.
    """
    unsafe: set[int] = set()
    for first, last in _valid_runs(text):
        unsafe.update(first + index for index in grapheme_unsafe(text[first:last]))
    return unsafe


def grapheme_offsets(payload: bytes) -> tuple[int, ...]:
    """Return local byte offsets, including both ends, safe for final splitting.

    Malformed UTF-8 bytes each form a recoverable atom. An actual grapheme may
    still exceed a caller's hard byte/token budget; that caller must explicitly
    report any unavoidable smaller cut rather than pretending it was safe.
    """
    return tuple(np.flatnonzero(grapheme_cut_mask(payload)).tolist())


def grapheme_cut_mask(payload: bytes) -> np.ndarray:
    """``mask[offset]`` is true when cutting ``payload`` at byte ``offset`` splits no grapheme."""
    view = _TextView.from_bytes(payload)
    if payload.isascii():
        mask = np.ones(len(payload) + 1, dtype=bool)
    else:
        mask = np.zeros(len(payload) + 1, dtype=bool)
        mask[np.frombuffer(view.byte_offsets, dtype=_OFFSET_DTYPE[view.byte_offsets.typecode])] = (
            True
        )
    for index in _grapheme_unsafe(view.text):
        mask[view.byte_offsets[index]] = False
    return mask


def _load_icu() -> ModuleType:
    try:
        return importlib.import_module("icu")
    except ImportError as error:
        raise RuntimeError(
            "ICU prose refinement was selected but PyICU is not installed; "
            "install the configured ICU dependency or select Unicode refinement"
        ) from error


def prose_runtime_identity(use_icu: bool = False) -> dict[str, object]:
    """Identify every selected algorithm/runtime that can change boundaries."""
    identity: dict[str, object] = {
        "rules": PROSE_RULES_VERSION,
        "uniseg": version("uniseg"),
        "unicode": UNICODE_VERSION,
        "icu": None,
    }
    if use_icu:
        icu = _load_icu()
        identity["icu"] = {
            "binding": str(icu.VERSION),
            "library": str(icu.ICU_VERSION),
            "unicode": str(icu.UNICODE_VERSION),
        }
    return identity


def _paragraph_ranges(text: str) -> Iterable[tuple[int, int]]:
    # Ideographic paragraph indentation is a useful extra signal for prose
    # whose paragraphs are not separated by blank lines. Ordinary soft line
    # wrapping stays inside one paragraph.
    ends = merge(
        (match.end() for match in _PARAGRAPH_SEPARATOR.finditer(text)),
        (match.end() for match in _IDEOGRAPHIC_INDENT.finditer(text)),
    )
    cursor = 0
    for end in chain(ends, (len(text),)):
        if cursor < end:
            yield cursor, end
        cursor = end


@lru_cache(maxsize=2048)
def _sentence_property(character: str) -> SentenceBreak:
    if 0xDC80 <= ord(character) <= 0xDCFF:
        return SentenceBreak.OTHER
    return sentence_break(character)


def _is_closer(character: str) -> bool:
    if character in _ASCII_CLOSERS:
        return True
    return not character.isascii() and _sentence_property(character) == SentenceBreak.CLOSE


def _is_terminal(character: str) -> bool:
    if character in ".!?" or character in _CULTURAL_TERMINALS:
        return True
    return not character.isascii() and _sentence_property(character) in {
        SentenceBreak.ATERM,
        SentenceBreak.STERM,
    }


class _BoundaryFinisher:
    """Attach punctuation/whitespace without rescanning long suffixes.

    UAX word boundaries can propose every character in a punctuation run.
    Scanning its remainder for each proposal would be quadratic on malformed
    text such as a megabyte of question marks. Run lookup stays logarithmic.
    """

    def __init__(self, text: str) -> None:
        self.spaces = [(match.start(), match.end()) for match in re.finditer(r"\s+", text)]
        self.space_starts = [span[0] for span in self.spaces]
        self.punctuation: list[tuple[int, int]] = []
        cursor: int | None = None
        for index, character in enumerate(text):
            if _is_terminal(character) or _is_closer(character) or character in _WORD_TRAILERS:
                if cursor is None:
                    cursor = index
            elif cursor is not None:
                self.punctuation.append((cursor, index))
                cursor = None
        if cursor is not None:
            self.punctuation.append((cursor, len(text)))
        self.punctuation_starts = [span[0] for span in self.punctuation]

    @staticmethod
    def _run_end(offset: int, spans: list[tuple[int, int]], starts: list[int]) -> int:
        index = bisect_right(starts, offset) - 1
        return spans[index][1] if index >= 0 and offset < spans[index][1] else offset

    def finish(self, offset: int, *, punctuation: bool = False) -> int:
        if punctuation:
            offset = self._run_end(offset, self.punctuation, self.punctuation_starts)
        return self._run_end(offset, self.spaces, self.space_starts)


def _terminal_before(text: str, offset: int) -> int | None:
    cursor = offset - 1
    while cursor >= 0:
        character = text[cursor]
        if character.isspace() or _is_closer(character):
            cursor -= 1
            continue
        if not character.isascii() and _sentence_property(character) in {
            SentenceBreak.EXTEND,
            SentenceBreak.FORMAT,
        }:
            cursor -= 1
            continue
        return cursor if _is_terminal(character) else None
    return None


def _abbreviation_boundary(text: str, offset: int, terminal: int, locale: str | None) -> bool:
    if text[terminal] != "." or offset >= len(text):
        return False
    # Do not misclassify decimals, versions, or dotted identifiers as sentences.
    if (
        terminal > 0
        and terminal + 1 < len(text)
        and text[terminal - 1].isascii()
        and text[terminal - 1].isalnum()
        and text[terminal + 1].isascii()
        and text[terminal + 1].isalnum()
    ):
        return True
    window = text[max(0, terminal - 96) : terminal + 1]
    match = _ABBREVIATION_SUFFIX.search(window)
    if match:
        abbreviation = re.sub(r"[ \t]", "", match[0]).casefold().rstrip(".")
        language = (locale or "und").replace("_", "-").split("-", 1)[0].split("@", 1)[0]
        known = _COMMON_ABBREVIATIONS | _LOCALE_ABBREVIATIONS.get(language.casefold(), frozenset())
        if any(abbreviation == item or abbreviation.endswith("." + item) for item in known):
            return True
        continuation = _ABBREVIATION_CONTINUATION.match(text, terminal + 1, terminal + 97)
        if continuation:
            following = re.sub(r"[ \t]", "", continuation[0]).casefold().rstrip(".")
            for item in known:
                for position, character in enumerate(item):
                    if character != ".":
                        continue
                    prefix, suffix = item[:position], item[position + 1 :]
                    if (abbreviation == prefix or abbreviation.endswith("." + prefix)) and (
                        following == suffix or following.startswith(suffix + ".")
                    ):
                        return True
    initial = _INITIAL_SUFFIX.search(window)
    if initial and text[offset].isupper():
        return any(character.isupper() for character in initial[0])
    return False


def _link_spans(text: str) -> Iterable[tuple[int, int, str]]:
    for match in _LINK.finditer(text):
        end = match.end()
        while end > match.start() and text[end - 1] in ".,;:!?)]}":
            end -= 1
        if match.start() < end:
            kind = "email" if "@" in match[0] and "://" not in match[0] else "url"
            yield match.start(), end, kind


def _protected_spans(text: str, limit: int) -> tuple[list[tuple[int, int, str]], bool]:
    spans = list(
        islice(
            merge(
                _link_spans(text),
                ((match.start(), match.end(), "escape") for match in _ESCAPE.finditer(text)),
            ),
            limit + 1,
        )
    )
    return spans[:limit], len(spans) > limit


class _CutSafety:
    def __init__(self, text: str, protected: list[tuple[int, int, str]]) -> None:
        self.text = text
        self.starts = [item[0] for item in protected]
        # Prefix maxima also handle a string escape contained in a URL.
        self.ends: list[int] = []
        for _, end, _ in protected:
            self.ends.append(max(end, self.ends[-1] if self.ends else 0))
        self.unsafe = _grapheme_unsafe(text)

    def safe(self, offset: int) -> bool:
        if not 0 < offset <= len(self.text):
            return False
        index = bisect_right(self.starts, offset - 1) - 1
        if index >= 0 and offset < self.ends[index]:
            return False
        return offset not in self.unsafe


def _icu_indices(
    text: str,
    locale: str | None,
    kind: Literal["sentence", "word"],
    icu: ModuleType,
    max_boundaries: int,
) -> set[int]:
    """Convert ICU's UTF-16 boundaries to Python indices; reject split pairs."""
    icu_locale = icu.Locale((locale or "und").replace("-", "_"))
    if kind == "sentence":
        icu_locale.setKeywordValue("ss", "standard")
        iterator = icu.BreakIterator.createSentenceInstance(icu_locale)
    else:
        iterator = icu.BreakIterator.createWordInstance(icu_locale)
    iterator.setText(text)
    utf16_to_python = {0: 0}
    units = 0
    for index, character in enumerate(text, 1):
        units += 1 if ord(character) <= 0xFFFF else 2
        utf16_to_python[units] = index
    result = {0, len(text)}
    for count, boundary in enumerate(islice(iterator, max_boundaries + 1)):
        if count == max_boundaries:
            raise ValueError("ICU boundary budget exceeded")
        if boundary not in utf16_to_python:
            raise ValueError("ICU returned a boundary outside scalar UTF-16 offsets")
        result.add(utf16_to_python[boundary])
    return result


def _segmentation_indices(
    text: str,
    locale: str | None,
    kind: Literal["sentence", "word"],
    icu: ModuleType | None,
    result: ParseResult,
    view: _TextView,
    first: int,
    max_boundaries: int,
) -> tuple[set[int], str]:
    if _ASCII_ATOM.fullmatch(text):
        return {0, len(text)}, "unicode"
    analyze = sentence_boundaries if kind == "sentence" else word_boundaries
    indices = {0, len(text)}
    icu_successes = icu_failures = 0
    for run_start, run_end in _valid_runs(text):
        run = text[run_start:run_end]
        candidates = list(islice(analyze(run), max_boundaries + 1))
        selected = set(candidates[:max_boundaries])
        if len(candidates) > max_boundaries:
            result.diagnostics.append(
                Diagnostic(
                    "prose_boundary_budget",
                    view.byte_offsets[first + run_start],
                    view.byte_offsets[first + run_end],
                    "unicode-prose",
                    f"{kind}: stopped after {max_boundaries} boundary proposals",
                )
            )
        if icu is not None:
            try:
                selected = _icu_indices(run, locale, kind, icu, max_boundaries)
                icu_successes += 1
            except Exception as error:
                icu_failures += 1
                # Data/backend errors degrade locally. Explicitly selecting an
                # uninstalled backend fails in _load_icu before reaching here.
                result.diagnostics.append(
                    Diagnostic(
                        "icu_refinement_failed",
                        view.byte_offsets[first + run_start],
                        view.byte_offsets[first + run_end],
                        "icu",
                        f"{kind}: {type(error).__name__}; retained Unicode boundaries",
                    )
                )
        indices.update(run_start + index for index in selected)
        if len(indices) > max_boundaries:
            indices = set(sorted(indices)[:max_boundaries]) | {len(text)}
            result.diagnostics.append(
                Diagnostic(
                    "prose_boundary_budget",
                    view.byte_offsets[first],
                    view.byte_offsets[first + len(text)],
                    "unicode-prose",
                    f"{kind}: cumulative boundary proposals reached {max_boundaries}",
                )
            )
            break
    source = "icu" if icu_successes else "unicode"
    if icu_successes and icu_failures:
        source = "unicode+icu"
    return indices, source


def _limit_evidence(result: ParseResult, limits: ParseLimits, end: int) -> bool:
    """Stop analysis once bounded metadata is full; retain the source endpoint."""
    if len(result.regions) <= limits.max_regions and len(result.boundaries) <= limits.max_nodes:
        return False
    result.regions[:] = result.regions[: limits.max_regions]
    result.boundaries[:] = result.boundaries[: max(0, limits.max_nodes - 1)]
    result.boundaries.append(Boundary(end, "paragraph", source="unicode"))
    result.diagnostics.append(
        Diagnostic(
            "prose_evidence_budget",
            0,
            end,
            "unicode-prose",
            "Prose evidence limit reached; remaining cuts use the bounded fallback planner",
        )
    )
    return True


def analyze_prose(
    payload: bytes,
    *,
    locale: str | None = None,
    refine: bool | Callable[[int, int], bool] = False,
    use_icu: bool = False,
    limits: ParseLimits | None = None,
) -> ParseResult:
    """Propose absolute byte boundaries without modifying or decoding output.

    ``refine=False`` performs paragraph detection only. The orchestrator should
    ask for refinement on oversized prose leaves, avoiding expensive linguistic
    work on paragraphs the packer can retain intact. A predicate receives each
    paragraph's absolute byte range and selects it for refinement. ``use_icu``
    is an explicit backend choice, never inferred from the host environment.
    """
    icu = _load_icu() if use_icu else None
    limits = limits or ParseLimits()
    view = _TextView.from_bytes(payload)
    result = ParseResult(parsers={"unicode-prose"}, languages={"text"})
    if use_icu:
        result.parsers.add("icu")
    for count, match in enumerate(
        islice(_INVALID_UTF8.finditer(view.text), limits.max_regions + 1)
    ):
        if count == limits.max_regions:
            result.diagnostics.append(
                Diagnostic(
                    "prose_diagnostic_budget",
                    view.byte_offsets[match.start()],
                    len(payload),
                    "unicode-prose",
                    "Additional invalid UTF-8 diagnostics omitted; every byte is retained",
                )
            )
            break
        result.diagnostics.append(
            Diagnostic(
                "invalid_utf8",
                view.byte_offsets[match.start()],
                view.byte_offsets[match.end()],
                "unicode-prose",
                "Invalid UTF-8 bytes retained as independent fallback atoms",
            )
        )
    for first, last in _paragraph_ranges(view.text):
        byte_first, byte_last = view.byte_offsets[first], view.byte_offsets[last]
        result.regions.append(Region(byte_first, byte_last, "paragraph", "text", source="unicode"))
        result.boundaries.append(Boundary(byte_last, "paragraph", source="unicode"))
        if _limit_evidence(result, limits, len(payload)):
            break
        if not (refine(byte_first, byte_last) if callable(refine) else refine):
            continue
        if byte_last - byte_first > limits.max_parse_bytes:
            result.diagnostics.append(
                Diagnostic(
                    "prose_parse_byte_budget",
                    byte_first,
                    byte_last,
                    "unicode-prose",
                    "Oversized paragraph retained without linguistic refinement",
                )
            )
            continue
        text = view.text[first:last]
        finisher = _BoundaryFinisher(text)
        protected, truncated = _protected_spans(text, limits.max_regions)
        if truncated:
            result.diagnostics.append(
                Diagnostic(
                    "prose_protected_region_budget",
                    byte_first,
                    byte_last,
                    "unicode-prose",
                    "Protected URL/escape candidates reached the configured region limit",
                )
            )
        safety = _CutSafety(text, protected)
        for begin, end, kind in protected:
            result.regions.append(
                Region(
                    view.byte_offsets[first + begin],
                    view.byte_offsets[first + end],
                    kind,
                    "text",
                    depth=2,
                    source="unicode",
                )
            )
        sentence_ends, sentence_source = _segmentation_indices(
            text, locale, "sentence", icu, result, view, first, limits.max_nodes
        )
        sentence_ends.update(
            islice(
                (
                    finisher.finish(index + 1, punctuation=True)
                    for index, character in enumerate(text)
                    if character in _CULTURAL_TERMINALS
                ),
                limits.max_nodes,
            )
        )
        chosen_sentences = set()
        for end in sentence_ends:
            end = finisher.finish(end)
            if not safety.safe(end):
                continue
            terminal = _terminal_before(text, end)
            if terminal is not None and not _abbreviation_boundary(text, end, terminal, locale):
                chosen_sentences.add(end)
        cursor = 0
        for end in sorted(chosen_sentences | {len(text)}):
            if cursor < end:
                result.regions.append(
                    Region(
                        view.byte_offsets[first + cursor],
                        view.byte_offsets[first + end],
                        "sentence",
                        "text",
                        depth=1,
                        source=sentence_source,
                    )
                )
            if end in chosen_sentences:
                result.boundaries.append(
                    Boundary(
                        view.byte_offsets[first + end],
                        "sentence",
                        depth=1,
                        source=sentence_source,
                    )
                )
            cursor = end
        words, word_source = _segmentation_indices(
            text, locale, "word", icu, result, view, first, limits.max_nodes
        )
        word_ends = {finisher.finish(end, punctuation=True) for end in words if end > 0}
        for end in sorted(word_ends):
            if safety.safe(end):
                result.boundaries.append(
                    Boundary(
                        view.byte_offsets[first + end],
                        "word",
                        depth=2,
                        source=word_source,
                    )
                )
        for match in islice(_LINE_SEPARATOR.finditer(text), limits.max_nodes):
            if safety.safe(match.end()):
                result.boundaries.append(
                    Boundary(
                        view.byte_offsets[first + match.end()],
                        "line",
                        depth=2,
                        source="unicode",
                    )
                )
        if _limit_evidence(result, limits, len(payload)):
            break
    result.boundaries.sort(key=lambda item: (item.offset, item.depth, item.kind, item.source))
    return result
