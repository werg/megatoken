"""Unicode (UAX #29) grapheme, sentence and word boundaries at C speed, equal to ``uniseg``.

``uniseg`` is the reference implementation and the source of every character property
used here, but it spends tens of microseconds per character. These functions return the
same boundaries by turning each text into a string of one-letter property classes (one
``str.translate``) and then looking only where a rule can matter:

* graphemes: a boundary between two "plain" characters (no rule ever joins them) is
  certain, so ``uniseg`` itself runs only on the few runs of characters that can join;
* sentences: a break can only follow a sentence terminator (plus closers and spaces) or
  a paragraph separator, so only those few positions evaluate rules SB6-SB11;
* words: the rules only ever remove breaks, so they are evaluated for every position at
  once with array comparisons.

Equality with ``uniseg`` is checked by randomized differential tests, which are the
specification of this module; if a text contains something these shortcuts do not cover
(regional-indicator pairs), ``uniseg`` answers directly.
"""

from __future__ import annotations

import re
from functools import cache

import numpy as np
from uniseg.derived import InCB, indic_conjunct_break
from uniseg.emoji import extended_pictographic
from uniseg.graphemecluster import GCB, grapheme_cluster_boundaries, grapheme_cluster_break
from uniseg.sentencebreak import SB, sentence_break
from uniseg.wordbreak import WB, word_break
from uniseg.wordbreak import word_boundaries as reference_word_boundaries

# One printable ASCII letter per property value; these strings are only ever compared
# inside this module.
_SB_SYMBOL = {
    SB.OTHER: "o", SB.CR: "r", SB.LF: "n", SB.EXTEND: "x", SB.SEP: "p", SB.FORMAT: "f",
    SB.SP: "s", SB.LOWER: "l", SB.UPPER: "u", SB.OLETTER: "a", SB.NUMERIC: "d",
    SB.ATERM: "A", SB.SCONTINUE: "c", SB.STERM: "T", SB.CLOSE: "e",
}  # fmt: skip
_WB_SYMBOL = {
    WB.OTHER: "o", WB.CR: "r", WB.LF: "n", WB.NEWLINE: "w", WB.EXTEND: "x", WB.ZWJ: "z",
    WB.REGIONAL_INDICATOR: "R", WB.FORMAT: "f", WB.KATAKANA: "k", WB.HEBREW_LETTER: "h",
    WB.ALETTER: "a", WB.SINGLE_QUOTE: "q", WB.DOUBLE_QUOTE: "d", WB.MIDNUMLET: "u",
    WB.MIDLETTER: "m", WB.MIDNUM: "b", WB.NUMERIC: "N", WB.EXTENDNUMLET: "e",
    WB.WSEGSPACE: "s",
}  # fmt: skip
_PLAIN_GCB = frozenset({GCB.OTHER, GCB.CONTROL, GCB.CR, GCB.LF})


class _Classes:
    """Lazily extended ``str.translate`` table from a character to its class letter."""

    def __init__(self, classify) -> None:
        self.table: dict[int, str] = {}
        self._classify = classify

    def __call__(self, text: str) -> str:
        table = self.table
        for character in set(text):
            if ord(character) not in table:
                table[ord(character)] = self._classify(character)
        return text.translate(table)


_sentence_classes = _Classes(lambda c: _SB_SYMBOL[sentence_break(c)])
_word_classes = _Classes(lambda c: _WB_SYMBOL[word_break(c)])
# "." plain (joins nothing), "X" may join its neighbours; see grapheme_unsafe.
_grapheme_classes = _Classes(
    lambda c: (
        "."
        if (
            grapheme_cluster_break(c) in _PLAIN_GCB
            and not extended_pictographic(c)
            and indic_conjunct_break(c) == InCB.NONE
        )
        else "X"
    )
)
_CRLF = re.compile(r"\r\n")
_PARAGRAPH_END = re.compile(r"[pn](?=.)|r(?=[^n])", re.DOTALL)
_SKIPPED = re.compile(r"[xf]")
_TERMINATOR = re.compile(r"[AT](e*)s*")
_JOINABLE_RUN = re.compile(r"X+")


def grapheme_unsafe(text: str) -> set[int]:
    """The indices in ``1..len(text)-1`` that are not grapheme-cluster boundaries.

    Every other index, and both ends, is a boundary. Between two plain characters
    (Other, Control, CR, LF that are neither emoji nor Indic consonants) UAX #29 always
    breaks, except inside CR LF, and no rule looks across such a seam; so only runs of
    characters that can join (plus the one plain character on each side that a run can
    attach to) go through ``uniseg``.
    """
    unsafe = {match.start() + 1 for match in _CRLF.finditer(text)}
    if text.isascii():
        return unsafe
    windows: list[list[int]] = []
    for run in _JOINABLE_RUN.finditer(_grapheme_classes(text)):
        start, end = max(0, run.start() - 1), min(len(text), run.end() + 1)
        if windows and start <= windows[-1][1]:  # touching windows are one region
            windows[-1][1] = end
        else:
            windows.append([start, end])
    for start, end in windows:
        region = text[start:end]
        boundaries = set(grapheme_cluster_boundaries(region))
        unsafe.difference_update(range(start + 1, end))
        unsafe.update(start + i for i in range(1, len(region)) if i not in boundaries)
    return unsafe


@cache
def _lut(symbols: str) -> np.ndarray:
    table = np.zeros(256, dtype=bool)
    table[np.frombuffer(symbols.encode("ascii"), dtype=np.uint8)] = True
    return table


def _skip_aware(classes: str, skipped: str):
    """The class bytes with skippable characters removed, and where each one came from."""
    symbols = np.frombuffer(classes.encode("ascii"), dtype=np.uint8)
    keep = ~_lut(skipped)[symbols]
    if keep.all():
        return symbols, None
    return symbols[keep], np.flatnonzero(keep)


def word_boundaries(text: str) -> list[int]:
    """``list(uniseg.wordbreak.word_boundaries(text))``, computed with array operations."""
    if not text:
        return []
    classes = _word_classes(text)
    if "R" in classes:  # regional-indicator pairing is rare; the reference handles it
        return list(reference_word_boundaries(text))
    length = len(text)
    literal = np.frombuffer(classes.encode("ascii"), dtype=np.uint8)
    previous, current = literal[:-1], literal[1:]
    newline = _lut("rnw")
    # WB3-WB4 over literal neighbours, in the reference's order: the first rule decides.
    crlf = (previous == ord("r")) & (current == ord("n"))
    force_break = ~crlf & (newline[previous] | newline[current])
    zwj_picto = np.zeros(length - 1, dtype=bool)
    for position in np.flatnonzero(previous == ord("z")):
        zwj_picto[position] = extended_pictographic(text[position + 1])
    spaces = (previous == ord("s")) & (current == ord("s"))
    decided_join = crlf | (~force_break & (zwj_picto | spaces | _lut("xzf")[current]))
    undecided = ~(decided_join | force_break)
    # WB5-WB13b on the sequence without Extend/Format/ZWJ.
    c, origin = _skip_aware(classes, "xzf")
    joined = np.zeros(len(c), dtype=bool)
    if len(c) > 1:
        zero = np.uint8(0)
        p1 = np.concatenate(([zero], c[:-1]))
        p2 = np.concatenate(([zero, zero], c[:-2]))
        n1 = np.concatenate((c[1:], [zero]))
        letter, mid_letter, mid_num = _lut("ah"), _lut("muq"), _lut("buq")
        h, quote, double, numeric = _lut("h"), _lut("q"), _lut("d"), _lut("N")
        katakana, extend_num = _lut("k"), _lut("e")
        continues_extend, starts_extend = _lut("ahNke"), _lut("ahNk")
        joined = (
            (letter[p1] & letter[c])  # WB5
            | (letter[p1] & mid_letter[c] & letter[n1])  # WB6
            | (letter[p2] & mid_letter[p1] & letter[c])  # WB7
            | (h[p1] & quote[c])  # WB7a
            | (h[p1] & double[c] & h[n1])  # WB7b
            | (h[p2] & double[p1] & h[c])  # WB7c
            | (numeric[p1] & numeric[c])  # WB8
            | (letter[p1] & numeric[c])  # WB9
            | (numeric[p1] & letter[c])  # WB10
            | (numeric[p2] & mid_num[p1] & numeric[c])  # WB11
            | (numeric[p1] & mid_num[c] & numeric[n1])  # WB12
            | (katakana[p1] & katakana[c])  # WB13
            | (continues_extend[p1] & extend_num[c])  # WB13a
            | (extend_num[p1] & starts_extend[c])  # WB13b
        )
        joined[0] = False  # the first kept character has no left neighbour to join
    joins = np.zeros(length, dtype=bool)
    joins[1:] = decided_join
    open_to_rules = np.zeros(length, dtype=bool)
    open_to_rules[1:] = undecided
    kept = np.arange(length) if origin is None else origin
    joins[kept] |= joined & open_to_rules[kept]
    return np.flatnonzero(~joins).tolist() + [length]


def sentence_boundaries(text: str) -> list[int]:
    """``list(uniseg.sentencebreak.sentence_boundaries(text))``, evaluating rules only
    where a sentence can end."""
    if not text:
        return []
    classes = _sentence_classes(text)
    # SB3-SB4: a paragraph separator ends a sentence, but not inside CR LF.
    breaks = {0}
    breaks.update(match.end() for match in _PARAGRAPH_END.finditer(classes))
    # SB5 ignores Extend and Format; the other rules see the text without them.
    skipped = _SKIPPED.sub("", classes)
    origin = (
        [i for i, symbol in enumerate(classes) if symbol not in "xf"]
        if len(skipped) != len(classes)
        else None
    )
    last = len(skipped) - 1
    for match in _TERMINATOR.finditer(skipped):
        terminator = match.start()
        after_closers = terminator + 1 + len(match[1])
        run_end = -1  # the end of the Close/Sp/Numeric/SContinue run being scanned for SB8
        for p in range(terminator + 1, min(match.end(), last) + 1):
            prev, curr = skipped[p - 1], skipped[p]
            if p == terminator + 1:
                if prev == "A" and curr == "d":  # SB6
                    continue
                if p >= 2 and skipped[p - 2] in "ul" and prev == "A" and curr == "u":  # SB7
                    continue
            if skipped[terminator] == "A":  # SB8
                if curr == "l":
                    continue
                if curr in "sdce":
                    if p > run_end:
                        run_end = p
                        while run_end + 1 <= last and skipped[run_end + 1] in "sdce":
                            run_end += 1
                    if run_end + 1 <= last and skipped[run_end + 1] == "l":
                        continue
            if curr in "cAT":  # SB8a
                continue
            if curr in "spnr":  # SB9 (before closers or spaces) and SB10 (after them)
                continue
            if curr == "e" and p <= after_closers:  # SB9
                continue
            breaks.add(p if origin is None else origin[p])
    return sorted(breaks) + [len(text)]
