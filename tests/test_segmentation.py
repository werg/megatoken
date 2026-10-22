"""The array-based UAX #29 segmenters must return exactly what ``uniseg`` returns."""

from __future__ import annotations

import random

import pytest
from uniseg.graphemecluster import grapheme_cluster_boundaries
from uniseg.sentencebreak import sentence_boundaries as reference_sentences
from uniseg.wordbreak import word_boundaries as reference_words

from megatoken.data.parsing.segmentation import (
    grapheme_unsafe,
    sentence_boundaries,
    word_boundaries,
)

# One or more characters of every property class a rule distinguishes (including
# Extend/Format/ZWJ, which the sentence and word rules skip), so short random strings
# reach every rule and its interactions.
SENTENCE_ALPHABET = [
    "-", "\r", "\n", "́", " ", "­", " ", "a", "A", "中", "5", ".", ",", "!",
    '"', ")", "?", "。", "é", "É", "\u0085", " ", " ",
]  # fmt: skip
WORD_ALPHABET = [
    "-", "\r", "\n", "\x0b", "́", "‍", "­", "ア", "א", "a", "'", '"', ".",
    ":", ",", "5", "_", " ", "\U0001f600", "A", "é", "⁠", "　", " ", ";",
]  # fmt: skip
GRAPHEME_ALPHABET = [
    "a", "\r", "\n", "\t", "́", "‍", "؀", "각", "ᄀ", "ᅡ", "ᆨ",
    "क", "्", "ख", "\U0001f600", "\U0001f1fa", "\U0001f1f8", "ः", "­",
    "é", "中", " ", "\U0001f3fb", "ำ", "ก",
]  # fmt: skip


def random_strings(alphabet: list[str], seed: int, count: int, longest: int):
    rng = random.Random(seed)
    for _ in range(count):
        size = rng.choice((2, 5, 9, longest))
        yield "".join(rng.choice(alphabet) for _ in range(rng.randint(0, size)))


@pytest.mark.parametrize("seed", [1, 2])
def test_sentence_boundaries_equal_the_reference(seed):
    for text in random_strings(SENTENCE_ALPHABET, seed, 4000, 24):
        assert sentence_boundaries(text) == list(reference_sentences(text)), repr(text)


@pytest.mark.parametrize("seed", [1, 2])
def test_word_boundaries_equal_the_reference(seed):
    for text in random_strings(WORD_ALPHABET, seed, 4000, 24):
        assert word_boundaries(text) == list(reference_words(text)), repr(text)


@pytest.mark.parametrize("seed", [1, 2])
def test_grapheme_boundaries_equal_the_reference(seed):
    for text in random_strings(GRAPHEME_ALPHABET, seed, 4000, 24):
        if text:
            expected = set(grapheme_cluster_boundaries(text))
            assert set(range(len(text) + 1)) - grapheme_unsafe(text) == expected, repr(text)


def test_regional_indicator_pairs_are_answered_by_the_reference():
    text = "\U0001f1fa\U0001f1f8\U0001f1fa\U0001f1f8 flags"
    assert word_boundaries(text) == list(reference_words(text))


def test_empty_text_has_no_boundaries():
    assert sentence_boundaries("") == [] == word_boundaries("")
    assert grapheme_unsafe("") == set()


def test_sentences_follow_terminators_closers_and_paragraph_breaks():
    text = 'He said "Stop." Then left!\r\nNext (really?) one. 3.5 is not a stop. e.g. this'
    assert sentence_boundaries(text) == list(reference_sentences(text))
    assert 0 in sentence_boundaries(text) and len(text) in sentence_boundaries(text)


def test_real_prose_paragraph_matches_the_reference():
    text = (
        "Dr. Smith's lab (founded in 1999) studies bees.  They buzz!  Why?  Nobody knows... "
        "The price rose 3.5% to $1,234.56 on 12:30 p.m. Monday.\n\n"
        "Zwischen „Anführungszeichen“ steht Text. 日本語の文章です。続きます？"
    )
    assert sentence_boundaries(text) == list(reference_sentences(text))
    assert word_boundaries(text) == list(reference_words(text))
    assert set(range(len(text) + 1)) - grapheme_unsafe(text) == set(
        grapheme_cluster_boundaries(text)
    )
