"""Contracts for lossless multilingual prose and explicit optional refinement."""

from __future__ import annotations

from types import ModuleType

import pytest

from megatoken.data.parsing import prose
from megatoken.data.parsing.prose import analyze_prose, grapheme_offsets, prose_runtime_identity
from megatoken.data.parsing.types import ParseLimits


def boundaries_of(result, kind):
    return {item.offset for item in result.boundaries if item.kind == kind}


def sentence_parts(text, *, locale=None, **kwargs):
    payload = text.encode("utf-8")
    result = analyze_prose(payload, refine=True, locale=locale, **kwargs)
    ends = sorted(boundaries_of(result, "sentence") | {len(payload)})
    cursor = 0
    pieces = []
    for end in ends:
        pieces.append(payload[cursor:end].decode("utf-8"))
        cursor = end
    return pieces


def test_paragraph_scan_is_lazy_and_keeps_soft_wrapping_and_crlf(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("A paragraph that fits must not run linguistic refinement")

    monkeypatch.setattr(prose, "sentence_boundaries", unexpected)
    monkeypatch.setattr(prose, "word_boundaries", unexpected)
    monkeypatch.setattr(prose, "grapheme_cluster_boundaries", unexpected)
    first = "第一段 e\u0301。\r\nA soft wrap stays here.\r\n\t\r\n"
    second = "下一段。\r\n"
    payload = (first + second).encode()
    result = analyze_prose(payload, start=17)
    assert boundaries_of(result, "paragraph") == {17 + len(first.encode()), 17 + len(payload)}
    assert {item.kind for item in result.boundaries} == {"paragraph"}
    paragraphs = [item for item in result.regions if item.kind == "paragraph"]
    assert [payload[item.start - 17 : item.end - 17] for item in paragraphs] == [
        first.encode(),
        second.encode(),
    ]
    result.validate(17 + len(payload))


def test_refinement_predicate_sees_absolute_byte_ranges_and_selects_only_oversized_paragraph():
    first = "短い段落。\n\n".encode()
    second = b"A much longer first sentence. A second sentence.\n\n"
    third = "終わり。".encode()
    base = 13
    seen = []

    def refine(start, end):
        seen.append((start, end))
        return end - start > 32

    result = analyze_prose(first + second + third, start=base, refine=refine)
    assert seen == [
        (base, base + len(first)),
        (base + len(first), base + len(first) + len(second)),
        (base + len(first) + len(second), base + len(first) + len(second) + len(third)),
    ]
    assert boundaries_of(result, "sentence") == {
        base + len(first) + len(b"A much longer first sentence. "),
        base + len(first) + len(second),
    }


@pytest.mark.parametrize(
    "separator", ["\n\n", "\r\n\r\n", "\r\r", "\x85\x85", "\u2028\u2028", "\u2029", "\f"]
)
def test_multicultural_paragraph_separators_are_preserved(separator):
    first = "Earlier paragraph." + separator
    payload = (first + "次の段落。").encode()
    result = analyze_prose(payload)
    assert boundaries_of(result, "paragraph") == {len(first.encode()), len(payload)}
    assert b"".join(payload[item.start : item.end] for item in result.regions) == payload


def test_single_crlf_is_neither_blank_line_nor_two_boundaries():
    payload = b"First physical line\r\nsecond physical line."
    result = analyze_prose(payload, refine=True)
    assert boundaries_of(result, "paragraph") == {len(payload)}
    assert boundaries_of(result, "sentence") == {len(payload)}
    assert boundaries_of(result, "line") == {len(b"First physical line\r\n")}
    assert all(payload[item.offset - 1 : item.offset + 1] != b"\r\n" for item in result.boundaries)


def test_ideographic_indentation_can_start_a_paragraph_without_blank_line():
    first = "第一段。\n"
    second = "\u3000\u3000第二段。\n"
    payload = (first + second + "\u3000第三段。").encode()
    result = analyze_prose(payload)
    assert boundaries_of(result, "paragraph") == {
        len(first.encode()),
        len((first + second).encode()),
        len(payload),
    }


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("你好。", "再见！"),
        ("「そうです。」", "次の文です。"),
        ("مرحبا؟", "نعم۔"),
        ("राम आया। ", "फिर गया॥"),
        ("প্রথম বাক্য। ", "পরের বাক্য।"),
        ("መጀመሪያ። ", "ሁለተኛ።"),
        ("Hello, “world!” ", "Another sentence."),
        ("처음 문장입니다. ", "다음 문장입니다."),
        ("זו שאלה? ", "זה משפט."),
        ("ภาษาไทยประโยคแรก! ", "ประโยคถัดไป?"),
    ],
)
def test_sentence_refinement_covers_scripts_and_closing_punctuation(first, second):
    text = first + second
    assert sentence_parts(text) == [first, second]
    result = analyze_prose(text.encode(), refine=True)
    safe = set(grapheme_offsets(text.encode()))
    assert all(item.offset in safe for item in result.boundaries)


@pytest.mark.parametrize(
    ("locale", "first", "second"),
    [
        ("en-US", "Mr. Smith and Dr. Jones arrived. ", "Then they left."),
        ("de-DE", "Das ist z. B. sinnvoll. ", "Danach geht es weiter."),
        ("de", "Prof. Müller arbeitet. ", "Morgen ist frei."),
        ("fr", "Mme. Dupont arrive. ", "Elle repart."),
        ("es", "La Sra. García llegó. ", "Luego salió."),
        ("pt", "O Sr. Silva chegou. ", "Depois saiu."),
        ("it", "Il dott. Rossi arriva. ", "Poi parte."),
        ("ru", "Это рис. Один в книге. ", "Дальше другой."),
        ("pl", "To np. Przykład zdania. ", "Potem następne."),
        ("tr", "Dr. Yılmaz geldi. ", "Sonra ayrıldı."),
    ],
)
def test_explicit_locale_filters_common_abbreviations(locale, first, second):
    assert sentence_parts(first + second, locale=locale) == [first, second]


def test_locale_does_not_require_a_known_profile():
    text = "First sentence. Second sentence."
    assert sentence_parts(text, locale="x-private-language") == [
        "First sentence. ",
        "Second sentence.",
    ]


def test_decimals_initials_urls_and_emails_are_not_sentence_fragments():
    first = "J. R. Smith paid 3.14 at https://example.org/Foo.Bar?x=1.2 using person@example.org. "
    second = "The next sentence remains separate."
    payload = (first + second).encode()
    result = analyze_prose(payload, refine=True)
    assert sentence_parts(first + second) == [first, second]
    atoms = [item for item in result.regions if item.kind in {"url", "email"}]
    assert {payload[item.start : item.end] for item in atoms} == {
        b"https://example.org/Foo.Bar?x=1.2",
        b"person@example.org",
    }
    assert all(
        not atom.start < boundary.offset < atom.end
        for atom in atoms
        for boundary in result.boundaries
    )


def test_program_string_escapes_remain_raw_and_atomic():
    payload = rb"First sentence. Literal\n\nseparators and \u2029 stay raw. End."
    result = analyze_prose(payload, refine=True)
    assert boundaries_of(result, "paragraph") == {len(payload)}
    escapes = [item for item in result.regions if item.kind == "escape"]
    assert [payload[item.start : item.end] for item in escapes] == [b"\\n", b"\\n", b"\\u2029"]
    assert all(
        not atom.start < boundary.offset < atom.end
        for atom in escapes
        for boundary in result.boundaries
    )


@pytest.mark.parametrize("cluster", ["e\u0301", "👩🏽\u200d🚀", "🇩🇪", "क्ष", "각", "\r\n"])
def test_grapheme_boundaries_do_not_split_combining_emoji_indic_or_crlf(cluster):
    payload = ("a" + cluster + "z").encode()
    assert grapheme_offsets(payload) == (0, 1, 1 + len(cluster.encode()), len(payload))
    result = analyze_prose(payload, refine=True)
    assert all(item.offset in grapheme_offsets(payload) for item in result.boundaries)


def test_malformed_utf8_retains_valid_graphemes_and_reports_exact_bad_ranges():
    first = "e\u0301".encode()
    last = "👩🏽\u200d🚀".encode()
    payload = b"\xff" + first + b"\x80\x81" + last + b"\xfe"
    base = 19
    result = analyze_prose(payload, start=base, refine=True)
    assert [
        (item.start - base, item.end - base)
        for item in result.diagnostics
        if item.code == "invalid_utf8"
    ] == [(0, 1), (1 + len(first), 3 + len(first)), (len(payload) - 1, len(payload))]
    offsets = grapheme_offsets(payload)
    assert offsets == (
        0,
        1,
        1 + len(first),
        2 + len(first),
        3 + len(first),
        len(payload) - 1,
        len(payload),
    )
    assert b"".join(payload[a:b] for a, b in zip(offsets, offsets[1:], strict=False)) == payload
    assert all(item.offset - base in offsets for item in result.boundaries)
    result.validate(base + len(payload))


def test_every_byte_value_has_a_lossless_fallback():
    payload = bytes(range(256)) * 3
    result = analyze_prose(payload, refine=True)
    result.validate(len(payload))
    offsets = grapheme_offsets(payload)
    assert offsets[0] == 0 and offsets[-1] == len(payload)
    assert all(a < b for a, b in zip(offsets, offsets[1:], strict=False))
    assert b"".join(payload[a:b] for a, b in zip(offsets, offsets[1:], strict=False)) == payload
    assert any(item.code == "invalid_utf8" for item in result.diagnostics)


def fake_icu(*, bad_offset=False):
    """Exercise real mapping/dispatch logic without a system ICU installation."""
    module = ModuleType("icu")
    module.VERSION = "test-binding"
    module.ICU_VERSION = "test-library"
    module.UNICODE_VERSION = "test-unicode"
    module.locales = []

    class Locale:
        def __init__(self, name):
            self.name = name
            self.keywords = {}
            module.locales.append(self)

        def setKeywordValue(self, key, value):
            self.keywords[key] = value

    class Iterator:
        def setText(self, text):
            self.text = text

        def __iter__(self):
            if bad_offset:
                yield 1  # Inside the opening emoji's surrogate pair.
            for position, character in enumerate(self.text):
                if character == " ":
                    yield len(self.text[: position + 1].encode("utf-16-le")) // 2
            yield len(self.text.encode("utf-16-le")) // 2

    class BreakIterator:
        @staticmethod
        def createSentenceInstance(locale):
            return Iterator()

        @staticmethod
        def createWordInstance(locale):
            return Iterator()

    module.Locale = Locale
    module.BreakIterator = BreakIterator
    return module


def test_icu_utf16_indices_are_mapped_back_to_exact_source_bytes(monkeypatch):
    icu = fake_icu()
    monkeypatch.setattr(prose, "_load_icu", lambda: icu)
    first = "🙂 First. "
    second = "Second."
    result = analyze_prose(
        (first + second).encode(), start=23, refine=True, use_icu=True, locale="de-DE"
    )
    assert boundaries_of(result, "sentence") == {
        23 + len(first.encode()),
        23 + len((first + second).encode()),
    }
    assert result.parsers == {"unicode-prose", "icu"}
    assert not result.diagnostics
    assert icu.locales[0].name == "de_DE"
    assert icu.locales[0].keywords == {"ss": "standard"}
    assert prose_runtime_identity(True)["icu"] == {
        "binding": "test-binding",
        "library": "test-library",
        "unicode": "test-unicode",
    }


def test_bad_icu_offsets_degrade_to_unicode_with_visible_provenance(monkeypatch):
    icu = fake_icu(bad_offset=True)
    monkeypatch.setattr(prose, "_load_icu", lambda: icu)
    payload = "🙂 First. Second.".encode()
    result = analyze_prose(payload, refine=True, use_icu=True)
    expected = analyze_prose(payload, refine=True)
    assert boundaries_of(result, "sentence") == boundaries_of(expected, "sentence")
    assert boundaries_of(result, "word") == boundaries_of(expected, "word")
    assert [item.code for item in result.diagnostics] == [
        "icu_refinement_failed",
        "icu_refinement_failed",
    ]
    assert all(item.source == "unicode" for item in result.boundaries)


def test_backend_selection_is_explicit_and_missing_selected_backend_fails(monkeypatch):
    def absent():
        raise RuntimeError("ICU prose refinement was selected but PyICU is not installed")

    monkeypatch.setattr(prose, "_load_icu", absent)
    assert prose_runtime_identity()["icu"] is None
    assert analyze_prose(b"One. Two.", refine=True).parsers == {"unicode-prose"}
    with pytest.raises(RuntimeError, match="selected but PyICU is not installed"):
        analyze_prose(b"One. Two.", refine=True, use_icu=True)
    with pytest.raises(RuntimeError, match="selected but PyICU is not installed"):
        prose_runtime_identity(True)


def test_uniseg_unicode_and_rules_versions_are_in_runtime_identity():
    identity = prose_runtime_identity()
    assert identity == {
        "rules": "unicode-prose-v1",
        "uniseg": "0.10.1",
        "unicode": "16.0.0",
        "icu": None,
    }


def test_very_long_identifier_avoids_unnecessary_unicode_passes(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("ASCII identifiers have no internal sentence/word/grapheme cuts")

    monkeypatch.setattr(prose, "sentence_boundaries", unexpected)
    monkeypatch.setattr(prose, "word_boundaries", unexpected)
    monkeypatch.setattr(prose, "grapheme_cluster_boundaries", unexpected)
    payload = b"a" * 250_000
    result = analyze_prose(payload, refine=True)
    assert boundaries_of(result, "word") == {len(payload)}
    assert not boundaries_of(result, "sentence")
    assert boundaries_of(result, "paragraph") == {len(payload)}


def test_long_punctuation_run_does_not_repeatedly_scan_its_suffix(monkeypatch):
    original = prose._is_terminal
    calls = 0

    def counted(character):
        nonlocal calls
        calls += 1
        return original(character)

    monkeypatch.setattr(prose, "_is_terminal", counted)
    payload = b"!" * 2_000 + b" Next sentence."
    result = analyze_prose(payload, refine=True)
    assert calls < 10 * len(payload)
    assert boundaries_of(result, "sentence") == {2_001, len(payload)}


def test_linguistic_byte_budget_skips_a_large_paragraph_without_changing_source(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("Oversized linguistic input must not enter the Unicode engine")

    monkeypatch.setattr(prose, "sentence_boundaries", unexpected)
    monkeypatch.setattr(prose, "word_boundaries", unexpected)
    payload = b"This sentence exceeds the configured analysis budget."
    result = analyze_prose(payload, refine=True, limits=ParseLimits(max_parse_bytes=16))
    assert boundaries_of(result, "paragraph") == {len(payload)}
    assert not boundaries_of(result, "sentence")
    assert [item.code for item in result.diagnostics] == ["prose_parse_byte_budget"]


def test_prose_evidence_limits_bound_regions_and_cuts_while_retaining_final_endpoint():
    payload = b"A sentence. Another sentence.\n\n" * 100
    limits = ParseLimits(max_regions=7, max_nodes=11)
    result = analyze_prose(payload, refine=True, limits=limits)
    assert len(result.regions) <= limits.max_regions
    assert len(result.boundaries) <= limits.max_nodes
    assert max(item.offset for item in result.boundaries) == len(payload)
    assert any(item.code == "prose_evidence_budget" for item in result.diagnostics)
    result.validate(len(payload))


def test_many_paragraphs_stop_at_the_evidence_budget_without_linguistic_work(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("The lazy paragraph path must not perform linguistic work")

    monkeypatch.setattr(prose, "sentence_boundaries", unexpected)
    payload = b"Tiny paragraph.\n\n" * 30_000
    result = analyze_prose(payload, limits=ParseLimits(max_regions=4, max_nodes=8))
    assert len(result.regions) == 4
    assert len(result.boundaries) <= 8
    assert max(item.offset for item in result.boundaries) == len(payload)
    assert any(item.code == "prose_evidence_budget" for item in result.diagnostics)


def test_refiner_proposals_are_bounded_even_if_a_backend_repeats_a_boundary(monkeypatch):
    def repeated(*args, **kwargs):
        while True:
            yield 0

    monkeypatch.setattr(prose, "sentence_boundaries", repeated)
    monkeypatch.setattr(prose, "word_boundaries", repeated)
    payload = b"Word followed by another."
    result = analyze_prose(payload, refine=True, limits=ParseLimits(max_nodes=8))
    assert max(item.offset for item in result.boundaries) == len(payload)
    assert any(item.code == "prose_boundary_budget" for item in result.diagnostics)


def test_invalid_utf8_diagnostics_are_bounded_without_losing_any_bytes():
    payload = b"\xffgood" * 1_000
    result = analyze_prose(payload, limits=ParseLimits(max_regions=3))
    assert sum(item.code == "invalid_utf8" for item in result.diagnostics) == 3
    assert result.diagnostics[-1].code == "prose_diagnostic_budget"
    assert result.diagnostics[-1].end == len(payload)
    offsets = grapheme_offsets(payload)
    assert b"".join(payload[a:b] for a, b in zip(offsets, offsets[1:], strict=False)) == payload


def test_empty_and_whitespace_only_payloads_and_offsets():
    assert not analyze_prose(b"").boundaries
    assert grapheme_offsets(b"") == (0,)
    payload = b" \t\r\n\r\n"
    result = analyze_prose(payload, refine=True)
    result.validate(len(payload))
    assert not boundaries_of(result, "sentence")
    assert boundaries_of(result, "paragraph") == {len(payload)}
    with pytest.raises(ValueError, match="nonnegative integer"):
        analyze_prose(b"hello", start=-1)
    with pytest.raises(ValueError, match="nonnegative integer"):
        analyze_prose(b"hello", start=True)


@pytest.mark.parametrize(
    ("locale", "text", "sentences"),
    [
        # Thai separates sentences with a space, which UAX #29 never breaks on:
        # ICU's contribution there is dictionary word segmentation.
        ("th-TH", "ภาษาไทยมีการแบ่งคำและประโยค ข้อความถัดไปมีความหมาย", False),
        ("ja-JP", "これは最初の文です。次の文章を読んでください。", True),
        ("ar", "هذه جملة واضحة. وهذه جملة أخرى؟", True),
        ("hi-IN", "यह पहला वाक्य है। यह अगला वाक्य है।", True),
        ("de-DE", "Dr. Müller liest z. B. dieses Buch. Danach schreibt er weiter.", True),
    ],
)
def test_installed_icu_refines_real_multilingual_text_without_offset_drift(locale, text, sentences):
    pytest.importorskip("icu", reason="Optional ICU integration runs in the i18n CI job")
    payload = ("👩🏾‍💻 " + text + "\r\n").encode()
    result = analyze_prose(payload, start=37, locale=locale, refine=True, use_icu=True)
    safe = {37 + offset for offset in grapheme_offsets(payload)}
    assert "icu" in result.parsers
    assert not result.diagnostics
    assert bool(boundaries_of(result, "sentence")) is sentences
    assert len(boundaries_of(result, "word")) > 2
    assert all(boundary.offset in safe for boundary in result.boundaries)
    assert max(boundary.offset for boundary in result.boundaries) == 37 + len(payload)
    assert prose_runtime_identity(True)["icu"]["library"]
