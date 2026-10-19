"""Regression cases for source-exact fallback parsing across language families.

These paths also run when no grammar can be loaded or a native worker crashes.
They must never classify arbitrary punctuation as comments or invent decoded
HTML/JSON source that is absent from the canonical byte stream.
"""

from __future__ import annotations

import random
from dataclasses import replace

import pytest

from megatoken.data.parsing.documents import scan_markup_fallback
from megatoken.data.parsing.engine import parse_content
from megatoken.data.parsing.syntax import scan_generic
from megatoken.data.parsing.types import ParseLimits


LIMITS = ParseLimits(isolate_native=False)


@pytest.mark.parametrize(
    ("language", "source", "expected_comments"),
    [
        ("css", b"#button { --color: red; background: #fff; /* note */ }", [b"/* note */"]),
        ("scss", b"$value: #fff; // note\r.next { color: #000; }", [b"// note"]),
        (
            "rust",
            b"#[derive(Clone)]\nfn f<'a>(x: &'a str) { /* a /* nested */ b */ }",
            [b"/* a /* nested */ b */"],
        ),
        ("c", b'#include <stdio.h>\rint x=1; // comment\r', [b"// comment"]),
        ("csharp", b"#nullable enable\nvar i = 2; // note", [b"// note"]),
        ("javascript", b'const x = "https://host/#a"; x--; // note\rnext();', [b"// note"]),
        ("python", b"x=1# comment\ry=2", [b"# comment"]),
        ("yaml", b"key: value#literal\rnext: value # comment\r", [b"# comment"]),
        ("sql", b"SELECT 1 -- comment\rSELECT 2 /* block */;", [b"-- comment", b"/* block */"]),
        ("haskell", b"item = 1 {- outer {- inner -} end -} + 2", [b"{- outer {- inner -} end -}"]),
        (
            "ocaml",
            b"let item = 1 (* outer (* inner *) end *) + 2",
            [b"(* outer (* inner *) end *)"],
        ),
        ("clojure", b'(def x 1) ; comment\r(def y 2)', [b"; comment"]),
        ("latex", b"\\% escaped\n\\section{A} % comment\n", [b"% comment"]),
        (
            "unknown",
            b"http://host/path/#anchor\n// comment\n# comment",
            [b"// comment", b"# comment"],
        ),
    ],
)
def test_dialect_aware_comments_preserve_operators_and_coordinates(
    language, source, expected_comments
):
    scan = scan_generic(source, language, LIMITS)
    scan.result.validate(len(source))
    actual = [
        source[region.start : region.end]
        for region in scan.result.regions
        if region.kind == "comment"
    ]
    assert actual == expected_comments
    assert "generic_lexical:v2" in scan.result.parsers


@pytest.mark.parametrize(
    ("language", "source", "expected_inner"),
    [
        ("rust", b'const X: &str = r###"http://host/*hi*/\n"#"###;', b'http://host/*hi*/\n"#'),
        ("rust", b'const X: &[u8] = br#"abc " quoted"#;', b'abc " quoted'),
        (
            "cpp",
            b'const char* s = R"sql(SELECT "value" /* not a comment */)sql";',
            b'SELECT "value" /* not a comment */',
        ),
        ("cpp", b'auto s = u8R"doc(<p>"hello"</p>)doc";', b'<p>"hello"</p>'),
    ],
)
def test_language_aware_raw_literals_keep_internal_quotes_and_comment_markers(
    language, source, expected_inner
):
    scan = scan_generic(source, language, LIMITS)
    scan.result.validate(len(source))
    strings = [r for r in scan.result.regions if r.kind == "string"]
    assert len(strings) == 1
    inner = next(item for item in scan.injections if item.source == "lexical_string")
    assert source[inner.start : inner.end] == expected_inner
    assert not any(r.kind == "comment" for r in scan.result.regions)
    assert not any(d.code == "unterminated_literal" for d in scan.result.diagnostics)


@pytest.mark.parametrize(
    ("language", "source"),
    [
        ("rust", b'const X: &str = r#"unterminated'),
        ("cpp", b'auto s = R"foo(unterminated'),
        ("haskell", b"let x = {- unterminated"),
        ("ocaml", b"let x = (* unterminated"),
        ("css", b"div { /* unterminated"),
    ],
)
def test_unfinished_comments_and_raw_strings_report_bounded_recovery(language, source):
    scan = scan_generic(source, language, LIMITS)
    scan.result.validate(len(source))
    assert any(
        d.code in {"unterminated_comment", "unterminated_literal"}
        for d in scan.result.diagnostics
    )
    assert all(0 <= item.start <= item.end <= len(source) for item in scan.result.regions)


def test_markup_recovery_recognizes_attribute_dialects_and_data_script_types():
    source = (
        b'<div title="Bonjour. Bienvenue!" onclick="show(1)" '
        b'style="color: red" srcdoc="<p>Hello.</p>" '
        b'data-hint="onclick=not_an_event">'
        b'<script type="text/plain">Not JavaScript.</script>'
        b'<script type="application/ld+json">{"message":"Hello."}</script>'
        b'<script type="module">const x = 1;</script>'
        b'<style lang="scss">$text: red;</style></div>'
    )
    scan = scan_markup_fallback(source, "html", LIMITS)
    scan.result.validate(len(source))
    values = {
        (i.language, source[i.start : i.end])
        for i in scan.injections
        if i.source in {"html_attribute", "html_raw_content"}
    }
    assert ("prose", b"Bonjour. Bienvenue!") in values
    assert ("javascript", b"show(1)") in values
    assert ("css", b"color: red") in values
    assert ("html", b"<p>Hello.</p>") in values
    assert ("prose", b"Not JavaScript.") in values
    assert ("json", b'{"message":"Hello."}') in values
    assert ("javascript", b"const x = 1;") in values
    assert ("scss", b"$text: red;") in values
    assert not any(b"not_an_event" == item[1] for item in values)
    result = parse_content(source, language="html", backend="fallback", limits=LIMITS)
    result.validate(len(source))
    assert {"html", "prose", "javascript", "css", "json", "scss"} <= result.languages


def test_markup_recovery_does_not_invent_events_inside_attribute_values():
    payload = b'<p title="onclick=not-an-event > here" once="plain" oninput="f(1)">Text.</p>'
    scan = scan_markup_fallback(payload, "html", LIMITS)
    javascript = [item for item in scan.injections if item.language == "javascript"]
    assert len(javascript) == 1
    assert payload[javascript[0].start : javascript[0].end] == b"f(1)"
    assert payload[scan.result.regions[0].start : scan.result.regions[0].end] == (
        b'<p title="onclick=not-an-event > here" once="plain" oninput="f(1)">'
    )


def test_markup_recovery_obeys_tiny_injection_limit():
    source = (
        b'<div title="First"><p alt="Second">Body</p>'
        b'<script type="text/javascript">third()</script>'
        b'<!-- fourth --><![CDATA[fifth]]></div>'
    )
    limits = replace(LIMITS, max_injections=1)
    scan = scan_markup_fallback(source, "html", limits)
    assert len(scan.injections) <= 1
    scan.result.validate(len(source))
    result = parse_content(source, language="html", backend="fallback", limits=limits)
    result.validate(len(source))
    assert not any(d.code == "invalid_injection_range" for d in result.diagnostics)


def test_recovery_fuzz_preserves_bounded_byte_evidence():
    rng = random.Random(324)
    limits = replace(LIMITS, max_nodes=72, max_regions=48, max_injections=12, max_parse_bytes=160)
    for language in ("unknown", "css", "javascript", "rust", "ocaml", "sql", "html"):
        for _ in range(24):
            payload = rng.randbytes(rng.randrange(1, 257))
            scan = scan_generic(payload, language, limits)
            scan.result.validate(len(payload))
            assert len(scan.result.regions) <= limits.max_regions
            assert len(scan.injections) <= limits.max_injections
            for item in scan.injections:
                assert 0 <= item.start < item.end <= len(payload)
                assert all(0 <= a < b <= len(payload) for a, b in item.ranges)
