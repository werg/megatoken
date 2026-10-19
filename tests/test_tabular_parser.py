"""Delimited documents preserve physical rows, quoted newlines, and original bytes."""

from __future__ import annotations

import random
from dataclasses import replace

import pytest

from megatoken.data.parsing.engine import parse_content
from megatoken.data.parsing.tabular import scan_delimited
from megatoken.data.parsing.types import ParseLimits


LIMITS = ParseLimits(isolate_native=False)


@pytest.mark.parametrize(
    ("language", "source", "rows", "fields"),
    [
        (
            "csv",
            b'name,note\r\nAlice,"Hello,\r\nworld"\r\nBob,"He said ""Hi"""\r\n',
            [
                b"name,note\r\n",
                b'Alice,"Hello,\r\nworld"\r\n',
                b'Bob,"He said ""Hi"""\r\n',
            ],
            [b"name", b"note", b"Alice", b'"Hello,\r\nworld"'],
        ),
        (
            "tsv",
            b'first\tsecond\rlast\t"two\tcolumns"\r',
            [b"first\tsecond\r", b'last\t"two\tcolumns"\r'],
            [b"first", b"second", b"last", b'"two\tcolumns"'],
        ),
        (
            "psv",
            b'left|right\n"a|b"|value\n',
            [b"left|right\n", b'"a|b"|value\n'],
            [b"left", b"right", b'"a|b"', b"value"],
        ),
    ],
)
def test_record_scanner_is_byte_exact_and_respects_quoted_line_endings(
    language, source, rows, fields
):
    scan = scan_delimited(source, language, LIMITS)
    scan.result.validate(len(source))
    actual_rows = [
        source[r.start : r.end] for r in scan.result.regions if r.kind == "row"
    ]
    actual_fields = [
        source[r.start : r.end] for r in scan.result.regions if r.kind == "table_cell"
    ]
    assert actual_rows == rows
    assert actual_fields[: len(fields)] == fields
    assert not scan.result.diagnostics
    assert f"delimited-records-v1:{language}" in scan.result.parsers


@pytest.mark.parametrize("language", ["csv", "tsv", "psv"])
def test_delimited_records_are_routed_without_native_grammar(language):
    delimiter = {"csv": b",", "tsv": b"\t", "psv": b"|"}[language]
    payload = b"first" + delimiter + b'"value with\nline"\r\nsecond' + delimiter + b"third"
    by_hint = parse_content(
        payload, language=language, limits=LIMITS, backend="fallback"
    )
    by_filename = parse_content(
        payload, filename="table." + language, limits=LIMITS
    )
    by_kind = parse_content(
        payload, content_kind=language, limits=LIMITS
    )
    for parsed in (by_hint, by_filename, by_kind):
        parsed.validate(len(payload))
        assert language in parsed.languages
        assert f"delimited-records-v1:{language}" in parsed.parsers
        rows = [r for r in parsed.regions if r.kind == "row"]
        assert len(rows) == 2
        assert payload[rows[0].start : rows[0].end].endswith(b"\r\n")
        assert not any(d.code == "grammar_unavailable" for d in parsed.diagnostics)


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (b'one,"unfinished\r\n', "unterminated_delimited_quote"),
        (b'one,"closed"unexpected\n', "malformed_delimited_quote"),
        (b'one,abc"stray\n', "malformed_delimited_quote"),
    ],
)
def test_malformed_delimiters_report_recoverable_diagnostics(payload, expected):
    parsed = parse_content(payload, language="csv", limits=LIMITS)
    parsed.validate(len(payload))
    assert any(d.code == expected for d in parsed.diagnostics)
    assert b"".join(
        payload[region.start : region.end]
        for region in parsed.regions
        if region.kind == "row"
    ) == payload


def test_tabular_scanner_budget_caps_and_reports_partial_analysis():
    source = b'a,"very long\nvalue",b\n' * 64
    limits = replace(
        LIMITS, max_parse_bytes=32, max_nodes=16, max_regions=3
    )
    scan = scan_delimited(source, "csv", limits)
    scan.result.validate(len(source))
    assert scan.nodes <= limits.max_nodes
    assert len(scan.result.regions) <= limits.max_regions
    assert any("budget_exhausted" in d.code for d in scan.result.diagnostics)


def test_table_scanner_fuzz_never_leaks_invalid_coordinates():
    randomizer = random.Random(8734)
    limits = replace(LIMITS, max_parse_bytes=192, max_nodes=256, max_regions=72)
    for language in ("csv", "tsv", "psv"):
        for _ in range(40):
            payload = randomizer.randbytes(randomizer.randrange(1, 257))
            parsed = parse_content(
                payload, language=language, backend="fallback", limits=limits
            )
            parsed.validate(len(payload))
            assert all(
                0 <= r.start <= r.end <= len(payload) for r in parsed.regions
            )
            assert len(parsed.regions) <= limits.max_regions
