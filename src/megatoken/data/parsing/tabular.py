"""Streaming, byte-exact recovery for CSV, TSV, and pipe-separated records.

Quoted fields may contain separators and physical line endings; doubled quotes
are escapes only while inside a quoted field. No decoding, dialect guessing,
schema inference, CSV evaluation, or normalization takes place. Rows protect
semantic chunk cohesion while field boundaries remain available when a row
exceeds the physical caps.
"""

from __future__ import annotations

from .syntax import Scan
from .types import Boundary, Diagnostic, ParseLimits, Region

DELIMITERS = {"csv": 44, "tsv": 9, "psv": 124}
PARSER_VERSION = "delimited-records-v1"


def scan_delimited(payload: bytes, language: str, limits: ParseLimits) -> Scan:
    """Produce bounded row and field evidence in original byte coordinates."""
    if language not in DELIMITERS:
        raise ValueError(f"Unsupported delimited record language: {language!r}")
    scan = Scan()
    source = f"{PARSER_VERSION}:{language}"
    scan.result.parsers.add(source)
    scan.result.languages.add(language)
    delimiter, maximum = DELIMITERS[language], min(len(payload), limits.max_parse_bytes)
    index = field_start = row_start = 0
    inside_quote = after_quote = False
    diagnostics = 0

    def report(code: str, start: int, end: int) -> None:
        nonlocal diagnostics
        if diagnostics < 32:
            scan.result.diagnostics.append(Diagnostic(code, start, end, source))
            diagnostics += 1

    def field(end: int) -> None:
        if field_start < end and len(scan.result.regions) < limits.max_regions:
            scan.result.regions.append(
                Region(field_start, end, "table_cell", language, 1, False, source)
            )
        if end < len(payload) and len(scan.result.boundaries) < limits.max_regions * 4:
            scan.result.boundaries.append(Boundary(end, "statement", 1, source))

    def row(end: int) -> None:
        if row_start < end and len(scan.result.regions) < limits.max_regions:
            scan.result.regions.append(
                Region(row_start, end, "row", language, 0, True, source)
            )
        if len(scan.result.boundaries) < limits.max_regions * 4:
            scan.result.boundaries.append(Boundary(end, "row", 0, source))

    while index < maximum and scan.nodes < limits.max_nodes:
        scan.nodes += 1
        value = payload[index]
        if inside_quote:
            if value == 34:
                if index + 1 < maximum and payload[index + 1] == 34:
                    index += 2
                    continue
                inside_quote = False
                after_quote = True
            index += 1
            continue
        if value == 34 and index == field_start:
            inside_quote = True
            after_quote = False
        elif value == delimiter:
            field(index)
            field_start = index + 1
            after_quote = False
        elif value in {10, 13}:
            field(index)
            stop = index + 2 if value == 13 and index + 1 < maximum and payload[index + 1] == 10 else index + 1
            row(stop)
            field_start = row_start = stop
            after_quote = False
            index = stop
            continue
        elif value == 34 or (after_quote and value not in {32, 9}):
            report("malformed_delimited_quote", index, index + 1)
            after_quote = False
        index += 1

    if row_start < index:
        field(index)
        row(index)
    if index < len(payload):
        report("tabular_budget_exhausted", index, len(payload))
    elif inside_quote:
        report("unterminated_delimited_quote", field_start, index)
    if len(scan.result.regions) >= limits.max_regions:
        report("tabular_region_budget_exhausted", 0, index)
    return scan
