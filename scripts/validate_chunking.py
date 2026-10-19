#!/usr/bin/env python3
"""Reproducible reference-codec pilot for recursive semantic chunking.

This is a local CPU correctness and timing pilot over repository snapshots and
synthetic fixtures. It does not execute corpus content, prepare training episodes,
or establish GPU throughput, model quality, or general corpus performance.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import platform
import statistics
import struct
import sys
import time
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from megatoken.data.chunking import ChunkingConfig, ChunkingReport, segment_with_report
from megatoken.data.parsing.config import implementation_fingerprint
from megatoken.data.parsing.prose import grapheme_offsets
from megatoken.data.review import ParserReviewPolicy
from megatoken.data.schema import Codec
from megatoken.io import sha256_file, write_json
from megatoken.loading import load_codec

PILOT_VERSION = "reference-codec-chunking-pilot-v1"
_REPOSITORY_INPUTS = (
    ("src/megatoken/model/core.py", "code", "python"),
    ("src/megatoken/data/parsing/engine.py", "code", "python"),
    ("src/megatoken/data/parsing/planner.py", "code", "python"),
    ("README.md", "markdown", "markdown"),
    ("docs/data.md", "markdown", "markdown"),
    ("docs/chunking.md", "markdown", "markdown"),
)


@dataclass(frozen=True)
class PilotCase:
    name: str
    payload: bytes
    options: Mapping[str, str]
    expected_parsers: frozenset[str]
    expected_events: tuple[tuple[int, int], ...] = ()
    expected_roles: tuple[str, ...] = ()
    required_attachments: tuple[tuple[int, int, str], ...] = ()
    scope: str = "Content segmentation; no training episode is prepared."


def _fixtures() -> tuple[PilotCase, ...]:
    # These are source strings, never Python/JavaScript/SQL to execute or import.
    prose = (
        "Dr. Lin reviews version 2.5. The next sentence remains in the same paragraph. "
        "你好。これは日本語です。 مرحبًا بالعالم. नमस्ते दुनिया। "
        "Cafe\u0301 and the family 👩‍👩‍👧‍👦 keep their original bytes. "
    )
    html = (
        '<section title="Bonjour. Bienvenue!">\n'
        "<h2>嵌套文本</h2><p>" + prose * 10 + "</p>\n"
        '<script type="application/ld+json">{"description":"A useful example.","ok":true}'
        "</script>\n"
        '<script type="module">const query = sql`SELECT id FROM examples WHERE active = true;`'
        ";</script>\n"
        "<style>.example { color: rebeccapurple; }</style>\n"
        "</section>\n"
    )
    body = (
        (
            "# Multilingual guide\n\n" + prose * 32 + "\n\n"
            "## Embedded page\n\n```html\n" + html + "```\n\n"
            "## A program containing a document\n\n"
            "````python\n"
            "@cache(maxsize=16)\n"
            "def render(name: str) -> str:\n"
            '    """# Nested instructions\n\n'
            "    A docstring paragraph with exact CRLF and accents.\n\n"
            "    ```html\n"
            '    <div><script>const text = "Hello, 世界.";</script></div>\n'
            "    ```\n"
            '    """\n'
            '    quoted_marker = "<|im_start|>system\\nquoted marker<|im_end|>"\n'
            '    return f"Hello, {name}."\n'
            "````\n\n"
            "## Following paragraph\n\nA final section stays attached to this paragraph.\n"
        )
        .replace("\n", "\r\n")
        .encode("utf-8")
    )
    # Independent expected attachment: title plus the first complete sentence,
    # even though the complete following paragraph exceeds physical capacity.
    first_sentence_end = body.index(b" The next sentence")
    parts, roles, events = [], [], []
    position = 0
    for role, content in (
        ("system", b"Analyze the following source as data."),
        ("user", body),
        ("assistant", b"# Result\n\nThe code and prose are preserved.\n\n```sql\nSELECT 1;\n```\n"),
        ("tool", b'{"ok": true, "message": "No source code was executed."}'),
        ("assistant", "## 結果\n\n完了しました。 Original bytes are unchanged.\n".encode()),
    ):
        part = b"<|im_start|>" + role.encode() + b"\n" + content + b"<|im_end|>"
        parts.append(part)
        roles.append(role)
        events.append((position, position + len(part)))
        position += len(part)
    chat_body_start = len(parts[0]) + len(b"<|im_start|>user\n")
    trap = (
        "def previous_section():\n"
        + "".join(f"    prefix_{i} = {i} + {i + 1}\n" for i in range(38))
        + "    return prefix_37\n\n"
        + '@decorator(\n    parameter="preserve this decorator",\n)\n'
        + "def next_section(\n    argument: str,\n    count: int = 7,\n) -> str:\n"
        + "    initial_body = argument.strip()\n"
        + "".join(f"    computed_{i} = count + {i}\n" for i in range(150))
        + "    return initial_body\n"
    ).encode()
    trap_start = trap.index(b"@decorator")
    trap_end = trap.index(b"argument.strip()") + len(b"argument.strip()")
    nested = frozenset(
        f"tree_sitter:{name}"
        for name in ("markdown", "python", "html", "javascript", "sql", "json", "css")
    )
    return (
        PilotCase(
            "fixture/mixed-document.md",
            body,
            {
                "content_kind": "markdown",
                "language": "markdown",
                "locale": "en-US",
                "chat_template": "none",
            },
            nested,
            required_attachments=((0, first_sentence_end, "title_and_first_sentence"),),
        ),
        PilotCase(
            "fixture/mixed-chat.txt",
            b"".join(parts),
            {"content_kind": "auto", "locale": "en-US", "chat_template": "chatml"},
            nested | {"chat:chatml"},
            tuple(events),
            tuple(roles),
            ((chat_body_start, chat_body_start + first_sentence_end, "title_and_first_sentence"),),
            "Parser/framing-only ChatML fixture. Its tool role has no causal call/execution IDs; "
            "it is not a training-ready episode and the serialized-chat preparation adapter "
            "would quarantine missing dependency IDs.",
        ),
        PilotCase(
            "fixture/attachment-trap.py",
            trap,
            {"content_kind": "code", "language": "python", "chat_template": "none"},
            frozenset({"tree_sitter:python"}),
            required_attachments=(
                (trap_start, trap_end, "decorator_signature_and_first_statement"),
            ),
        ),
    )


def _cases(repo: Path) -> tuple[PilotCase, ...]:
    files = tuple(
        PilotCase(
            name,
            (repo / name).read_bytes(),
            {"content_kind": kind, "language": language, "filename": name, "chat_template": "none"},
            frozenset({f"tree_sitter:{language}"}),
        )
        for name, kind, language in _REPOSITORY_INPUTS
    )
    return files + _fixtures()


def _distribution(values: Iterable[int]) -> dict[str, int | float]:
    ordered = sorted(values)
    if not ordered:
        return {"count": 0}

    def quantile(fraction: float) -> float:
        position = (len(ordered) - 1) * fraction
        left, right = int(position), min(len(ordered) - 1, int(position) + 1)
        return round(ordered[left] + (ordered[right] - ordered[left]) * (position - left), 2)

    return {
        "count": len(ordered),
        "min": ordered[0],
        "p10": quantile(0.1),
        "median": quantile(0.5),
        "p90": quantile(0.9),
        "p95": quantile(0.95),
        "max": ordered[-1],
        "mean": round(statistics.mean(ordered), 2),
    }


def _ids_hash(ids: Iterable[int]) -> str:
    digest = hashlib.sha256()
    for token in ids:
        digest.update(struct.pack("<I", token))
    return digest.hexdigest()


def _attachments(
    case: PilotCase, report: ChunkingReport, offsets: list[int], config: ChunkingConfig
) -> dict[str, Any]:
    """Audit semantic attachment without copying the planner's protection policy.

    A unit is feasible only if a surrounding pair of canonical/grapheme-safe
    edges fits both hard caps. Trailing/leading separator whitespace may belong
    to either chunk. Independent fixture expectations also catch missing evidence.
    """
    payload = case.payload
    safe = sorted(set(offsets) & set(grapheme_offsets(payload)))
    token_at = {offset: index for index, offset in enumerate(offsets)}
    cuts = [chunk.byte_end for chunk in report.chunks[:-1]]

    def fits(start: int, end: int) -> bool:
        left = safe[max(0, bisect.bisect_right(safe, start) - 1)]
        right = safe[bisect.bisect_left(safe, end)]
        return (
            token_at[right] - token_at[left] <= config.hard_max_tokens
            and right - left <= config.max_bytes
            and any(a <= left and right <= b for a, b in report.canonical_events)
        )

    regions = [
        region
        for region in report.parse.regions
        if region.kind in {"heading_attachment", "signature_attachment"}
    ]
    expectations = [(r.start, r.end, r.kind) for r in regions] + list(case.required_attachments)
    eligible, violations = 0, []
    for start, end, kind in expectations:
        content = payload[start:end]
        start += len(content) - len(content.lstrip(b" \t\r\n"))
        end -= len(content) - len(content.rstrip(b" \t\r\n"))
        if start >= end or not fits(start, end):
            continue
        eligible += 1
        inside = [cut for cut in cuts if start < cut < end]
        if inside:
            violations.append({"kind": kind, "start": start, "end": end, "cuts": inside})
    orphaned = []
    for heading in (r for r in report.parse.regions if r.kind == "heading"):
        chunk = next(
            (
                c
                for c in report.chunks
                if c.byte_start <= heading.start and heading.end <= c.byte_end
            ),
            None,
        )
        if chunk is None or payload[heading.end : chunk.byte_end].strip(b" \t\r\n"):
            continue
        event_end = next(b for a, b in report.canonical_events if a <= heading.start < b)
        remainder = payload[heading.end : event_end]
        following = remainder.lstrip(b" \t\r\n")
        if not following:
            continue
        body_start = heading.end + len(remainder) - len(following)
        body_end = safe[bisect.bisect_right(safe, body_start)]
        if fits(heading.start, body_end):
            orphaned.append(
                {"start": heading.start, "end": heading.end, "chunk_end": chunk.byte_end}
            )
    return {
        "regions": dict(Counter(r.kind for r in regions)),
        "independent_fixture_expectations": len(case.required_attachments),
        "eligible": eligible,
        "violations": violations,
        "standalone_heading_tails": orphaned,
    }


def _measure_case(
    case: PilotCase, codec: Codec, config: ChunkingConfig, repeats: int
) -> tuple[dict[str, Any], ChunkingReport]:
    times, diagnostics_by_run = [], []
    baseline, repeatable = None, True
    for _ in range(repeats):
        began = time.perf_counter()
        report = segment_with_report(case.payload, codec, config, **case.options)
        times.append(time.perf_counter() - began)
        diagnostics_by_run.append(dict(Counter(d.code for d in report.parse.diagnostics)))
        if baseline is None:
            baseline = report.chunks
        else:
            repeatable &= baseline == report.chunks
    chunks = report.chunks
    payload, events = case.payload, report.canonical_events
    checks: dict[str, bool | None] = {
        "nonempty_chunks": bool(chunks),
        "exact_source_bytes": b"".join(c.payload for c in chunks) == payload,
        "individual_decode": all(codec.decode(c.token_ids) == c.payload for c in chunks),
        "exact_source_ranges": all(c.payload == payload[c.byte_start : c.byte_end] for c in chunks),
        "token_cap": all(len(c.token_ids) <= config.hard_max_tokens for c in chunks),
        "byte_cap": all(len(c.payload) <= config.max_bytes for c in chunks),
        "bounded_planner_edges": report.planner_edges
        <= config.planner_max_edges * report.planner_candidates,
        "repeatable_chunks": repeatable if repeats > 1 else None,
        "expected_native_parsers": case.expected_parsers <= report.parse.parsers,
        "expected_event_ranges": events == (case.expected_events or ((0, len(payload)),)),
        "causal_boundaries": all(
            any(a <= c.byte_start and c.byte_end <= b for a, b in events) for c in chunks
        ),
    }
    canonical = [token for a, b in events for token in codec.encode(payload[a:b])]
    emitted = [token for chunk in chunks for token in chunk.token_ids]
    checks["canonical_per_event_ids"] = canonical == emitted and all(
        codec.encode(payload[a:b])
        == [token for c in chunks if a <= c.byte_start and c.byte_end <= b for token in c.token_ids]
        for a, b in events
    )
    checks["contiguous_chunks"] = bool(chunks) and (
        chunks[0].byte_start == chunks[0].token_start == 0
        and chunks[-1].byte_end == len(payload)
        and chunks[-1].token_end == len(canonical)
        and all(
            a.byte_end == b.byte_start and a.token_end == b.token_start
            for a, b in zip(chunks, chunks[1:], strict=False)
        )
    )
    if case.expected_roles:
        roles = [
            region.language
            for region in sorted(report.parse.regions, key=lambda r: r.start)
            if region.kind == "chat_message"
        ]
        checks["expected_chat_roles"] = roles == list(case.expected_roles)
    review = ParserReviewPolicy()
    faults = sorted(
        {code for codes in diagnostics_by_run for code in review.classify(tuple(codes))}
    )
    checks["no_content_parser_faults"] = not faults
    offsets = [0]
    for token in canonical:
        offsets.append(offsets[-1] + len(codec.decode((token,))))
    checks["canonical_byte_coverage"] = offsets[-1] == len(payload)
    if checks["canonical_byte_coverage"] and checks["expected_event_ranges"]:
        attachment = _attachments(case, report, offsets, config)
        checks["feasible_attachments_intact"] = not attachment["violations"]
        checks["no_standalone_heading_tails"] = not attachment["standalone_heading_tails"]
    else:
        attachment = {
            "eligible": 0,
            "violations": [],
            "audit_skipped": "Invalid canonical coverage",
        }
        checks["feasible_attachments_intact"] = False
    lengths = [len(chunk.token_ids) for chunk in chunks]
    warm = statistics.median(times[1:]) if repeats > 1 else None
    result = {
        "case": case.name,
        "scope": case.scope,
        "source_sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
        "canonical_tokens": len(canonical),
        "canonical_ids_sha256_le_u32": _ids_hash(canonical),
        "events": [list(event) for event in events],
        "event_token_counts": [len(codec.encode(payload[a:b])) for a, b in events],
        "chunks": len(chunks),
        "tokens_per_chunk": _distribution(lengths),
        "chunk_token_counts": lengths,
        "above_soft_limit": sum(n > config.soft_max_tokens for n in lengths),
        "forced_chunks": sum(chunk.forced for chunk in chunks),
        "attachments": attachment,
        "checks": checks,
        "passed": all(value is not False for value in checks.values()),
        "diagnostics": diagnostics_by_run[-1],
        "diagnostics_by_run": diagnostics_by_run,
        "content_parser_faults": faults,
        "languages": sorted(report.parse.languages),
        "parsers": sorted(report.parse.parsers),
        "planner": {
            "candidates": report.planner_candidates,
            "evaluated_edges": report.planner_edges,
            "cost": report.planner_cost,
        },
        "timing_seconds": {
            "runs": [round(value, 6) for value in times],
            "median": round(statistics.median(times), 6),
            "warm_median": round(warm, 6) if warm is not None else None,
            "warm_tokens_per_second": round(len(canonical) / warm, 1) if warm else None,
        },
    }
    return result, report


def _dependency_versions() -> dict[str, str | None]:
    result = {}
    for name in (
        "tokenizers",
        "regex",
        "tree-sitter",
        "tree-sitter-language-pack",
        "markdown-it-py",
        "uniseg",
    ):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--codec", type=Path, required=True, help="Existing reference codec asset or tokenizer JSON"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Fresh directory for input snapshots and receipts",
    )
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--repeats", type=int, default=3, help="Timing repetitions per case (default: 3)"
    )
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if not args.codec.exists():
        parser.error("--codec must be an existing asset path; fetch it with megatoken codec")
    if args.output.exists():
        parser.error("--output must be a fresh directory; previous pilot evidence is retained")
    output, repo = args.output.resolve(), args.repo_root.resolve()
    output.mkdir(parents=True)
    receipt: dict[str, Any] = {
        "version": PILOT_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "repository": str(repo),
        "output": str(output),
        "driver_sha256": sha256_file(__file__),
        "scope": {
            "tested": "Reference-codec content segmentation and serialized-chat framing on nine fixed cases.",
            "excluded": "Training-episode admission, source/quality/dedup gates, GPU throughput, model quality, and general corpus performance.",
            "chat_fixture": "Framing only: no causal tool-call/execution IDs; not a preparation-accepted training episode.",
            "parser_fault_gate": "Classifies content report diagnostics only; does not replace the full preparation adapter and review gate.",
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "dependencies": _dependency_versions(),
        },
        "repeats": args.repeats,
        "timing_method": "Each case runs repeatedly in one process. The first case's first run includes native-worker startup. Warm medians use runs after the first. Encoding, parsing, planning and materialization are timed; asset loading and verification are separate. Shared CPU environment, not a controlled throughput benchmark.",
        "cases": [],
        "passed": False,
    }
    try:
        began = time.perf_counter()
        codec = load_codec(args.codec.resolve())
        receipt["codec"] = {
            "path": str(args.codec.resolve()),
            "fingerprint": codec.fingerprint,
            "vocab_size": codec.vocab_size,
            "lexical_rows": len(codec.valid_token_ids),
            "load_seconds": round(time.perf_counter() - began, 6),
        }
        if args.codec.is_dir():
            receipt["codec"]["asset"] = json.loads((args.codec / "codec.json").read_text())
        else:
            receipt["codec"]["tokenizer_sha256"] = sha256_file(args.codec)
        config = ChunkingConfig()
        identity = config.identity(codec)
        receipt["chunking_identity"] = identity
        for case in _cases(repo):
            snapshot = output / "inputs" / case.name
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            snapshot.write_bytes(case.payload)
            try:
                result, report = _measure_case(case, codec, config, args.repeats)
                if not result["passed"]:
                    failure_path = output / "reports" / (case.name + ".json")
                    write_json(failure_path, report.to_dict())
                    result["failure_report"] = str(failure_path.relative_to(output))
            except Exception as error:
                result = {
                    "case": case.name,
                    "scope": case.scope,
                    "source_sha256": sha256_file(snapshot),
                    "bytes": len(case.payload),
                    "passed": False,
                    "error": f"{type(error).__name__}: {error}",
                }
            result["snapshot"] = str(snapshot.relative_to(output))
            receipt["cases"].append(result)
            print(
                json.dumps(
                    {
                        key: result.get(key)
                        for key in (
                            "case",
                            "passed",
                            "canonical_tokens",
                            "chunks",
                            "content_parser_faults",
                            "error",
                        )
                    }
                ),
                flush=True,
            )
            write_json(output / "receipt.json", receipt)
        # The normal runtime caches its implementation fingerprint. This fresh
        # check catches a developer editing parser files while the pilot runs.
        implementation_fingerprint.cache_clear()
        receipt["unchanged_implementation"] = config.identity(codec) == identity
        results = receipt["cases"]
        lengths = [count for result in results for count in result.get("chunk_token_counts", ())]
        warm = [
            result["timing_seconds"]["warm_median"]
            for result in results
            if result.get("timing_seconds", {}).get("warm_median") is not None
        ]
        receipt["summary"] = {
            "cases": len(results),
            "bytes": sum(result["bytes"] for result in results),
            "canonical_tokens": sum(result.get("canonical_tokens", 0) for result in results),
            "chunks": len(lengths),
            "tokens_per_chunk": _distribution(lengths),
            "planner_candidates": sum(
                result.get("planner", {}).get("candidates", 0) for result in results
            ),
            "planner_edges": sum(
                result.get("planner", {}).get("evaluated_edges", 0) for result in results
            ),
            "eligible_attachments": sum(
                result.get("attachments", {}).get("eligible", 0) for result in results
            ),
            "warm_total_seconds": round(sum(warm), 6) if len(warm) == len(results) else None,
            "failed_cases": [result["case"] for result in results if not result["passed"]],
        }
        receipt["passed"] = receipt["unchanged_implementation"] and all(
            result["passed"] for result in results
        )
    except Exception as error:
        receipt["error"] = f"{type(error).__name__}: {error}"
    write_json(output / "receipt.json", receipt)
    print(
        json.dumps(
            {
                "receipt": str(output / "receipt.json"),
                "passed": receipt["passed"],
                "summary": receipt.get("summary"),
                "error": receipt.get("error"),
            }
        ),
        flush=True,
    )
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
