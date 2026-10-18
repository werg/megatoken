"""Cross-language parser integration and exact nested-source coordinate contracts."""

from __future__ import annotations

from dataclasses import replace

import pytest

from megatoken.data.parsing import registry
from megatoken.data.parsing.documents import scan_markdown, scan_markup_fallback
from megatoken.data.parsing.engine import SourceView, parse_content
from megatoken.data.parsing.syntax import scan_syntax
from megatoken.data.parsing.types import ParseLimits

LIMITS = ParseLimits(isolate_native=False)


def parse(payload: bytes, **kwargs):
    return parse_content(payload, limits=LIMITS, **kwargs)


def test_bundled_catalog_is_actually_loadable_without_network(monkeypatch):
    import socket

    def offline(*args, **kwargs):
        raise AssertionError("Parser loading attempted network access")

    monkeypatch.setattr(socket, "create_connection", offline)
    report = registry.coverage_report()
    assert report["grammar_count"] == 173
    assert report["failures"] == {}
    assert report["network"] == "disabled"
    assert report["tree_sitter_language_pack"] == "0.13.0"
    assert {"python", "java", "csharp", "yaml", "markdown", "html", "rust", "cobol"} <= set(
        report["grammars"]
    )


@pytest.mark.parametrize(
    ("hints", "expected"),
    [
        ({"language": "{.python}"}, "python"),
        ({"language": "language-C++"}, "cpp"),
        ({"language": "source.js"}, "javascript"),
        ({"language": "C#"}, "csharp"),
        ({"language": "text/x-python"}, "python"),
        ({"filename": "/src/App.tsx"}, "tsx"),
        ({"filename": r"C:\src\Dockerfile.release"}, "dockerfile"),
        ({"filename": "CMakeLists.txt"}, "cmake"),
        ({"filename": "go.mod"}, "gomod"),
        ({"filename": "src/component.vue"}, "vue"),
        ({"mime_type": "application/ld+json; charset=utf-8"}, "json"),
        ({"mime_type": "application/vnd.example+json"}, "json"),
        ({"mime_type": "image/svg+xml"}, "xml"),
        ({"payload": b"#!/usr/bin/env -S python3.12 -u\nprint(1)\n"}, "python"),
        ({"payload": b"Text.\n\n```rust\nfn main() {}\n```\n"}, "markdown"),
        ({"language": "newlang", "filename": "script.py"}, "newlang"),
        ({"payload": "هذا نص طبيعي. It contains {braces}, version 1.2, and x = y.".encode()}, None),
    ],
)
def test_language_resolution_is_explicit_first_and_conservative(hints, expected):
    assert registry.resolve_language(**hints) == expected


@pytest.mark.parametrize(
    ("language", "source"),
    [
        ("python", 'def greet():\n    return "Hello, 世界."\n'),
        ("javascript", 'function greet() { return "Hello, 世界."; }'),
        ("typescript", 'function greet(): string { return "Hello."; }'),
        ("tsx", "const View = () => <div>Hello, 世界.</div>;"),
        ("java", 'class A { String greet() { return "Hello."; } }'),
        ("csharp", 'class A { string Greet() { return "Hello."; } }'),
        ("cpp", 'const char* greet() { return "Hello."; }'),
        ("c", 'const char* greet(void) { return "Hello."; }'),
        ("rust", 'fn greet() -> String { String::from("Hello.") }'),
        ("go", 'package main\nfunc greet() string { return "Hello." }'),
        ("ruby", 'def greet\n  "Hello."\nend\n'),
        ("bash", 'greet() { printf "Hello."; }\n'),
        ("kotlin", 'fun greet(): String { return "Hello." }'),
        ("swift", 'func greet() -> String { return "Hello." }'),
        ("php", '<?php function greet() { return "Hello."; }'),
        ("lua", 'function greet() return "Hello." end'),
        ("julia", 'function greet()\n  return "Hello."\nend'),
        ("elixir", 'defmodule A do\n def greet, do: "Hello."\nend'),
        ("erlang", 'greet() -> "Hello.".'),
        ("haskell", 'greet :: String\ngreet = "Hello."\n'),
        ("clojure", '(defn greet [] "Hello.")'),
        ("scala", 'def greet(): String = "Hello."'),
        ("ocaml", 'let greet () = "Hello."'),
        ("r", 'greet <- function() { "Hello." }'),
        ("perl", 'sub greet { return "Hello."; }'),
        ("zig", 'fn greet() []const u8 { return "Hello."; }'),
        ("dart", 'String greet() { return "Hello."; }'),
        ("fortran", 'program hello\n print *, "Hello."\nend program hello\n'),
        ("sql", "SELECT 'Hello.' AS greeting;"),
        ("yaml", 'greeting: "Hello."\nitems:\n  - one\n  - two\n'),
        ("toml", '[greeting]\ntext = "Hello."\n'),
        ("json", '{"greeting": "Hello.", "items": [1, 2]}'),
        ("html", "<section><p>Hello.</p></section>"),
        ("xml", "<root><message>Hello.</message></root>"),
        ("css", ".greeting { color: red; }"),
    ],
)
def test_native_grammars_produce_structural_evidence_across_language_families(language, source):
    payload = source.encode()
    result = parse(payload, language=language)
    result.validate(len(payload))
    assert f"tree_sitter:{language}" in result.parsers
    assert result.regions
    assert not any(
        item.code in {"grammar_unavailable", "native_parser_error"} for item in result.diagnostics
    )


def test_markdown_to_python_docstring_to_html_to_javascript_to_prose():
    payload = b'''# Example\n\n```python\ndef page():\n    """<div>Hello. <script>const m = "Good morning. Welcome!";</script></div>"""\n    return 1\n```\n'''
    result = parse(payload, language="markdown", needs_refinement=lambda start, end: True)
    assert {"markdown", "python", "html", "javascript", "prose"} <= result.languages
    assert not result.diagnostics
    text = b"Good morning. Welcome!"
    start = payload.index(text)
    assert any(
        region.start == start and region.end == start + len(text) and region.kind == "paragraph"
        for region in result.regions
    )
    assert any(
        boundary.offset == start + len(b"Good morning. ") and boundary.kind == "sentence"
        for boundary in result.boundaries
    )


def test_markdown_nested_quote_list_fence_uses_disjoint_original_ranges():
    payload = b'> - ```python\r\n>   def greet():\r\n>       return "Hello world."\r\n>   ```\r\n'
    result = parse(payload, language="markdown")
    functions = [
        region
        for region in result.regions
        if region.kind == "function" and region.language == "python"
    ]
    assert functions
    region = functions[0]
    assert payload[region.start : region.start + len(b"def greet")] == b"def greet"
    assert payload[region.end - 1 : region.end] == b'"'
    assert b">       " in payload[region.start : region.end]
    assert not any(
        item.code == "recovered_syntax" and item.parser == "tree_sitter:python"
        for item in result.diagnostics
    )


@pytest.mark.parametrize(
    ("language", "payload", "expected"),
    [
        ("javascript", b"const query = sql`SELECT 1;`;", "sql"),
        ("javascript", b"const markup = html`<p>Hello.</p>`;", "html"),
        ("cpp", b'const char *s = R"sql(SELECT 1;)sql";', "sql"),
        ("bash", b"cat <<HTML\n<div>Hello.</div>\nHTML\n", "html"),
        ("ruby", b"x = <<~HTML\n<div>Hello.</div>\nHTML\n", "html"),
        ("php", b"<?php $x = <<<HTML\n<div>Hello.</div>\nHTML;", "html"),
        ("lua", b"local s = [[# Guide\n\n```python\nx=1\n```\n]]", "python"),
        ("yaml", b"text: |\n  # Guide\n\n  ```python\n  x=1\n  ```\n", "python"),
        ("python", b'# language=sql\nquery = "SELECT 1;"\n', "sql"),
    ],
)
def test_literal_and_heredoc_injection_hints(language, payload, expected):
    result = parse(payload, language=language)
    assert f"tree_sitter:{expected}" in result.parsers


def test_html_script_style_attributes_and_multilingual_text():
    payload = """<section title="Bonjour. Bienvenue!" onclick="alert('Hello.')">
<p>你好。世界！</p><script type="application/ld+json">{"text":"مرحبا. أهلا."}</script>
<script lang="ts">const x: number = 1;</script><style lang="scss">$c: red; p {color: $c;}</style></section>""".encode()
    result = parse(payload, language="html", needs_refinement=lambda start, end: True)
    assert {"html", "javascript", "json", "typescript", "scss", "prose"} <= result.languages
    assert any(
        payload[region.start : region.end] == "你好。世界！".encode()
        for region in result.regions
        if region.kind == "paragraph"
    )
    assert any(
        payload[region.start : region.end] == b"Bonjour. Bienvenue!"
        for region in result.regions
        if region.kind == "paragraph"
    )


@pytest.mark.parametrize(
    ("language", "payload"),
    [
        (
            "rst",
            b'Title\n=====\n\n.. code-block:: python\n   :linenos:\n\n   def greet():\n       return "Hello."\n',
        ),
        ("org", b'* Title\n\n#+BEGIN_SRC python\ndef greet():\n    return "Hello."\n#+END_SRC\n'),
        (
            "asciidoc",
            b'= Title\n\n[source,python]\n----\ndef greet():\n    return "Hello."\n----\n',
        ),
        (
            "latex",
            b'\\section{Title}\n\\begin{minted}{python}\ndef greet():\n    return "Hello."\n\\end{minted}',
        ),
        (
            "latex",
            b'\\begin{lstlisting}[language=Python]\ndef greet():\n    return "Hello."\n\\end{lstlisting}',
        ),
    ],
)
def test_other_document_containers_recurse_into_code(language, payload):
    result = parse(payload, language=language)
    assert "tree_sitter:python" in result.parsers
    assert any(
        region.kind == "function" and region.language == "python" for region in result.regions
    )


def test_headers_and_function_signatures_are_attached_to_first_body():
    markdown = b"# Heading\n\nFirst paragraph.\n\nSecond paragraph.\n"
    result = parse(markdown, language="markdown")
    assert any(
        region.kind == "heading_attachment"
        and region.start == 0
        and b"First paragraph." in markdown[region.start : region.end]
        for region in result.regions
    )
    python = b'def greet():\n    """Help text."""\n    return 1\n'
    result = parse(python, language="python")
    attachment = next(region for region in result.regions if region.kind == "signature_attachment")
    assert python[attachment.start : attachment.end] == b'def greet():\n    """Help text."""'
    function = next(region for region in result.regions if region.kind == "function")
    assert function.path
    assert attachment.path == function.path


def test_recovered_syntax_remains_useful_and_unknown_grammar_is_honest():
    broken = b'def broken(:\n    return "Hello. World!"\n\ndef good():\n    return 2\n'
    result = parse(broken, language="python")
    assert "tree_sitter:python" in result.parsers
    assert any(item.code == "recovered_syntax" for item in result.diagnostics)
    assert any(region.kind == "string" for region in result.regions)
    unknown = parse(b'function thing { a = "Hello. World!"; }\n', language="futurelang")
    assert "generic_lexical:v1" in unknown.parsers
    assert any(item.code == "grammar_unavailable" for item in unknown.diagnostics)
    assert any(region.kind == "paragraph" for region in unknown.regions)


def test_string_escapes_and_html_entities_never_change_source_coordinates():
    payload = rb'const x = "Hello.\nWorld! \u4f60\u597d";'
    result = parse(payload, language="javascript", needs_refinement=lambda start, end: True)
    string = next(region for region in result.regions if region.kind == "string")
    assert payload[string.start : string.end] == rb'"Hello.\nWorld! \u4f60\u597d"'
    markup = b"<p>Fish &amp; chips. &#x4f60;&#x597d;!</p>"
    result = parse(markup, language="html")
    assert all(region.end <= len(markup) for region in result.regions)
    assert any(
        markup[region.start : region.end] == b"Fish &amp; chips."
        for region in result.regions
        if region.kind == "paragraph"
    ) or any(
        b"&amp;" in markup[region.start : region.end]
        for region in result.regions
        if region.kind == "paragraph"
    )


@pytest.mark.parametrize("payload", [b"", b"\xff\xfe\x00hello", b"\r\r\n", b"<", b"`" * 4096])
def test_arbitrary_bytes_are_bounded_and_offset_valid(payload):
    result = parse_content(
        payload,
        limits=replace(
            LIMITS, max_nodes=1000, max_regions=100, max_parse_bytes=8192, max_injections=30
        ),
    )
    result.validate(len(payload))
    assert len(result.regions) <= 100
    assert len(result.boundaries) <= 400


def test_shared_depth_byte_and_injection_budgets_emit_diagnostics():
    payload = b'```python\nx = "<p>Hello world.</p>"\n```\n'
    shallow = parse_content(payload, language="markdown", limits=replace(LIMITS, max_depth=1))
    assert any(item.code == "depth_budget_exhausted" for item in shallow.diagnostics)
    short = parse_content(payload, language="markdown", limits=replace(LIMITS, max_parse_bytes=16))
    assert any(item.code == "parse_byte_budget_exhausted" for item in short.diagnostics)
    limited = parse_content(
        b'x = "a"; y = "b"; z = "c";',
        language="javascript",
        limits=replace(LIMITS, max_injections=1),
    )
    assert any(item.code == "injection_budget_exhausted" for item in limited.diagnostics)


def test_native_tree_depth_budget_prunes_descendants_and_keeps_later_siblings():
    from megatoken.data.chunking import ChunkingConfig, segment_with_report
    from megatoken.loading import load_codec

    tail = b"<p>Retained sibling.</p>"
    payload = b"<div>" * 40 + b"Deep content." + b"</div>" * 40 + tail
    limits = replace(LIMITS, max_tree_depth=6)
    scan = scan_syntax(payload, "html", limits)
    full_scan = scan_syntax(payload, "html", LIMITS)
    assert scan.nodes < full_scan.nodes
    assert all(region.depth <= 6 and len(region.path) <= 7 for region in scan.result.regions)
    assert any(
        region.start == len(payload) - len(tail) and region.end == len(payload)
        for region in scan.result.regions
    )
    pruned = [
        item for item in scan.result.diagnostics if item.code == "tree_depth_budget_exhausted"
    ]
    assert len(pruned) == 1
    assert "max_tree_depth=6" in pruned[0].detail
    assert not any(
        item.code == "tree_depth_budget_exhausted" for item in full_scan.result.diagnostics
    )

    policy = ChunkingConfig(limits=limits)
    report = segment_with_report(payload, load_codec("byte"), policy, content_kind="html")
    assert b"".join(chunk.payload for chunk in report.chunks) == payload
    assert all(len(chunk.token_ids) <= policy.hard_max_tokens for chunk in report.chunks)
    assert any(item.code == "tree_depth_budget_exhausted" for item in report.parse.diagnostics)


def test_source_views_compose_without_decoding_or_dropping_original_coordinates():
    payload = b"> abc\r\n> def\r\n"
    view = SourceView.original(payload).subview(((2, 7), (9, 14)))
    assert view.data == b"abc\r\ndef\r\n"
    assert view.position(5) == 9
    assert view.position(5, end=True) == 7
    child = view.subview(((1, 8),))
    assert child.data == b"bc\r\ndef"
    assert child.position(0) == 3
    assert child.position(7, end=True) == 12
    with pytest.raises(ValueError, match="sorted, disjoint"):
        view.subview(((2, 5), (1, 3)))


def test_commonmark_fallback_maps_cr_only_fences_and_recovers_unclosed_blocks():
    payload = b'Intro.\r\r```python\rx = "Hello."\r'
    scan = scan_markdown(payload, LIMITS)
    scan.result.validate(len(payload))
    assert any(item.code == "unclosed_fence" for item in scan.result.diagnostics)
    injection = next(item for item in scan.injections if item.language == "python")
    assert b'x = "Hello."' in payload[injection.start : injection.end]


def test_markup_fallback_is_quote_aware_and_does_not_execute_entities():
    payload = b'<div title="a > b"><script>const s="Hello.";</script><p>World.</p></div>'
    scan = scan_markup_fallback(payload, "html", LIMITS)
    scan.result.validate(len(payload))
    tag = next(region for region in scan.result.regions if region.kind == "tag")
    assert payload[tag.start : tag.end] == b'<div title="a > b">'
    script = next(item for item in scan.injections if item.language == "javascript")
    assert payload[script.start : script.end] == b'const s="Hello.";'


def test_provider_extension_is_explicit_and_fingerprinted(monkeypatch):
    monkeypatch.setattr(registry, "_providers", {})
    registry.register_parser_provider(
        "test-local",
        "v1",
        ["testlang"],
        lambda language: registry.get_parser("python"),
        aliases={"test-lang": "testlang"},
        extensions={"testsrc": "testlang"},
    )
    assert registry.resolve_language(filename="source.testsrc") == "testlang"
    assert registry.resolve_language(language="test-lang") == "testlang"
    identity = registry.parser_runtime_identity()
    assert identity["providers"][0]["name"] == "test-local"
    result = scan_syntax(b'def f():\n return "Hello."\n', "testlang", LIMITS)
    assert "tree_sitter:testlang" in result.result.parsers
    with pytest.raises(ValueError, match="already registered"):
        registry.register_parser_provider("test-local", "v2", ["otherlang"], lambda language: None)


def test_untrusted_future_pack_api_is_never_called(monkeypatch):
    registry._bundled_pack.cache_clear()
    original = registry._version
    monkeypatch.setattr(
        registry,
        "_version",
        lambda name: "1.22.1" if name == "tree-sitter-language-pack" else original(name),
    )
    try:
        with pytest.raises(registry.ParserUnavailable):
            registry.get_parser("python")
    finally:
        registry._bundled_pack.cache_clear()


@pytest.mark.parametrize(
    ("language", "payload", "nested"),
    [
        (
            "cpp",
            b'// Example:\n// ```python\n// def f():\n//     return "Hello."\n// ```\nint main() { return 0; }\n',
            "python",
        ),
        (
            "python",
            b'def f():\n    """Doc.\n\n    ```javascript\n    const x = html`<p>Hello.</p>`;\n    ```\n    """\n    return 1\n',
            "javascript",
        ),
        ("markdown", b'```python\ndef f():\n    return "Hello."\n```', "python"),
        ("xml", b"<root><![CDATA[<div>Hello.</div>]]></root>", "html"),
        (
            "vue",
            b'<template><p>Hello {{ name }}!</p></template><script lang="ts">const name: string = "world";</script>',
            "typescript",
        ),
        (
            "svelte",
            b'<script lang="ts">let name: string = "world";</script><p>Hello {name}!</p>',
            "javascript",
        ),
        ("astro", b'---\nconst name: string = "world";\n---\n<p>Hello {name}!</p>', "typescript"),
        ("twig", b'<p>Hello {{ name }}!</p>{% set message = "Good morning." %}', "html"),
    ],
)
def test_source_view_regressions_for_comments_docstrings_and_templates(language, payload, nested):
    result = parse(payload, language=language)
    assert f"tree_sitter:{nested}" in result.parsers
    assert not any(
        item.code in {"syntax_error", "low_syntax_coverage", "invalid_injection_range"}
        and item.parser == f"tree_sitter:{nested}"
        for item in result.diagnostics
    )


@pytest.mark.parametrize(
    ("filename", "source", "expected"),
    [
        ("page.ejs", b'<p>Hello <%= name %>!</p><% const m = "Hello."; %>', "javascript"),
        ("page.erb", b'<p>Hello <%= name %>!</p><% m = "Hello." %>', "ruby"),
    ],
)
def test_template_filename_controls_embedded_code_dialect(filename, source, expected):
    result = parse(source, filename=filename)
    assert {"embeddedtemplate", "html", expected, "prose"} <= result.languages
    assert not any(item.code == "low_syntax_coverage" for item in result.diagnostics)


def test_native_worker_failure_keeps_observable_generic_fallback(monkeypatch):
    from megatoken.data.parsing import workers
    from megatoken.data.parsing.syntax import Scan
    from megatoken.data.parsing.types import Diagnostic, ParseResult

    def failed(payload, language, limits, backend):
        return Scan(
            ParseResult(diagnostics=[Diagnostic("native_parser_crash", 0, len(payload), "test")])
        )

    monkeypatch.setattr(workers, "scan_syntax_isolated", failed)
    result = parse_content(b'def f():\n    return "Hello."\n', language="python")
    assert "generic_lexical:v1" in result.parsers
    assert any(item.code == "native_parser_crash" for item in result.diagnostics)
    assert any(region.kind == "string" for region in result.regions)


def test_almost_total_syntax_failure_is_distinct_from_local_recovery():
    unreadable = parse(b"@@@@ ???? \\ \\ %%% }}} {{{", language="python")
    assert any(item.code == "low_syntax_coverage" for item in unreadable.diagnostics)
    mostly_valid = parse(
        b'def f():\n    return "Hello."\n\ndef g():\n    return 1\n\n@\n', language="python"
    )
    assert any(item.code == "recovered_syntax" for item in mostly_valid.diagnostics)
    assert not any(item.code == "low_syntax_coverage" for item in mostly_valid.diagnostics)


@pytest.mark.parametrize("locale", ["en-US", "fr", "zh-Hant-TW", "ar", "ja-JP"])
def test_explicit_prose_accepts_recognized_bcp47_language_hints(locale):
    result = parse("Hello. 你好。مرحبا.".encode(), content_kind="text", language=locale)
    assert "prose" in result.languages
    assert not any(item.code == "grammar_unavailable" for item in result.diagnostics)
    code = parse(b"something()", content_kind="code", language=locale)
    assert any(item.code == "grammar_unavailable" for item in code.diagnostics)


def test_empty_cdata_and_non_markup_angles_do_not_create_invalid_views():
    result = parse_content(
        b"a < b < c\n<![CDATA[]]>", language="html", limits=LIMITS, backend="fallback"
    )
    assert not any(item.code == "invalid_injection_range" for item in result.diagnostics)
    result.validate(len(b"a < b < c\n<![CDATA[]]>"))


def test_empty_comment_lines_preserve_a_single_embedded_code_block():
    source = b'// ```python\n// def f():\n//\n//     return "Hello."\n// ```\n'
    result = parse(source, language="cpp")
    assert "tree_sitter:python" in result.parsers
    assert not any(
        item.code in {"syntax_error", "unclosed_fence", "low_syntax_coverage"}
        for item in result.diagnostics
    )
