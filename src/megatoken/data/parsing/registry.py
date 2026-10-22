"""Offline grammar providers and conservative language-hint resolution.

The bundled provider is deliberately pinned: newer language-pack releases perform
implicit downloads from ``get_parser``. Such behaviour must never happen inside a
data worker. Additional locally provisioned grammars use the explicit provider
interface below and participate in the preparation fingerprint.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import os
import re
import threading
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import PurePosixPath
from typing import Any, get_args

BUNDLED_PACK_VERSION = "0.13.0"
REGISTRY_VERSION = "offline-grammar-registry-v1"

# These are hint aliases, not a claim that every alias has an installed grammar.
# Availability is checked against the actual provider before every native parse.
ALIASES = {
    "c++": "cpp",
    "cxx": "cpp",
    "cc": "cpp",
    "c#": "csharp",
    "c_sharp": "csharp",
    "cs": "csharp",
    "objective-c": "objc",
    "objectivec": "objc",
    "objc++": "objc",
    "js": "javascript",
    "jsx": "javascript",
    "node": "javascript",
    "nodejs": "javascript",
    "mjs": "javascript",
    "cjs": "javascript",
    "ts": "typescript",
    "tscript": "typescript",
    "py": "python",
    "python3": "python",
    "python2": "python",
    "pycon": "python",
    "rb": "ruby",
    "rs": "rust",
    "golang": "go",
    "kt": "kotlin",
    "kts": "kotlin",
    "sh": "bash",
    "shell": "bash",
    "shellscript": "bash",
    "zsh": "bash",
    "console": "bash",
    "ps1": "powershell",
    "pwsh": "powershell",
    "f#": "fsharp",
    "fs": "fsharp",
    "elisp": "elisp",
    "emacs-lisp": "elisp",
    "lisp": "commonlisp",
    "cl": "commonlisp",
    "clj": "clojure",
    "cljs": "clojure",
    "cljc": "clojure",
    "edn": "clojure",
    "pl": "perl",
    "pm": "perl",
    "raku": "raku",
    "perl6": "raku",
    "jl": "julia",
    "hs": "haskell",
    "ml": "ocaml",
    "mli": "ocaml_interface",
    "ex": "elixir",
    "exs": "elixir",
    "erl": "erlang",
    "hrl": "erlang",
    "erlastic": "erlang",
    "yml": "yaml",
    "jsonc": "json",
    "json5": "json",
    "jsonl": "json",
    "ndjson": "json",
    "tf": "terraform",
    "protobuf": "proto",
    "protobuf3": "proto",
    "gql": "graphql",
    "postgres": "sql",
    "postgresql": "sql",
    "mysql": "sql",
    "sqlite": "sql",
    "tsql": "sql",
    "md": "markdown",
    "mdown": "markdown",
    "mkd": "markdown",
    "commonmark": "markdown",
    "gfm": "markdown",
    "mdx": "markdown",
    "rmarkdown": "markdown",
    "qmd": "markdown",
    "htm": "html",
    "xhtml": "html",
    "svg": "xml",
    "xsl": "xml",
    "xslt": "xml",
    "tex": "latex",
    "restructuredtext": "rst",
    "rest": "rst",
    "asciidoc": "asciidoc",
    "adoc": "asciidoc",
    "orgmode": "org",
    "text": "prose",
    "plain": "prose",
    "plaintext": "prose",
    "txt": "prose",
    "natural": "prose",
    "natural_language": "prose",
    "makefile": "make",
    "gnumakefile": "make",
    "cmakelists": "cmake",
    "docker": "dockerfile",
    "containerfile": "dockerfile",
    "cu": "cuda",
    "glslang": "glsl",
    "sv": "verilog",
    "systemverilog": "verilog",
    "vhd": "vhdl",
    "asm": "asm",
    "assembly": "asm",
    "dosini": "ini",
    "conf": "ini",
    "cfg": "ini",
    "jinja": "embeddedtemplate",
    "jinja2": "embeddedtemplate",
    "erb": "embeddedtemplate",
    "j2": "embeddedtemplate",
    "ejs": "embeddedtemplate",
    "html+erb": "embeddedtemplate",
    "html+jinja": "embeddedtemplate",
    "bash-session": "bash",
    "shell-session": "bash",
    "obj-c": "objc",
    "ocaml-interface": "ocaml_interface",
    "fennel": "fennel",
    "rkt": "racket",
    "scm": "scheme",
    "sml": "sml",
    "vb": "vb",
    "visualbasic": "vb",
}

_EXTENSION_GROUPS = {
    "python": "py pyi pyw pyx pxd",
    "javascript": "js jsx mjs cjs",
    "typescript": "ts mts cts",
    "tsx": "tsx",
    "c": "c h",
    "cpp": "cc cpp cxx c++ hh hpp hxx h++ ino",
    "csharp": "cs csx",
    "objc": "m mm",
    "java": "java",
    "kotlin": "kt kts",
    "scala": "scala sc",
    "groovy": "groovy gvy gradle",
    "go": "go",
    "rust": "rs",
    "ruby": "rb rake gemspec ru",
    "bash": "sh bash zsh bashrc zshrc",
    "fish": "fish",
    "powershell": "ps1 psm1 psd1",
    "lua": "lua",
    "luau": "luau",
    "perl": "pl pm t",
    "php": "php php3 php4 php5 phtml",
    "hack": "hack hhi",
    "swift": "swift",
    "dart": "dart",
    "elixir": "ex exs",
    "erlang": "erl hrl",
    "haskell": "hs lhs",
    "ocaml": "ml mll mly",
    "ocaml_interface": "mli",
    "fsharp": "fs fsx",
    "fsharp_signature": "fsi",
    "clojure": "clj cljs cljc edn",
    "commonlisp": "lisp lsp cl",
    "elisp": "el",
    "scheme": "scm ss sld",
    "racket": "rkt rktd rktl",
    "julia": "jl",
    "r": "r R",
    "matlab": "matlab",
    "fortran": "f f90 f95 f03 f08 for fpp",
    "cobol": "cob cbl cpy",
    "ada": "ada adb ads",
    "pascal": "pas pp lpr",
    "d": "d di",
    "nim": "nim nims",
    "zig": "zig zon",
    "v": "v vv",
    "odin": "odin",
    "hare": "ha",
    "pony": "pony",
    "elm": "elm",
    "purescript": "purs",
    "fennel": "fnl",
    "janet": "janet",
    "gleam": "gleam",
    "agda": "agda lagda",
    "cairo": "cairo",
    "clarity": "clar",
    "solidity": "sol",
    "gdscript": "gd",
    "haxe": "hx",
    "apex": "cls trigger",
    "actionscript": "as",
    "squirrel": "nut",
    "tcl": "tcl tk",
    "vim": "vim",
    "json": "json jsonc json5 jsonl ndjson ipynb",
    "yaml": "yaml yml",
    "toml": "toml",
    "xml": "xml xsd xsl xslt svg rss atom plist props targets",
    "html": "html htm xhtml",
    "markdown": "md markdown mdown mkd mdx rmd qmd",
    "rst": "rst rest",
    "org": "org",
    "asciidoc": "adoc asciidoc asc",
    "latex": "tex sty cls ltx",
    "bibtex": "bib",
    "typst": "typ",
    "css": "css",
    "scss": "scss sass",
    "less": "less",
    "vue": "vue",
    "svelte": "svelte",
    "astro": "astro",
    "heex": "heex",
    "embeddedtemplate": "erb ejs jinja jinja2 j2",
    "twig": "twig",
    "sql": "sql ddl dml",
    "graphql": "graphql gql",
    "proto": "proto",
    "thrift": "thrift",
    "capnp": "capnp",
    "prisma": "prisma",
    "hcl": "hcl",
    "terraform": "tf tfvars",
    "nix": "nix",
    "bicep": "bicep bicepparam",
    "cmake": "cmake",
    "make": "mk mak make",
    "meson": "meson",
    "ninja": "ninja",
    "starlark": "bzl bazel",
    "dockerfile": "dockerfile",
    "ini": "ini cfg conf editorconfig",
    "properties": "properties",
    "csv": "csv",
    "tsv": "tsv",
    "psv": "psv",
    "kdl": "kdl",
    "jsonnet": "jsonnet libsonnet",
    "ron": "ron",
    "rego": "rego",
    "puppet": "pp",
    "bitbake": "bb bbappend bbclass",
    "verilog": "v sv svh vh",
    "vhdl": "vhd vhdl",
    "glsl": "glsl vert frag geom comp tesc tese",
    "hlsl": "hlsl hlsli fx",
    "wgsl": "wgsl",
    "cuda": "cu cuh",
    "ispc": "ispc",
    "asm": "s asm inc",
    "llvm": "ll",
    "wat": "wat",
    "wast": "wast",
    "tablegen": "td",
    "linkerscript": "ld lds",
    "query": "scm",
    "ungrammar": "ungram",
    "re2c": "re",
    "beancount": "beancount bean",
    "pgn": "pgn",
    "po": "po pot",
    "pem": "pem crt key",
    "mermaid": "mmd mermaid",
    "smali": "smali",
    "smithy": "smithy",
    "yuck": "yuck",
    "bsl": "bsl os",
    "func": "fc",
    "firrtl": "fir",
    "netlinx": "axs axi",
    "qmljs": "qml",
    "gn": "gn gni",
    "magik": "magik",
    "uxntal": "tal",
    "prose": "txt text log nfo license readme",
}
EXTENSIONS = {
    extension.lower(): language
    for language, group in _EXTENSION_GROUPS.items()
    for extension in group.split()
}
# Ambiguous extensions have explicit, stable defaults; a supplied language wins.
EXTENSIONS.update(
    {"h": "c", "m": "objc", "v": "verilog", "pp": "puppet", "scm": "scheme", "cls": "latex"}
)
FILENAMES = {
    "dockerfile": "dockerfile",
    "containerfile": "dockerfile",
    "makefile": "make",
    "gnumakefile": "make",
    "cmakelists.txt": "cmake",
    "meson.build": "meson",
    "meson_options.txt": "meson",
    "build": "starlark",
    "workspace": "starlark",
    "gemfile": "ruby",
    "rakefile": "ruby",
    "vagrantfile": "ruby",
    "guardfile": "ruby",
    "jenkinsfile": "groovy",
    "justfile": "bash",
    "kconfig": "kconfig",
    "go.mod": "gomod",
    "go.sum": "gosum",
    "requirements.txt": "requirements",
    ".gitignore": "gitignore",
    ".gitattributes": "gitattributes",
    ".gitmodules": "ini",
    ".editorconfig": "ini",
    ".bashrc": "bash",
    ".bash_profile": "bash",
    ".zshrc": "bash",
    ".profile": "bash",
    "readme": "prose",
    "license": "prose",
    "copying": "prose",
    "authors": "prose",
    "changelog": "prose",
    "qmldir": "qmldir",
    "hyprland.conf": "hyprlang",
}
MIME_TYPES = {
    "text/plain": "prose",
    "text/markdown": "markdown",
    "text/x-markdown": "markdown",
    "text/html": "html",
    "application/xhtml+xml": "html",
    "text/xml": "xml",
    "application/xml": "xml",
    "image/svg+xml": "xml",
    "application/json": "json",
    "application/ld+json": "json",
    "application/x-ndjson": "json",
    "application/yaml": "yaml",
    "text/yaml": "yaml",
    "application/x-yaml": "yaml",
    "application/toml": "toml",
    "text/toml": "toml",
    "text/css": "css",
    "text/javascript": "javascript",
    "application/javascript": "javascript",
    "application/ecmascript": "javascript",
    "text/ecmascript": "javascript",
    "application/typescript": "typescript",
    "text/typescript": "typescript",
    "text/x-python": "python",
    "application/x-python-code": "python",
    "text/x-shellscript": "bash",
    "application/x-sh": "bash",
    "text/x-rst": "rst",
    "text/x-tex": "latex",
    "application/sql": "sql",
    "text/x-sql": "sql",
    "text/csv": "csv",
    "text/tab-separated-values": "tsv",
    "application/graphql": "graphql",
}


class ParserUnavailable(LookupError):
    """A grammar is not safely loadable without network access."""


@dataclass(frozen=True)
class ParserProvider:
    name: str
    version: str
    languages: tuple[str, ...]
    parser_factory: Callable[[str], Any]
    aliases: Mapping[str, str]
    extensions: Mapping[str, str]
    mime_types: Mapping[str, str]


_providers: dict[str, ParserProvider] = {}
_lock = threading.RLock()


def _reset_after_fork() -> None:
    global _lock
    _lock = threading.RLock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_after_fork)


def register_parser_provider(
    name: str,
    version: str,
    languages: Iterable[str],
    parser_factory: Callable[[str], Any],
    *,
    aliases: Mapping[str, str] | None = None,
    extensions: Mapping[str, str] | None = None,
    mime_types: Mapping[str, str] | None = None,
) -> None:
    """Register pre-provisioned grammars; the factory must be offline and return a fresh parser.

    Registration is process-local and should happen before preparing any data.
    Stable provider names and versions become part of the corpus fingerprint.
    Existing canonical grammars cannot be silently overridden.
    """
    names = tuple(sorted(set(languages)))
    if not name or not version or not names or not callable(parser_factory):
        raise ValueError("Provider name, version, languages, and parser factory are required")
    if any(not re.fullmatch(r"[a-z][a-z0-9_+-]*", item) for item in names):
        raise ValueError("Provider language names must be lowercase canonical identifiers")
    maps = [dict(aliases or {}), dict(extensions or {}), dict(mime_types or {})]
    if any(value not in names for mapping in maps for value in mapping.values()):
        raise ValueError("Provider aliases must resolve to a grammar owned by the provider")
    with _lock:
        owned = set(bundled_languages()) | {
            item for p in _providers.values() for item in p.languages
        }
        if name in _providers or owned.intersection(names):
            raise ValueError("Provider or canonical grammar is already registered")
        _providers[name] = ParserProvider(name, version, names, parser_factory, *maps)


def parser_provider_specs() -> tuple[ParserProvider, ...]:
    """Export local provider configuration for a spawned syntax worker.

    Factories used with process isolation must be importable module-level
    callables. The worker validates picklability before startup. A lambda or local
    closure is supported only by the explicitly selected in-process mode.
    """
    with _lock:
        return tuple(_providers[name] for name in sorted(_providers))


def install_parser_provider_specs(specs: Iterable[ParserProvider]) -> None:
    """Install an exported registry in a fresh worker, preserving its identity."""
    for spec in specs:
        register_parser_provider(
            spec.name,
            spec.version,
            spec.languages,
            spec.parser_factory,
            aliases=spec.aliases,
            extensions=spec.extensions,
            mime_types=spec.mime_types,
        )


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


@lru_cache(maxsize=1)
def _bundled_pack() -> Any:
    version = _version("tree-sitter-language-pack")
    if version != BUNDLED_PACK_VERSION:
        raise ParserUnavailable(
            f"Offline bundled provider requires tree-sitter-language-pack=={BUNDLED_PACK_VERSION}; "
            f"installed={version!r}. Install megatoken[data] or register a local provider."
        )
    try:
        return importlib.import_module("tree_sitter_language_pack")
    except (ImportError, OSError) as exc:
        raise ParserUnavailable(f"Bundled grammar provider cannot be imported: {exc}") from exc


def bundled_languages() -> tuple[str, ...]:
    try:
        return tuple(sorted(get_args(_bundled_pack().SupportedLanguage)))
    except ParserUnavailable:
        return ()


@lru_cache(maxsize=256)
def _bundled_language(language: str) -> Any:
    try:
        return _bundled_pack().get_language(language)
    except (ImportError, LookupError, OSError, ValueError, TypeError) as exc:
        raise ParserUnavailable(f"Cannot load bundled grammar {language!r}: {exc}") from exc


def get_parser(language: str) -> Any:
    """Get an independent native parser, without downloads or shared mutable parse state."""
    canonical = canonical_language(language)
    with _lock:
        for provider in _providers.values():
            if canonical in provider.languages:
                return provider.parser_factory(canonical)
    if canonical not in bundled_languages():
        raise ParserUnavailable(f"No locally provisioned grammar for {canonical!r}")
    from tree_sitter import Parser

    return Parser(_bundled_language(canonical))


def canonical_language(hint: str | None) -> str | None:
    if not hint:
        return None
    value = hint.strip().lower()
    if len(value) > 200:
        return None
    # CommonMark/Pandoc/Quarto fence identifiers and editor language hints.
    value = value.strip("{} ")
    if value.startswith("."):
        value = value[1:]
    value = value.split()[0] if value else ""
    for prefix in ("language-", "lang-", "source."):
        if value.startswith(prefix):
            value = value[len(prefix) :]
            break
    value = value.split(";", 1)[0]
    with _lock:
        for provider in _providers.values():
            if value in provider.aliases:
                return provider.aliases[value]
            if value in provider.mime_types:
                return provider.mime_types[value]
    if value in MIME_TYPES:
        return MIME_TYPES[value]
    if value.startswith("text/x-"):
        value = value[7:]
    elif value.startswith("application/x-"):
        value = value[14:]
    return ALIASES.get(value, value) or None


def resolve_language(
    language: str | None = None,
    filename: str | None = None,
    mime_type: str | None = None,
    payload: bytes = b"",
) -> str | None:
    """Prefer explicit language, filename, then MIME; content guessing uses strong signatures.

    Unknown explicit languages are retained so unavailable-grammar diagnostics are
    visible. Unlabelled prose is not classified as code merely for punctuation.
    """
    if language and language.lower() not in {"auto", "unknown", "code"}:
        return canonical_language(language)
    if filename:
        name = PurePosixPath(filename.replace("\\", "/")).name.lower()
        if name in FILENAMES:
            return FILENAMES[name]
        if name.startswith(("dockerfile.", "containerfile.")):
            return "dockerfile"
        ext = name.rsplit(".", 1)[-1] if "." in name else ""
        with _lock:
            for provider in _providers.values():
                if ext in provider.extensions:
                    return provider.extensions[ext]
        if ext in EXTENSIONS:
            return EXTENSIONS[ext]
    if mime_type:
        mime = mime_type.split(";", 1)[0].strip().lower()
        with _lock:
            for provider in _providers.values():
                if mime in provider.mime_types:
                    return provider.mime_types[mime]
        if mime in MIME_TYPES:
            return MIME_TYPES[mime]
        if mime.endswith("+json"):
            return "json"
        if mime.endswith("+xml"):
            return "xml"
        if mime.startswith(("text/x-", "application/x-")):
            candidate = canonical_language(mime)
            if candidate in bundled_languages():
                return candidate
    return detect_content_language(payload)


_TOP_LEVEL_PYTHON = re.compile(
    rb"(?m)^(?:from\s+[\w.]+\s+import\s+\S|import\s+[\w.]+(?:\s+as\s+\w+)?\s*$"
    rb"|(?:async\s+)?def\s+\w+\s*\([^\n]*\)[^\n]*:\s*$|class\s+\w+[^\n]*:\s*$)"
)


def detect_content_language(payload: bytes) -> str | None:
    """Bounded, high-confidence format detection; ordinary prose is the fallback."""
    head = payload[:8192].lstrip(b"\xef\xbb\xbf \t\r\n")
    if head.startswith(b"#!"):
        line = head.splitlines()[0].decode("ascii", "ignore")
        interpreters = {
            "python": "python",
            "pypy": "python",
            "ruby": "ruby",
            "node": "javascript",
            "deno": "typescript",
            "bash": "bash",
            "dash": "bash",
            "zsh": "bash",
            "sh": "bash",
            "perl": "perl",
            "lua": "lua",
            "fish": "fish",
            "pwsh": "powershell",
            "Rscript": "r",
        }
        for token in line[2:].split():
            name = token.rsplit("/", 1)[-1]
            base = re.sub(r"[0-9.]+$", "", name)
            if base in interpreters:
                return interpreters[base]
    if re.match(
        rb"(?is)(?:<!doctype\s+html|<html(?:\s|>)|<(?:div|p|section|article|script|style|body|head|main|h[1-6])(?:\s|>))",
        head,
    ):
        return "html"
    if head.startswith(b"<?xml"):
        return "xml"
    if re.match(rb'(?s)^[\[{]\s*(?:[\[{"]|[-0-9]|true\b|false\b|null\b|[\]}])', head):
        return "json"
    markdown = re.search(
        rb"(?m)^[ \t]{0,3}(?:`{3,}|~{3,}|#{1,6}\s|>\s|[-*+]\s|[0-9]{1,9}[.)]\s)", head
    )
    if markdown:
        # "# comment" opens Python and shell as readily as it opens a Markdown
        # heading. Top-level Python before any fence settles which it was.
        code = _TOP_LEVEL_PYTHON.search(head)
        fence = re.search(rb"(?m)^[ \t]{0,3}(?:`{3,}|~{3,})", head)
        if (
            code
            and markdown.group().lstrip().startswith(b"#")
            and (not fence or code.start() < fence.start())
        ):
            return "python"
        return "markdown"
    if head.startswith((b"---\n", b"---\r\n")) and re.search(rb"(?m)^[A-Za-z_][\w-]*:\s", head):
        return "yaml"
    if re.match(
        rb"(?m)^(?:from\s+[\w.]+\s+import\s|(?:async\s+)?def\s+\w+\s*\(|class\s+\w+[^\n]*:\s*$)",
        head,
    ):
        return "python"
    if re.match(rb'(?s)^(?:#include\s*[<"]|(?:package\s+[\w.]+;?\s+)?import\s+java\.)', head):
        return "c" if head.startswith(b"#include") else "java"
    if re.match(rb"(?:pub\s+)?(?:async\s+)?fn\s+\w+\s*\(", head):
        return "rust"
    if re.match(rb"package\s+\w+\s*\r?\n", head) and b"func " in head:
        return "go"
    if re.match(
        rb"(?:(?:export\s+)?(?:async\s+)?function\s+\w+\s*\(|(?:const|let|var)\s+\w+\s*=)", head
    ):
        return "javascript"
    return None


def parser_runtime_identity(backend: str = "tree_sitter") -> dict[str, Any]:
    with _lock:
        providers = [
            {
                "name": p.name,
                "version": p.version,
                "languages": list(p.languages),
                "aliases": dict(sorted(p.aliases.items())),
                "extensions": dict(sorted(p.extensions.items())),
                "mime_types": dict(sorted(p.mime_types.items())),
            }
            for p in sorted(_providers.values(), key=lambda p: p.name)
        ]
    return {
        "registry": REGISTRY_VERSION,
        "backend": backend,
        "network": "disabled",
        "tree_sitter": _version("tree-sitter"),
        "tree_sitter_language_pack": _version("tree-sitter-language-pack"),
        "grammar_dependencies": {
            name: _version(name)
            for name in ("tree-sitter-c-sharp", "tree-sitter-yaml", "tree-sitter-embedded-template")
        },
        "markdown_it_py": _version("markdown-it-py"),
        "providers": providers,
    }


def coverage_report(*, validate: bool = True) -> dict[str, Any]:
    """Report installed grammars, optionally loading each binding to verify actual availability."""
    with _lock:
        candidates = sorted(
            set(bundled_languages()) | {n for p in _providers.values() for n in p.languages}
        )
    available, failures = [], {}
    for language in candidates:
        try:
            if validate:
                get_parser(language)
            available.append(language)
        except Exception as exc:
            failures[language] = f"{type(exc).__name__}: {exc}"
    return {
        **parser_runtime_identity(),
        "validated": validate,
        "grammar_count": len(available),
        "grammars": available,
        "failures": failures,
        "aliases": dict(sorted(ALIASES.items())),
        "extensions": dict(sorted(EXTENSIONS.items())),
        "fallback": "bounded lexical structure and Unicode prose segmentation",
    }
