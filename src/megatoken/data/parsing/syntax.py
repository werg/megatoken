"""Recovering syntax-tree analysis and a bounded, language-agnostic fallback.

Only byte-coordinate evidence leaves this module. Native nodes, source text
normalisation, evaluation, and parser-specific trees never enter the chunk planner.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from typing import Any

from .registry import ParserUnavailable, canonical_language, detect_content_language, get_parser
from .types import Boundary, Diagnostic, ParseLimits, ParseResult, Region


@dataclass(frozen=True, slots=True)
class Injection:
    start: int
    end: int
    kind: str = "auto"
    language: str | None = None
    source: str = "injection"
    # Disjoint source views remove structural comment/blockquote prefixes only.
    # Escapes and entities are never decoded, and each resulting byte is mapped
    # back to its original source coordinate by the recursive engine.
    ranges: tuple[tuple[int, int], ...] = ()


@dataclass
class Scan:
    result: ParseResult = field(default_factory=ParseResult)
    injections: list[Injection] = field(default_factory=list)
    nodes: int = 0


_STRING_TYPES = {
    "string",
    "string_literal",
    "raw_string_literal",
    "interpreted_string_literal",
    "template_string",
    "template_literal",
    "concatenated_string",
    "encapsed_string",
    "quoted_string",
    "long_string",
    "long_string_literal",
    "raw_string",
    "verbatim_string_literal",
    "multiline_string_literal",
    "multiline_string",
    "triple_quoted_string",
    "quoted_atom",
    "heredoc_body",
    "heredoc_content",
    "heredoc",
    "nowdoc",
    "nowdoc_body",
    "block_scalar",
    "double_quote_scalar",
    "single_quote_scalar",
    "string_scalar",
    "string_expression",
    "string_literal_expression",
    "string_content_expression",
    "r_string_literal",
    "character_string",
    "line_string_literal",
    "interpolated_string_literal",
    "str_lit",
    "STRINGLITERALSINGLE",
    "text_block",
    "quoted_attribute_value",
}
_CONTENT_TYPES = {
    "string_content",
    "string_fragment",
    "raw_string_content",
    "heredoc_content",
    "string_value",
    "string_text",
    "string_body",
    "text_fragment",
    "raw_text",
    "line_str_text",
}
_PROSE_TYPES = {"text", "CharData", "char_data", "jsx_text", "cdata_text", "text_node"}
_TEXT_ENTITY_TYPES = _PROSE_TYPES | {
    "entity",
    "character_reference",
    "entity_reference",
    "Reference",
    "EntityRef",
    "CharRef",
}
_MARKUP_LANGUAGES = {"html", "xml", "vue", "svelte", "astro", "heex", "embeddedtemplate", "twig"}
_DOCUMENT_LANGUAGES = {"markdown", "markdown_inline", "rst", "org", "latex", "typst"}
_LANGUAGE_DIRECTIVE = re.compile(
    rb"(?i)(?:language|lang|syntax)\s*(?:=|:)\s*[\"']?([a-z0-9_+#.-]+)"
)
_TAG_LANGUAGES = {
    "sql": "sql",
    "gql": "graphql",
    "graphql": "graphql",
    "html": "html",
    "svg": "xml",
    "css": "css",
    "scss": "scss",
    "js": "javascript",
    "javascript": "javascript",
    "json": "json",
    "yaml": "yaml",
    "xml": "xml",
    "md": "markdown",
    "markdown": "markdown",
    "sh": "bash",
    "bash": "bash",
    "python": "python",
    "py": "python",
    "rust": "rust",
    "lua": "lua",
    "ruby": "ruby",
    "toml": "toml",
    "ts": "typescript",
}


def _is_string_node(node: Any, payload: bytes) -> bool:
    if node.type in _STRING_TYPES:
        return True
    name = node.type.lower()
    if name in _CONTENT_TYPES | {"info_string"} or any(
        part in name
        for part in (
            "string_type",
            "string_start",
            "string_end",
            "string_delimiter",
            "string_format",
            "interpolation",
            "string_parameter",
            "string_escape",
        )
    ):
        return False
    if "string" in name and ("literal" in name or name.endswith(("_string", "_string_expression"))):
        return True
    if name in {"literal", "str", "str_literal", "character_literal"}:
        return payload[node.start_byte : node.start_byte + 1] in {b"'", b'"', b"`"}
    return False


def _literal_ancestor(node: Any, payload: bytes) -> bool:
    parent = node.parent
    for _ in range(128):
        if parent is None:
            return False
        if _is_string_node(parent, payload):
            return True
        parent = parent.parent
    return False


def _walk(root: Any, maximum: int, max_depth: int):
    """Bound traversal and ancestry before constructing evidence for IPC.

    The last flag reports a subtree whose descendants were deliberately pruned.
    Its siblings are still visited, preserving useful evidence after a deeply
    nested construct. No child list or recursive Python stack is materialized.
    """
    cursor = root.walk()
    depth, visited = 0, 0
    while visited < maximum:
        node = cursor.node
        visited += 1
        pruned = depth >= max_depth and node.child_count > 0
        yield node, depth, pruned
        if not pruned and cursor.goto_first_child():
            depth += 1
            continue
        while not cursor.goto_next_sibling():
            if not cursor.goto_parent():
                return
            depth -= 1


def _kind(node: Any, language: str, depth: int, payload: bytes) -> str | None:
    name = node.type.lower()
    if "comment" in name:
        return "comment"
    if _is_string_node(node, payload) or name in {"code_span", "inline_code"}:
        return "string" if name not in {"code_span", "inline_code"} else "inline_code"
    if name in {"fenced_code_block", "indented_code_block", "literal_block", "code_block"}:
        return "code_block"
    if name in {"list_item", "item", "listitem"} or name.endswith("list_item"):
        return "list_item"
    if "heading" in name or name in {"title", "section_heading", "section_title"}:
        return "heading"
    if name in {"paragraph", "text_block", "prose"}:
        return "paragraph"
    if name in {"section", "subsection", "chapter"}:
        return "section"
    if name in {"element", "script_element", "style_element", "jsx_element", "html_block"}:
        return "element"
    if name in {
        "start_tag",
        "end_tag",
        "self_closing_tag",
        "stag",
        "etag",
        "emptyelemtag",
        "jsx_opening_element",
        "jsx_closing_element",
        "jsx_self_closing_element",
    }:
        return "tag"
    if (
        "function" in name
        or "method" in name
        or name in {"constructor_declaration", "decorated_definition"}
    ):
        if not any(part in name for part in ("call", "type", "name", "pointer", "parameter")):
            return "function"
    if any(
        part in name
        for part in (
            "class_",
            "struct_",
            "interface_",
            "enum_",
            "module_",
            "namespace_",
            "trait_",
            "impl_",
        )
    ):
        if any(part in name for part in ("declaration", "definition", "item", "body")):
            return "definition"
    if name in {
        "pair",
        "property",
        "block_mapping_pair",
        "flow_pair",
        "key_value",
        "key_value_pair",
    }:
        return "json" if language in {"json", "jsonnet"} else "statement"
    if name in {
        "array",
        "object",
        "table",
        "table_array_element",
        "block_mapping",
        "block_sequence",
        "flow_mapping",
        "flow_sequence",
        "block_sequence_item",
        "document",
        "row",
        "record",
    }:
        if depth:
            return "json" if language in {"json", "jsonnet"} else "statement"
    if name.endswith(("_statement", "_declaration", "_definition", "_directive", "_clause")):
        return "statement"
    if name in {"block", "suite", "compound_statement", "declaration_list", "body", "do_block"}:
        return "code_block"
    if depth == 1 and language not in _DOCUMENT_LANGUAGES | _MARKUP_LANGUAGES:
        return "statement"
    return None


def _ancestor(node: Any, names: set[str]) -> Any | None:
    parent = node.parent
    for _ in range(128):
        if parent is None:
            return None
        if parent.type in names:
            return parent
        parent = parent.parent
    return None


def _exclude_ranges(
    start: int, end: int, excluded: list[tuple[int, int]]
) -> tuple[tuple[int, int], ...]:
    result, cursor = [], start
    for left, right in sorted(excluded):
        left, right = max(start, left), min(end, right)
        if left > cursor:
            result.append((cursor, left))
        cursor = max(cursor, right)
    if cursor < end:
        result.append((cursor, end))
    return tuple(result)


def _comment_ranges(payload: bytes, start: int, end: int) -> tuple[tuple[int, int], ...]:
    """Remove only actual comment delimiters/prefixes from the analysis view."""
    raw = payload[start:end]
    if raw.startswith(b"<!--"):
        start += 4
        if raw.endswith(b"-->"):
            end -= 3
    elif raw.startswith(b"/*"):
        start += 2
        if raw.endswith(b"*/"):
            end -= 2
    elif raw.startswith(b"{-"):
        start += 2
        if raw.endswith(b"-}"):
            end -= 2
    elif raw.startswith(b"(*"):
        start += 2
        if raw.endswith(b"*)"):
            end -= 2
    else:
        match = re.match(rb"(?:/{2,}|\#{1,}|--+|;+|%+)[ \t]?", raw)
        if match:
            start += match.end()
    excluded = []
    for match in re.finditer(rb"(?m)^[ \t]*\*[ \t]?", payload[start:end]):
        excluded.append((start + match.start(), start + match.end()))
    return _exclude_ranges(start, end, excluded)


def _literal_ranges(node: Any, payload: bytes) -> tuple[tuple[int, int], ...]:
    start, end = node.start_byte, node.end_byte
    pieces = []
    for child in node.named_children:
        if child.type in _CONTENT_TYPES:
            pieces.append((child.start_byte, child.end_byte))
    if pieces:
        raw = payload[start:end]
        if len(pieces) == 1 and re.match(rb"(?i)[rubf]*('{3}|\"{3})", raw):
            return _dedent_literal(payload, *pieces[0], skip_first=True)
        return tuple(pieces)
    raw = payload[start:end]
    if node.type == "block_scalar":
        newline = raw.find(b"\n")
        return _dedent_literal(payload, start + newline + 1, end) if newline >= 0 else ()
    if node.type in {"heredoc_body", "heredoc_content", "nowdoc_body", "string_scalar"}:
        return ((start, end),)
    match = re.match(rb"(?i)(?:[rubf]{0,3}|@|\$@?|@\$)?(\"{3,}|'{3}|\"|'|`)", raw)
    if match:
        quote = match.group(1)
        inner_start = start + match.end()
        inner_end = (
            end - len(quote)
            if raw.endswith(quote) and end - start >= match.end() + len(quote)
            else end
        )
        return ((inner_start, inner_end),) if inner_start < inner_end else ()
    match = re.match(rb'r(\#*)"', raw)
    if match:
        closer = b'"' + match.group(1)
        return ((start + match.end(), end - len(closer) if raw.endswith(closer) else end),)
    match = re.match(rb"\[(=*)\[", raw)
    if match:
        closer = b"]" + match.group(1) + b"]"
        return ((start + match.end(), end - len(closer) if raw.endswith(closer) else end),)
    # Unknown grammar literal nodes remain exact lexical views, never unescaped.
    return ((start, end),)


def _dedent_literal(
    payload: bytes, start: int, end: int, *, skip_first: bool = False
) -> tuple[tuple[int, int], ...]:
    lines, cursor = [], start
    for match in re.finditer(rb"\r\n|\n|\r", payload[start:end]):
        finish = start + match.end()
        lines.append((cursor, finish))
        cursor = finish
    if cursor < end:
        lines.append((cursor, end))
    candidates = lines[1:] if skip_first else lines
    indents = [
        len(payload[a:b]) - len(payload[a:b].lstrip(b" \t"))
        for a, b in candidates
        if payload[a:b].strip()
    ]
    indent = min(indents, default=0)
    if not indent:
        return ((start, end),) if start < end else ()
    ranges = []
    for index, (left, right) in enumerate(lines):
        if not (skip_first and index == 0):
            removed = 0
            while left < right and removed < indent and payload[left] in {32, 9}:
                left += 1
                removed += 1
        if left < right:
            ranges.append((left, right))
    return tuple(ranges)


def _literal_language(node: Any, payload: bytes) -> str | None:
    prefix = payload[max(0, node.start_byte - 256) : node.start_byte]
    directive = list(_LANGUAGE_DIRECTIVE.finditer(prefix))
    if directive:
        # A directive belongs to the immediately following literal, not every
        # later literal on a line or in the next statement.
        tail = prefix[directive[-1].end() :]
        if b";" not in tail and tail.count(b"\n") <= 2:
            return canonical_language(directive[-1].group(1).decode("ascii"))
    parent = node.parent
    if parent is not None and parent.type in {
        "call_expression",
        "tagged_template",
        "tagged_template_expression",
    }:
        function = parent.child_by_field_name("function") or parent.child_by_field_name("tag")
        if function is not None:
            name = (
                payload[function.start_byte : function.end_byte].decode("ascii", "ignore").lower()
            )
            if name in _TAG_LANGUAGES:
                return _TAG_LANGUAGES[name]
    # Heredoc labels and C++ raw delimiters are explicit, conventional hints.
    for ancestor in (node, parent):
        if ancestor is None:
            continue
        for child in ancestor.named_children[:16]:
            if child.type in {"heredoc_start", "heredoc_beginning", "raw_string_delimiter"}:
                label = (
                    payload[child.start_byte : child.end_byte]
                    .strip(b"<~-\"' ")
                    .decode("ascii", "ignore")
                    .lower()
                )
                if label in _TAG_LANGUAGES:
                    return _TAG_LANGUAGES[label]
    # Ruby's heredoc start and body are siblings rather than one AST node.
    if node.type.startswith("heredoc"):
        match = re.search(rb"<<[~-]?[\"']?([A-Za-z_][A-Za-z_0-9]*)[\"']?[^\n]*\n$", prefix)
        if match:
            return _TAG_LANGUAGES.get(match.group(1).decode("ascii").lower())
    return None


def _html_attributes(start_tag: Any, payload: bytes) -> dict[str, str]:
    result = {}
    for child in start_tag.named_children:
        if child.type != "attribute":
            continue
        name, value = None, None
        for part in child.named_children:
            if part.type == "attribute_name":
                name = payload[part.start_byte : part.end_byte].decode("ascii", "ignore").lower()
            elif "value" in part.type:
                value = (
                    payload[part.start_byte : part.end_byte]
                    .strip(b"\"'")
                    .decode("ascii", "ignore")
                    .lower()
                )
        if name is not None:
            result[name] = value or ""
    return result


def _html_raw_language(node: Any, payload: bytes) -> str | None:
    parent = node.parent
    if parent is None:
        return None
    start_tag = next((n for n in parent.named_children if n.type == "start_tag"), None)
    if start_tag is None:
        return None
    attrs = _html_attributes(start_tag, payload)
    if attrs.get("lang"):
        return canonical_language(attrs["lang"])
    mime = attrs.get("type", "").split(";", 1)[0]
    if parent.type == "style_element":
        return canonical_language(mime) if mime else "css"
    if mime in {"", "module", "text/javascript", "application/javascript"}:
        return "javascript"
    if mime.endswith("+json") or mime in {"application/json", "importmap", "speculationrules"}:
        return "json"
    if mime in {"text/x-template", "text/ng-template", "text/html"}:
        return "html"
    return canonical_language(mime)


def _attribute_language(node: Any, payload: bytes) -> str | None:
    parent = _ancestor(node, {"attribute"})
    if parent is None:
        return None
    name_node = next((n for n in parent.named_children if n.type == "attribute_name"), None)
    if name_node is None:
        return None
    name = payload[name_node.start_byte : name_node.end_byte].lower()
    if name.startswith(b"on") or name.startswith((b"@", b"v-on:")):
        return "javascript"
    if name == b"style":
        return "css"
    if name == b"srcdoc":
        return "html"
    return (
        "prose"
        if name in {b"title", b"alt", b"placeholder", b"aria-label", b"aria-description"}
        else None
    )


def _node_injections(node: Any, payload: bytes, language: str) -> list[Injection]:
    name, start, end = node.type, node.start_byte, node.end_byte
    if name == "template" and language in {"embeddedtemplate", "twig"}:
        ranges = tuple(
            (child.start_byte, child.end_byte)
            for child in node.named_children
            if child.type == "content" and child.start_byte < child.end_byte
        )
        if ranges:
            return [
                Injection(ranges[0][0], ranges[-1][1], "html", "html", "template_markup", ranges)
            ]
    if language == "embeddedtemplate" and name == "code":
        return [Injection(start, end, "code", None, "template_code")]
    if language == "heex" and name == "expression_value":
        return [Injection(start, end, "code", "elixir", "heex_expression")]
    if language == "astro" and name == "frontmatter_js_block":
        return [Injection(start, end, "code", "typescript", "astro_frontmatter")]
    if (
        language == "astro"
        and name == "permissible_text"
        and node.parent
        and node.parent.type == "html_interpolation"
    ):
        return [Injection(start, end, "code", "tsx", "astro_expression")]
    if language == "svelte" and name == "raw_text_expr":
        return [Injection(start, end, "code", "javascript", "svelte_expression")]
    if language == "xml" and name in {"CData", "cdata_content"}:
        return [Injection(start, end, "auto", None, "cdata_content")]
    if name in {
        "element",
        "content",
        "jsx_element",
        "fragment",
    } and language in _MARKUP_LANGUAGES | {"javascript", "tsx"}:
        groups: list[Injection] = []
        group_start, group_end = None, None
        for child in node.named_children:
            if child.type in _TEXT_ENTITY_TYPES:
                if group_start is None:
                    group_start = child.start_byte
                group_end = child.end_byte
            elif group_start is not None:
                groups.append(Injection(group_start, group_end, "auto", None, "markup_text"))
                group_start, group_end = None, None
        if group_start is not None:
            groups.append(Injection(group_start, group_end, "auto", None, "markup_text"))
        if groups:
            return groups
    if name == "fenced_code_block":
        info = next((n for n in node.named_children if n.type == "info_string"), None)
        content = next((n for n in node.named_children if n.type == "code_fence_content"), None)
        if content is None:
            return []
        hint = (
            canonical_language(payload[info.start_byte : info.end_byte].decode("utf-8", "replace"))
            if info
            else None
        )
        excluded = [
            (n.start_byte, n.end_byte)
            for n in content.named_children
            if n.type == "block_continuation"
        ]
        content_end = content.end_byte
        # The bundled Markdown grammar can include an EOF closing fence in the
        # content when the document lacks a final newline. CommonMark permits
        # that fence; recognise it on original bytes instead of adding a newline.
        delimiters = [n for n in node.named_children if n.type == "fenced_code_block_delimiter"]
        if len(delimiters) == 1:
            marker = payload[delimiters[0].start_byte : delimiters[0].end_byte]
            last_line = (
                max(
                    payload.rfind(b"\n", content.start_byte, content_end),
                    payload.rfind(b"\r", content.start_byte, content_end),
                )
                + 1
            )
            tail = payload[last_line:content_end]
            tail = re.sub(rb"^(?:[ \t]*>[ \t]?)*", b"", tail)
            if marker and re.fullmatch(
                rb"[ \t]{0,3}"
                + re.escape(marker[:1])
                + rb"{"
                + str(len(marker)).encode()
                + rb",}[ \t]*",
                tail,
            ):
                content_end = max(content.start_byte, last_line)
        ranges = _exclude_ranges(content.start_byte, content_end, excluded)
        return [
            Injection(
                content.start_byte,
                content_end,
                "code" if hint else "auto",
                hint,
                "markdown_fence",
                ranges,
            )
        ]
    if name == "indented_code_block":
        return [Injection(start, end, "code", None, "markdown_indented_code")]
    if language == "markdown" and name in {"html_block", "html_inline"}:
        return [Injection(start, end, "html", "html", "markdown_html")]
    if language == "markdown" and name == "inline":
        return [Injection(start, end, "markdown_inline", "markdown_inline", "markdown_inline")]
    if language == "markdown_inline" and name == "code_span":
        raw = payload[start:end]
        prefix = len(raw) - len(raw.lstrip(b"`"))
        suffix = len(raw) - len(raw.rstrip(b"`"))
        if prefix and suffix and start + prefix < end - suffix:
            return [Injection(start + prefix, end - suffix, "code", None, "inline_code")]
    if language in _DOCUMENT_LANGUAGES - {"markdown", "markdown_inline"} and name in {
        "paragraph",
        "text",
        "text_span",
    }:
        if not _ancestor(node, {"fenced_code_block", "indented_code_block", "code_span"}):
            return [Injection(start, end, "prose", "prose", "document_prose")]
    if name in _PROSE_TYPES and (language in _MARKUP_LANGUAGES or name == "jsx_text"):
        if node.parent is not None and node.parent.type in {
            "element",
            "content",
            "jsx_element",
            "fragment",
        }:
            return []
        return [Injection(start, end, "auto", None, "markup_text")]
    if name == "raw_text" and language in _MARKUP_LANGUAGES:
        hint = (
            "javascript"
            if language == "vue" and node.parent and node.parent.type == "interpolation"
            else _html_raw_language(node, payload)
        )
        return [Injection(start, end, "code" if hint else "auto", hint, "html_raw_content")]
    if name == "attribute_value" and language in _MARKUP_LANGUAGES:
        hint = _attribute_language(node, payload)
        return [Injection(start, end, "auto", hint, "html_attribute")] if hint else []
    if "comment" in name.lower() and not _ancestor(
        node, {"comment", "block_comment", "line_comment"}
    ):
        ranges = _comment_ranges(payload, start, end)
        return [Injection(start, end, "auto", None, "comment_content", ranges)] if ranges else []
    if _is_string_node(node, payload) and name != "quoted_attribute_value":
        if _literal_ancestor(node, payload):
            return []
        ranges = _literal_ranges(node, payload)
        if not ranges:
            return []
        hint = _literal_language(node, payload)
        return [Injection(ranges[0][0], ranges[-1][1], "auto", hint, "literal_content", ranges)]
    return []


def _signature_attachment(node: Any) -> tuple[int, int] | None:
    body = node.child_by_field_name("body")
    if body is None:
        for child in node.named_children[:32]:
            if child.type in {"block", "function_body", "compound_statement", "statement_block"}:
                body = child
                break
            if child.type in {"function_definition", "function_declaration", "method_definition"}:
                nested = _signature_attachment(child)
                return (node.start_byte, nested[1]) if nested else None
    if body is None:
        return None
    first = body.named_child(0) if body.named_child_count else None
    if first is None:
        return (node.start_byte, body.end_byte)
    return (node.start_byte, first.end_byte)


def _group_comments(scan: Scan, payload: bytes, language: str, limits: ParseLimits) -> None:
    comments = [item for item in scan.injections if item.source == "comment_content"]
    occupied = {(item.start, item.end) for item in comments}
    for region in scan.result.regions:
        if (
            region.kind == "comment"
            and region.source == f"tree_sitter:{language}"
            and (region.start, region.end) not in occupied
            and not _comment_ranges(payload, region.start, region.end)
        ):
            # A bare // or # line still separates lines inside a fenced example.
            # It contributes no text bytes, but must not break the comment group.
            comments.append(Injection(region.start, region.end, "auto", None, "comment_content"))
    comments.sort(key=lambda item: (item.start, item.end))
    if len(comments) < 2:
        return
    grouped: list[Injection] = []
    current = comments[0]
    for following in comments[1:]:
        gap = payload[current.end : following.start]
        if current.end <= following.start and not gap.strip():
            gap_end = following.start - (len(gap) - len(gap.rstrip(b" \t")))
            gap_range = ((current.end, gap_end),) if current.end < gap_end else ()
            current = Injection(
                current.start,
                following.end,
                "auto",
                None,
                "comment_content",
                (*current.ranges, *gap_range, *following.ranges),
            )
        else:
            grouped.append(current)
            current = following
    grouped.append(current)
    scan.injections = [item for item in scan.injections if item.source != "comment_content"] + [
        item for item in grouped if item.ranges
    ]
    for item in grouped:
        if len(item.ranges) > 1 and len(scan.result.regions) < limits.max_regions:
            scan.result.regions.append(
                Region(
                    item.start,
                    item.end,
                    "comment",
                    language,
                    0,
                    True,
                    f"tree_sitter:{language}:comment_group",
                )
            )


def scan_syntax(
    payload: bytes,
    language: str,
    limits: ParseLimits,
    backend: str = "tree_sitter",
) -> Scan:
    """Parse an exact byte view and return serializable structural evidence.

    Production calls use the isolated worker wrapper. The direct function is also
    useful for locally trusted grammars and testing. Native progress callbacks are
    intentionally unused: the pinned Python binding has an unsafe callback path.
    """
    if backend == "fallback":
        return scan_generic(payload, language, limits)
    parser_name = f"tree_sitter:{language}"
    try:
        parser = get_parser(language)
        timeout_ms = limits.max_parse_milliseconds
        if timeout_ms:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                parser.timeout_micros = timeout_ms * 1000
        tree = parser.parse(payload)
        if tree is None:
            scan = scan_generic(payload, language, limits)
            scan.result.diagnostics.append(
                Diagnostic("native_parser_timeout", 0, len(payload), parser_name)
            )
            return scan
    except ParserUnavailable as exc:
        scan = scan_generic(payload, language, limits)
        scan.result.diagnostics.append(
            Diagnostic("grammar_unavailable", 0, len(payload), parser_name, str(exc))
        )
        return scan
    except Exception as exc:
        scan = scan_generic(payload, language, limits)
        scan.result.diagnostics.append(
            Diagnostic("native_parser_error", 0, len(payload), parser_name, type(exc).__name__)
        )
        return scan

    scan = Scan()
    result = scan.result
    result.parsers.add(parser_name)
    result.languages.add(language)
    root = tree.root_node
    if root.has_error:
        result.diagnostics.append(
            Diagnostic(
                "recovered_syntax",
                0,
                len(payload),
                parser_name,
                "Recovered tree retained despite missing or invalid syntax",
            )
        )
    errors = 0
    error_spans: list[tuple[int, int]] = []
    hierarchy: list[tuple[int, str]] = []
    pruned_subtrees, pruned_start, pruned_end = 0, len(payload), 0
    for node, depth, pruned in _walk(root, limits.max_nodes, limits.max_tree_depth):
        scan.nodes += 1
        start, end = node.start_byte, node.end_byte
        if pruned:
            pruned_subtrees += 1
            pruned_start, pruned_end = min(pruned_start, start), max(pruned_end, end)
        while hierarchy and hierarchy[-1][0] >= depth:
            hierarchy.pop()
        if node.is_error or node.is_missing:
            if node.is_error and start < end:
                error_spans.append((start, end))
            if errors < 32:
                result.diagnostics.append(
                    Diagnostic(
                        "syntax_error" if node.is_error else "missing_syntax",
                        start,
                        end,
                        parser_name,
                        node.type,
                    )
                )
                errors += 1
        if not node.is_named or start >= end:
            continue
        kind = _kind(node, language, depth, payload)
        if kind and len(result.regions) < limits.max_regions:
            key = f"{language}:{node.type}:{start}:{end}"
            path = (*[item[1] for item in hierarchy], key)
            result.regions.append(
                Region(start, end, kind, language, depth, True, parser_name, path)
            )
            result.boundaries.extend(
                (Boundary(start, kind, depth, parser_name), Boundary(end, kind, depth, parser_name))
            )
            if kind in {"function", "definition", "section", "code_block", "element", "list_item"}:
                hierarchy.append((depth, key))
            if kind in {"function", "definition"}:
                attachment = _signature_attachment(node)
                if (
                    attachment
                    and attachment[0] < attachment[1]
                    and len(result.regions) < limits.max_regions
                ):
                    result.regions.append(
                        Region(
                            *attachment,
                            "signature_attachment",
                            language,
                            depth,
                            True,
                            parser_name,
                            path,
                        )
                    )
        if len(scan.injections) < limits.max_injections:
            for injection in _node_injections(node, payload, language):
                if injection.end > injection.start and len(scan.injections) < limits.max_injections:
                    scan.injections.append(injection)
    if pruned_subtrees:
        result.diagnostics.append(
            Diagnostic(
                "tree_depth_budget_exhausted",
                pruned_start,
                pruned_end,
                parser_name,
                f"max_tree_depth={limits.max_tree_depth}; pruned_subtrees={pruned_subtrees}",
            )
        )
    error_bytes, cursor = 0, 0
    for start, end in sorted(error_spans):
        error_bytes += max(0, end - max(cursor, start))
        cursor = max(cursor, end)
    if payload and error_bytes / len(payload) >= 0.6:
        result.diagnostics.append(
            Diagnostic(
                "low_syntax_coverage",
                0,
                len(payload),
                parser_name,
                f"ERROR nodes cover {error_bytes}/{len(payload)} source bytes",
            )
        )
    if scan.nodes >= limits.max_nodes:
        result.diagnostics.append(
            Diagnostic("node_budget_exhausted", 0, len(payload), parser_name, str(limits.max_nodes))
        )
    if len(result.regions) >= limits.max_regions:
        result.diagnostics.append(
            Diagnostic("region_budget_exhausted", 0, len(payload), parser_name)
        )
    if len(scan.injections) >= limits.max_injections:
        result.diagnostics.append(
            Diagnostic("injection_budget_exhausted", 0, len(payload), parser_name)
        )
    if language in _DOCUMENT_LANGUAGES:
        headings = [region for region in result.regions if region.kind == "heading"]
        bodies = sorted(
            (
                region
                for region in result.regions
                if region.kind in {"paragraph", "code_block", "list_item", "element"}
            ),
            key=lambda region: (region.start, region.end),
        )
        for heading in headings:
            following = next((body for body in bodies if body.start >= heading.end), None)
            if following and len(result.regions) < limits.max_regions:
                result.regions.append(
                    Region(
                        heading.start,
                        following.end,
                        "heading_attachment",
                        language,
                        heading.depth,
                        True,
                        parser_name,
                        heading.path,
                    )
                )
    _group_comments(scan, payload, language, limits)
    # Weak line candidates make malformed leaves splittable without trusting a
    # recovered tree to cover the entire source. Protected AST regions still win.
    remaining = max(0, limits.max_regions * 2 - len(result.boundaries))
    for index, match in enumerate(re.finditer(rb"\r\n|\n|\r", payload)):
        if index >= remaining:
            break
        result.boundaries.append(Boundary(match.end(), "line", 0, "source_lines"))
    return scan


# The lexical recovery path must not treat every punctuation sequence as a
# comment in every programming language. These families are conservative:
# ambiguous or unsupported dialects retain text instead of inventing structure.
_HASH_COMMENTS = frozenset(
    {
        "python",
        "ruby",
        "perl",
        "bash",
        "fish",
        "powershell",
        "r",
        "elixir",
        "nix",
        "make",
        "dockerfile",
        "cmake",
        "yaml",
        "toml",
        "ini",
        "starlark",
        "terraform",
        "hcl",
        "julia",
        "nim",
        "beancount",
    }
)
_HASH_AFTER_SPACE = frozenset(
    {
        "yaml",
        "toml",
        "bash",
        "fish",
        "powershell",
        "make",
        "dockerfile",
        "ini",
        "hcl",
        "terraform",
    }
)
_SLASH_COMMENTS = frozenset(
    {
        "c",
        "cpp",
        "csharp",
        "objc",
        "java",
        "javascript",
        "typescript",
        "tsx",
        "rust",
        "go",
        "kotlin",
        "swift",
        "scala",
        "groovy",
        "dart",
        "php",
        "hack",
        "solidity",
        "zig",
        "scss",
        "less",
        "jsonc",
        "json5",
        "proto",
        "graphql",
        "glsl",
        "hlsl",
        "wgsl",
        "cuda",
        "verilog",
        "svelte",
        "vue",
        "astro",
        "luau",
        "d",
        "odin",
        "hcl",
        "terraform",
    }
)
_DASH_COMMENTS = frozenset(
    {
        "sql",
        "lua",
        "haskell",
        "ada",
        "elm",
        "purescript",
        "vhdl",
    }
)
_SEMICOLON_COMMENTS = frozenset(
    {
        "clojure",
        "commonlisp",
        "elisp",
        "scheme",
        "racket",
        "fennel",
        "janet",
        "asm",
    }
)
_PERCENT_COMMENTS = frozenset(
    {
        "latex",
        "tex",
        "matlab",
        "octave",
        "erlang",
        "prolog",
        "bibtex",
    }
)
_MARKUP_COMMENTS = frozenset(
    {
        "html",
        "xml",
        "markdown",
        "markdown_inline",
        "vue",
        "svelte",
        "astro",
        "twig",
        "heex",
        "embeddedtemplate",
    }
)


def scan_generic(payload: bytes, language: str | None, limits: ParseLimits) -> Scan:
    """Bounded, dialect-aware lexical recovery, never a substitute native AST.

    Line endings, invalid UTF-8, nested comments, and unclosed constructs retain
    their exact byte positions. An unknown dialect only recognises line-leading
    comment markers, not punctuation embedded in identifiers or operators.
    """
    scan = Scan()
    result = scan.result
    result.parsers.add("generic_lexical:v2")
    if language:
        result.languages.add(language)
    dialect = canonical_language(language) or ""
    maximum = min(len(payload), limits.max_parse_bytes)
    index, line_prefix = 0, True
    stack: list[tuple[int, int]] = []

    def comment(start: int, stop: int) -> None:
        _generic_region(scan, start, stop, "comment", language, limits)
        ranges = _comment_ranges(payload, start, stop)
        if ranges and len(scan.injections) < limits.max_injections:
            scan.injections.append(Injection(start, stop, "auto", None, "lexical_comment", ranges))

    while index < maximum and scan.nodes < limits.max_nodes:
        scan.nodes += 1
        value = payload[index]
        block = None
        if dialect in _MARKUP_COMMENTS and payload.startswith(b"<!--", index):
            block = (b"<!--", b"-->", False)
        elif dialect in {"ocaml", "ocaml_interface", "sml", "mathematica"} and payload.startswith(
            b"(*", index
        ):
            block = (b"(*", b"*)", True)
        elif dialect in {"haskell", "purescript", "agda"} and payload.startswith(b"{-", index):
            block = (b"{-", b"-}", True)
        elif dialect in _SLASH_COMMENTS | {"css", "sql"} and payload.startswith(b"/*", index):
            block = (b"/*", b"*/", dialect in {"rust", "swift"})

        if block is not None:
            opener, closer, nested = block
            start, depth = index, 1
            index += len(opener)
            while index < maximum and depth:
                closing = payload.find(closer, index, maximum)
                opening = payload.find(opener, index, maximum) if nested else -1
                if opening >= 0 and (closing < 0 or opening < closing):
                    depth += 1
                    index = opening + len(opener)
                elif closing >= 0:
                    depth -= 1
                    index = closing + len(closer)
                else:
                    index = maximum
            comment(start, index)
            line_prefix = False
            if depth and len(result.diagnostics) < 32:
                result.diagnostics.append(
                    Diagnostic("unterminated_comment", start, index, "generic_lexical")
                )
            continue

        marker = None
        if dialect in _SLASH_COMMENTS and payload.startswith(b"//", index):
            marker = b"//"
        elif dialect in _DASH_COMMENTS and payload.startswith(b"--", index):
            marker = b"--"
        elif dialect in _SEMICOLON_COMMENTS and value == 59:
            marker = b";"
        elif (
            dialect in _PERCENT_COMMENTS
            and value == 37
            and (index == 0 or payload[index - 1] != 92)
        ):
            marker = b"%"
        elif dialect in _HASH_COMMENTS and value == 35:
            if dialect not in _HASH_AFTER_SPACE or index == 0 or payload[index - 1] in b" \t\r\n":
                marker = b"#"
        elif dialect in {"", "unknown", "code"} and line_prefix:
            if payload.startswith(b"//", index):
                marker = b"//"
            elif value == 35:
                marker = b"#"

        if marker is not None:
            start = index
            lf = payload.find(b"\n", index, maximum)
            cr = payload.find(b"\r", index, maximum)
            index = min(x for x in (lf, cr, maximum) if x >= 0)
            comment(start, index)
            line_prefix = False
            continue

        raw_literal = None
        if index == 0 or payload[index - 1] not in (
            b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"
        ):
            if dialect == "rust" and value in {98, 114}:
                match = re.match(rb"(?:br|r)(#{0,16})\"", payload[index : index + 64])
                if match:
                    raw_literal = (match.end(), b'"' + match.group(1))
            elif dialect in {"cpp", "cuda"} and value in {82, 76, 85, 117}:
                match = re.match(
                    rb'(?:u8|u|U|L)?R"([^()\s\\]{0,16})\(',
                    payload[index : index + 64],
                )
                if match:
                    raw_literal = (match.end(), b")" + match.group(1) + b'"')
        if raw_literal is not None:
            opening_length, closer = raw_literal
            start, content_start = index, index + opening_length
            stop = payload.find(closer, content_start, maximum)
            content_end = stop if stop >= 0 else maximum
            index = stop + len(closer) if stop >= 0 else maximum
            _generic_region(scan, start, index, "string", language, limits)
            if content_start < content_end and len(scan.injections) < limits.max_injections:
                hint = detect_content_language(payload[content_start:content_end])
                scan.injections.append(
                    Injection(content_start, content_end, "auto", hint, "lexical_string")
                )
            if stop < 0 and len(result.diagnostics) < 32:
                result.diagnostics.append(
                    Diagnostic("unterminated_literal", start, index, "generic_lexical")
                )
            line_prefix = False
            continue

        if value in {34, 39, 96}:
            if dialect == "rust" and value == 39:
                # Lifetime labels such as 'a and '_ must not eat an entire line.
                lifetime = re.match(rb"'[A-Za-z_][A-Za-z_0-9]*", payload[index : index + 256])
                if lifetime and (
                    index + len(lifetime.group()) >= maximum
                    or payload[index + len(lifetime.group())] != 39
                ):
                    index += len(lifetime.group())
                    line_prefix = False
                    continue
            start = index
            quote = bytes((value,))
            delimiter = quote * 3 if payload.startswith(quote * 3, index) else quote
            index += len(delimiter)
            content_start, closed = index, False
            while index < maximum:
                if payload[index] == 92 and value != 96:
                    index = min(maximum, index + 2)
                elif payload.startswith(delimiter, index):
                    closed = True
                    break
                elif payload[index] in {10, 13} and len(delimiter) == 1 and value != 96:
                    break
                else:
                    index += 1
            content_end = index
            if closed:
                index += len(delimiter)
            _generic_region(scan, start, index, "string", language, limits)
            if content_start < content_end and len(scan.injections) < limits.max_injections:
                hint = detect_content_language(payload[content_start:content_end])
                scan.injections.append(
                    Injection(content_start, content_end, "auto", hint, "lexical_string")
                )
            if not closed and len(result.diagnostics) < 32:
                result.diagnostics.append(
                    Diagnostic("unterminated_literal", start, index, "generic_lexical")
                )
            line_prefix = False
            continue

        if value in {40, 91, 123}:
            if len(stack) < limits.max_nodes:
                stack.append((value, index))
        elif value in {41, 93, 125}:
            opener = {41: 40, 93: 91, 125: 123}[value]
            if stack and stack[-1][0] == opener:
                _, start = stack.pop()
                _generic_region(scan, start, index + 1, "code_block", language, limits)
                result.boundaries.append(
                    Boundary(index + 1, "statement", len(stack), "generic_lexical")
                )
            elif len(result.diagnostics) < 32:
                result.diagnostics.append(
                    Diagnostic("unmatched_delimiter", index, index + 1, "generic_lexical")
                )
        elif value == 59:
            result.boundaries.append(
                Boundary(index + 1, "statement", len(stack), "generic_lexical")
            )
        elif value in {10, 13}:
            if value == 13 and index + 1 < maximum and payload[index + 1] == 10:
                index += 1
            result.boundaries.append(Boundary(index + 1, "line", len(stack), "generic_lexical"))
            line_prefix = True
        if value not in {9, 10, 13, 32}:
            line_prefix = False
        index += 1
        if len(result.boundaries) >= limits.max_regions * 2:
            break
    if index < len(payload):
        result.diagnostics.append(
            Diagnostic("lexical_budget_exhausted", index, len(payload), "generic_lexical")
        )
    if stack:
        result.diagnostics.append(
            Diagnostic("unclosed_delimiter", stack[0][1], len(payload), "generic_lexical")
        )
    return scan


def _generic_region(
    scan: Scan, start: int, end: int, kind: str, language: str | None, limits: ParseLimits
) -> None:
    if start < end and len(scan.result.regions) < limits.max_regions:
        scan.result.regions.append(Region(start, end, kind, language, 0, True, "generic_lexical"))
        scan.result.boundaries.extend(
            (Boundary(start, kind, 0, "generic_lexical"), Boundary(end, kind, 0, "generic_lexical"))
        )
