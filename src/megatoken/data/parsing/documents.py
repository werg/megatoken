"""Lenient document containers and exact-source embedded-content views.

Markdown uses CommonMark block parsing, including nested containers. Other
document syntaxes have conservative, recoverable container recognisers alongside
their native grammar. None of these adapters render markup, evaluate directives,
read include paths, fetch URLs, or decode source escapes/entities.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from .registry import canonical_language
from .syntax import Injection, Scan
from .types import Boundary, Diagnostic, ParseLimits, Region

DOCUMENT_LANGUAGES = {"markdown", "markdown_inline", "rst", "org", "asciidoc", "latex", "typst"}
MARKUP_LANGUAGES = {"html", "xml", "vue", "svelte", "astro", "heex", "embeddedtemplate", "twig"}


def line_spans(payload: bytes) -> list[tuple[int, int]]:
    """CommonMark line numbering, including CR-only inputs, without normalisation."""
    result, start = [], 0
    for match in re.finditer(rb"\r\n|\n|\r", payload):
        result.append((start, match.end()))
        start = match.end()
    if start < len(payload):
        result.append((start, len(payload)))
    return result


def _region(
    scan: Scan,
    start: int,
    end: int,
    kind: str,
    language: str,
    source: str,
    limits: ParseLimits,
    *,
    depth: int = 0,
    protect: bool = True,
) -> None:
    if start < end and len(scan.result.regions) < limits.max_regions:
        scan.result.regions.append(Region(start, end, kind, language, depth, protect, source))
        scan.result.boundaries.extend(
            (Boundary(start, kind, depth, source), Boundary(end, kind, depth, source))
        )


def _nonempty_ranges(
    payload: bytes, spans: Iterable[tuple[int, int]], prefix: int = 0
) -> tuple[tuple[int, int], ...]:
    result = []
    for start, end in spans:
        cursor = start
        # Remove only structural indentation, never whitespace beyond the
        # container's common indent (which may be meaningful in its language).
        while cursor < end and cursor - start < prefix and payload[cursor] in {32, 9}:
            cursor += 1
        if cursor < end:
            result.append((cursor, end))
    return tuple(result)


def _fence_ranges(
    payload: bytes, spans: list[tuple[int, int]], left: int, right: int, markup: str
) -> tuple[tuple[int, int], ...]:
    if right <= left + 1:
        return ()
    opening = payload[slice(*spans[left])]
    marker = markup.encode("ascii")
    marker_at = opening.find(marker)
    prefix = opening[:marker_at] if marker_at >= 0 else b""
    quote_depth = prefix.count(b">")
    # CommonMark list indentation belongs to the container; code's extra indent
    # remains present in every source range.
    list_match = re.search(rb"(?:[-+*]|[0-9]{1,9}[.)])[ \t]+", prefix)
    indent = len(prefix) - (list_match.start() if list_match else len(prefix))
    if not quote_depth and not list_match:
        indent = min(3, len(prefix))
    last = payload[slice(*spans[right - 1])]
    trimmed = re.sub(rb"^(?:[ \t]*>[ \t]?)*[ \t]*", b"", last)
    closing = re.fullmatch(
        re.escape(marker[:1]) + rb"{" + str(len(marker)).encode() + rb",}[ \t\r\n]*", trimmed
    )
    finish = right - 1 if closing else right
    result = []
    for row in range(left + 1, finish):
        start, end = spans[row]
        cursor = start
        if quote_depth:
            for _ in range(quote_depth):
                while cursor < end and payload[cursor] in {32, 9}:
                    cursor += 1
                if cursor < end and payload[cursor] == 62:
                    cursor += 1
                    if cursor < end and payload[cursor] in {32, 9}:
                        cursor += 1
        removed = 0
        while cursor < end and removed < indent and payload[cursor] in {32, 9}:
            cursor += 1
            removed += 1
        if cursor < end:
            result.append((cursor, end))
    return tuple(result)


def scan_markdown(payload: bytes, limits: ParseLimits) -> Scan:
    """CommonMark/GFM blocks with direct byte-line maps and conservative recovery."""
    scan = Scan()
    source = "commonmark:markdown-it-py"
    scan.result.parsers.add(source)
    scan.result.languages.add("markdown")
    spans = line_spans(payload)
    if not spans:
        return scan
    try:
        from markdown_it import MarkdownIt

        parser = MarkdownIt("commonmark", {"html": True, "maxNesting": limits.max_depth * 4})
        parser.enable("table")
        tokens = parser.parse(payload.decode("utf-8", "surrogateescape"))
    except ImportError:
        scan.result.diagnostics.append(
            Diagnostic(
                "document_parser_unavailable",
                0,
                len(payload),
                source,
                "Install markdown-it-py for CommonMark parsing",
            )
        )
        return scan_document_containers(payload, "markdown", limits)
    except Exception as exc:
        scan.result.diagnostics.append(
            Diagnostic("document_parser_error", 0, len(payload), source, type(exc).__name__)
        )
        fallback = scan_document_containers(payload, "markdown", limits)
        scan.result.extend(fallback.result)
        scan.injections.extend(fallback.injections)
        return scan
    headings: list[tuple[int, int, int]] = []
    blocks: list[tuple[int, int, str]] = []
    kinds = {
        "paragraph_open": "paragraph",
        "heading_open": "heading",
        "list_item_open": "list_item",
        "blockquote_open": "blockquote",
        "table_open": "table",
        "tr_open": "table_row",
        "hr": "section",
        "fence": "code_block",
        "code_block": "code_block",
        "html_block": "element",
    }
    for token in tokens:
        scan.nodes += 1
        if scan.nodes > limits.max_nodes or len(scan.result.regions) >= limits.max_regions:
            scan.result.diagnostics.append(
                Diagnostic("node_budget_exhausted", 0, len(payload), source)
            )
            break
        if token.map is None:
            continue
        left, right = token.map
        if not 0 <= left < right <= len(spans):
            continue
        start, end = spans[left][0], spans[right - 1][1]
        kind = kinds.get(token.type)
        if kind:
            _region(scan, start, end, kind, "markdown", source, limits, depth=token.level)
            blocks.append((start, end, kind))
        if token.type == "heading_open":
            headings.append((start, end, token.level))
        if len(scan.injections) >= limits.max_injections:
            continue
        if token.type == "fence":
            hint = canonical_language(token.info)
            ranges = _fence_ranges(payload, spans, left, right, token.markup)
            if ranges:
                scan.injections.append(
                    Injection(
                        ranges[0][0],
                        ranges[-1][1],
                        "code" if hint else "auto",
                        hint,
                        "markdown_fence",
                        ranges,
                    )
                )
            last = payload[slice(*spans[right - 1])].strip()
            if right == len(spans) and not last.lstrip(b"> ").startswith(token.markup.encode()):
                scan.result.diagnostics.append(
                    Diagnostic(
                        "unclosed_fence",
                        start,
                        end,
                        source,
                        "Fence recovered through end of source",
                    )
                )
        elif token.type == "code_block":
            ranges = _nonempty_ranges(payload, spans[left:right], 4)
            if ranges:
                scan.injections.append(
                    Injection(
                        ranges[0][0], ranges[-1][1], "code", None, "markdown_indented_code", ranges
                    )
                )
        elif token.type == "html_block":
            scan.injections.append(Injection(start, end, "html", "html", "markdown_html"))
        elif token.type == "inline":
            # Inline grammar recognises code, links, and markup at original byte
            # offsets; the NLP pass sees the same bytes and can refine large text.
            scan.injections.append(
                Injection(start, end, "markdown_inline", "markdown_inline", "markdown_inline")
            )
    _attach_headings(scan, headings, blocks, "markdown", source, limits)
    return scan


def _attach_headings(
    scan: Scan,
    headings: list[tuple[int, int, int]],
    blocks: list[tuple[int, int, str]],
    language: str,
    source: str,
    limits: ParseLimits,
) -> None:
    ordered = sorted(set(blocks))
    for start, end, depth in headings:
        first = next(
            (
                block
                for block in ordered
                if block[0] >= end and block[2] not in {"heading", "section"}
            ),
            None,
        )
        if first:
            _region(
                scan, start, first[1], "heading_attachment", language, source, limits, depth=depth
            )


def scan_document_containers(payload: bytes, language: str, limits: ParseLimits) -> Scan:
    """Recognise embedded-language containers in RST, Org, AsciiDoc, TeX and prose.

    These bounded stateful recognisers supplement native trees. For an unsupported
    document grammar (such as AsciiDoc in the pinned bundle), their provenance is
    explicitly ``document_containers`` rather than a native AST claim.
    """
    scan = Scan()
    source = f"document_containers:{language}:v1"
    scan.result.parsers.add(source)
    scan.result.languages.add(language)
    spans = line_spans(payload)
    rows = [payload[start:end].rstrip(b"\r\n") for start, end in spans]
    headings: list[tuple[int, int, int]] = []
    blocks: list[tuple[int, int, str]] = []
    index, pending_hint = 0, None
    while index < len(rows) and scan.nodes < limits.max_nodes:
        scan.nodes += 1
        raw, (start, end) = rows[index], spans[index]
        stripped = raw.strip()
        if len(scan.result.regions) >= limits.max_regions:
            break
        match = re.match(rb"[ \t]*\[source\s*,\s*([A-Za-z0-9_+#.-]+)[^]]*\]", raw, re.I)
        if match and language == "asciidoc":
            pending_hint = canonical_language(match.group(1).decode("ascii"))
            index += 1
            continue
        # CommonMark fallback, Org source/example/export blocks, AsciiDoc literal
        # blocks, and reStructuredText explicit code/raw directives.
        fence = re.match(rb"[ \t]{0,3}(`{3,}|~{3,})[ \t]*(.*)", raw)
        org = re.match(rb"[ \t]*#\+begin_(src|example|export)\b[ \t]*(.*)", raw, re.I)
        rst = re.match(
            rb"([ \t]*)\.\.[ \t]+(code-block|code|sourcecode|raw|parsed-literal|literalinclude)::[ \t]*(.*)",
            raw,
            re.I,
        )
        if fence and language in {"markdown", "prose"}:
            marker, info = fence.groups()
            hint = canonical_language(info.decode("utf-8", "replace"))
            closing = re.compile(
                rb"[ \t]{0,3}"
                + re.escape(marker[:1])
                + rb"{"
                + str(len(marker)).encode()
                + rb",}[ \t]*$"
            )
            finish = index + 1
            while finish < len(rows) and not closing.fullmatch(rows[finish]):
                finish += 1
            inner_end = spans[finish][0] if finish < len(rows) else len(payload)
            whole_end = spans[finish][1] if finish < len(rows) else len(payload)
            _region(scan, start, whole_end, "code_block", language, source, limits)
            if end < inner_end:
                scan.injections.append(
                    Injection(end, inner_end, "code" if hint else "auto", hint, "document_fence")
                )
            if finish == len(rows):
                scan.result.diagnostics.append(
                    Diagnostic("unclosed_fence", start, whole_end, source)
                )
            blocks.append((start, whole_end, "code_block"))
            index = finish + 1
            continue
        if org and language == "org":
            block_name, info = org.groups()
            hint = (
                canonical_language(info.decode("utf-8", "replace"))
                if block_name.lower() != b"example"
                else None
            )
            closing = re.compile(rb"[ \t]*#\+end_" + block_name + rb"[ \t]*$", re.I)
            finish = index + 1
            while finish < len(rows) and not closing.fullmatch(rows[finish]):
                finish += 1
            inner_end = spans[finish][0] if finish < len(rows) else len(payload)
            whole_end = spans[finish][1] if finish < len(rows) else len(payload)
            _region(scan, start, whole_end, "code_block", language, source, limits)
            if end < inner_end:
                scan.injections.append(
                    Injection(end, inner_end, "code" if hint else "auto", hint, "org_block")
                )
            if finish == len(rows):
                scan.result.diagnostics.append(
                    Diagnostic("unclosed_document_block", start, whole_end, source)
                )
            blocks.append((start, whole_end, "code_block"))
            index = finish + 1
            continue
        if language == "asciidoc" and stripped in {b"----", b"....", b"++++"}:
            marker, finish = stripped, index + 1
            while finish < len(rows) and rows[finish].strip() != marker:
                finish += 1
            inner_end = spans[finish][0] if finish < len(rows) else len(payload)
            whole_end = spans[finish][1] if finish < len(rows) else len(payload)
            hint = "html" if marker == b"++++" else pending_hint
            _region(scan, start, whole_end, "code_block", language, source, limits)
            if end < inner_end:
                scan.injections.append(
                    Injection(end, inner_end, "code" if hint else "auto", hint, "asciidoc_block")
                )
            if finish == len(rows):
                scan.result.diagnostics.append(
                    Diagnostic("unclosed_document_block", start, whole_end, source)
                )
            blocks.append((start, whole_end, "code_block"))
            pending_hint, index = None, finish + 1
            continue
        if rst and language == "rst":
            indent_bytes, directive, info = rst.groups()
            if directive.lower() == b"literalinclude":
                scan.result.diagnostics.append(
                    Diagnostic(
                        "external_reference_unread",
                        start,
                        end,
                        source,
                        "Include references are preserved without filesystem access",
                    )
                )
                index += 1
                continue
            hint = canonical_language(info.decode("utf-8", "replace"))
            first, finish = index + 1, index + 1
            while first < len(rows) and (
                not rows[first].strip() or rows[first].lstrip().startswith(b":")
            ):
                first += 1
            if first >= len(rows):
                index += 1
                continue
            indent = len(rows[first]) - len(rows[first].lstrip(b" \t"))
            if indent <= len(indent_bytes):
                index += 1
                continue
            finish = first
            while finish < len(rows):
                current = rows[finish]
                if current.strip() and len(current) - len(current.lstrip(b" \t")) < indent:
                    break
                finish += 1
            whole_end = spans[finish - 1][1]
            ranges = _nonempty_ranges(payload, spans[first:finish], indent)
            _region(scan, start, whole_end, "code_block", language, source, limits)
            if ranges:
                scan.injections.append(
                    Injection(
                        ranges[0][0],
                        ranges[-1][1],
                        "code" if hint else "auto",
                        hint,
                        "rst_directive",
                        ranges,
                    )
                )
            blocks.append((start, whole_end, "code_block"))
            index = finish
            continue
        heading = False
        if language == "org":
            heading = bool(re.match(rb"\*+[ \t]+\S", raw))
        elif language == "asciidoc":
            heading = bool(re.match(rb"={1,6}[ \t]+\S", raw))
        elif language in {"rst", "markdown"} and index + 1 < len(rows):
            underline = rows[index + 1].strip()
            heading = bool(
                stripped
                and len(underline) >= 2
                and len(set(underline)) == 1
                and underline[:1] in (b"=", b"-", b"~", b"^", b'"', b"+", b"*", b"#")
            )
            if heading:
                end = spans[index + 1][1]
                index += 1
        if heading:
            _region(scan, start, end, "heading", language, source, limits)
            headings.append((start, end, 0))
        elif stripped:
            blocks.append((start, end, "paragraph"))
        index += 1
        if len(scan.injections) >= limits.max_injections:
            break
    if language == "latex":
        _latex_blocks(payload, scan, limits)
    _attach_headings(scan, headings, blocks, language, source, limits)
    if index < len(rows):
        scan.result.diagnostics.append(
            Diagnostic("document_budget_exhausted", spans[index][0], len(payload), source)
        )
    return scan


def _latex_blocks(payload: bytes, scan: Scan, limits: ParseLimits) -> None:
    source = "document_containers:latex:v1"
    opening = re.compile(
        rb"\\begin\{(minted|lstlisting|verbatim\*?|Verbatim)\}(?:\[([^\]\r\n]*)\])?(?:\{([^}\r\n]+)\})?"
    )
    for match in opening.finditer(payload):
        if len(scan.injections) >= limits.max_injections:
            break
        name, options, argument = match.groups()
        closer = b"\\end{" + name + b"}"
        stop = payload.find(closer, match.end())
        inner_end = stop if stop >= 0 else len(payload)
        end = stop + len(closer) if stop >= 0 else len(payload)
        hint = (
            canonical_language(argument.decode("ascii", "ignore"))
            if name == b"minted" and argument
            else None
        )
        if options and hint is None:
            value = re.search(rb"(?:^|,)\s*language\s*=\s*\{?([A-Za-z0-9_+#.-]+)", options, re.I)
            if value:
                hint = canonical_language(value.group(1).decode("ascii"))
        _region(scan, match.start(), end, "code_block", "latex", source, limits)
        if match.end() < inner_end:
            scan.injections.append(
                Injection(
                    match.end(), inner_end, "code" if hint else "auto", hint, "latex_environment"
                )
            )
        if stop < 0:
            scan.result.diagnostics.append(
                Diagnostic("unclosed_document_block", match.start(), end, source)
            )


# Work only on original attribute-value spans: decoding HTML entities or
# normalizing quotes would make nested parser offsets invalid.
_ATTRIBUTE = re.compile(
    rb"([:@A-Za-z_][:@A-Za-z0-9_.-]*)[ \t\r\n]*=[ \t\r\n]*"
    rb"(?:\"([^\"]*)\"|'([^']*)'|([^\s>]+))"
)
_EVENT_NAMES = frozenset(
    name.encode("ascii")
    for name in (
        "click", "dblclick", "input", "change", "submit", "reset", "load",
        "error", "focus", "blur", "keydown", "keyup", "keypress", "contextmenu",
        "mousedown", "mouseup", "mouseenter", "mouseleave", "mouseover",
        "mouseout", "mousemove", "wheel", "scroll", "resize", "select",
        "drag", "dragstart", "dragend", "dragover", "drop", "touchstart",
        "touchmove", "touchend", "pointerdown", "pointerup", "pointermove",
        "copy", "cut", "paste", "animationstart", "animationend",
        "transitionend", "beforeunload", "message",
    )
)
_PROSE_ATTRIBUTES = frozenset({
    b"title", b"alt", b"placeholder", b"aria-label", b"aria-description",
})


def _markup_attributes(
    payload: bytes, start: int, end: int
) -> list[tuple[bytes, bytes, int, int]]:
    result = []
    for match in _ATTRIBUTE.finditer(payload, start, end):
        for group in (2, 3, 4):
            if match.start(group) >= 0:
                left, right = match.span(group)
                if left < right:
                    result.append((match.group(1).lower(), payload[left:right], left, right))
                break
    return result


def _raw_body_hint(name: bytes, attributes: dict[bytes, bytes]) -> str:
    from .registry import resolve_language

    explicit = attributes.get(b"lang", b"").decode("ascii", "ignore").strip()
    if explicit:
        return canonical_language(explicit) or "unknown"
    media = attributes.get(b"type", b"").decode("ascii", "ignore").strip().lower()
    if not media:
        return "javascript" if name == b"script" else "css"
    if media in {"module", "text/javascript", "application/javascript"}:
        return "javascript"
    if media in {"importmap", "speculationrules"} or media.endswith("+json"):
        return "json"
    if media in {"text/x-template", "text/ng-template"}:
        return "html"
    return resolve_language(mime_type=media) or canonical_language(media) or "unknown"


def scan_markup_fallback(payload: bytes, language: str, limits: ParseLimits) -> Scan:
    """Recover tags, raw script/style bodies and text when a native grammar is absent.

    The scanner is deliberately quote-aware and does not use a broad ``<.*?>``
    expression, which would split tags at ``>`` inside attributes. Open tags are
    recoverable containers; no DTD/entity processing or external reads occur.
    """
    scan = Scan()
    source = f"markup_recovery:{language}:v1"
    scan.result.parsers.add(source)
    scan.result.languages.add(language)
    index, text_start, length = 0, 0, len(payload)
    stack: list[tuple[bytes, int]] = []
    while index < length and scan.nodes < limits.max_nodes:
        scan.nodes += 1
        if payload[index] != 60:
            index += 1
            continue
        tag_match = re.match(
            rb"<(/?)([A-Za-z_][A-Za-z_0-9:.-]*)", payload[index : min(length, index + 256)]
        )
        if not tag_match and not payload.startswith((b"<!--", b"<![CDATA["), index):
            index += 1
            continue
        if index > text_start and len(scan.injections) < limits.max_injections:
            scan.injections.append(Injection(text_start, index, "auto", None, "markup_text"))
        if payload.startswith(b"<!--", index):
            stop = payload.find(b"-->", index + 4)
            end = stop + 3 if stop >= 0 else length
            _region(scan, index, end, "comment", language, source, limits)
            if index + 4 < (stop if stop >= 0 else end) and len(scan.injections) < limits.max_injections:
                scan.injections.append(
                    Injection(index + 4, stop if stop >= 0 else end, "auto", None, "markup_comment")
                )
            index = text_start = end
            continue
        if payload.startswith(b"<![CDATA[", index):
            stop = payload.find(b"]]>", index + 9)
            end = stop + 3 if stop >= 0 else length
            _region(scan, index, end, "string", language, source, limits)
            if index + 9 < (stop if stop >= 0 else end) and len(scan.injections) < limits.max_injections:
                scan.injections.append(
                    Injection(index + 9, stop if stop >= 0 else end, "auto", None, "cdata_content")
                )
            index = text_start = end
            continue
        match = tag_match
        if not match:
            index += 1
            text_start = min(text_start, index)
            continue
        closing, name = bool(match.group(1)), match.group(2).lower()
        cursor, quote = index + match.end(), None
        while cursor < length:
            value = payload[cursor]
            if quote is not None:
                if value == quote:
                    quote = None
            elif value in {34, 39}:
                quote = value
            elif value == 62:
                cursor += 1
                break
            cursor += 1
        _region(scan, index, cursor, "tag", language, source, limits)
        attributes = _markup_attributes(payload, index + match.end(), cursor) if not closing else []
        if not closing:
            for attribute, value, left, right in attributes:
                if len(scan.injections) >= limits.max_injections:
                    break
                if attribute in _PROSE_ATTRIBUTES:
                    hint = "prose"
                elif attribute == b"style":
                    hint = "css"
                elif attribute == b"srcdoc":
                    hint = "html"
                elif attribute[2:] in _EVENT_NAMES and attribute.startswith(b"on"):
                    hint = "javascript"
                elif attribute.startswith((b"@", b"v-on:")):
                    hint = "javascript"
                else:
                    continue
                scan.injections.append(
                    Injection(left, right, "auto", hint, "html_attribute")
                )
        if closing:
            for position in range(len(stack) - 1, -1, -1):
                if stack[position][0] == name:
                    _region(scan, stack[position][1], cursor, "element", language, source, limits)
                    del stack[position:]
                    break
        elif name in {b"script", b"style"}:
            close = re.compile(rb"</" + name + rb"\s*>", re.I).search(payload, cursor)
            inner_end = close.start() if close else length
            whole_end = close.end() if close else length
            hint = _raw_body_hint(
                name, {attribute: value for attribute, value, _, _ in attributes}
            )
            if cursor < inner_end and len(scan.injections) < limits.max_injections:
                scan.injections.append(
                    Injection(cursor, inner_end, "code", hint, "html_raw_content")
                )
            _region(scan, index, whole_end, "element", language, source, limits)
            cursor = whole_end
        elif not payload[index:cursor].rstrip().endswith(b"/>") and name not in {
            b"area",
            b"base",
            b"br",
            b"col",
            b"embed",
            b"hr",
            b"img",
            b"input",
            b"link",
            b"meta",
            b"param",
            b"source",
            b"track",
            b"wbr",
        }:
            stack.append((name, index))
        index = text_start = cursor
        if (
            len(scan.injections) >= limits.max_injections
            or len(scan.result.regions) >= limits.max_regions
        ):
            break
    if text_start < length and len(scan.injections) < limits.max_injections:
        scan.injections.append(Injection(text_start, length, "auto", None, "markup_text"))
    if stack:
        scan.result.diagnostics.append(Diagnostic("unclosed_markup", stack[0][1], length, source))
    return scan
