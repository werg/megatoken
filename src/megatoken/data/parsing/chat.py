"""Conservative, byte-exact framing of explicitly serialized conversations.

These parsers identify source ranges; they do not render templates, interpret
message bodies, or execute tool markup. Structured dataset messages are already
framed and MUST use ``template="none"`` when their content is parsed. A text dump
cannot distinguish an unescaped reserved token from its identical literal byte
spelling. Anchored recognition and literal shielding reduce that ambiguity, but
trusted source message boundaries always take precedence over inference here.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from types import MappingProxyType

from .types import Diagnostic

CHAT_PARSER_VERSION = "chat-framing-v1"
_SPACE = b" \t\r\n"
_HEADER_LIMIT = 1024
_ROLE = re.compile(rb"[A-Za-z_][A-Za-z0-9_.:/-]{0,127}\Z")
_ATTRIBUTE = re.compile(rb"(?:\A|[ \t])([a-z_]+)=([^ \t<>]+)")
_FENCE = re.compile(rb" {0,3}(`{3,}|~{3,})([^\r\n]*)")
_RUST_RAW = re.compile(rb'(?:r|br)(#{0,32})"')
_CPP_RAW = re.compile(rb'R"([^()\\\s]{0,16})\(')
_ROLES = ("system", "developer", "user", "assistant", "tool", "function", "environment")
_ALIASES = {"human": "user", "model": "assistant", "gpt": "assistant", "function": "tool"}


@dataclass(frozen=True, slots=True)
class ChatFrame:
    """One contiguous source frame, including its unmodified delimiters.

    ``[start, header_end)`` owns the header (and an initial BOS, if present),
    ``[header_end, body_end)`` owns content, and ``[body_end, end)`` owns the
    footer and separating whitespace. ``role=None`` denotes an unparsed tail.
    Empty bodies and incomplete generation headers remain representable.
    """

    start: int
    header_end: int
    body_end: int
    end: int
    role: str | None
    name: str | None = None
    channel: str | None = None
    complete: bool = True
    recipient: str | None = None

    def __post_init__(self) -> None:
        if not 0 <= self.start <= self.header_end <= self.body_end <= self.end:
            raise ValueError("Chat frame ranges must be ordered nonnegative byte offsets")

    @property
    def header(self) -> tuple[int, int]:
        return self.start, self.header_end

    @property
    def body(self) -> tuple[int, int]:
        return self.header_end, self.body_end

    @property
    def footer(self) -> tuple[int, int]:
        return self.body_end, self.end


@dataclass(frozen=True, slots=True)
class ChatParse:
    template: str
    frames: tuple[ChatFrame, ...] = ()
    diagnostics: tuple[Diagnostic, ...] = ()
    detected: bool = False

    def validate(self, payload_length: int) -> None:
        if not self.detected:
            if self.frames:
                raise ValueError("Undetected chat cannot assign message frames")
            return
        cursor = 0
        for frame in self.frames:
            if frame.start != cursor or frame.end > payload_length:
                raise ValueError("Chat frames must partition the exact source bytes")
            cursor = frame.end
        if cursor != payload_length:
            raise ValueError("Chat frames do not cover the complete source")


@dataclass(frozen=True, slots=True)
class ChatTemplate:
    """Declarative description of one delimiter-based text-dump template.

    The built-in set is the only registry: there is no registration hook or
    remote code loading, and :func:`template_identity` fingerprints it for the
    chunking policy. ``mode`` is ``delimited``, ``inst``, or ``deepseek``; only
    the two non-delimited families need specialized state machines.
    """

    name: str
    start: bytes
    separator: bytes
    ends: tuple[bytes, ...]
    prefixes: tuple[bytes, ...] = ()
    aliases: tuple[str, ...] = ()
    channel_markers: tuple[bytes, ...] = ()
    format_markers: tuple[bytes, ...] = ()
    body_newlines: int = 0
    allow_named_tools: bool = False
    mode: str = "delimited"

    def __post_init__(self) -> None:
        if not self.name or not self.start or not self.separator or not self.ends:
            raise ValueError("Chat templates require a name and nonempty framing delimiters")
        if any(not marker for marker in self.ends):
            raise ValueError("Chat terminators cannot be empty")
        if self.mode not in {"delimited", "inst", "deepseek"}:
            raise ValueError("Unknown chat template parser mode")
        if not 0 <= self.body_newlines <= 2:
            raise ValueError("A template can consume at most two body-separator newlines")


_BUILTINS = (
    ChatTemplate("chatml", b"<|im_start|>", b"\n", (b"<|im_end|>",), aliases=("qwen",)),
    ChatTemplate(
        "llama3",
        b"<|start_header_id|>",
        b"<|end_header_id|>",
        (b"<|eot_id|>", b"<|eom_id|>", b"<|end_of_text|>"),
        prefixes=(b"<|begin_of_text|>",),
        aliases=("llama-3", "llama3.1", "llama3.2", "llama3.3"),
        body_newlines=2,
        allow_named_tools=True,
    ),
    ChatTemplate(
        "gemma",
        b"<start_of_turn>",
        b"\n",
        (b"<end_of_turn>",),
        prefixes=(b"<bos>",),
        aliases=("gemma2", "gemma3"),
    ),
    ChatTemplate(
        "harmony",
        b"<|start|>",
        b"<|message|>",
        (b"<|end|>", b"<|return|>", b"<|call|>"),
        channel_markers=(b"<|channel|>",),
        format_markers=(b"<|constrain|>",),
        allow_named_tools=True,
    ),
    ChatTemplate(
        "harmony_legacy",
        b"<|im_start|>",
        b"<|im_sep|>",
        (b"<|im_end|>", b"<|fim_suffix|>", b"<|ghissue|>"),
        prefixes=(b"<|endoftext|>",),
        channel_markers=(b"<|meta_sep|>",),
        format_markers=(b"<|meta_start|>",),
        allow_named_tools=True,
    ),
    ChatTemplate(
        "phi3",
        b"<|",
        b"|>",
        (b"<|end|>", b"<|endoftext|>"),
        aliases=("phi-3", "phi4"),
        body_newlines=1,
    ),
    ChatTemplate(
        "inst",
        b"[INST]",
        b"[/INST]",
        (b"</s>",),
        prefixes=(b"<s>",),
        aliases=("llama2", "llama-2", "mistral"),
        mode="inst",
    ),
    ChatTemplate(
        "deepseek",
        "<｜User｜>".encode(),
        "<｜Assistant｜>".encode(),
        ("<｜end▁of▁sentence｜>".encode(),),
        prefixes=("<｜begin▁of▁sentence｜>".encode(),),
        aliases=("deepseek-v3", "deepseek-r1"),
        mode="deepseek",
    ),
)
CHAT_TEMPLATES: Mapping[str, ChatTemplate] = MappingProxyType(
    {item.name: item for item in _BUILTINS}
)


def template_identity() -> dict:
    """JSON-safe parser identity, including every configured delimiter byte."""

    def encode(value):
        if isinstance(value, bytes):
            return value.hex()
        if isinstance(value, (tuple, list)):
            return [encode(item) for item in value]
        return value

    return {
        "version": CHAT_PARSER_VERSION,
        "templates": {
            name: {key: encode(value) for key, value in asdict(spec).items()}
            for name, spec in sorted(CHAT_TEMPLATES.items())
        },
    }


def _diagnostic(code: str, start: int, end: int, detail: str = "") -> Diagnostic:
    return Diagnostic(code, start, end, CHAT_PARSER_VERSION, detail)


def _whitespace(payload: bytes, position: int) -> int:
    while position < len(payload) and payload[position] in _SPACE:
        position += 1
    return position


def _prefix(payload: bytes, spec: ChatTemplate, position: int = 0) -> int:
    if position == 0 and payload.startswith(b"\xef\xbb\xbf"):
        position = 3
    position = _whitespace(payload, position)
    for marker in spec.prefixes:
        if payload.startswith(marker, position):
            return _whitespace(payload, position + len(marker))
    return position


def _header(payload: bytes, position: int, spec: ChatTemplate) -> ChatFrame | None:
    if not payload.startswith(spec.start, position):
        return None
    begin = position + len(spec.start)
    sep = payload.find(spec.separator, begin, min(len(payload), begin + _HEADER_LIMIT))
    if sep < 0:
        return None
    raw = payload[begin:sep].rstrip(b"\r")
    if not raw.isascii() or b"\n" in raw or b"\r" in raw or b"\x00" in raw:
        return None
    # Channel/type delimiters are header syntax, never body syntax.
    channel = None
    normalized = raw
    for marker in spec.channel_markers:
        if marker in raw:
            _, rest = raw.split(marker, 1)
            candidate = re.split(rb"[ \t<]", rest.strip(), maxsplit=1)[0]
            if not _ROLE.fullmatch(candidate):
                return None
            channel = candidate.decode("ascii")
        normalized = normalized.replace(marker, b" ")
    for marker in spec.format_markers:
        normalized = normalized.replace(marker, b" ")
    if b"<" in normalized or b">" in normalized:
        return None
    fields = normalized.split()
    if not fields or not _ROLE.fullmatch(fields[0]):
        return None
    original = fields[0].decode("ascii")
    role = _ALIASES.get(original, original)
    name = None
    if role not in _ROLES:
        if not spec.allow_named_tools:
            return None
        role, name = "tool", original
    first_section = raw
    for marker in (*spec.channel_markers, *spec.format_markers):
        first_section = first_section.split(marker, 1)[0]
    if any(not re.fullmatch(rb"[a-z_]+=[^ \t<>]+", item) for item in first_section.split()[1:]):
        return None
    attributes = {
        key.decode(): value.decode("ascii") for key, value in _ATTRIBUTE.findall(normalized)
    }
    name = attributes.get("name", name)
    end = sep + len(spec.separator)
    for _ in range(spec.body_newlines):
        if payload.startswith(b"\r\n", end):
            end += 2
        elif payload.startswith(b"\n", end):
            end += 1
        else:
            break
    return ChatFrame(position, end, end, end, role, name, channel, recipient=attributes.get("to"))


def _quoted_end(payload: bytes, position: int) -> int | None:
    """Shield common literal spellings without treating prose apostrophes as code.

    Delimiters inside fenced/inline code, ordinary/triple strings, JS templates,
    C++ raw strings, Rust raw strings and HTML/C-style comments are literal.
    Ordinary quotes and inline code end at their line; block literals do not.
    This is a framing guard, not a claim of a universal programming grammar.
    """
    if payload[position] not in b"<'\"`/rbR":
        return None
    if payload.startswith(b"<!--", position):
        close = payload.find(b"-->", position + 4)
        return len(payload) if close < 0 else close + 3
    if payload.startswith(b"/*", position):
        close = payload.find(b"*/", position + 2)
        return len(payload) if close < 0 else close + 2
    raw = _RUST_RAW.match(payload, position, position + 36)
    if raw and (
        position == 0
        or payload[position - 1]
        not in b"_abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    ):
        delimiter = b'"' + raw[1]
        close = payload.find(delimiter, raw.end())
        return len(payload) if close < 0 else close + len(delimiter)
    raw_cpp = _CPP_RAW.match(payload, position, position + 20)
    if raw_cpp:
        delimiter = b")" + raw_cpp[1] + b'"'
        close = payload.find(delimiter, raw_cpp.end())
        return len(payload) if close < 0 else close + len(delimiter)
    value = payload[position : position + 1]
    if value not in {b'"', b"'", b"`"}:
        return None
    if value == b"'" and position and chr(payload[position - 1]).isalnum():
        return None
    if value == b'"' and position and payload[position - 1 : position].isdigit():
        return None  # An inch or second mark (5"), not the start of a string.
    delimiter = value * 3 if value != b"`" and payload.startswith(value * 3, position) else value
    if value == b"`":
        end = position + 1
        while end < len(payload) and payload[end : end + 1] == value:
            end += 1
        delimiter = payload[position:end]
    cursor = position + len(delimiter)
    # Only triple quotes and backtick fences span lines. An ordinary quotation
    # mark or inline code span ends at its line: a stray inch mark or backtick
    # in one turn must not swallow the turn boundary and the next speaker's
    # own quotation.
    block = len(delimiter) >= 3
    line_end = len(payload) if block else payload.find(b"\n", cursor)
    if line_end < 0:
        line_end = len(payload)
    while True:
        close = payload.find(delimiter, cursor, line_end)
        if close < 0:
            # An unmatched ordinary quotation mark may just be prose punctuation.
            # Unclosed block/triple literals are protected through truncation.
            return len(payload) if block else None
        escapes = 0
        prior = close - 1
        while prior >= cursor and payload[prior] == 92:
            escapes += 1
            prior -= 1
        if value == b"`" or not escapes % 2:
            return close + len(delimiter)
        cursor = close + len(delimiter)


def _candidates(payload: bytes, start: int, markers: tuple[bytes, ...]):
    """Yield unquoted structural marker occurrences in one linear scan."""
    cursor = start
    line_start = start
    fence: bytes | None = None
    first_bytes = {marker[0] for marker in markers}
    while cursor < len(payload):
        if cursor == line_start:
            match = _FENCE.match(payload, cursor)
            if fence is not None:
                if match and match[1][:1] == fence[:1] and len(match[1]) >= len(fence):
                    after_run = match.start(2)
                    while after_run < len(payload) and payload[after_run] in b" \t":
                        after_run += 1
                    if any(payload.startswith(marker, after_run) for marker in markers):
                        # Templates commonly append EOS directly after a closing
                        # Markdown fence, with no separating newline.
                        fence = None
                        cursor = after_run
                        line_start = -1
                        continue
                    if not match[2].strip():
                        fence = None
                newline = payload.find(b"\n", cursor)
                cursor = len(payload) if newline < 0 else newline + 1
                line_start = cursor
                continue
            if match:
                fence = match[1]
                newline = payload.find(b"\n", cursor)
                cursor = len(payload) if newline < 0 else newline + 1
                line_start = cursor
                continue
        if payload[cursor] in first_bytes:
            marker = next((item for item in markers if payload.startswith(item, cursor)), None)
            if marker is not None:
                yield cursor, marker
                cursor += len(marker)
                continue
        quoted = _quoted_end(payload, cursor)
        if quoted is not None:
            # A quote can span lines; subsequent fence detection starts only at
            # a true source newline, not just after an arbitrary quoted range.
            cursor = quoted
            line_start = cursor if cursor and payload[cursor - 1] == 10 else -1
            continue
        if payload[cursor] == 92:
            cursor = min(len(payload), cursor + 2)
            continue
        if payload[cursor] in (10, 13):
            line_start = cursor + 1
        cursor += 1


def has_structural_marker(payload: bytes, markers: tuple[bytes, ...]) -> bool:
    """Check reserved body markup using the same literal guards as chat framing."""
    if not markers or any(not isinstance(marker, bytes) or not marker for marker in markers):
        raise ValueError("Structural markers must be nonempty byte strings")
    return next(_candidates(payload, 0, markers), None) is not None


def _delimited(payload: bytes, spec: ChatTemplate, max_frames: int) -> ChatParse:
    position = _prefix(payload, spec)
    frames: list[ChatFrame] = []
    diagnostics: list[Diagnostic] = []
    owned = 0
    while position < len(payload):
        header = _header(payload, position, spec)
        if header is None:
            raw = (
                payload[position + len(spec.start) :]
                if payload.startswith(spec.start, position)
                else b""
            )
            partial_role = re.split(rb"[ \t\r\n<]", raw, maxsplit=1)[0]
            role = _ALIASES.get(
                partial_role.decode("ascii", "ignore"), partial_role.decode("ascii", "ignore")
            )
            recognized_partial = (
                role in _ROLES
                and len(raw) < _HEADER_LIMIT
                and not any(marker in raw for marker in spec.ends)
            )
            if not frames and not recognized_partial:
                return ChatParse(
                    spec.name,
                    diagnostics=(
                        _diagnostic("chat_unrecognized_header", position, len(payload), spec.name),
                    ),
                )
            frames.append(
                ChatFrame(
                    owned,
                    len(payload),
                    len(payload),
                    len(payload),
                    role if recognized_partial else None,
                    complete=False,
                )
            )
            diagnostics.append(
                _diagnostic("chat_incomplete_header", position, len(payload), spec.name)
            )
            break
        if len(frames) >= max_frames:
            frames.append(ChatFrame(owned, owned, len(payload), len(payload), None, complete=False))
            diagnostics.append(
                _diagnostic("chat_frame_limit", owned, len(payload), str(max_frames))
            )
            break
        header = replace(header, start=owned)
        found = False
        for end, marker in _candidates(payload, header.header_end, spec.ends):
            following = _whitespace(payload, end + len(marker))
            if following < len(payload) and not payload.startswith(spec.start, following):
                continue
            # A lone token spelling in text is not an end; a complete frame must
            # be followed by another header or EOF (a final partial header is OK).
            frames.append(replace(header, body_end=end, end=following))
            owned = position = following
            found = True
            break
        if not found:
            frames.append(replace(header, body_end=len(payload), end=len(payload), complete=False))
            diagnostics.append(
                _diagnostic("chat_incomplete_body", header.header_end, len(payload), spec.name)
            )
            break
    return ChatParse(spec.name, tuple(frames), tuple(diagnostics), bool(frames))


def _inst(payload: bytes, spec: ChatTemplate, max_frames: int) -> ChatParse:
    position = _prefix(payload, spec)
    owned = 0
    frames: list[ChatFrame] = []
    diagnostics: list[Diagnostic] = []
    while position < len(payload):
        if not payload.startswith(spec.start, position):
            if not frames:
                return ChatParse(spec.name)
            frames.append(ChatFrame(owned, owned, len(payload), len(payload), None, complete=False))
            diagnostics.append(_diagnostic("chat_unframed_tail", owned, len(payload), spec.name))
            break
        if len(frames) >= max_frames:
            frames.append(ChatFrame(owned, owned, len(payload), len(payload), None, complete=False))
            diagnostics.append(
                _diagnostic("chat_frame_limit", owned, len(payload), str(max_frames))
            )
            break
        body = position + len(spec.start)
        # Llama 2 embeds the initial system message inside its first INST block.
        system = _whitespace(payload, body)
        if not frames and payload.startswith(b"<<SYS>>", system):
            close = next(_candidates(payload, system + 7, (b"<</SYS>>",)), None)
            if close is not None:
                system_end = _whitespace(payload, close[0] + 8)
                frames.append(ChatFrame(owned, system + 7, close[0], system_end, "system"))
                owned = body = system_end
        close = next(_candidates(payload, body, (spec.separator,)), None)
        if close is None:
            frames.append(
                ChatFrame(owned, body, len(payload), len(payload), "user", complete=False)
            )
            diagnostics.append(_diagnostic("chat_incomplete_body", body, len(payload), spec.name))
            break
        prompt_end = close[0] + len(spec.separator)
        frames.append(ChatFrame(owned, body, close[0], prompt_end, "user"))
        owned = prompt_end
        next_end = next(_candidates(payload, prompt_end, spec.ends), None)
        if next_end is None:
            if prompt_end < len(payload):
                frames.append(
                    ChatFrame(owned, owned, len(payload), len(payload), "assistant", complete=False)
                )
                diagnostics.append(
                    _diagnostic("chat_incomplete_body", owned, len(payload), spec.name)
                )
            break
        end, marker = next_end
        following = _prefix(payload, spec, end + len(marker))
        frames.append(ChatFrame(owned, owned, end, following, "assistant"))
        owned = position = following
    return ChatParse(spec.name, tuple(frames), tuple(diagnostics), bool(frames))


def _deepseek(payload: bytes, spec: ChatTemplate, max_frames: int) -> ChatParse:
    position = _prefix(payload, spec)
    user, assistant = spec.start, spec.separator
    markers = (user, assistant, *spec.ends)
    frames: list[ChatFrame] = []
    diagnostics: list[Diagnostic] = []
    owned = 0
    if not payload.startswith((user, assistant), position):
        # An unlabeled system prefix is legal only behind this family's BOS.
        if position == _whitespace(payload, 0):
            return ChatParse(spec.name)
        first = next(_candidates(payload, position, (user, assistant)), None)
        if first is None:
            return ChatParse(spec.name)
        frames.append(ChatFrame(0, position, first[0], first[0], "system"))
        owned = position = first[0]
    while position < len(payload):
        marker = (
            user
            if payload.startswith(user, position)
            else assistant
            if payload.startswith(assistant, position)
            else None
        )
        if marker is None or len(frames) >= max_frames:
            frames.append(ChatFrame(owned, owned, len(payload), len(payload), None, complete=False))
            diagnostics.append(_diagnostic("chat_unframed_tail", owned, len(payload), spec.name))
            break
        role = "user" if marker == user else "assistant"
        body = position + len(marker)
        next_marker = next(_candidates(payload, body, markers), None)
        if next_marker is None:
            frames.append(ChatFrame(owned, body, len(payload), len(payload), role, complete=False))
            diagnostics.append(_diagnostic("chat_incomplete_body", body, len(payload), spec.name))
            break
        end, terminator = next_marker
        following = (
            end if terminator in {user, assistant} else _whitespace(payload, end + len(terminator))
        )
        frames.append(ChatFrame(owned, body, end, following, role))
        owned = position = following
    return ChatParse(spec.name, tuple(frames), tuple(diagnostics), bool(frames))


def parse_chat(
    payload: bytes,
    template: str = "auto",
    *,
    max_frames: int = 20_000,
) -> ChatParse:
    """Frame a serialized chat only from a recognized, anchored source prefix.

    ``none`` disables inference for already-framed messages. ``auto`` requires
    at least one complete frame and refuses competing template parses. Unknown
    explicit names, malformed prefixes and ambiguous auto detection fall back
    to ordinary content with diagnostics. Explicit known templates retain final
    incomplete bodies/headers. Source ranges are contiguous and lossless.
    """
    if not isinstance(payload, bytes):
        raise TypeError("Chat framing requires immutable source bytes")
    if not isinstance(template, str):
        raise TypeError("Chat template must be a string")
    if type(max_frames) is not int or max_frames < 1:
        raise ValueError("Chat max_frames must be a positive integer")
    selected = template.strip().casefold()
    if selected == "none":
        return ChatParse("none")
    candidates = [
        spec for name, spec in CHAT_TEMPLATES.items() if selected in {name, *spec.aliases}
    ]
    if selected != "auto" and not candidates:
        return ChatParse(
            selected, diagnostics=(_diagnostic("chat_unknown_template", 0, len(payload), selected),)
        )
    if selected == "auto":
        candidates = [
            spec
            for spec in CHAT_TEMPLATES.values()
            if payload.startswith(spec.start, _prefix(payload, spec))
            or (
                spec.mode == "deepseek"
                and any(
                    payload.removeprefix(b"\xef\xbb\xbf").lstrip(_SPACE).startswith(marker)
                    for marker in spec.prefixes
                )
            )
        ]
    parsed: list[ChatParse] = []
    attempted: list[Diagnostic] = []
    for spec in candidates:
        parser = {"delimited": _delimited, "inst": _inst, "deepseek": _deepseek}[spec.mode]
        result = parser(payload, spec, max_frames)
        result.validate(len(payload))
        if result.detected and (
            selected != "auto" or any(frame.complete for frame in result.frames)
        ):
            parsed.append(result)
        else:
            attempted.extend(result.diagnostics)
    if len(parsed) == 1:
        return parsed[0]
    if len(parsed) > 1:
        return ChatParse(
            selected,
            diagnostics=(
                _diagnostic(
                    "chat_ambiguous_template",
                    0,
                    len(payload),
                    ",".join(item.template for item in parsed),
                ),
            ),
        )
    return ChatParse(
        selected,
        diagnostics=tuple(attempted)
        or (_diagnostic("chat_no_template", 0, len(payload), "No anchored known chat template"),),
    )
