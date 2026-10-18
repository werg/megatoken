"""Source-faithful framing, literal shielding, and trusted adapter boundaries."""

from __future__ import annotations

import json
import random
from dataclasses import replace

import pytest

from megatoken.data.adapters import AdapterContext, ChatFramingError, SchemaError, adapt_row
from megatoken.data.chunking import ChunkingConfig
from megatoken.data.parsing.chat import (
    CHAT_TEMPLATES,
    ChatFrame,
    ChatParse,
    parse_chat,
    template_identity,
)
from megatoken.data.schema import make_records


class ByteCodec:
    fingerprint = "chat-test-bytes-v1"

    def encode(self, payload):
        return list(payload)

    def decode(self, ids):
        return bytes(ids)


class CrossFrameCodec(ByteCodec):
    fingerprint = "chat-cross-frame-review-v1"
    merged = b"<|im_end|><|im_start|>"

    def encode(self, payload):
        ids = []
        offset = 0
        while offset < len(payload):
            if payload.startswith(self.merged, offset):
                ids.append(256)
                offset += len(self.merged)
            else:
                ids.append(payload[offset])
                offset += 1
        return ids

    def decode(self, ids):
        return b"".join(self.merged if token == 256 else bytes([token]) for token in ids)


def assert_partition(payload, result):
    assert result.detected
    result.validate(len(payload))
    assert b"".join(payload[frame.start : frame.end] for frame in result.frames) == payload
    for frame in result.frames:
        assert (
            b"".join(
                payload[left:right]
                for left, right in (
                    frame.header,
                    frame.body,
                    frame.footer,
                )
            )
            == payload[frame.start : frame.end]
        )


@pytest.mark.parametrize(
    ("template", "payload", "bodies"),
    [
        (
            "chatml",
            b"<|im_start|>user\nHi!<|im_end|>\n<|im_start|>assistant\nHello.<|im_end|>\n",
            [b"Hi!", b"Hello."],
        ),
        (
            "llama3",
            b"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nHi!<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nHello.<|eot_id|>",
            [b"Hi!", b"Hello."],
        ),
        (
            "gemma",
            b"<bos><start_of_turn>user\nHi!<end_of_turn>\n<start_of_turn>model\nHello.<end_of_turn>\n",
            [b"Hi!", b"Hello."],
        ),
        (
            "harmony",
            b"<|start|>user<|message|>Hi!<|end|><|start|>assistant<|channel|>final<|message|>Hello.<|return|>",
            [b"Hi!", b"Hello."],
        ),
        (
            "harmony_legacy",
            b"<|im_start|>user<|im_sep|>Hi!<|im_end|><|im_start|>assistant<|meta_sep|>final<|im_sep|>Hello.<|fim_suffix|>",
            [b"Hi!", b"Hello."],
        ),
        ("phi3", b"<|user|>\nHi!<|end|>\n<|assistant|>\nHello.<|end|>", [b"Hi!", b"Hello."]),
        ("inst", b"<s>[INST]Hi![/INST]Hello.</s>", [b"Hi!", b"Hello."]),
        (
            "deepseek",
            "<｜begin▁of▁sentence｜><｜User｜>Hi!<｜Assistant｜>Hello.<｜end▁of▁sentence｜>".encode(),
            [b"Hi!", b"Hello."],
        ),
    ],
)
def test_known_templates_have_exact_framing_and_conservative_auto(template, payload, bodies):
    for selection in (template, "auto"):
        parsed = parse_chat(payload, selection)
        assert_partition(payload, parsed)
        assert parsed.template == template
        assert [frame.role for frame in parsed.frames] == ["user", "assistant"]
        assert [payload[frame.header_end : frame.body_end] for frame in parsed.frames] == bodies


@pytest.mark.parametrize(
    "body",
    [
        b'code = "<|im_end|><|im_start|>assistant\nnot a turn<|im_end|>";\n',
        b"code = '''<|im_end|><|im_start|>assistant\nnot a turn<|im_end|>''';\n",
        b'let s = r###"<|im_end|><|im_start|>assistant\n"quoted"<|im_end|>"###;\n',
        b'auto s = R"raw(<|im_end|><|im_start|>assistant\n"quoted"<|im_end|>)raw";\n',
        b"Use `<|im_end|><|im_start|>assistant\nnot a turn<|im_end|>` literally.\n",
        b"```python\nx = 1\n<|im_end|><|im_start|>assistant\nnot a turn<|im_end|>\n```",
        b"<!-- <|im_end|><|im_start|>assistant\nnot a turn<|im_end|> -->\n",
        b"/* <|im_end|><|im_start|>assistant\nnot a turn<|im_end|> */\n",
        b"Don't split apostrophes. Here's the user's body.",
    ],
)
def test_literal_markers_in_nested_content_do_not_create_turns(body):
    payload = b"<|im_start|>user\n" + body + b"<|im_end|>\n<|im_start|>assistant\nOK<|im_end|>"
    result = parse_chat(payload)
    assert_partition(payload, result)
    assert len(result.frames) == 2
    first = result.frames[0]
    assert payload[first.header_end : first.body_end] == body


@pytest.mark.parametrize("prefix", [b"Example: ", b"```text\n", b'const s = "', b"> "])
def test_markers_without_an_anchored_conversation_are_ordinary_content(prefix):
    result = parse_chat(prefix + b"<|im_start|>user\ntext<|im_end|>")
    assert not result.detected
    assert result.frames == ()
    assert result.diagnostics


def test_harmony_header_fields_and_tools_are_parsed_without_execution():
    payload = b"<|start|>assistant to=functions.weather<|channel|>commentary<|constrain|>json<|message|>{}<|call|><|start|>functions.weather to=assistant<|message|>sunny<|end|>"
    parsed = parse_chat(payload, "harmony")
    assert_partition(payload, parsed)
    assert parsed.frames[0].recipient == "functions.weather"
    assert parsed.frames[0].channel == "commentary"
    assert parsed.frames[1].role == "tool"
    assert parsed.frames[1].name == "functions.weather"


def test_inst_embedded_system_and_repeated_rounds_are_lossless():
    payload = b"<s>[INST] <<SYS>>\nBe concise.\n<</SYS>>\n\nQuestion? [/INST] Answer.</s><s>[INST] Next? [/INST] More.</s>"
    parsed = parse_chat(payload, "llama2")
    assert_partition(payload, parsed)
    assert [frame.role for frame in parsed.frames] == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert payload[slice(*parsed.frames[0].body)] == b"\nBe concise.\n"


def test_deepseek_system_and_unicode_coordinates():
    payload = "<｜begin▁of▁sentence｜>系统。<｜User｜>你好。<｜Assistant｜>欢迎。<｜end▁of▁sentence｜>".encode()
    parsed = parse_chat(payload)
    assert_partition(payload, parsed)
    assert [frame.role for frame in parsed.frames] == ["system", "user", "assistant"]
    assert payload[slice(*parsed.frames[1].body)] == "你好。".encode()


def test_incomplete_headers_bodies_bom_crlf_and_invalid_utf8_are_preserved():
    payload = b"\xef\xbb\xbf \r\n<|im_start|>user\r\nHi\xff!<|im_end|>\r\n<|im_start|>assistant"
    parsed = parse_chat(payload)
    assert_partition(payload, parsed)
    assert parsed.frames[-1].role == "assistant"
    assert not parsed.frames[-1].complete
    assert parsed.frames[-1].body == (len(payload), len(payload))
    assert any(item.code == "chat_incomplete_header" for item in parsed.diagnostics)
    incomplete = b"<|im_start|>assistant\npartial answer"
    assert not parse_chat(incomplete).detected
    assert_partition(incomplete, parse_chat(incomplete, "chatml"))
    header = b"<|im_start|>assistant"
    assert not parse_chat(header).detected
    assert_partition(header, parse_chat(header, "chatml"))


def test_unknown_ambiguous_disabled_and_invalid_templates_fail_conservatively():
    payload = b"<|im_start|>user\nHi<|im_end|>"
    assert parse_chat(payload, "none") == ChatParse("none")
    assert parse_chat(payload, "made-up").diagnostics[0].code == "chat_unknown_template"
    registry = dict(CHAT_TEMPLATES)
    registry["duplicate"] = replace(registry["chatml"], name="duplicate")
    ambiguous = parse_chat(payload, templates=registry)
    assert not ambiguous.detected
    assert ambiguous.diagnostics[0].code == "chat_ambiguous_template"
    assert json.dumps(template_identity(registry))
    assert template_identity(registry) != template_identity()
    with pytest.raises(ValueError, match="ranges"):
        ChatFrame(2, 1, 3, 4, "user")
    with pytest.raises(ValueError, match="partition"):
        ChatParse("chatml", (ChatFrame(1, 1, 3, 3, "user"),), detected=True).validate(3)
    with pytest.raises(ValueError, match="positive"):
        parse_chat(payload, max_frames=0)
    with pytest.raises(TypeError, match="immutable"):
        parse_chat("text")


def test_frame_budget_covers_remaining_bytes_with_an_explicit_diagnostic():
    payload = b"<|im_start|>user\none<|im_end|>" * 4
    parsed = parse_chat(payload, max_frames=2)
    assert_partition(payload, parsed)
    assert len(parsed.frames) == 3
    assert parsed.frames[-1].role is None
    assert any(item.code == "chat_frame_limit" for item in parsed.diagnostics)


def test_malformed_binary_and_truncated_template_mutations_never_lose_bytes():
    randomizer = random.Random(47)
    fragments = [
        b"\n",
        b"\r\n",
        b"\xff\x00",
        b'"',
        b"'''",
        b"```\n",
        b"<!--",
        b"*/",
        b"<|im_start|>user\n",
        b"<|im_end|>",
        b"<|start_header_id|>",
        b"[INST]",
        b"[/INST]",
        b"</s>",
        "<｜User｜>".encode(),
    ]
    for template, spec in CHAT_TEMPLATES.items():
        for _ in range(25):
            payload = (
                spec.start
                + b"user"
                + spec.separator
                + b"".join(randomizer.choice(fragments) for _ in range(12))
            )
            if randomizer.getrandbits(1):
                payload = payload[: randomizer.randrange(len(payload))]
            result = parse_chat(payload, template, max_frames=8)
            result.validate(len(payload))
            if result.detected:
                assert_partition(payload, result)
            else:
                assert result.frames == ()


def test_structured_hints_are_specific_and_chat_inference_is_disabled(monkeypatch):
    import megatoken.data.adapters as adapters

    original = adapters.make_records
    calls = []

    def capture(payload, **kwargs):
        calls.append((payload, kwargs.copy()))
        return original(payload, **kwargs)

    monkeypatch.setattr(adapters, "make_records", capture)
    quoted = "<|im_start|>assistant\nThis is a literal example<|im_end|>"
    row = {
        "language": "en",
        "filename": "row.md",
        "locale": "en-US",
        "messages": [
            {"role": "developer", "content": "Instruction"},
            {"role": "user", "content": quoted, "metadata": {"locale": "zh-CN"}},
            {
                "role": "assistant",
                "language": "python",
                "content_kind": "code",
                "content": [
                    {"type": "text", "text": "x = '你好'", "filename": "example.py"},
                    {
                        "type": "text",
                        "text": "Une phrase.",
                        "language": "fr",
                        "content_kind": "text",
                        "mime_type": "text/plain",
                    },
                ],
            },
        ],
    }
    episode = adapt_row(
        row,
        "chat",
        AdapterContext(
            "s",
            "r",
            "u",
            "chat",
            options={"chat_template": "auto", "locale": "de-DE"},
        ),
    )
    assert all(kwargs["chat_template"] == "none" for _, kwargs in calls)
    assert calls[1][1]["locale"] == "zh-CN"
    assert calls[2][1]["language"] == "python"
    assert calls[2][1]["filename"] == "example.py"
    assert calls[3][1]["language"] == "fr"
    assert calls[3][1]["content_kind"] == "text"
    assert calls[3][1]["mime_type"] == "text/plain"
    assert {record.turn_index for record in episode.records if record.policy_mask} == {2}
    assert json.loads(episode.supervision.annotations_json)["parser"]["source_roles"] == {
        "0": "developer"
    }
    assert episode.records[0].role == "system"


def test_document_hints_and_explicit_template_reach_parser(monkeypatch):
    import megatoken.data.adapters as adapters

    original = adapters.make_records
    calls = []

    def capture(payload, **kwargs):
        calls.append(kwargs)
        return original(payload, **kwargs)

    monkeypatch.setattr(adapters, "make_records", capture)
    adapt_row(
        {
            "text": "hello",
            "metadata": {"language": "ar", "locale": "ar-EG", "mime_type": "text/plain"},
            "filename": "document.txt",
        },
        "document",
        AdapterContext(
            "s",
            "r",
            "u",
            "text",
            options={"chat_template": "none"},
        ),
    )
    assert calls[0]["language"] == "ar"
    assert calls[0]["locale"] == "ar-EG"
    assert calls[0]["filename"] == "document.txt"
    assert calls[0]["mime_type"] == "text/plain"
    assert calls[0]["chat_template"] == "none"


def test_role_owned_records_preserve_one_canonical_stream_for_literal_template_text():
    payload = b"<|im_start|>user\nHi<|im_end|><|im_start|>assistant\nHello<|im_end|>"
    codec = CrossFrameCodec()
    records = make_records(
        payload,
        logical_id="user-message",
        role="user",
        codec=codec,
        content_kind="text",
    )
    assert len(records) == 1
    assert records[0].payload == payload
    assert records[0].cached_ids == tuple(codec.encode(payload))
    assert records[0].chunk_boundary != "chat"
    for template in ("auto", "chatml"):
        with pytest.raises(ValueError, match="serialized_chat adapter"):
            make_records(payload, logical_id="field", role="document", chat_template=template)


def test_document_auto_framing_owns_roles_masks_and_canonical_fields_before_bpe():
    payload = (
        b"<|im_start|>developer\nRules<|im_end|>"
        b"<|im_start|>user\nQuestion?<|im_end|>"
        b"<|im_start|>assistant\nEarlier answer<|im_end|>"
        b"<|im_start|>user\nFollow up?<|im_end|>"
        b"<|im_start|>assistant\nFinal answer<|im_end|>"
    )
    codec = CrossFrameCodec()
    context = AdapterContext("s", "r", "u", "documents", codec=codec, max_record_tokens=11)
    row = {"text": payload.decode(), "train_turns_mask": [False, False, False, False, True]}
    episode = adapt_row(row, "document", context)
    assert b"".join(record.payload for record in episode.records) == payload
    assert (
        b"".join(record.payload for record in episode.records if record.policy_mask)
        == b"Final answer"
    )
    expected_roles = {0: "system", 1: "user", 2: "assistant", 3: "user", 4: "assistant"}
    assert {record.turn_index: record.role for record in episode.records} == expected_roles
    assert all(record.provenance == "natural" for record in episode.records)
    assert all(
        payload[record.source_span.start : record.source_span.end] == record.payload
        for record in episode.records
    )
    groups = list(episode.iter_logical_records())
    assert len(groups) == 15  # Header, body, and footer are distinct logical fields.
    for group in groups:
        assert tuple(token for record in group for token in record.cached_ids) == tuple(
            codec.encode(b"".join(record.payload for record in group))
        )
    # Cross-frame BPE merges are deliberately unavailable after authoritative framing.
    assert tuple(token for record in episode.records for token in record.cached_ids) != tuple(
        codec.encode(payload)
    )
    assert adapt_row(episode.to_dict(), "native", context).records == episode.records
    framing = json.loads(episode.supervision.annotations_json)["parser"]["chat_framing"]
    assert framing["template"] == "chatml" and framing["frames"][0]["role"] == "developer"


@pytest.mark.parametrize("disable_in", ["options", "row", "policy", "binary"])
def test_document_can_explicitly_preserve_a_template_as_one_literal_field(disable_in):
    payload = b"<|im_start|>user\nHi<|im_end|><|im_start|>assistant\nHello<|im_end|>"
    codec = CrossFrameCodec()
    row = {"text": payload.decode()}
    options = {"chat_template": "none"} if disable_in == "options" else {}
    if disable_in == "row":
        row["chat_template"] = "none"
    if disable_in == "binary":
        row["content_kind"] = "binary"
    policy = ChunkingConfig(chat_template="none" if disable_in == "policy" else "auto")
    context = AdapterContext(
        "s", "r", "u", "documents", codec=codec, options=options, chunking=policy
    )
    episode = adapt_row(row, "document", context)
    assert len(list(episode.iter_logical_records())) == 1
    assert all(record.role == "document" and record.policy_mask for record in episode.records)
    assert tuple(token for record in episode.records for token in record.cached_ids) == tuple(
        codec.encode(payload)
    )
    assert adapt_row(episode.to_dict(), "native", context).records == episode.records


def test_document_framing_fallback_keeps_diagnostics_and_original_bytes():
    payload = b"Unrecognized template content."
    episode = adapt_row(
        {"text": payload.decode()},
        "document",
        AdapterContext("s", "r", "u", "documents", options={"chat_template": "unknown"}),
    )
    assert b"".join(record.payload for record in episode.records) == payload
    assert all("chat_unknown_template" in record.chunk_diagnostics for record in episode.records)
    framing = json.loads(episode.supervision.annotations_json)["parser"]["chat_framing"]
    assert framing["frames"] == []
    assert framing["diagnostics"][0]["code"] == "chat_unknown_template"


def test_document_framing_honors_the_configured_frame_resource_limit():
    from megatoken.data.parsing.types import ParseLimits

    policy = ChunkingConfig(limits=ParseLimits(max_regions=2))
    context = AdapterContext("s", "r", "u", "documents", chunking=policy, codec=ByteCodec())
    with pytest.raises(ChatFramingError) as caught:
        adapt_row({"text": "<|im_start|>user\nHi<|im_end|>" * 4}, "document", context)
    assert "chat_frame_limit" in caught.value.diagnostics


def test_serialized_adapter_has_exact_spans_masks_and_no_cross_turn_records():
    payload = b"<|im_start|>developer\nRules.<|im_end|>\n<|im_start|>user\nQuestion?<|im_end|>\n<|im_start|>assistant\nAnswer.<|im_end|>\n"
    episode = adapt_row(
        {
            "text": payload.decode(),
            "train_turns_mask": [False, False, True],
            "metadata": {"gold_outcome": "SECRET"},
        },
        "serialized_chat",
        AdapterContext(
            "s",
            "r",
            "u",
            "chat",
            codec=ByteCodec(),
            max_record_bytes=8,
            options={"chat_template": "chatml"},
        ),
    )
    assert b"".join(record.payload for record in episode.records) == payload
    assert (
        b"".join(record.payload for record in episode.records if record.policy_mask) == b"Answer."
    )
    assert {record.turn_index for record in episode.records if record.policy_mask} == {2}
    cursor = 0
    for record in episode.records:
        assert record.source_span.start == cursor
        assert record.source_span.field == "text"
        assert record.payload == payload[cursor : record.source_span.end]
        assert record.cached_ids == tuple(record.payload)
        cursor = record.source_span.end
    assert b"SECRET" not in b"".join(record.payload for record in episode.model_records())
    for group in episode.iter_logical_records():
        assert tuple(token for record in group for token in record.cached_ids) == tuple(
            b"".join(record.payload for record in group)
        )
        assert len({record.turn_index for record in group}) == 1


def test_serialized_adapter_rejects_unsafe_roles_and_handles_incomplete_context():
    context = AdapterContext("s", "r", "u", "chat", options={"chat_template": "chatml"})
    with pytest.raises(SchemaError, match="no unambiguous"):
        adapt_row(
            {"text": "A literal <|im_start|>user\nExample<|im_end|>"}, "serialized_chat", context
        )
    with pytest.raises(SchemaError, match="call/result IDs"):
        adapt_row({"text": "<|im_start|>tool\nResult<|im_end|>"}, "serialized_chat", context)
    with pytest.raises(SchemaError, match="incomplete"):
        adapt_row({"text": "<|im_start|>assistant\nunfinished"}, "serialized_chat", context)
    partial = adapt_row(
        {"text": "<|im_start|>assistant\nunfinished"},
        "serialized_chat",
        replace(
            context,
            options={"chat_template": "chatml", "allow_incomplete_serialized": True},
        ),
    )
    assert not any(record.policy_mask for record in partial.records)
    with pytest.raises(SchemaError, match="one boolean"):
        adapt_row(
            {"text": "<|im_start|>user\nHi<|im_end|>", "train_turns_mask": []},
            "serialized_chat",
            context,
        )


def test_serialized_literal_tool_examples_and_public_recipient_are_content():
    body = b"Example:\n```xml\n<tool_call>{}</tool_call>\n```"
    payload = b"<|start|>assistant to=all<|channel|>final<|message|>" + body + b"<|return|>"
    episode = adapt_row(
        {"text": payload.decode()},
        "serialized_chat",
        AdapterContext("s", "r", "u", "chat", options={"chat_template": "harmony"}),
    )
    assert b"".join(record.payload for record in episode.records if record.policy_mask) == body


def test_tiny_user_and_tool_events_never_share_a_logical_record():
    row = {
        "messages": [
            {"role": "user", "content": "a"},
            {
                "role": "assistant",
                "tool_calls": [{"id": "c1", "function": {"name": "f", "arguments": {}}}],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "b"},
            {"role": "user", "content": "c"},
            {"role": "assistant", "content": "d"},
        ]
    }
    episode = adapt_row(row, "chat", AdapterContext("s", "r", "u", "chat", codec=ByteCodec()))
    groups = list(episode.iter_logical_records())
    assert len(groups) == 5
    assert [group[0].turn_index for group in groups] == [0, 1, 2, 3, 4]
    assert [group[0].role for group in groups] == ["user", "assistant", "tool", "user", "assistant"]
    assert groups[2][0].result_of == episode.actions[0].action_id
