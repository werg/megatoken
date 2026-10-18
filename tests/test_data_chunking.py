import json
from dataclasses import replace

import pytest

from megatoken.data import (
    ChunkingConfig,
    DedupConfig,
    Dependency,
    Episode,
    Lineage,
    PhysicalRecord,
    Supervision,
    TermsReview,
    WindowConfig,
    encode_record,
    iter_windows,
    make_records,
    pin_local_source,
    prepare_sources,
    segment_bytes,
)
from megatoken.generation.codec import ByteCodec


def test_chunking_defaults_explicit_stage_caps_and_identity():
    codec = ByteCodec.byte_fallback()
    config = ChunkingConfig()
    assert (config.target_tokens, config.soft_max_tokens, config.hard_max_tokens) == (
        256,
        512,
        1024,
    )
    assert config.with_hard_limit(256) == ChunkingConfig(256, 256, 256)
    assert config.with_hard_limit(64) == ChunkingConfig(64, 64, 64)
    assert ChunkingConfig(**config.to_dict()).identity(codec) == config.identity(codec)
    assert ChunkingConfig.from_identity(config.identity(codec), codec) == config
    altered = config.identity(codec) | {"python_ast": [0, 0]}
    with pytest.raises(ValueError, match="Incompatible chunker"):
        ChunkingConfig.from_identity(altered, codec)
    with pytest.raises(ValueError, match="Incompatible chunker"):
        ChunkingConfig.from_identity(config.identity(codec) | {"fingerprint": "tampered"}, codec)
    assert replace(config, target_tokens=128).fingerprint != config.fingerprint
    with pytest.raises(ValueError, match="Unsupported chunking version"):
        ChunkingConfig(version="unknown")


def test_coherent_paragraphs_up_to_soft_cap_and_small_paragraph_packing():
    codec = ByteCodec.byte_fallback()
    first = b"First paragraph. " * 25 + b"\r\n\r\n"
    second = b"Second paragraph. " * 24 + b"\r\n"
    chunks = segment_bytes(first + second, codec)
    assert [chunk.payload for chunk in chunks] == [first, second]
    assert all(256 < len(chunk.token_ids) <= 512 for chunk in chunks)
    small = b"Small paragraph.\n\n"
    chunks = segment_bytes(small * 30, codec)
    assert len(chunks) > 1
    assert all(chunk.payload == small * (len(chunk.payload) // len(small)) for chunk in chunks)
    assert abs(len(chunks[0].token_ids) - 256) <= len(small)
    assert b"".join(chunk.payload for chunk in chunks) == small * 30


def test_fenced_code_python_functions_and_list_items_are_not_cut_internally():
    codec = ByteCodec.byte_fallback()
    function = b"@decorator\ndef example():\n" + b"    value = 3\n" * 24 + b"    return value\n"
    fence = b"```python\n" + function + b"```\n"
    chunks = segment_bytes(fence + b"\n" + fence, codec)
    ends = {chunk.byte_end for chunk in chunks}
    assert not any(0 < end < len(fence) for end in ends)
    assert not any(len(fence) + 1 < end < 2 * len(fence) + 1 for end in ends)
    plain = function + b"\n" + function.replace(b"example", b"another")
    chunks = segment_bytes(plain, codec, content_kind="code", language="python")
    assert not any(0 < chunk.byte_end < len(function) for chunk in chunks)
    assert b"".join(chunk.payload for chunk in chunks) == plain

    item = b"- First item " + b"with meaningful content " * 13 + b"\n  continuation.\n"
    second = item.replace(b"First", b"Other")
    chunks = segment_bytes(item + second, codec)
    assert [chunk.payload for chunk in chunks] == [item, second]


def test_brace_code_ignores_braces_and_semicolons_inside_strings_and_comments():
    codec = ByteCodec.byte_fallback()
    function = (
        b"function first() {\n"
        b'  const quoted = "a } brace; and an escaped \\" quote";\n'
        b"  /* comments { are not blocks } */\n" + b"  value += 1;\n" * 19 + b"  return value;\n}\n"
    )
    payload = function + function.replace(b"first", b"other")
    chunks = segment_bytes(payload, codec, content_kind="code", language="javascript")
    assert not any(0 < chunk.byte_end < len(function) for chunk in chunks)
    assert b"".join(chunk.payload for chunk in chunks) == payload


def test_oversized_text_uses_sentences_then_reports_unavoidable_token_splits():
    codec = ByteCodec.byte_fallback()
    payload = b"A sentence with several distinct words. " * 100
    chunks = segment_bytes(payload, codec)
    assert all(len(chunk.token_ids) <= 512 for chunk in chunks)
    assert all(chunk.boundary == "sentence" for chunk in chunks[:-1])
    assert all(not chunk.forced for chunk in chunks)
    atom = b"x" * 2500
    chunks = segment_bytes(atom, codec)
    assert b"".join(chunk.payload for chunk in chunks) == atom
    assert any(chunk.forced for chunk in chunks)
    assert all(len(chunk.token_ids) <= 1024 for chunk in chunks)


def test_exact_unicode_crlf_binary_and_empty_payloads():
    codec = ByteCodec.byte_fallback()
    payload = ("日本語 🙂 e\u0301\r\n" * 90).encode()
    chunks = segment_bytes(payload, codec, ChunkingConfig(17, 23, 29, max_bytes=35))
    assert b"".join(chunk.payload for chunk in chunks) == payload
    assert all(chunk.payload.decode("utf-8") for chunk in chunks)
    assert all(chunk.payload == payload[chunk.byte_start : chunk.byte_end] for chunk in chunks)
    assert not any(
        left.payload.endswith(b"\r") and right.payload.startswith(b"\n")
        for left, right in zip(chunks, chunks[1:], strict=False)
    )
    assert tuple(token for chunk in chunks for token in chunk.token_ids) == tuple(
        codec.encode(payload)
    )
    binary = bytes(range(256)) * 9
    assert b"".join(chunk.payload for chunk in segment_bytes(binary, codec)) == binary
    empty = segment_bytes(b"", codec)
    assert len(empty) == 1 and empty[0].payload == b"" and empty[0].token_ids == ()
    for payload in ("🙂".encode(), b"\r\n"):
        chunks = segment_bytes(payload, codec, ChunkingConfig(1, 1, 4))
        assert len(chunks) == 1 and chunks[0].payload == payload
        tiny = segment_bytes(payload, codec, ChunkingConfig(1, 1, 1))
        assert b"".join(chunk.payload for chunk in tiny) == payload
        assert any(chunk.forced for chunk in tiny)


class BoundaryMergeCodec:
    fingerprint = "boundary-merge-v1"
    merged = b"\n\nB"

    def encode(self, payload):
        ids = []
        index = 0
        while index < len(payload):
            if payload.startswith(self.merged, index):
                ids.append(256)
                index += len(self.merged)
            else:
                ids.append(payload[index])
                index += 1
        return ids

    def decode(self, ids):
        return b"".join(self.merged if token == 256 else bytes([token]) for token in ids)


def test_bpe_crossing_semantic_boundary_preserves_reference_ids_and_inventory():
    codec = BoundaryMergeCodec()
    payload = (b"a" * 30 + b"\n\nB" + b"b" * 30) * 8
    totals = set()
    for hard in (24, 48, 96):
        config = ChunkingConfig(min(16, hard), min(24, hard), hard)
        records = make_records(
            payload,
            logical_id="logical",
            role="document",
            policy_mask=True,
            codec=codec,
            max_tokens=hard,
            chunking=config,
        )
        assert b"".join(record.payload for record in records) == payload
        assert tuple(token for record in records for token in record.cached_ids) == tuple(
            codec.encode(payload)
        )
        totals.add(sum(encode_record(record, codec).tokens for record in records))
    assert totals == {len(codec.encode(payload))}


def test_document_windows_are_bounded_and_writer_reader_extensions_are_explicit():
    codec = ByteCodec.byte_fallback()
    payload = (b"A paragraph with a complete statement. " * 3 + b"\n\n") * 30
    records = make_records(
        payload,
        logical_id="document",
        role="document",
        policy_mask=True,
        codec=codec,
        chunking=ChunkingConfig(64, 128, 256),
    )
    episode = Episode("long", records, Lineage("fixture", "v1", "family"))
    config = WindowConfig(max_window_tokens=10000, max_decisions=99, max_records=3)
    windows = list(iter_windows(episode, codec, config))
    assert len(windows) > 1 and all(len(window.records) <= 3 for window in windows)
    assert [entry.record for window in windows for entry in window.records] == list(records)
    for window in windows:
        assert tuple(entry.record for entry in window.prefix) == records[: window.start]
    dependent = replace(
        episode, dependencies=(Dependency(records[0].record_id, records[-1].record_id),)
    )
    extended = list(iter_windows(dependent, codec, replace(config, respect_dependencies=True)))
    assert len(extended) == 1 and len(extended[0].writer_reader_pairs) == 1


def test_prepared_chunking_stats_and_policy_identity_are_durable(tmp_path):
    codec = ByteCodec.byte_fallback()
    text = ("A paragraph containing exact text. " * 8 + "\n\n") * 10
    path = tmp_path / "raw.jsonl"
    path.write_text(json.dumps({"id": "doc", "text": text, "family_id": "original"}) + "\n")
    source = pin_local_source(
        path,
        source_id="fixture",
        adapter="document",
        stream="natural_documents",
        terms=TermsReview(True, "fixture", reviewed_by="test", reviewed_at="2026-10-18"),
    )
    with prepare_sources(
        [source],
        tmp_path / "ready",
        codec,
        max_record_tokens=256,
        dedup=DedupConfig(train_fraction=1, validation_fraction=0),
    ) as prepared:
        manifest = prepared.manifest
        stats = manifest["chunking_stats"]
        assert manifest["chunking"]["config"] == ChunkingConfig(256, 256, 256).to_dict()
        assert (
            stats["reference_tokens"] == manifest["census"]["U"] == len(codec.encode(text.encode()))
        )
        assert stats["max_positions"] <= 256 and stats["unversioned_records"] == 0
        assert stats["records"] == sum(stats["histogram"].values())
        assert all(
            record.chunking_fingerprint == manifest["chunking"]["fingerprint"]
            for record in prepared[0].records
        )
    with pytest.raises(ValueError, match="different pipeline"):
        prepare_sources(
            [source],
            tmp_path / "ready",
            codec,
            max_record_tokens=256,
            chunking=ChunkingConfig(128, 256, 256),
            dedup=DedupConfig(train_fraction=1, validation_fraction=0),
        )


@pytest.mark.parametrize(
    ("payload_bytes", "physical_limit"),
    [(1025, {"max_record_tokens": 1024}), (1024, {"max_record_bytes": 1023})],
)
def test_oversized_native_generated_record_is_rejected_without_resegmentation(
    tmp_path, payload_bytes, physical_limit
):
    codec = ByteCodec.byte_fallback()
    payload = b"x" * payload_bytes
    record = PhysicalRecord(
        "native",
        "assistant",
        payload,
        policy_mask=True,
        provenance="student",
        emitted_ids=tuple(codec.encode(payload)),
        codec_fingerprint=codec.fingerprint,
    )
    episode = Episode("native", (record,), Lineage("fixture", "v1", "family"))
    path = tmp_path / "native.jsonl"
    path.write_text(json.dumps(episode.to_dict()) + "\n")
    original = path.read_bytes()
    source = pin_local_source(
        path,
        source_id="fixture",
        adapter="native",
        stream="student_corrections",
        terms=TermsReview(True, "fixture", reviewed_by="test", reviewed_at="2026-10-18"),
    )
    with pytest.raises(ValueError, match="cannot be silently rechunked"):
        prepare_sources([source], tmp_path / "ready", codec, **physical_limit)
    assert path.read_bytes() == original


def test_legacy_episode_and_native_tape_versions_fail_explicitly():
    episode = Episode("e", (PhysicalRecord("record", "user", b"source"),), Lineage("s", "v", "f"))
    value = episode.to_dict()
    assert value["schema_version"] == 2
    value["schema_version"] = 1
    with pytest.raises(ValueError, match="legacy records"):
        Episode.from_dict(value)
    with pytest.raises(ValueError, match="incompatible legacy"):
        Supervision(
            annotations_json=json.dumps(
                {"native_action_tapes": [{"tape": {"version": "v1", "carry": [0.0]}}]}
            )
        )
