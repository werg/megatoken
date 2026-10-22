import random
from dataclasses import replace

import pytest
from test_data_chunking import BoundaryMergeCodec

from megatoken.data.chunking import ChunkingConfig, segment_bytes, segment_with_report
from megatoken.data.parsing.planner import TokenIndex, plan_chunks
from megatoken.data.parsing.types import Boundary, ParseLimits, ParseResult, Region
from megatoken.generation.codec import ByteCodec


def test_lookahead_avoids_greedy_twenty_token_tail():
    codec = ByteCodec.byte_fallback()
    source = b"a" * 110 + b"b" * 100 + b"c" * 20
    parsed = ParseResult(
        boundaries=[Boundary(end, "paragraph") for end in (110, 210, 230)],
        regions=[
            Region(start, end, "paragraph") for start, end in ((0, 110), (110, 210), (210, 230))
        ],
    )
    plan = plan_chunks(TokenIndex.encode(source, codec), parsed, ChunkingConfig(100, 150, 200))
    # A closest-boundary greedy pass chooses 110, 100, 20. Full-path evaluation
    # sees that merging the final two intact paragraphs is the better plan.
    assert [part.end - part.start for part in plan.chunks] == [110, 120]
    assert all(not part.forced for part in plan.chunks)


def test_latency_boundaries_precede_tokenization_even_when_bpe_would_merge_them():
    class CrossFrameCodec(BoundaryMergeCodec):
        fingerprint = "cross-frame-fixture"
        merged = b"<|im_end|><|im_start|>"

    codec = CrossFrameCodec()
    source = b"<|im_start|>user\nHi<|im_end|><|im_start|>assistant\nHello<|im_end|>"
    report = segment_with_report(source, codec, chat_template="chatml")
    assert len(report.canonical_events) == 2
    assert len(report.chunks) == 2
    expected = tuple(
        token for start, end in report.canonical_events for token in codec.encode(source[start:end])
    )
    assert tuple(token for chunk in report.chunks for token in chunk.token_ids) == expected
    assert expected != tuple(codec.encode(source))
    assert b"".join(chunk.payload for chunk in report.chunks) == source
    assert report.chunks[0].byte_end == report.canonical_events[0][1]


def test_headings_and_long_function_signatures_stay_with_their_own_content():
    codec = ByteCodec.byte_fallback()
    first = b"# First chapter\n\n" + b"First paragraph has useful content. " * 12 + b"\n\n"
    following = (
        b"# Following chapter\n\n" + (b"This belongs to the following chapter. " * 20 + b"\n\n") * 3
    )
    report = segment_with_report(first + following, codec, content_kind="markdown")
    heading = len(first)
    assert not any(c.byte_start < heading < c.byte_end for c in report.chunks)
    assert not any(c.payload.rstrip().endswith(b"# Following chapter") for c in report.chunks)
    previous = b"def previous():\n" + b"    value = 1\n" * 28 + b"    return value\n\n"
    function = (
        b"@decorator\ndef next_function(argument):\n"
        + b"    value = argument + 1\n" * 100
        + b"    return value\n"
    )
    report = segment_with_report(previous + function, codec, content_kind="code", language="python")
    entry = len(previous)
    assert not any(c.byte_start < entry < c.byte_end for c in report.chunks)
    containing = next(c for c in report.chunks if c.byte_start <= entry < c.byte_end)
    assert b"def next_function(argument):\n    value" in containing.payload


def test_tree_distance_changes_packing_without_breaking_complete_small_functions():
    codec = ByteCodec.byte_fallback()
    function = b"def fn_00():\n" + b"    value = 1\n" * 6 + b"    return value\n\n"
    source = b"".join(function.replace(b"fn_00", f"fn_{i:02d}".encode()) for i in range(10))
    config = ChunkingConfig()
    plain = segment_with_report(
        source,
        codec,
        replace(config, tree_distance_weight=0),
        content_kind="code",
        language="python",
    )
    aware = segment_with_report(source, codec, config, content_kind="code", language="python")
    assert max(c.tree_distance for c in aware.chunks) < max(c.tree_distance for c in plain.chunks)
    assert len(aware.chunks) < 10  # Cohesion must not force singleton short functions.
    for chunk in aware.chunks:
        assert chunk.payload.count(b"def ") == chunk.payload.count(b"    return value")
    assert b"".join(c.payload for c in aware.chunks) == source


def test_attachment_expands_outward_past_a_bpe_edge_inside_crlf():
    class HeadingMergeCodec(BoundaryMergeCodec):
        fingerprint = "heading-crlf-fixture"
        merged = b"\n#"

    source = b"A" * 120 + b"\r\n#" + b"h" * 999 + b"\n" + b"z" * 120
    parsed = ParseResult(
        boundaries=[Boundary(122, "heading"), Boundary(1122, "paragraph")],
        regions=[Region(122, 1122, "heading_attachment", "markdown")],
    )
    index = TokenIndex.encode(source, HeadingMergeCodec())
    plan = plan_chunks(index, parsed, ChunkingConfig())
    cuts = [index.offsets[chunk.end] for chunk in plan.chunks]
    assert not any(122 < cut < 1122 for cut in cuts)
    assert 121 not in cuts  # CRLF is intact as well as the semantic attachment.
    assert all(chunk.end - chunk.start <= 1024 for chunk in plan.chunks)


@pytest.mark.parametrize("hard", [1, 2, 4, 17, 64, 257])
def test_arbitrary_bytes_tiny_caps_and_token_merges_always_make_progress(hard):
    generator = random.Random(260018 + hard)
    codec = BoundaryMergeCodec()
    config = ChunkingConfig(min(13, hard), min(29, hard), hard, max_bytes=max(3, hard))
    for _ in range(20):
        source = bytes(generator.randrange(256) for _ in range(generator.randrange(1, 300)))
        source += b"\n\nB" * 3
        chunks = segment_bytes(source, codec, config, content_kind="binary")
        assert b"".join(c.payload for c in chunks) == source
        assert tuple(t for c in chunks for t in c.token_ids) == tuple(codec.encode(source))
        assert all(
            0 < len(c.token_ids) <= hard and len(c.payload) <= config.max_bytes for c in chunks
        )
        assert chunks[0].byte_start == 0 and chunks[-1].byte_end == len(source)
        assert all(
            a.byte_end == b.byte_start and a.token_end == b.token_start
            for a, b in zip(chunks, chunks[1:], strict=False)
        )


def test_bounded_candidate_graph_for_large_indivisible_input():
    source = b"x" * 100_000
    config = ChunkingConfig(limits=ParseLimits(max_parse_bytes=4096))
    report = segment_with_report(
        source, ByteCodec.byte_fallback(), config, content_kind="text", chat_template="none"
    )
    assert report.planner_edges <= report.planner_candidates * config.planner_max_edges
    assert report.planner_candidates < len(source) // 8
    assert all(len(c.token_ids) <= config.hard_max_tokens for c in report.chunks)
    assert b"".join(c.payload for c in report.chunks) == source
    assert any(c.forced for c in report.chunks)


def test_config_roundtrip_pins_parser_limits_and_implementation_identity():
    codec = ByteCodec.byte_fallback()
    config = ChunkingConfig(limits=ParseLimits(max_depth=7, max_regions=1234))
    assert ChunkingConfig(**config.to_dict()) == config
    identity = config.identity(codec)
    assert ChunkingConfig.from_identity(identity, codec) == config
    assert "implementation" in identity["parsers"]
    assert (
        replace(config, limits=replace(config.limits, max_depth=6)).fingerprint
        != config.fingerprint
    )


def test_partial_utf8_and_crlf_only_when_the_physical_cap_requires_it():
    codec = ByteCodec.byte_fallback()
    text = ("e\u0301 👩🏾‍💻 🇮🇳\r\n" * 40).encode()
    chunks = segment_bytes(text, codec, ChunkingConfig(23, 37, 64), content_kind="text")
    from megatoken.data.parsing.prose import grapheme_offsets

    safe = set(grapheme_offsets(text))
    assert all(c.byte_start in safe and c.byte_end in safe for c in chunks)
    tiny = segment_bytes("👩🏾‍💻".encode(), codec, ChunkingConfig(1, 1, 1), content_kind="text")
    assert any("forced_token_cut" in c.diagnostics for c in tiny)
    assert b"".join(c.payload for c in tiny) == "👩🏾‍💻".encode()


def _assert_canonical_plan(index, plan, codec, config):
    assert plan.chunks[0].start == 0 and plan.chunks[-1].end == len(index.ids)
    assert all(
        left.end == right.start for left, right in zip(plan.chunks, plan.chunks[1:], strict=False)
    )
    pieces = [codec.decode(index.ids[chunk.start : chunk.end]) for chunk in plan.chunks]
    assert b"".join(pieces) == index.payload
    assert all(
        0 < chunk.end - chunk.start <= config.hard_max_tokens and len(piece) <= config.max_bytes
        for chunk, piece in zip(plan.chunks, pieces, strict=True)
    )


@pytest.mark.parametrize(
    ("sentence", "merged"),
    [
        (b"The next sentence has enough useful detail. ", b" The"),
        ("これは次の文です。".encode(), "。これ".encode()),
    ],
)
def test_adjacent_sentences_sharing_a_codec_token_remain_independently_packable(sentence, merged):
    class SentenceMergeCodec(BoundaryMergeCodec):
        fingerprint = "sentence-boundary-merge"

    codec = SentenceMergeCodec()
    codec.merged = merged
    source = sentence * 100
    ends = list(range(len(sentence), len(source) + 1, len(sentence)))
    parsed = ParseResult(
        boundaries=[Boundary(end, "sentence") for end in ends],
        regions=[
            Region(start, end, "sentence", "text")
            for start, end in zip([0, *ends[:-1]], ends, strict=True)
        ],
    )
    index, config = TokenIndex.encode(source, codec), ChunkingConfig()
    plan = plan_chunks(index, parsed, config)
    _assert_canonical_plan(index, plan, codec, config)
    assert all(256 <= chunk.end - chunk.start <= config.soft_max_tokens for chunk in plan.chunks)
    assert not any(chunk.forced for chunk in plan.chunks)
    assert not any(d.code == "crossing_protected_regions" for d in parsed.diagnostics)
    assert any(d.code == "semantic_boundary_projected" for d in parsed.diagnostics)
    assert all(0 <= d.start <= d.end <= len(source) for d in parsed.diagnostics)


def test_genuine_crossing_source_constraints_still_report_conflict():
    codec, config = ByteCodec.byte_fallback(), ChunkingConfig(40, 75, 100)
    index = TokenIndex.encode(b"a" * 140, codec)
    parsed = ParseResult(regions=[Region(0, 70, "sentence"), Region(60, 130, "sentence")])
    plan = plan_chunks(index, parsed, config)
    _assert_canonical_plan(index, plan, codec, config)
    crossing = [d for d in parsed.diagnostics if d.code == "crossing_protected_regions"]
    assert [(d.start, d.end) for d in crossing] == [(60, 130)]


def test_adjacent_atomic_source_units_can_relax_shared_token_ownership_without_quarantine():
    class AtomMergeCodec(BoundaryMergeCodec):
        fingerprint = "atomic-boundary-merge"
        merged = b"ab"

    codec, config = AtomMergeCodec(), ChunkingConfig(3, 4, 4)
    source = b"xxabyyy"
    index = TokenIndex.encode(source, codec)
    parsed = ParseResult(
        regions=[Region(0, 3, "heading_attachment"), Region(3, 7, "heading_attachment")],
        boundaries=[Boundary(3, "statement")],
    )
    plan = plan_chunks(index, parsed, config)
    _assert_canonical_plan(index, plan, codec, config)
    assert not any(d.code == "crossing_protected_regions" for d in parsed.diagnostics)
    assert any(d.code == "token_overlapping_protection" for d in parsed.diagnostics)
    assert len(plan.chunks) == 2


def test_oversized_heading_attachment_keeps_its_first_sentence_across_a_shared_bpe_edge():
    class SentenceMergeCodec(BoundaryMergeCodec):
        fingerprint = "heading-first-sentence-merge"
        merged = b" The"

    codec = SentenceMergeCodec()
    heading = b"# A useful title\r\n\r\n"
    first = b"The first sentence must stay with this title. "
    sentence = b"The next sentence has enough useful detail. "
    source = heading + first + sentence * 60
    body_start, first_end = len(heading), len(heading + first)
    ends = [first_end, *range(first_end + len(sentence), len(source) + 1, len(sentence))]
    parsed = ParseResult(
        boundaries=[Boundary(body_start, "heading"), *[Boundary(end, "sentence") for end in ends]],
        regions=[
            Region(0, body_start, "heading", "markdown"),
            Region(0, len(source), "heading_attachment", "markdown"),
            Region(body_start, len(source), "paragraph", "text"),
            *[
                Region(start, end, "sentence", "text")
                for start, end in zip([body_start, *ends[:-1]], ends, strict=True)
            ],
        ],
    )
    index, config = TokenIndex.encode(source, codec), ChunkingConfig()
    plan = plan_chunks(index, parsed, config)
    _assert_canonical_plan(index, plan, codec, config)
    assert index.offsets[plan.chunks[0].end] >= first_end
    assert all(256 <= part.end - part.start <= config.soft_max_tokens for part in plan.chunks)
    assert not any(d.code == "crossing_protected_regions" for d in parsed.diagnostics)


@pytest.mark.parametrize("unit_kind", ["heading_attachment", "signature_attachment"])
def test_oversized_first_unit_attaches_useful_prefix_after_indentation(unit_kind):
    codec, config = ByteCodec.byte_fallback(), ChunkingConfig(80, 120, 160, max_bytes=144)
    header = (
        b"# An attached title\r\n" if unit_kind == "heading_attachment" else b"def function():\r\n"
    )
    spaces, body = b" \t" * 38, b"meaningful " * 100
    source = header + spaces + body
    first = len(header + spaces)
    kind = "heading" if unit_kind == "heading_attachment" else "code_block"
    region_start, region_end = (0, len(header)) if kind == "heading" else (len(header), len(source))
    parsed = ParseResult(
        boundaries=[
            Boundary(len(header), "line"),
            *[Boundary(end, "word") for end in range(first + 11, len(source) + 1, 11)],
        ],
        regions=[
            Region(0, len(source), unit_kind),
            Region(region_start, region_end, kind),
            Region(first, len(source), "sentence" if kind == "heading" else "statement"),
        ],
    )
    index = TokenIndex.encode(source, codec)
    plan = plan_chunks(index, parsed, config)
    _assert_canonical_plan(index, plan, codec, config)
    end = index.offsets[plan.chunks[0].end]
    assert end > first  # Heading/signature plus actual body, not whitespace alone.
    assert b"meaningful" in source[:end]
    assert source[:end].rstrip() not in {header.rstrip(), (header + spaces).rstrip()}


def test_mandatory_source_event_boundary_may_not_be_projected_across_a_codec_token():
    class EventMergeCodec(BoundaryMergeCodec):
        fingerprint = "mandatory-event-merge"
        merged = b"ab"

    index = TokenIndex.encode(b"ab", EventMergeCodec())
    with pytest.raises(ValueError, match="frame events before tokenization"):
        plan_chunks(
            index, ParseResult(boundaries=[Boundary(1, "chat", mandatory=True)]), ChunkingConfig()
        )
