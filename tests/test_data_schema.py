import dataclasses
import json

import pytest

from megatoken.data import (
    AdapterContext,
    ByteSpan,
    Episode,
    Lineage,
    PhysicalRecord,
    SchemaError,
    WindowConfig,
    adapt_row,
    encode_record,
    iter_windows,
    make_demo_episodes,
    make_records,
)
from megatoken.data.masking import apply_tool_masks, tool_free_spans
from megatoken.generation.codec import ByteCodec


def test_binary_and_utf8_record_roundtrip_and_exact_spans():
    payload = ("é A🙂\n".encode() + b"\x00\xff\xfe") * 4
    records = make_records(payload, logical_id="a", role="document", policy_mask=True, max_bytes=9)
    assert b"".join(record.payload for record in records) == payload
    assert all(payload[r.source_span.start : r.source_span.end] == r.payload for r in records)
    episode = Episode("e", records, Lineage("local", "sha256:a", "family:a"))
    assert Episode.from_dict(episode.to_dict()) == episode
    with pytest.raises(dataclasses.FrozenInstanceError):
        records[0].payload = b"changed"


class AmbiguousCodec:
    fingerprint = "ambiguous-test-v1"

    def encode(self, payload):
        return [256] if payload == b"ab" else list(payload)

    def decode(self, ids):
        return b"".join(b"ab" if token == 256 else bytes([token]) for token in ids)


def test_emitted_ids_are_not_retokenized_and_fixed_fields_are_masked():
    codec = AmbiguousCodec()
    record = PhysicalRecord(
        "r",
        "assistant",
        b"ab",
        policy_mask=True,
        emitted_ids=(97, 98),
        emitted_free_mask=(True, False),
        codec_fingerprint=codec.fingerprint,
    )
    encoded = encode_record(record, codec)
    assert encoded.token_ids == (97, 98)
    assert encoded.free_mask == (True, False)
    assert encoded.transcript_tokens == encoded.tokens == 1
    assert encoded.positions == 2
    # A merged token crossing from free to deterministic bytes cannot be scored.
    other = PhysicalRecord("r2", "assistant", b"ab", policy_mask=True, free_spans=(ByteSpan(0, 1),))
    assert encode_record(other, codec).free_mask == (False,)
    with pytest.raises(ValueError, match="different codec"):
        encode_record(record, ByteCodec.byte_fallback())


def test_reference_transcript_meters_are_separate_from_native_policy_positions():
    codec = AmbiguousCodec()
    record = PhysicalRecord(
        "action",
        "assistant",
        b"ab",
        turn_index=1,
        policy_mask=True,
        emitted_ids=(97, 98),
        codec_fingerprint=codec.fingerprint,
    )
    episode = Episode(
        "episode",
        (PhysicalRecord("context", "user", b"ab"), record),
        Lineage("fixture", "v1", "family"),
    )
    windows = list(iter_windows(episode, codec, WindowConfig(max_window_tokens=1)))
    assert windows[-1].D == 1
    assert windows[-1].I == 2
    assert windows[-1].P == 2


def test_codec_chunk_limit_and_meter_do_not_change_logical_source_ids():
    codec = ByteCodec.byte_fallback()
    payload = b"one two three four five six\n" * 7
    for limit in (4, 11, 32):
        records = make_records(
            payload,
            logical_id=f"limit-{limit}",
            role="document",
            policy_mask=True,
            codec=codec,
            max_tokens=limit,
            max_bytes=1000,
        )
        assert b"".join(record.payload for record in records) == payload
        assert sum(encode_record(record, codec).tokens for record in records) == len(
            codec.encode(payload)
        )
        assert max(encode_record(record, codec).tokens for record in records) <= limit
        assert all(
            record.emitted_ids is None and record.cached_ids is not None for record in records
        )


def test_public_tool_masks_preserve_free_json_values_across_physical_parts():
    codec = ByteCodec.byte_fallback()
    episode = next(make_demo_episodes(1, max_record_bytes=8))
    masked = apply_tool_masks(episode)
    action_records = [record for record in masked.records if record.action_id]
    assert len(action_records) > 1
    assert sum(encode_record(record, codec).P for record in action_records) == len(
        episode.actions[0].arguments_json.encode()
    )
    assert episode.content_sha256 == masked.content_sha256
    assert b"".join(record.payload for record in episode.records) == b"".join(
        record.payload for record in masked.records
    )
    tools = '[{"name":"finish","parameters":{"type":"object","properties":{},"additionalProperties":false}}]'
    assert tool_free_spans(b'{"arguments":{},"name":"finish"}', tools) == ()
    assert tool_free_spans(b'{"arguments":{},"name":"finish"}', "[]") is None


def test_masks_follow_original_message_turns_and_hide_outcomes():
    row = {
        "messages": [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "not trained"},
            {"role": "user", "content": "next"},
            {"role": "assistant", "content": "trained"},
        ],
        "metadata": {
            "train_turns_mask": [False, False, False, True],
            "gold_outcome": "SECRET_GOLD",
        },
    }
    context = AdapterContext("chat-v4", "abc", "local", "chat", max_record_bytes=3)
    episode = adapt_row(row, "nemotron_chat_v4", context)
    assert {record.turn_index for record in episode.records if record.policy_mask} == {3}
    assert b"SECRET_GOLD" not in b"".join(record.payload for record in episode.model_records())
    assert "SECRET_GOLD" in episode.supervision.annotations_json
    row["metadata"]["train_turns_mask"] = [True]
    with pytest.raises(SchemaError, match="one boolean"):
        adapt_row(row, "nemotron_chat_v4", context)
    with pytest.raises(ValueError, match="Only assistant"):
        PhysicalRecord("x", "tool", b"observation", policy_mask=True)


def test_wrong_reordered_and_repeated_action_results_are_rejected():
    episode = next(make_demo_episodes(max_record_bytes=1000))
    records = list(episode.records)
    action_position = next(index for index, record in enumerate(records) if record.action_id)
    result_position = next(index for index, record in enumerate(records) if record.result_of)
    records[action_position], records[result_position] = (
        records[result_position],
        records[action_position],
    )
    with pytest.raises(ValueError):
        dataclasses.replace(episode, records=tuple(records))
    records = list(episode.records)
    records[result_position] = dataclasses.replace(
        records[result_position], execution_id="different-execution"
    )
    with pytest.raises(ValueError, match="execution identity"):
        dataclasses.replace(episode, records=tuple(records))
    duplicate = dataclasses.replace(
        episode.records[result_position],
        record_id="duplicate-result",
        logical_id="duplicate-result",
        turn_index=9,
    )
    with pytest.raises(ValueError, match="execute exactly once"):
        dataclasses.replace(episode, records=episode.records + (duplicate,))


def test_failed_trace_imitates_only_explicit_recovery_turns():
    row = {
        "repo": "acme/repo",
        "instance_id": "acme__repo-2",
        "resolved": 0,
        "messages": [
            {"role": "user", "content": "fix"},
            {"role": "assistant", "content": "bad branch"},
            {"role": "user", "content": "try a recovery"},
            {"role": "assistant", "content": "good repair"},
        ],
        "recovery_turns": [3],
    }
    episode = adapt_row(
        row, "open_swe", AdapterContext("openswe", "rev", "file", "executed_trajectories")
    )
    assert [record.turn_index for record in episode.records if record.policy_mask] == [3]
    assert episode.supervision.outcome == "failure"


def test_orchard_json_metadata_and_complete_tool_boundaries():
    row = {
        "metadata": json.dumps(
            {
                "instance_id": "acme__repo-1",
                "repo": "acme/repo",
                "verify_status": "resolved",
                "gold": "private",
            }
        ),
        "tools": json.dumps([{"type": "function", "function": {"name": "read"}}]),
        "messages": [
            {"role": "user", "content": "read"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call1", "function": {"name": "read", "arguments": '{"path":"a.py"}'}}
                ],
            },
            {"role": "tool", "tool_call_id": "call1", "content": "file bytes"},
            {"role": "assistant", "content": "done"},
        ],
    }
    episode = adapt_row(
        row,
        "orchard_swe",
        AdapterContext("orchard", "rev", "hf", "verified_trajectories", max_record_bytes=8),
    )
    assert episode.supervision.verified_success
    assert len(episode.actions[0].record_ids) > 1
    assert not any(record.policy_mask for record in episode.records if record.role == "tool")
    assert episode.lineage.repository == "acme/repo"
    assert b"private" not in b"".join(record.payload for record in episode.records)


def test_windows_count_reconstruction_separately_and_preserve_dependency_credit():
    episode = next(make_demo_episodes(max_record_bytes=20))
    codec = ByteCodec.byte_fallback()
    windows = list(
        iter_windows(episode, codec, WindowConfig(max_window_tokens=30, max_decisions=1))
    )
    assert tuple(
        record.record.record_id for window in windows for record in window.records
    ) == tuple(record.record_id for record in episode.records)
    assert sum(window.D for window in windows) == sum(
        len(record.payload) for record in episode.records
    )
    assert sum(window.I for window in windows) > sum(window.D for window in windows)
    for window in windows:
        actions = [entry.record for entry in window.records if entry.record.action_id]
        if actions:
            assert actions[0].part_index == 0
            assert actions[-1].part_index + 1 == actions[-1].part_count
        assert tuple(entry.record for entry in window.prefix) == episode.records[: window.start]
    merged = list(
        iter_windows(
            episode,
            codec,
            WindowConfig(30, 1, respect_dependencies=True, max_dependency_tokens=1000),
        )
    )
    assert any(window.writer_reader_pairs for window in merged)


def test_repository_patch_is_target_and_future_controls_are_forbidden():
    row = {
        "repo": "a/b",
        "instance_id": "a/b-1",
        "issue": "repair",
        "before": "old code",
        "patch": "SECRET_PATCH",
    }
    episode = adapt_row(
        row, "repository", AdapterContext("repo", "rev", "file", "issue_pr_sequences")
    )
    assert b"SECRET_PATCH" not in b"".join(
        record.payload for record in episode.records if not record.policy_mask
    )
    with pytest.raises(ValueError, match="causal origin"):
        dataclasses.replace(
            episode, control_reference=b"SECRET_PATCH", reference_origin="future_target"
        )
