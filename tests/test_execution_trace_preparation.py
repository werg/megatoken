"""Native trace admission verifies execution, media readiness and asset bytes."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from native_support import group_tape

from megatoken.data.dedup import DedupConfig
from megatoken.data.execution_traces import prepare_traces
from megatoken.data.media_assets import MEDIA_RECORD_TYPE, MediaRecord, TensorAssetStore
from megatoken.data.schema import (
    Episode,
    Lineage,
    LogicalAction,
    MediaSpan,
    Meridian,
    PhysicalRecord,
    Supervision,
    TrackEvent,
    TrackSpec,
    canonical_json,
    digest_bytes,
)
from megatoken.data.sources import TermsReview
from megatoken.data.storage import PreparedDataset
from megatoken.generation.codec import ByteCodec
from megatoken.model.multitrack import GroupActionTape
from megatoken.teacher import PINNED_TEACHER

TERMS = TermsReview(
    approved=True,
    license="CC-BY-4.0",
    license_url="https://example.org/license",
    reviewed_by="reviewer",
    reviewed_at="2026-10-23",
    redistribute="permissive",
    redistribute_basis="The local fixture is explicitly licensed for redistribution.",
)


def fixture(tmp_path, *, stream="duplex_execution", verified=True, with_tape=True):
    codec = ByteCodec.byte_fallback()
    source_root = tmp_path / "source-assets"
    source_assets = TensorAssetStore(source_root)
    source_family = hashlib.sha256(b"reviewed source recording").hexdigest()
    features = source_assets.put(
        np.arange(1024, dtype=np.float32).reshape(2, 512),
        modality="audio",
        kind="features",
        extractor=PINNED_TEACHER,
        source_family=source_family,
    )
    embedding = source_assets.put(
        np.zeros(768, dtype=np.float32),
        modality="audio",
        kind="embedding",
        extractor=PINNED_TEACHER,
        source_family=source_family,
    )
    media = MediaRecord(
        "audio",
        features,
        embedding=embedding,
        prompt="",
    )
    audio_bytes = media.to_bytes()
    audio_records = (
        PhysicalRecord("audio-a", "user", audio_bytes, media_type=MEDIA_RECORD_TYPE),
        PhysicalRecord("audio-b", "user", audio_bytes, media_type=MEDIA_RECORD_TYPE),
    )
    action_bytes = canonical_json({"name": "write", "arguments": {"path": "result.txt"}}).encode()
    action_id, execution_id = "action-1", "execution-1"
    action_record = PhysicalRecord(
        "action-record",
        "assistant",
        action_bytes,
        turn_index=1,
        policy_mask=True,
        provenance="teacher",
        emitted_ids=tuple(codec.encode(action_bytes)),
        codec_fingerprint=codec.fingerprint,
        action_id=action_id,
    )
    result_bytes = b"wrote result.txt"
    result_record = PhysicalRecord(
        "result-record",
        "tool",
        result_bytes,
        turn_index=2,
        provenance="executed",
        result_of=action_id,
        execution_id=execution_id,
    )
    action = LogicalAction(
        action_id,
        "call-1",
        "write",
        canonical_json({"path": "result.txt"}),
        (action_record.record_id,),
        digest_bytes(action_bytes),
        execution_id=execution_id,
    )
    tapes = []
    if with_tape:
        tape = group_tape(
            codec,
            action_bytes,
            emitted_ids=tuple(codec.encode(action_bytes)),
            track="agent",
            role=2,
        )
        tape = replace(tape, meridian_id=1, event_ids=("selected-action",))
        tapes.append(
            {
                "record_cursor": 2,
                "record_ids": [action_record.record_id],
                "tape": json.loads(tape.to_json()),
            }
        )
    annotations = {"native_action_tapes": tapes} if tapes else {}
    episode = Episode(
        "trace-1",
        (*audio_records, action_record, result_record),
        Lineage(
            "reviewed-source",
            "recording-v2",
            "source-family-1",
            frozen_split="train",
        ),
        stream=stream,
        actions=(action,),
        tracks=(
            TrackSpec("speaker-a", "user", "audio"),
            TrackSpec("speaker-b", "user", "audio"),
            TrackSpec("agent", "assistant"),
        ),
        meridians=(
            Meridian(
                0,
                (
                    TrackEvent(
                        "audio-a",
                        "speaker-a",
                        "audio",
                        0,
                        "user",
                        media_span=MediaSpan("recording-clock", 0, 500, 0, 8000, 16000),
                        speaker="speaker-a",
                        direction="incoming",
                        available_at_ns=500_000_000,
                        source_revision="recording-v2",
                        asset_sha256=features.sha256,
                    ),
                    TrackEvent(
                        "audio-b",
                        "speaker-b",
                        "audio",
                        0,
                        "user",
                        media_span=MediaSpan("recording-clock", 250, 750, 4000, 12000, 16000),
                        speaker="speaker-b",
                        direction="incoming",
                        available_at_ns=750_000_000,
                        source_revision="recording-v2",
                        asset_sha256=features.sha256,
                    ),
                ),
                selected_at_ns=1_000_000_000,
            ),
            Meridian(1, (TrackEvent("action-record", "agent", "text", 1, "assistant"),)),
            Meridian(2, (TrackEvent("result-record", "agent", "text", 2, "tool"),)),
        ),
        supervision=Supervision(
            outcome="success" if verified else "failure",
            verification="local_assertions" if verified else "none",
            annotations_json=canonical_json(annotations),
        ),
    )
    return episode, codec, source_root


def write_episode(path: Path, episode):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(episode.to_dict(), sort_keys=True) + "\n")


def prepare(tmp_path, episode, codec, source_root, *, terms=TERMS):
    input_path = tmp_path / "traces.jsonl"
    write_episode(input_path, episode)
    return prepare_traces(
        input_path,
        source_assets=source_root,
        output=tmp_path / "prepared",
        source_terms=terms,
        codec=codec,
        source_id="reviewed-source",
        dedup=DedupConfig(train_fraction=1, validation_fraction=0),
    )


def test_verified_full_duplex_trace_copies_assets_and_uses_regular_export(tmp_path):
    episode, codec, source_root = fixture(tmp_path, stream="executed_trajectories")
    manifest = prepare(tmp_path, episode, codec, source_root)
    assert manifest["counts"]["split:train|stream:duplex_execution"] == 1
    assert manifest["sources"][0]["terms"]["approved"]
    receipt = json.loads((tmp_path / "prepared" / "execution-traces-report.json").read_text())
    assert receipt["source_families"] == ["source-family-1"]

    with PreparedDataset(tmp_path / "prepared", split="train", stream="duplex_execution") as data:
        stored = data[0]
        assert stored.stream == "duplex_execution"
        media = MediaRecord.from_bytes(stored.records[0].payload)
        assert (
            media.source.source_family == hashlib.sha256(b"reviewed source recording").hexdigest()
        )
        values = data.assets.load(media.source)
        assert values.shape == (2, 512)
        assert np.allclose(values, np.arange(1024, dtype=np.float32).reshape(2, 512))
        assert (tmp_path / "prepared" / media.source.relative_path).is_file()


def test_declared_duplex_stream_without_overlapping_speaker_evidence_is_rejected(tmp_path):
    episode, codec, source_root = fixture(tmp_path)
    broken = replace(
        episode,
        meridians=(
            replace(
                episode.meridians[0],
                events=(
                    replace(episode.meridians[0].events[0], speaker=""),
                    episode.meridians[0].events[1],
                ),
            ),
            *episode.meridians[1:],
        ),
    )
    with pytest.raises(ValueError, match="speaker, exact source interval"):
        prepare(tmp_path, broken, codec, source_root)


def test_unverified_episode_cannot_be_remapped_and_unapproved_terms_are_not_inferred(tmp_path):
    episode, codec, source_root = fixture(tmp_path, stream="timed_conversation", verified=False)
    with pytest.raises(ValueError, match="without verified success"):
        prepare(tmp_path, episode, codec, source_root)

    episode, codec, source_root = fixture(tmp_path / "second")
    unapproved = replace(TERMS, approved=False)
    with pytest.raises(ValueError, match="unapproved"):
        prepare(tmp_path / "second", episode, codec, source_root, terms=unapproved)


def test_native_action_tapes_must_match_actual_records_and_codec(tmp_path):
    episode, codec, source_root = fixture(tmp_path)
    tape = GroupActionTape.from_json(
        json.dumps(
            json.loads(episode.supervision.annotations_json)["native_action_tapes"][0]["tape"]
        )
    )
    changed = b"changed"
    bad_tape = replace(
        tape,
        payloads=(changed,),
        emitted_ids=(tuple(codec.encode(changed)),),
    )
    annotations = json.loads(episode.supervision.annotations_json)
    annotations["native_action_tapes"][0]["tape"] = json.loads(bad_tape.to_json())
    broken = replace(
        episode,
        supervision=replace(
            episode.supervision,
            annotations_json=canonical_json(annotations),
        ),
    )
    with pytest.raises(ValueError, match="native text action differs"):
        prepare(tmp_path, broken, codec, source_root)
