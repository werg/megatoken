"""Behavior coverage counts verified evidence and unique source families."""

import json
import sqlite3
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest
from native_support import group_tape

from megatoken.data.coverage import Coverage, behaviors, require_coverage
from megatoken.data.packing import TrainingWindow
from megatoken.data.schema import (
    Episode,
    Lineage,
    LogicalAction,
    Meridian,
    PhysicalRecord,
    Supervision,
    TrackEvent,
    TrackSpec,
    canonical_json,
    digest_bytes,
    encode_record,
)
from megatoken.generation.codec import ByteCodec
from megatoken.training.outcome import OutcomeCredit


def parallel_episode(*, family="family", intervals=None, verified=True):
    records = []
    actions = []
    events = []
    for index, execution_id in enumerate(("exec-a", "exec-b")):
        action_id = f"action-{index}"
        record_id = f"record-{index}"
        payload = canonical_json({"name": "read", "arguments": {"path": str(index)}}).encode()
        records.append(
            PhysicalRecord(
                record_id,
                "assistant",
                payload,
                action_id=action_id,
                execution_id=execution_id,
                provenance="executed",
            )
        )
        actions.append(
            LogicalAction(
                action_id,
                action_id,
                "read",
                canonical_json({"path": str(index)}),
                (record_id,),
                digest_bytes(payload),
                execution_id=execution_id,
            )
        )
        events.append(TrackEvent(record_id, f"track-{index}", "text", 0, "assistant"))
    annotations = {} if intervals is None else {"execution_intervals": intervals}
    episode = Episode(
        "parallel",
        tuple(records),
        Lineage("fixture", "v1", family),
        actions=tuple(actions),
        tracks=(TrackSpec("track-0", "assistant"), TrackSpec("track-1", "assistant")),
        meridians=(Meridian(0, tuple(events)),),
        supervision=Supervision(
            outcome="success" if verified else "unknown",
            verification="local_assertions" if verified else "",
            annotations_json=canonical_json(annotations),
        ),
    )
    return episode


def part(report):
    return SimpleNamespace(manifest={"behavior_coverage": report})


def corpus(*reports):
    return SimpleNamespace(parts=[part(report) for report in reports])


def test_co_selected_calls_are_distinct_from_measured_execution_overlap():
    episode = parallel_episode()
    found = behaviors(episode)
    assert "executed_parallel_calls" in found
    assert "executed_overlapping_calls" not in found

    overlap = [
        {
            "action_id": "action-0",
            "execution_id": "exec-a",
            "started_at_ns": 10,
            "completed_at_ns": 30,
        },
        {
            "action_id": "action-1",
            "execution_id": "exec-b",
            "started_at_ns": 20,
            "completed_at_ns": 40,
        },
    ]
    assert "executed_overlapping_calls" in behaviors(parallel_episode(intervals=overlap))
    touching = [
        {
            "action_id": "action-0",
            "execution_id": "exec-a",
            "started_at_ns": 10,
            "completed_at_ns": 20,
        },
        {
            "action_id": "action-1",
            "execution_id": "exec-b",
            "started_at_ns": 20,
            "completed_at_ns": 40,
        },
    ]
    assert "executed_overlapping_calls" not in behaviors(parallel_episode(intervals=touching))
    assert "executed_parallel_calls" not in behaviors(parallel_episode(verified=False))


def test_family_ids_deduplicate_across_roots_and_split_filter_is_exact():
    first = Coverage(sqlite3.connect(":memory:"))
    first.add(parallel_episode(family="same"), "train", "calls")
    first_report = first.report()
    duplicate = Coverage(sqlite3.connect(":memory:"))
    duplicate.add(parallel_episode(family="same"), "train", "calls")
    duplicate_report = duplicate.report()
    validation = Coverage(sqlite3.connect(":memory:"))
    validation.add(parallel_episode(family="heldout"), "validation", "calls")
    validation_report = validation.report()

    assert (
        require_coverage(
            corpus(first_report, duplicate_report), {"executed_parallel_calls": 1}, {"calls"}
        )["executed_parallel_calls"]
        == 1
    )
    with pytest.raises(ValueError, match="Required validation behaviors"):
        require_coverage(
            corpus(first_report, validation_report),
            {"executed_parallel_calls": 2},
            {"calls"},
            split="validation",
        )
    assert (
        require_coverage(
            corpus(first_report, validation_report),
            {"executed_parallel_calls": 1},
            {"calls"},
            split="validation",
        )["executed_parallel_calls"]
        == 1
    )


def test_legacy_family_counts_cannot_be_summed_across_roots():
    legacy = {
        "version": 1,
        "original_families": {"train|calls|executed_parallel_calls": 1},
    }
    with pytest.raises(ValueError, match="Malformed original family identities"):
        require_coverage(
            corpus(legacy, legacy),
            {"executed_parallel_calls": 1},
            {"calls"},
        )


def test_verified_outcome_only_group_advances_q_without_lexical_targets():
    original = parallel_episode()
    records = tuple(replace(record, policy_mask=False) for record in original.records)
    codec = ByteCodec.byte_fallback()
    tape = group_tape(codec, b"", valid=False, emitted_ids=(), track="track-0")
    supervision = replace(
        original.supervision,
        annotations_json=canonical_json(
            {
                "native_action_tapes": [
                    {
                        "record_cursor": 0,
                        "record_ids": [None],
                        "outcome_credit": asdict(OutcomeCredit(1, 0, 1, "local_assertions")),
                        "tape": json.loads(tape.to_json()),
                    }
                ]
            }
        ),
    )
    episode = replace(original, records=records, supervision=supervision)
    window = TrainingWindow(
        episode,
        (),
        tuple(encode_record(record, codec) for record in records),
        0,
        len(records),
        0,
    )
    assert window.P == 0
    assert window.Q == 1
