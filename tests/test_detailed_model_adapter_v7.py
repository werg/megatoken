"""Native serving contract tests for selected-detailed/1."""

from __future__ import annotations

import copy
import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from megatoken.controls import build_request_control
from megatoken.data.detailed_context_v7 import (
    detailed_execution_metadata,
    detailed_selection_instruction,
)
from megatoken.data.detailed_harness_v7 import adapt_detailed_harness_v7
from megatoken.data.knowledge import episode_metadata
from megatoken.data.semantic_contract_v7 import validate_envelope_record
from megatoken.detailed_model_adapter_v7 import DetailedModelAdapterV7, DetailedServingError
from megatoken.generation.codec import ByteCodec
from megatoken.loading import build_model
from megatoken.model import ModelConfig
from megatoken.model.multitrack import ROLE_IDS, MultiTrackGenerator
from megatoken.multitrack_runtime import MultiTrackRuntime

ROOT = Path(__file__).resolve().parents[1]


def _exports():
    path = ROOT / "tests" / "fixtures" / "semantic-v7-detailed-interoperability.json.gz"
    with gzip.open(path, "rt", encoding="utf-8") as source:
        return json.load(source)


class _FakeDetailedRuntime:
    """Only tests framing, prefix validation and restore; never predicts targets."""

    def __init__(self, pieces=()):
        self.generator = SimpleNamespace(
            generator=SimpleNamespace(config=SimpleNamespace(max_capacity=128))
        )
        self.pieces = list(pieces)
        self.sessions = []
        self.calls = []

    def open(self, **kwargs):
        assert kwargs["metadata"] == detailed_execution_metadata()
        assert kwargs["memory_writes"] is False
        session = SimpleNamespace(context_archive=SimpleNamespace(events={}), visible_events=set())
        self.sessions.append(session)
        return session

    def observe_text(self, session, track, payload, *, role, **kwargs):
        identity = f"event{len(session.context_archive.events)}"
        session.context_archive.events[identity] = payload
        session.visible_events.add(identity)
        return 1

    def context_status(self, session, *, track):
        return {"visible_events": tuple(session.visible_events)}

    def generate(self, session, *, tracks, specs, controls, effort, continuation_mode):
        assert continuation_mode == "encoded_text"
        assert len(tracks) == len(specs) == len(controls) == 1
        assert specs[0].text_kind == "utf8"
        assert controls[0] in {
            build_request_control(instruction=detailed_selection_instruction()),
            build_request_control(instruction=detailed_selection_instruction(continuation=True)),
        }
        self.calls.append(controls[0])
        payload, terminal = self.pieces.pop(0)
        identity = f"event{len(session.context_archive.events)}"
        session.context_archive.events[identity] = payload
        session.visible_events.add(identity)
        return SimpleNamespace(
            tape=SimpleNamespace(
                valid=(True,),
                modalities=("text",),
                tracks=tracks,
                event_ids=(identity,),
                payloads=(payload,),
                terminal=(terminal,),
            )
        )


def _target(row):
    return {"sequence": row["sequence"], "track": row["track"], "invocationId": row["invocationId"]}


def test_detailed_adapter_rejects_mutated_context_and_uses_native_continuation_controls():
    export = _exports()[0]["export"]
    timeline = export["public"]["timeline"]
    target_index = next(index for index, row in enumerate(timeline) if row["kind"] == "semantic")
    prior = timeline[:target_index]
    target = timeline[target_index]
    context = export["contexts"][0]
    session_id = timeline[0]["session"]
    calls = []

    def validate_observation(history, row):
        assert row == timeline[len(history)]
        assert validate_envelope_record(row)

    def expected_state_provider(history, next_target):
        assert history == prior
        assert next_target == _target(target)
        return context["prefix"]["selectedState"]

    runtime = _FakeDetailedRuntime([(b"not a detailed ", False), (b"semantic record", True)])
    adapter = DetailedModelAdapterV7(runtime)
    adapter.open(session_id)
    for row in prior:
        adapter.observe(session_id, row, validate_observation=validate_observation)

    forged = copy.deepcopy(context)
    forged["prefix"]["selectedState"]["stack"] = ["forged"]
    from megatoken.data.algorithm_v7 import stable_algorithm_hash

    forged["prefixDigest"] = stable_algorithm_hash(forged["prefix"])
    with pytest.raises(DetailedServingError, match="current prefix|continuation"):
        adapter.select_record(
            session_id,
            forged,
            _target(target),
            expected_state_provider=expected_state_provider,
            commit_record=lambda line, record: calls.append((line, record)),
        )
    assert runtime.calls == []

    with pytest.raises(DetailedServingError, match="invalid v7 record"):
        adapter.select_record(
            session_id,
            context,
            _target(target),
            expected_state_provider=expected_state_provider,
            commit_record=lambda line, record: calls.append((line, record)),
        )
    assert runtime.calls == [
        build_request_control(instruction=detailed_selection_instruction()),
        build_request_control(instruction=detailed_selection_instruction(continuation=True)),
    ]
    assert not calls
    with pytest.raises(DetailedServingError, match="poisoned"):
        adapter.select_record(
            session_id,
            context,
            _target(target),
            expected_state_provider=expected_state_provider,
            commit_record=lambda line, record: calls.append((line, record)),
        )
    assert len(runtime.calls) == 2


def test_detailed_restore_replays_public_rows_and_contexts_without_dispatch():
    export = _exports()[0]["export"]
    timeline = export["public"]["timeline"]
    semantic_rows = [row for row in timeline if row["kind"] == "semantic"]
    target = semantic_rows[1]
    prior = timeline[: timeline.index(target)]
    session_id = timeline[0]["session"]
    contexts = export["contexts"][:1]
    states = {
        row["sequence"]: context["prefix"]["selectedState"]
        for row, context in zip(semantic_rows, export["contexts"], strict=True)
    }

    def expected_state_provider(history, next_target):
        return states[next_target["sequence"]]

    def validate_observation(history, row):
        assert row == timeline[len(history)]

    runtime = _FakeDetailedRuntime()
    adapter = DetailedModelAdapterV7(runtime)
    restored = adapter.restore(
        session_id,
        prior_timeline=prior,
        contexts=contexts,
        target=_target(target),
        expected_state_provider=expected_state_provider,
        validate_observation=validate_observation,
    )
    assert restored["timeline"] == prior
    assert restored["expectedState"] == states[target["sequence"]]
    assert runtime.calls == []
    session = adapter.sessions[session_id]
    assert session.driver.sessions[session_id].latest_context_ids


def test_real_detailed_runtime_prefix_roles_and_controls_match_second_invocation_training_episode():
    class RecordingRuntime(MultiTrackRuntime):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.generated_groups = []
            self.capture = None

        def generate(self, session, **kwargs):
            if self.capture is None:
                self.capture = {
                    "tracks": tuple(kwargs["tracks"]),
                    "specs": tuple(kwargs["specs"]),
                    "controls": tuple(kwargs["controls"]),
                    "event_ids": tuple(session.memory.state.history_event_ids),
                    "visible_ids": tuple(
                        self.context_status(session, track=kwargs["tracks"][0])["visible_events"]
                    ),
                }
            group = super().generate(session, **kwargs)
            self.generated_groups.append(group)
            return group

    case = next(item for item in _exports() if item["case"] == "callback-captured-write")
    export = case["export"]
    timeline = export["public"]["timeline"]
    semantic_rows = [row for row in timeline if row["kind"] == "semantic"]
    second_invocation = next(row for row in timeline if row.get("invocationId") == "event1")
    target = next(
        row for row in semantic_rows if row["invocationId"] == second_invocation["invocationId"]
    )
    prior = timeline[: timeline.index(target)]
    target_context_index = next(
        index for index, row in enumerate(semantic_rows) if row["sequence"] == target["sequence"]
    )
    contexts = export["contexts"]
    state_by_sequence = {
        row["sequence"]: context["prefix"]["selectedState"]
        for row, context in zip(semantic_rows, contexts, strict=True)
    }

    def expected_state_provider(history, next_target):
        return state_by_sequence[next_target["sequence"]]

    def validate_observation(history, row):
        assert row == timeline[len(history)]

    def stop_commit(line, record):
        raise DetailedServingError("stop after observing real model bytes")

    torch.set_num_threads(1)
    torch.manual_seed(61)
    codec = ByteCodec.byte_fallback()
    capacity = 256
    max_history = 1024
    model = build_model(
        ModelConfig.tiny(
            max_capacity=capacity,
            max_history_chunks=max_history,
            context_local_events=max_history,
            context_cross_events=max_history,
            vocab_size=codec.vocab_size,
        ),
        codec,
    ).eval()
    episode, _ = adapt_detailed_harness_v7(
        export,
        codec=codec,
        max_positions=capacity,
        max_history=max_history,
        allow_local=True,
    )
    runtime = RecordingRuntime(
        MultiTrackGenerator.of(model),
        context_archive_events=4096,
    )
    session_id = timeline[0]["session"]
    adapter = DetailedModelAdapterV7(runtime, max_tokens=8, max_record_bytes=8)
    restored = adapter.restore(
        session_id,
        prior_timeline=prior,
        contexts=contexts[:target_context_index],
        target={
            "sequence": target["sequence"],
            "track": target["track"],
            "invocationId": target["invocationId"],
        },
        expected_state_provider=expected_state_provider,
        validate_observation=validate_observation,
    )
    assert restored["expectedState"] == contexts[target_context_index]["prefix"]["selectedState"]
    with pytest.raises(DetailedServingError):
        adapter.select_record(
            session_id,
            contexts[target_context_index],
            {
                "sequence": target["sequence"],
                "track": target["track"],
                "invocationId": target["invocationId"],
            },
            expected_state_provider=expected_state_provider,
            commit_record=stop_commit,
        )

    assert runtime.generated_groups
    assert episode_metadata(episode) == detailed_execution_metadata()
    runtime_session = adapter.sessions[session_id].runtime_state.runtime_session
    metadata_ids = [
        event_id
        for event_id in runtime.capture["event_ids"]
        if event_id not in runtime_session.context_archive.events
    ]
    assert metadata_ids and metadata_ids[0] == "__metadata__"
    visible = [
        event_id
        for event_id in runtime.capture["visible_ids"]
        if event_id in runtime_session.context_archive.events
    ]
    role_names = {value: key for key, value in ROLE_IDS.items()}
    actual = [
        (
            runtime_session.context_archive.events[event_id][0].track,
            role_names[runtime_session.context_archive.events[event_id][0].role_id],
            runtime_session.context_archive.events[event_id][0].payload,
        )
        for event_id in visible
    ]
    target_record_id = f"semantic:r{target['sequence']}"
    target_index = next(
        index
        for index, record in enumerate(episode.records)
        if record.record_id == target_record_id or record.logical_id == target_record_id
    )
    expected = [
        (episode.meridians[index].events[0].track_id, record.role, record.payload)
        for index, record in enumerate(episode.records[:target_index])
    ]
    assert actual == expected
    assert runtime.capture["tracks"] == ("assistant",)
    assert runtime.capture["specs"][0].text_kind == "utf8"
    assert runtime.capture["controls"] == (
        build_request_control(instruction=detailed_selection_instruction()),
    )
    assert adapter.sessions[session_id].timeline == prior
    assert target_context_index == 5
