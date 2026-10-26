"""Causal ready-event runtime and JSONL mixed-media transport."""

import base64
import hashlib
import io
import json
import time
from types import SimpleNamespace

import pytest
import torch

from megatoken.generation.codec import ByteCodec
from megatoken.generation.policy import FacetedGenerator
from megatoken.generation.structure import ActionSpec
from megatoken.model import FacetedCore, ModelConfig
from megatoken.model.media import MediaOutputSpec
from megatoken.model.multitrack import MultiTrackGenerator, ObservedEvent
from megatoken.multitrack_runtime import ConcurrentToolRunner, MultiTrackRuntime
from megatoken.multitrack_service import MultiTrackService, serve_multitrack_jsonl
from megatoken.output_store import FileOutputStore
from megatoken.teacher import PINNED_TEACHER
from tests.media_support import features, install_fake_realizers

# Opening commits the metadata prefix, then the first lookup's query and knowledge events.
OPENED = 3
# A finished user message or assistant turn, or an arriving tool result, makes a lookup due: its query
# and knowledge events.
LOOKUP = 2


def _runtime():
    torch.manual_seed(27)
    model = FacetedGenerator(FacetedCore(ModelConfig.tiny()), ByteCodec.byte_fallback())
    install_fake_realizers(model)
    return model, MultiTrackRuntime(MultiTrackGenerator(model))


def test_speculative_groups_do_not_mutate_canonical_history():
    _, runtime = _runtime()
    session = runtime.open()
    runtime.observe(session, (ObservedEvent("user", "text", b"Speak briefly"),))
    committed = session.state.history.keys.clone()
    draft = runtime.speculate(
        session,
        tracks=("voice",),
        specs=(MediaOutputSpec("audio", sample_count=128, sample_rate=24000),),
        effort=1,
        generator=torch.Generator().manual_seed(5),
    )
    torch.testing.assert_close(session.state.history.keys, committed)
    runtime.cancel(session)
    with pytest.raises(ValueError, match="stale|no longer"):
        runtime.accept(session, draft)


def test_translation_timing_uses_observed_arrival_and_selection_clocks():
    _, runtime = _runtime()
    session = runtime.open()
    runtime.observe(session, (ObservedEvent("source", "text", b"hello", event_id="source-1"),))
    revision = runtime.start_generation(
        session,
        tracks=("target",),
        event_ids=("target-1",),
        specs=(ActionSpec(max_tokens=4),),
        effort=1,
        generator=torch.Generator().manual_seed(10),
    )
    work = session.rendering[revision]
    work.future.result(timeout=10)
    work = runtime.poll_generation(session, revision)
    payload_extent = len(work.result.tape.payloads[0])
    runtime.delivered(session, revision, ("target-1",), extents={"target-1": payload_extent})
    report = runtime.translation_timing_report(session, revision, ("source-1",), "target-1")
    assert report["clock"] == "monotonic_elapsed_v1"
    assert report["source_to_selection_seconds"] >= 0
    assert report["source_to_delivery_ack_seconds"] >= report["source_to_selection_seconds"]
    assert report["delivery_ack_is_client_ack_not_audio_playback"] is True
    with pytest.raises(ValueError, match="unobserved source"):
        runtime.translation_timing_report(session, session.revision, ("future-source",), "target-1")
    with pytest.raises(ValueError, match="output is absent"):
        runtime.translation_timing_report(session, session.revision, ("source-1",), "other-output")
    runtime.release_generation(session, revision)
    assert revision not in session.selection_at_ns
    assert revision not in session.selection_visible_event_ids
    retained_ids = set(session.state.history_event_ids) | set(session.state.frontier_event_ids)
    assert set(session.event_arrived_at_ns) <= retained_ids


def test_empty_text_delivery_and_retries_do_not_invent_progress_clocks():
    _, runtime = _runtime()
    session = runtime.open()
    event_ids = ("empty-text", "partial-text")
    tape = SimpleNamespace(event_ids=event_ids)
    selection = SimpleNamespace(tape=tape, specs=(ActionSpec(max_tokens=1),) * 2)
    result = SimpleNamespace(tape=SimpleNamespace(payloads=(b"", b"abc"), valid=(True, True)))
    work = SimpleNamespace(
        status="rendered",
        selection=selection,
        result=result,
        delivery_progress={},
        delivery_log=[],
        delivered_events=set(),
    )
    revision = 77
    session.rendering[revision] = work
    session.event_arrived_at_ns["source"] = time.monotonic_ns()
    session.selection_at_ns[revision] = time.monotonic_ns()
    session.selection_visible_event_ids[revision] = frozenset({"source"})

    runtime.delivered(
        session,
        revision,
        event_ids,
        extents={"empty-text": 0, "partial-text": 2},
    )
    assert work.delivery_progress == {"empty-text": 0, "partial-text": 2}
    assert work.delivered_events == {"empty-text"}
    assert [row["event_id"] for row in work.delivery_log] == ["partial-text"]
    first_clock = work.delivery_log[0]["available_at_ns"]

    runtime.delivered(
        session,
        revision,
        event_ids,
        extents={"empty-text": 0, "partial-text": 2},
    )
    assert len(work.delivery_log) == 1
    assert work.delivery_log[0]["available_at_ns"] == first_clock
    timing = runtime.translation_timing_report(session, revision, ("source",), "empty-text")
    assert timing["delivery_acknowledged_at_ns"] is None
    assert timing["delivery_acknowledged_extent"] is None
    assert timing["source_to_delivery_ack_seconds"] is None

    runtime.delivered(session, revision, ("partial-text",), extents={"partial-text": 3})
    assert [row["delivered"] for row in work.delivery_log if row["event_id"] == "partial-text"] == [
        2,
        3,
    ]
    assert work.delivered_events == set(event_ids)
    assert work.status == "delivered"

    for invalid in (-1, 4, True, 1):
        with pytest.raises(ValueError, match="Delivery extent"):
            runtime.delivered(
                session,
                revision,
                ("partial-text",),
                extents={"partial-text": invalid},
            )


def test_output_store_budget_is_reserved_before_committing_media_selection(tmp_path):
    _, runtime = _runtime()
    service = MultiTrackService(runtime, output_store=FileOutputStore(tmp_path, max_bytes=16))
    opened = service.handle({"op": "open"})
    session_id = opened["session"]
    session = service.sessions[session_id]
    before = session.revision
    failed = service.handle(
        {
            "op": "generate",
            "session": session_id,
            "outputs": [{"track": "voice", "modality": "audio", "sample_count": 16}],
            "effort": 1,
        }
    )
    assert not failed["ok"] and "before generation" in failed["error"]
    assert session.revision == before
    service.close()


def test_partial_delivery_is_timed_and_stop_revokes_future_media_bytes(tmp_path):
    _, runtime = _runtime()
    service = MultiTrackService(runtime, output_store=FileOutputStore(tmp_path))
    session_id = service.handle({"op": "open"})["session"]
    service.handle(
        {
            "op": "observe",
            "session": session_id,
            "events": [{"track": "source", "text": "source", "event_id": "source-clock"}],
        }
    )
    started = service.handle(
        {
            "op": "start_generation",
            "session": session_id,
            "effort": 1,
            "outputs": [
                {
                    "track": "voice",
                    "event_id": "voice-clock",
                    "modality": "audio",
                    "sample_count": 16,
                }
            ],
        }
    )
    revision = started["selection_revision"]
    work = service.sessions[session_id].rendering[revision]
    work.future.result(timeout=10)
    rendered = service.handle(
        {"op": "poll_generation", "session": session_id, "selection_revision": revision}
    )
    media_ref = rendered["outputs"][0]["media_ref"]
    service.handle(
        {
            "op": "delivered",
            "session": session_id,
            "selection_revision": revision,
            "event_ids": ["voice-clock"],
            "extents": {"voice-clock": 4},
        }
    )
    stopped = service.handle(
        {"op": "stop_delivery", "session": session_id, "selection_revision": revision}
    )
    assert stopped["status"] == "cancelled"
    report = service.handle(
        {
            "op": "translation_timing",
            "session": session_id,
            "selection_revision": revision,
            "source_event_ids": ["source-clock"],
            "output_event_id": "voice-clock",
        }
    )
    assert report["delivery_acknowledged_extent"] == 4
    assert report["source_to_delivery_ack_seconds"] >= report["source_to_selection_seconds"]
    denied = service.handle(
        {
            "op": "read_media",
            "session": session_id,
            "selection_revision": revision,
            "handle": media_ref["handle"],
            "offset": 4,
            "length": 4,
        }
    )
    assert not denied["ok"]
    assert service.sessions[session_id].rendering[revision].delivery_log
    service.close()


def test_accept_and_fetch_release_residual_media_reservations(tmp_path, monkeypatch):
    _, runtime = _runtime()
    store = FileOutputStore(tmp_path, max_bytes=1 << 20)
    service = MultiTrackService(runtime, output_store=store)
    session_id = service.handle({"op": "open"})["session"]
    draft = service.handle(
        {
            "op": "speculate",
            "session": session_id,
            "effort": 1,
            "outputs": [{"track": "voice", "modality": "audio", "sample_count": 16}],
        }
    )
    assert draft["ok"] and "outputs" not in draft
    owner = f"{session_id}:{service.sessions[session_id].revision + 1}"
    original_put = store.put

    def put_with_duplicate_alternative_reservation(entry_owner, data):
        reference = original_put(entry_owner, data)
        store.reserve(entry_owner, len(data))
        return reference

    store.put = put_with_duplicate_alternative_reservation
    accepted = service.handle({"op": "accept", "session": session_id})
    assert accepted["ok"]
    assert owner not in store._reservations

    # A recovered response can inherit an unused reservation after a prior partial
    # serialization failure; successful replay must release that residual capacity.
    store.reserve(owner, 16)
    import megatoken.multitrack_service as service_module

    original_response = service_module._output_response

    def fail_response(*_args, **_kwargs):
        raise OSError("simulated media serialization interruption")

    monkeypatch.setattr(service_module, "_output_response", fail_response)
    failed_replay = service.handle(
        {
            "op": "fetch_generation",
            "session": session_id,
            "selection_revision": accepted["selection_revision"],
        }
    )
    assert not failed_replay["ok"]
    assert store._reservations[owner] == 16
    monkeypatch.setattr(service_module, "_output_response", original_response)
    replay = service.handle(
        {
            "op": "fetch_generation",
            "session": session_id,
            "selection_revision": accepted["selection_revision"],
        }
    )
    assert replay["ok"]
    assert owner not in store._reservations
    service.close()


def test_arriving_observation_invalidates_pending_generation():
    _, runtime = _runtime()
    session = runtime.open()
    draft = runtime.speculate(
        session, tracks=("answer",), specs=(ActionSpec(max_tokens=8),), effort=1
    )
    runtime.observe(session, (ObservedEvent("user", "text", b"Actually, stop"),))
    with pytest.raises(ValueError, match="stale|no longer"):
        runtime.accept(session, draft)
    assert session.revision == 1


def test_accept_commits_whole_meridian():
    _, runtime = _runtime()
    session = runtime.open()
    # A finished assistant turn, so the lookup it makes due is deterministic.
    draft = runtime.speculate(
        session,
        tracks=("answer",),
        specs=(ActionSpec(max_tokens=8, require_terminal=True),),
        effort=1,
    )
    accepted = runtime.accept(session, draft)
    assert session.revision == 1
    assert session.state.meridian_sizes == (1,) * (OPENED + 1 + LOOKUP)
    assert accepted.tape.tracks == ("answer",)


def test_parallel_tool_result_is_not_committed_automatically():
    _, runtime = _runtime()
    session = runtime.open()
    opened = session.state.history.slots
    pool = ConcurrentToolRunner(2)
    try:
        pool.submit("fetch-1", lambda: b"one")
        pool.submit("fetch-2", lambda: b"two")
        for task in pool.pending.values():
            task.result(timeout=5)
        assert pool.collect_ready() == ({"fetch-1": b"one", "fetch-2": b"two"}, {})
        assert session.state.history.slots == opened
        runtime.observe(
            session,
            (
                ObservedEvent("tool-1", "text", b"one", role_id=3),
                ObservedEvent("tool-2", "text", b"two", role_id=3),
            ),
        )
        assert session.state.meridian_sizes[OPENED:] == (2,) + (1,) * LOOKUP
    finally:
        pool.close()


def test_a_failed_parallel_call_does_not_lose_the_finished_ones():
    def fail():
        raise OSError("connection reset")

    pool = ConcurrentToolRunner(3)
    try:
        pool.submit("good", lambda: b"ok")
        pool.submit("bad", fail)
        pool.submit("wrong-type", lambda: "text")
        for task in pool.pending.values():
            task.exception(timeout=5)
        results, failures = pool.collect_ready()
        assert results == {"good": b"ok"}
        assert isinstance(failures["bad"], OSError) and isinstance(
            failures["wrong-type"], TypeError
        )
        assert not pool.pending  # nothing finished is left behind, and nothing is reported twice
        assert pool.collect_ready() == ({}, {})
    finally:
        pool.close()


def _feature_event(track, tokens=9, seed=0, extractor=PINNED_TEACHER):
    values = features(tokens, seed=seed)[0].numpy().astype("<f4")
    return {
        "track": track,
        "modality": "audio",
        "payload_f32_b64": base64.b64encode(values.tobytes()).decode("ascii"),
        "shape": [1, tokens, values.shape[-1]],
        "extractor": extractor,
    }


def _serve(model, requests):
    results = io.StringIO()
    serve_multitrack_jsonl(
        MultiTrackRuntime(MultiTrackGenerator(model)), requests(results), results
    )
    return [json.loads(line) for line in results.getvalue().splitlines()]


def _line(value):
    return json.dumps(value) + "\n"


def test_jsonl_multitrack_roundtrip():
    model, _ = _runtime()

    def requests(results):
        yield _line({"op": "open", "id": 1})
        sid = json.loads(results.getvalue().splitlines()[-1])["session"]
        yield _line(
            {
                "op": "observe",
                "id": 2,
                "session": sid,
                "events": [
                    {
                        "track": "user",
                        "modality": "text",
                        "payload_b64": "aGk=",
                        "event_id": "source-start",
                    },
                    _feature_event("mic"),
                ],
            }
        )
        yield _line(
            {
                "op": "generate",
                "id": 3,
                "session": sid,
                "effort": 1,
                "seed": 11,
                "outputs": [
                    {
                        "track": "reply",
                        "event_id": "reply-output",
                        "modality": "text",
                        "max_tokens": 8,
                        "terminal": True,
                    },
                    {
                        "track": "voice",
                        "event_id": "voice-output",
                        "modality": "audio",
                        "sample_count": 64,
                    },
                    {
                        "track": "still",
                        "event_id": "still-output",
                        "modality": "image",
                        "height": 16,
                        "width": 16,
                    },
                ],
            }
        )
        generated = json.loads(results.getvalue().splitlines()[-1])
        voice_ref = generated["outputs"][1]["media_ref"]
        yield _line(
            {
                "op": "fetch_generation",
                "session": sid,
                "selection_revision": generated["selection_revision"],
            }
        )
        yield _line(
            {
                "op": "translation_timing",
                "session": sid,
                "selection_revision": generated["selection_revision"],
                "source_event_ids": ["source-start"],
                "output_event_id": "voice-output",
            }
        )
        yield _line(
            {
                "op": "read_media",
                "session": sid,
                "selection_revision": generated["selection_revision"],
                "handle": voice_ref["handle"],
                "offset": 0,
                "length": voice_ref["byte_length"],
            }
        )
        yield _line({"op": "open", "id": 5})
        other = json.loads(results.getvalue().splitlines()[-1])["session"]
        yield _line(
            {
                "op": "read_media",
                "session": other,
                "selection_revision": generated["selection_revision"],
                "handle": voice_ref["handle"],
                "offset": 0,
                "length": 8,
            }
        )
        yield _line(
            {
                "op": "release_generation",
                "session": sid,
                "selection_revision": generated["selection_revision"],
            }
        )
        yield _line(
            {
                "op": "read_media",
                "session": sid,
                "selection_revision": generated["selection_revision"],
                "handle": voice_ref["handle"],
                "offset": 0,
                "length": 8,
            }
        )
        yield _line({"op": "status", "id": 4, "session": sid})

    rows = _serve(model, requests)
    assert len(rows) == 11
    assert all(rows[i]["ok"] for i in (0, 1, 2, 3, 4, 5, 6, 8, 10))
    assert not rows[7]["ok"] and not rows[9]["ok"]
    # The user's finished message makes a lookup due: its query and knowledge events follow it.
    assert rows[1]["next_meridian"] == OPENED + 1 + LOOKUP
    assert rows[1]["events_seen"] == OPENED + 2 + LOOKUP
    generated = rows[2]
    voice, still = generated["outputs"][1:]
    assert voice["shape"] == [64] and still["shape"] == [1, 3, 16, 16]
    assert generated["tape"]["modalities"] == ["text", "audio", "image"]
    # The realized media bytes come back, bound to the tape by SHA256 and realizer identity.
    assert rows[3]["outputs"][1]["media_ref"] == voice["media_ref"]
    timing = rows[4]
    assert timing["clock"] == "monotonic_elapsed_v1"
    assert timing["source_to_selection_seconds"] >= 0
    assert timing["source_to_delivery_ack_seconds"] is None
    received_voice = base64.b64decode(rows[5]["data_b64"])
    for output, tape_payload in zip((voice, still), generated["tape"]["payloads"][1:], strict=True):
        digest = json.loads(output["tape_payload"])
        assert digest["kind"] == "media-sha256-v1"
        assert digest["shape"] == output["shape"]
        assert base64.b64decode(tape_payload) == output["tape_payload"].encode()
        assert output["media_ref"]["sha256"] == digest["sha256"]
    assert len(received_voice) == 4 * int(torch.Size(voice["shape"]).numel())
    assert hashlib.sha256(received_voice).hexdigest() == voice["media_ref"]["sha256"]
    assert [json.loads(o["tape_payload"])["realizer"] for o in (voice, still)] == (
        generated["tape"]["pack_fingerprints"][1:]
    )
    assert "tape_payload" not in generated["outputs"][0]
    assert rows[10]["meridians"] == OPENED + 2 + 2 * LOOKUP
    assert rows[10]["retained_meridians"] == OPENED + 2
    assert rows[10]["events_seen"] == OPENED + 5 + 2 * LOOKUP
    # The tiny model keeps 8 events of history and drops whole oldest meridians to fit.
    assert rows[10]["retained_events"] == 7


def test_jsonl_media_must_be_pinned_features():
    model, _ = _runtime()
    raw_pcm = base64.b64encode(torch.zeros(2400).numpy().astype("<f4").tobytes()).decode()

    def requests(results):
        yield _line({"op": "open", "id": 1})
        sid = json.loads(results.getvalue().splitlines()[-1])["session"]

        def observe(identifier, event):
            return _line({"op": "observe", "id": identifier, "session": sid, "events": [event]})

        yield observe(2, _feature_event("mic", extractor="another-teacher"))
        yield observe(3, _feature_event("mic", extractor=""))
        yield observe(
            4,
            {
                "track": "mic",
                "modality": "audio",
                "payload_f32_b64": raw_pcm,
                "shape": [1, 2400],
                "extractor": PINNED_TEACHER,
            },
        )
        yield _line({"op": "status", "id": 5, "session": sid})

    rows = _serve(model, requests)
    assert [row["ok"] for row in rows] == [True, False, False, False, True]
    assert "different teacher" in rows[1]["error"]
    assert "ordered" in rows[3]["error"]  # raw PCM is not an observation
    assert rows[4]["events_seen"] == OPENED  # nothing was committed


def test_jsonl_speculative_media_is_provisional_until_accepted():
    model, _ = _runtime()

    def requests(results):
        yield _line({"op": "open", "id": 1})
        sid = json.loads(results.getvalue().splitlines()[-1])["session"]
        yield _line(
            {
                "op": "speculate",
                "id": 2,
                "session": sid,
                "effort": 1,
                "outputs": [{"track": "voice", "modality": "audio", "sample_count": 64}],
            }
        )
        yield _line({"op": "status", "id": 3, "session": sid})
        yield _line({"op": "accept", "id": 4, "session": sid})
        yield _line({"op": "status", "id": 5, "session": sid})

    rows = _serve(model, requests)
    assert all(row["ok"] for row in rows)
    assert rows[1]["provisional"] is True
    assert "outputs" not in rows[1]
    assert rows[2]["pending"] is True and rows[2]["retained_events"] == OPENED
    assert rows[3]["outputs"][0]["media_ref"]["version"] == 1
    assert rows[4]["pending"] is False and rows[4]["retained_events"] == OPENED + 1 + LOOKUP


def test_jsonl_a_result_that_is_not_strict_json_fails_only_its_own_request(monkeypatch):
    import dataclasses

    model, _ = _runtime()
    original = MultiTrackGenerator.render_selected

    def nonfinite(self, *args, **kwargs):
        group = original(self, *args, **kwargs)
        return dataclasses.replace(group, log_prob=torch.tensor(float("nan")))

    monkeypatch.setattr(MultiTrackGenerator, "render_selected", nonfinite)

    def requests(results):
        yield _line({"op": "open", "id": 1})
        sid = json.loads(results.getvalue().splitlines()[-1])["session"]
        outputs = [{"track": "reply", "modality": "text", "max_tokens": 4}]
        yield _line({"op": "speculate", "id": 2, "session": sid, "effort": 1, "outputs": outputs})
        yield _line({"op": "status", "id": 3, "session": sid})

    rows = _serve(model, requests)
    assert [row["ok"] for row in rows] == [True, False, True]
    assert "ValueError" in rows[1]["error"] and rows[1]["id"] == 2
