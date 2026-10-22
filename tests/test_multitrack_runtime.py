"""Causal ready-event runtime and JSONL mixed-media transport."""

import base64
import hashlib
import io
import json

import pytest
import torch

from megatoken.generation.codec import ByteCodec
from megatoken.generation.policy import FacetedGenerator
from megatoken.generation.structure import ActionSpec
from megatoken.model import FacetedCore, ModelConfig
from megatoken.model.media import MediaOutputSpec
from megatoken.model.multitrack import MultiTrackGenerator, ObservedEvent
from megatoken.multitrack_runtime import ConcurrentToolRunner, MultiTrackRuntime
from megatoken.multitrack_service import serve_multitrack_jsonl
from megatoken.teacher import PINNED_TEACHER
from tests.media_support import features, install_fake_realizers


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


def test_accept_commits_whole_meridian_and_delivery_cannot_reverse():
    _, runtime = _runtime()
    session = runtime.open()
    draft = runtime.speculate(
        session, tracks=("answer",), specs=(ActionSpec(max_tokens=8),), effort=1
    )
    accepted = runtime.accept(session, draft)
    assert session.revision == 1
    assert session.state.meridian_sizes == (1,)
    assert accepted.tape.tracks == ("answer",)
    runtime.record_delivery(session, "answer:0", 10, complete=False)
    runtime.record_delivery(session, "answer:0", 20, complete=True)
    with pytest.raises(ValueError, match="rolled back|incomplete"):
        runtime.record_delivery(session, "answer:0", 5, complete=False)


def test_parallel_tool_result_is_not_committed_automatically():
    _, runtime = _runtime()
    session = runtime.open()
    pool = ConcurrentToolRunner(2)
    try:
        pool.submit("fetch-1", lambda: b"one")
        pool.submit("fetch-2", lambda: b"two")
        for task in pool.pending.values():
            task.result(timeout=5)
        assert pool.collect_ready() == {"fetch-1": b"one", "fetch-2": b"two"}
        assert session.state.history.slots == 0
        runtime.observe(
            session,
            (
                ObservedEvent("tool-1", "text", b"one", role_id=3),
                ObservedEvent("tool-2", "text", b"two", role_id=3),
            ),
        )
        assert session.state.meridian_sizes == (2,)
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
    class Engine:
        device = torch.device("cpu")
        config = model.config

    engine = Engine()
    engine.model = model
    results = io.StringIO()
    serve_multitrack_jsonl(engine, requests(results), results)
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
                    {"track": "user", "modality": "text", "payload_b64": "aGk="},
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
                    {"track": "reply", "modality": "text", "max_tokens": 8},
                    {"track": "voice", "modality": "audio", "sample_count": 64},
                    {"track": "still", "modality": "image", "height": 16, "width": 16},
                ],
            }
        )
        yield _line({"op": "status", "id": 4, "session": sid})

    rows = _serve(model, requests)
    assert len(rows) == 4 and all(row["ok"] for row in rows)
    assert rows[1]["next_meridian"] == 1
    assert rows[1]["events_seen"] == 2
    generated = rows[2]
    voice, still = generated["outputs"][1:]
    assert voice["shape"] == [64] and still["shape"] == [1, 3, 16, 16]
    assert generated["tape"]["modalities"] == ["text", "audio", "image"]
    # The realized media bytes come back, bound to the tape by SHA256 and realizer identity.
    for output, tape_payload in zip((voice, still), generated["tape"]["payloads"][1:], strict=True):
        media = base64.b64decode(output["payload_f32_b64"])
        assert len(media) == 4 * int(torch.Size(output["shape"]).numel())
        digest = json.loads(output["tape_payload"])
        assert digest["kind"] == "media-sha256-v1"
        assert digest["sha256"] == hashlib.sha256(media).hexdigest()
        assert digest["shape"] == output["shape"]
        assert base64.b64decode(tape_payload) == output["tape_payload"].encode()
    assert [json.loads(o["tape_payload"])["realizer"] for o in (voice, still)] == (
        generated["tape"]["pack_fingerprints"][1:]
    )
    assert "tape_payload" not in generated["outputs"][0]
    assert rows[3]["meridians"] == 2
    assert rows[3]["retained_meridians"] == 2
    assert rows[3]["events_seen"] == 5
    assert rows[3]["retained_events"] == 5


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
    assert rows[4]["events_seen"] == 0  # nothing was committed


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
    assert rows[1]["outputs"][0]["shape"] == [64]
    assert rows[2]["pending"] is True and rows[2]["retained_events"] == 0
    assert rows[4]["pending"] is False and rows[4]["retained_events"] == 1
