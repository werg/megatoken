"""Causal ready-event runtime and JSONL mixed-media transport."""

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


def _runtime():
    torch.manual_seed(27)
    model = FacetedGenerator(FacetedCore(ModelConfig.tiny()), ByteCodec.byte_fallback())
    return model, MultiTrackRuntime(MultiTrackGenerator(model))


def test_speculative_groups_do_not_mutate_canonical_history():
    _, runtime = _runtime()
    session = runtime.open()
    runtime.observe(session, (ObservedEvent("user", "text", b"Speak briefly"),))
    committed = session.state.history.keys.clone()
    draft = runtime.speculate(
        session, tracks=("voice",),
        specs=(MediaOutputSpec("audio", sample_count=128),), effort=1,
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
        runtime.observe(session, (
            ObservedEvent("tool-1", "text", b"one", role_id=3),
            ObservedEvent("tool-2", "text", b"two", role_id=3),
        ))
        assert session.state.meridian_sizes == (2,)
    finally:
        pool.close()


def test_jsonl_multitrack_roundtrip():
    model, _ = _runtime()

    class Engine:
        device = torch.device("cpu")
        config = model.config

    engine = Engine()
    engine.model = model
    results = io.StringIO()

    def requests():
        yield json.dumps({"op": "open", "id": 1}) + "\n"
        opened = json.loads(results.getvalue().splitlines()[-1])
        sid = opened["session"]
        yield json.dumps({
            "op": "observe", "id": 2, "session": sid,
            "events": [{"track":"user","modality":"text","payload_b64":"aGk="}],
        }) + "\n"
        yield json.dumps({
            "op":"generate", "id":3, "session":sid, "effort":1,
            "outputs":[
                {"track":"reply","modality":"text","max_tokens":8},
                {"track":"voice","modality":"audio","sample_count":64},
            ],
        }) + "\n"
        yield json.dumps({"op":"status","id":4,"session":sid}) + "\n"

    serve_multitrack_jsonl(engine, requests(), results)
    rows = [json.loads(line) for line in results.getvalue().splitlines()]
    assert len(rows) == 4 and all(row["ok"] for row in rows)
    assert rows[2]["outputs"][1]["shape"] == [64]
    assert rows[2]["tape"]["modalities"] == ["text", "audio"]
    assert rows[3]["retained_events"] == 3
