"""The JSON-lines service survives bad input, and a client can release what it generated."""

from __future__ import annotations

import io
import json

import pytest
import torch
from test_multitrack_runtime import _line, _runtime, _serve

from megatoken.multitrack_runtime import MultiTrackGenerator, MultiTrackRuntime
from megatoken.multitrack_service import MultiTrackService


def _service():
    model, _ = _runtime()
    return MultiTrackService(MultiTrackRuntime(MultiTrackGenerator(model)))


def test_undecodable_input_bytes_fail_that_request_only():
    raw = b"\xff\xfe not json\n" + json.dumps({"op": "open", "id": 7}).encode() + b"\n"
    results = io.StringIO()
    _service().serve(io.TextIOWrapper(io.BytesIO(raw), encoding="utf-8"), results)
    first, second = (json.loads(line) for line in results.getvalue().splitlines())
    assert first["ok"] is False
    assert second["ok"] is True and second["id"] == 7


def test_a_lone_surrogate_in_a_request_id_is_answered_in_strict_json():
    wire = io.BytesIO()
    output = io.TextIOWrapper(wire, encoding="utf-8")  # strict: a raw surrogate would raise
    request = '{"op": "status", "id": "\\ud800"}\n'
    _service().serve(io.StringIO(request), output)
    output.flush()
    answer = json.loads(wire.getvalue().decode("utf-8"))
    assert answer["ok"] is False  # no such session, but the reply was written
    assert answer["id"] == "\ud800"


def test_a_generated_answer_reports_the_revision_that_releases_it():
    model, _ = _runtime()

    def requests(results):
        yield _line({"op": "open", "id": 1})
        sid = json.loads(results.getvalue().splitlines()[-1])["session"]
        yield _line(
            {
                "op": "observe",
                "id": 2,
                "session": sid,
                "events": [{"track": "user", "modality": "text", "payload_b64": "aGk="}],
            }
        )
        yield _line(
            {
                "op": "generate",
                "id": 3,
                "session": sid,
                "effort": 1,
                "seed": 1,
                "outputs": [{"track": "reply", "modality": "text", "max_tokens": 4}],
            }
        )
        reply = json.loads(results.getvalue().splitlines()[-1])
        yield _line(
            {
                "op": "release_generation",
                "id": 4,
                "session": sid,
                "selection_revision": reply["selection_revision"],
            }
        )

    rows = _serve(model, requests)
    assert all(row["ok"] for row in rows), rows
    assert rows[2]["selection_revision"] == rows[2]["revision"]


def test_add_files_without_a_knowledge_base_root_is_refused(tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("not for the model")
    request = {"op": "add_files", "id": 1, "collection": "c", "paths": [str(secret)]}
    results = io.StringIO()
    _service().serve(io.StringIO(json.dumps(request) + "\n"), results)
    answer = json.loads(results.getvalue())
    assert answer["ok"] is False and "kb-root" in answer["error"]


def test_a_hung_tool_call_times_out_and_does_not_block_closing():
    import threading
    import time

    from megatoken.multitrack_runtime import ConcurrentToolRunner

    release = threading.Event()
    runner = ConcurrentToolRunner(1, call_timeout=0.05)
    try:
        runner.submit("hung", lambda: (release.wait(30), b"late")[1])
        time.sleep(0.15)
        results, failures = runner.collect_ready()
        assert results == {} and isinstance(failures["hung"], TimeoutError)
        assert not runner.pending
        started = time.monotonic()
        runner.close()  # must not wait for the hung thread
        assert time.monotonic() - started < 5
    finally:
        release.set()


def test_non_finite_media_features_are_refused():
    import base64

    import numpy as np

    from megatoken.multitrack_service import _float_tensor

    bad = np.array([1.0, np.nan, 2.0, 3.0], dtype="<f4")
    payload = base64.b64encode(bad.tobytes()).decode("ascii")
    with pytest.raises(ValueError, match="non-finite"):
        _float_tensor(payload, [4], torch.device("cpu"))


def test_an_oversized_request_line_fails_alone_and_a_string_flag_is_refused(monkeypatch):
    import megatoken.multitrack_service as service

    monkeypatch.setattr(service, "MAX_REQUEST_CHARS", 64)
    big = '{"op": "open", "id": 1, "pad": "' + "x" * 500 + '"}\n'
    ok = json.dumps({"op": "open", "id": 2}) + "\n"
    results = io.StringIO()
    _service().serve(io.StringIO(big + ok), results)
    first, second = (json.loads(line) for line in results.getvalue().splitlines())
    assert first["ok"] is False and "exceeds" in first["error"]
    assert second["ok"] is True and second["id"] == 2

    with pytest.raises(ValueError, match="boolean"):
        service._flag("false", "writable")
