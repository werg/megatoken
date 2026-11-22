"""Exercise the JSONL loop without replacing parsing with a direct adapter call."""

import io
import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace


def test_jsonl_rejects_ambiguous_or_nonstandard_json_and_recovers(monkeypatch, capsys):
    script = Path(__file__).parents[1] / "scripts" / "serve_semantic_v8.py"
    main = runpy.run_path(str(script))["main"]
    calls = []
    adapter = SimpleNamespace(sessions={}, metrics=lambda: calls.append("metrics") or {})
    runtime = SimpleNamespace(close=lambda: calls.append("close"))
    bundle = SimpleNamespace(
        model=None,
        chunking=None,
        manifest={"model_sha256": "a" * 64, "semantic_v8_lineage_digest": "b" * 64},
    )
    monkeypatch.setitem(main.__globals__, "ServedBundle", lambda *args, **kwargs: bundle)
    monkeypatch.setitem(main.__globals__, "MultiTrackGenerator", SimpleNamespace(of=lambda _: None))
    monkeypatch.setitem(main.__globals__, "MultiTrackRuntime", lambda *args, **kwargs: runtime)

    def construct(*args, **kwargs):
        assert kwargs["training_identity"] == "b" * 64
        return adapter

    monkeypatch.setitem(main.__globals__, "NativeSemanticInferenceV8", construct)
    monkeypatch.setattr(sys, "argv", [str(script), "--served", "unused"])
    monkeypatch.setattr(
        sys,
        "stdin",
        SimpleNamespace(
            buffer=io.BytesIO(
                b'{"id":"duplicate","op":"unknown","op":"metrics"}\n'
                b'{"id":"nonfinite","op":"metrics","extra":NaN}\n'
                b'{"id":"utf8","op":"metrics","extra":"\xff"}\n'
                b'{"id":"valid","op":"metrics"}\n'
            )
        ),
    )
    assert main() == 0
    replies = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert len(replies) == 4
    assert [reply["ok"] for reply in replies] == [False, False, False, True]
    assert "duplicate JSON object key" in replies[0]["error"]["message"]
    assert "nonstandard JSON constant" in replies[1]["error"]["message"]
    assert replies[-1]["id"] == "valid"
    assert calls == ["metrics", "close"]


def test_opt_in_jsonl_commit_readback_requires_same_locked_ofd_and_restores_checkpoint(
    monkeypatch, capsys, tmp_path
):
    import copy
    import fcntl
    import os

    from megatoken.data.semantic_json import stable_algorithm_hash

    script = Path(__file__).parents[1] / "scripts" / "serve_semantic_v8.py"
    main = runpy.run_path(str(script))["main"]
    namespace = main.__globals__
    model = {"checkpoint": "a" * 64, "tokenizer": "b" * 64}
    public = {"timeline": [{"mode": "generated", "fields": {"line": "answer"}}]}
    request_identity = {
        "schema": "semantic-native-request-identity/1",
        "session_id": "lock-session",
        "stream_id": "lock-stream",
        "session_generation": 4,
        "ordinal": 1,
        "operation": "commit",
        "model_identity": model,
    }
    request = {
        "id": "v8-native-4-1",
        "op": "commit",
        "nativeRequest": request_identity,
        "session": "session-a",
        "selection": {"selectionId": "v8:0"},
        "public": public,
        "contexts": [],
        "invocationId": "invocation-a",
        "record": "answer",
    }
    line = json.dumps(request, separators=(",", ":")).encode() + b"\n"
    calls = []

    class Adapter:
        def __init__(self, runtime, **kwargs):
            self.runtime = runtime
            self.model_identity = "a" * 64
            self.codec = SimpleNamespace(fingerprint="b" * 64)
            self.training_identity = None
            self.initialization = {
                "checkpointIdentity": self.model_identity,
                "tokenizerIdentity": self.codec.fingerprint,
                "trainingIdentity": None,
            }
            self.attempts = []
            self.sessions = {}
            self.state = None

        def checkpoint(self):
            if self.state is None:
                raise ValueError("fresh adapter has no state")
            return copy.deepcopy(self.state)

        def restore(self, *, checkpoint):
            assert self.state is None
            self.state = copy.deepcopy(checkpoint)
            self.attempts = copy.deepcopy(checkpoint["attempts"])

        def cancel(self, **kwargs):
            return {"closed": True}

        def metrics(self):
            return {}

    class Runtime:
        memory = SimpleNamespace(sessions={})
        def close(self):
            return None

    bundle = SimpleNamespace(
        model=None,
        chunking=None,
        manifest={"model_sha256": model["checkpoint"]},
    )
    monkeypatch.setitem(namespace, "ServedBundle", lambda *args, **kwargs: bundle)
    monkeypatch.setitem(namespace, "MultiTrackGenerator", SimpleNamespace(of=lambda _: None))
    monkeypatch.setitem(namespace, "MultiTrackRuntime", lambda *args, **kwargs: Runtime())
    monkeypatch.setitem(namespace, "NativeSemanticInferenceV8", Adapter)

    def dispatch(adapter, value):
        calls.append(value["id"])
        attempt = {"selectionId": "v8:0", "status": "committed", "session": value["session"]}
        adapter.attempts = [attempt]
        adapter.state = {
            "schema": "semantic-model-checkpoint/8",
            "initialization": adapter.initialization,
            "attempts": adapter.attempts,
            "sessions": {
                value["session"]: {
                    "public": copy.deepcopy(value["public"]),
                    "unacknowledgedSelection": None,
                }
            },
        }
        adapter.state["digest"] = stable_algorithm_hash(adapter.state)
        return {
            "schema": "semantic-native-commit/8",
            "selectionId": "v8:0",
            "publicDigest": stable_algorithm_hash(value["public"]),
            "committed": True,
        }

    monkeypatch.setitem(namespace, "dispatch", dispatch)
    lock_path = tmp_path / "execution.lock"
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    fcntl.flock(fd, fcntl.LOCK_EX)
    argv = [
        str(script),
        "--served",
        "unused",
        "--request-journal",
        str(tmp_path / "journal"),
        "--execution-lock-fd",
        str(fd),
        "--execution-lock-session",
        "lock-session",
        "--execution-lock-stream",
        "lock-stream",
        "--execution-lock-path",
        str(lock_path),
    ]
    try:
        for expected_calls in (["v8-native-4-1"], ["v8-native-4-1"]):
            monkeypatch.setattr(sys, "argv", argv)
            monkeypatch.setattr(sys, "stdin", SimpleNamespace(buffer=io.BytesIO(line)))
            assert main() == 0
            output = capsys.readouterr().out.splitlines()
            assert len(output) == 1
            reply = json.loads(output[0])
            assert reply["id"] == request["id"]
            assert reply["ok"] is True
            assert reply["result"]["publicDigest"] == stable_algorithm_hash(public)
            assert calls == expected_calls
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def test_real_jsonl_subprocess_reads_exact_committed_reply_after_process_restart(
    tmp_path,
):
    import fcntl
    import json
    import os
    import subprocess
    import sys

    from megatoken.data.semantic_json import stable_algorithm_hash

    fixture = Path(__file__).parent / "fixtures" / "serve_semantic_v8_bridge_fixture.py"
    lock_path = tmp_path / "execution.lock"
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    fcntl.flock(fd, fcntl.LOCK_EX)
    public = {"timeline": [{"mode": "generated", "fields": {"line": "persisted"}}]}
    request = {
        "id": "v8-native-8-1",
        "op": "commit",
        "nativeRequest": {
            "schema": "semantic-native-request-identity/1",
            "session_id": "scope-session",
            "stream_id": "scope-stream",
            "session_generation": 8,
            "ordinal": 1,
            "operation": "commit",
            "model_identity": {"checkpoint": "a" * 64, "tokenizer": "b" * 64},
        },
        "session": "native-session",
        "selection": {"selectionId": "v8:0"},
        "public": public,
        "contexts": [],
        "invocationId": "invocation-a",
        "record": "persisted",
    }
    request_line = json.dumps(request, separators=(",", ":")).encode() + b"\n"
    calls_path = tmp_path / "calls.txt"
    env = dict(
        os.environ,
        JOURNAL_PATH=str(tmp_path / "journal"),
        LOCK_FD=str(fd),
        LOCK_PATH=str(lock_path),
        LOCK_SESSION="scope-session",
        LOCK_STREAM="scope-stream",
        CALLS_PATH=str(calls_path),
    )
    try:
        outputs = []
        for _ in range(2):
            child = subprocess.run(
                [sys.executable, str(fixture)],
                input=request_line,
                capture_output=True,
                timeout=45,
                pass_fds=(fd,),
                env=env,
                check=False,
            )
            assert child.returncode == 0, child.stderr.decode("utf-8", errors="replace")
            outputs.append(child.stdout)
        assert outputs[0] == outputs[1]
        reply = json.loads(outputs[1])
        assert reply["id"] == request["id"]
        assert reply["ok"] is True
        assert reply["result"]["publicDigest"] == stable_algorithm_hash(public)
        assert calls_path.read_text(encoding="utf-8").splitlines() == [request["id"]]
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)

def test_jsonl_stops_and_never_retries_after_commit_outcome_becomes_unknown(tmp_path):
    import fcntl
    import json
    import os
    import subprocess
    import sys

    fixture = Path(__file__).parent / "fixtures" / "serve_semantic_v8_bridge_fixture.py"
    lock_path = tmp_path / "execution.lock"
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    fcntl.flock(fd, fcntl.LOCK_EX)
    request = {
        "id": "v8-native-8-1",
        "op": "commit",
        "nativeRequest": {
            "schema": "semantic-native-request-identity/1",
            "session_id": "scope-session",
            "stream_id": "scope-stream",
            "session_generation": 8,
            "ordinal": 1,
            "operation": "commit",
            "model_identity": {"checkpoint": "a" * 64, "tokenizer": "b" * 64},
        },
        "session": "native-session",
        "selection": {"selectionId": "v8:0"},
        "public": {"timeline": [{"mode": "generated", "fields": {"line": "unknown"}}]},
        "contexts": [],
        "invocationId": "invocation-b",
        "record": "unknown",
    }
    line = json.dumps(request, separators=(",", ":")).encode() + b"\n"
    calls_path = tmp_path / "calls.txt"
    env = dict(
        os.environ,
        JOURNAL_PATH=str(tmp_path / "journal"),
        LOCK_FD=str(fd),
        LOCK_PATH=str(lock_path),
        LOCK_SESSION="scope-session",
        LOCK_STREAM="scope-stream",
        CALLS_PATH=str(calls_path),
        FAIL_AFTER_DISPATCH="1",
    )
    try:
        fixture_outputs = []
        for input_bytes in (line + line, line):
            child = subprocess.run(
                [sys.executable, str(fixture)],
                input=input_bytes,
                capture_output=True,
                timeout=45,
                pass_fds=(fd,),
                env=env,
                check=False,
            )
            assert child.returncode == 0, child.stderr.decode("utf-8", errors="replace")
            replies = [json.loads(row) for row in child.stdout.splitlines()]
            assert len(replies) == 1
            assert replies[0]["id"] == request["id"]
            assert replies[0]["ok"] is False
            assert replies[0]["error"]["category"] == "NativeRequestOutcomeUnknown"
            fixture_outputs.append(child.stdout)
        assert calls_path.read_text(encoding="utf-8").splitlines() == [request["id"]]
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def test_jsonl_server_refuses_abbreviated_recovery_flags_before_loading_model(monkeypatch, capsys):
    import pytest

    script = Path(__file__).parents[1] / "scripts" / "serve_semantic_v8.py"
    main = runpy.run_path(str(script))["main"]
    monkeypatch.setattr(sys, "argv", [str(script), "--served", "unused", "--request-j", "unused"])
    with pytest.raises(SystemExit) as refused:
        main()
    assert refused.value.code == 2
    assert "unrecognized arguments: --request-j" in capsys.readouterr().err
