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
