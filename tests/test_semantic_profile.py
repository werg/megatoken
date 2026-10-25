"""Measure actual serving and preserve failed output, without taking an optimizer step."""

import io
import json
import runpy
import sys
from pathlib import Path

import torch

from megatoken.io import content_hash
from megatoken.semantic_profile import percentile
from megatoken.served import export_served


def test_real_serving_process_profile_is_separate_from_rpc(tmp_path, monkeypatch, capsys):
    from test_semantic_model_adapter import fixture
    from test_served import checkpoint_model

    torch.set_num_threads(2)
    _, checkpoint = checkpoint_model(tmp_path / "checkpoint")
    served = tmp_path / "served"
    manifest = export_served(checkpoint, served)
    report_path = tmp_path / "profile.json"
    requests = [
        {
            "id": "init",
            "op": "initialize",
            "seed": 17,
            "checkpointIdentity": manifest["model_sha256"],
        },
        {"id": "first", "op": "select", "prefix": fixture()},
        {"id": "cancel", "op": "cancel", "session": "session1", "invocation": "i1"},
        {"id": "bad", "op": "unknown"},
    ]
    monkeypatch.setattr(sys, "stdin", io.StringIO("\n".join(json.dumps(row) for row in requests)))
    script = Path(__file__).parents[1] / "scripts/serve_selected_semantic.py"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(script),
            "--served",
            str(served),
            "--max-tokens",
            "4",
            "--profile-output",
            str(report_path),
        ],
    )
    entry = runpy.run_path(str(script))
    assert entry["main"]() == 0
    replies = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert len(replies) == len(requests) and replies[0]["ok"]
    report = json.loads(report_path.read_text())
    assert report["schema"] == "semantic-serving-profile/1"
    assert report["declaredOrigin"] == "fixed" and report["optimizerSteps"] == 0
    assert report["learnedSuccess"] is None and report["runtimeClosed"]
    assert report["coldLoadSeconds"] > 0 and report["operations"]["select"]["count"] == 1
    assert report["operations"]["select"]["warmP50Seconds"] is None
    assert report["memory"]["hostLifetimePeakRssBytes"] > 0
    assert report["memory"]["cudaPeakAllocatedBytes"] is None
    assert report["stateBeforeRuntimeClose"]["selectedGroups"] == 0
    assert report["stateBeforeRuntimeClose"]["reservedOutputBytes"] == 0
    assert report["stateBeforeRuntimeClose"]["activeInvocations"] == 0
    for reply, row in zip(replies, report["records"], strict=True):
        assert row["failure"] == reply.get("error")
        assert row["wallSeconds"] >= 0
    assert report.pop("digest") == content_hash(report)


def test_percentiles_do_not_invent_warm_samples():
    assert percentile([], 0.95) is None
    assert percentile([5, 1, 2, 4, 3], 0.5) == 3
    assert percentile([5, 1, 2, 4, 3], 0.95) == 5
