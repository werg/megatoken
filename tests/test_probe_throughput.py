"""Smoke test for scripts/probes/probe_throughput.py (tiny model, CPU, harness check only)."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "probes" / "probe_throughput.py"


def load_probe():
    spec = importlib.util.spec_from_file_location("probe_throughput", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["probe_throughput"] = module
    spec.loader.exec_module(module)
    return module


def finite_numbers(value) -> bool:
    if isinstance(value, dict):
        return all(finite_numbers(v) for v in value.values())
    if isinstance(value, list):
        return all(finite_numbers(v) for v in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def test_smoke_produces_expected_keys_and_finite_numbers(tmp_path):
    probe = load_probe()
    output = tmp_path / "result.json"
    result = probe.main(["--smoke", "--output", str(output)])
    assert json.loads(output.read_text()).keys() == json.loads(json.dumps(result)).keys()
    assert finite_numbers(result)
    cell = result["train"]["tiny"]["cells"][0]
    assert cell["status"] == "ok" and cell["tokens_per_s"] > 0
    assert result["decode"]["tiny"]["rows"][0]["ms_per_chunk"] > 0
    history = result["history"]["tiny"]
    assert history["bound"]["enforced"] and history["bound"]["matches_docs"]
    for row in history["rows"]:
        assert row["kv_bytes"] == row["kv_bytes_documented"]
        assert row["predict_ms"] > 0
    needle = result["needle"]["tiny"]["rows"]
    assert len(needle) == 2 and all(row["stages_read"] == 4 for row in needle)
    assert all(row["ratio_to_uniform"] > 0 for row in needle)
    projection = result["projection"]["tiny"]
    assert projection["tokens"] == pytest.approx(
        projection["local_tokens_per_s"] * projection["assumed_speed_ratio"] * 3600
    )


def test_projection_requires_explicit_speed_ratio():
    probe = load_probe()
    with pytest.raises(SystemExit):
        probe.main(["--smoke", "--gpu-hours", "10", "--speed-ratio", "0"])
