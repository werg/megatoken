"""The rehearsal and serve-soak probes run in smoke mode and report what they should.

These tests run the probe scripts as subprocesses (the way an owner would), so they take about a
minute.  They check the shape of the reports and that every number is finite; they do not require
every stage to pass, because a probe that finds a real fault in the pipeline must still be able to
write its report.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REHEARSAL = ROOT / "scripts/probes/probe_rehearsal.py"
SOAK = ROOT / "scripts/probes/probe_serve_soak.py"

STAGES = [
    "launch_readiness",
    "build_data",
    "train_b0",
    "resume_b0",
    "resume_exactness",
    "train_p1_text",
    "train_p1_mixture",
    "train_p1_mixture_no_eval",
    "resume_p1_mixture",
    "export_bundle",
    "serve_client",
    "serve_add_files",
    "serve_media_bundle",
    "evaluate",
    "serve_soak",
]


def strict_json(path: Path):
    """Parse a report, rejecting NaN and infinity (Python's json accepts them by default)."""

    def refuse(token):
        raise ValueError(f"non-finite number {token} in {path}")

    return json.loads(path.read_text(), parse_constant=refuse)


def assert_all_finite(value):
    if isinstance(value, float):
        assert math.isfinite(value)
    elif isinstance(value, dict):
        for item in value.values():
            assert_all_finite(item)
    elif isinstance(value, list):
        for item in value:
            assert_all_finite(item)


def run(script: Path, tmp_path: Path, *arguments: str, timeout: int = 1500):
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    output = tmp_path / "result.json"
    process = subprocess.run(
        [sys.executable, str(script), *arguments, "--output", str(output)],
        capture_output=True,
        text=True,
        env=env,
        timeout=timeout,
        cwd=tmp_path,
    )
    return process, output


def load_script(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


def test_launch_readiness_catches_the_known_configuration_gaps():
    probe = load_script(REHEARSAL)
    result = probe.launch_readiness(ROOT)
    findings = result["findings"]
    # Every stage stream now has a producer (M1 and M2 used to name `base_replay`, which had none).
    assert findings["streams_without_any_producer"] == []
    assert not findings["streams_without_producer_by_stage"].get("M1")
    assert not findings["streams_without_producer_by_stage"].get("M2")
    # The one pretraining example declares a root for aligned tasks.
    for name in ("pretraining",):
        config = next(c for c in result["configs"] if f"{name}.example" in c["config"])
        assert config["roots_listed"] == 9
        assert config["missing_streams"] == []
        assert config["unfilled_placeholders"], "PIN_ placeholders must be reported"
    assert findings["phases_without_an_example_config"] == []
    assert findings["configs_without_default_memory"] == []
    # nothing is silently dropped: every example config is audited
    audited = {Path(c["config"]).name for c in result["configs"]}
    assert {p.name for p in (ROOT / "configs/train").glob("*.yaml")} <= audited
    assert_all_finite(result)
    json.dumps(result, allow_nan=False)


def test_readiness_only_command_writes_a_report(tmp_path):
    process, output = run(REHEARSAL, tmp_path, "--readiness-only", timeout=300)
    assert process.returncode == 0, process.stdout + process.stderr
    report = strict_json(output)
    assert [s["name"] for s in report["stages"]] == ["launch_readiness"]
    assert report["stages"][0]["status"] == "pass"


def test_rehearsal_smoke_reports_every_stage_with_finite_numbers(tmp_path):
    process, output = run(
        REHEARSAL, tmp_path, "--smoke", "--workdir", str(tmp_path / "work"), "--seed", "17"
    )
    assert output.exists(), process.stdout + process.stderr
    report = strict_json(output)
    assert_all_finite(report)
    assert report["probe"] == "rehearsal" and report["smoke"] is True
    names = [s["name"] for s in report["stages"]]
    assert names == STAGES
    for stage in report["stages"]:
        assert stage["status"] in {"pass", "fail", "skip"}
        assert stage["seconds"] >= 0
        if stage["status"] == "fail":
            assert stage["error"] and stage["traceback"]
    status = {s["name"]: s["status"] for s in report["stages"]}
    # the exit code agrees with the report
    assert (process.returncode != 0) == bool(report["failed"])
    # the core chain must work end to end
    for name in (
        "launch_readiness",
        "build_data",
        "train_b0",
        "resume_b0",
        "resume_exactness",
        "train_p1_text",
        "export_bundle",
        "serve_client",
        "evaluate",
        "serve_soak",
    ):
        assert status[name] == "pass", report["stages"][names.index(name)]["error"]
    # the stage mixture must run, with held-out evaluation or (if that fails) without it
    assert "pass" in (status["train_p1_mixture"], status["train_p1_mixture_no_eval"])
    details = {s["name"]: s["detail"] for s in report["stages"]}
    resume = details["resume_b0"]
    assert resume["steps_contiguous"] and resume["global_step"] == 2
    assert resume["counters"]["I"] > resume["counters_before_resume"]["I"]
    assert details["resume_exactness"]["max_abs_difference"] == 0.0
    data = details["build_data"]
    assert data["p1_streams_covered"] == data["p1_streams"] and not data["p1_streams_not_built"]
    assert set(data["streams"]["dynamic_execution"]) >= {"kind", "constructor"}
    served = details["serve_client"]
    assert served["requests"] >= 20 and not served["failures"] and served["exit_code"] == 0
    soak = details["serve_soak"]
    assert soak["with_release"]["first_failure"] is None
    assert soak["limit_confirmed"] in (True, False)


def test_serve_soak_smoke_reproduces_the_retained_output_limit(tmp_path):
    process, output = run(SOAK, tmp_path, "--smoke", "--seed", "17", timeout=900)
    assert output.exists(), process.stdout + process.stderr
    report = strict_json(output)
    assert_all_finite(report)
    assert report["probe"] == "serve_soak" and report["requests"] == 70
    # a client that releases every reply never fails
    released = report["with_release"]["summary"]
    assert released["first_failure"] is None
    assert released["completed_rounds"] == 70 and released["unreleased_revisions"] == 0
    assert released["tracks_created"] >= 5 and released["lifecycle_actions"] >= 10
    assert released["retained_events_max"] <= 16
    assert report["server_exit_with_release"] == 0
    # a client that never releases hits the limit at the 65th generate
    finding = report["finding"]
    unreleased = report["without_release"]["summary"]
    if finding["limit_confirmed"]:
        assert finding["generates_that_succeeded_without_release"] == 64
        assert finding["failed_at_generate"] == 65
        assert "Retained output capacity exhausted" in finding["error"]
        failure = unreleased["first_failure"]
        assert failure["failed_request"] == "generate" and failure["round"] == 65
        assert failure["exchanges_leading_to_failure"], "the exact exchanges must be reported"
        assert report["recovery"]["recovered"] is True
    else:  # the limit was removed or raised: then nothing may fail in 70 rounds
        assert unreleased["first_failure"] is None
    assert report["ok"] is True
    assert process.returncode == 0, process.stdout + process.stderr


@pytest.mark.parametrize("flag", ["--requests", "--seed"])
def test_soak_rejects_nonsense_arguments(tmp_path, flag):
    process = subprocess.run(
        [sys.executable, str(SOAK), flag, "not-a-number"], capture_output=True, text=True
    )
    assert process.returncode == 2
