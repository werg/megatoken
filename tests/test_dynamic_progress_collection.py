"""Recover exact committed bytes without stopping training or copying credentials."""

import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


def capture_script():
    path = Path(__file__).resolve().parents[1] / "scripts/collect_dynamic_progress.py"
    spec = importlib.util.spec_from_file_location("dynamic_collection", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SNAPSHOT


def capture(work, root):
    result = subprocess.run(
        [sys.executable, "-c", capture_script(), str(work), str(root)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    marker = json.loads(result.stdout)
    return json.loads((work / marker["manifest"]).read_text())


def test_progress_capture_retains_pruned_checkpoint_and_immutable_reports(tmp_path):
    work = tmp_path / "work"
    root = work / "out/experiment"
    checkpoint = root / "training/checkpoints/current"
    checkpoint.mkdir(parents=True)
    (checkpoint / "state.pt").write_bytes(b"exact model checkpoint")
    (checkpoint / "transducer.pt").write_bytes(b"exact bank checkpoint")
    metadata = checkpoint / "metadata.json"
    metadata.write_text(
        json.dumps(
            {
                "format_version": 16,
                "checkpoint_name": checkpoint.name,
                "state_sha256": hashlib.sha256((checkpoint / "state.pt").read_bytes()).hexdigest(),
                "transducer_checkpoint": {
                    "file": "transducer.pt",
                    "sha256": hashlib.sha256(
                        (checkpoint / "transducer.pt").read_bytes()
                    ).hexdigest(),
                },
            }
        )
    )
    (checkpoint.parent / "latest.json").write_text(
        json.dumps(
            {
                "checkpoint": checkpoint.name,
                "metadata_sha256": hashlib.sha256(metadata.read_bytes()).hexdigest(),
            }
        )
    )
    report = root / "status.json"
    report.write_text('{"state":"training"}')
    secret = root / "packs/secret.json"
    secret.parent.mkdir()
    secret.write_text('{"access_token":"must-not-be-copied"}')
    receipt = capture(work, root)
    files = {item["original"]: item for item in receipt["files"]}
    assert not receipt["capture_errors"]
    assert "packs/secret.json" not in files
    shutil.rmtree(checkpoint)
    report.write_text('{"state":"changed"}')
    assert (
        work / files["training/checkpoints/current/state.pt"]["path"]
    ).read_bytes() == b"exact model checkpoint"
    assert json.loads((work / files["status.json"]["path"]).read_text()) == {"state": "training"}
    for item in files.values():
        assert hashlib.sha256((work / item["path"]).read_bytes()).hexdigest() == item["sha256"]


def test_uncommitted_data_and_symlinks_cannot_enter_a_prepared_backup(tmp_path):
    work = tmp_path / "work"
    root = work / "out/experiment"
    data = root / "prepared"
    (data / "media").mkdir(parents=True)
    (data / "index.sqlite").write_bytes(b"in progress")
    assert not capture(work, root)["files"]
    (data / "manifest.json").write_text('{"schema_version":6,"shards":[]}')
    (data / "episodes-000000.jsonl").write_bytes(b"{}\n")
    outside = work / "secret"
    outside.write_bytes(b"private credentials")
    (data / "media/escape.npy").symlink_to(outside)
    receipt = capture(work, root)
    files = {item["original"] for item in receipt["files"]}
    assert "prepared/index.sqlite" in files
    assert "prepared/episodes-000000.jsonl" in files
    assert "prepared/media/escape.npy" not in files
    assert receipt["capture_errors"]


@pytest.mark.parametrize("relative", ["out", "private", "out/../../other"])
def test_capture_scope_is_one_experiment_directory(tmp_path, relative):
    work = tmp_path / "work"
    result = subprocess.run(
        [sys.executable, "-c", capture_script(), str(work), str(work / relative)],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
