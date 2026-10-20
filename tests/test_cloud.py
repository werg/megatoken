"""Offline checks of the CoreWeave job contract (no API calls)."""

from __future__ import annotations

import json
import subprocess

import pytest

from megatoken import cloud


def _job(**overrides):
    return {
        "schema": cloud.SCHEMA,
        "id": "unit",
        "image": "nvcr.io/nvidia/pytorch:26.09-py3",
        "resources": {"cpu": "8", "memory": "64Gi", "gpu": 1},
        "lifetime_hours": 2,
        "command": ["python", "-c", "print(1)"],
        "outputs": ["out"],
        "inputs": [
            {"path": "inputs/a.bin", "url": "https://example.invalid/a", "sha256": "0" * 64}
        ],
        **overrides,
    }


def test_request_mounts_bootstrap_and_never_carries_credentials(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "secret-value")
    request = cloud.request(cloud.validate(_job(env={"HF_HOME": "/work/hf"}, pip=["x==1"])))
    assert request["resources"] == {"cpu": "8", "memory": "64Gi", "gpu": {"count": 1}}
    assert request["max_lifetime_seconds"] == 7200
    assert cloud.BOOTSTRAP in request["mounted_files"]
    assert request["environment_variables"]["MEGATOKEN_CW_OUTPUTS"] == "out"
    assert request["environment_variables"]["MEGATOKEN_CW_PIP"] == "x==1"
    assert "secret-value" not in json.dumps(request)


@pytest.mark.parametrize(
    "change",
    [
        {"schema": "other"},
        {"lifetime_hours": 0},
        {"resources": {"cpu": "1", "memory": "1Gi", "gpu": 2}},
        {"outputs": ["/abs"]},
        {"inputs": [{"path": "../x", "url": "u", "sha256": "0" * 64}]},
        {"inputs": [{"path": "x", "sha256": "0" * 64}]},
        {"inputs": [{"path": "x", "url": "u", "source": "s", "sha256": "0" * 64}]},
    ],
)
def test_invalid_jobs_are_rejected(change):
    with pytest.raises(cloud.CloudError):
        cloud.validate(_job(**change))


def test_code_snapshot_requires_a_clean_commit(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    git = ["git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t"]
    subprocess.run([*git, "init", "-q"], check=True)
    (repo / "a.py").write_text("x = 1\n")
    subprocess.run([*git, "add", "a.py"], check=True)
    subprocess.run([*git, "commit", "-qm", "init"], check=True)
    snapshot = cloud.code_snapshot(repo, tmp_path / "code.tar.gz")
    assert len(snapshot["commit"]) == 40
    assert snapshot["sha256"] == cloud.sha256_file(tmp_path / "code.tar.gz")
    (repo / "a.py").write_text("x = 2\n")
    with pytest.raises(cloud.CloudError, match="Commit"):
        cloud.code_snapshot(repo, tmp_path / "again.tar.gz")


def test_shared_credential_files_are_refused(tmp_path):
    secrets = tmp_path / "coreweave.env"
    secrets.write_text("WANDB_API_KEY=x\n")
    secrets.chmod(0o644)
    with pytest.raises(cloud.CloudError, match="chmod 600"):
        cloud.load_credentials(secrets)
