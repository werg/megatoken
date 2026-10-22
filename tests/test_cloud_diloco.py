"""Offline checks of the DiLoCo-on-sandboxes launcher (no API calls)."""

from __future__ import annotations

import json
import stat
import subprocess

import pytest

from megatoken import cloud, cloud_diloco


def _spec(**overrides):
    return {
        "schema": cloud_diloco.SCHEMA,
        "id": "unit-run",
        "image": "nvcr.io/nvidia/pytorch:26.09-py3",
        "lifetime_hours": 4,
        "workers": 2,
        "config": "configs/train/x.yaml",
        "pip": ["pyarrow"],
        "data": {"inputs": [{"path": "out/a", "url": "https://e.invalid/a", "sha256": "0" * 64}]},
        "coordinator": {"resources": {"cpu": "4", "memory": "16Gi"}, "args": ["--quorum", "2"]},
        "learners": {"resources": {"cpu": "8", "memory": "32Gi", "gpu": 1}},
        **overrides,
    }


def test_role_jobs_are_valid_and_never_carry_secrets():
    spec = cloud_diloco.validate(_spec())
    inputs = spec["data"]["inputs"]
    coordinator = cloud.validate(
        {"schema": cloud.SCHEMA, **cloud_diloco.coordinator_job(spec, inputs)}
    )
    learner = cloud.validate({"schema": cloud.SCHEMA, **cloud_diloco.learner_job(spec, inputs, 1)})
    assert coordinator["serve_port"] == cloud_diloco.PORT and "gpu" not in coordinator["resources"]
    assert learner["resources"]["gpu"] == 1 and learner["env"]["MEGATOKEN_WORKER_ID"] == "1"
    for job in (coordinator, learner):
        command = job["command"][2]
        # Tokens are read from secret files inside the sandbox, never spelled out.
        assert "$MEGATOKEN_CW_SECRETS/diloco-token" in command
        assert "TOKEN=" not in json.dumps(job["env"])
        request = cloud.request(job)
        assert "services" in request if job is coordinator else "services" not in request
    assert "megatoken diloco serve" in coordinator["command"][2]
    assert "megatoken diloco export" in coordinator["command"][2]
    assert "gateway-token" in learner["command"][2]


@pytest.mark.parametrize(
    "change",
    [
        {"schema": "other"},
        {"workers": 0},
        {"data": {"peer": "x"}},
        {"learners": {"resources": {"cpu": "8", "memory": "32Gi"}}},
    ],
)
def test_invalid_runs_are_rejected(change):
    with pytest.raises(cloud.CloudError):
        cloud_diloco.validate(_spec(**change))


def test_derive_sets_the_coordinator_url_and_checks_the_worker_count(tmp_path):
    config = tmp_path / "train.json"
    config.write_text(
        json.dumps(
            {
                "model": "tiny",
                "budget": 64,
                "decoupled": {
                    "coordinator_url": "https://placeholder",
                    "run_id": "r",
                    "workers": 2,
                },
            }
        )
    )
    output = tmp_path / "run" / "train.json"
    cloud_diloco.derive(config, "https://8080-abc.ep1.cwsandbox.com", 2, output)
    assert json.loads(output.read_text())["decoupled"]["coordinator_url"] == (
        "https://8080-abc.ep1.cwsandbox.com"
    )
    with pytest.raises(SystemExit, match="workers"):
        cloud_diloco.derive(config, "https://x", 3, output)


class _Done:
    def __init__(self, value=None):
        self.value = value

    def result(self):
        return self.value


class _Token:
    def get_secret_value(self):
        return "share-secret"


def test_start_delivers_generated_files_secrets_and_peer_share_tokens(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    git = ["git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t"]
    subprocess.run([*git, "init", "-q"], check=True)
    (repo / "a.py").write_text("x = 1\n")
    subprocess.run([*git, "add", "a.py"], check=True)
    subprocess.run([*git, "commit", "-qm", "init"], check=True)
    monkeypatch.setattr(cloud, "STATE", tmp_path / "state")
    monkeypatch.setattr(cloud, "SECRETS", tmp_path / "state" / "secrets")
    written = {}

    class Sandbox:
        sandbox_id = "coord-1"
        endpoint_share_token = _Token()
        service_urls = ((8080, "", "https://8080-coord-1.example"),)

        def wait(self, timeout):
            pass

        def write_file(self, path, data):
            written[path] = data
            return _Done()

    monkeypatch.setattr(cloud, "_create", lambda job: Sandbox())
    monkeypatch.setattr(cloud, "_push", lambda sandbox, files: None)
    (tmp_path / "state" / "secrets").mkdir(parents=True)
    (tmp_path / "state" / "secrets" / "prep-9.share").write_text("peer-secret")
    job = {
        "schema": cloud.SCHEMA,
        "id": "coordinator",
        "image": "i",
        "resources": {"cpu": "1", "memory": "1Gi"},
        "lifetime_hours": 1,
        "command": ["true"],
        "outputs": ["out"],
        "serve_port": 8080,
        "inputs": [
            {"path": "out/a", "url": "https://prep/out/a", "sha256": "0" * 64, "peer": "prep-9"}
        ],
    }
    entry, _ = cloud.start(
        cloud.validate(job),
        repo,
        secrets={"diloco-token": b"bearer"},
        files=lambda sandbox: {"run/coordinator-url": cloud.service_url(sandbox).encode()},
    )
    assert entry["url"] == "https://8080-coord-1.example"
    secrets = cloud.SECRETS / "coord-1.share"
    assert secrets.read_text() == "share-secret"
    assert stat.S_IMODE(secrets.stat().st_mode) == 0o600
    assert written[f"{cloud.WORK}/.cloud/secrets/diloco-token"] == b"bearer"
    assert written[f"{cloud.WORK}/.cloud/secrets/share-prep-9"] == b"peer-secret"
    spec = json.loads(written[f"{cloud.WORK}/.cloud/inputs.json"])
    paths = {item["path"]: item for item in spec["inputs"]}
    assert paths["out/a"]["peer"] == "prep-9"
    assert "run/coordinator-url" in paths
    # No secret value ever reaches the input spec or the run records.
    records = (tmp_path / "state" / "sandboxes.jsonl").read_text()
    for value in ("share-secret", "peer-secret", "bearer"):
        assert value not in json.dumps(spec) and value not in records
