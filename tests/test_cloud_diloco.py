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


CACHED = {"cache": {"run": "configs/cloud/prepare-full-agentic.json"}}


def test_stored_data_is_restored_by_learners_before_they_train():
    spec = cloud_diloco.validate(_spec(data=CACHED))
    assert cloud_diloco.data_inputs(spec["data"]) == []
    learner = cloud.validate({"schema": cloud.SCHEMA, **cloud_diloco.learner_job(spec, [], 0)})
    coordinator = cloud_diloco.coordinator_job(spec, [])
    command = learner["command"][2]
    assert (
        "megatoken.publish fetch --pin /work/in/cache-pin.json --dest /work/data/prepared" in command
    )
    assert command.index("publish fetch") < command.index("cloud_diloco derive")
    assert "cryptography>=43,<51" in learner["pip"] and "cache" not in coordinator["command"][2]
    assert "cache" not in json.dumps(
        cloud_diloco.learner_job(cloud_diloco.validate(_spec()), [], 0)
    )
    for bad in ({"cache": {}}, {"cache": "x"}, {"cache": {"run": "a", "other": 1}}):
        with pytest.raises(cloud.CloudError):
            cloud_diloco.validate(_spec(data=bad))


def test_launch_ensures_the_data_then_gives_the_key_to_learners_only(tmp_path, monkeypatch):
    from megatoken import cache, cloud_prepare

    repo, run = _launch_env(tmp_path, monkeypatch)
    run.write_text(json.dumps(_spec(data=CACHED)))
    key_file = cache.keygen(tmp_path / "keys" / "cache.key")
    token = tmp_path / "token"
    token.write_text("hf_secret\n")
    pin = tmp_path / "pin.json"
    pin.write_text('{"pin": true}')
    asked = []
    monkeypatch.setattr(
        cloud_prepare,
        "ensure",
        lambda path, repo_, **kwargs: (
            asked.append((path, kwargs)) or {"mode": "stored", "pin": str(pin)}
        ),
    )
    started = []

    def start(job, repo_, *, secrets=None, files=None, snapshot=None):
        started.append((job, sorted(secrets), files(object()) if files else {}))
        return {"sandbox_id": job["id"], "url": "https://c", "commit": "x"}, None

    monkeypatch.setattr(cloud, "start", start)
    monkeypatch.setattr(cloud_diloco, "_status", lambda *args: {})
    monkeypatch.setattr(cloud, "service_url", lambda sandbox: "https://c")
    _stoppable(monkeypatch)
    cloud_diloco.launch(run, repo, hf_token=token, cache_key=key_file)
    assert asked[0][0] == repo / "configs/cloud/prepare-full-agentic.json"
    (coordinator, c_secrets, c_files), *learners = started
    assert c_secrets == ["diloco-token"] and "in/cache-pin.json" not in c_files
    assert len(learners) == 2
    for job, secrets, files in learners:
        assert secrets == ["cache-key", "diloco-token", "gateway-token", "hf-token"]
        assert files["in/cache-pin.json"] == b'{"pin": true}'
        assert key_file.read_text().strip() not in json.dumps(job)


def test_the_coordinator_outlives_the_learners():
    spec = cloud_diloco.validate(_spec(lifetime_hours=4))
    inputs = spec["data"]["inputs"]
    coordinator = cloud_diloco.coordinator_job(spec, inputs)
    learner = cloud_diloco.learner_job(spec, inputs, 0)
    assert coordinator["lifetime_hours"] == learner["lifetime_hours"] + 2
    with pytest.raises(cloud.CloudError, match="lifetime_hours"):
        cloud_diloco.validate(_spec(lifetime_hours=720))


def _stoppable(monkeypatch, failing=()):
    stopped = []

    class Done:
        def result(self):
            return None

    class Sandbox:
        def __init__(self, sandbox_id):
            self.sandbox_id = sandbox_id

        def stop(self, **kwargs):
            if self.sandbox_id in failing:
                raise RuntimeError("gateway timeout")
            stopped.append(self.sandbox_id)
            return Done()

    monkeypatch.setattr(cloud, "_attach", Sandbox)
    monkeypatch.setattr(cloud, "record", lambda entry: None)
    return stopped


def test_every_sandbox_is_stopped_even_when_one_will_not_stop(monkeypatch, capsys):
    stopped = _stoppable(monkeypatch, failing={"b"})
    failed = cloud.stop_all(["a", "b", "c"], "launch failed")
    assert stopped == ["a", "c"] and failed == ["b"]
    assert "COULD NOT STOP b" in capsys.readouterr().err


def _launch_env(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    git = ["git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t"]
    subprocess.run([*git, "init", "-q"], check=True)
    (repo / "a.py").write_text("x = 1\n")
    subprocess.run([*git, "add", "a.py"], check=True)
    subprocess.run([*git, "commit", "-qm", "init"], check=True)
    monkeypatch.setattr(cloud, "STATE", tmp_path / "state")
    monkeypatch.setattr(cloud, "SECRETS", tmp_path / "state" / "secrets")
    monkeypatch.setattr(cloud_diloco, "RUNS", tmp_path / "state" / "diloco")
    monkeypatch.setattr(cloud, "share_token", lambda sandbox_id: "gateway")
    run = tmp_path / "run.json"
    run.write_text(json.dumps(_spec()))
    return repo, run


def test_launch_fails_fast_when_the_coordinator_exits_and_stops_everything(tmp_path, monkeypatch):
    repo, run = _launch_env(tmp_path, monkeypatch)
    stopped = _stoppable(monkeypatch)
    monkeypatch.setattr(
        cloud,
        "start",
        lambda *args, **kwargs: ({"sandbox_id": "coord", "url": "https://c", "commit": "x"}, None),
    )

    def unreachable(*args):
        raise OSError("connection refused")

    monkeypatch.setattr(cloud_diloco, "_status", unreachable)
    monkeypatch.setattr(cloud, "outputs_ready", lambda sandbox_id: True)
    monkeypatch.setattr(cloud_diloco.time, "sleep", lambda _: pytest.fail("kept waiting"))
    with pytest.raises(cloud.CloudError, match="exited before it became ready"):
        cloud_diloco.launch(run, repo)
    assert stopped == ["coord"]
    assert not cloud_diloco._run_path("unit-run").exists()


def test_launch_stops_the_rest_when_a_learner_cannot_start(tmp_path, monkeypatch):
    repo, run = _launch_env(tmp_path, monkeypatch)
    stopped = _stoppable(monkeypatch, failing={"learner-0"})
    starts = iter(
        [
            ({"sandbox_id": "coord", "url": "https://c", "commit": "x"}, None),
            ({"sandbox_id": "learner-0"}, None),
        ]
    )

    def start(*args, **kwargs):
        try:
            return next(starts)
        except StopIteration:
            raise cloud.CloudError("no capacity") from None

    monkeypatch.setattr(cloud, "start", start)
    monkeypatch.setattr(cloud_diloco, "_status", lambda *args: {})
    with pytest.raises(cloud.CloudError, match="no capacity"):
        cloud_diloco.launch(run, repo)
    # learner-0 would not stop, yet the coordinator behind it was still stopped.
    assert stopped == ["coord"]


def test_a_terminated_launcher_tears_its_sandboxes_down(monkeypatch):
    import signal

    monkeypatch.setattr(cloud, "load_credentials", lambda: None)
    monkeypatch.setattr(cloud_diloco, "status", lambda run_id: {})
    previous = signal.getsignal(signal.SIGTERM)
    try:
        cloud_diloco.main(["status", "r"])
        handler = signal.getsignal(signal.SIGTERM)
        assert handler is not previous and callable(handler)
        with pytest.raises(SystemExit) as exit_:
            handler(signal.SIGTERM, None)  # SystemExit unwinds through launch's teardown
        assert exit_.value.code == 143
    finally:
        signal.signal(signal.SIGTERM, previous)
