"""Run one DiLoCo phase on CoreWeave sandboxes: a CPU coordinator and N GPU learners.

A run file (``megatoken.cloud-diloco/1``, JSON) names the training config (a repo
path whose ``decoupled`` block describes the run), the worker count, the data
and both roles' resources:

    {"schema": "megatoken.cloud-diloco/1", "id": "qa-p1", "image": "...",
     "lifetime_hours": 24, "pip": ["pyarrow"], "workers": 4,
     "config": "configs/train/qa-pilot-p1-diloco.yaml",
     "data": {"peer": "<sandbox id>", "prefixes": ["out/corpus/datasets/rc-train"]},
     "coordinator": {"resources": {"cpu": "8", "memory": "32Gi"}, "args": ["--quorum", "3"]},
     "learners": {"resources": {"cpu": "8", "memory": "32Gi", "gpu": 1}}}

Data is built once in the cloud: ``data.peer`` is a finished ``megatoken.cloud``
job with ``serve_outputs``, whose outputs every sandbox fetches over HTTPS at
datacenter speed (``data.inputs`` may instead list ordinary job inputs). The
coordinator serves ``megatoken diloco serve`` on a share-token HTTPS endpoint;
learners reach it with the share token and a fresh DiLoCo bearer token, both
delivered as secret files. ``finish`` waits for every learner, stops the
coordinator, exports its global model (the next phase's seed) and collects it.

  python -m megatoken.cloud_diloco launch RUN.json [--repo .]
  python -m megatoken.cloud_diloco finish RUN_ID --dest DIR
  python -m megatoken.cloud_diloco status RUN_ID
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any

from megatoken import cloud

SCHEMA = "megatoken.cloud-diloco/1"
PORT = 8080
COORDINATOR_ROOT = f"{cloud.WORK}/coordinator"
EXPORT = "out/export"
RUN_CONFIG = f"{cloud.WORK}/run/train.json"
URL_FILE = "run/coordinator-url"  # an input: the URL exists only once the coordinator does
RUNS = cloud.STATE / "diloco"


def validate(spec: dict[str, Any]) -> dict[str, Any]:
    if spec.get("schema") != SCHEMA:
        raise cloud.CloudError(f"Run schema must be {SCHEMA}")
    for key in ("id", "image", "lifetime_hours", "workers", "config", "data"):
        if key not in spec:
            raise cloud.CloudError(f"Run lacks {key!r}")
    if type(spec["workers"]) is not int or not 1 <= spec["workers"] <= 64:
        raise cloud.CloudError("workers must be in [1, 64]")
    if set(spec["data"]) not in ({"peer", "prefixes"}, {"inputs"}):
        raise cloud.CloudError("data is {peer, prefixes} or {inputs}")
    if not spec.get("learners", {}).get("resources", {}).get("gpu"):
        raise cloud.CloudError("Learners train on one GPU each")
    return spec


def data_inputs(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Every pinned data file, as ordinary job inputs (a peer's are fetched from it)."""
    if "inputs" in data:
        return list(data["inputs"])
    peer = cloud._attach(data["peer"])
    if peer.exec(["test", "-f", f"{cloud.WORK}/.cloud/outputs-ready"]).result().returncode:
        raise cloud.CloudError(f"{data['peer']} has not finished its outputs")
    url = cloud.service_url(peer)
    files = json.loads(peer.read_file(f"{cloud.WORK}/.cloud/outputs.json").result())["files"]
    chosen = [
        {"path": item["path"], "sha256": item["sha256"], "url": f"{url}/{item['path']}"}
        for item in files
        if any(item["path"].startswith(prefix.rstrip("/") + "/") for prefix in data["prefixes"])
    ]
    if not chosen:
        raise cloud.CloudError(f"{data['peer']} serves no files under {data['prefixes']}")
    return [dict(item, peer=data["peer"]) for item in chosen]


def _shell(*lines: str) -> list[str]:
    return ["bash", "-c", "set -e; " + "; ".join(lines)]


# Each role first writes the run config: the repo config with this run's
# coordinator URL, checked against the declared worker count.
def _derive(workers: int) -> str:
    return (
        f'python -m megatoken.cloud_diloco derive --config "$MEGATOKEN_RUN_CONFIG" '
        f'--url "$(cat {cloud.WORK}/{URL_FILE})" --workers {workers} --output {RUN_CONFIG}'
    )


_TOKEN = 'export MEGATOKEN_DILOCO_TOKEN="$(cat "$MEGATOKEN_CW_SECRETS/diloco-token")"'


def coordinator_job(spec: dict[str, Any], inputs: list[dict[str, Any]]) -> dict[str, Any]:
    args = " ".join(spec.get("coordinator", {}).get("args", []))
    finish = f"{cloud.WORK}/.cloud/finish"
    return {
        "id": f"{spec['id']}-coordinator",
        "image": spec["image"],
        "resources": spec.get("coordinator", {}).get("resources", {"cpu": "8", "memory": "32Gi"}),
        "lifetime_hours": spec["lifetime_hours"],
        "pip": spec.get("pip", []),
        "inputs": inputs,
        "serve_port": PORT,
        "env": {"MEGATOKEN_RUN_CONFIG": spec["config"]},
        "command": _shell(
            _TOKEN,
            _derive(spec["workers"]),
            f"megatoken diloco init --config {RUN_CONFIG} --root {COORDINATOR_ROOT} {args}",
            f"megatoken diloco serve --root {COORDINATOR_ROOT} --host 0.0.0.0 --port {PORT} & "
            "server=$!",
            # `finish` ends the phase: stop serving, then export the global model.
            f'until [ -f {finish} ]; do kill -0 "$server"; sleep 10; done',
            'kill -TERM "$server"; wait "$server" || true',
            f"megatoken diloco export --config {RUN_CONFIG} --root {COORDINATOR_ROOT} "
            f"--output {cloud.WORK}/{EXPORT}",
        ),
        "outputs": [EXPORT],
    }


def learner_job(spec: dict[str, Any], inputs: list[dict[str, Any]], worker: int) -> dict[str, Any]:
    return {
        "id": f"{spec['id']}-learner-{worker}",
        "image": spec["image"],
        "resources": spec["learners"]["resources"],
        "lifetime_hours": spec["lifetime_hours"],
        "pip": spec.get("pip", []),
        "inputs": inputs,
        "env": {
            "MEGATOKEN_RUN_CONFIG": spec["config"],
            "MEGATOKEN_WORKER_ID": str(worker),
        },
        "command": _shell(
            _TOKEN,
            'export MEGATOKEN_DILOCO_GATEWAY_TOKEN="$(cat "$MEGATOKEN_CW_SECRETS/gateway-token")"',
            _derive(spec["workers"]),
            f"megatoken train --config {RUN_CONFIG} --output {cloud.WORK}/out/learner",
        ),
        "outputs": ["out/learner"],
    }


def _status(url: str, token: str, gateway: str) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{url}/v1/status",
        headers={"Authorization": f"Bearer {token}", "X-Sandbox-Share-Token": gateway},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read())


def _run_path(run_id: str) -> Path:
    return RUNS / f"{run_id}.json"


def _keep_secret(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w") as file:
        file.write(value)


def launch(path: Path, repo: Path, *, ready_timeout: float = 3 * 3600) -> dict[str, Any]:
    spec = validate(json.loads(path.read_text()))
    if _run_path(spec["id"]).exists():
        raise cloud.CloudError(f"Run {spec['id']} exists; finish it or choose a new id")
    inputs = data_inputs(spec["data"])
    token = secrets.token_urlsafe(48)
    _keep_secret(cloud.SECRETS / f"diloco-{spec['id']}.token", token)
    run: dict[str, Any] = {"id": spec["id"], "workers": spec["workers"], "learners": []}
    started: list[str] = []
    with tempfile.TemporaryDirectory() as temporary:
        # Every role runs the same commit.
        archive = Path(temporary) / "code.tar.gz"
        snapshot = (archive, cloud.code_snapshot(repo, archive))
        try:
            entry, _ = cloud.start(
                coordinator_job(spec, inputs),
                repo,
                secrets={"diloco-token": token.encode()},
                files=lambda sandbox: {URL_FILE: cloud.service_url(sandbox).encode()},
                snapshot=snapshot,
            )
            started.append(entry["sandbox_id"])
            run.update(coordinator=entry["sandbox_id"], url=entry["url"], commit=entry["commit"])
            _save(run)
            gateway = cloud.share_token(entry["sandbox_id"])
            # Learners start only once the coordinator answers: init must finish first.
            deadline = time.time() + ready_timeout
            while True:
                try:
                    _status(entry["url"], token, gateway)
                    break
                except OSError as error:
                    if time.time() > deadline:
                        raise cloud.CloudError(
                            f"Coordinator never became ready: {error}"
                        ) from error
                    time.sleep(30)
            for worker in range(spec["workers"]):
                learner, _ = cloud.start(
                    learner_job(spec, inputs, worker),
                    repo,
                    secrets={"diloco-token": token.encode(), "gateway-token": gateway.encode()},
                    files=lambda sandbox: {URL_FILE: entry["url"].encode()},
                    snapshot=snapshot,
                )
                started.append(learner["sandbox_id"])
                run["learners"].append(learner["sandbox_id"])
                _save(run)
        except BaseException:
            # A partial phase must not bill: stop every sandbox this launch started.
            for sandbox_id in started:
                cloud._attach(sandbox_id).stop(missing_ok=True).result()
                cloud.record({"event": "stop", "sandbox_id": sandbox_id, "reason": "launch failed"})
            _run_path(spec["id"]).unlink(missing_ok=True)
            raise
    cloud.record({"event": "diloco_launched", **{k: v for k, v in run.items() if k != "url"}})
    return run


def _save(run: dict[str, Any]) -> None:
    _run_path(run["id"]).parent.mkdir(parents=True, exist_ok=True)
    _run_path(run["id"]).write_text(json.dumps(run, indent=2))


def _exited(sandbox_id: str) -> bool:
    sandbox = cloud._attach(sandbox_id)
    probe = sandbox.exec(["test", "-f", f"{cloud.WORK}/.cloud/outputs-ready"]).result()
    return probe.returncode == 0


def status(run_id: str) -> dict[str, Any]:
    run = json.loads(_run_path(run_id).read_text())
    token = (cloud.SECRETS / f"diloco-{run_id}.token").read_text()
    return {
        "coordinator": _status(run["url"], token, cloud.share_token(run["coordinator"])),
        "learners_finished": [_exited(learner) for learner in run["learners"]],
    }


def finish(run_id: str, destination: Path, *, poll: float = 300) -> dict[str, Any]:
    """Wait for every learner, then stop the coordinator, export and collect everything."""
    run = json.loads(_run_path(run_id).read_text())
    while not all(_exited(learner) for learner in run["learners"]):
        time.sleep(poll)
    learners = [
        cloud.collect(learner, destination / f"learner-{index}")
        for index, learner in enumerate(run["learners"])
    ]
    cloud._attach(run["coordinator"]).write_file(f"{cloud.WORK}/.cloud/finish", b"").result()
    exported = cloud.collect(run["coordinator"], destination / "coordinator")
    for sandbox_id in (run["coordinator"], *run["learners"]):
        cloud._attach(sandbox_id).stop(graceful_shutdown_seconds=60, missing_ok=True).result()
    result = {"run": run_id, "coordinator": exported, "learners": learners}
    cloud.record({"event": "diloco_finished", "run": run_id})
    return result


def derive(config: Path, url: str, workers: int, output: Path) -> None:
    """In the sandbox: the repo config with this run's coordinator URL."""
    from dataclasses import asdict, replace

    from megatoken.training.config import TrainConfig

    loaded = TrainConfig.load(config)
    if loaded.decoupled is None or loaded.decoupled.workers != workers:
        raise SystemExit(f"{config}: decoupled.workers must equal the run's {workers} learners")
    derived = replace(loaded, decoupled=replace(loaded.decoupled, coordinator_url=url))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(asdict(derived), indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m megatoken.cloud_diloco", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    item = commands.add_parser("launch")
    item.add_argument("run", type=Path)
    item.add_argument("--repo", type=Path, default=Path.cwd())
    item = commands.add_parser("finish")
    item.add_argument("run_id")
    item.add_argument("--dest", type=Path, required=True)
    item = commands.add_parser("status")
    item.add_argument("run_id")
    item = commands.add_parser("derive", help="(in the sandbox) write the run config")
    item.add_argument("--config", type=Path, required=True)
    item.add_argument("--url", required=True)
    item.add_argument("--workers", type=int, required=True)
    item.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "derive":
        derive(args.config, args.url, args.workers, args.output)
        return 0
    cloud.load_credentials()
    if args.command == "launch":
        result: Any = launch(args.run, args.repo)
    elif args.command == "finish":
        result = finish(args.run_id, args.dest)
    else:
        result = status(args.run_id)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
