"""Run one DiLoCo phase on CoreWeave sandboxes: a CPU coordinator and N GPU learners.

A run file (``megatoken.cloud-diloco/1``, JSON) names the training config (a repo
path whose ``decoupled`` block describes the run), the worker count, the data
and both roles' resources:

    {"schema": "megatoken.cloud-diloco/1", "id": "p1", "image": "...",
     "lifetime_hours": 24, "pip": ["pyarrow"], "workers": 4,
     "config": "configs/train/pretraining.example.yaml",
     "data": {"peer": "<sandbox id>", "prefixes": ["out/datasets/train"]},
     "coordinator": {"resources": {"cpu": "8", "memory": "32Gi"}, "args": ["--quorum", "3"]},
     "learners": {"resources": {"cpu": "8", "memory": "32Gi", "gpu": 1}}}

Data is built once in the cloud: ``data.peer`` is a finished ``megatoken.cloud``
job with ``serve_outputs``, whose outputs every sandbox fetches over HTTPS at
datacenter speed (``data.inputs`` may instead list ordinary job inputs).
``data.cache`` names a ``megatoken.cloud-prepare/1`` run file with a ``cache`` block
(``{"cache": {"run": "configs/cloud/prepare-full-agentic.json"}}``): the launcher first runs
``cloud_prepare ensure``, which checks the published copy on the Hub (the public Parquet
subsets and the encrypted folder) and, if it is missing or does not verify, rebuilds the data
from the pinned sources. Each learner then downloads and verifies the pinned commit, rebuilds
the prepared roots and decrypts the encrypted part into ``/work/data/prepared`` before training
(the training config's ``datasets`` may say ``/work/data/prepared/*``); the cache key and Hub
token reach learners as secrets only. The
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
import signal
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any

from megatoken import cache, cloud, cloud_prepare, publish

SCHEMA = "megatoken.cloud-diloco/1"
PORT = 8080
COORDINATOR_ROOT = f"{cloud.WORK}/coordinator"
EXPORT = "out/export"
RUN_CONFIG = f"{cloud.WORK}/run/train.json"
URL_FILE = "run/coordinator-url"  # an input: the URL exists only once the coordinator does
PIN_FILE = "in/cache-pin.json"  # an input: the pin of the stored data
CORPUS_FILE = "in/corpus-metadata.json"  # small identity bootstrap for the coordinator
DATA_DIR = f"{cloud.WORK}/data/prepared"  # where a learner restores the stored roots
HINDSIGHT_DIR = "data/hindsight"
RUNS = cloud.STATE / "diloco"
# The coordinator outlives its learners, so it can still export when the last one ends.
COORDINATOR_EXTRA_HOURS = 2


def validate(spec: dict[str, Any]) -> dict[str, Any]:
    if spec.get("schema") != SCHEMA:
        raise cloud.CloudError(f"Run schema must be {SCHEMA}")
    for key in ("id", "image", "lifetime_hours", "workers", "config", "data"):
        if key not in spec:
            raise cloud.CloudError(f"Run lacks {key!r}")
    if type(spec["workers"]) is not int or not 1 <= spec["workers"] <= 64:
        raise cloud.CloudError("workers must be in [1, 64]")
    if not 0 < float(spec["lifetime_hours"]) <= 720 - COORDINATOR_EXTRA_HOURS:
        raise cloud.CloudError(
            f"lifetime_hours must be in (0, {720 - COORDINATOR_EXTRA_HOURS}]: "
            f"the coordinator lives {COORDINATOR_EXTRA_HOURS} h longer than the learners"
        )
    if set(spec["data"]) not in ({"peer", "prefixes"}, {"inputs"}, {"cache"}):
        raise cloud.CloudError("data is {peer, prefixes}, {inputs} or {cache}")
    if "assets" in spec and (
        not isinstance(spec["assets"], dict)
        or set(spec["assets"]) != {"inputs"}
        or not isinstance(spec["assets"]["inputs"], list)
    ):
        raise cloud.CloudError("assets is {inputs: ordinary verified sandbox input files}")
    if "hindsight" in spec and (
        not isinstance(spec["hindsight"], dict)
        or set(spec["hindsight"]) != {"peer", "prefix"}
        or not all(isinstance(v, str) and v for v in spec["hindsight"].values())
        or Path(spec["hindsight"]["prefix"]).is_absolute()
        or ".." in Path(spec["hindsight"]["prefix"]).parts
    ):
        raise cloud.CloudError("hindsight is {peer, prefix: a relative completed artifact path}")
    if "cache" in spec["data"]:
        stored = spec["data"]["cache"]
        if (
            not isinstance(stored, dict)
            or not isinstance(stored.get("run"), str)
            or set(stored) - {"run", "pin"}
        ):
            raise cloud.CloudError("data.cache is {run: a cloud-prepare run file, pin?: a file}")
    if not spec.get("learners", {}).get("resources", {}).get("gpu"):
        raise cloud.CloudError("Learners train on one GPU each")
    return spec


def data_inputs(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Every pinned data file, as ordinary job inputs (a peer's are fetched from it)."""
    if "inputs" in data:
        return list(data["inputs"])
    if "cache" in data:
        return []  # learners restore the stored data themselves, from the Hub
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


def hindsight_inputs(spec):
    """Deliver one prepared artifact to every learner, without regenerating per worker."""
    if "hindsight" not in spec:
        return []
    source = spec["hindsight"]
    prefix = source["prefix"].rstrip("/") + "/"
    chosen = data_inputs({"peer": source["peer"], "prefixes": [source["prefix"]]})
    return [{**item, "path": HINDSIGHT_DIR + "/" + item["path"][len(prefix) :]} for item in chosen]


def validate_memory_delivery(config, inputs):
    paths = {item["path"] for item in inputs}
    if config.hindsight_targets:
        try:
            expected = str(Path(config.hindsight_targets).relative_to(cloud.WORK))
        except ValueError as error:
            raise cloud.CloudError("Cloud teacher targets must be located under /work") from error
        required = {f"{expected}/manifest.json", f"{expected}/targets.sqlite"}
        if not required <= paths:
            raise cloud.CloudError(
                "Teacher-backed training requires the prepared manifest and index "
                "as verified inputs; supply hindsight: {peer, prefix} or assets.inputs"
            )
    required_assets = [config.packs, config.text_anchors]
    for root in [config.engram_import] if config.engram_import else config.engram_imports or []:
        required_assets.extend(
            [str(Path(root) / "engram.json"), str(Path(root) / "descriptors.npz")]
        )
    if config.codec != "byte":
        codec = Path(config.codec)
        required_assets.append(str(codec if codec.suffix else codec / "codec.json"))
    for path in required_assets:
        if path and Path(path).is_relative_to(cloud.WORK):
            relative = str(Path(path).relative_to(cloud.WORK))
            if relative not in paths:
                raise cloud.CloudError(
                    f"Required training asset is absent from verified launch inputs: {path}"
                )


def _coordinator_inputs(inputs):
    # The CPU coordinator needs the identity, not every teacher vector or its lookup index.
    prefix = HINDSIGHT_DIR + "/"
    return [
        item
        for item in inputs
        if not item["path"].startswith(prefix) or item["path"] == prefix + "manifest.json"
    ]


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
    if "cache" in spec["data"]:
        args = f"{args} --corpus-metadata {cloud.WORK}/{CORPUS_FILE}".strip()
    finish = f"{cloud.WORK}/.cloud/finish"
    return {
        "id": f"{spec['id']}-coordinator",
        "image": spec["image"],
        "resources": spec.get("coordinator", {}).get("resources", {"cpu": "8", "memory": "32Gi"}),
        "lifetime_hours": float(spec["lifetime_hours"]) + COORDINATOR_EXTRA_HOURS,
        "pip": spec.get("pip", []),
        "inputs": _coordinator_inputs(inputs),
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


def _restore(spec: dict[str, Any]) -> list[str]:
    """A learner's first step when the data is stored: fetch and verify the pinned commit,
    rebuild the public roots and decrypt the rest."""
    if "cache" not in spec["data"]:
        return []
    return [
        'export HF_TOKEN="$(cat "$MEGATOKEN_CW_SECRETS/hf-token")"',
        f"python -m megatoken.publish fetch --pin {cloud.WORK}/{PIN_FILE} --dest {DATA_DIR}",
    ]


def learner_job(spec: dict[str, Any], inputs: list[dict[str, Any]], worker: int) -> dict[str, Any]:
    return {
        "id": f"{spec['id']}-learner-{worker}",
        "image": spec["image"],
        "resources": spec["learners"]["resources"],
        "lifetime_hours": spec["lifetime_hours"],
        "pip": [*spec.get("pip", []), *(publish.PIP if "cache" in spec["data"] else [])],
        "inputs": inputs,
        "env": {
            "MEGATOKEN_RUN_CONFIG": spec["config"],
            "MEGATOKEN_WORKER_ID": str(worker),
        },
        "command": _shell(
            _TOKEN,
            'export MEGATOKEN_DILOCO_GATEWAY_TOKEN="$(cat "$MEGATOKEN_CW_SECRETS/gateway-token")"',
            *_restore(spec),
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


def launch(
    path: Path,
    repo: Path,
    *,
    ready_timeout: float = 3 * 3600,
    hf_token: Path | None = None,
    cache_key: Path | None = None,
) -> dict[str, Any]:
    spec = validate(json.loads(path.read_text()))
    if _run_path(spec["id"]).exists():
        raise cloud.CloudError(f"Run {spec['id']} exists; finish it or choose a new id")
    from megatoken.training.config import TrainConfig

    config = TrainConfig.load(repo / spec["config"], expand_data=False)
    if config.purpose == "production" and config.decoupled.workers != spec["workers"]:
        raise cloud.CloudError("Cloud workers must match the stage's decoupled worker count")
    unresolved = [
        *config.datasets,
        config.codec,
        config.packs,
        config.text_anchors,
        config.hindsight_targets,
        config.engram_import,
        *(config.engram_imports or []),
    ]
    if config.decoupled is not None:
        unresolved.append(config.decoupled.run_id)
    if any(isinstance(path, str) and "PIN_" in path for path in unresolved):
        raise cloud.CloudError("Training config still contains unresolved PIN_* artifact paths")
    # Later stages start from the preceding export; reject omissions before billable work.
    resume_path = None
    if config.purpose == "production":
        from megatoken.data.recipes import stage_recipe

        arguments = spec.get("coordinator", {}).get("args", [])
        if "--resume" in arguments:
            index = arguments.index("--resume")
            if index + 1 >= len(arguments):
                raise cloud.CloudError(
                    "Coordinator --resume requires the preceding checkpoint path"
                )
            resume_path = arguments[index + 1]
        if stage_recipe(config.phase).previous_phase is not None and resume_path is None:
            raise cloud.CloudError(
                "Production stage requires the preceding checkpoint via coordinator --resume"
            )
    inputs = [
        *data_inputs(spec["data"]),
        *spec.get("assets", {}).get("inputs", []),
        *hindsight_inputs(spec),
    ]
    if len({item["path"] for item in inputs}) != len(inputs):
        raise cloud.CloudError("Launch inputs contain duplicate destination paths")
    if resume_path is not None:
        try:
            relative = str(Path(resume_path).relative_to(cloud.WORK))
        except ValueError as error:
            raise cloud.CloudError(
                "The preceding checkpoint must be delivered under /work"
            ) from error
        if relative not in {item["path"] for item in inputs}:
            raise cloud.CloudError("Preceding checkpoint is absent from verified launch inputs")
    validate_memory_delivery(config, inputs)
    learner_secrets: dict[str, bytes] = {}
    pin = b""
    corpus_metadata = b""
    if "cache" in spec["data"]:
        stored = spec["data"]["cache"]
        # Data stored and verified, or rebuilt first (which can take hours, before any GPU bills).
        found = cloud_prepare.ensure(
            repo / stored["run"],
            repo,
            pin=Path(stored["pin"]).expanduser() if "pin" in stored else None,
            hf_token=hf_token,
            cache_key=cache_key,
        )
        print(json.dumps({"data": found}), file=sys.stderr)
        pin = Path(found["pin"]).read_bytes()
        corpus_metadata = json.dumps(
            publish.prepared_corpus_metadata(json.loads(pin)), sort_keys=True
        ).encode()
        learner_secrets = {
            "hf-token": (hf_token or Path.home() / ".cache/huggingface/token").read_bytes().strip(),
            cache.SECRET_NAME: cache.read_key_file(cache_key).hex().encode(),
        }
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
                files=lambda sandbox: {
                    URL_FILE: cloud.service_url(sandbox).encode(),
                    **({CORPUS_FILE: corpus_metadata} if corpus_metadata else {}),
                },
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
                    if cloud.outputs_ready(entry["sandbox_id"]):
                        raise cloud.CloudError(
                            f"Coordinator exited before it became ready (see its log): {error}"
                        ) from error
                    if time.time() > deadline:
                        raise cloud.CloudError(
                            f"Coordinator never became ready: {error}"
                        ) from error
                    time.sleep(30)
            for worker in range(spec["workers"]):
                learner, _ = cloud.start(
                    learner_job(spec, inputs, worker),
                    repo,
                    secrets={
                        "diloco-token": token.encode(),
                        "gateway-token": gateway.encode(),
                        **learner_secrets,
                    },
                    files=lambda sandbox: {
                        URL_FILE: entry["url"].encode(),
                        **({PIN_FILE: pin} if pin else {}),
                    },
                    snapshot=snapshot,
                )
                started.append(learner["sandbox_id"])
                run["learners"].append(learner["sandbox_id"])
                _save(run)
        except BaseException:
            # A partial phase must not bill: stop every sandbox this launch started.
            cloud.stop_all(started, "launch failed")
            _run_path(spec["id"]).unlink(missing_ok=True)
            raise
    cloud.record({"event": "diloco_launched", **{k: v for k, v in run.items() if k != "url"}})
    return run


def _save(run: dict[str, Any]) -> None:
    _run_path(run["id"]).parent.mkdir(parents=True, exist_ok=True)
    _run_path(run["id"]).write_text(json.dumps(run, indent=2))


def status(run_id: str) -> dict[str, Any]:
    run = json.loads(_run_path(run_id).read_text())
    token = (cloud.SECRETS / f"diloco-{run_id}.token").read_text()
    return {
        "coordinator": _status(run["url"], token, cloud.share_token(run["coordinator"])),
        "learners_finished": [cloud.outputs_ready(learner) for learner in run["learners"]],
    }


def finish(run_id: str, destination: Path, *, poll: float = 300) -> dict[str, Any]:
    """Wait for every learner, then stop the coordinator, export and collect everything."""
    run = json.loads(_run_path(run_id).read_text())
    while not all(cloud.outputs_ready(learner) for learner in run["learners"]):
        time.sleep(poll)
    learners = [
        cloud.collect(learner, destination / f"learner-{index}")
        for index, learner in enumerate(run["learners"])
    ]
    failed = [entry["sandbox_id"] for entry in learners if entry["status"] != 0]
    if failed:
        # No export of a phase whose learners failed; their logs are collected.
        cloud.stop_all([run["coordinator"], *run["learners"]], "learners failed", grace=60)
        cloud.record({"event": "diloco_failed", "run": run_id, "learners": failed})
        raise cloud.CloudError(f"Learners failed (logs under {destination}): {failed}")
    cloud._attach(run["coordinator"]).write_file(f"{cloud.WORK}/.cloud/finish", b"").result()
    exported = cloud.collect(run["coordinator"], destination / "coordinator")
    unstopped = cloud.stop_all([run["coordinator"], *run["learners"]], "finished", grace=60)
    if exported["status"] != 0:
        cloud.record({"event": "diloco_failed", "run": run_id, "export_status": exported["status"]})
        raise cloud.CloudError(
            f"The coordinator's export failed (status {exported['status']}); "
            f"the global model is missing, logs under {destination / 'coordinator'}"
        )
    cloud.record({"event": "diloco_finished", "run": run_id})
    if unstopped:
        raise cloud.CloudError(f"Collected, but these sandboxes would not stop: {unstopped}")
    return {"run": run_id, "coordinator": exported, "learners": learners}


def derive(config: Path, url: str, workers: int, output: Path) -> None:
    """In the sandbox: the repo config with this run's coordinator URL."""
    from dataclasses import asdict, replace

    from megatoken.training.config import TrainConfig

    # The cached-data coordinator intentionally has no corpus directories to glob.
    loaded = TrainConfig.load(config, expand_data=False)
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
    item.add_argument("--hf-token", type=Path, help="default ~/.cache/huggingface/token")
    item.add_argument("--cache-key", type=Path, help=f"default {cache.KEY_FILE}")
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
    # A terminated launcher must still stop the sandboxes it started (launch's
    # BaseException handler), not leave them billing.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    if args.command == "launch":
        result: Any = launch(args.run, args.repo, hf_token=args.hf_token, cache_key=args.cache_key)
    elif args.command == "finish":
        result = finish(args.run_id, args.dest)
    else:
        result = status(args.run_id)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
