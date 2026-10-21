"""Run memory-heavy megatoken jobs in CoreWeave serverless GPU sandboxes.

A job file (``megatoken.cloud-job/1``, JSON) declares the image, resources,
lifetime, inputs (public URLs or local files, each pinned by sha256), the
command, and the output directories. ``launch`` starts the sandbox with
``cloud_bootstrap.sh`` as its main process, then streams a git snapshot of a
*clean, committed* working tree plus the local inputs into it. ``collect`` pulls
the outputs once the job finishes and verifies each file's sha256.

  python -m megatoken.cloud launch JOB.json [--repo .]   start + push; prints the sandbox id
  python -m megatoken.cloud collect SANDBOX_ID --dest DIR pull and verify outputs, then release
  python -m megatoken.cloud logs SANDBOX_ID [--follow]
  python -m megatoken.cloud stop SANDBOX_ID
  python -m megatoken.cloud records                     sandboxes launched from this machine

Credentials come from the environment or ``MEGATOKEN_CW_ENV_FILE`` (mode 600):
``CWSANDBOX_AUTH=wandb`` with ``WANDB_API_KEY``, or ``CWSANDBOX_API_KEY``.
They are never printed and never passed into a sandbox. Only the standard
library and ``cwsandbox`` are imported, so this runs from a slim client venv.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

SCHEMA = "megatoken.cloud-job/1"
WORK = "/work"
BOOTSTRAP = "/opt/megatoken/bootstrap.sh"
ENV_FILE = Path(
    os.environ.get("MEGATOKEN_CW_ENV_FILE", Path.home() / ".config/megatoken/coreweave.env")
)
STATE = Path(os.environ.get("MEGATOKEN_CW_STATE", Path.home() / ".local/state/megatoken/cloud"))
CHUNK = 8 << 20
# Gateway streams are cut after ~25 s, so every transfer moves bounded parts.
PART_BYTES = 32 << 20
PUSH_IN_FLIGHT = 8
CREATE_ATTEMPTS = 6


class CloudError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def load_credentials(path: Path = ENV_FILE) -> None:
    """KEY=VALUE lines into the environment (existing variables win); refuses a shared file."""
    if not path.is_file():
        return
    if path.stat().st_mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise CloudError(f"{path} is readable by others; chmod 600 it")
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            name, value = line.removeprefix("export ").split("=", 1)
            os.environ.setdefault(name.strip(), value.strip().strip("\"'"))


def _auth():
    from cwsandbox import AuthStrategy

    strategy = os.environ.get("CWSANDBOX_AUTH") or (
        "coreweave_api_key" if os.environ.get("CWSANDBOX_API_KEY") else "wandb"
    )
    needed = "WANDB_API_KEY" if strategy == "wandb" else "CWSANDBOX_API_KEY"
    if not os.environ.get(needed):
        raise CloudError(f"No {needed} for auth {strategy!r}; set it or MEGATOKEN_CW_ENV_FILE")
    return AuthStrategy(strategy)


def validate(job: dict[str, Any]) -> dict[str, Any]:
    if job.get("schema") != SCHEMA:
        raise CloudError(f"Job schema must be {SCHEMA}")
    for key in ("id", "image", "resources", "lifetime_hours", "command", "outputs"):
        if key not in job:
            raise CloudError(f"Job lacks {key!r}")
    if not 0 < float(job["lifetime_hours"]) <= 720:
        raise CloudError("lifetime_hours must be in (0, 720]")
    if job["resources"].get("gpu", 0) > 1:
        raise CloudError("Serverless sandboxes have exactly one GPU")
    for item in job.get("inputs", []):
        if (
            not item.get("path")
            or Path(item["path"]).is_absolute()
            or ".." in Path(item["path"]).parts
        ):
            raise CloudError(f"Input path must be relative to {WORK}: {item}")
        if len(item.get("sha256", "")) != 64 or bool(item.get("url")) == bool(item.get("source")):
            raise CloudError(f"Input needs sha256 and exactly one of url/source: {item['path']}")
    for item in job.get("push_dirs", []):
        if Path(item.get("path", "")).is_absolute() or ".." in Path(item.get("path", "")).parts:
            raise CloudError(f"push_dirs path must be relative to {WORK}: {item}")
        if not item.get("source"):
            raise CloudError(f"push_dirs entry lacks a local source: {item}")
    for output in job["outputs"]:
        if Path(output).is_absolute() or ".." in Path(output).parts:
            raise CloudError(f"Output must be relative to {WORK}: {output}")
    return job


def request(job: dict[str, Any]) -> dict[str, Any]:
    """Keyword arguments for ``cwsandbox.Sandbox.run``."""
    resources: dict[str, Any] = {
        "cpu": str(job["resources"]["cpu"]),
        "memory": str(job["resources"]["memory"]),
    }
    if job["resources"].get("gpu"):
        resources["gpu"] = {"count": 1}
    bootstrap = (Path(__file__).with_name("cloud_bootstrap.sh")).read_text()
    request = {
        "container_image": job["image"],
        "resources": resources,
        "max_lifetime_seconds": int(float(job["lifetime_hours"]) * 3600),
        "tags": ["megatoken", f"job-{job['id']}"],
        "placement_mode": "serverless",
        # Mapping form {path: content} is the one cwsandbox 1.19 accepts unambiguously.
        "mounted_files": {BOOTSTRAP: bootstrap},
        "environment_variables": {
            "MEGATOKEN_CW_JOB": job["id"],
            "MEGATOKEN_CW_WORK": WORK,
            "MEGATOKEN_CW_OUTPUTS": " ".join(job["outputs"]),
            "MEGATOKEN_CW_PIP": " ".join(job.get("pip", [])),
            **job.get("env", {}),
        },
        # Under W&B auth the direct data plane fails ("project not found"); the gateway works.
        "data_plane_mode": "gateway",
    }
    if job.get("scratch"):
        # Optional: measured 2026-10-21, GPU requests with a scratch volume sat in "creating"
        # until reconciled away, while identical requests without one placed. The root
        # filesystem of a GPU sandbox is node-local NVMe with terabytes free.
        request["volumes"] = [{"name": "work", "mount_path": WORK, "size": job["scratch"]}]
    return request


def record(entry: dict[str, Any]) -> None:
    STATE.mkdir(parents=True, exist_ok=True)
    stamped = {"at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **entry}
    with (STATE / "sandboxes.jsonl").open("a") as file:
        file.write(json.dumps(stamped, sort_keys=True) + "\n")


def code_snapshot(repo: Path, destination: Path) -> dict[str, str]:
    """``git archive`` of HEAD; refuses uncommitted changes so every run names a commit."""
    dirty = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if dirty:
        raise CloudError("Commit the working tree first; cloud runs execute an exact commit")
    commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    subprocess.run(
        ["git", "-C", str(repo), "archive", "--format=tar.gz", "-o", str(destination), commit],
        check=True,
    )
    return {"commit": commit, "sha256": sha256_file(destination)}


def _create(job: dict[str, Any]):
    from cwsandbox import Sandbox

    keywords = request(job)
    for attempt in range(CREATE_ATTEMPTS):
        try:
            return Sandbox.run("/bin/bash", BOOTSTRAP, *job["command"], auth=_auth(), **keywords)
        except Exception as error:  # capacity and throttling answers are retried with backoff
            text = f"{type(error).__name__}: {error}"
            retryable = "ResourceExhausted" in text or "RESOURCE_EXHAUSTED" in text or "429" in text
            record(
                {
                    "event": "create_refused",
                    "job": job["id"],
                    "attempt": attempt,
                    "error": text[:400],
                }
            )
            if not retryable or attempt == CREATE_ATTEMPTS - 1:
                raise
            time.sleep(min(600, 30 * 2**attempt))
    raise AssertionError("unreachable")


def _push(
    sandbox, files: dict[str, Path], *, in_flight: int = PUSH_IN_FLIGHT, attempts: int = 4
) -> None:
    """Upload each ``{digest: source}`` as ``inbox/<digest>.NNNN`` parts (the bootstrap
    joins and verifies them), keeping ``in_flight`` part writes outstanding at once:
    one gateway stream is latency-bound, so throughput scales with concurrency."""
    from cwsandbox.exceptions import SandboxError

    parts = [
        (digest, source, index)
        for digest, source in files.items()
        for index in range(max(1, -(-source.stat().st_size // PART_BYTES)))
    ]

    def read(source: Path, index: int) -> bytes:
        with source.open("rb") as stream:
            stream.seek(index * PART_BYTES)
            return stream.read(PART_BYTES)

    def write(digest: str, source: Path, index: int):
        return sandbox.write_file(f"{WORK}/.cloud/inbox/{digest}.{index:04d}", read(source, index))

    pending: dict[tuple[str, int], tuple[Any, Path, int]] = {}
    queue = list(reversed(parts))
    while queue or pending:
        while queue and len(pending) < in_flight:
            digest, source, index = queue.pop()
            pending[(digest, index)] = (write(digest, source, index), source, 1)
        (digest, index), (future, source, attempt) = next(iter(pending.items()))
        try:
            future.result()
            del pending[(digest, index)]
        except SandboxError as error:
            if attempt == attempts:
                raise CloudError(f"{source}: part {index}: {error}") from error
            time.sleep(10 * attempt)
            pending[(digest, index)] = (write(digest, source, index), source, attempt + 1)


def expand_push_dirs(job: dict[str, Any]) -> list[dict[str, str]]:
    """Every file under each ``push_dirs`` source, as a sha256-pinned pushed input."""
    items = []
    for entry in job.get("push_dirs", []):
        root = Path(entry["source"]).expanduser()
        if not root.is_dir():
            raise CloudError(f"push_dirs source is not a directory: {root}")
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            items.append(
                {
                    "path": str(Path(entry["path"]) / path.relative_to(root)),
                    "source": str(path),
                    "sha256": sha256_file(path),
                }
            )
    return items


def launch(job_path: Path, repo: Path, *, start_timeout: float = 5400) -> dict[str, Any]:
    job = validate(json.loads(job_path.read_text()))
    job["inputs"] = [*job.get("inputs", []), *expand_push_dirs(job)]
    with tempfile.TemporaryDirectory() as temporary:
        archive = Path(temporary) / "code.tar.gz"
        code = code_snapshot(repo, archive)
        pushed = []
        for item in job["inputs"]:
            if item.get("source"):
                source = Path(item["source"]).expanduser()
                if sha256_file(source) != item["sha256"]:
                    raise CloudError(f"{source} does not match its declared sha256")
                pushed.append((source, item["sha256"]))
        sandbox = _create(job)
        entry = {
            "event": "launched",
            "sandbox_id": sandbox.sandbox_id,
            "job": job["id"],
            **code,
            "inputs_sha256": hashlib.sha256(
                json.dumps(job["inputs"], sort_keys=True).encode()
            ).hexdigest(),
        }
        record(entry)
        # Waiting in "creating" is not billed, but a sandbox the controller gave up
        # on must never start later and idle billed while it waits for inputs.
        try:
            started = time.time()
            sandbox.wait(timeout=start_timeout)
            record(
                {
                    "event": "running",
                    "sandbox_id": sandbox.sandbox_id,
                    "seconds": time.time() - started,
                }
            )
            _push(sandbox, {code["sha256"]: archive, **{d: s for s, d in pushed}})
            inputs = {
                "code": code,
                "inputs": [
                    {key: item[key] for key in ("path", "sha256", "url") if key in item}
                    for item in job.get("inputs", [])
                ],
            }
            sandbox.write_file(f"{WORK}/.cloud/inputs.json", json.dumps(inputs).encode()).result()
            sandbox.write_file(f"{WORK}/.cloud/inputs-ready", b"").result()
        except BaseException as error:
            sandbox.stop(graceful_shutdown_seconds=10, missing_ok=True).result()
            record(
                {
                    "event": "stop",
                    "sandbox_id": sandbox.sandbox_id,
                    "reason": f"launch failed: {type(error).__name__}: {error}"[:400],
                }
            )
            raise
        record(
            {"event": "inputs_pushed", "sandbox_id": sandbox.sandbox_id, "files": len(pushed) + 1}
        )
    return entry


def _attach(sandbox_id: str):
    from cwsandbox import Sandbox

    return Sandbox.from_id(sandbox_id, auth=_auth(), data_plane_mode="gateway").result()


def _stream(sandbox, remote: str, size: int, out, *, attempts: int = 4) -> None:
    """Append one remote file of ``size`` bytes to ``out``, retrying dropped streams."""
    from cwsandbox.exceptions import SandboxError

    start = out.tell()
    for attempt in range(1, attempts + 1):
        out.seek(start)
        out.truncate()
        try:
            with contextlib.closing(
                sandbox.read_file_streaming(remote, timeout_seconds=7200)
            ) as reader:
                for block in reader:
                    out.write(block)
        except SandboxError as error:
            if attempt == attempts:
                raise CloudError(f"{remote}: {error}") from error
        else:
            if out.tell() - start == size:
                return
            if attempt == attempts:
                raise CloudError(f"{remote}: pulled {out.tell() - start} of {size} bytes")
        time.sleep(10 * attempt)


def _pull(sandbox, item: dict[str, Any], target: Path) -> None:
    """Pull one output file into ``target``, verified against the sandbox manifest.

    Single gateway streams drop on multi-GB files, so larger outputs are split
    into PART_BYTES parts in the sandbox and each part is streamed (and retried)
    on its own.
    """
    remote, size = f"{WORK}/{item['path']}", item["bytes"]
    parts = [(remote, size)]
    if size > PART_BYTES:
        prefix = f"{WORK}/.cloud/parts/{item['sha256']}."
        # Detached: the gateway also ends long exec streams, which would kill split.
        sandbox.exec(
            [
                "bash",
                "-c",
                'mkdir -p "${2%/*}" && rm -f "$2"* && '
                '(setsid bash -c \'split -b "$3" -d -a 4 "$1" "$2" && touch "$2done"\' '
                '_ "$1" "$2" "$3" >/dev/null 2>&1 &)',
                "split",
                remote,
                prefix,
                str(PART_BYTES),
            ]
        ).result()
        deadline = time.time() + 1800
        while sandbox.exec(["test", "-f", f"{prefix}done"]).result().returncode != 0:
            if time.time() > deadline:
                raise CloudError(f"{item['path']}: splitting into parts did not finish")
            time.sleep(5)
        parts = [
            (f"{prefix}{index:04d}", min(PART_BYTES, size - offset))
            for index, offset in enumerate(range(0, size, PART_BYTES))
        ]
    staging = target.with_name(f".{target.name}.partial")
    with staging.open("wb") as out:
        for part, part_size in parts:
            _stream(sandbox, part, part_size, out)
    if size > PART_BYTES:
        sandbox.exec(["rm", "-f", f"{prefix}done", *(part for part, _ in parts)]).result()
    if sha256_file(staging) != item["sha256"]:
        staging.unlink()
        raise CloudError(f"{item['path']}: pulled bytes do not match the sandbox manifest")
    staging.replace(target)


def collect(sandbox_id: str, destination: Path, *, timeout: float = 30 * 86400) -> dict[str, Any]:
    """Wait for outputs-ready, pull and verify every output file, then release the sandbox."""
    sandbox = _attach(sandbox_id)
    base = f"{WORK}/.cloud"
    deadline = time.time() + timeout
    while sandbox.exec(["test", "-f", f"{base}/outputs-ready"]).result().returncode != 0:
        if time.time() > deadline:
            raise CloudError(f"{sandbox_id}: outputs not ready within {timeout} s")
        time.sleep(30)
    status = sandbox.exec(["cat", f"{base}/exit-status"]).result().stdout.strip()
    probe = sandbox.exec(["test", "-f", f"{base}/outputs.json"]).result()
    manifest = (
        json.loads(sandbox.read_file(f"{base}/outputs.json").result())
        if probe.returncode == 0
        else {"files": []}
    )
    pulled = 0
    for item in manifest["files"]:
        target = destination / item["path"]
        if target.is_file() and sha256_file(target) == item["sha256"]:
            continue  # already pulled by an earlier, interrupted collect
        target.parent.mkdir(parents=True, exist_ok=True)
        _pull(sandbox, item, target)
        pulled += 1
    sandbox.write_file(f"{base}/outputs-collected", b"").result()
    entry = {
        "event": "collected",
        "sandbox_id": sandbox_id,
        "status": int(status) if status.lstrip("-").isdigit() else None,
        "files": pulled,
        "bytes": sum(item["bytes"] for item in manifest["files"]),
    }
    record(entry)
    return entry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m megatoken.cloud", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    item = commands.add_parser("launch")
    item.add_argument("job", type=Path)
    item.add_argument("--repo", type=Path, default=Path.cwd())
    item.add_argument(
        "--start-timeout-hours",
        type=float,
        default=1.5,
        help="How long to wait for GPU capacity before stopping the sandbox",
    )
    item = commands.add_parser("collect")
    item.add_argument("sandbox_id")
    item.add_argument("--dest", type=Path, required=True)
    item = commands.add_parser("logs")
    item.add_argument("sandbox_id")
    item.add_argument("--follow", action="store_true")
    item.add_argument("--tail", type=int)
    item = commands.add_parser("stop")
    item.add_argument("sandbox_id")
    item.add_argument("--grace-seconds", type=float, default=600)
    commands.add_parser("records")
    args = parser.parse_args(argv)
    load_credentials()
    if args.command == "launch":
        result: Any = launch(args.job, args.repo, start_timeout=args.start_timeout_hours * 3600)
    elif args.command == "collect":
        result = collect(args.sandbox_id, args.dest)
    elif args.command == "logs":
        for line in _attach(args.sandbox_id).stream_logs(follow=args.follow, tail_lines=args.tail):
            sys.stdout.write(line if line.endswith("\n") else line + "\n")
        return 0
    elif args.command == "stop":
        _attach(args.sandbox_id).stop(
            graceful_shutdown_seconds=args.grace_seconds, missing_ok=True
        ).result()
        result = {"stopped": args.sandbox_id}
        record({"event": "stop", "sandbox_id": args.sandbox_id})
    else:
        path = STATE / "sandboxes.jsonl"
        result = (
            [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
