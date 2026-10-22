"""Prepare every row of the data sources on many CPU sandboxes, deduplicated across all of them.

A run file (``megatoken.cloud-prepare/1``, JSON) names a source manifest, the file listing
that fixes how it is split, and how many sandboxes and shards to use:

    {"schema": "megatoken.cloud-prepare/1", "id": "agentic-full", "image": "...",
     "sources": "configs/data/sources.agentic.json",
     "listing": "configs/data/files.agentic.json",
     "review_config": "configs/data/parser-review.agentic.json",
     "pip": ["datasets>=3,<5"], "resources": {"cpu": "16", "memory": "32Gi"},
     "sandboxes": 24, "shards_per_sandbox": 15, "processes": 15,
     "hours": {"ingest": 24, "reconcile": 8, "finish": 4}}

The manifest is split into ``sandboxes * shards_per_sandbox`` balanced shards without any
coordinator (``megatoken.data.sharding``: every machine computes the same plan from the
committed listing and keeps its own part) and prepared in three waves of sandboxes:

1. ``ingest``: each sandbox reads its shards (``processes`` at a time), removes duplicates
   inside each shard and leaves, per shard, a digest of what it kept.
2. ``reconcile``: one sandbox fetches only the digests (about 2 KB per episode, no text),
   finds duplicates, families and benchmark overlaps across shards, and freezes the splits
   once over everything (``megatoken reconcile``).
3. ``finish``: each group of shards is pulled, with its verdicts, into a fresh sandbox that
   writes the final prepared roots. Nothing reads the sources twice, and those sandboxes
   serve ``out/prepared`` to training like any job with ``serve_outputs``.

Every wave is an ordinary ``megatoken.cloud`` job (share-token HTTPS between sandboxes, each
file pinned by sha256). ``launch`` records its progress, so after a failure it redoes only
the failed sandboxes: the earlier waves stay up, holding the data the redo needs, until
every root is finished (or ``stop`` is run). Sandboxes of waves 1 and 2 wait idle for the
next wave, so the reconcile and finish hours in the run file are billed for all of them.

  python -m megatoken.cloud_prepare plan RUN.json [--dest DIR]  write and validate every job
  python -m megatoken.cloud_prepare launch RUN.json [--repo .] [--hf-token FILE]
  python -m megatoken.cloud_prepare status RUN_ID
  python -m megatoken.cloud_prepare stop RUN_ID
"""

from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from megatoken import cloud

SCHEMA = "megatoken.cloud-prepare/1"
PORT = 8080
CODEC = f"{cloud.WORK}/codec"
SHARDS = "out/shards"  # wave 1 output: <SHARDS>/<shard>/{digest,preparation}.sqlite
PREPARED = "out/prepared"  # wave 3 output: the finished roots
DIGESTS = "in/digests"
DECISIONS = "in/decisions"
DECISIONS_OUT = "out/decisions"
RUNS = cloud.STATE / "prepare"
# Slack on top of the waves that follow, for sandbox placement and transfers.
PULL_HOURS = 2
ATTEMPTS = 2  # a shard resumes from its durable cursor, so one retry costs little
MAX_SHARDS = 99999  # shard numbers are zero-padded to five digits so that they sort
WAVES = ("ingest", "reconcile", "finish")
START_THREADS = 8  # sandboxes created at once


def validate(spec: dict[str, Any]) -> dict[str, Any]:
    if spec.get("schema") != SCHEMA:
        raise cloud.CloudError(f"Run schema must be {SCHEMA}")
    for key in (
        "id",
        "image",
        "sources",
        "listing",
        "review_config",
        "resources",
        "sandboxes",
        "shards_per_sandbox",
        "processes",
        "hours",
    ):
        if key not in spec:
            raise cloud.CloudError(f"Run lacks {key!r}")
    for key in ("sandboxes", "shards_per_sandbox", "processes"):
        if type(spec[key]) is not int or spec[key] < 1:
            raise cloud.CloudError(f"{key} must be a positive integer")
    if spec["sandboxes"] * spec["shards_per_sandbox"] > MAX_SHARDS:
        raise cloud.CloudError(f"At most {MAX_SHARDS} shards")
    if spec["resources"].get("gpu"):
        raise cloud.CloudError("Preparation is CPU-only; drop the gpu")
    if set(spec["hours"]) != {"ingest", "reconcile", "finish"}:
        raise cloud.CloudError("hours is {ingest, reconcile, finish}")
    if not all(0 < float(hours) for hours in spec["hours"].values()):
        raise cloud.CloudError("Every wave needs a positive number of hours")
    if lifetimes(spec)["ingest"] > 720:
        raise cloud.CloudError("The ingest sandboxes would live longer than 720 hours")
    return spec


def lifetimes(spec: Mapping[str, Any]) -> dict[str, float]:
    """Each wave's sandbox lifetime: its own work, then until the last roots are finished
    (a failed finish is redone from the earlier waves' data)."""
    hours = spec["hours"]
    return {
        "ingest": hours["ingest"] + hours["reconcile"] + hours["finish"] + PULL_HOURS,
        "reconcile": hours["reconcile"] + hours["finish"] + PULL_HOURS,
        "finish": hours["finish"],
    }


def shard_count(spec: Mapping[str, Any]) -> int:
    return spec["sandboxes"] * spec["shards_per_sandbox"]


def shards_of(spec: Mapping[str, Any], sandbox: int) -> range:
    each = spec["shards_per_sandbox"]
    return range(sandbox * each, (sandbox + 1) * each)


def shard_name(shard: int) -> str:
    return f"{shard:05d}"


def _decision_name(shard: int) -> str:
    return f"decisions-{shard:04d}.sqlite"  # reconcile's own naming


# ---- jobs ---------------------------------------------------------------------------------


def _shell(*lines: str) -> list[str]:
    return ["bash", "-c", "set -eo pipefail; " + "; ".join(lines)]


_HF_TOKEN = 'export HF_TOKEN="$(cat "$MEGATOKEN_CW_SECRETS/hf-token")"'


def _job(spec: Mapping[str, Any], name: str, wave: str, **fields: Any) -> dict[str, Any]:
    lifetime = lifetimes(spec)[wave]
    return {
        "schema": cloud.SCHEMA,
        "id": f"{spec['id']}-{name}",
        "image": spec["image"],
        "resources": spec["resources"],
        "lifetime_hours": lifetime,
        "pip": spec.get("pip", []),
        "serve_port": PORT,
        "serve_outputs": True,
        # Stay up, serving, until stopped: the next wave fetches from this sandbox.
        "env": {"MEGATOKEN_CW_COLLECT_WAIT": str(int(lifetime * 3600))},
        **fields,
    }


def ingest_job(spec: Mapping[str, Any], sandbox: int, spec_path: str) -> dict[str, Any]:
    return _job(
        spec,
        f"ingest-{sandbox}",
        "ingest",
        command=_shell(
            _HF_TOKEN,
            f"megatoken codec --output {CODEC}",
            f"python -m megatoken.cloud_prepare ingest --run {spec_path} --sandbox {sandbox}",
        ),
        outputs=[SHARDS, "out/logs"],
    )


def reconcile_job(spec: Mapping[str, Any], inputs: list[dict[str, Any]]) -> dict[str, Any]:
    return _job(
        spec,
        "reconcile",
        "reconcile",
        inputs=inputs,
        # The digests are named by shard number, so the glob lists them in shard order.
        command=_shell(
            f"megatoken reconcile --digests {cloud.WORK}/{DIGESTS}/*.sqlite "
            f"--output {cloud.WORK}/{DECISIONS_OUT}"
        ),
        outputs=[DECISIONS_OUT],
    )


def finish_job(
    spec: Mapping[str, Any], sandbox: int, inputs: list[dict[str, Any]], spec_path: str
) -> dict[str, Any]:
    return _job(
        spec,
        f"finish-{sandbox}",
        "finish",
        inputs=inputs,
        command=_shell(
            _HF_TOKEN,
            f"megatoken codec --output {CODEC}",
            f"python -m megatoken.cloud_prepare finish --run {spec_path} --sandbox {sandbox}",
        ),
        outputs=[PREPARED, "out/logs"],
    )


def peer_inputs(
    sandbox_id: str, url: str, files: Sequence[Mapping[str, Any]], wanted: Mapping[str, str]
) -> list[dict[str, Any]]:
    """Pinned inputs from a finished sandbox's outputs: ``wanted`` maps its published
    path to where this sandbox should put the file."""
    by_path = {item["path"]: item for item in files}
    missing = [published for published in wanted if published not in by_path]
    if missing:
        raise cloud.CloudError(f"{sandbox_id} did not publish {missing[:3]} (of {len(missing)})")
    return [
        {
            "path": path,
            "sha256": by_path[published]["sha256"],
            "url": f"{url}/{published}",
            "peer": sandbox_id,
        }
        for published, path in wanted.items()
    ]


def digest_inputs(
    spec: Mapping[str, Any], sandbox: int, sandbox_id: str, url: str, files: Sequence[Any]
) -> list[dict[str, Any]]:
    return peer_inputs(
        sandbox_id,
        url,
        files,
        {
            f"{SHARDS}/{shard_name(shard)}/digest.sqlite": f"{DIGESTS}/{shard_name(shard)}.sqlite"
            for shard in shards_of(spec, sandbox)
        },
    )


def finish_inputs(
    spec: Mapping[str, Any],
    sandbox: int,
    ingest: tuple[str, str, Sequence[Any]],
    reconcile: tuple[str, str, Sequence[Any]],
) -> list[dict[str, Any]]:
    """A group's shard databases (from its ingest sandbox) and its verdicts (from reconcile)."""
    shards = shards_of(spec, sandbox)
    return [
        *peer_inputs(
            ingest[0],
            ingest[1],
            ingest[2],
            {
                f"{SHARDS}/{shard_name(s)}/preparation.sqlite": (
                    f"{PREPARED}/{shard_name(s)}/preparation.sqlite"
                )
                for s in shards
            },
        ),
        *peer_inputs(
            reconcile[0],
            reconcile[1],
            reconcile[2],
            {
                f"{DECISIONS_OUT}/{_decision_name(s)}": f"{DECISIONS}/{_decision_name(s)}"
                for s in shards
            },
        ),
    ]


# ---- inside a sandbox ---------------------------------------------------------------------


def shard_plan(spec: Mapping[str, Any], root: Path = Path()) -> list[dict[str, Any]]:
    """Every shard's source manifest, the same on every machine (paths are relative to ``root``,
    the repository). A shard with nothing to read would fail its sandbox, so none may be empty."""
    from megatoken.data.sharding import plan_shards, read_listings

    manifest = json.loads((root / spec["sources"]).read_text())
    listings = read_listings(manifest, root / spec["listing"])
    shards = plan_shards(manifest, shard_count(spec), listings, full=True)
    empty = [index for index, shard in enumerate(shards) if not shard["sources"]]
    if empty:
        raise cloud.CloudError(
            f"{len(empty)} of {len(shards)} shards would have no rows to read (first: {empty[0]}): "
            "the sources are too few or too small for this many shards; lower the shard count"
        )
    return shards


def shard_commands(
    spec: Mapping[str, Any], sandbox: int, *, stage: str, work: Path, codec: str
) -> dict[int, list[str]]:
    """The ``megatoken prepare`` command of each shard of one sandbox, after writing its
    source manifest. Stage ``ingest`` reads the sources; ``finish`` applies the verdicts."""
    plan = shard_plan(spec)
    scratch = work / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    commands = {}
    for shard in shards_of(spec, sandbox):
        sources = scratch / f"sources-{shard_name(shard)}.json"
        sources.write_text(json.dumps(plan[shard], indent=1))
        root = work / (SHARDS if stage == "ingest" else PREPARED) / shard_name(shard)
        command = [
            sys.executable,
            "-m",
            "megatoken",
            "prepare",
            "--sources",
            str(sources),
            "--output",
            str(root),
            "--codec",
            codec,
            "--parser-review-config",
            spec["review_config"],
            *spec.get("prepare_args", ["--reject-invalid"]),
        ]
        command += (
            ["--ingest-only"]
            if stage == "ingest"
            else ["--decisions", str(work / DECISIONS / _decision_name(shard))]
        )
        commands[shard] = command
    return commands


def run_shards(commands: Mapping[int, list[str]], processes: int, logs: Path) -> list[int]:
    """Run each shard's command, ``processes`` at a time; returns the shards that failed.

    A shard that dies is run once more: it resumes from its durable cursor, so a lost
    connection costs the rows since the last commit, not the shard.
    """
    logs.mkdir(parents=True, exist_ok=True)

    def one(item: tuple[int, list[str]]) -> tuple[int, bool]:
        shard, command = item
        for _ in range(ATTEMPTS):
            with (logs / f"{shard_name(shard)}.log").open("ab") as log:
                if subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode == 0:
                    return shard, True
        return shard, False

    with ThreadPoolExecutor(processes) as pool:
        results = list(pool.map(one, commands.items()))
    failed = [shard for shard, ok in results if not ok]
    for shard in failed:
        print(f"shard {shard} failed; see out/logs/{shard_name(shard)}.log", file=sys.stderr)
    return failed


def _work_command(args: argparse.Namespace, stage: str) -> int:
    spec = validate(json.loads(Path(args.run).read_text()))
    work = Path(args.work)
    commands = shard_commands(spec, args.sandbox, stage=stage, work=work, codec=args.codec)
    failed = run_shards(commands, spec["processes"], work / "out" / "logs")
    print(json.dumps({"stage": stage, "shards": len(commands), "failed": failed}))
    return 1 if failed else 0


# ---- the launcher -------------------------------------------------------------------------


def _run_path(run_id: str) -> Path:
    return RUNS / f"{run_id}.json"


def _save(run: dict[str, Any]) -> None:
    _run_path(run["id"]).parent.mkdir(parents=True, exist_ok=True)
    _run_path(run["id"]).write_text(json.dumps(run, indent=2))


def _load(run_id: str) -> dict[str, Any]:
    return json.loads(_run_path(run_id).read_text())


def _everything(run: Mapping[str, Any]) -> list[str]:
    return [sandbox_id for wave in WAVES for sandbox_id in run["waves"][wave].values()]


def _published(run: Mapping[str, Any], sandbox_id: str) -> tuple[str, str, list[Any]]:
    """A finished sandbox's id, URL and the files it serves, with their sha256."""
    sandbox = cloud._attach(sandbox_id)
    manifest = json.loads(sandbox.read_file(f"{cloud.WORK}/.cloud/outputs.json").result())
    return sandbox_id, run["urls"][sandbox_id], manifest["files"]


def _start_all(
    jobs: Mapping[str, dict[str, Any]],
    repo: Path,
    secrets: dict[str, bytes],
    snapshot: tuple[Path, dict[str, str]],
) -> dict[str, dict[str, Any]]:
    """Start every job (several at once: waiting for placement dominates); entries by key."""
    started: dict[str, dict[str, Any]] = {}

    def one(item: tuple[str, dict[str, Any]]) -> None:
        entry, _ = cloud.start(item[1], repo, secrets=secrets, snapshot=snapshot)
        started[item[0]] = entry

    try:
        with ThreadPoolExecutor(min(START_THREADS, max(1, len(jobs)))) as pool:
            list(pool.map(one, jobs.items()))
    except BaseException:
        # A partly started wave must not bill: stop what this call started.
        cloud.stop_all([entry["sandbox_id"] for entry in started.values()], "wave failed")
        raise
    return started


def _wave(
    run: dict[str, Any],
    wave: str,
    jobs: Mapping[str, dict[str, Any]],
    repo: Path,
    secrets: dict[str, bytes],
    snapshot: tuple[Path, dict[str, str]],
    poll: float,
) -> None:
    """Start the wave's sandboxes that are not running yet and wait for all of them.

    A sandbox whose job failed keeps its log under the run's state directory, is stopped
    and forgotten, so running ``launch`` again redoes just that one; the waves before it
    keep their sandboxes, which still hold the data the redo needs.
    """
    members = run["waves"][wave]
    missing = {key: job for key, job in jobs.items() if key not in members}
    for key, entry in _start_all(missing, repo, secrets, snapshot).items():
        members[key] = entry["sandbox_id"]
        run["urls"][entry["sandbox_id"]] = entry["url"]
    _save(run)
    pending = list(members.values())
    while pending:
        pending = [sandbox_id for sandbox_id in pending if not cloud.outputs_ready(sandbox_id)]
        if pending:
            time.sleep(poll)
    failed = [sandbox_id for sandbox_id in members.values() if cloud.exit_status(sandbox_id) != 0]
    if failed:
        _keep_logs(run, failed)
        cloud.stop_all(failed, f"{wave} failed")
        for key in [key for key, sandbox_id in members.items() if sandbox_id in failed]:
            del members[key]
        _save(run)
        raise cloud.CloudError(
            f"{wave} failed in {failed}; logs are under {_run_path(run['id']).with_suffix('.logs')}. "
            f"The earlier waves are still running and bill until the run is stopped "
            f"({sys.argv[0]} stop {run['id']}) or their lifetime ends; fix the cause and launch "
            "again to redo only the failed sandboxes."
        )


def launch(
    path: Path, repo: Path, *, hf_token: Path | None = None, poll: float = 300
) -> dict[str, Any]:
    """Run the three waves to the end, or resume a recorded run; returns the run record.

    The sandboxes serving the finished roots keep running (training fetches from them);
    ``stop`` releases them. Waves 1 and 2 are stopped once every root is finished.
    """
    spec = validate(json.loads(path.read_text()))
    spec_path = str(path.resolve().relative_to(repo.resolve()))
    if not (repo / spec["listing"]).exists():
        raise cloud.CloudError(
            f"{spec['listing']} is missing: build it once with scripts/shard_sources.py --listing "
            "and commit it, so every sandbox plans the same shards"
        )
    shard_plan(spec, repo)
    token = (hf_token or Path.home() / ".cache/huggingface/token").read_bytes().strip()
    run = (
        _load(spec["id"])
        if _run_path(spec["id"]).exists()
        else {"id": spec["id"], "urls": {}, "waves": {wave: {} for wave in WAVES}}
    )
    secrets = {"hf-token": token}
    with tempfile.TemporaryDirectory() as temporary:
        archive = Path(temporary) / "code.tar.gz"
        snapshot = (archive, cloud.code_snapshot(repo, archive))
        if run.setdefault("commit", snapshot[1]["commit"]) != snapshot[1]["commit"]:
            raise cloud.CloudError(f"{spec['id']} started at {run['commit']}: resume from there")
        waves = run["waves"]
        _wave(
            run,
            "ingest",
            {str(i): ingest_job(spec, i, spec_path) for i in range(spec["sandboxes"])},
            repo,
            secrets,
            snapshot,
            poll,
        )
        if not waves["reconcile"]:
            inputs = []
            for index in range(spec["sandboxes"]):
                sandbox_id = waves["ingest"][str(index)]
                inputs += digest_inputs(spec, index, *_published(run, sandbox_id))
            reconcile = {"0": reconcile_job(spec, inputs)}
        else:
            reconcile = {}
        _wave(run, "reconcile", reconcile, repo, secrets, snapshot, poll)
        finish = {}
        for index in range(spec["sandboxes"]):
            if str(index) not in waves["finish"]:
                finish[str(index)] = finish_job(
                    spec,
                    index,
                    finish_inputs(
                        spec,
                        index,
                        _published(run, waves["ingest"][str(index)]),
                        _published(run, waves["reconcile"]["0"]),
                    ),
                    spec_path,
                )
        _wave(run, "finish", finish, repo, secrets, snapshot, poll)
    cloud.stop_all(
        [*waves["ingest"].values(), *waves["reconcile"].values()], "every root is finished"
    )
    cloud.record({"event": "prepare_finished", "run": spec["id"]})
    return run


def _keep_logs(run: Mapping[str, Any], sandbox_ids: Sequence[str]) -> None:
    """Save the tail of each sandbox's log before it stops: logs vanish with the sandbox."""
    logs = _run_path(run["id"]).with_suffix(".logs")
    logs.mkdir(parents=True, exist_ok=True)
    for sandbox_id in sandbox_ids:
        try:
            tail = (
                cloud._attach(sandbox_id)
                .exec(["tail", "-c", "200000", f"{cloud.WORK}/.cloud/job.log"])
                .result()
                .stdout
            )
            (logs / f"{sandbox_id}.log").write_text(tail)
        except Exception as error:
            print(
                f"No log for {sandbox_id}: {type(error).__name__}: {error}"[:300], file=sys.stderr
            )


def status(run_id: str) -> dict[str, Any]:
    run = _load(run_id)
    return {
        wave: {
            key: cloud.outputs_ready(sandbox_id) for key, sandbox_id in run["waves"][wave].items()
        }
        for wave in WAVES
    }


def stop(run_id: str) -> list[str]:
    """Stop every sandbox of a run, also the ones serving the finished roots; returns the
    sandboxes that would not stop."""
    return cloud.stop_all(_everything(_load(run_id)), "stopped by user", grace=60)


def write_jobs(spec: dict[str, Any], spec_path: str, destination: Path) -> list[Path]:
    """Write every job of the run (with placeholder inputs for the later waves) and validate
    each against ``megatoken.cloud``; the way to check a run file before it costs anything."""
    validate(spec)
    sha = "0" * 64
    ingest_files = [
        {"path": f"{SHARDS}/{shard_name(s)}/{name}.sqlite", "sha256": sha}
        for s in range(shard_count(spec))
        for name in ("digest", "preparation")
    ]
    verdicts = [
        {"path": f"{DECISIONS_OUT}/{_decision_name(s)}", "sha256": sha}
        for s in range(shard_count(spec))
    ]
    jobs = {f"ingest-{i}": ingest_job(spec, i, spec_path) for i in range(spec["sandboxes"])}
    jobs["reconcile"] = reconcile_job(
        spec,
        [
            item
            for i in range(spec["sandboxes"])
            for item in digest_inputs(
                spec, i, f"ingest-{i}", f"https://ingest-{i}.invalid", ingest_files
            )
        ],
    )
    for i in range(spec["sandboxes"]):
        jobs[f"finish-{i}"] = finish_job(
            spec,
            i,
            finish_inputs(
                spec,
                i,
                (f"ingest-{i}", f"https://ingest-{i}.invalid", ingest_files),
                ("reconcile", "https://reconcile.invalid", verdicts),
            ),
            spec_path,
        )
    destination.mkdir(parents=True, exist_ok=True)
    written = []
    for name, job in jobs.items():
        cloud.validate(job)
        cloud.request(job)
        written.append(destination / f"{name}.json")
        written[-1].write_text(json.dumps(job, indent=1))
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m megatoken.cloud_prepare", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("ingest", "finish"):  # run inside a sandbox
        item = commands.add_parser(name)
        item.add_argument("--run", required=True, help="the run file, relative to the repository")
        item.add_argument("--sandbox", type=int, required=True)
        item.add_argument("--work", default=cloud.WORK)
        item.add_argument("--codec", default=CODEC)
    item = commands.add_parser("plan")
    item.add_argument("run", type=Path)
    item.add_argument("--dest", type=Path, default=Path("prepare-jobs"))
    item = commands.add_parser("launch")
    item.add_argument("run", type=Path)
    item.add_argument("--repo", type=Path, default=Path.cwd())
    item.add_argument("--hf-token", type=Path, help="default ~/.cache/huggingface/token")
    for name in ("status", "stop"):
        commands.add_parser(name).add_argument("run_id")
    args = parser.parse_args(argv)
    if args.command in ("ingest", "finish"):
        return _work_command(args, args.command)
    cloud.load_credentials()
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    if args.command == "plan":
        shard_plan(json.loads(args.run.read_text()))
        result: Any = [
            str(path)
            for path in write_jobs(json.loads(args.run.read_text()), str(args.run), args.dest)
        ]
    elif args.command == "launch":
        result = launch(args.run, args.repo, hf_token=args.hf_token)
    elif args.command == "status":
        result = status(args.run_id)
    else:
        result = {"not_stopped": stop(args.run_id)}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
