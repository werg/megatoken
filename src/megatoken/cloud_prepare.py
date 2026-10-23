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

Every shard holds the sources of one licence group only (``terms.redistribute`` in the source
manifests: permissive, share_alike, non_commercial, or no), and shards are numbered in that
order. ``reconcile`` replays digests in shard order, so a text two groups share is kept in the
more permissive one.

With a ``"cache": {"repo": "<account>/<name>"}`` block, each finish sandbox also publishes its
roots to that one public Hugging Face dataset repository (``megatoken.publish``): roots of the
three public groups as readable Parquet, roots of group ``no`` encrypted (AES-256-GCM, key from
``megatoken cache keygen``, delivered to finish sandboxes only, as a secret). Each sandbox writes
only its own folders. When every sandbox is done the launcher rebuilds the dataset card, records
a *pin* (the repository, one commit that holds every part, the sha256 of every public file; no
secret) and stops every sandbox: the roots outlive them. ``ensure`` is how a training run gets
its data: if the pin's commit still verifies it says "fetch", otherwise it rebuilds the data.

A run with ``"kind": "media"`` prepares the media sources (``data/media_sharding.py``) the same
way on GPU sandboxes: ingest runs ``scripts/prepare_open_media.py --shard-spec`` (the teacher
and the codec encode every clip, so each ingest sandbox has a GPU), reconcile is unchanged,
and the finish wave runs *inside the ingest sandboxes*: a shard's media assets are far larger
than its episodes (about a megabyte of float32 per image), so moving them to a fresh sandbox
would cost more than the GPU idling while reconcile runs. The launcher gives each ingest
sandbox the address of its shards' verdicts and runs ``finish`` there; the sandbox then serves
``out/prepared`` (its manifest of served files is refreshed to include the finished roots).

  python -m megatoken.cloud_prepare plan RUN.json [--dest DIR]  write and validate every job
  python -m megatoken.cloud_prepare launch RUN.json [--repo .] [--hf-token FILE]
  python -m megatoken.cloud_prepare ensure RUN.json [--pin FILE]  fetch-or-rebuild decision
  python -m megatoken.cloud_prepare status RUN_ID
  python -m megatoken.cloud_prepare stop RUN_ID
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from megatoken import cache, cloud, hub, publish

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
KINDS = ("text", "media")
# A media run's reconcile sandbox reads digests only; it needs no GPU.
RECONCILE_RESOURCES = {"cpu": "16", "memory": "32Gi", "gpu": 0}
FINISH_STATUS = f"{cloud.WORK}/.cloud/finish-status"
FINISH_INPUTS = f"{cloud.WORK}/.cloud/finish-inputs.json"
FINISH_LOG = f"{cloud.WORK}/.cloud/finish.log"
START_THREADS = 8  # sandboxes created at once
CACHE_WORK = f"{cloud.WORK}/scratch/cache"  # converted and sealed files waiting for upload
CACHE_OUT = "out/cache"  # the finish sandbox's receipt (file names and hashes only)
CACHE_RECEIPT = f"{CACHE_OUT}/receipt.json"


def is_media(spec: Mapping[str, Any]) -> bool:
    return spec.get("kind", "text") == "media"


def validate(spec: dict[str, Any]) -> dict[str, Any]:
    if spec.get("schema") != SCHEMA:
        raise cloud.CloudError(f"Run schema must be {SCHEMA}")
    if spec.get("kind", "text") not in KINDS:
        raise cloud.CloudError(f"kind is one of {KINDS}")
    media = is_media(spec)
    for key in (
        "id",
        "image",
        *(("media_sources",) if media else ("sources", "review_config")),
        "listing",
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
    if media:
        sources = spec["media_sources"]
        if not (
            sources == "all" or (isinstance(sources, list) and sources and all(map(str, sources)))
        ):
            raise cloud.CloudError('media_sources is "all" or a list of media source ids')
        if spec["resources"].get("gpu") != 1:
            raise cloud.CloudError("Media preparation encodes on a GPU: resources.gpu must be 1")
    elif spec["resources"].get("gpu"):
        raise cloud.CloudError("Preparation is CPU-only; drop the gpu")
    if "cache" in spec:
        stored = spec["cache"]
        if media:
            raise cloud.CloudError("Media assets are not cached: their sources stream from the Hub")
        if not isinstance(stored, dict) or set(stored) - {
            "repo",
            "contact",
            "piece_bytes",
            "parquet_bytes",
        }:
            raise cloud.CloudError("cache is {repo, contact?, piece_bytes?, parquet_bytes?}")
        if not isinstance(stored.get("repo"), str) or stored["repo"].count("/") != 1:
            raise cloud.CloudError("cache.repo is the Hub dataset repository, <account>/<name>")
        if "contact" in stored and not isinstance(stored["contact"], str):
            raise cloud.CloudError("cache.contact is a sentence saying how to reach the owner")
        for key in ("piece_bytes", "parquet_bytes"):
            if key in stored and (type(stored[key]) is not int or stored[key] < 1):
                raise cloud.CloudError(f"cache.{key} must be a positive integer")
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


def shards_dir(spec: Mapping[str, Any]) -> str:
    """Where ingest leaves each shard: a media shard keeps its assets, so it is built where it
    is finished (the final root); a text shard moves on to a fresh sandbox."""
    return PREPARED if is_media(spec) else SHARDS


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
    resources = spec["resources"]
    if is_media(spec) and wave == "reconcile":
        resources = spec.get("reconcile_resources", RECONCILE_RESOURCES)
    return {
        "schema": cloud.SCHEMA,
        "id": f"{spec['id']}-{name}",
        "image": spec["image"],
        "resources": resources,
        "lifetime_hours": lifetime,
        "pip": spec.get("pip", []),
        "serve_port": PORT,
        "serve_outputs": True,
        # Stay up, serving, until stopped: the next wave fetches from this sandbox.
        "env": {"MEGATOKEN_CW_COLLECT_WAIT": str(int(lifetime * 3600))},
        **fields,
    }


_FFMPEG = (
    "command -v ffmpeg >/dev/null || (apt-get update -qq && "
    "DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ffmpeg >/dev/null)"
)


def ingest_job(spec: Mapping[str, Any], sandbox: int, spec_path: str) -> dict[str, Any]:
    ingest = f"python -m megatoken.cloud_prepare ingest --run {spec_path} --sandbox {sandbox}"
    if is_media(spec):
        # The teacher and the codec packs are fetched into each GPU sandbox.
        lines = (
            _HF_TOKEN,
            _FFMPEG,
            f"mkdir -p {cloud.WORK}/packs {cloud.WORK}/out/logs",
            f"python scripts/acquire_packs.py --root {cloud.WORK}/packs",
            f"megatoken codec --output {CODEC}",
            ingest,
        )
    else:
        lines = (_HF_TOKEN, f"megatoken codec --output {CODEC}", ingest)
    return _job(
        spec,
        f"ingest-{sandbox}",
        "ingest",
        command=_shell(*lines),
        outputs=[shards_dir(spec), "out/logs"],
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
            *_cache_lines(spec, sandbox, spec_path),
        ),
        outputs=[PREPARED, "out/logs", *([CACHE_OUT] if spec.get("cache") else [])],
        **({"pip": [*spec.get("pip", []), *publish.PIP]} if spec.get("cache") else {}),
    )


def _cache_lines(spec: Mapping[str, Any], sandbox: int, spec_path: str) -> list[str]:
    """After finishing: publish this sandbox's roots. The key is read from the sandbox's secret
    file by the program itself, so it is on no command line."""
    if not spec.get("cache"):
        return []
    command = f"python -m megatoken.cloud_prepare publish --run {spec_path} --sandbox {sandbox}"
    return [command, f"rm -rf {CACHE_WORK}"]


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
            f"{shards_dir(spec)}/{shard_name(shard)}/digest.sqlite": (
                f"{DIGESTS}/{shard_name(shard)}.sqlite"
            )
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


def media_sources(spec: Mapping[str, Any]) -> dict[str, Any]:
    """The run's media sources, with the run's option overrides applied (before planning:
    FLEURS divides by its languages)."""
    from megatoken.data.media_sharding import all_sources

    chosen = all_sources(None if spec["media_sources"] == "all" else spec["media_sources"])
    overrides = spec.get("options", {})
    unknown = set(overrides) - set(chosen)
    if unknown:
        raise cloud.CloudError(f"options name sources the run does not build: {sorted(unknown)}")
    return {
        source_id: source.with_options(**overrides.get(source_id, {}))
        for source_id, source in chosen.items()
    }


def media_shard_plan(spec: Mapping[str, Any], root: Path = Path()) -> list[dict[str, Any]]:
    """Every shard's ``{source_id: {files | stripe, options, procedural_every}}`` entries."""
    from megatoken.data.media_sharding import plan_media_shards, read_listing

    sources = media_sources(spec)
    listings = read_listing(sources, root / spec["listing"])
    shards = plan_media_shards(sources, shard_count(spec), listings)
    every = spec.get("procedural_every", {})
    unknown = set(every) - set(sources)
    if unknown:
        raise cloud.CloudError(f"procedural_every names unknown sources: {sorted(unknown)}")
    for shard in shards:
        for source_id, entry in shard.items():
            entry["options"] = {
                **spec.get("options", {}).get(source_id, {}),
                **entry.get("options", {}),
            }
            if source_id in every:
                entry["procedural_every"] = every[source_id]
    empty = [index for index, shard in enumerate(shards) if not shard]
    if empty:
        raise cloud.CloudError(
            f"{len(empty)} of {len(shards)} shards would have nothing to read (first: {empty[0]}): "
            "the sources are too few or too small for this many shards; lower the shard count"
        )
    return shards


def shard_plan(spec: Mapping[str, Any], root: Path = Path()) -> list[dict[str, Any]]:
    """Every shard's source manifest, the same on every machine (paths are relative to ``root``,
    the repository). A shard holds the sources of one licence group, groups in the order of
    ``REDISTRIBUTION_GROUPS``. A shard with nothing to read would fail its sandbox, so none may
    be empty."""
    from megatoken.data.sharding import plan_group_shards, read_listings

    if is_media(spec):
        return media_shard_plan(spec, root)

    manifest = json.loads((root / spec["sources"]).read_text())
    listings = read_listings(manifest, root / spec["listing"])
    shards = plan_group_shards(manifest, shard_count(spec), listings, full=True)
    empty = [index for index, shard in enumerate(shards) if not shard["sources"]]
    if empty:
        raise cloud.CloudError(
            f"{len(empty)} of {len(shards)} shards would have no rows to read (first: {empty[0]}): "
            "the sources are too few or too small for this many shards; lower the shard count"
        )
    return shards


def shard_groups(spec: Mapping[str, Any], root: Path = Path()) -> list[str]:
    """The licence group of every shard, in shard order."""
    from megatoken.data.sharding import shard_group

    return [shard_group(shard) for shard in shard_plan(spec, root)]


def shard_commands(
    spec: Mapping[str, Any], sandbox: int, *, stage: str, work: Path, codec: str
) -> dict[int, list[str]]:
    """The ``megatoken prepare`` command of each shard of one sandbox, after writing its
    source manifest. Stage ``ingest`` reads the sources; ``finish`` applies the verdicts."""
    plan = shard_plan(spec)
    scratch = work / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    commands = {}
    if is_media(spec):
        return media_shard_commands(spec, sandbox, plan, stage=stage, work=work, codec=codec)
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


def media_shard_commands(
    spec: Mapping[str, Any],
    sandbox: int,
    plan: Sequence[Mapping[str, Any]],
    *,
    stage: str,
    work: Path,
    codec: str,
) -> dict[int, list[str]]:
    """The ``prepare_open_media.py`` command of each shard of one sandbox (stage ``ingest``
    reads the sources and stops at the digest; ``finish`` applies the verdicts). A shard whose
    root is already written needs no finishing."""
    scratch = work / "scratch"
    commands = {}
    for shard in shards_of(spec, sandbox):
        root = work / PREPARED / shard_name(shard)
        if stage == "finish" and (root / "manifest.json").exists():
            continue
        shard_spec = scratch / f"media-shard-{shard_name(shard)}.json"
        shard_spec.write_text(json.dumps(plan[shard], indent=1))
        command = [
            sys.executable,
            "scripts/prepare_open_media.py",
            *("--root", str(root), "--codec", codec, "--shard-spec", str(shard_spec)),
        ]
        if stage == "ingest":
            command += [
                *("--packs", f"{work}/packs/lock.json", "--inputs", f"{work}/inputs"),
                *("--work", str(scratch / shard_name(shard)), "--ingest-only"),
            ]
        else:
            command += ["--decisions", str(work / DECISIONS / _decision_name(shard))]
        commands[shard] = command
    return commands


def fetch_verdicts(spec: Mapping[str, Any], sandbox: int, work: Path) -> None:
    """Fetch this sandbox's shards' verdicts from the reconcile sandbox (its address and the
    files' hashes were written here by the launcher), each checked against its sha256."""
    import hashlib
    import urllib.request

    given = json.loads(Path(FINISH_INPUTS.replace(cloud.WORK, str(work), 1)).read_text())
    token = (Path(os.environ["MEGATOKEN_CW_SECRETS"]) / f"share-{given['peer']}").read_text()
    wanted = {
        f"{DECISIONS_OUT}/{_decision_name(shard)}": f"{DECISIONS}/{_decision_name(shard)}"
        for shard in shards_of(spec, sandbox)
    }
    for published, path in wanted.items():
        target = work / path
        target.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(
            f"{given['url']}/{published}", headers={"X-Sandbox-Share-Token": token}
        )
        digest = hashlib.sha256()
        with urllib.request.urlopen(request, timeout=600) as response, target.open("wb") as out:
            for block in iter(lambda: response.read(1 << 22), b""):
                digest.update(block)
                out.write(block)
        if digest.hexdigest() != given["files"][published]:
            target.unlink()
            raise cloud.CloudError(f"{published}: checksum mismatch")


def refresh_outputs(work: Path) -> int:
    """List the finished roots among the sandbox's served files: the bootstrap listed the
    outputs when ingest ended, before the roots existed. Returns the number of files."""
    import hashlib

    manifest = work / ".cloud" / "outputs.json"
    files = json.loads(manifest.read_text())["files"] if manifest.exists() else []
    kept = [item for item in files if not item["path"].startswith((f"{PREPARED}/", "out/logs/"))]
    for name in (PREPARED, "out/logs"):
        for path in sorted(p for p in (work / name).rglob("*") if p.is_file()):
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1 << 22), b""):
                    digest.update(block)
            kept.append(
                {
                    "path": str(path.relative_to(work)),
                    "bytes": path.stat().st_size,
                    "sha256": digest.hexdigest(),
                }
            )
    manifest.write_text(json.dumps({"files": kept}))
    return len(kept)


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


def publish_roots(spec: Mapping[str, Any], sandbox: int, work: Path) -> dict[str, Any]:
    """In a finish sandbox: publish its finished roots to the run's repository and write the
    receipt (``megatoken.publish``). The cache key is read from the sandbox's secret file."""
    stored = spec["cache"]
    groups = shard_groups(spec)
    manifest = json.loads(Path(spec["sources"]).read_text())
    options = {}
    if "piece_bytes" in stored:
        options["piece_bytes"] = stored["piece_bytes"]
    if "parquet_bytes" in stored:
        options["parquet_bytes"] = stored["parquet_bytes"]
    receipt = publish.store_part(
        keys=cache.load_keys(),
        api=hub.api(),
        repo=stored["repo"],
        run=spec["id"],
        part=sandbox,
        prepared=work / PREPARED,
        work=work / CACHE_WORK.removeprefix(f"{cloud.WORK}/"),
        groups={shard_name(shard): groups[shard] for shard in shards_of(spec, sandbox)},
        sources=publish.attribution(manifest["sources"]),
        **options,
    )
    target = work / CACHE_RECEIPT
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(receipt, indent=1))
    return {
        "part": sandbox,
        "public_files": len(receipt["public"]),
        "encrypted": bool(receipt["encrypted"]),
    }


def _work_command(args: argparse.Namespace, stage: str) -> int:
    spec = validate(json.loads(Path(args.run).read_text()))
    work = Path(args.work)
    media_finish = is_media(spec) and stage == "finish"
    if media_finish:
        fetch_verdicts(spec, args.sandbox, work)
    commands = shard_commands(spec, args.sandbox, stage=stage, work=work, codec=args.codec)
    failed = run_shards(commands, spec["processes"], work / "out" / "logs")
    if media_finish and not failed:
        refresh_outputs(work)
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
    # A media run's finish wave is its ingest sandboxes again.
    return list(
        dict.fromkeys(sandbox_id for wave in WAVES for sandbox_id in run["waves"][wave].values())
    )


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


def _finish_in_place(
    run: dict[str, Any], spec: Mapping[str, Any], spec_path: str, poll: float
) -> None:
    """Media's third wave: run ``finish`` inside each ingest sandbox, which still holds its
    shards' assets, once the verdicts exist. Only the verdicts' address and hashes are sent
    here (the sandbox fetches the files from the reconcile sandbox, behind its share token).

    The command runs in the background of the sandbox and is polled through short commands, as
    sandbox calls are cut after a few minutes. A failed sandbox keeps its log, stays up and is
    redone by the next ``launch``; sandboxes already finished are left alone.
    """
    waves = run["waves"]
    reconcile = _published(run, waves["reconcile"]["0"])
    running: dict[str, Any] = {}
    for index in range(spec["sandboxes"]):
        if str(index) in waves["finish"]:
            continue
        sandbox = cloud._attach(waves["ingest"][str(index)])
        wanted = {
            f"{DECISIONS_OUT}/{_decision_name(shard)}": f"{DECISIONS}/{_decision_name(shard)}"
            for shard in shards_of(spec, index)
        }
        files = peer_inputs(reconcile[0], reconcile[1], reconcile[2], wanted)
        given = {
            "peer": reconcile[0],
            "url": reconcile[1],
            "files": {
                published: item["sha256"] for published, item in zip(wanted, files, strict=True)
            },
        }
        secrets = f"{cloud.WORK}/.cloud/secrets"
        sandbox.write_file(
            f"{secrets}/share-{reconcile[0]}", cloud.share_token(reconcile[0]).encode()
        ).result()
        sandbox.write_file(FINISH_INPUTS, json.dumps(given).encode()).result()
        command = (
            f"rm -f {FINISH_STATUS}; cd {cloud.WORK}/megatoken && "
            f"(MEGATOKEN_CW_SECRETS={secrets} python -m megatoken.cloud_prepare finish "
            f"--run {spec_path} --sandbox {index} > {FINISH_LOG} 2>&1; echo $? > {FINISH_STATUS}) "
            "> /dev/null 2>&1 &"
        )
        sandbox.exec(["bash", "-c", command]).result()
        running[str(index)] = sandbox
    failed = []
    while running:
        for key, sandbox in list(running.items()):
            probe = sandbox.exec(["cat", FINISH_STATUS]).result()
            if probe.returncode != 0:
                continue
            del running[key]
            if probe.stdout.strip() == "0":
                waves["finish"][key] = waves["ingest"][key]
            else:
                failed.append(waves["ingest"][key])
        _save(run)
        if running:
            time.sleep(poll)
    if failed:
        _keep_logs(run, failed, FINISH_LOG)
        raise cloud.CloudError(
            f"finish failed in {failed}; logs are under {_run_path(run['id']).with_suffix('.logs')}. "
            "The sandboxes are still running and bill until the run is stopped "
            f"({sys.argv[0]} stop {run['id']}); fix the cause and launch again to redo only them."
        )


def launch(
    path: Path,
    repo: Path,
    *,
    hf_token: Path | None = None,
    cache_key: Path | None = None,
    api: Any = None,
    poll: float = 300,
) -> dict[str, Any]:
    """Run the three waves to the end, or resume a recorded run; returns the run record.

    Without a ``cache`` block the sandboxes serving the finished roots keep running (training
    fetches from them); ``stop`` releases them. Waves 1 and 2 are stopped once every root is
    finished. With one, the finish sandboxes also publish their roots to the run's repository
    (public groups as Parquet, the rest encrypted; the key goes to them alone, as a secret);
    once every upload is verified the dataset card is rebuilt, the pin is saved and every
    sandbox is stopped.
    """
    spec = validate(json.loads(path.read_text()))
    spec_path = str(path.resolve().relative_to(repo.resolve()))
    if not (repo / spec["listing"]).exists():
        raise cloud.CloudError(
            f"{spec['listing']} is missing: build it once with "
            f"{'scripts/shard_media.py --listing' if is_media(spec) else 'scripts/shard_sources.py --listing'}"
            " and commit it, so every sandbox plans the same shards"
        )
    shard_plan(spec, repo)
    token = (hf_token or Path.home() / ".cache/huggingface/token").read_bytes().strip()
    run = (
        _load(spec["id"])
        if _run_path(spec["id"]).exists()
        else {"id": spec["id"], "urls": {}, "waves": {wave: {} for wave in WAVES}}
    )
    if run.get("finished"):
        return run
    secrets = {"hf-token": token}
    # Only the finish wave seals data: no other sandbox ever holds the cache key.
    finish_secrets = dict(secrets)
    keys = None
    if spec.get("cache"):
        keys = cache.load_keys(cache_key)
        finish_secrets[cache.SECRET_NAME] = cache.read_key_file(cache_key).hex().encode()
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
        if is_media(spec):
            _finish_in_place(run, spec, spec_path, poll)
        else:
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
            _wave(run, "finish", finish, repo, finish_secrets, snapshot, poll)
        if spec.get("cache"):
            run["cache_pin"] = str(_save_pin(spec, waves["finish"], keys, api or hub.api()))
    # A media run's ingest sandboxes hold (and serve) the finished roots: only reconcile goes.
    # A cached run's roots are safe in the Hub repos, so its finish sandboxes go too.
    cloud.stop_all(
        [
            *([] if is_media(spec) else waves["ingest"].values()),
            *waves["reconcile"].values(),
            *(waves["finish"].values() if spec.get("cache") else []),
        ],
        "every root is finished",
    )
    run["finished"] = True
    _save(run)
    cloud.record({"event": "prepare_finished", "run": spec["id"]})
    return run


def cache_pin_path(run_id: str) -> Path:
    """Where the launcher keeps a run's pin (copy it into the repo to share it)."""
    return RUNS / f"{run_id}.cache-pin.json"


def _save_pin(
    spec: Mapping[str, Any], finish: Mapping[str, str], keys: cache.Keys | None, api: Any
) -> Path:
    """Gather each finish sandbox's receipt, rebuild the dataset card once, and write the run's
    pin (one commit of the repository that holds every part, checked against the receipts).
    Only now is the data durable: a missing receipt means that sandbox's upload did not
    complete."""
    receipts = []
    for key in sorted(finish, key=int):
        raw = cloud._attach(finish[key]).read_file(f"{cloud.WORK}/{CACHE_RECEIPT}").result()
        receipts.append(json.loads(raw))
    with tempfile.TemporaryDirectory() as scratch:
        pin = publish.finalize(
            api=api,
            keys=keys,
            repo=spec["cache"]["repo"],
            run=spec["id"],
            receipts=receipts,
            contact=spec["cache"].get("contact", publish.DEFAULT_CONTACT),
            scratch=Path(scratch),
        )
    target = cache_pin_path(spec["id"])
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(pin, indent=1))
    return target


def ensure(
    path: Path,
    repo: Path,
    *,
    pin: Path | None = None,
    hf_token: Path | None = None,
    cache_key: Path | None = None,
    api: Any = None,
    poll: float = 300,
) -> dict[str, Any]:
    """Have the run's prepared data stored and verified, rebuilding it if it is not.

    The rule: the pin (``pin``, else the one the last build recorded) is checked against the
    Hub without downloading the data; if the commit, every public file and every encrypted
    manifest and piece verify the answer is ``stored`` and training fetches it
    (``megatoken.publish fetch``). If there is no pin, the key no longer opens the manifests, or
    anything is missing or altered, the data is prepared again from the pinned sources
    (``launch``), which publishes and pins it afresh.
    """
    spec = validate(json.loads(path.read_text()))
    if not spec.get("cache"):
        raise cloud.CloudError(f"{path} has no cache block: nothing is stored to fetch or rebuild")
    keys = cache.load_keys(cache_key)
    pin_path = Path(pin) if pin else cache_pin_path(spec["id"])
    with tempfile.TemporaryDirectory() as scratch:
        decision = publish.decide(pin_path, keys, api or hub.api(), Path(scratch) / "verify")
    if decision.mode == "stored":
        return {"mode": "stored", "pin": str(pin_path), "reason": decision.reason}
    record = _run_path(spec["id"])
    if record.exists():  # the old build is spent: start a fresh one rather than resume it
        record.rename(record.with_name(f"{record.stem}.superseded-{int(time.time())}.json"))
    run = launch(path, repo, hf_token=hf_token, cache_key=cache_key, api=api, poll=poll)
    return {"mode": "rebuilt", "pin": run["cache_pin"], "reason": decision.reason}


def _keep_logs(
    run: Mapping[str, Any], sandbox_ids: Sequence[str], log: str = f"{cloud.WORK}/.cloud/job.log"
) -> None:
    """Save the tail of each sandbox's log before it stops: logs vanish with the sandbox."""
    logs = _run_path(run["id"]).with_suffix(".logs")
    logs.mkdir(parents=True, exist_ok=True)
    for sandbox_id in sandbox_ids:
        try:
            tail = cloud._attach(sandbox_id).exec(["tail", "-c", "200000", log]).result().stdout
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
        {"path": f"{shards_dir(spec)}/{shard_name(s)}/{name}.sqlite", "sha256": sha}
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
    for i in range(0 if is_media(spec) else spec["sandboxes"]):
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
    for name in ("ingest", "finish", "publish"):  # run inside a sandbox
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
    item.add_argument("--cache-key", type=Path, help=f"default {cache.KEY_FILE}")
    item = commands.add_parser("ensure", help="fetch-or-rebuild: have the run's data stored")
    item.add_argument("run", type=Path)
    item.add_argument("--repo", type=Path, default=Path.cwd())
    item.add_argument("--pin", type=Path, help="default: the pin the last build recorded")
    item.add_argument("--hf-token", type=Path, help="default ~/.cache/huggingface/token")
    item.add_argument("--cache-key", type=Path, help=f"default {cache.KEY_FILE}")
    for name in ("status", "stop"):
        commands.add_parser(name).add_argument("run_id")
    args = parser.parse_args(argv)
    if args.command == "publish":
        spec = validate(json.loads(Path(args.run).read_text()))
        print(json.dumps(publish_roots(spec, args.sandbox, Path(args.work))))
        return 0
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
        result = launch(args.run, args.repo, hf_token=args.hf_token, cache_key=args.cache_key)
    elif args.command == "ensure":
        result = ensure(
            args.run, args.repo, pin=args.pin, hf_token=args.hf_token, cache_key=args.cache_key
        )
    elif args.command == "status":
        result = status(args.run_id)
    else:
        result = {"not_stopped": stop(args.run_id)}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
