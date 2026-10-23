#!/usr/bin/env python3
"""Dress rehearsal: run the whole megatoken pipeline once, at toy scale, with everything on.

THE QUESTION THIS ANSWERS
    "If I spend real cloud credits on a training run, will the chain
    prepare data -> train -> resume -> export -> serve -> evaluate actually hold together, and is the
    launch configuration itself complete?"  This probe finds out in a minute or two on one machine,
    with the tiny model.  It checks plumbing, not model quality: a tiny model trained for a few steps
    learns nothing useful, so do not read the loss values as quality.

WHAT IT DOES (each step is a "stage" with pass/fail, wall time and the first error with its
traceback; the run keeps going after a failure whenever the later stages do not depend on it)
    launch_readiness   Static audit, no training.  Reads every configs/train/*.yaml (plus any file
                       given with --audit-config) and the stage recipes in
                       src/megatoken/data/stages.json and lists (a) every data stream a stage needs
                       that the config's listed data roots cannot supply, each with the script or
                       cloud job that would produce it (or "no producer found"), (b) config keys the
                       training code does not read, (c) PIN_ placeholders still unfilled.
    build_data         Builds small synthetic prepared data for every stream of the P1 recipe, from
                       the repository's test fixtures: text and chat, tool use, executed
                       trajectories, dynamic (lifecycle) execution, full-duplex execution traces,
                       reading from a collection, and tiny media (audio, image and video with fake
                       teacher-feature tensors).  Each stream is labelled "real" (made by the
                       repository's own constructor) or "stand_in" (same machinery, relabelled
                       fixture content); about 90 percent of the P1 mix is stand-in.
    train_b0           Bootstrap stage (B0) for a few steps, a checkpoint at every step.
    resume_b0          One resume from the checkpoint: checks the step number, the token counters and
                       the logged steps carry on without a gap or repeat.
    resume_exactness   Compares an uninterrupted run with the stopped-and-resumed one; their weights,
                       optimizer state, random state, data position and counters must be identical.
    train_p1_text      Base-pretraining stage (P1) continued from B0 (a "phase transition") on the
                       text data.  This checkpoint is the one that gets exported.
    train_p1_mixture   P1 with the real stage mixture over ALL data roots (every stream of the
                       recipe, media included; tiny fake stand-ins replace the Pocket, Sana and
                       Sana-Video donor models) with held-out evaluation on.
    train_p1_mixture_no_eval
                       Only runs if the previous stage failed: the same run with held-out evaluation
                       off, so the rest of the mixture path is still covered.
    resume_p1_mixture  One resume of the mixture run.
    export_bundle      `megatoken export` of the P1 text checkpoint into a served bundle.
    serve_client       Starts `megatoken serve` on it and drives it over JSON lines: open a session
                       (with a mounted collection), observe text, generate, status, create a track and
                       run lifecycle actions, speculate/accept/cancel, asynchronous generation with
                       poll/deliver/release, a raw observe, protocol errors, a malformed line, close.
    serve_add_files    Adds a file to a mounted collection (in this process with a stand-in file
                       preparer, because the real one needs the large media model packs) and reads it.
    serve_media_bundle Exports the stage-mixture checkpoint together with its (fake) packs and
                       generates audio, an image and text from one request, after observing an image.
    evaluate           `megatoken evaluate` on the held-out episodes.
    serve_soak         Runs scripts/probes/probe_serve_soak.py on the bundle (150 long-session rounds,
                       with and without release calls).

HOW TO RUN
    Smoke (about 40 seconds on a quiet 20-core CPU; used by the test):
        /home/werg/megatoken-runs/env/dev/bin/python scripts/probes/probe_rehearsal.py \\
            --smoke --output /tmp/rehearsal-smoke.json
    Realistic (more steps, training through the real `megatoken train` command, every stream gets a
    held-out check, the stage mixture draws every stream in every step, soak of 150; ~2-3 minutes):
        /home/werg/megatoken-runs/env/dev/bin/python scripts/probes/probe_rehearsal.py \\
            --output /tmp/rehearsal.json
    Only the static launch audit, optionally of your own filled-in config (about a second):
        /home/werg/megatoken-runs/env/dev/bin/python scripts/probes/probe_rehearsal.py \\
            --readiness-only --audit-config my-launch.yaml --output /tmp/readiness.json
    On the GPU (waits for and holds /tmp/megatoken-gpu.lock for the whole run):
        ... --device cuda --model 32x256 --b0-steps 4 --p1-steps 4
    Training uses one CPU thread per process (OMP_NUM_THREADS=1): with several threads on a busy
    machine one tiny step took a minute instead of two seconds.

    The process exits non-zero when any stage failed.  Skipped stages (an earlier stage they depend on
    failed) are listed but are not failures.  Use --keep-workdir --workdir DIR to keep the data,
    checkpoints, logs and the exported bundle for inspection.

WHAT THE REPORTED NUMBERS MEAN
    stages[].status          pass / fail / skip.  Healthy: all "pass".
    stages[].seconds         wall time of the stage; a stage far slower than its siblings deserves a look.
    train_*.loss_first/last  the training objective terms at the first and last logged step (cross
                             entropy of the bytes, facet prediction, router balance, ...).  Healthy:
                             all finite.  They need not fall in a few steps on a toy model.
    train_*.counters         work done, in the stage's own units: D = new tokens read, I = tokens
                             ingested, P = tokens scored by the decoder, G = physical records, Q =
                             decisions scored.  Healthy: nonzero and growing across a resume.
    train_*.steps_contiguous true when the logged steps run one after another (no gap or repeat).
    train_*.transducer       lookups_total: times the memory bank was searched; hindsight_writes_total
                             and write_outs_total: rows written into / added to the bank.  In a toy run
                             the bank stays empty (see LIMITATIONS).
    train_p1_mixture.data_drawn
                             how many of the recipe's streams the run actually drew windows from;
                             healthy: all of them (needs --mixture-windows >= number of streams).
    resume_exactness.max_abs_difference
                             largest difference between weights etc. of an uninterrupted run and a
                             resumed one.  Healthy: exactly 0.0 on CPU.
    build_data.streams       per stream: kind real / stand_in, the root, the constructor, counts.
    launch_readiness.configs[].missing_streams
                             streams the config's stage needs that its listed roots cannot supply.
                             Healthy: an empty list for the config you will launch.
    launch_readiness.findings
                             the one-screen summary: streams with no producer at all, per-stage;
                             unfilled placeholder counts; configs with missing streams.
    serve_client.steps[]     one entry per request sent; "ok" is the service's own answer and "pass"
                             says whether it matched what the script expected.

IF IT FAILS, WHAT TO CHANGE
    launch_readiness: fill the PIN_ values, add the missing roots to `datasets:` (a stream is only
        read if some listed root holds it), or add a producer (a stream with "no producer found" has
        nothing that prepares it: add a preparation step or take it out of the recipe).
    build_data / train_*: the first traceback names the code path; fewer steps reproduce it faster.
    train_p1_mixture failing only with "Required validation behaviors ...": see KNOWN FINDINGS.
    resume_exactness difference > 0: something saved at a checkpoint is not restored (optimizer,
        random generators, data position); keep the workdir and compare the two checkpoints.
    serve_*: run the printed `megatoken serve` command by hand and paste the failing line in.

KNOWN FINDINGS (when this probe was written, 2026-10-23)
    * Held-out evaluation with a multi-root stage mixture fails at launch with "Required validation
      behaviors lack prepared original families ... available: {}" even when the validation data
      exists: src/megatoken/training/heldout.py (HeldOutEvaluator.__init__, the require_coverage
      call) checks the first stream's corpus only, which contains only the roots holding that
      stream's validation episodes.  eval_every > 0 on a real multi-root P1 config would fail the
      same way.  `--patch-heldout-check` works around it inside this probe so the rest of the
      evaluation path can be seen working (with it, all validated streams are scored).
    * This is an explicit bounded diagnostic without prepared teacher values. It exercises causal
      retrieval but does not invent targets or grow memory from a deleted learned-only curriculum.
      The joint teacher/structure checks exercise bank growth with prepared targets.

LIMITATIONS
    Media here is fake (random teacher features, tiny stand-in realizers): protocol and plumbing only.
    The streams base_reading, timed_conversation and streaming_translation have no cheap real
    constructor, so they are stand-ins.  The real file preparer for add_files needs the model packs.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import dataclasses
import fcntl
import json
import math
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
PYTHON = sys.executable
TESTS = REPO / "tests"
GPU_LOCK = "/tmp/megatoken-gpu.lock"


# ---------------------------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------------------------


class Skip(Exception):
    """Raised by a stage that cannot run because something it needs is unavailable."""


def finite_tree(value: Any, path: str = "") -> list[str]:
    """Paths of non-finite floats anywhere in a JSON-like tree."""
    bad: list[str] = []
    if isinstance(value, float) and not math.isfinite(value):
        bad.append(path or "<root>")
    elif isinstance(value, dict):
        for key, item in value.items():
            bad.extend(finite_tree(item, f"{path}.{key}" if path else str(key)))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            bad.extend(finite_tree(item, f"{path}[{index}]"))
    return bad


def clean(value: Any) -> Any:
    """Make a tree strictly JSON-serializable (paths to strings, non-finite floats to None)."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [clean(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


class Report:
    """Collects stage results; a failing stage never stops the stages that do not need it."""

    def __init__(self) -> None:
        self.stages: list[dict[str, Any]] = []
        self.by_name: dict[str, dict[str, Any]] = {}

    def run(self, name: str, function, *, needs: tuple[str, ...] = ()) -> Any:
        entry: dict[str, Any] = {
            "name": name,
            "status": "pass",
            "seconds": 0.0,
            "detail": None,
            "error": None,
            "traceback": None,
        }
        self.stages.append(entry)
        self.by_name[name] = entry
        blockers = [n for n in needs if self.by_name.get(n, {}).get("status") != "pass"]
        started = time.time()
        result = None
        if blockers:
            entry["status"] = "skip"
            entry["error"] = f"needs {', '.join(blockers)}, which did not pass"
        else:
            try:
                result = function()
                entry["detail"] = result
                bad = finite_tree(result)
                if bad:
                    raise ValueError(f"non-finite numbers in the stage result: {bad[:5]}")
            except Skip as exc:
                entry["status"] = "skip"
                entry["error"] = str(exc)
            except Exception as exc:  # noqa: BLE001 - a stage failure is data, not a crash
                entry["status"] = "fail"
                entry["error"] = f"{type(exc).__name__}: {exc}"
                entry["traceback"] = traceback.format_exc()
        entry["seconds"] = round(time.time() - started, 3)
        marker = {"pass": "PASS", "fail": "FAIL", "skip": "SKIP"}[entry["status"]]
        line = f"[{marker}] {name:<22} {entry['seconds']:8.1f}s"
        if entry["error"]:
            line += f"  {entry['error'].splitlines()[0][:150]}"
        print(line, flush=True)
        return result


def run_command(
    argv: list[str], log: Path, *, env: dict[str, str], timeout: float = 1800
) -> subprocess.CompletedProcess:
    """Run a command, saving combined output to ``log``; raise with the tail on failure."""
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as handle:
        handle.write("$ " + " ".join(argv) + "\n")
        handle.flush()
        process = subprocess.run(  # noqa: S603 - fixed argv
            argv, stdout=handle, stderr=subprocess.STDOUT, env=env, timeout=timeout, text=True
        )
    if process.returncode != 0:
        tail = "\n".join(log.read_text().splitlines()[-25:])
        raise RuntimeError(f"command exited {process.returncode}: {' '.join(argv[:6])}\n{tail}")
    return process


def child_env(threads: int = 1) -> dict[str, str]:
    env = dict(os.environ)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env.setdefault(name, str(threads))
    env["PYTHONUNBUFFERED"] = "1"
    return env


# ---------------------------------------------------------------------------------------------
# part 1: launch readiness (static)
# ---------------------------------------------------------------------------------------------

PLACEHOLDER = re.compile(r"PIN_[A-Z0-9_]+|REPLACE-WITH-[A-Z0-9-]+")

# Which preparation steps fill a data root, judging by the name the example configs give it.
# This is a declared assumption (the configs only carry placeholders); edit it if the roots'
# meaning changes.  Each entry lists producer *kinds*: "sources" = any `megatoken prepare
# --sources configs/data/sources*.json` output; otherwise a script path.
ROOT_ROLE_HINTS: dict[str, tuple[str, ...]] = {
    "TEXT_AND_CODE": ("sources",),
    "LATER_STAGE_TEXT": ("sources",),
    "STUDENT_CORRECTIONS": ("cli",),
    "ALIGNED_TASKS": ("megatoken.data.concordance:compile_task",),
    "TEXT_ROOT": ("sources",),
    "CAUSAL_MERIDIAN_EPISODES": ("sources",),
    "MEDIA_IDENTITY": ("scripts/prepare_media_identity.py", "scripts/prepare_open_media.py"),
    "MEDIA_TASKS": ("scripts/prepare_open_media.py",),
    "LONG_DOCUMENT_READING": ("scripts/prepare_long_document_reading.py",),
    "BASE_AND_COLLECTION_READING": (
        "scripts/prepare_base_reading.py",
        "scripts/prepare_collection_reading.py",
    ),
    "DYNAMIC_EXECUTION": (
        "scripts/accept_dynamic_tracks.py",
        "scripts/prepare_execution_traces.py",
    ),
    "TIMED_MEDIA": ("scripts/prepare_timed_media.py", "scripts/prepare_meeting_prefix.py"),
}
# Scripts that really write a prepared root (the others measure, verify or acquire).
PRODUCER_PREFIXES = ("prepare_", "import_")
PRODUCER_EXTRA = ("accept_dynamic_tracks.py",)
# Library modules that mention stream names without producing them.
NOT_PRODUCERS = {
    "recipes.py",
    "coverage.py",
    "storage.py",
    "schema.py",
    "chunking.py",
    "sharding.py",
}


def _read(path: Path) -> str:
    try:
        return path.read_text(errors="replace")
    except OSError:
        return ""


def _quoted(name: str) -> re.Pattern:
    return re.compile("[\"']" + re.escape(name) + "[\"']")


def producer_index(repo: Path, streams: set[str]) -> dict[str, list[dict[str, str]]]:
    """For each stream: scripts, library modules and cloud jobs that can produce it."""
    scripts = sorted((repo / "scripts").glob("*.py"))
    producer_scripts = [
        p for p in scripts if p.name.startswith(PRODUCER_PREFIXES) or p.name in PRODUCER_EXTRA
    ]
    modules = [
        p for p in sorted((repo / "src/megatoken/data").glob("*.py")) if p.name not in NOT_PRODUCERS
    ]
    sources_files = sorted((repo / "configs/data").glob("sources*.json"))
    cloud_jobs = sorted((repo / "configs/cloud").glob("*.json"))
    cloud_text = {job: _read(job) for job in cloud_jobs}
    data_cli = _read(repo / "src/megatoken/data/cli.py")
    # which streams each sources file declares
    declared: dict[Path, set[str]] = {}
    for path in sources_files:
        try:
            data = json.loads(_read(path))
        except ValueError:
            continue
        found: set[str] = set()

        def walk(node, found=found):
            if isinstance(node, dict):
                if isinstance(node.get("stream"), str):
                    found.add(node["stream"])
                for item in node.values():
                    walk(item)
            elif isinstance(node, list):
                for item in node:
                    walk(item)

        walk(data)
        declared[path] = found
    index: dict[str, list[dict[str, str]]] = {}
    for stream in sorted(streams):
        pattern = _quoted(stream)
        found_list: list[dict[str, str]] = []
        for path, names in declared.items():
            if stream in names:
                rel = str(path.relative_to(repo))
                found_list.append({"kind": "sources_file", "path": rel, "how": "megatoken prepare"})
                for job, text in cloud_text.items():
                    if rel in text or path.name in text:
                        found_list.append(
                            {
                                "kind": "cloud_job",
                                "path": str(job.relative_to(repo)),
                                "how": f"runs megatoken prepare on {path.name}",
                            }
                        )
        module_hits = [m for m in modules if pattern.search(_read(m))]
        for module in module_hits:
            dotted = f"megatoken.data.{module.stem}"
            via_scripts = [s for s in producer_scripts if dotted in _read(s)]
            direct = [s for s in producer_scripts if pattern.search(_read(s))]
            for script in {*via_scripts, *direct}:
                rel = str(script.relative_to(repo))
                found_list.append({"kind": "script", "path": rel, "how": f"uses {dotted}"})
            if (
                not via_scripts
                and not direct
                and re.search(
                    rf"megatoken\.data\.{module.stem}\b|from \.{module.stem} import|"
                    rf"from \. import [^\n]*\b{module.stem}\b",
                    data_cli,
                )
            ):
                found_list.append(
                    {
                        "kind": "cli",
                        "path": "src/megatoken/data/cli.py",
                        "how": f"`megatoken data ...` commands call {dotted}",
                    }
                )
            elif not via_scripts and not direct:
                found_list.append(
                    {
                        "kind": "library",
                        "path": str(module.relative_to(repo)),
                        "how": "library code only; no script or cloud job calls it",
                    }
                )
        for script in producer_scripts:
            if pattern.search(_read(script)) and not any(
                f["path"] == str(script.relative_to(repo)) for f in found_list
            ):
                found_list.append(
                    {
                        "kind": "script",
                        "path": str(script.relative_to(repo)),
                        "how": "names the stream",
                    }
                )
        for item in [f for f in found_list if f["kind"] == "script"]:
            for job, text in cloud_text.items():
                if item["path"] in text:
                    found_list.append(
                        {
                            "kind": "cloud_job",
                            "path": str(job.relative_to(repo)),
                            "how": f"runs {Path(item['path']).name}",
                        }
                    )
        unique = {(f["kind"], f["path"]): f for f in found_list}
        index[stream] = list(unique.values())
    return index


def root_streams(hints: tuple[str, ...], index: dict[str, list[dict[str, str]]]) -> set[str]:
    """Streams a placeholder root could hold, given its role hint."""
    from megatoken.data.recipes import STREAM_PRODUCERS

    result: set[str] = {
        stream for stream, producer in STREAM_PRODUCERS.items() if producer in hints
    }
    for stream, producers in index.items():
        for producer in producers:
            for hint in hints:
                if (
                    (hint == "sources" and producer["kind"] == "sources_file")
                    or (hint == "cli" and producer["kind"] == "cli")
                    or (
                        hint != "sources"
                        and producer["kind"] == "script"
                        and producer["path"] == hint
                    )
                ):
                    result.add(stream)
    return result


def manifest_streams(root: Path) -> set[str] | None:
    try:
        manifest = json.loads((root / "manifest.json").read_text())
    except (OSError, ValueError):
        return None
    return {
        key.split("stream:", 1)[1]
        for key in manifest.get("counts", {})
        if key.startswith("stream:")
    }


def find_placeholders(value: Any, path: str = "") -> list[dict[str, str]]:
    found: list[dict[str, str]] = []
    if isinstance(value, str):
        for match in PLACEHOLDER.finditer(value):
            found.append({"key": path, "placeholder": match.group(0), "value": value})
    elif isinstance(value, dict):
        for key, item in value.items():
            found.extend(find_placeholders(item, f"{path}.{key}" if path else str(key)))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(find_placeholders(item, f"{path}[{index}]"))
    return found


def dead_config_keys(repo: Path) -> dict[str, list[str]]:
    """Fields the configuration classes define that no other source file ever reads.

    A field counts as read when some other file mentions ``config.NAME`` (or a close variant such
    as ``decoupled.NAME``), ``getattr(config, "NAME")`` or the quoted name inside a mapping.
    """
    from megatoken.training.config import DecoupledConfig, TrainConfig

    blob = "\n".join(
        _read(p)
        for p in (repo / "src/megatoken").rglob("*.py")
        if not (p.name == "config.py" and p.parent.name == "training")
    )
    owners = {
        "TrainConfig": r"(?:config|cfg|train_config|self\.config|base|settings|args)",
        "DecoupledConfig": r"(?:decoupled|self\.config|config\.decoupled|settings|self\.settings)",
    }
    result: dict[str, list[str]] = {}
    for cls in (TrainConfig, DecoupledConfig):
        owner = owners[cls.__name__]
        unread = []
        for f in dataclasses.fields(cls):
            name = re.escape(f.name)
            pattern = (
                rf"\b{owner}\.{name}\b|getattr\([A-Za-z_.]+, [\"']{name}[\"']\)"
                rf"|[\"']{name}[\"']\s*[:,\]]"
            )
            if not re.search(pattern, blob):
                unread.append(f.name)
        result[cls.__name__] = unread
    return result


def audit_config(path: Path, repo: Path, recipes: dict, index: dict) -> dict[str, Any]:
    from megatoken.data.coverage import behaviors  # noqa: F401  (import check only)
    from megatoken.io import read_mapping
    from megatoken.training.config import DecoupledConfig, TrainConfig

    raw = read_mapping(path)
    name = str(path.relative_to(repo)) if path.is_relative_to(repo) else str(path)
    entry: dict[str, Any] = {"config": name, "phase": raw.get("phase")}
    known = {f.name for f in dataclasses.fields(TrainConfig)}
    entry["unread_keys"] = sorted(set(raw) - known)
    decoupled = raw.get("decoupled")
    entry["unread_decoupled_keys"] = (
        sorted(set(decoupled) - {f.name for f in dataclasses.fields(DecoupledConfig)})
        if isinstance(decoupled, dict)
        else []
    )
    entry["unfilled_placeholders"] = find_placeholders(raw)
    resolved = None
    try:
        resolved = TrainConfig.from_dict(raw, expand_data=False)
        entry["loader_error"] = None
    except Exception as exc:  # noqa: BLE001
        entry["loader_error"] = f"{type(exc).__name__}: {exc}"
    entry["default_memory"] = {
        "required": resolved is not None and resolved.purpose == "production",
        "teacher_artifact": None if resolved is None else resolved.hindsight_targets,
        "navigation_enabled": resolved is not None and resolved.navigation_weight > 0,
        "distillation_enabled": resolved is not None and resolved.hindsight_distill_weight > 0,
    }
    entry["default_memory"]["configured"] = all(entry["default_memory"].values())
    datasets = [d for d in raw.get("datasets", []) if isinstance(d, str)]
    roots: list[dict[str, Any]] = []
    supplied: set[str] = set()
    for value in datasets:
        item: dict[str, Any] = {"value": value}
        if PLACEHOLDER.search(value):
            hint_key = next((k for k in ROOT_ROLE_HINTS if k in value), None)
            item["placeholder"] = True
            item["role_hint"] = hint_key
            streams = (
                root_streams(ROOT_ROLE_HINTS[hint_key], index) if hint_key is not None else set()
            )
            item["streams_expected"] = sorted(streams)
            if hint_key is None:
                item["note"] = "role unknown from the name; its streams cannot be attributed"
        else:
            item["placeholder"] = False
            streams = manifest_streams(Path(value)) or set()
            item["streams_found"] = sorted(streams)
            if not Path(value).exists():
                item["note"] = "path does not exist here"
        supplied |= streams
        roots.append(item)
    entry["roots"] = roots
    phase = raw.get("phase")
    recipe = recipes["phases"].get(phase)
    entry["uses_stage_mixture"] = (
        resolved.use_stage_mixture
        if resolved is not None
        else bool(raw.get("use_stage_mixture", False))
    )
    missing: list[dict[str, Any]] = []
    if recipe is not None and entry["uses_stage_mixture"]:
        for stream, share in recipe["mixture"].items():
            if stream not in supplied:
                producers = index.get(stream, [])
                missing.append(
                    {
                        "stream": stream,
                        "share": round(share, 6),
                        "producers": producers or "no producer found",
                    }
                )
    elif recipe is not None:
        entry["note"] = (
            "use_stage_mixture is off: the whole concatenation of the roots is one stream, "
            "so no particular stream is required"
        )
    entry["missing_streams"] = missing
    entry["missing_share"] = round(sum(m["share"] for m in missing), 6)
    entry["roots_listed"] = len(datasets)
    return entry


def launch_readiness(repo: Path, extra: tuple[Path, ...] = ()) -> dict[str, Any]:
    from megatoken.data.coverage import behaviors
    from megatoken.data.dynamic_execution import execution_episode
    from megatoken.data.recipes import load_recipes

    recipes = load_recipes()
    streams = {s for p in recipes["phases"].values() for s in p["mixture"]}
    index = producer_index(repo, streams)
    configs = [*sorted((repo / "configs/train").glob("*.yaml")), *extra]
    audited = [audit_config(path, repo, recipes, index) for path in configs]
    stages: dict[str, Any] = {}
    for phase, recipe in recipes["phases"].items():
        no_producer = sorted(s for s in recipe["mixture"] if not index.get(s))
        library_only = sorted(
            s
            for s in recipe["mixture"]
            if index.get(s) and all(p["kind"] == "library" for p in index[s])
        )
        stages[phase] = {
            "streams": len(recipe["mixture"]),
            "no_producer_found": no_producer,
            "library_code_only": library_only,
            "required_behaviors": recipe["constraints"].get("required_behaviors", {}),
        }
    # Which required behaviours this repository's own authored producer already demonstrates.
    demonstrated = sorted(behaviors(execution_episode(1, delegated=True)))
    findings = {
        "streams_without_any_producer": sorted(
            {s for st in stages.values() for s in st["no_producer_found"]}
        ),
        "streams_without_producer_by_stage": {
            phase: st["no_producer_found"]
            for phase, st in stages.items()
            if st["no_producer_found"]
        },
        "streams_with_library_code_only": sorted(
            {s for st in stages.values() for s in st["library_code_only"]}
        ),
        "phases_without_an_example_config": sorted(
            set(recipes["phases"]) - {a["phase"] for a in audited}
        ),
        "unread_keys": {a["config"]: a["unread_keys"] for a in audited if a["unread_keys"]},
        "unfilled_placeholder_count": {
            a["config"]: len(a["unfilled_placeholders"]) for a in audited
        },
        "configs_with_missing_streams": {
            a["config"]: [m["stream"] for m in a["missing_streams"]]
            for a in audited
            if a["missing_streams"]
        },
        "configs_without_default_memory": [
            a["config"] for a in audited if not a["default_memory"]["configured"]
        ],
        "config_loader_errors": {
            a["config"]: a["loader_error"] for a in audited if a["loader_error"]
        },
    }
    for a in audited:
        if a["phase"] == "P1" and a["uses_stage_mixture"]:
            findings.setdefault("p1_roots_listed", {})[a["config"]] = a["roots_listed"]
    return {
        "configs": audited,
        "stages": stages,
        "dead_config_fields": dead_config_keys(repo),
        "behaviors_shown_by_authored_executions": demonstrated,
        "producer_assumptions": {k: list(v) for k, v in ROOT_ROLE_HINTS.items()},
        "findings": findings,
    }


def print_readiness(result: dict[str, Any]) -> None:
    findings = result["findings"]
    print("launch readiness:")
    print(f"  streams no script or cloud job produces: {findings['streams_without_any_producer']}")
    for phase, names in findings["streams_without_producer_by_stage"].items():
        print(f"      stage {phase} names {names} -> no producer found")
    for config in result["configs"]:
        placeholders = len(config["unfilled_placeholders"])
        print(
            f"  {config['config']}: phase {config['phase']}, {config['roots_listed']} roots, "
            f"{placeholders} unfilled placeholders, {len(config['unread_keys'])} unread keys, "
            f"{len(config['missing_streams'])} missing streams "
            f"(share {config['missing_share']})"
        )
        for item in config["missing_streams"]:
            producers = item["producers"]
            if isinstance(producers, list):
                producers = "; ".join(f"{p['kind']} {p['path']}" for p in producers[:3])
            print(f"      missing {item['stream']}: {producers}")


# ---------------------------------------------------------------------------------------------
# part 2: synthetic data
# ---------------------------------------------------------------------------------------------

MEDIA_STREAM_MODALITY = {
    "audio_identity": "audio",
    "image_identity": "image",
    "video_identity": "video",
    "audio_description": "audio",
    "video_description": "video",
    "transcription": "audio",
    "verbalization": "audio",
    "speech_translation": "audio",
    "image_description": "image",
    "image_creation": "image",
    "av_creation": "video",
    "image_edit": "image",
    "edit_judging": "image",
    "edit_plan": "image",
    "speech_edit": "audio",
    "av_edit": "video",
}
EXECUTED_STANDINS = (
    "executed_trajectories",
    "verified_trajectories",
    "tool_use",
    "environment_trajectories",
)


def setup_paths() -> None:
    for path in (str(TESTS), str(REPO / "src")):
        if path not in sys.path:
            sys.path.insert(0, path)


class DataBuilder:
    """Builds the synthetic prepared roots and a per-stream report of what is real."""

    def __init__(self, workdir: Path, seed: int, per_stream: int, validation: str, model: str):
        setup_paths()
        self.model = model
        self.workdir = workdir
        self.seed = seed
        self.per_stream = per_stream
        self.validation = validation
        self.counter = 1000 * (seed % 97 + 1)
        self.report: dict[str, dict[str, Any]] = {}
        self.roots: dict[str, Path] = {}

    # -- bookkeeping ---------------------------------------------------------------------
    def next_index(self) -> int:
        self.counter += 1
        return self.counter

    def note(self, stream: str, kind: str, root: str, constructor: str, train: int, val: int):
        self.report[stream] = {
            "kind": kind,
            "root": root,
            "constructor": constructor,
            "train_episodes": train,
            "validation_episodes": val,
        }

    def wants_validation(self, stream: str) -> bool:
        from megatoken.data.recipes import STREAM_PRODUCERS

        # The launch check wants held-out supply for every procedurally produced stream.
        if self.validation == "all" or stream in STREAM_PRODUCERS:
            return True
        return stream in {
            "natural_documents",
            "chat",
            "dynamic_execution",
            "tool_use",
            "audio_identity",
            "image_identity",
            "video_identity",
            "collection_reading",
        }

    def export(self, root: Path, episodes: list) -> Path:
        from megatoken.data import DedupConfig, DedupStore
        from megatoken.data.storage import export_prepared
        from megatoken.generation.codec import ByteCodec

        codec = ByteCodec.byte_fallback()
        root.mkdir(parents=True, exist_ok=True)
        spool = root.parent / f"{root.name}-spool.sqlite"
        spool.unlink(missing_ok=True)
        with DedupStore(spool, DedupConfig(train_fraction=1.0, validation_fraction=0.0)) as store:
            with store.transaction():
                for episode in episodes:
                    store.add(episode, codec)
            store.freeze()
            export_prepared(
                store,
                root,
                codec=codec,
                source_receipts=[],
                pipeline_fingerprint=f"rehearsal-{self.seed}",
            )
        return root

    def split(self, episode, split: str):
        return dataclasses.replace(
            episode, lineage=dataclasses.replace(episode.lineage, frozen_split=split)
        )

    # -- text-like streams ---------------------------------------------------------------
    def text_episode(self, stream: str, *, doc_bytes: int = 0, turns: int = 1):
        from training_fixture import make_episode

        index = self.next_index()
        episode = make_episode(index, turns=turns)
        if doc_bytes:
            filler = (episode.records[0].payload + b" ") * (doc_bytes // 40 + 1)
            first = dataclasses.replace(episode.records[0], payload=filler[:doc_bytes])
            episode = dataclasses.replace(episode, records=(first, *episode.records[1:]))
        return dataclasses.replace(
            episode,
            stream=stream,
            lineage=dataclasses.replace(episode.lineage, family_id=f"{stream}-family-{index}"),
        )

    def executed_episode(self, stream: str, *, delegated: bool, variant: int | None = None):
        from megatoken.data.dynamic_execution import execution_episode

        index = self.next_index()
        # The stage recipes require each way of handling slow, cheap and dependent work to be
        # shown; the variants cycle through them (the short real delays keep it fast).
        options = {
            0: dict(delegated=True, read_delay_seconds=0.2, delegation_overhead_seconds=0.02),
            1: dict(delegated=False, read_delay_seconds=0.02, delegation_overhead_seconds=0.2),
            2: dict(delegated=False, dependent=True),
            3: dict(delegated=delegated),
        }
        episode = execution_episode(index, **options[3 if variant is None else variant % 4])
        return dataclasses.replace(episode, stream=stream)

    def streaming_episode(self, stream: str):
        """A real sustained-translation episode (its waits and emits), over a fake preparer."""
        from test_open_media import FakePreparer, make_source

        from megatoken.data.media_assets import TensorAssetStore
        from megatoken.data.open_media import CachingPreparer, translation_text_episodes

        index = self.next_index()
        scratch = self.workdir / "streaming" / str(index)
        scratch.mkdir(parents=True, exist_ok=True)
        audio = scratch / "recording.wav"
        import numpy as np
        import soundfile

        soundfile.write(audio, 0.3 * np.sin(np.arange(48000) / (3 + index % 5)), 16000, format="WAV")
        source = make_source(
            "translation", "translation_text", options={"sustained_translation": True}
        )
        episodes = translation_text_episodes(
            CachingPreparer(FakePreparer(TensorAssetStore(self.workdir / "root-text"))),
            source,
            {
                "id": f"row-{index}",
                "family": f"speaker-{index}",
                "audio": audio,
                "text": "Bonjour.",
                "translation": "Hello.",
                "language": {"source": "fr", "target": "en"},
            },
        )
        return next(e for e in episodes if e.stream == stream)

    def build_text_root(self, recipe_streams: list[str]) -> Path:
        episodes = []
        for stream in recipe_streams:
            if stream in MEDIA_STREAM_MODALITY or stream in {
                "collection_reading",
                "duplex_execution",
            }:
                continue
            train, val = 0, 0
            # Dynamic execution shows four ways of handling work, in each split.
            train_count = max(self.per_stream, 4) if stream == "dynamic_execution" else self.per_stream
            val_count = (4 if stream == "dynamic_execution" else 1) if self.wants_validation(stream) else 0
            for number in range(train_count + val_count):
                is_val = number >= train_count
                if stream == "dynamic_execution":
                    episode = self.executed_episode(
                        stream, delegated=is_val or number % 2 == 0, variant=number
                    )
                    kind, constructor = "real", "megatoken.data.dynamic_execution.execution_episode"
                elif stream == "streaming_translation":
                    episode = self.streaming_episode(stream)
                    kind, constructor = "real", "megatoken.data.open_media.translation_text_episodes"
                elif stream in EXECUTED_STANDINS:
                    episode = self.executed_episode(stream, delegated=number % 2 == 1)
                    kind = "stand_in"
                    constructor = "execution_episode relabelled as this stream"
                elif stream == "long_document_reading":
                    episode = self.text_episode(stream, doc_bytes=900)
                    kind, constructor = "stand_in", "tests/training_fixture.make_episode, long doc"
                else:
                    episode = self.text_episode(stream, turns=1 + number % 2)
                    kind, constructor = "stand_in", "tests/training_fixture.make_episode"
                if is_val:
                    episode = self.split(episode, "validation")
                    val += 1
                else:
                    train += 1
                episodes.append(episode)
            self.note(stream, kind, "text", constructor, train, val)
        self.roots["text"] = self.export(self.workdir / "root-text", episodes)
        return self.roots["text"]

    # -- media ---------------------------------------------------------------------------
    def build_media_root(self, recipe_streams: list[str]) -> Path:
        from media_support import prepared_record

        from megatoken.data.media_assets import TensorAssetStore
        from megatoken.data.open_media import identity_episode

        root = self.workdir / "root-media"
        store = TensorAssetStore(root)
        episodes = []
        for stream in recipe_streams:
            modality = MEDIA_STREAM_MODALITY.get(stream)
            if modality is None:
                continue
            train = val = 0
            for number in range(self.per_stream + (1 if self.wants_validation(stream) else 0)):
                seed = self.next_index()
                record = prepared_record(store, modality, seed=seed)
                episode = identity_episode(record, family=f"{stream}-{seed}")
                episode = dataclasses.replace(episode, stream=stream)
                if number >= self.per_stream:
                    episode = self.split(episode, "validation")
                    val += 1
                else:
                    episode = self.split(episode, "train")
                    train += 1
                episodes.append(episode)
            real = stream.endswith("_identity")
            self.note(
                stream,
                "real" if real else "stand_in",
                "media",
                "megatoken.data.open_media.identity_episode"
                + ("" if real else " relabelled as this stream"),
                train,
                val,
            )
        self.roots["media"] = self.export(root, episodes)
        return self.roots["media"]

    def with_model_geometry(self, episode, codec):
        """The fixture's recorded action tape has a 1x1 facet shape; give it this model's."""
        import numpy as np
        from native_support import group_tape

        from megatoken.data.schema import canonical_json
        from megatoken.loading import load_model_config
        from megatoken.model.multitrack import GroupActionTape

        config = load_model_config(self.model)
        shape = (config.n_facets, config.facet_dim)
        annotations = json.loads(episode.supervision.annotations_json)
        entry = annotations["native_action_tapes"][0]
        old = GroupActionTape.from_json(json.dumps(entry["tape"]))
        rng = np.random.default_rng(self.seed)
        facets = (0.1 * rng.standard_normal(shape[0] * shape[1])).astype("<f4")
        tape = group_tape(
            codec,
            old.payloads[0],
            emitted_ids=old.emitted_ids[0],
            track="agent",
            role=2,
            shape=shape,
            facets=facets,
        )
        tape = dataclasses.replace(tape, meridian_id=1, event_ids=("selected-action",))
        entry["tape"] = json.loads(tape.to_json())
        supervision = dataclasses.replace(
            episode.supervision, annotations_json=canonical_json(annotations)
        )
        return dataclasses.replace(episode, supervision=supervision)

    def build_duplex_root(self) -> Path:
        import importlib

        module = importlib.import_module("test_execution_trace_preparation")
        for label, split in (("duplex", "train"), ("duplex_heldout", "validation")):
            base = self.workdir / f"{label}-work"
            base.mkdir(parents=True, exist_ok=True)
            episode, codec, source_root = module.fixture(base, stream="duplex_execution")
            episode = self.with_model_geometry(episode, codec)
            if split == "validation":  # the held-out check needs its own family
                episode = dataclasses.replace(
                    episode,
                    episode_id="trace-heldout",
                    lineage=dataclasses.replace(
                        episode.lineage, family_id="source-family-heldout", frozen_split=split
                    ),
                )
            module.prepare(base, episode, codec, source_root)
            self.roots[label] = base / "prepared"
        self.note(
            "duplex_execution",
            "real",
            "duplex",
            "megatoken.data.execution_traces.prepare_traces on the test fixture trace",
            1,
            1,
        )
        return self.roots["duplex"]

    def build_collection_root(self) -> Path:
        from retrieval_support import reading_root

        root = self.workdir / "root-collection"
        root.mkdir(parents=True, exist_ok=True)
        self.roots["collection"] = reading_root(root, tasks=3)
        # one held-out reading root, so the stream has validation episodes
        heldout = self.workdir / "root-collection-heldout"
        heldout.mkdir(parents=True, exist_ok=True)
        self.roots["collection_heldout"] = reading_root(heldout, tasks=1, first=100, heldout=True)
        self.note(
            "collection_reading",
            "real",
            "collection",
            "tests/retrieval_support.reading_root (episodes + source collection)",
            3,
            1,
        )
        return self.roots["collection"]


def build_data(workdir: Path, args) -> dict[str, Any]:
    from megatoken.data.recipes import stage_recipe

    streams = list(stage_recipe("P1").mixture)
    builder = DataBuilder(
        workdir / "data", args.seed, args.episodes_per_stream, args.validation, args.model
    )
    builder.build_text_root(streams)
    builder.build_media_root(streams)
    builder.build_duplex_root()
    builder.build_collection_root()
    covered = set(builder.report)
    shares = stage_recipe("P1").mixture
    missing = [s for s in streams if s not in covered]
    return {
        "roots": {k: str(v) for k, v in builder.roots.items()},
        "streams": builder.report,
        "p1_streams": len(streams),
        "p1_streams_covered": len(covered & set(streams)),
        "p1_streams_not_built": missing,
        "p1_share_real": round(
            sum(shares[s] for s, r in builder.report.items() if r["kind"] == "real"), 6
        ),
        "p1_share_stand_in": round(
            sum(shares[s] for s, r in builder.report.items() if r["kind"] == "stand_in"), 6
        ),
        "builder": builder,
    }


# ---------------------------------------------------------------------------------------------
# part 3: training stages
# ---------------------------------------------------------------------------------------------


def read_events(output: Path) -> list[dict[str, Any]]:
    path = output / "metrics.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@contextlib.contextmanager
def fake_realizers():
    """Install tiny stand-ins for the Pocket/Sana/Sana-Video donors (protocol only)."""
    setup_paths()
    from media_support import install_fake_realizers

    import megatoken.omni.loader as loader

    original = loader.install_realizers

    def install(model, path, *, device):
        install_fake_realizers(model)
        for realizer in model.media_realizers.packs.values():
            realizer.to(device)  # the stand-ins are built on the CPU
        return model

    loader.install_realizers = install
    try:
        yield
    finally:
        loader.install_realizers = original


@contextlib.contextmanager
def patched_heldout_check():
    """Work around the known held-out coverage bug: check the unscoped validation corpus.

    HeldOutEvaluator hands require_coverage the first stream's corpus, which holds only the roots
    that contain that one stream (see KNOWN FINDINGS in the module docstring).
    """
    import megatoken.data.coverage as coverage
    from megatoken.data import PreparedCorpus

    original = coverage.require_coverage

    def wrapper(corpus, requirements, streams, *, split="train"):
        if split == "validation" and getattr(corpus, "stream", None) is not None:
            with PreparedCorpus(list(corpus.all_roots), split="validation") as whole:
                return original(whole, requirements, streams, split=split)
        return original(corpus, requirements, streams, split=split)

    coverage.require_coverage = wrapper
    try:
        yield
    finally:
        coverage.require_coverage = original


class Trainer:
    """Runs a training stage through the real command line, or in this process."""

    def __init__(self, workdir: Path, args, env: dict[str, str]):
        self.workdir, self.args, self.env = workdir, args, env

    def base_config(self, name: str, **overrides) -> dict[str, Any]:
        args = self.args
        config: dict[str, Any] = dict(
            purpose="diagnostic",
            phase="B0",
            model=args.model,
            datasets=[],
            output=str(self.workdir / "runs" / name),
            codec="byte",
            allow_byte_codec=True,
            device=args.device,
            precision="fp32",
            seed=args.seed,
            microbatch_episodes=1,
            max_accumulation_windows=2,
            accumulate_scored=1,
            accumulate_ingested=2048,
            max_window_records=4,
            scored_decisions=2,
            max_history_tokens=512,
            generated_capacity=512,
            max_record_bytes=4096,
            effort=1,
            gradient_checkpointing=False,
            expert_backend="torch",
            checkpoint_every=1,
            keep_checkpoints=8,
            eval_every=0,
            eval_windows=1,
            log_every=1,
            use_stage_mixture=False,
            use_effort_mixture=False,
            model_overrides={"max_capacity": 512, "vocab_size": 256},
        )
        if args.model != "tiny":
            config["model_overrides"].update({"ngram_rows": 131072, "max_history_chunks": 64})
        config.update(overrides)
        return config

    def run(
        self,
        name: str,
        config: dict[str, Any],
        *,
        max_steps: int,
        resume: str | None = None,
        phase_transition: bool = False,
        cli: bool | None = None,
        media: bool = False,
    ) -> dict[str, Any]:
        config = dict(config)
        path = self.workdir / "configs" / f"{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(config, indent=2))
        use_cli = self.args.real_cli if cli is None else cli
        if media:
            use_cli = False  # fake realizers have to be installed in this process
        if use_cli:
            out = self.workdir / "logs" / f"{name}.result.json"
            argv = [PYTHON, "-m", "megatoken", "--json-output", str(out), "train"]
            argv += ["--config", str(path), "--max-steps", str(max_steps)]
            if resume:
                argv += ["--resume", resume]
            if phase_transition:
                argv += ["--phase-transition"]
            run_command(argv, self.workdir / "logs" / f"{name}.log", env=self.env)
            result = json.loads(out.read_text())
            result["via"] = "cli"
        else:
            from megatoken.training.config import TrainConfig
            from megatoken.training.engine import train

            parsed = TrainConfig.load(path, max_steps=max_steps)
            log = self.workdir / "logs" / f"{name}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            with contextlib.ExitStack() as stack:
                if media:
                    stack.enter_context(fake_realizers())
                    if self.args.patch_heldout_check:
                        stack.enter_context(patched_heldout_check())
                handle = stack.enter_context(log.open("w"))
                stack.enter_context(contextlib.redirect_stdout(handle))
                result = train(parsed, resume=resume, phase_transition=phase_transition)
            result["via"] = "library"
        return result


def summarize_training(result: dict[str, Any], output: Path) -> dict[str, Any]:
    events = read_events(output)
    steps = [e for e in events if e.get("event") == "train"]
    if not steps:
        raise RuntimeError(f"no train events were logged in {output / 'metrics.jsonl'}")
    for event in steps:
        for key, value in (event.get("loss") or {}).items():
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"step {event['global_step']}: loss term {key} is {value!r}")
        if not math.isfinite(event.get("gradient_norm", 0.0)):
            raise ValueError(f"step {event['global_step']}: gradient norm is not finite")
    evaluations = [e for e in events if e.get("event") == "evaluation"]
    numbers = [e["global_step"] for e in steps]
    summary = {
        "via": result.get("via"),
        "global_step": result["global_step"],
        "phase": result["phase"],
        "complete": result["complete"],
        "phase_consumed": result["phase_consumed"],
        "counters": {k: v for k, v in result["counters"].items() if v},
        "checkpoint": result["checkpoint"],
        "logged_steps": numbers,
        "steps_contiguous": numbers == list(range(numbers[0], numbers[0] + len(numbers))),
        "first_logged_step": numbers[0],
        "loss_first": steps[0]["loss"],
        "loss_last": steps[-1]["loss"],
        "gradient_norm_last": steps[-1]["gradient_norm"],
        "learning_rate_last": steps[-1].get("lr"),
        "step_seconds_mean": round(
            sum(e.get("step_seconds", 0.0) for e in steps) / len(steps),
            3,
        ),
        "transducer": {
            "edges_last": steps[-1].get("transducer_edges"),
            "lookups_total": sum(
                e.get("generator", {}).get("transducer_lookups", 0) for e in steps
            ),
            "hindsight_writes_total": sum(
                e.get("generator", {}).get("transducer_writes", 0) for e in steps
            ),
            "write_outs_total": sum(e.get("transducer_writeouts", 0) for e in steps),
        },
        "evaluations": len(evaluations),
    }
    if evaluations:
        last = evaluations[-1]
        summary["evaluation_streams"] = {
            name: {"windows": item["windows"], "weighted_loss": item.get("weighted_loss")}
            for name, item in last["streams"].items()
        }
        summary["evaluation_pooled_weighted_loss"] = last.get("pooled_weighted_loss")
    return summary


def drawn_streams(checkpoint: str) -> dict[str, Any]:
    """Which stage-mixture streams the run actually drew windows from (from the saved cursor)."""
    from megatoken.training.checkpoint import load_checkpoint

    payload, _ = load_checkpoint(checkpoint, map_location="cpu")
    cursor = payload["rank_state"][0]["cursor"]
    exposure = cursor.get("exposure", {})
    drawn = sorted(name for name, amount in exposure.items() if amount > 0)
    return {
        "streams_in_mixture": len(exposure),
        "streams_drawn": len(drawn),
        "streams_never_drawn": sorted(set(exposure) - set(drawn)),
    }


def tree_equal(left: Any, right: Any, path: str = "") -> tuple[float, list[str]]:
    """Largest absolute difference between two checkpoint trees, and the paths that differ."""
    import torch

    worst, bad = 0.0, []
    if isinstance(left, torch.Tensor):
        if left.shape != right.shape or left.dtype != right.dtype:
            return math.inf, [path + " (shape/dtype)"]
        if left.numel() == 0:
            return 0.0, []
        if left.dtype.is_floating_point or left.dtype.is_complex:
            diff = (left.double() - right.double()).abs()
            diff = diff[~(torch.isnan(left.double()) & torch.isnan(right.double()))]
            value = float(diff.max()) if diff.numel() else 0.0
        else:
            value = float((left != right).any())
        if value:
            bad.append(path)
        return value, bad
    if isinstance(left, dict):
        if left.keys() != right.keys():
            return math.inf, [path + " (keys)"]
        for key in left:
            w, b = tree_equal(left[key], right[key], f"{path}.{key}")
            worst, bad = max(worst, w), bad + b
        return worst, bad
    if isinstance(left, (list, tuple)):
        if len(left) != len(right):
            return math.inf, [path + " (length)"]
        for index, (a, b_) in enumerate(zip(left, right, strict=True)):
            w, b = tree_equal(a, b_, f"{path}[{index}]")
            worst, bad = max(worst, w), bad + b
        return worst, bad
    return (0.0, []) if left == right else (math.inf, [path])


# ---------------------------------------------------------------------------------------------
# part 4: JSON-lines service client
# ---------------------------------------------------------------------------------------------


class ServeProcess:
    """`megatoken serve` as a subprocess, spoken to one JSON line at a time."""

    def __init__(self, argv: list[str], env: dict[str, str], stderr_path: Path):
        self.argv = argv
        self.stderr_path = stderr_path
        stderr_path.parent.mkdir(parents=True, exist_ok=True)
        self.stderr = stderr_path.open("w")
        self.process = subprocess.Popen(  # noqa: S603 - fixed argv
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.stderr,
            env=env,
            text=True,
            bufsize=1,
        )
        self.lines: queue.Queue = queue.Queue()
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self) -> None:
        for line in self.process.stdout:  # type: ignore[union-attr]
            self.lines.put(line)
        self.lines.put(None)

    def call(self, request: dict[str, Any], timeout: float = 300.0) -> dict[str, Any]:
        assert self.process.stdin is not None
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        return self.read(timeout)

    def send_raw(self, text: str, timeout: float = 120.0) -> dict[str, Any]:
        assert self.process.stdin is not None
        self.process.stdin.write(text + "\n")
        self.process.stdin.flush()
        return self.read(timeout)

    def read(self, timeout: float) -> dict[str, Any]:
        try:
            line = self.lines.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError(f"no answer within {timeout:.0f}s") from None
        if line is None:
            tail = "\n".join(self.stderr_path.read_text().splitlines()[-15:])
            raise RuntimeError(f"the service exited (code {self.process.poll()}); stderr:\n{tail}")
        return json.loads(line)

    def finish(self) -> int:
        with contextlib.suppress(Exception):
            assert self.process.stdin is not None
            self.process.stdin.close()
        try:
            code = self.process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            self.process.kill()
            code = self.process.wait()
        self.stderr.close()
        return code


def trim(value: Any, limit: int = 120) -> Any:
    """A short rendition of a request or response for the transcript."""
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if key in {"tape", "selection"}:
                out[key] = f"<{key}: {len(json.dumps(item))} chars>"
            elif key in {"payload_f32_b64"}:
                out[key] = f"<{len(item)} base64 chars>"
            else:
                out[key] = trim(item, limit)
        return out
    if isinstance(value, list):
        return [trim(item, limit) for item in value[:6]] + (["..."] if len(value) > 6 else [])
    if isinstance(value, str) and len(value) > limit:
        return value[:limit] + "..."
    return value


class Script:
    """A scripted conversation with checks; every request is recorded."""

    def __init__(self, server: ServeProcess):
        self.server = server
        self.steps: list[dict[str, Any]] = []
        self.failures: list[str] = []
        self.session: str | None = None

    def step(self, name: str, request: dict[str, Any], *, expect_ok: bool = True, check=None):
        started = time.time()
        request = dict(request)
        if (
            self.session is not None
            and "session" not in request
            and request.get("op")
            not in {
                "open",
                "add_files",
                "kb_list",
                "kb_create",
            }
        ):
            request["session"] = self.session
        request["id"] = len(self.steps) + 1
        response = self.server.call(request)
        problem = None
        if response.get("ok") is not expect_ok:
            problem = f"expected ok={expect_ok}, got {trim(response)}"
        elif check is not None:
            try:
                check(response)
            except AssertionError as exc:
                problem = f"check failed: {exc}"
        self.steps.append(
            {
                "name": name,
                "request": trim(request),
                "ok": response.get("ok"),
                "pass": problem is None,
                "problem": problem,
                "seconds": round(time.time() - started, 3),
                "response": trim(response),
            }
        )
        if problem:
            self.failures.append(f"{name}: {problem}")
        return response


def text_output(max_tokens: int = 8, **extra) -> dict[str, Any]:
    return {"track": "assistant", "modality": "text", "max_tokens": max_tokens, **extra}


def check_generation(response: dict[str, Any]) -> None:
    assert response["outputs"], "no outputs"
    out = response["outputs"][0]
    raw = base64.b64decode(out["payload_b64"])
    assert isinstance(out["valid"], bool), "valid flag missing"
    assert isinstance(out["emitted_ids"], list), "emitted ids missing"
    assert math.isfinite(response["log_prob"]), "log density not finite"
    assert len(raw) <= 64 * 8, "output longer than the request allows"
    assert response["tape"], "tape missing"


def drive_service(server: ServeProcess, seed: int, extra_collection: bool) -> dict[str, Any]:
    script = Script(server)

    def opened(response):
        assert response["session"], "no session id"
        script.session = response["session"]

    open_request: dict[str, Any] = {"op": "open", "effort": 1, "metadata": {"task": "chat"}}
    if extra_collection:
        open_request.update({"collections": ["main"], "max_sources": 2})
    script.step("open", open_request, check=opened)
    script.step(
        "observe_text",
        {
            "op": "observe_text",
            "track": "user",
            "role": "user",
            "text": "document 3: the key 24 opens the vault. Which key opens the vault?",
            "effort": 1,
        },
        check=lambda r: r["chunks"] >= 1 or (_ for _ in ()).throw(AssertionError("no chunks")),
    )
    generation = script.step(
        "generate",
        {"op": "generate", "outputs": [text_output(8)], "seed": seed, "effort": 1},
        check=check_generation,
    )
    revision = generation.get("revision")
    script.step("status", {"op": "status"}, check=lambda r: r["events_seen"] > 0 or None)
    script.step("lifecycle_tools", {"op": "lifecycle_tools"})
    created = script.step(
        "track_create",
        {
            "op": "track_action",
            "operation": "create",
            "values": {"purpose": "Check the vault key", "completion": "Key reported", "budget": 8},
        },
        check=lambda r: r["track"]["address"] or None,
    )
    address = created.get("track", {}).get("address")
    if address:
        script.step(
            "track_progress",
            {
                "op": "track_action",
                "operation": "progress",
                "track": address,
                "values": {"progress": "halfway"},
            },
        )
        script.step(
            "track_complete", {"op": "track_action", "operation": "complete", "track": address}
        )
    script.step(
        "speculate",
        {"op": "speculate", "outputs": [text_output(6)], "seed": seed + 1, "effort": 1},
        check=lambda r: r["provisional"] is True or None,
    )
    script.step("accept", {"op": "accept"})
    script.step(
        "speculate_then_cancel",
        {"op": "speculate", "outputs": [text_output(6)], "seed": seed + 2, "effort": 1},
    )
    script.step("cancel", {"op": "cancel"})
    if revision is not None:
        script.step(
            "release_generated", {"op": "release_generation", "selection_revision": revision}
        )
    started = script.step(
        "start_generation",
        {
            "op": "start_generation",
            "outputs": [text_output(6, event_id="async-1")],
            "seed": seed + 3,
            "effort": 1,
        },
    )
    selection = started.get("selection_revision")
    if selection is not None:
        for _attempt in range(60):
            polled = script.server.call(
                {
                    "op": "poll_generation",
                    "session": script.session,
                    "selection_revision": selection,
                }
            )
            if polled.get("status") in {"rendered", "delivered", "failed", "cancelled"}:
                break
            time.sleep(0.2)
        script.steps.append(
            {
                "name": "poll_generation",
                "pass": polled.get("status") == "rendered",
                "ok": polled.get("ok"),
                "problem": None
                if polled.get("status") == "rendered"
                else f"status {polled.get('status')}: {polled.get('error')}",
                "response": trim(polled),
                "request": {"op": "poll_generation", "selection_revision": selection},
                "seconds": 0.0,
            }
        )
        if polled.get("status") != "rendered":
            script.failures.append(f"poll_generation: {polled.get('status')} {polled.get('error')}")
        script.step(
            "delivered",
            {"op": "delivered", "selection_revision": selection, "event_ids": ["async-1"]},
        )
        script.step("release_async", {"op": "release_generation", "selection_revision": selection})
    script.step(
        "observe_raw_event",
        {
            "op": "observe",
            "events": [{"track": "tool", "role": "tool", "modality": "text", "text": "ok: 42"}],
            "effort": 1,
        },
    )
    script.step("generate_again", {"op": "generate", "outputs": [text_output(8)], "seed": seed + 4})
    script.step("kb_list", {"op": "kb_list"}, expect_ok=True)
    script.step(
        "reject_unknown_session",
        {"op": "status", "session": "no-such-session"},
        expect_ok=False,
    )
    script.step("reject_unknown_op", {"op": "frobnicate"}, expect_ok=False)
    script.step(
        "reject_obsolete_sampling",
        {
            "op": "generate",
            "outputs": [text_output(4, temperature=0.7)],
        },
        expect_ok=False,
    )
    bad = script.server.send_raw("this is not json {")
    script.steps.append(
        {
            "name": "malformed_line_does_not_end_service",
            "pass": bad.get("ok") is False,
            "ok": bad.get("ok"),
            "problem": None if bad.get("ok") is False else f"unexpected answer {trim(bad)}",
            "response": trim(bad),
            "request": "<not json>",
            "seconds": 0.0,
        }
    )
    if bad.get("ok") is not False:
        script.failures.append("malformed_line: service answered ok")
    script.step("status_after_errors", {"op": "status"})
    script.step("close", {"op": "close"})
    code = script.server.finish()
    if code != 0:
        script.failures.append(f"service exit code {code}")
    return {
        "requests": len(script.steps),
        "failures": script.failures,
        "exit_code": code,
        "steps": script.steps,
    }


# ---------------------------------------------------------------------------------------------
# the rehearsal
# ---------------------------------------------------------------------------------------------


def rehearsal(args, workdir: Path, report: Report) -> None:
    env = child_env()
    trainer = Trainer(workdir, args, env)
    state: dict[str, Any] = {}
    b0_steps, p1_steps = args.b0_steps, args.p1_steps

    def stage_data():
        built = build_data(workdir, args)
        state["builder"] = built.pop("builder")
        state["roots"] = state["builder"].roots
        return built

    report.run("launch_readiness", lambda: launch_readiness(REPO, tuple(args.audit_config)))
    report.run("build_data", stage_data)

    # -- B0 ------------------------------------------------------------------------------
    def stage_b0():
        config = trainer.base_config("b0", phase="B0", datasets=[str(state["roots"]["text"])])
        state["b0_config"] = config
        result = trainer.run("b0", config, max_steps=b0_steps)
        summary = summarize_training(result, Path(config["output"]))
        if summary["global_step"] != b0_steps:
            raise AssertionError(f"expected step {b0_steps}, got {summary['global_step']}")
        state["b0_result"], state["b0_summary"] = result, summary
        return summary

    report.run("train_b0", stage_b0, needs=("build_data",))

    def stage_resume_b0():
        config = state["b0_config"]
        before = state["b0_summary"]
        checkpoint = str(Path(config["output"]) / "checkpoints" / "latest.json")
        result = trainer.run("b0-resume", config, max_steps=args.resume_steps, resume=checkpoint)
        summary = summarize_training(result, Path(config["output"]))
        expect = b0_steps + args.resume_steps
        problems = []
        if summary["global_step"] != expect:
            problems.append(f"global_step {summary['global_step']} != {expect}")
        if not summary["steps_contiguous"]:
            problems.append(f"logged steps not contiguous: {summary['logged_steps']}")
        for meter in ("I", "D", "P"):
            if summary["counters"].get(meter, 0) <= before["counters"].get(meter, 0):
                problems.append(f"counter {meter} did not grow across the resume")
        if summary["phase_consumed"] <= before["phase_consumed"]:
            problems.append("phase_consumed did not grow across the resume")
        if problems:
            raise AssertionError("; ".join(problems))
        summary["steps_before_resume"] = before["global_step"]
        summary["counters_before_resume"] = before["counters"]
        state["b0_resumed"] = result
        return summary

    report.run("resume_b0", stage_resume_b0, needs=("train_b0",))

    def stage_exactness():
        from megatoken.training.checkpoint import load_checkpoint

        config = dict(state["b0_config"])
        config["output"] = str(workdir / "runs" / "b0-uninterrupted")
        full = trainer.run("b0-full", config, max_steps=b0_steps + args.resume_steps, cli=False)
        left, _ = load_checkpoint(full["checkpoint"], map_location="cpu")
        right, _ = load_checkpoint(state["b0_resumed"]["checkpoint"], map_location="cpu")
        worst = 0.0
        differing: list[str] = []
        for key in (
            "model",
            "optimizer",
            "rank_state",
            "phase_consumed",
            "counters",
            "global_step",
        ):
            w, bad = tree_equal(left[key], right[key], key)
            worst = max(worst, w)
            differing += bad
        if differing:
            raise AssertionError(
                f"resumed run differs from an uninterrupted one (max abs diff {worst}): "
                f"{differing[:6]}"
            )
        return {"max_abs_difference": worst, "steps_compared": b0_steps + args.resume_steps}

    if args.exact_resume:
        report.run("resume_exactness", stage_exactness, needs=("resume_b0",))

    # -- P1 on text (the checkpoint that is exported) ---------------------------------------
    def stage_p1_text():
        config = trainer.base_config(
            "p1-text",
            phase="P1",
            datasets=[str(state["roots"]["text"])],
            representation_mode="recurrent",
        )
        resume = str(Path(state["b0_config"]["output"]) / "checkpoints" / "latest.json")
        result = trainer.run(
            "p1-text", config, max_steps=p1_steps, resume=resume, phase_transition=True
        )
        summary = summarize_training(result, Path(config["output"]))
        state["p1_text_result"] = result
        state["p1_text_config"] = config
        return summary

    report.run("train_p1_text", stage_p1_text, needs=("resume_b0",))

    # -- P1 with the full stage mixture (every stream, every root, fake realizers) ----------------
    def mixture_config():
        packs = workdir / "fake-packs-lock.json"
        # Only the names matter here: the realizer loader is replaced by the fake one.
        names = ("pocket", "sana", "sana_video")
        packs.write_text(json.dumps({"packs": {name: {"fake": True} for name in names}}))
        roots = state["roots"]
        return trainer.base_config(
            "p1-mixture",
            phase="P1",
            datasets=[
                str(roots["text"]),
                str(roots["media"]),
                str(roots["duplex"]),
                str(roots["duplex_heldout"]),
                str(roots["collection"]),
                str(roots["collection_heldout"]),
            ],
            representation_mode="recurrent",
            use_stage_mixture=True,
            packs=str(packs),
            eval_every=args.p1_steps,
            eval_windows=1,
            checkpoint_every=1,
            max_accumulation_windows=args.mixture_windows,
            accumulate_scored=10**9,
        )

    def run_mixture(name: str, **overrides):
        config = {**mixture_config(), **overrides}
        config["output"] = str(workdir / "runs" / name)
        state["mix_config"] = config
        resume = str(Path(state["b0_config"]["output"]) / "checkpoints" / "latest.json")
        result = trainer.run(
            name, config, max_steps=p1_steps, resume=resume, phase_transition=True, media=True
        )
        summary = summarize_training(result, Path(config["output"]))
        summary["data_drawn"] = drawn_streams(result["checkpoint"])
        state["mix_result"] = result
        return config, summary

    def stage_p1_mixture():
        config, summary = run_mixture("p1-mixture")
        needed = {s for s, r in state["builder"].report.items() if r["validation_episodes"]}
        got = set(summary.get("evaluation_streams", {}))
        if summary["evaluations"] == 0:
            raise AssertionError("held-out evaluation never ran")
        if not needed <= got:
            raise AssertionError(
                f"streams with validation data missing from evaluation: {needed - got}"
            )
        summary["evaluation_streams_expected"] = len(needed)
        return summary

    def stage_p1_mixture_no_eval():
        if state.get("mix_result") is not None:
            raise Skip("not needed: the run with held-out evaluation worked")
        _, summary = run_mixture("p1-mixture-noeval", eval_every=0)
        summary["note"] = (
            "ran WITHOUT held-out evaluation because train_p1_mixture failed; "
            "this covers the rest of the stage mixture path"
        )
        return summary

    report.run("train_p1_mixture", stage_p1_mixture, needs=("resume_b0",))
    report.run("train_p1_mixture_no_eval", stage_p1_mixture_no_eval, needs=("resume_b0",))

    def stage_resume_mixture():
        if state.get("mix_result") is None:
            raise Skip("no stage-mixture run succeeded")
        config = state["mix_config"]
        before = summarize_training(state["mix_result"], Path(config["output"]))
        checkpoint = str(Path(config["output"]) / "checkpoints" / "latest.json")
        result = trainer.run(
            "p1-mixture-resume", config, max_steps=1, resume=checkpoint, media=True
        )
        summary = summarize_training(result, Path(config["output"]))
        problems = []
        if summary["global_step"] != before["global_step"] + 1:
            problems.append(f"global_step {summary['global_step']} after {before['global_step']}")
        if not summary["steps_contiguous"]:
            problems.append(f"logged steps not contiguous: {summary['logged_steps']}")
        if summary["phase_consumed"] <= before["phase_consumed"]:
            problems.append("phase_consumed did not grow")
        if problems:
            raise AssertionError("; ".join(problems))
        return summary

    report.run("resume_p1_mixture", stage_resume_mixture)

    # -- export, serve, evaluate ----------------------------------------------------------------
    def stage_export():
        bundle = workdir / "bundle"
        checkpoint = str(Path(state["p1_text_config"]["output"]) / "checkpoints" / "latest.json")
        out = workdir / "logs" / "export.json"
        run_command(
            [PYTHON, "-m", "megatoken", "--json-output", str(out), "export"]
            + ["--checkpoint", checkpoint, "--output", str(bundle), "--codec", "byte"],
            workdir / "logs" / "export.log",
            env=env,
        )
        manifest = json.loads(out.read_text())
        state["bundle"] = bundle
        return {
            "bundle": str(bundle),
            "parameters": manifest["parameters"],
            "model_version": manifest["model_version"],
            "output_modalities": manifest["output_modalities"],
            "bytes": sum(p.stat().st_size for p in bundle.iterdir()),
        }

    report.run("export_bundle", stage_export, needs=("train_p1_text",))

    def stage_serve():
        collection = Path(state["roots"]["collection"]) / "collections" / "main"
        argv = [PYTHON, "-m", "megatoken", "serve", "--model", str(state["bundle"])]
        argv += ["--device", "cpu", "--collection", f"main={collection}"]
        server = ServeProcess(argv, env, workdir / "logs" / "serve.stderr")
        state["serve_command"] = " ".join(argv)
        try:
            result = drive_service(server, args.seed, True)
        finally:
            if server.process.poll() is None:
                server.process.kill()
        result["command"] = state["serve_command"]
        if result["failures"]:
            raise AssertionError("; ".join(result["failures"][:5]))
        return result

    report.run("serve_client", stage_serve, needs=("export_bundle",))

    def stage_add_files():
        from retrieval_support import FakePreparer

        from megatoken.model.multitrack import MultiTrackGenerator
        from megatoken.multitrack_runtime import MultiTrackRuntime
        from megatoken.multitrack_service import MultiTrackService
        from megatoken.served import ServedBundle

        collection = Path(state["roots"]["collection"]) / "collections" / "main"
        scratch = workdir / "add-files-scratch"
        kb_root = workdir / "kb-root"  # files can only be added from inside the server's root
        kb_root.mkdir(parents=True, exist_ok=True)
        new_file = kb_root / "new-fact.txt"
        new_file.write_text("A fact that arrived while serving: the vault key is 24.")
        bundle = ServedBundle(state["bundle"], device="cpu")
        runtime = MultiTrackRuntime(
            MultiTrackGenerator(bundle.model),
            chunking=bundle.chunking,
            collections={"main": collection},
            preparer=lambda: FakePreparer(scratch),
        )
        try:
            service = MultiTrackService(runtime, kb_root=kb_root)
            added = service.handle(
                {"op": "add_files", "collection": "main", "paths": [str(new_file)], "id": 1}
            )
            if not added["ok"]:
                raise AssertionError(f"add_files failed: {added}")
            opened = service.handle(
                {"op": "open", "effort": 1, "collections": ["main"], "max_sources": 3, "id": 2}
            )
            if not opened["ok"]:
                raise AssertionError(f"open after add_files failed: {opened}")
            return {
                "added": added["collection"],
                "retrieved_at_open": len(opened.get("retrieved", [])),
                "note": "the file preparer is a stand-in; the real one needs the media model packs",
            }
        finally:
            runtime.close()

    report.run("serve_add_files", stage_add_files, needs=("export_bundle",))

    def stage_serve_media():
        """Export the stage-mixture checkpoint with its (fake) packs and serve audio/image."""
        if state.get("mix_result") is None:
            raise Skip("no stage-mixture checkpoint to export")
        from media_support import features

        from megatoken.model.multitrack import MultiTrackGenerator
        from megatoken.multitrack_runtime import MultiTrackRuntime
        from megatoken.multitrack_service import MultiTrackService
        from megatoken.served import ServedBundle, export_served
        from megatoken.teacher import PINNED_TEACHER

        packs = state["mix_config"]["packs"]
        bundle_dir = workdir / "bundle-media"
        with fake_realizers():
            manifest = export_served(state["mix_result"]["checkpoint"], bundle_dir, packs=packs)
            bundle = ServedBundle(bundle_dir, packs=packs, device="cpu")
            runtime = MultiTrackRuntime(MultiTrackGenerator(bundle.model), chunking=bundle.chunking)
            try:
                service = MultiTrackService(runtime)
                opened = service.handle({"op": "open", "effort": 1, "id": 1})
                sid = opened["session"]
                tokens = features(5, seed=3)[0:1]
                raw = tokens.detach().float().numpy().astype("<f4").tobytes()
                observed = service.handle(
                    {
                        "op": "observe",
                        "session": sid,
                        "events": [
                            {
                                "track": "camera",
                                "modality": "image",
                                "role": "user",
                                "payload_f32_b64": base64.b64encode(raw).decode(),
                                "shape": list(tokens.shape),
                                "extractor": PINNED_TEACHER,
                            }
                        ],
                        "effort": 1,
                        "id": 2,
                    }
                )
                generated = service.handle(
                    {
                        "op": "generate",
                        "session": sid,
                        "effort": 1,
                        "seed": 5,
                        "outputs": [
                            {"track": "voice", "modality": "audio", "sample_count": 480},
                            {"track": "picture", "modality": "image", "height": 16, "width": 16},
                            {"track": "caption", "modality": "text", "max_tokens": 6},
                        ],
                        "id": 3,
                    }
                )
            finally:
                runtime.close()
        for name, answer in (("open", opened), ("observe", observed), ("generate", generated)):
            if not answer["ok"]:
                raise AssertionError(f"{name} failed: {trim(answer)}")
        shapes = {
            item["modality"]: item.get("shape", len(item.get("payload_b64", "")))
            for item in generated["outputs"]
        }
        return {
            "output_modalities": manifest["output_modalities"],
            "generated_output_shapes": shapes,
            "log_prob": generated["log_prob"],
            "note": "the audio/image realizers are tiny stand-ins: this checks protocol and "
            "export/load with packs, not media quality",
        }

    report.run("serve_media_bundle", stage_serve_media)

    def stage_evaluate():
        out = workdir / "logs" / "evaluate.json"
        checkpoint = str(Path(state["p1_text_config"]["output"]) / "checkpoints" / "latest.json")
        run_command(
            [PYTHON, "-m", "megatoken", "--json-output", str(out), "evaluate"]
            + ["--checkpoint", checkpoint, "--datasets", str(state["roots"]["text"])]
            + ["--episodes", str(args.eval_episodes), "--effort", "1", "--device", "cpu"]
            + ["--seed", str(args.seed)],
            workdir / "logs" / "evaluate.log",
            env=env,
        )
        result = json.loads(out.read_text())
        bad = finite_tree(result)
        if bad:
            raise ValueError(f"evaluate reported non-finite numbers at {bad[:5]}")
        scalars = {k: v for k, v in result.items() if isinstance(v, (int, float, str, bool))}
        return {"keys": sorted(result), "scalars": scalars}

    report.run("evaluate", stage_evaluate, needs=("train_p1_text",))

    def stage_soak():
        out = workdir / "logs" / "soak.json"
        argv = [PYTHON, str(Path(__file__).with_name("probe_serve_soak.py"))]
        argv += ["--bundle", str(state["bundle"]), "--requests", str(args.soak_requests)]
        argv += ["--output", str(out), "--seed", str(args.seed)]
        run_command(argv, workdir / "logs" / "soak.log", env=env, timeout=3600)
        result = json.loads(out.read_text())
        if not result.get("ok", False):
            raise AssertionError(f"soak reported problems: {result.get('problems')}")
        return {
            "requests": result["requests"],
            "without_release": result["without_release"]["summary"],
            "with_release": result["with_release"]["summary"],
            "limit_confirmed": result["finding"]["limit_confirmed"],
            "finding": result["finding"]["text"],
        }

    report.run("serve_soak", stage_soak, needs=("export_bundle",))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0], formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--smoke", action="store_true", help="smallest settings (CPU, ~1-2 min)")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, help="write the full result JSON here")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--model", default="tiny", choices=("tiny", "32x256", "32x512"))
    parser.add_argument("--workdir", type=Path, help="directory for artifacts (default: temporary)")
    parser.add_argument("--keep-workdir", action="store_true", help="do not delete the artifacts")
    parser.add_argument("--readiness-only", action="store_true", help="run only the static audit")
    parser.add_argument(
        "--audit-config",
        type=Path,
        action="append",
        default=[],
        help="also audit this training config (e.g. your filled-in launch config); repeatable",
    )
    parser.add_argument("--b0-steps", type=int)
    parser.add_argument("--resume-steps", type=int)
    parser.add_argument("--p1-steps", type=int)
    parser.add_argument("--episodes-per-stream", type=int)
    parser.add_argument("--validation", choices=("few", "all"))
    parser.add_argument("--eval-episodes", type=int)
    parser.add_argument(
        "--patch-heldout-check",
        action="store_true",
        help="work around the known held-out coverage bug so train_p1_mixture can evaluate",
    )
    parser.add_argument(
        "--mixture-windows",
        type=int,
        help="windows per optimizer step in the stage-mixture run; at least the number of "
        "streams (35 for P1) makes every stream appear in every step",
    )
    parser.add_argument("--soak-requests", type=int, help="generate round trips in the soak")
    parser.add_argument(
        "--real-cli",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="run training through `megatoken train` subprocesses (default: yes unless --smoke)",
    )
    parser.add_argument(
        "--exact-resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="also compare an uninterrupted run with the resumed one",
    )
    args = parser.parse_args(argv)
    small = args.smoke
    defaults = {
        "b0_steps": 1 if small else 4,
        "resume_steps": 1 if small else 2,
        "p1_steps": 1 if small else 4,
        "episodes_per_stream": 1 if small else 2,
        "validation": "few" if small else "all",
        "eval_episodes": 2 if small else 6,
        "mixture_windows": 3 if small else 40,
        "soak_requests": 70 if small else 150,
    }
    for name, value in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    if args.real_cli is None:
        args.real_cli = not small
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(name, "1")
    setup_paths()
    report = Report()
    started = time.time()
    lock = None
    if args.device == "cuda":
        lock = open(GPU_LOCK, "w")  # noqa: SIM115 - held for the whole run
        print(f"waiting for the shared GPU lock {GPU_LOCK} ...", flush=True)
        fcntl.flock(lock, fcntl.LOCK_EX)
    temp = None
    try:
        if args.readiness_only:
            report.run("launch_readiness", lambda: launch_readiness(REPO, tuple(args.audit_config)))
        else:
            if args.workdir:
                workdir = args.workdir
                workdir.mkdir(parents=True, exist_ok=True)
            else:
                temp = tempfile.mkdtemp(prefix="megatoken-rehearsal-")
                workdir = Path(temp)
            print(f"working directory: {workdir}", flush=True)
            rehearsal(args, workdir, report)
    finally:
        if lock is not None:
            fcntl.flock(lock, fcntl.LOCK_UN)
            lock.close()
    failed = [s["name"] for s in report.stages if s["status"] == "fail"]
    skipped = [s["name"] for s in report.stages if s["status"] == "skip"]
    result = {
        "probe": "rehearsal",
        "smoke": args.smoke,
        "device": args.device,
        "seed": args.seed,
        "model": args.model,
        "ok": not failed,
        "failed": failed,
        "skipped": skipped,
        "total_seconds": round(time.time() - started, 1),
        "stages": report.stages,
    }
    readiness = report.by_name.get("launch_readiness", {}).get("detail")
    if readiness:
        print_readiness(readiness)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(clean(result), indent=2, allow_nan=False))
    print(
        f"\nrehearsal {'OK' if not failed else 'FAILED'}: "
        f"{len(report.stages) - len(failed) - len(skipped)} passed, {len(failed)} failed, "
        f"{len(skipped)} skipped, {result['total_seconds']}s"
    )
    if failed:
        for name in failed:
            print(f"  first error in {name}: {report.by_name[name]['error'].splitlines()[0]}")
    if temp and not args.keep_workdir:
        shutil.rmtree(temp, ignore_errors=True)
    elif temp:
        print(f"artifacts kept in {temp}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
