"""Thin command adapters over the same streaming preparation/collection APIs."""

from __future__ import annotations

import argparse
import base64
import itertools
import json
import sys
from collections import Counter
from collections.abc import Iterator
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from megatoken.io import content_hash, read_mapping, sha256_file

from .adapters import PROFILES
from .collection import (
    CollectionConfig,
    CollectionLedger,
    Collector,
    OpenAICompatibleTeacher,
    TaskSeed,
    TeacherResponse,
)
from .dedup import DedupConfig, DedupStore
from .demo import write_demo_source
from .prepare import prepare_sources
from .schema import canonical_json
from .sources import (
    TermsReview,
    load_source_manifest,
    pin_hf_source,
    pin_local_source,
    write_source_manifest,
)


def _codec(path: str):
    from megatoken.loading import load_codec

    return load_codec(path)


def _log(value: dict[str, Any]) -> None:
    print(json.dumps(value, sort_keys=True), file=sys.stderr, flush=True)


def _demo(args: argparse.Namespace) -> dict[str, Any]:
    source = write_demo_source(
        args.output, count=args.count, max_record_bytes=args.max_record_bytes, stream=args.stream
    )
    manifest = Path(args.manifest or str(args.output) + ".sources.json")
    write_source_manifest(manifest, (source,))
    return {
        "source": str(Path(args.output).resolve()),
        "sources": str(manifest.resolve()),
        "episodes": args.count,
        "fixture_only": True,
    }


def _prepare(args: argparse.Namespace) -> dict[str, Any]:
    manifest = args.sources
    if args.demo:
        manifest = str(args.demo) + ".sources.json"
        if not Path(manifest).is_file():
            raise ValueError(
                "--demo requires a source made by 'megatoken demo' and its .sources.json sidecar; use sources pin-local for your own corpus"
            )
    config = DedupConfig(**read_mapping(args.dedup_config)) if args.dedup_config else DedupConfig()
    quarantine = read_mapping(args.benchmark_quarantine) if args.benchmark_quarantine else None
    with prepare_sources(
        manifest,
        args.output,
        _codec(args.codec),
        dedup=config,
        max_record_bytes=args.max_record_bytes,
        max_record_tokens=args.max_record_tokens,
        shard_bytes=args.shard_mib << 20,
        benchmark_quarantine=quarantine,
        reject_invalid=args.reject_invalid,
        progress=_log,
    ) as dataset:
        return {
            "output": str(dataset.root),
            "fingerprint": dataset.manifest["fingerprint"],
            "codec_fingerprint": dataset.manifest["codec_fingerprint"],
            "counts": dataset.manifest["counts"],
            "census": dataset.manifest["census"],
        }


def _terms(args: argparse.Namespace) -> TermsReview:
    review = TermsReview(
        approved=args.approve_terms,
        license=args.license,
        license_url=args.license_url,
        reviewed_by=args.reviewed_by,
        reviewed_at=args.reviewed_at,
        notes=args.terms_notes,
    )
    if review.approved:
        review.require_approval()
    return review


def _pin(args: argparse.Namespace) -> dict[str, Any]:
    common = {
        "source_id": args.source_id,
        "adapter": args.adapter,
        "stream": args.stream,
        "terms": _terms(args),
        "max_rows": None if args.all_rows else args.max_rows,
    }
    if args.source_command == "pin-hf":
        source = pin_hf_source(
            args.repository, revision=args.revision, subset=args.subset, split=args.split, **common
        )
    else:
        source = pin_local_source(args.path, **common)
    if args.mapping:
        mapping = read_mapping(args.mapping)
        unexpected = set(mapping) - {"field_map", "options"}
        if unexpected:
            raise ValueError(
                f"Source mapping only supports field_map/options, got {sorted(unexpected)}"
            )
        source = replace(source, **mapping)
    sources = (
        load_source_manifest(args.output) if args.append and Path(args.output).exists() else ()
    )
    if any(previous.source_id == source.source_id for previous in sources):
        raise ValueError("A source with this source_id already exists in the output manifest")
    write_source_manifest(args.output, (*sources, source))
    return {
        "manifest": str(Path(args.output).resolve()),
        "source": asdict(source),
        "metadata_only": args.source_command == "pin-hf",
    }


def iter_task_seeds(path: str | Path, *, fixture_tools: bool = False) -> Iterator[TaskSeed]:
    """Stream explicit task identities and public messages; private state stays private."""
    path = Path(path)
    revision = "sha256:" + sha256_file(path)
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    raise ValueError("Task rows must be JSON objects")
                if ("prompt" in obj) == ("prompt_base64" in obj):
                    raise ValueError("Provide exactly one of prompt or prompt_base64")
                prompt = (
                    obj["prompt"].encode("utf-8")
                    if "prompt" in obj
                    else base64.b64decode(obj["prompt_base64"], validate=True)
                )
                policy = (
                    obj.get("policy", "").encode("utf-8")
                    if "policy_base64" not in obj
                    else base64.b64decode(obj["policy_base64"], validate=True)
                )
                environment = obj.get(
                    "environment_json", canonical_json(obj.get("environment", {}))
                )
                tools = obj.get("tools_json")
                if tools is None:
                    schemas = obj.get("tools")
                    if schemas is None and fixture_tools:
                        from megatoken.environments import FIXTURE_TOOLS

                        schemas = FIXTURE_TOOLS
                    if schemas is None:
                        raise ValueError("Explicit public tools are required for a remote task")
                    tools = canonical_json(schemas)
                yield TaskSeed(
                    seed_id=obj["seed_id"],
                    family_id=obj["family_id"],
                    category=obj["category"],
                    prompt=prompt,
                    environment_json=environment,
                    tools_json=tools,
                    revision=obj.get("revision", revision),
                    ancestor_ids=tuple(obj.get("ancestor_ids", ())),
                    mode=obj.get("mode", "executed"),
                    policy=policy,
                )
            except (KeyError, TypeError, ValueError, AttributeError) as error:
                raise ValueError(f"Task source {path}, line {index}: {error}") from error


def _collection_config(args: argparse.Namespace) -> CollectionConfig:
    value = read_mapping(args.config) if args.config else {}
    for name in (
        "distinct_tasks",
        "attempts_per_task",
        "teacher_output_limit",
        "max_output_per_request",
        "max_requests_per_attempt",
        "max_record_bytes",
        "max_record_tokens",
    ):
        override = getattr(args, name, None)
        if override is not None:
            value[name] = override
    if "seed_mix" in value:
        mix = value["seed_mix"]
        value["seed_mix"] = tuple(mix.items()) if isinstance(mix, dict) else tuple(map(tuple, mix))
    return CollectionConfig(**value)


def _collect(args: argparse.Namespace) -> dict[str, Any]:
    from megatoken.environments import FixtureEnvironment, HTTPEnvironment

    config = _collection_config(args)
    codec = _codec(args.codec)
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    if args.max_new_attempts is not None and args.max_new_attempts < 1:
        raise ValueError("max-new-attempts must be positive")
    if args.student_prefix_steps and not args.student_bundle:
        raise ValueError("Student prefix execution requires --student-bundle")
    student_steps = args.student_prefix_steps or (1 if args.student_bundle else 0)
    if student_steps < 0:
        raise ValueError("student-prefix-steps cannot be negative")
    fixture = args.environment == "fixture"
    quarantine = read_mapping(args.benchmark_quarantine) if args.benchmark_quarantine else None
    student_manifest = (
        sha256_file(Path(args.student_bundle) / "manifest.json") if args.student_bundle else None
    )
    inventory: Counter[str] = Counter()
    registry_config = (
        DedupConfig(**read_mapping(args.dedup_config)) if args.dedup_config else DedupConfig()
    )
    families_path = args.families or root / "families.sqlite"
    with (
        DedupStore(families_path, registry_config) as families,
        CollectionLedger(root / "collection.sqlite", config) as ledger,
    ):
        identity = content_hash(
            {
                "tasks_sha256": sha256_file(args.tasks),
                "codec": codec.fingerprint,
                "teacher_endpoint": args.teacher_endpoint,
                "teacher_model": args.teacher_model,
                "environment": args.environment,
                "student_bundle": str(args.student_bundle or ""),
                "student_manifest_sha256": student_manifest,
                "student_prefix_steps": student_steps,
                "benchmark_quarantine": quarantine,
            }
        )
        old = ledger.connection.execute(
            "SELECT value FROM meta WHERE key='collection_identity'"
        ).fetchone()
        if old and old[0] != identity:
            raise ValueError(
                "Task source, teacher, student, environment or codec changed; start a separately budgeted collection directory"
            )
        with ledger.connection:
            ledger.connection.execute(
                "INSERT OR IGNORE INTO meta VALUES('collection_identity',?)", (identity,)
            )
        # Complete the original seed pass and freeze connected ancestors BEFORE
        # constructing actors or sending a single teacher completion request.
        with families.transaction():
            for task in iter_task_seeds(args.tasks, fixture_tools=fixture):
                if task.category not in dict(config.seed_mix):
                    raise ValueError(
                        f"Task category {task.category!r} is absent from the collection mix"
                    )
                families.register_family((task.family_id, *task.ancestor_ids))
                inventory[task.category] += 1
        if not families.frozen:
            if quarantine:
                families.quarantine_benchmarks(
                    family_keys=quarantine.get("family_keys", ()),
                    content_sha256=quarantine.get("content_sha256", ()),
                )
            families.freeze()
        elif quarantine:
            for key, kind in itertools.chain(
                ((key, "family") for key in quarantine.get("family_keys", ())),
                ((key, "content") for key in quarantine.get("content_sha256", ())),
            ):
                if not families.connection.execute(
                    "SELECT 1 FROM benchmarks WHERE key=? AND kind=?", (key, kind)
                ).fetchone():
                    raise ValueError(
                        "An existing frozen registry does not include the requested benchmark snapshot"
                    )
        for attempt_id in args.recover_attempt:
            ledger.recover_interrupted(attempt_id)
        if args.reconcile_usage:
            with Path(args.reconcile_usage).open(encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        receipt = json.loads(line)
                        ledger.settle(
                            receipt["request_id"],
                            TeacherResponse(
                                provider_input_tokens=receipt.get("provider_input_tokens", 0),
                                provider_output_tokens=receipt.get("provider_output_tokens", 0),
                                provider_cache_read_tokens=receipt.get(
                                    "provider_cache_read_tokens", 0
                                ),
                                provider_cache_write_tokens=receipt.get(
                                    "provider_cache_write_tokens", 0
                                ),
                                request_id=receipt.get("provider_request_id", ""),
                            ),
                            raw_output=receipt["raw_output"],
                        )
        if args.dry_run:
            return {
                "output": str(root.resolve()),
                "dry_run": True,
                "seed_rows_by_category": dict(inventory),
                "config": asdict(config),
                "usage": ledger.meters(),
                "families_frozen": True,
            }
        student = None
        if args.student_bundle:
            from .student import ModelTeacher

            student = ModelTeacher(
                args.student_bundle, device=args.student_device, effort=args.student_effort
            )
            if student.codec.fingerprint != codec.fingerprint:
                raise ValueError("Student bundle and collection codecs differ")
        teacher = OpenAICompatibleTeacher(
            args.teacher_endpoint,
            args.teacher_model,
            codec,
            token_env=args.token_env,
            timeout=args.timeout,
            include_reasoning=config.include_reasoning,
        )
        factory = (
            FixtureEnvironment
            if fixture
            else lambda: HTTPEnvironment(
                args.environment, token_env=args.environment_token_env, timeout=args.timeout
            )
        )
        collector = Collector(teacher, factory, codec, ledger, families)
        episodes = collector.run_pilot(
            iter_task_seeds(args.tasks, fixture_tools=fixture),
            student=student,
            student_snapshot=student.snapshot if student else None,
            student_prefix_steps=student_steps,
        )
        limited = (
            itertools.islice(episodes, args.max_new_attempts) if args.max_new_attempts else episodes
        )
        completed = 0
        try:
            for episode in limited:
                completed += 1
                _log(
                    {
                        "episode_id": episode.episode_id,
                        "outcome": episode.supervision.outcome,
                        "verified": episode.supervision.verified_success,
                        "T": ledger.meters()["T"],
                    }
                )
        finally:
            manifest = ledger.export(root / "export")
        return {
            "output": str(root.resolve()),
            "sources": str(manifest.resolve()),
            "completed_attempts_this_run": completed,
            "usage": ledger.meters(),
        }


def add_data_commands(subparsers: argparse._SubParsersAction) -> None:
    demo = subparsers.add_parser("demo", help="Write a deterministic executed smoke corpus")
    demo.add_argument("--output", required=True)
    demo.add_argument("--manifest")
    demo.add_argument("--count", type=int, default=32)
    demo.add_argument("--max-record-bytes", type=int, default=128)
    demo.add_argument("--stream", default="bootstrap")
    demo.set_defaults(func=_demo)

    prepare = subparsers.add_parser(
        "prepare", help="Stream pinned sources into causally valid indexed shards"
    )
    inputs = prepare.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--sources")
    inputs.add_argument("--demo")
    prepare.add_argument("--output", required=True)
    prepare.add_argument(
        "--codec",
        default="byte",
        help="Pinned codec directory/tokenizer JSON, or byte for smoke tests",
    )
    prepare.add_argument("--max-record-bytes", type=int, default=16384)
    prepare.add_argument("--max-record-tokens", type=int, default=4096)
    prepare.add_argument("--shard-mib", type=int, default=128)
    prepare.add_argument("--dedup-config")
    prepare.add_argument("--benchmark-quarantine")
    prepare.add_argument("--reject-invalid", action="store_true")
    prepare.set_defaults(func=_prepare)

    sources = subparsers.add_parser(
        "sources", help="Pin source files/revisions without accepting their terms"
    )
    source_commands = sources.add_subparsers(dest="source_command", required=True)
    for name in ("pin-local", "pin-hf"):
        command = source_commands.add_parser(name)
        command.add_argument("path" if name == "pin-local" else "repository")
        command.add_argument("--output", required=True)
        command.add_argument("--source-id", required=True)
        command.add_argument("--adapter", required=True, choices=sorted(PROFILES))
        command.add_argument("--stream", required=True)
        command.add_argument("--max-rows", type=int, default=1000)
        command.add_argument("--all-rows", action="store_true")
        command.add_argument("--mapping", help="JSON/YAML with explicit field_map and options")
        command.add_argument("--append", action="store_true")
        command.add_argument("--approve-terms", action="store_true")
        command.add_argument("--license", default="unreviewed")
        command.add_argument("--license-url", default="")
        command.add_argument("--reviewed-by", default="")
        command.add_argument("--reviewed-at", default="")
        command.add_argument("--terms-notes", default="")
        if name == "pin-hf":
            command.add_argument("--revision", default="main")
            command.add_argument("--subset")
            command.add_argument("--split", default="train")
        command.set_defaults(func=_pin)

    collect = subparsers.add_parser(
        "collect", help="Execute a bounded teacher pilot or continue its durable ledger"
    )
    collect.add_argument("--tasks", required=True)
    collect.add_argument("--output", required=True)
    collect.add_argument("--teacher-endpoint", required=True, help="Explicit Chat Completions URL")
    collect.add_argument("--teacher-model", required=True)
    collect.add_argument("--codec", default="byte")
    collect.add_argument("--token-env")
    collect.add_argument(
        "--environment", default="fixture", help="fixture or an executed HTTP environment URL"
    )
    collect.add_argument("--environment-token-env")
    collect.add_argument("--timeout", type=float, default=60)
    collect.add_argument("--config")
    collect.add_argument("--families", help="Existing frozen preparation.sqlite, or a new registry")
    collect.add_argument("--dedup-config", help="Must match an existing family registry")
    collect.add_argument("--benchmark-quarantine")
    collect.add_argument(
        "--dry-run", action="store_true", help="Validate and freeze seeds without calling a teacher"
    )
    collect.add_argument("--max-new-attempts", type=int)
    collect.add_argument(
        "--recover-attempt",
        action="append",
        default=[],
        help="Mark a stopped worker's attempt interrupted; reservations stay charged",
    )
    collect.add_argument(
        "--reconcile-usage", help="JSONL with request_id/raw_output from known provider receipts"
    )
    collect.add_argument(
        "--student-bundle",
        help="Checksum-verified frozen served bundle for actual student-state corrections",
    )
    collect.add_argument("--student-prefix-steps", type=int, default=0)
    collect.add_argument("--student-device", default="auto")
    collect.add_argument("--student-effort", type=int, default=6)
    for option in (
        "distinct-tasks",
        "attempts-per-task",
        "teacher-output-limit",
        "max-output-per-request",
        "max-requests-per-attempt",
        "max-record-bytes",
        "max-record-tokens",
    ):
        collect.add_argument("--" + option, type=int)
    collect.set_defaults(func=_collect)
