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

from megatoken.contracts import CONVERSATION_PROFILE_VERSION
from megatoken.environments import environment_factory
from megatoken.io import content_hash, read_mapping, sha256_file, write_json

from .adapters import PROFILES
from .chunking import ChunkingConfig, segment_with_report
from .collection import (
    CollectionConfig,
    CollectionLedger,
    Collector,
    OpenAICompatibleTeacher,
    TaskSeed,
    TeacherResponse,
)
from .compression import CompressionPolicy
from .dedup import DedupConfig, DedupStore, reconcile
from .demo import write_demo_source
from .prepare import prepare_sources
from .review import ParserReviewPolicy, ParserReviewQueue
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


def _chunking_config(args: argparse.Namespace) -> ChunkingConfig:
    values = read_mapping(args.chunking_config) if getattr(args, "chunking_config", None) else {}
    for argument, setting in (
        ("chunk_target_tokens", "target_tokens"),
        ("chunk_soft_max_tokens", "soft_max_tokens"),
        ("parser_backend", "parser_backend"),
    ):
        value = getattr(args, argument, None)
        if value is not None:
            values[setting] = value
    return ChunkingConfig(**values)


def _parser_review_policy(path: str | None) -> ParserReviewPolicy:
    values = read_mapping(path) if path else {}
    if "hard_codes" in values:
        values["hard_codes"] = tuple(values["hard_codes"])
    return ParserReviewPolicy(**values)


def _chunks(args: argparse.Namespace) -> dict[str, Any]:
    if args.chunks_command == "coverage":
        from .parsing.registry import coverage_report

        result = coverage_report(validate=not args.catalog_only)
    elif args.chunks_command == "inspect":
        codec = _codec(args.codec)
        policy = _chunking_config(args).with_record_limits(
            max_tokens=args.max_record_tokens, max_bytes=args.max_record_bytes
        )
        payload = Path(args.input).read_bytes()
        report = segment_with_report(
            payload,
            codec,
            policy,
            content_kind=args.content_kind,
            language=args.language,
            filename=args.filename or Path(args.input).name,
            mime_type=args.mime_type,
            locale=args.locale,
            chat_template=args.chat_template,
        )
        result = {
            "source_sha256": sha256_file(args.input),
            "bytes": len(payload),
            "identity": policy.identity(codec),
            **report.to_dict(include_payload=args.include_payload),
        }
    else:
        with ParserReviewQueue(args.queue) as review:
            command = args.review_command
            if command == "list":
                return {
                    "summary": review.summary(),
                    "items": review.list_items(status=args.status, limit=args.limit),
                }
            if command == "export":
                return review.export_tasks(args.output, limit=args.limit)
            if command == "claim":
                return {
                    "task": review.claim(
                        args.owner, sample_id=args.sample, lease_seconds=args.lease_seconds
                    ),
                    "summary": review.summary(),
                }
            if command == "release":
                review.release(args.sample, args.owner)
                return {"released": args.sample, "summary": review.summary()}
            if command == "renew":
                review.renew(args.sample, args.owner, lease_seconds=args.lease_seconds)
                return {"renewed": args.sample, "summary": review.summary()}
            if command == "replay":
                policy = _chunking_config(args) if args.chunking_config else None
                return review.replay(
                    args.sample,
                    _codec(args.codec),
                    owner=args.owner,
                    fix_ref=args.fix_ref,
                    chunking=policy,
                )
            raise ValueError(f"Unknown review command: {command}")
    if args.output:
        write_json(args.output, result)
        if args.chunks_command == "coverage":
            return {
                "output": str(Path(args.output).resolve()),
                "validated": result["validated"],
                "grammar_count": result["grammar_count"],
                "failures": result["failures"],
            }
        return {
            "output": str(Path(args.output).resolve()),
            "bytes": result.get("bytes"),
            "chunks": len(result.get("chunks", ())),
            "diagnostics": len(result.get("diagnostics", ())),
        }
    return result


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
    dataset = prepare_sources(
        manifest,
        args.output,
        _codec(args.codec),
        dedup=config,
        max_record_bytes=args.max_record_bytes,
        max_record_tokens=args.max_record_tokens,
        chunking=_chunking_config(args),
        review=_parser_review_policy(args.parser_review_config),
        shard_bytes=args.shard_mib << 20,
        benchmark_quarantine=quarantine,
        reject_invalid=args.reject_invalid,
        compression=None
        if args.compress_min_bytes is None
        else CompressionPolicy(min_bytes=args.compress_min_bytes),
        teacher_packs=args.teacher_packs,
        progress=_log,
        ingest_only=args.ingest_only,
        decisions=args.decisions,
    )
    if dataset is None:
        return {"output": args.output, "digest": str(Path(args.output) / "digest.sqlite")}
    with dataset:
        return {
            "output": str(dataset.root),
            "fingerprint": dataset.manifest["fingerprint"],
            "codec_fingerprint": dataset.manifest["codec_fingerprint"],
            "counts": dataset.manifest["counts"],
            "census": dataset.manifest["census"],
            "chunking": dataset.manifest["chunking"],
            "chunking_stats": dataset.manifest["chunking_stats"],
            "parser_review": dataset.manifest["parser_review"],
        }


def _reconcile(args: argparse.Namespace) -> dict[str, Any]:
    return reconcile(args.digests, args.output)


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
                outputs = obj.get("outputs_json", obj.get("outputs", []))
                if isinstance(outputs, str):
                    try:
                        outputs = json.loads(outputs)
                    except json.JSONDecodeError as error:
                        raise ValueError("outputs_json must contain valid JSON") from error
                if not isinstance(outputs, list) or any(
                    not isinstance(item, dict) for item in outputs
                ):
                    raise ValueError("outputs/outputs_json must be a JSON list of output objects")
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
                    outputs_json=canonical_json(outputs),
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
    if "chunking" in value:
        value["chunking"] = ChunkingConfig(**value["chunking"])
    return CollectionConfig(**value)


def _collect(args: argparse.Namespace) -> dict[str, Any]:
    config = _collection_config(args)
    codec = _codec(args.codec)
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    if args.max_new_attempts is not None and args.max_new_attempts < 1:
        raise ValueError("max-new-attempts must be positive")
    if (args.student_prefix_steps or args.student_only) and not args.student_bundle:
        raise ValueError("Student prefix execution requires --student-bundle")
    if (args.student_packs or args.student_media_root) and not args.student_bundle:
        raise ValueError("Student media setup requires --student-bundle")
    if bool(args.student_packs) != bool(args.student_media_root):
        raise ValueError("--student-packs and --student-media-root must be supplied together")
    if args.student_only and args.student_prefix_steps:
        raise ValueError("--student-only and --student-prefix-steps are separate collection modes")
    student_steps = (
        0 if args.student_only else args.student_prefix_steps or (1 if args.student_bundle else 0)
    )
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
                "student_manifest_sha256": student_manifest,
                "conversation_profile_version": CONVERSATION_PROFILE_VERSION,
                "student_packs_sha256": sha256_file(args.student_packs)
                if args.student_packs
                else None,
                "student_media_root": str(Path(args.student_media_root).resolve())
                if args.student_media_root
                else None,
                "student_prefix_steps": student_steps,
                "student_only": args.student_only,
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
        media_preparer = None
        if args.student_bundle:
            from .media_prepare import MediaPreparer
            from .student import ModelTeacher

            packs_manifest = args.student_packs
            media_preparer = None
            if args.student_media_root or packs_manifest:
                if not args.student_media_root or not packs_manifest:
                    raise ValueError(
                        "--student-packs and --student-media-root must be supplied together"
                    )
                media_preparer = MediaPreparer(
                    packs_manifest, args.student_media_root, device=args.student_device
                )

            student = ModelTeacher(
                args.student_bundle,
                device=args.student_device,
                effort=args.student_effort,
                packs=packs_manifest,
                media_preparer=media_preparer,
            )
            if student.codec.fingerprint != codec.fingerprint:
                raise ValueError("Student bundle and collection codecs differ")
        teacher = OpenAICompatibleTeacher(
            args.teacher_endpoint,
            args.teacher_model,
            codec,
            token_env=args.token_env,
            timeout=args.timeout,
        )
        factory = environment_factory(
            args.environment, token_env=args.environment_token_env, timeout=args.timeout
        )
        collector = Collector(teacher, factory, codec, ledger, families)
        episodes = collector.run_pilot(
            iter_task_seeds(args.tasks, fixture_tools=fixture),
            student=student,
            student_snapshot=student.snapshot if student else None,
            student_prefix_steps=student_steps,
            student_only=args.student_only,
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
            manifest = ledger.export(
                root / "export", assets=None if media_preparer is None else media_preparer.store
            )
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
    prepare.add_argument("--max-record-bytes", type=int, help="Hard limit (default: the policy's)")
    prepare.add_argument("--max-record-tokens", type=int, help="Hard limit (default: the policy's)")
    prepare.add_argument("--chunk-target-tokens", type=int, help="Packing target (default: 512)")
    prepare.add_argument("--chunk-soft-max-tokens", type=int, help="Soft limit (default: 768)")
    prepare.add_argument("--chunking-config", help="Versioned parser/planner policy JSON or YAML")
    prepare.add_argument("--parser-backend", choices=["tree_sitter", "fallback"])
    prepare.add_argument(
        "--parser-review-config",
        help="Parser review policy JSON or YAML; quarantine is enabled by default",
    )
    prepare.add_argument("--shard-mib", type=int, default=128)
    prepare.add_argument("--dedup-config")
    prepare.add_argument("--benchmark-quarantine")
    prepare.add_argument("--reject-invalid", action="store_true")
    prepare.add_argument(
        "--compress-min-bytes",
        type=int,
        help="Read observed document/tool fields of at least this many bytes as "
        "EmbeddingGemma2 features instead of byte chunks (needs --teacher-packs)",
    )
    prepare.add_argument("--teacher-packs", help="Locked pack manifest with EmbeddingGemma2")
    prepare.add_argument(
        "--ingest-only",
        action="store_true",
        help="Read and deduplicate this shard, write its digest, and stop (see 'reconcile')",
    )
    prepare.add_argument(
        "--decisions",
        help="This shard's decisions-<i>.sqlite from 'reconcile'; finishes an --ingest-only shard",
    )
    prepare.set_defaults(func=_prepare)

    joined = subparsers.add_parser(
        "reconcile",
        help="Join shards prepared with --ingest-only: global duplicates, families and splits",
    )
    joined.add_argument(
        "--digests", nargs="+", required=True, help="Each shard's digest.sqlite, in a fixed order"
    )
    joined.add_argument("--output", required=True, help="Directory for the per-shard decisions")
    joined.set_defaults(func=_reconcile)

    chunks = subparsers.add_parser(
        "chunks", help="Inspect recursive parsing, grammar coverage, and parser repair tasks"
    )
    chunk_commands = chunks.add_subparsers(dest="chunks_command", required=True)
    coverage = chunk_commands.add_parser(
        "coverage", help="Verify locally installed grammars without network access"
    )
    coverage.add_argument("--catalog-only", action="store_true")
    coverage.add_argument("--output")
    coverage.set_defaults(func=_chunks)
    inspect = chunk_commands.add_parser(
        "inspect", help="Explain original byte ranges and the selected chunk plan"
    )
    inspect.add_argument("--input", required=True)
    inspect.add_argument("--codec", default="byte")
    inspect.add_argument(
        "--content-kind",
        default="auto",
        choices=[
            "auto",
            "text",
            "prose",
            "code",
            "binary",
            "markdown",
            "markdown_inline",
            "rst",
            "org",
            "asciidoc",
            "latex",
            "typst",
            "html",
            "xml",
            "vue",
            "svelte",
            "astro",
            "heex",
            "embeddedtemplate",
            "twig",
            "json",
            "yaml",
            "toml",
            "csv",
            "tsv",
            "psv",
        ],
    )
    inspect.add_argument("--language")
    inspect.add_argument("--filename")
    inspect.add_argument("--mime-type")
    inspect.add_argument("--locale")
    inspect.add_argument("--chat-template")
    inspect.add_argument("--chunking-config")
    inspect.add_argument("--chunk-target-tokens", type=int)
    inspect.add_argument("--chunk-soft-max-tokens", type=int)
    inspect.add_argument("--parser-backend", choices=["tree_sitter", "fallback"])
    inspect.add_argument("--max-record-tokens", type=int)
    inspect.add_argument("--max-record-bytes", type=int)
    inspect.add_argument("--include-payload", action="store_true")
    inspect.add_argument("--output")
    inspect.set_defaults(func=_chunks)
    review = chunk_commands.add_parser(
        "review", help="Claim, export, and replay quarantined parser cases"
    )
    review_commands = review.add_subparsers(dest="review_command", required=True)
    for name in ("list", "export", "claim", "renew", "release", "replay"):
        command = review_commands.add_parser(name)
        command.add_argument(
            "--queue",
            required=True,
            help="Preparation's preparation.sqlite or a parser review SQLite ledger",
        )
        if name in {"list", "export"}:
            command.add_argument("--limit", type=int, default=100)
        if name == "list":
            command.add_argument("--status", choices=["pending", "claimed", "verified"])
        if name == "export":
            command.add_argument("--output", required=True)
        if name in {"claim", "renew", "release", "replay"}:
            command.add_argument("--owner", required=True)
            command.add_argument("--sample", required=name != "claim")
        if name in {"claim", "renew"}:
            command.add_argument("--lease-seconds", type=int, default=1800)
        if name == "replay":
            command.add_argument("--codec", required=True)
            command.add_argument(
                "--fix-ref", required=True, help="Commit or PR identifying the parser amendment"
            )
            command.add_argument("--chunking-config")
        command.set_defaults(func=_chunks)

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
    collect.add_argument(
        "--student-only",
        action="store_true",
        help="Run only the frozen student; independently verify its own outcome without teacher repair",
    )
    collect.add_argument("--student-device", default="auto")
    collect.add_argument("--student-effort", type=int, default=6)
    collect.add_argument(
        "--student-packs", help="Locked media pack manifest matching the served student bundle"
    )
    collect.add_argument(
        "--student-media-root", help="Asset directory for the student's collected media records"
    )
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
