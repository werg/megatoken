"""Small, lazy CLI over the same public APIs used in Python and tests."""

from __future__ import annotations

import argparse
import base64
import fnmatch
import json
import sys
from dataclasses import asdict
from pathlib import Path

from megatoken import __version__
from megatoken.io import write_json


def _doctor(args):
    from megatoken.runtime import doctor

    result = doctor(require_cuda=args.require_cuda, require_blackwell=args.require_blackwell)
    if not result["ok"]:
        print(json.dumps(result, indent=2), file=sys.stderr)
        raise RuntimeError("Runtime preflight failed")
    return result


def _codec(args):
    from megatoken.assets import fetch_codec

    return fetch_codec(args.output, local_tokenizer=args.local_tokenizer)


def _cache(args):
    from megatoken.cache import main as cache_main

    code = cache_main(args.rest)  # prints its own result
    if code:
        raise SystemExit(code)


def _inspect(args):
    import torch

    from megatoken.loading import build_model, load_codec, load_model_config, parameter_report
    from megatoken.model.audit import audit_parameters

    config = load_model_config(args.model)
    codec = load_codec(args.codec, vocab_size=config.vocab_size)
    with torch.device("meta"):
        model = build_model(config, codec, device="meta", precision="bf16")
    audit_parameters(model)  # Check distinct owners before publishing the census.
    result = {
        "config": config.to_dict(),
        "codec_fingerprint": codec.fingerprint,
        **parameter_report(model),
    }
    if not args.owners:
        result.pop("owners")
    return result


def _recipes(args):
    from megatoken.data import load_recipes, stage_recipe

    return asdict(stage_recipe(args.phase)) if args.phase else load_recipes()


def _train(args):
    from megatoken.training.config import TrainConfig
    from megatoken.training.engine import train

    overrides = {
        name: getattr(args, name)
        for name in (
            "phase",
            "model",
            "datasets",
            "output",
            "codec",
            "device",
            "precision",
            "max_steps",
            "budget",
            "generated_capacity",
            "microbatch_episodes",
            "require_blackwell",
            "compile",
            "expert_backend",
            "representation_mode",
        )
        if getattr(args, name) is not None
    }
    config = TrainConfig.load(args.config, **overrides)
    return train(config, resume=args.resume, phase_transition=args.phase_transition)


def _diloco_initialize(args):
    """Create immutable coordinator bootstrap from the exact train architecture."""
    import json

    import torch

    from megatoken.data import PreparedCorpus
    from megatoken.data.chunking import ChunkingConfig
    from megatoken.io import content_hash, sha256_file
    from megatoken.loading import build_model, load_codec, load_model_config, restore_weights
    from megatoken.runtime import seed_everything
    from megatoken.training.checkpoint import load_checkpoint, resolve_checkpoint
    from megatoken.training.config import TrainConfig
    from megatoken.training.diloco_protocol import training_identity
    from megatoken.training.diloco_server import initialize_coordinator
    from megatoken.training.transducer import HindsightSchedule
    from megatoken.training.transducer_sync import load_transducer_state

    config = TrainConfig.load(
        args.config,
        expand_data=not bool(getattr(args, "corpus_metadata", None)),
    )
    if config.decoupled is None:
        raise ValueError("Set decoupled settings in the training config before initializing")
    seed_everything(config.seed, config.deterministic)
    overrides = {
        **config.model_overrides,
        "expert_backend": config.expert_backend,
        "checkpoint_stages": config.gradient_checkpointing,
    }
    model_config = load_model_config(config.model, **overrides)
    codec = load_codec(config.codec, vocab_size=model_config.vocab_size)
    model = build_model(
        model_config,
        codec,
        device=torch.device("cpu"),
        precision=config.precision,
        packs=config.packs,
    )
    weights: dict = {}
    transducer_state = None
    if args.resume:
        # A new coordinator is the DiLoCo phase transition: it may install
        # realizers the previous phase lacked, whose masters start from init.
        payload, _ = load_checkpoint(args.resume, map_location="cpu")
        if payload["model_config"] != model_config.to_dict():
            raise ValueError("Bootstrap checkpoint model architecture changed")
        if payload["codec_fingerprint"] != codec.fingerprint:
            raise ValueError("Bootstrap checkpoint codec differs")
        restore_weights(
            model,
            payload,
            sha256_file(config.packs) if config.packs else None,
            phase_transition=True,
        )
        # A coordinator export carries the global FP32 masters; a local
        # training checkpoint carries its optimizer's.
        weights = (
            payload["master_weights"]
            if payload.get("export_only")
            else payload["optimizer"]["master_weights"]
        )
        # The previous phase's bank seeds this coordinator's canonical bank.
        transducer_state = load_transducer_state(resolve_checkpoint(args.resume))
    elif config.engram_import or config.engram_imports:
        from megatoken.assets import install_engram_asset

        paths = [config.engram_import] if config.engram_import else config.engram_imports
        for path in paths:
            install_engram_asset(model, path)
    masters = {
        # Owners new in this phase start from their initialization.
        name: weights.get(name, parameter).detach().cpu().float()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if getattr(args, "corpus_metadata", None):
        corpus = json.loads(Path(args.corpus_metadata).read_text())
        roots = corpus.get("roots")
        if not isinstance(roots, list) or not roots:
            raise ValueError("Coordinator corpus metadata has no prepared roots")
        fingerprints = [item.get("fingerprint") for item in roots if isinstance(item, dict)]
        names = [item.get("name") for item in roots if isinstance(item, dict)]
        if (
            len(fingerprints) != len(roots)
            or len(set(fingerprints)) != len(roots)
            or len(set(names)) != len(roots)
        ):
            raise ValueError("Coordinator corpus metadata has invalid root fingerprints")
        if content_hash(fingerprints) != corpus.get("fingerprint"):
            raise ValueError("Coordinator corpus fingerprint differs from its roots")
        if corpus.get("codec_fingerprint") != codec.fingerprint:
            raise ValueError("Coordinator corpus codec differs from the training config")
        chunking_identity = ChunkingConfig.from_identity(corpus.get("chunking"), codec).identity(
            codec
        )
        if not isinstance(config.datasets, list) or not config.datasets:
            raise ValueError("Coordinator config must name its selected prepared roots")
        selected = {}
        for pattern in config.datasets:
            if not isinstance(pattern, str):
                raise ValueError("Coordinator dataset selectors must be prepared-root paths")
            matches = {
                item["name"]: item
                for item in roots
                if fnmatch.fnmatchcase(str(Path("/work/data/prepared") / item["name"]), pattern)
                or fnmatch.fnmatchcase(item["name"], pattern)
            }
            if not matches:
                raise ValueError(
                    f"Coordinator dataset selector {pattern!r} matches no roots in the verified pin"
                )
            selected.update(matches)
        if not selected:
            raise ValueError("Coordinator corpus metadata selected no prepared roots")
        selected_items = sorted(selected.values(), key=lambda item: item["fingerprint"])
        if len({item["fingerprint"] for item in selected_items}) != len(selected_items):
            raise ValueError("Selected roots contain duplicate manifest fingerprints")
        dataset_fingerprint = content_hash([item["fingerprint"] for item in selected_items])
        # PreparedCorpus rejects roots produced by different media teachers.
        if (
            len({extractor for item in selected_items for extractor in item.get("extractors", [])})
            > 1
        ):
            raise ValueError("Selected prepared roots use different feature teachers")
    else:
        with PreparedCorpus(config.datasets) as prepared:
            dataset_fingerprint = prepared.identity
            chunking_identity = prepared.chunking
    return initialize_coordinator(
        args.root,
        run_id=config.decoupled.run_id,
        model_state=model.state_dict(),
        masters=masters,
        model_config=model_config.to_dict(),
        codec_fingerprint=codec.fingerprint,
        train_identity=training_identity(config),
        dataset_fingerprint=dataset_fingerprint,
        chunking_identity=chunking_identity,
        workers=config.decoupled.workers,
        min_quorum=args.quorum,
        fragment_bytes=args.fragment_mib * 1024 * 1024,
        outer_lr=args.outer_lr,
        outer_momentum=args.outer_momentum,
        grace_seconds=args.grace_seconds,
        max_staleness=args.max_staleness,
        token_budget=config.schedule.budget,
        transducer={
            "geometry": dict(vars(model_config.transducer_geometry)),
            "evidence_from": HindsightSchedule().warmup_steps,
        },
        transducer_state=transducer_state,
    )


def _diloco_serve(args):
    import os
    import signal
    import sys

    from megatoken.training.diloco_server import serve_coordinator

    # Exit through the server's cleanup, which saves the canonical bank.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    serve_coordinator(
        args.root,
        args.host,
        args.port,
        token=os.environ.get("MEGATOKEN_DILOCO_TOKEN", ""),
    )


def _diloco_export(args):
    from megatoken.training.config import TrainConfig
    from megatoken.training.diloco_export import export_coordinator

    return export_coordinator(
        args.root, args.output, TrainConfig.load(args.config, expand_data=False)
    )


def _pipeline(args):
    from megatoken.training.checkpoint import load_checkpoint
    from megatoken.training.config import TrainConfig
    from megatoken.training.engine import train

    previous = args.resume
    configs = [TrainConfig.load(filename, device=args.device) for filename in args.configs]
    names = [config.phase for config in configs]
    if len(set(names)) != len(names):
        raise ValueError("Each phase may appear only once in a pipeline")
    previous_phase = None
    if previous:
        payload, _ = load_checkpoint(previous)
        previous_phase = payload["train_config"]["phase"]
        if previous_phase in names:
            configs = configs[names.index(previous_phase) :]
    result = []
    for config in configs:
        step = train(
            config,
            resume=previous,
            phase_transition=previous is not None and previous_phase != config.phase,
        )
        result.append(step)
        previous = step["checkpoint"]
        previous_phase = config.phase
        if not step["complete"]:
            return {
                "complete": False,
                "stages": result,
                "resume": previous,
                "note": "Phase paused at its update limit or signal; resume it before transitioning.",
            }
    return {"complete": True, "stages": result}


def _evaluate(args):
    import torch

    from megatoken.data import PreparedCorpus
    from megatoken.data.chunking import ChunkingConfig
    from megatoken.loading import load_trained
    from megatoken.runtime import seed_everything
    from megatoken.training.config import TrainConfig
    from megatoken.training.evaluation import evaluate_records, evaluate_served_records

    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    )
    trained = load_trained(
        args.checkpoint, device=device, codec_path=args.codec, packs=getattr(args, "packs", None)
    )
    seed_everything(args.seed)
    train_settings = {**trained.payload["train_config"], "effort": args.effort}
    if args.datasets:
        train_settings["datasets"] = args.datasets
    config = TrainConfig.from_dict(train_settings)
    dataset = PreparedCorpus(config.datasets, split=args.split)
    try:
        if args.core_only:
            if args.collection or args.base or getattr(args, "output_specs", None):
                raise ValueError(
                    "Mounted collections and knowledge bases require served evaluation"
                )
            return evaluate_records(
                trained.model, dataset, trained.codec, config, device, episodes=args.episodes
            )
        collections = _named_paths(args.collection, "--collection")
        bases = _named_paths(args.base, "--base")
        capabilities = {}
        if getattr(args, "output_specs", None):
            from megatoken.multitrack_service import _output

            declared = json.loads(Path(args.output_specs).read_text())
            if not isinstance(declared, dict):
                raise ValueError("Public output specifications must map modality to a request")
            capabilities["output_specs"] = {
                kind: _output({**spec, "modality": kind}, trained.model.config.max_capacity, [])[0]
                for kind, spec in declared.items()
            }
        return evaluate_served_records(
            trained.model,
            dataset,
            config,
            device,
            episodes=args.episodes,
            seed=args.seed,
            collections=collections,
            bases=bases,
            trained_weight=args.trained_weight,
            max_sources=args.max_sources,
            threshold=args.threshold,
            chunking=ChunkingConfig.from_identity(trained.payload["chunking"], trained.codec),
            **capabilities,
        )
    finally:
        dataset.close()


def _named_paths(values, option):
    result = {}
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or not name or not path or name in result:
            raise ValueError(f"{option} entries must be unique NAME=PATH values")
        result[name] = path
    return result


def _kb_build(args):
    from megatoken.kb import build

    return build(args)


def _kb_inspect(args):
    from megatoken.kb import inspect

    return inspect(args)


def _benchmark(args):
    from megatoken.benchmark import benchmark

    if args.packs and not args.bundle:
        raise ValueError("--packs is used with the verified --bundle")
    if args.backward and args.model_only:
        raise ValueError("--model-only is an inference benchmark, not a training measurement")
    if (args.backward or args.core_only or args.model_only) and (args.collection or args.base):
        raise ValueError("Mounted collections and knowledge bases require served-session inference")
    if (args.backward or args.core_only or args.model_only) and args.bundle:
        raise ValueError("--bundle applies to served-session inference, not this diagnostic")
    if not args.backward and not args.core_only and not args.model_only:
        from megatoken.benchmark import benchmark_served_session

        return benchmark_served_session(
            config=args.model,
            codec=args.codec,
            bundle=args.bundle,
            packs=args.packs,
            device=args.device,
            steps=args.steps,
            warmup=args.warmup,
            batch=args.batch,
            record_bytes=args.record_bytes,
            capacity=args.capacity,
            history_records=args.history_records,
            backend=args.backend,
            precision=args.precision,
            compile_mode=args.compile_mode,
            effort=args.effort,
            seed=17,
            collections=_named_paths(args.collection, "--collection"),
            bases=_named_paths(args.base, "--base"),
            trained_weight=args.trained_weight,
            max_sources=args.max_sources,
            threshold=args.threshold,
            profile=args.profile,
        )

    return benchmark(
        config=args.model,
        codec=args.codec,
        device=args.device,
        steps=args.steps,
        warmup=args.warmup,
        batch=args.batch,
        record_bytes=args.record_bytes,
        capacity=args.capacity,
        backward=args.backward,
        backend=args.backend,
        compile_mode=args.compile_mode,
        precision=args.precision,
        history_records=args.history_records,
        profile=args.profile,
        core_only=args.core_only,
        effort=args.effort,
        representation_mode=args.representation_mode,
        prediction_chunks=args.prediction_chunks,
    )


def _export(args):
    from megatoken.served import export_served

    return export_served(args.checkpoint, args.output, codec_path=args.codec, packs=args.packs)


def _runtime(args, *, preparer=None):
    """The one serving runtime over a served bundle, with its mounted collections."""
    from megatoken.model.multitrack import MultiTrackGenerator
    from megatoken.multitrack_runtime import MultiTrackRuntime
    from megatoken.served import ServedBundle

    collections = dict(item.partition("=")[::2] for item in args.collection)
    if not all(collections.values()):
        raise ValueError("--collection is NAME=DIRECTORY")
    bundle = ServedBundle(args.model, packs=args.packs, device=args.device, compile=args.compile)
    runtime = MultiTrackRuntime(
        MultiTrackGenerator(bundle.model),
        chunking=bundle.chunking,
        collections=collections,
        preparer=preparer,
        memory_root=getattr(args, "kb_root", None),
    )
    return bundle, runtime


def _generate(args):
    """One-shot session: a prompt as one user message, then one assistant answer."""
    from megatoken.controls import build_control_payload
    from megatoken.multitrack_service import MultiTrackService

    bundle, runtime = _runtime(args)
    bases = {}
    for item in args.base:
        name, _, path = item.partition("=")
        if not path:
            raise ValueError("--base is NAME=FILE")
        runtime.memory.open_base(name, path)
        bases[name] = 1.0
    service = MultiTrackService(runtime)
    metadata = json.loads(args.metadata)

    retrieved = []

    def call(**request):
        answer = service.handle(request)
        if not answer["ok"]:
            raise RuntimeError(answer["error"])
        retrieved.extend(answer.pop("retrieved", ()))
        return answer

    opened = call(op="open", metadata=metadata, bases=bases, effort=args.effort)
    session = opened["session"]
    payload = Path(args.prompt_file).read_bytes() if args.prompt_file else args.prompt.encode()
    call(
        op="observe_text",
        session=session,
        track="assistant",
        role="user",
        payload_b64=base64.b64encode(payload).decode(),
        effort=args.effort,
    )
    output = {
        "track": "assistant",
        "modality": "text",
        "kind": args.kind,
        "min_tokens": args.min_tokens,
        "max_tokens": min(512, bundle.config.max_capacity)
        if args.max_tokens is None
        else args.max_tokens,
    }
    tools_json = Path(args.tools).read_text(encoding="utf-8") if args.tools else None
    if tools_json is not None:
        output["tools_json"] = tools_json
    # Raw control files may hold any bytes, so the control is framed here and sent whole.
    output["control_b64"] = base64.b64encode(
        build_control_payload(
            seed=Path(args.control_seed_file).read_bytes() if args.control_seed_file else b"",
            reference=(
                Path(args.control_reference_file).read_bytes()
                if args.control_reference_file
                else b""
            ),
            reference_origin=args.reference_origin,
            tools_json="[]" if tools_json is None else tools_json,
        )
    ).decode()
    answer = call(
        op="generate", session=session, outputs=[output], effort=args.effort, seed=args.seed
    )
    (reply,) = answer.pop("outputs")
    # Everything the session retrieved on the way, lookup by lookup.
    return {**reply, **answer, **({"retrieved": retrieved} if retrieved else {})}


def _serve(args):
    import tempfile

    from megatoken.multitrack_service import serve_multitrack_jsonl

    scratch = tempfile.TemporaryDirectory()
    runtime = None

    def preparer():
        from megatoken.data.media_prepare import MediaPreparer

        if not args.packs:
            raise ValueError("Adding files needs the pack manifest (--packs)")
        # Prepared features of a knowledge base's files are kept beside it; a collection
        # copies what it keeps, so its files are prepared in scratch space.
        place = Path(args.kb_root) / "prepared" if args.kb_root else Path(scratch.name)
        return MediaPreparer(args.packs, place, device=str(bundle.device))

    try:
        bundle, runtime = _runtime(
            args, preparer=preparer if args.packs and (args.collection or args.kb_root) else None
        )
        serve_multitrack_jsonl(
            runtime, sys.stdin, sys.stdout, max_sessions=args.max_sessions, kb_root=args.kb_root
        )
    finally:
        try:
            if runtime is not None:
                runtime.close()
        finally:
            scratch.cleanup()


def _engram_extract(args):
    from megatoken.assets import extract_engram

    return extract_engram(
        args.embedding_file,
        args.tensor_name,
        args.donor_codec,
        args.output,
        donor_profile=args.donor_profile,
        donor_root=args.donor_root,
        model=args.model,
        seed=args.seed,
    )


def _engram_bake(args):
    from megatoken.assets import bake_checkpoint

    return bake_checkpoint(args.checkpoint, args.output, codec_path=args.codec, packs=args.packs)


def _engram_native_extract(args):
    from megatoken.native_engram import extract_native_engram

    return extract_native_engram(
        args.donor_profile,
        args.donor_root,
        args.corpus,
        args.output,
        model=args.model,
        max_contexts=args.max_contexts,
        seed=args.seed,
        split=args.split,
        cache_dir=args.cache_dir,
        source_mode=args.source_mode,
    )


def _serving_arguments(item):
    item.add_argument("--model", required=True, help="Exported served bundle")
    item.add_argument("--packs", help="Locked realizer pack manifest the model was trained with")
    item.add_argument("--device", default="auto")
    item.add_argument("--compile", action="store_true")
    item.add_argument(
        "--collection",
        action="append",
        default=[],
        metavar="NAME=DIRECTORY",
        help="Mount a source collection (repeatable); when serving with --packs, "
        "files can be added to it",
    )


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="megatoken", description="Prepare, train, evaluate, and serve causal faceted models."
    )
    root.add_argument("--version", action="version", version=__version__)
    root.add_argument("--debug", action="store_true", help="Show Python tracebacks on errors")
    root.add_argument("--json-output", help="Also save the command's final JSON result")
    commands = root.add_subparsers(dest="command", required=True)
    from megatoken.data.cli import add_data_commands

    add_data_commands(commands)
    from megatoken.retrieval.cli import add_collection_commands

    add_collection_commands(commands)
    item = commands.add_parser(
        "doctor", help="Validate the installed runtime and real device execution"
    )
    item.add_argument("--require-cuda", action="store_true")
    item.add_argument("--require-blackwell", action="store_true")
    item.set_defaults(func=_doctor)
    item = commands.add_parser("codec", help="Fetch and verify the pinned raw byte codec")
    item.add_argument("--output", default="artifacts/codec")
    item.add_argument(
        "--local-tokenizer",
        help="Use a previously downloaded copy; the pinned hash is still required",
    )
    item.set_defaults(func=_codec)
    item = commands.add_parser(
        "cache",
        help="The key of the encrypted part of the published data: keygen (see docs/data.md)",
        add_help=False,
    )
    item.add_argument("rest", nargs=argparse.REMAINDER)
    item.set_defaults(func=_cache)
    item = commands.add_parser(
        "inspect", help="Audit actual parameter owners without allocating model storage"
    )
    item.add_argument("--model", default="tiny")
    item.add_argument("--codec", default="byte")
    item.add_argument("--owners", action="store_true")
    item.set_defaults(func=_inspect)
    item = commands.add_parser(
        "recipes", help="Print canonical data budgets, mixtures, and curriculum recommendations"
    )
    item.add_argument("--phase")
    item.set_defaults(func=_recipes)
    item = commands.add_parser(
        "train", help="Run one supervised stage, under torchrun when distributed"
    )
    item.add_argument("--config", required=True)
    for name in (
        "phase",
        "model",
        "output",
        "codec",
        "device",
        "precision",
        "expert-backend",
        "representation-mode",
    ):
        item.add_argument("--" + name, type=str)
    item.add_argument("--datasets", nargs="+", help="Prepared roots, read as one corpus")
    for name in ("max-steps", "budget", "generated-capacity", "microbatch-episodes"):
        item.add_argument("--" + name, type=int)
    item.add_argument("--require-blackwell", action=argparse.BooleanOptionalAction, default=None)
    item.add_argument("--compile", action=argparse.BooleanOptionalAction, default=None)
    item.add_argument("--resume", help="Committed checkpoint directory or checkpoints/latest.json")
    item.add_argument(
        "--phase-transition",
        action="store_true",
        help="Preserve optimizer state, reset the new phase's data cursor and counters",
    )
    item.set_defaults(func=_train)
    group = commands.add_parser(
        "diloco", help="Initialize or serve a resilient asynchronous DiLoCo coordinator"
    )
    sub = group.add_subparsers(dest="diloco_command", required=True)
    initial = sub.add_parser("init", help="Initialize persistent global weights and fragments")
    initial.add_argument("--config", required=True)
    initial.add_argument("--root", required=True)
    initial.add_argument(
        "--resume",
        help="Start this phase from the previous phase's coordinator export or checkpoint",
    )
    initial.add_argument(
        "--corpus-metadata",
        help="Verified compact prepared-data identity; avoids fetching a stored corpus on coordinator",
    )
    initial.add_argument("--quorum", type=int, default=4)
    initial.add_argument("--fragment-mib", type=int, default=64)
    initial.add_argument("--outer-lr", type=float, default=0.7)
    initial.add_argument("--outer-momentum", type=float, default=0.9)
    initial.add_argument("--grace-seconds", type=float, default=8.0)
    initial.add_argument("--max-staleness", type=int, default=2)
    initial.set_defaults(func=_diloco_initialize)
    serving = sub.add_parser("serve", help="Start the persistent authenticated HTTP syncer")
    serving.add_argument("--root", required=True)
    serving.add_argument("--host", default="127.0.0.1")
    serving.add_argument("--port", type=int, default=8823)
    serving.set_defaults(func=_diloco_serve)
    exporting = sub.add_parser(
        "export",
        help="Export a stopped syncer's global model (evaluation and the next phase's seed)",
    )
    exporting.add_argument("--root", required=True)
    exporting.add_argument("--output", required=True)
    exporting.add_argument("--config", required=True)
    exporting.set_defaults(func=_diloco_export)
    item = commands.add_parser(
        "pipeline", help="Run complete supervised phases in order; pause when a phase is incomplete"
    )
    item.add_argument("--configs", nargs="+", required=True)
    item.add_argument("--resume")
    item.add_argument("--device")
    item.set_defaults(func=_pipeline)
    item = commands.add_parser(
        "evaluate", help="Fixed-holdout generated samples through served sessions"
    )
    item.add_argument("--checkpoint", required=True)
    item.add_argument("--datasets", nargs="+")
    item.add_argument("--codec")
    item.add_argument("--packs", help="Relocate the checkpoint's matching media pack manifest")
    item.add_argument(
        "--output-specs",
        help="JSON of public per-modality output bounds, independent of held-out targets",
    )
    item.add_argument("--device", default="auto")
    item.add_argument("--split", choices=("validation", "test"), default="validation")
    item.add_argument("--episodes", type=int, default=16)
    item.add_argument("--effort", type=int, choices=(1, 2, 3, 6), default=6)
    item.add_argument("--seed", type=int, default=17)
    item.add_argument(
        "--core-only",
        action="store_true",
        help="Run the lower-level model-only loss and sampling diagnostic without session memory",
    )
    item.add_argument("--collection", action="append", default=[], metavar="NAME=DIRECTORY")
    item.add_argument("--base", action="append", default=[], metavar="NAME=FILE")
    item.add_argument("--trained-weight", type=float, default=1.0)
    item.add_argument("--max-sources", type=int, default=2)
    item.add_argument("--threshold", type=float)
    item.set_defaults(func=_evaluate)
    item = commands.add_parser(
        "benchmark", help="Measure real end-to-end work, memory, and dispatch"
    )
    item.add_argument("--model", default="tiny")
    item.add_argument("--codec", default="byte")
    item.add_argument(
        "--bundle", help="Run the exact shipped weights so model-bound knowledge bases can mount"
    )
    item.add_argument("--packs", help="The realizer pack manifest paired with --bundle")
    item.add_argument("--device", default="cpu")
    item.add_argument("--precision", choices=("bf16", "fp32"))
    for name, default in (
        ("steps", 5),
        ("warmup", 2),
        ("batch", 1),
        ("record-bytes", 256),
        ("capacity", 64),
        ("effort", 6),
        ("history-records", 0),
        ("prediction-chunks", 1),
    ):
        item.add_argument("--" + name, type=int, default=default)
    item.add_argument("--backend", choices=("torch", "triton"), default="torch")
    item.add_argument(
        "--representation-mode", choices=("bootstrap", "recurrent"), default="bootstrap"
    )
    item.add_argument("--compile-mode")
    for name in ("backward", "profile"):
        item.add_argument("--" + name, action="store_true")
    diagnostics = item.add_mutually_exclusive_group()
    diagnostics.add_argument(
        "--core-only",
        action="store_true",
        help="Measure the encoder/core without lexical generation",
    )
    diagnostics.add_argument(
        "--model-only",
        action="store_true",
        help="Measure full generation without session memory lookups",
    )
    item.add_argument("--collection", action="append", default=[], metavar="NAME=DIRECTORY")
    item.add_argument("--base", action="append", default=[], metavar="NAME=FILE")
    item.add_argument("--trained-weight", type=float, default=1.0)
    item.add_argument("--max-sources", type=int, default=2)
    item.add_argument("--threshold", type=float)
    item.set_defaults(func=_benchmark)
    item = commands.add_parser("export", help="Create a checksummed served-only safetensors bundle")
    item.add_argument("--checkpoint", required=True)
    item.add_argument("--output", required=True)
    item.add_argument("--codec", default="byte")
    item.add_argument("--packs", help="Locked realizer pack manifest the model was trained with")
    item.set_defaults(func=_export)
    item = commands.add_parser(
        "generate",
        help="Run one session on the serving runtime: a prompt, then one answer and its tape",
    )
    _serving_arguments(item)
    prompts = item.add_mutually_exclusive_group(required=True)
    prompts.add_argument("--prompt")
    prompts.add_argument("--prompt-file")
    item.add_argument("--metadata", default="{}", help="Session metadata, a JSON object")
    item.add_argument(
        "--base",
        action="append",
        default=[],
        metavar="NAME=FILE",
        help="Mount a knowledge base file built for this model (repeatable)",
    )
    item.add_argument("--tools", help="Public JSON tool schemas")
    item.add_argument("--control-seed-file", help="Raw-byte causal policy control file")
    item.add_argument("--control-reference-file", help="Raw-byte causal reference evidence file")
    item.add_argument(
        "--reference-origin",
        choices=("external_evidence", "earlier_history"),
        help="Required causal provenance when a nonempty reference control is supplied",
    )
    item.add_argument("--kind", choices=("bytes", "utf8", "ascii"), default="utf8")
    for name, default in (
        ("min-tokens", 0),
        ("max-tokens", None),
        ("effort", 6),
        ("seed", 17),
    ):
        item.add_argument("--" + name, type=int, default=default)
    item.set_defaults(func=_generate)
    item = commands.add_parser(
        "serve",
        help="Serve stateful text and media sessions over a local JSONL protocol",
    )
    _serving_arguments(item)
    item.add_argument("--max-sessions", type=int, default=32)
    item.add_argument(
        "--kb-root",
        help="Persistent memory directory; scoped sessions remember automatically; enables kb_* operations",
    )
    item.set_defaults(func=_serve)
    group = commands.add_parser(
        "kb", help="Build and inspect knowledge bases that ship apart from the model"
    )
    sub = group.add_subparsers(dest="kb_command", required=True)
    build = sub.add_parser(
        "build",
        help="Write a base file from prepared roots or plain files, reading each item alone",
    )
    build.add_argument("--name", required=True)
    build.add_argument(
        "--items", nargs="+", required=True, help="Prepared roots, or text/image/audio/video files"
    )
    build.add_argument("--model", required=True, help="Exported bundle or training checkpoint")
    build.add_argument("--packs", help="Locked realizer pack manifest (needed to prepare files)")
    build.add_argument("--codec", help="Codec location, if a checkpoint's has moved")
    build.add_argument("--device", default="auto")
    build.add_argument("--output", help="Base file to write (default: <name>.kb)")
    build.add_argument(
        "--prepared",
        default="kb-prepared",
        help="Where prepared features of plain files are kept (a new prepared root)",
    )
    build.add_argument("--split", choices=("train", "validation", "test", "all"), default="train")
    build.add_argument("--batch", type=int, default=16, help="Items read together")
    for name in ("task", "topic", "language", "date"):
        build.add_argument(f"--{name}", help=f"File every plain file under this {name}")
    build.set_defaults(func=_kb_build)
    inspecting = sub.add_parser(
        "inspect", help="Show what a base file holds and which model wrote it"
    )
    inspecting.add_argument("base")
    inspecting.set_defaults(func=_kb_inspect)
    item = commands.add_parser(
        "engram", help="Offline donor memory import, token-embedding baseline, and exact baking"
    )
    sub = item.add_subparsers(dest="engram_command", required=True)
    extract = sub.add_parser("extract")
    extract.add_argument("--embedding-file")
    extract.add_argument("--tensor-name")
    extract.add_argument("--donor-codec")
    extract.add_argument(
        "--donor-profile", choices=("deepseek-v4.1-flash-1017", "qwen3.8-flash-next")
    )
    extract.add_argument(
        "--donor-root",
        help="Local snapshot containing pinned model/tokenizer metadata and embedding shard",
    )
    extract.add_argument("--output", required=True)
    extract.add_argument("--model", default="32x512")
    extract.add_argument("--seed", type=int, default=17)
    extract.set_defaults(func=_engram_extract)
    native = sub.add_parser("extract-native", help="Import a selected donor's n-gram memory")
    native.add_argument(
        "--donor-profile", choices=("deepseek-v4.1-flash-1017", "qwen3.8-flash-next"), required=True
    )
    native.add_argument("--donor-root", required=True)
    native.add_argument("--corpus", action="append", required=True)
    native.add_argument("--output", required=True)
    native.add_argument("--model", default="32x512")
    native.add_argument("--max-contexts", type=int, required=True)
    native.add_argument("--seed", type=int, default=17)
    native.add_argument("--split", default="train")
    native.add_argument("--cache-dir")
    native.add_argument("--source-mode", choices=("local", "ranges"), default="local")
    native.set_defaults(func=_engram_native_extract)
    bake = sub.add_parser("bake")
    bake.add_argument("--checkpoint", required=True)
    bake.add_argument("--output", required=True)
    bake.add_argument("--codec", default="byte")
    bake.add_argument("--packs")
    bake.set_defaults(func=_engram_bake)
    return root


def install_hub_retries() -> None:
    """Hub reads wait out rate limits (see ``megatoken.hub``) when the Hub client is installed."""
    try:
        from megatoken.hub import install
    except ImportError:  # httpx and huggingface_hub come with the optional "data" extra.
        return
    install()


def main(argv: list[str] | None = None) -> int:
    install_hub_retries()
    root = parser()
    args = root.parse_args(argv)
    try:
        result = args.func(args)
        if result is not None:
            if args.json_output:
                write_json(args.json_output, result)
            print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
        # Keep the verified-replay receipt available to automation while making
        # a failed repair a failing shell gate. Other commands retain their
        # existing return contract.
        if (
            getattr(args, "chunks_command", None) == "review"
            and getattr(args, "review_command", None) == "replay"
            and result is not None
            and not result["success"]
        ):
            return 1
        if (
            getattr(args, "chunks_command", None) == "coverage"
            and result is not None
            and result["validated"]
            and result["failures"]
        ):
            return 1
        return 0
    except (ValueError, OSError, RuntimeError, ImportError) as error:
        if args.debug:
            raise
        root.exit(2, f"megatoken: {type(error).__name__}: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
