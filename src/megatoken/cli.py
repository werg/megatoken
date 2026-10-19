"""Small, lazy CLI over the same public APIs used in Python and tests."""

from __future__ import annotations

import argparse
import base64
import json
import sys
from dataclasses import asdict
from pathlib import Path

from megatoken import __version__
from megatoken.io import read_mapping, write_json


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
            "dataset",
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
    config = (
        TrainConfig.load(args.config, **overrides)
        if args.config
        else TrainConfig.from_dict(overrides)
    )
    return train(config, resume=args.resume, phase_transition=args.phase_transition)


def _diloco_initialize(args):
    """Create immutable coordinator bootstrap from the exact train architecture."""
    import torch

    from megatoken.data import PreparedDataset
    from megatoken.loading import build_model, load_codec, load_model_config
    from megatoken.runtime import seed_everything
    from megatoken.training.checkpoint import load_checkpoint
    from megatoken.training.config import TrainConfig
    from megatoken.training.diloco_protocol import training_identity
    from megatoken.io import content_hash
    from megatoken.training.diloco_server import initialize_coordinator
    from megatoken.training.schedule import PHASES

    config = TrainConfig.load(args.config)
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
    model = build_model(model_config, codec, device=torch.device("cpu"), precision=config.precision)
    weights = None
    if args.resume:
        payload, _ = load_checkpoint(args.resume, map_location="cpu")
        if payload["model_config"] != model_config.to_dict():
            raise ValueError("Bootstrap checkpoint model architecture changed")
        if payload["codec_fingerprint"] != codec.fingerprint:
            raise ValueError("Bootstrap checkpoint codec differs")
        model.load_state_dict(payload["model"], strict=True)
        weights = payload["optimizer"]["master_weights"]
    elif config.engram_import:
        from megatoken.assets import install_engram_asset

        install_engram_asset(model, config.engram_import)
    masters = {
        name: (weights[name] if weights is not None else parameter).detach().cpu().float()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    with PreparedDataset(config.dataset) as prepared:
        dataset_fingerprint = content_hash(prepared.manifest)
    return initialize_coordinator(
        args.root,
        run_id=config.decoupled.run_id,
        model_state=model.state_dict(),
        masters=masters,
        model_config=model_config.to_dict(),
        codec_fingerprint=codec.fingerprint,
        train_identity=training_identity(config),
        dataset_fingerprint=dataset_fingerprint,
        workers=config.decoupled.workers,
        min_quorum=args.quorum,
        fragment_bytes=args.fragment_mib * 1024 * 1024,
        outer_lr=args.outer_lr,
        outer_momentum=args.outer_momentum,
        grace_seconds=args.grace_seconds,
        max_staleness=args.max_staleness,
        token_budget=config.budget or PHASES[config.phase].budget,
    )


def _diloco_serve(args):
    import os

    from megatoken.training.diloco_server import serve_coordinator

    serve_coordinator(
        args.root, args.host, args.port,
        token=os.environ.get("MEGATOKEN_DILOCO_TOKEN", ""),
    )


def _diloco_export(args):
    from megatoken.training.config import TrainConfig
    from megatoken.training.diloco_export import export_coordinator

    return export_coordinator(args.root, args.output, TrainConfig.load(args.config))


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

    from megatoken.data import PreparedDataset
    from megatoken.loading import build_model, load_codec
    from megatoken.model.config import ModelConfig
    from megatoken.runtime import seed_everything
    from megatoken.training.checkpoint import load_checkpoint
    from megatoken.training.config import TrainConfig
    from megatoken.training.evaluation import evaluate_records

    payload, _ = load_checkpoint(args.checkpoint)
    model_config = ModelConfig.from_dict(payload["model_config"])
    codec_path = args.codec or payload["train_config"]["codec"]
    codec = load_codec(codec_path, vocab_size=model_config.vocab_size)
    if codec.fingerprint != payload["codec_fingerprint"]:
        raise ValueError("Evaluation codec differs from the checkpoint")
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    )
    model = build_model(
        model_config, codec, device=device, precision=payload["train_config"]["precision"]
    )
    model.load_state_dict(payload["model"], strict=True)
    model.core.set_model_version(payload["global_step"])
    seed_everything(args.seed)
    config = TrainConfig.from_dict(
        {**payload["train_config"], "eval_episodes": args.episodes, "effort": args.effort}
    )
    dataset = PreparedDataset(args.dataset or config.dataset, split=args.split)
    try:
        return evaluate_records(model, dataset, codec, config, device)
    finally:
        dataset.close()


def _benchmark(args):
    from megatoken.benchmark import benchmark

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


def _rl(args):
    from megatoken.training.rl import RLConfig, train_rl

    base = read_mapping(args.config) if args.config else {}
    names = {key for key in RLConfig.__dataclass_fields__}
    unknown = set(base) - names
    if unknown:
        raise ValueError(f"Unknown RL settings: {sorted(unknown)}")
    for name in names:
        value = getattr(args, name, None)
        if value is not None:
            base[name] = value
    path = Path(args.tasks)
    tasks = (
        json.loads(path.read_text())
        if path.suffix == ".json"
        else [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    )
    if not isinstance(tasks, list) or any(not isinstance(task, dict) for task in tasks):
        raise ValueError("RL tasks must be a JSON array or JSONL objects")
    return train_rl(RLConfig(**base), tasks)


def _export(args):
    from megatoken.inference import export_served

    return export_served(args.checkpoint, args.output, codec_path=args.codec)


def _generate(args):
    from megatoken.generation.structure import ActionSpec
    from megatoken.inference import InferenceEngine

    engine = InferenceEngine(args.model, device=args.device, compile=args.compile)
    session = engine.new_session()
    payload = Path(args.prompt_file).read_bytes() if args.prompt_file else args.prompt.encode()
    engine.observe(session, payload)
    maximum = min(512, engine.config.max_capacity) if args.max_tokens is None else args.max_tokens
    spec = ActionSpec(
        min_tokens=args.min_tokens,
        max_tokens=maximum,
        text_kind=args.kind,
    )
    tools_json = Path(args.tools).read_text(encoding="utf-8") if args.tools else "[]"
    if args.tools:
        from megatoken.tool_schema import compile_tool_spec

        spec = compile_tool_spec(tools_json, maximum)
    result = engine.generate(
        session,
        spec=spec,
        effort=args.effort,
        seed=args.seed,
        control_seed=Path(args.control_seed_file).read_bytes() if args.control_seed_file else b"",
        control_reference=(
            Path(args.control_reference_file).read_bytes() if args.control_reference_file else b""
        ),
        reference_origin=args.reference_origin,
        tools_json=tools_json,
    )
    response = {
        "payload_b64": base64.b64encode(result.payload).decode(),
        "emitted_ids": list(result.emitted_ids),
        "valid": result.tape.valid,
        "latent_dimensions": result.tape.latent_dimensions,
        "likelihood": "canonical_facet_gaussian_density",
        "terminal": result.tape.terminal,
        "tape": json.loads(result.tape.to_json()),
    }
    try:
        response["text"] = result.payload.decode("utf-8")
    except UnicodeError:
        pass
    return response


def _serve(args):
    from megatoken.inference import InferenceEngine, serve_jsonl

    serve_jsonl(
        InferenceEngine(args.model, device=args.device, compile=args.compile),
        sys.stdin,
        sys.stdout,
        max_sessions=args.max_sessions,
    )


def _serve_multitrack(args):
    from megatoken.inference import InferenceEngine
    from megatoken.multitrack_service import serve_multitrack_jsonl

    serve_multitrack_jsonl(
        InferenceEngine(args.model, device=args.device, compile=args.compile),
        sys.stdin,
        sys.stdout,
        max_sessions=args.max_sessions,
    )


def _smoke(args):
    from megatoken.smoke import run_smoke

    return run_smoke(args.output, device=args.device, precision=args.precision)


def _engram_extract(args):
    from megatoken.assets import extract_engram

    return extract_engram(
        args.embedding_file,
        args.tensor_name,
        args.donor_codec,
        args.output,
        model=args.model,
        seed=args.seed,
    )


def _engram_bake(args):
    from megatoken.assets import bake_checkpoint

    return bake_checkpoint(args.checkpoint, args.output, codec_path=args.codec)


def _preset_name(value: str) -> str:
    """Keep historical user-facing aliases out of saved model configurations."""
    return {"recurrent32": "32x512", "compact32": "32x256"}.get(value, value)


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
        "inspect", help="Audit actual parameter owners without allocating model storage"
    )
    item.add_argument("--model", type=_preset_name, default="tiny")
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
    item.add_argument("--config")
    for name in (
        "phase",
        "model",
        "dataset",
        "output",
        "codec",
        "device",
        "precision",
        "expert-backend",
        "representation-mode",
    ):
        item.add_argument("--" + name, type=_preset_name if name == "model" else str)
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
    initial.add_argument("--resume", help="Warm-start global weights from an earlier checkpoint")
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
    exporting = sub.add_parser("export", help="Export a stopped syncer's global model for evaluation")
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
        "evaluate", help="Fixed-holdout served-prior sampling and lexical CE"
    )
    item.add_argument("--checkpoint", required=True)
    item.add_argument("--dataset")
    item.add_argument("--codec")
    item.add_argument("--device", default="auto")
    item.add_argument("--split", choices=("validation", "test"), default="validation")
    item.add_argument("--episodes", type=int, default=16)
    item.add_argument("--effort", type=int, choices=(1, 2, 3, 6), default=6)
    item.add_argument("--seed", type=int, default=17)
    item.set_defaults(func=_evaluate)
    item = commands.add_parser(
        "benchmark", help="Measure real end-to-end work, memory, and dispatch"
    )
    item.add_argument("--model", type=_preset_name, default="tiny")
    item.add_argument("--codec", default="byte")
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
    for name in ("backward", "profile", "core-only"):
        item.add_argument("--" + name, action="store_true")
    item.set_defaults(func=_benchmark)
    item = commands.add_parser("rl", help="Collect and replay fresh executed policy episodes")
    item.add_argument("--config")
    item.add_argument("--tasks", required=True)
    for name in ("checkpoint", "output", "environment", "codec", "device"):
        item.add_argument("--" + name)
    for name in (
        "task-seeds",
        "rollouts-per-seed",
        "max-steps-per-episode",
        "max-updates",
        "max-task-groups",
        "max-latent-coordinates",
        "max-total-episodes",
        "capacity",
        "seed",
        "checkpoint-every",
    ):
        item.add_argument("--" + name, type=int)
    item.add_argument("--require-blackwell", action=argparse.BooleanOptionalAction, default=None)
    item.set_defaults(func=_rl)
    item = commands.add_parser("export", help="Create a checksummed served-only safetensors bundle")
    item.add_argument("--checkpoint", required=True)
    item.add_argument("--output", required=True)
    item.add_argument("--codec", default="byte")
    item.set_defaults(func=_export)
    item = commands.add_parser(
        "generate", help="Generate one complete chunk and return its exact tape"
    )
    item.add_argument("--model", required=True, help="Exported served bundle")
    item.add_argument("--device", default="auto")
    item.add_argument("--compile", action="store_true")
    prompts = item.add_mutually_exclusive_group(required=True)
    prompts.add_argument("--prompt")
    prompts.add_argument("--prompt-file")
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
        "serve", help="Serve stateful exact-byte sessions over a local JSONL protocol"
    )
    item.add_argument("--model", required=True)
    item.add_argument("--device", default="auto")
    item.add_argument("--compile", action="store_true")
    item.add_argument("--max-sessions", type=int, default=32)
    item.set_defaults(func=_serve)
    item = commands.add_parser(
        "serve-multitrack", help="Serve v8 joint semantic meridians via local JSONL"
    )
    item.add_argument("--model", required=True)
    item.add_argument("--device", default="auto")
    item.add_argument("--compile", action="store_true")
    item.add_argument("--max-sessions", type=int, default=32)
    item.set_defaults(func=_serve_multitrack)
    item = commands.add_parser(
        "smoke", help="Run offline preparation, all stages, executed RL, export, and generation"
    )
    item.add_argument("--output", default="runs/smoke")
    item.add_argument("--device", default="cpu")
    item.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    item.set_defaults(func=_smoke)
    item = commands.add_parser(
        "engram", help="Optional offline native-table initialization and exact baking"
    )
    sub = item.add_subparsers(dest="engram_command", required=True)
    extract = sub.add_parser("extract")
    extract.add_argument("--embedding-file", required=True)
    extract.add_argument("--tensor-name", required=True)
    extract.add_argument("--donor-codec", required=True)
    extract.add_argument("--output", required=True)
    extract.add_argument("--model", type=_preset_name, default="32x512")
    extract.add_argument("--seed", type=int, default=17)
    extract.set_defaults(func=_engram_extract)
    bake = sub.add_parser("bake")
    bake.add_argument("--checkpoint", required=True)
    bake.add_argument("--output", required=True)
    bake.add_argument("--codec", default="byte")
    bake.set_defaults(func=_engram_bake)
    return root


def main(argv: list[str] | None = None) -> int:
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
