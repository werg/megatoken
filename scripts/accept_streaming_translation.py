#!/usr/bin/env python3
"""Train sustained translation, restart, and archive held-out served generations."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, replace
from pathlib import Path

from megatoken.io import write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, action="append", required=True)
    p.add_argument("--packs", type=Path)
    p.add_argument("--evaluation-stream", default="streaming_translation")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--steps", type=int, default=256)
    p.add_argument("--model", choices=["tiny", "32x256", "32x512"], default="tiny")
    p.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    p.add_argument("--microbatch-episodes", type=int, default=1)
    p.add_argument("--evaluation-episodes", type=int, default=4)
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--phase", choices=["B0", "P1", "MB"], default="B0")
    p.add_argument("--budget", type=int)
    p.add_argument("--resume", type=Path)
    p.add_argument("--audio-samples", type=int)
    p.add_argument("--expert-backend", choices=["torch", "auto", "triton"], default="torch")
    p.add_argument(
        "--visual-output",
        action="store_true",
        help="Declare image/video outputs at 256x192, nine video frames, 8 fps and four decoder steps",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--_stage", default="train", choices=["train", "restart", "export", "evaluate"])
    args = p.parse_args()
    if min(args.steps, args.microbatch_episodes, args.evaluation_episodes) < 1:
        p.error("Training steps, batch size and evaluation count must be positive")
    from megatoken.training.config import TrainConfig

    config = TrainConfig(
        purpose="diagnostic",
        model=args.model,
        phase=args.phase,
        trainable="media_bridge" if args.phase == "MB" else "all",
        budget=args.budget,
        datasets=[str(path) for path in args.data],
        output=str(args.root / "training"),
        codec="byte",
        allow_byte_codec=True,
        device=args.device,
        precision=args.precision,
        packs=None if args.packs is None else str(args.packs),
        model_overrides={"max_capacity": 512},
        max_steps=args.steps,
        microbatch_episodes=args.microbatch_episodes,
        max_accumulation_windows=1,
        accumulate_scored=1,
        accumulate_ingested=4096,
        max_window_records=4,
        max_history_tokens=1024,
        max_record_bytes=8192,
        generated_capacity=512,
        effort=1,
        gradient_checkpointing=args.gradient_checkpointing,
        expert_backend=args.expert_backend,
        checkpoint_every=32,
        use_stage_mixture=False,
        use_effort_mixture=False,
    )
    checkpoint = args.root / "training/checkpoints/latest.json"

    def next_stage(stage):
        argv = sys.argv[1:]
        if "--_stage" in argv:
            at = argv.index("--_stage")
            del argv[at : at + 2]
        os.execv(sys.executable, [sys.executable, __file__, *argv, "--_stage", stage])

    if args._stage == "train":
        args.root.mkdir(parents=True, exist_ok=False)
        write_json(args.root / "train-config.json", asdict(config))
        from megatoken.training.engine import train

        result = train(
            config,
            resume=None if args.resume is None else str(args.resume),
            phase_transition=args.resume is not None,
        )
        write_json(args.root / "training.json", result)
        if result["termination"] == "requested":
            raise SystemExit(130)
        next_stage("restart")
    # Stage transitions use the recipe recorded before the first update. Later
    # helper defaults must not silently alter an already running experiment.
    config = TrainConfig.load(args.root / "train-config.json")
    if args._stage == "restart":
        from megatoken.training.engine import train

        result = train(replace(config, max_steps=2), resume=str(checkpoint))
        write_json(args.root / "restart.json", result)
        if result["termination"] == "requested":
            raise SystemExit(130)
        next_stage("export")
    if args._stage == "export":
        from megatoken.served import export_served

        write_json(
            args.root / "export.json",
            export_served(checkpoint, args.root / "served", codec_path="byte"),
        )
        next_stage("evaluate")
    import torch

    from megatoken.data.storage import PreparedCorpus
    from megatoken.generation.structure import ActionSpec
    from megatoken.model.media import MediaOutputSpec
    from megatoken.omni.quality import DecodedQualityArchive
    from megatoken.served import ServedBundle
    from megatoken.training.evaluation import evaluate_served_records

    bundle = ServedBundle(args.root / "served", packs=args.packs, device=args.device)
    archive = DecodedQualityArchive(
        args.root / "decoded",
        evaluator_metadata={
            "reference": (
                "publisher supplied human English translation"
                if args.evaluation_stream == "streaming_translation"
                else "exact independently executed fixture output"
            ),
            "text_normalization": "lowercase whitespace tokenization for WER; exact UTF-8 bytes archived",
        },
    )
    with PreparedCorpus(config.datasets, split="validation", stream=args.evaluation_stream) as data:
        if not len(data):
            raise ValueError("The selected held-out stream contains no prepared episodes")
        output_specs = {"text": ActionSpec(max_tokens=config.generated_capacity)}
        if args.audio_samples is not None:
            output_specs["audio"] = MediaOutputSpec(
                "audio", sample_count=args.audio_samples, sample_rate=24000
            )
        if args.visual_output:
            output_specs.update(
                image=MediaOutputSpec("image", height=192, width=256, steps=4),
                video=MediaOutputSpec("video", frames=9, height=192, width=256, fps_num=8, steps=4),
            )
        report = evaluate_served_records(
            bundle.model,
            data,
            config,
            torch.device(args.device),
            episodes=args.evaluation_episodes,
            decoded_evaluator=archive,
            output_specs=output_specs,
        )
    write_json(
        args.root / "acceptance.json",
        {
            "training": json.loads((args.root / "training.json").read_text()),
            "restart": json.loads((args.root / "restart.json").read_text()),
            "export": json.loads((args.root / "export.json").read_text()),
            "heldout_generated": report,
            "fresh_process_boundaries": True,
            "independent_task_quality_established": False,
            "autonomous_lifecycle_established": False,
        },
    )


if __name__ == "__main__":
    main()
