#!/usr/bin/env python3
"""Optimize checked recorded media, restart, export and exercise real decoding.

The generated checks use caller-declared output bounds. They establish executable
media training and serving, not intelligibility, translation or synchronization.
"""

from __future__ import annotations

import argparse
import gc
from dataclasses import asdict, replace
from pathlib import Path

import torch

from megatoken.data.storage import PreparedDataset
from megatoken.io import sha256_file, write_json
from megatoken.model.media import MediaOutputSpec
from megatoken.served import ServedBundle, export_served
from megatoken.training.config import TrainConfig
from megatoken.training.engine import train
from megatoken.training.evaluation import evaluate_served_records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--packs", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--codec", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--audio-samples", type=int, default=24000)
    parser.add_argument("--sample-rate", type=int, default=24000)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=False)
    if min(args.steps, args.audio_samples, args.sample_rate) < 1:
        raise ValueError("Update and audio bounds must be positive")
    config = TrainConfig(
        model="tiny",
        phase="B0",
        datasets=[str(args.data)],
        output=str(args.root / "training"),
        codec=args.codec,
        allow_byte_codec=args.codec == "byte",
        device=args.device,
        precision="fp32",
        packs=str(args.packs),
        model_overrides={"max_capacity": 512},
        max_steps=args.steps,
        microbatch_episodes=1,
        max_accumulation_windows=1,
        accumulate_scored=1,
        accumulate_ingested=2048,
        max_window_records=4,
        max_history_tokens=512,
        max_record_bytes=8192,
        generated_capacity=512,
        effort=1,
        gradient_checkpointing=False,
        expert_backend="torch",
        checkpoint_every=4,
        use_stage_mixture=False,
        use_effort_mixture=False,
    )
    write_json(args.root / "train-config.json", asdict(config))
    first = train(config)
    gc.collect()
    checkpoint = args.root / "training/checkpoints/latest.json"
    restarted = train(replace(config, max_steps=2), resume=str(checkpoint))
    gc.collect()
    exported = export_served(checkpoint, args.root / "served", codec_path=args.codec)
    bundle = ServedBundle(args.root / "served", packs=args.packs, device=args.device)
    output_specs = {
        "audio": MediaOutputSpec(
            "audio",
            sample_count=args.audio_samples,
            sample_rate=args.sample_rate,
        )
    }
    with PreparedDataset(args.data, split="train") as data:
        report = evaluate_served_records(
            bundle.model,
            data,
            config,
            torch.device(args.device),
            episodes=1,
            output_specs=output_specs,
        )
    write_json(
        args.root / "acceptance.json",
        {
            "script_sha256": sha256_file(__file__),
            "data_manifest_sha256": sha256_file(args.data / "manifest.json"),
            "packs_manifest_sha256": sha256_file(args.packs),
            "training": first,
            "restart": restarted,
            "export": exported,
            "generated": report,
            "output_specs": {k: asdict(v) for k, v in output_specs.items()},
            "evaluation_split": "train",
            "heldout_quality_established": False,
        },
    )


if __name__ == "__main__":
    main()
