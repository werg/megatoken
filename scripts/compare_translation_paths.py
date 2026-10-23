#!/usr/bin/env python3
"""Compare actual exported joint and text-then-speech translation continuations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from megatoken.data.storage import PreparedDataset
from megatoken.generation.structure import ActionSpec
from megatoken.io import sha256_file, write_json
from megatoken.model.media import MediaOutputSpec
from megatoken.omni.quality import DecodedQualityArchive
from megatoken.served import ServedBundle
from megatoken.training.config import TrainConfig
from megatoken.training.evaluation import evaluate_served_records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--packs", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--audio-samples", type=int, default=192000)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    if args.audio_samples < 1:
        p.error("The same positive public speech bound is required for both paths")
    args.root.mkdir(parents=True, exist_ok=False)
    bundle = ServedBundle(args.bundle, packs=args.packs, device=args.device)
    config = TrainConfig(device=args.device, generated_capacity=512, effort=1)
    reports = {}
    with PreparedDataset(args.data, split="validation", stream="speech_translation") as data:
        for path in ("direct_joint", "text_then_speech"):
            selected = []
            for episode in data.iter_episodes():
                annotation = json.loads(episode.supervision.annotations_json)
                if annotation.get("verification", {}).get("translation_path") == path:
                    selected.append(episode)
            if not selected:
                raise ValueError(f"Held-out source families do not supply {path}")

            class View:
                assets = data.assets

                def iter_episodes(self, examples=tuple(selected)):
                    return iter(examples)

            archive = DecodedQualityArchive(
                args.root / path,
                evaluator_metadata={
                    "reference": "publisher human translated text, held out by sentence family",
                    "language": "French to English",
                    "sample_rate": 24000,
                    "public_sample_count": args.audio_samples,
                    "normalization": "lowercase whitespace words; exact sampled bytes preserved separately",
                    "speech_recognizer": "not installed in the generation process; use independent post-generation evaluation",
                },
            )
            reports[path] = evaluate_served_records(
                bundle.model,
                View(),
                config,
                torch.device(args.device),
                episodes=len(selected),
                output_specs={
                    "text": ActionSpec(max_tokens=512),
                    "audio": MediaOutputSpec(
                        "audio", sample_count=args.audio_samples, sample_rate=24000
                    ),
                },
                decoded_evaluator=archive,
            )
    families = {
        path: sorted(r["family_id"] for r in result["decoded_quality"])
        for path, result in reports.items()
    }
    if families["direct_joint"] != families["text_then_speech"]:
        raise ValueError("Translation paths used different held-out source families")
    write_json(
        args.root / "comparison.json",
        {
            "format": "translation-path-comparison/v1",
            "paths": reports,
            "matched_original_families": families["direct_joint"],
            "served_manifest_sha256": sha256_file(args.bundle / "manifest.json"),
            "data_manifest_sha256": sha256_file(args.data / "manifest.json"),
            "same_public_audio_samples": args.audio_samples,
            "same_public_sample_rate": 24000,
            "timing_definition": "wall time for actual served generation; first output includes available source ingestion; archive writing excluded from per-generation time",
            "quality_winner_established": False,
            "human_review": "pending",
        },
    )


if __name__ == "__main__":
    main()
