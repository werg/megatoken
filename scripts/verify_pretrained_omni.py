#!/usr/bin/env python3
"""Validate pinned pretrained omnimodal packs without dataset preparation/training.

Use --manifest only with an existing, fully checksummed local pack snapshot.
The script probes the *real* donor code, not the native debug media heads.
Large video checkpoints require accelerator memory; no downloads occur.
"""
from __future__ import annotations

import argparse
import json

import torch

from megatoken.omni.loader import read_locked_manifest
from megatoken.omni.donors import PocketFacetsStudent, SanaFacetsStudent
from megatoken.omni.validation import check_pretrained_student


def _device(name: str) -> torch.device:
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA donor smoke requested but no CUDA device is present")
    return device


def probe(manifest: str, *, device: str, packs: tuple[str, ...]) -> dict:
    base, items = read_locked_manifest(manifest)
    result = {}
    torch_device = _device(device)
    for name in packs:
        if name not in items:
            raise ValueError(f"Requested pack {name} is missing from pinned manifest")
        item = items[name]
        if name == "pocket":
            student = PocketFacetsStudent.from_checkpoint(
                config=str(base / item["config"]),
                checkpoint=(str(base / item["checkpoint"]) if item.get("checkpoint") else None),
            ).to(torch_device)
            sample = torch.zeros(1, int(student.sample_rate), device=torch_device)
            with torch.no_grad():
                latent = student.encode_waveform(sample, sample_rate=student.sample_rate)
            sample_rate = student.sample_rate
        else:
            student = SanaFacetsStudent.from_checkpoint(
                snapshot=str(base / item["snapshot"]),
                modality="image" if name == "sana" else "video",
                torch_dtype=torch.bfloat16 if torch_device.type == "cuda" else torch.float32,
            ).to(torch_device)
            # Minimal dummy latent geometry is for exercising the flow objective,
            # not a meaningful image/video quality comparison.
            config = student.transformer.config
            channels = int(getattr(config, "in_channels"))
            if name == "sana":
                latent = torch.zeros(1, channels, 16, 16, device=torch_device)
            else:
                latent = torch.zeros(1, channels, 9, 16, 16, device=torch_device)
            sample_rate = 0
        raw = torch.randn(1, 32, 512, device=torch_device)
        student.train()
        contract = check_pretrained_student(name, student, raw, latent)
        student.eval()
        result[name] = {
            "latent_shape": contract.latent_shape,
            "finite_native_loss": contract.loss,
            "trainable_parameters": contract.trainable_parameters,
            "frozen_codec_parameters": contract.frozen_parameters,
            "sample_rate": sample_rate,
            "weights_revision": item["revision"],
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--packs", nargs="+", choices=("pocket", "sana", "sana_video"),
                        default=["pocket", "sana", "sana_video"])
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    report = probe(args.manifest, device=args.device, packs=tuple(args.packs))
    value = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        from pathlib import Path
        Path(args.output).write_text(value + "\n", encoding="utf-8")
    print(value)


if __name__ == "__main__":
    main()
