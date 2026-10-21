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

from megatoken.omni.donors import PocketFacetsStudent, SanaFacetsStudent
from megatoken.omni.loader import read_locked_manifest
from megatoken.omni.validation import check_pretrained_student


def _device(name: str) -> torch.device:
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA donor smoke requested but no CUDA device is present")
    return device


def probe(
    manifest: str,
    *,
    device: str,
    packs: tuple[str, ...],
    rollout: bool = False,
    image_size: int = 1024,
    video_height: int = 704,
    video_width: int = 1280,
    video_frames: int = 81,
    steps: int = 16,
    roundtrip: bool = False,
) -> dict:
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
            channels = int(config.in_channels)
            if name == "sana":
                latent = torch.zeros(1, channels, 16, 16, device=torch_device)
            else:
                latent = torch.zeros(1, channels, 9, 16, 16, device=torch_device)
            sample_rate = 0
        raw = torch.randn(1, 32, 512, device=torch_device)
        student.train()
        contract = check_pretrained_student(name, student, raw, latent)
        student.eval()
        # Objective gradients are not sufficient to certify a generator. The
        # optional expensive native free rollout starts from RNG, not gold U.
        # Only its physical contract is checked; intelligibility/quality still
        # requires separate task-based evaluation after adapter training.
        generated = {"status": "not_run"}
        if rollout:
            with torch.inference_mode():
                rng = torch.Generator(device=torch_device).manual_seed(78117)
                if name == "pocket":
                    output = student.realize(
                        raw,
                        max_frames=16,
                        steps=steps,
                        generator=rng,
                    )
                    expected_rank = 2
                elif name == "sana":
                    output = student.realize(
                        raw,
                        height=image_size,
                        width=image_size,
                        steps=steps,
                        generator=rng,
                    )
                    expected_rank = 5
                else:
                    output = student.realize(
                        raw,
                        height=video_height,
                        width=video_width,
                        frames=video_frames,
                        steps=steps,
                        generator=rng,
                    )
                    expected_rank = 5
                if not isinstance(output, torch.Tensor):
                    raise ValueError("Donor rollout must materialize as a Torch tensor")
                if (
                    output.ndim != expected_rank
                    or not output.is_floating_point()
                    or not torch.isfinite(output).all()
                ):
                    raise ValueError("Full donor rollout returned invalid media")
                if output.shape[0] != raw.shape[0]:
                    raise ValueError("Donor rollout changed the selected event batch")
                generated = {
                    "status": "physical_contract_pass",
                    "shape": list(output.shape),
                    "min": float(output.min()),
                    "max": float(output.max()),
                }
        codec_check = {"status": "not_run"}
        if roundtrip:
            with torch.inference_mode():
                if name == "pocket":
                    decoded = student.decode(latent)
                    original = sample
                else:
                    if name == "sana":
                        original = torch.zeros(
                            1,
                            3,
                            64,
                            64,
                            device=torch_device,
                            dtype=student.vae.dtype,
                        )
                    else:
                        original = torch.zeros(
                            1,
                            3,
                            9,
                            64,
                            64,
                            device=torch_device,
                            dtype=student.vae.dtype,
                        )
                    encoded = student.vae.encode(original, return_dict=True)
                    if hasattr(encoded, "latent_dist"):
                        latent_codec = encoded.latent_dist.mode()
                    elif hasattr(encoded, "latents"):
                        latent_codec = encoded.latents
                    else:
                        raise ValueError("Donor codec has no explicit encoded latent result")
                    decoded = student.vae.decode(latent_codec, return_dict=False)[0]
                if (
                    not isinstance(decoded, torch.Tensor)
                    or not torch.isfinite(decoded).all()
                    or decoded.ndim != original.ndim
                ):
                    raise ValueError("Matched codec round trip has invalid shape/values")
                # MIME and temporal codecs may introduce deterministic padding,
                # which is excluded from the valid reconstruction RMSE.
                slices = tuple(
                    slice(0, min(n, m))
                    for n, m in zip(
                        original.shape,
                        decoded.shape,
                        strict=True,
                    )
                )
                aligned = decoded[slices].float()
                valid = original[slices].float()
                error = float((aligned - valid).square().mean().sqrt().cpu())
                codec_check = {
                    "status": "physical_contract_pass",
                    "native_shape": list(original.shape),
                    "decoded_shape": list(decoded.shape),
                    "measured_synthetic_rmse": error,
                }
        result[name] = {
            "codec_roundtrip": codec_check,
            "latent_shape": contract.latent_shape,
            "finite_native_loss": contract.loss,
            "trainable_parameters": contract.trainable_parameters,
            "frozen_codec_parameters": contract.frozen_parameters,
            "sample_rate": sample_rate,
            "weights_revision": item["revision"],
            "free_rollout": generated,
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--packs",
        nargs="+",
        choices=("pocket", "sana", "sana_video"),
        default=["pocket", "sana", "sana_video"],
    )
    parser.add_argument("--output", default="")
    parser.add_argument(
        "--roundtrip",
        action="store_true",
        help="Verify exact frozen codec encode/decode and synthetic RMSE",
    )
    parser.add_argument(
        "--rollout",
        action="store_true",
        help="Actually generate raw-facets-conditioned speech/image/video",
    )
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--video-height", type=int, default=704)
    parser.add_argument("--video-width", type=int, default=1280)
    parser.add_argument("--video-frames", type=int, default=81)
    parser.add_argument("--steps", type=int, default=16)
    args = parser.parse_args()
    report = probe(
        args.manifest,
        device=args.device,
        packs=tuple(args.packs),
        rollout=args.rollout,
        image_size=args.image_size,
        video_height=args.video_height,
        video_width=args.video_width,
        video_frames=args.video_frames,
        steps=args.steps,
        roundtrip=args.roundtrip,
    )
    value = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        from pathlib import Path

        Path(args.output).write_text(value + "\n", encoding="utf-8")
    print(value)


if __name__ == "__main__":
    main()
