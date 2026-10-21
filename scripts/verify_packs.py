#!/usr/bin/env python3
"""Probe the locked packs on real weights, on an accelerator; no weights change.

- realizers: codec encode, training objective and gradients (check_realizer),
  optionally a free rollout;
- ``--prepare``: the complete preparation path (``MediaPreparer``) on synthetic
  text, image, speech and video files: teacher hidden states for every event,
  plus realizer latents and output geometry where the realizer pack is locked.

Run after ``scripts/acquire_packs.py``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import tempfile
import wave
from pathlib import Path

import torch

from megatoken.model.media import MediaOutputSpec
from megatoken.omni.loader import build_realizer, read_locked_manifest
from megatoken.omni.validation import check_realizer


def probe(manifest: str, *, device: str, names: tuple[str, ...], rollout: bool, steps: int):
    base, packs = read_locked_manifest(manifest)
    results = {}
    for name in names:
        realizer = build_realizer(
            base, name, packs[name], facet_count=32, facet_width=512, device=device
        )
        generator = torch.Generator(device=device).manual_seed(78117)
        if name == "pocket":
            media = 0.1 * torch.randn(1, realizer.sample_rate, device=device, generator=generator)
            spec = MediaOutputSpec("audio", sample_count=realizer.sample_rate, sample_rate=24000)
        elif name == "sana":
            media = torch.rand(1, 3, 256, 256, device=device, generator=generator)
            spec = MediaOutputSpec("image", height=256, width=256)
        else:
            media = torch.rand(1, 9, 3, 256, 256, device=device, generator=generator)
            spec = MediaOutputSpec("video", frames=9, height=256, width=256, fps_num=8)
        latents = realizer.encode(media)
        raw = torch.randn(1, 32, 512, device=device, generator=generator)
        result = {"probe": check_realizer(realizer, raw, latents).__dict__}
        if rollout:
            output = realizer.realize(raw, dataclasses.replace(spec, steps=steps))
            if not torch.isfinite(output).all():
                raise ValueError(f"{name} free rollout returned nonfinite media")
            result["rollout_shape"] = list(output.shape)
        results[name] = result
    return results


def synthetic_media(directory: Path) -> dict[str, Path]:
    """A 2 s 24 kHz tone, a 64x64 gradient image and a 9-frame (8k+1) 8 fps 64x64 clip."""
    import numpy as np
    from PIL import Image
    from torchcodec.encoders import VideoEncoder

    speech = directory / "tone.wav"
    samples = (0.3 * np.sin(2 * math.pi * 220 * np.arange(48000) / 24000) * 32767).astype("<i2")
    with wave.open(str(speech), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(24000)
        stream.writeframes(samples.tobytes())
    image = directory / "gradient.png"
    ramp = np.linspace(0, 255, 64, dtype=np.uint8)
    pixels = np.stack(np.broadcast_arrays(ramp[None], ramp[:, None], np.uint8(128)), -1)
    Image.fromarray(pixels.astype(np.uint8)).save(image)
    video = directory / "clip.mp4"
    frames = torch.stack([torch.full((3, 64, 64), 16 * i, dtype=torch.uint8) for i in range(9)])
    VideoEncoder(frames, frame_rate=8).to_file(str(video))
    return {"audio": speech, "image": image, "video": video}


def probe_preparation(manifest: str, *, device: str) -> dict:
    from megatoken.data.media_prepare import REALIZER, MediaPreparer

    with tempfile.TemporaryDirectory() as scratch:
        directory = Path(scratch)
        preparer = MediaPreparer(manifest, directory / "prepared", device=device)
        anchor = preparer.text(b"A short sentence for the byte-E anchor.")
        results = {"text": {"features": list(anchor.shape), "extractor": anchor.extractor}}
        for modality, path in synthetic_media(directory).items():
            record = preparer.media(modality, path, target=REALIZER[modality] in preparer.packs)
            results[modality] = {
                "features": list(record.source.shape),
                "latents": None if record.target is None else list(record.target.shape),
                "geometry": None
                if record.output_geometry is None
                else dataclasses.asdict(record.output_geometry),
            }
        return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pack", action="append", choices=("pocket", "sana", "sana_video"))
    parser.add_argument("--rollout", action="store_true")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    _, packs = read_locked_manifest(args.manifest)
    names = tuple(args.pack or sorted(set(packs) - {"embeddinggemma2"}))
    report = {
        "realizers": probe(
            args.manifest, device=args.device, names=names, rollout=args.rollout, steps=args.steps
        )
    }
    if args.prepare:
        report["preparation"] = probe_preparation(args.manifest, device=args.device)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
