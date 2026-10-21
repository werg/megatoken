#!/usr/bin/env python3
"""Probe the locked realizers on real weights: codec round trip, objective, rollout.

No data is prepared and no weights change. Run on an accelerator after locking
the pack manifest (scripts/lock_pretrained_omni.py).
"""

from __future__ import annotations

import argparse
import dataclasses
import json

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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pack", action="append", choices=("pocket", "sana", "sana_video"))
    parser.add_argument("--rollout", action="store_true")
    parser.add_argument("--steps", type=int, default=8)
    args = parser.parse_args()
    names = tuple(args.pack or ("pocket", "sana", "sana_video"))
    report = probe(
        args.manifest, device=args.device, names=names, rollout=args.rollout, steps=args.steps
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
