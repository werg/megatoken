#!/usr/bin/env python3
"""Offline native-codec target export; NOT dataset acquisition or training.

Waveforms/images/videos must already be quality-checked, normalized, aligned
and saved as local float32 .npy arrays. No model is downloaded at invocation.
Codec checkpoint SHA manifests are verified before any extraction.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from megatoken.data.media_assets import MediaRecord, TensorAssetStore
from megatoken.omni.loader import read_locked_manifest


def _distribution_mean(encoded):
    if hasattr(encoded, "latent_dist"):
        return encoded.latent_dist.mode()
    if hasattr(encoded, "latents"):
        return encoded.latents
    if isinstance(encoded, tuple):
        return _distribution_mean(encoded[0])
    if torch.is_tensor(encoded):
        return encoded
    if hasattr(encoded, "mode"):
        return encoded.mode()
    raise TypeError("Unrecognized donor codec output; pin matching implementation")


def cache_codec(
    *, manifest: str, mode: str, source_file: str, root: str,
    extractor: str, sample_rate: int = 0, fps_num: int = 0,
    fps_den: int = 1, artifact_version: str = "",
    source_family: str = "",
):
    base, packs = read_locked_manifest(manifest)
    if mode not in ("pocket", "sana", "sana_video") or mode not in packs:
        raise ValueError("Target codec requires an available locked donor pack")
    if not extractor or not artifact_version or not source_family:
        raise ValueError("Native codec assets require identity, ancestry and pinned extractor")
    array = np.load(source_file, allow_pickle=False).astype(np.float32)
    if not np.isfinite(array).all():
        raise ValueError("Source contains nonfinite coordinates")
    modality = {"pocket": "audio", "sana": "image", "sana_video": "video"}[mode]
    expected = {"audio": 1, "image": 3, "video": 4}[modality]
    if array.ndim != expected:
        raise ValueError("Native source requires mono samples, CHW RGB or TCHW RGB")
    if mode != "pocket" and (array.shape[-3] != 3 or np.min(array) < 0
                                  or np.max(array) > 1):
        raise ValueError("Visual source must be normalized RGB float [0,1]")
    if mode == "pocket":
        if sample_rate != 24000:
            raise ValueError("Pocket codec only accepts explicitly resampled 24kHz mono")
        from pocket_tts import TTSModel
        if not packs["pocket"].get("config"):
            raise ValueError("Pocket configuration missing")
        donor = TTSModel.load_model(
            config=str(base / packs["pocket"]["config"]),
            checkpoint=str(base / packs["pocket"]["checkpoint"])
            if packs["pocket"].get("checkpoint") else None,
        )
        codec = donor.mimi.eval()
        with torch.inference_mode():
            latent = codec.encode_to_latent(
                torch.from_numpy(array.copy())[None, None]
            )[0].cpu().float().numpy()
        valid_count = latent.shape[0]
    else:
        from diffusers import SanaPipeline, SanaVideoPipeline
        pipe_type = SanaPipeline if mode == "sana" else SanaVideoPipeline
        pipe = pipe_type.from_pretrained(
            str(base / packs[mode]["snapshot"]), local_files_only=True,
            torch_dtype=torch.float32,
        )
        codec = pipe.vae.eval()
        expected_class = "AutoencoderDC" if mode == "sana" else "AutoencoderKLLTX2Video"
        if type(codec).__name__ != expected_class:
            raise ValueError("Native checkpoint/codec family mismatch")
        pixels = torch.from_numpy(array.copy())
        if mode == "sana":
            pixels = pixels[None]
        else:
            pixels = pixels.permute(1, 0, 2, 3)[None]
        with torch.inference_mode():
            encoded = codec.encode(pixels * 2 - 1, return_dict=True)
            latent = _distribution_mean(encoded)
            if mode == "sana":
                latent = latent * float(codec.config.scaling_factor)
                valid_count = 0  # Image codec has no temporal padding axis.
                latent = latent[0].cpu().float().numpy()
            else:
                channels = int(codec.config.latent_channels)
                mean = codec.latents_mean.view(1, channels, 1, 1, 1).to(latent)
                std = codec.latents_std.view(1, channels, 1, 1, 1).to(latent)
                latent = ((latent - mean) / std)[0].permute(1, 0, 2, 3)
                valid_count = latent.shape[0]
                latent = latent.cpu().float().numpy()
    store = TensorAssetStore(root)
    input_ref = store.put(
        array, modality=modality, kind="raw", sample_rate=sample_rate,
        fps_num=fps_num, fps_den=fps_den,
        valid_count=array.shape[0],
        normalization="mono-fp32" if modality == "audio" else "rgb01-fp32",
        artifact_version=artifact_version, source_family=source_family,
    )
    target_ref = store.put(
        latent, modality=modality, kind="codec_latents", extractor=extractor,
        sample_rate=sample_rate, fps_num=fps_num, fps_den=fps_den,
        valid_count=valid_count,
        normalization="pocket-mimi-continuous" if mode == "pocket"
        else "sana-latent-scaled" if mode == "sana" else "ltx2-normalized",
        artifact_version=artifact_version, source_family=source_family,
    )
    # This descriptor represents a supervised event, not an observed decoder
    # bypass. Target latents never enter prospective P or R from source features.
    return MediaRecord(
        modality=modality, source=input_ref, target=target_ref,
        source_views=(target_ref,) if mode == "pocket" else (),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--mode", choices=("pocket", "sana", "sana_video"), required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--extractor", required=True)
    p.add_argument("--sample-rate", type=int, default=0)
    p.add_argument("--fps-num", type=int, default=0)
    p.add_argument("--fps-den", type=int, default=1)
    p.add_argument("--artifact-version", required=True)
    p.add_argument("--source-family", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    reference = cache_codec(
        manifest=args.manifest, mode=args.mode, source_file=args.input,
        root=args.root, extractor=args.extractor,
        sample_rate=args.sample_rate, fps_num=args.fps_num,
        fps_den=args.fps_den, artifact_version=args.artifact_version,
        source_family=args.source_family,
    )
    Path(args.output).write_bytes(reference.to_bytes() + b"\n")


if __name__ == "__main__":
    main()
