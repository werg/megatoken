"""Frozen Whistle, Pocket-Mimi, and exported MiniCPM-V 4.7 input stems.

Everything here produces ordered features, masks, and explicit physical time.
No donor language model, transcript, future target, or hidden decoder KV is
exposed as a Megatoken semantic input.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class OrderedView:
    features: Tensor          # [1, positions, donor_width]
    valid: Tensor             # [1, positions]
    timestamps: Tensor        # [positions] native-media seconds
    extractor: str
    def __post_init__(self):
        if (self.features.ndim != 3 or self.features.shape[0] != 1
                or self.valid.shape != self.features.shape[:2]
                or self.valid.dtype != torch.bool
                or self.timestamps.shape != (self.features.shape[1],)
                or not bool(torch.isfinite(self.features).all())
                or not bool(torch.isfinite(self.timestamps).all())):
            raise ValueError("Frozen ordered feature geometry or clock is invalid")


class WhistleFeatures:
    """Whistle encoder embeddings, one frame every 80 ms, no ASR transcript."""

    def __init__(self, weights: str, *, feature_width: int,
                 extractor: str, max_window_seconds=30):
        from needle import Whistle
        path = Path(weights)
        if not path.is_file() or path.suffix != ".cact":
            raise ValueError("Whistle requires an explicit local .cact checkpoint")
        if feature_width <= 0 or not extractor:
            raise ValueError("Whistle output needs pinned width and revision")
        self.model = Whistle(weights=str(path))
        self.feature_width, self.extractor = feature_width, extractor
        self.window = min(30, max_window_seconds) * 16000

    def __call__(self, signal: np.ndarray, *, sample_rate=16000,
                 available_samples: int | None = None) -> OrderedView:
        if sample_rate != 16000:
            raise ValueError("Whistle needs an explicitly resampled 16kHz mono view")
        audio = np.asarray(signal, dtype=np.float32)
        if audio.ndim != 1 or not np.isfinite(audio).all():
            raise ValueError("Whistle expects a finite mono waveform")
        if available_samples is not None:
            if not 0 < available_samples <= len(audio):
                raise ValueError("Future audio is not an available source prefix")
            audio = audio[:available_samples]
        if not len(audio):
            raise ValueError("An audio event cannot be empty")
        frames, times = [], []
        for offset in range(0,len(audio),self.window):
            chunk = audio[offset:offset+self.window]
            values = np.asarray(self.model.embed(chunk), dtype=np.float32).reshape(-1)
            if values.size % self.feature_width or not values.size:
                raise ValueError("Whistle embed size differs from pinned donor geometry")
            features = values.reshape(-1,self.feature_width)
            frames.append(features)
            times.extend((offset/16000) + 0.08*np.arange(len(features)))
        merged = np.concatenate(frames, axis=0)
        return OrderedView(
            torch.from_numpy(merged.copy())[None],
            torch.ones((1,len(merged)),dtype=torch.bool),
            torch.tensor(np.asarray(times),dtype=torch.float32), self.extractor
        )


class PocketAcousticFeatures:
    """Pocket's matching continuous acoustic codec view at its native rate."""
    def __init__(self, mimi: nn.Module, extractor: str):
        self.mimi, self.extractor = mimi, extractor
        self.sample_rate = int(mimi.sample_rate)
        self.frame_rate = float(mimi.frame_rate)
        if not extractor or self.sample_rate != 24000:
            raise ValueError("Pocket acoustic feature identity/rate mismatch")
        self.mimi.eval()
        for p in self.mimi.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def __call__(self, waveform: Tensor, *, sample_rate: int,
                 available_samples: int | None = None):
        if sample_rate != self.sample_rate or waveform.ndim != 2 or waveform.shape[0] != 1:
            raise ValueError("Pocket acoustic view expects [1, native-rate mono samples]")
        if available_samples is not None:
            if not 0 < available_samples <= waveform.shape[1]:
                raise ValueError("Acoustic view cannot see future media")
            waveform = waveform[:, :available_samples]
        latents = self.mimi.encode_to_latent(waveform[:, None])
        seconds = torch.arange(latents.shape[1],device=latents.device)/self.frame_rate
        return OrderedView(latents, torch.ones(latents.shape[:2],device=latents.device,
                                              dtype=torch.bool), seconds,self.extractor)


def verify_minicpm_export(directory: str) -> dict:
    """Check trained tower/merger bytes and the original exported revision."""
    import hashlib
    import json
    import re

    root = Path(directory).resolve()
    receipt = root / "export_receipt.json"
    if not receipt.is_file():
        raise FileNotFoundError("MiniCPM export requires a pinned source receipt")
    value = json.loads(receipt.read_text(encoding="utf-8"))
    if (value.get("schema_version") != 1
            or not re.fullmatch(r"[0-9a-f]{40}", value.get("source_revision", ""))):
        raise ValueError("Invalid pinned MiniCPM export revision")
    expected = value.get("sha256")
    names = {"vision_tower.safetensors", "merger.safetensors", "config.json"}
    if not isinstance(expected, dict) or set(expected) != names:
        raise ValueError("Visual export must include exact tower, merger and config")
    for name in names:
        artifact = root / name
        if not artifact.is_file() or artifact.is_symlink():
            raise ValueError("Missing or symlinked pretrained visual export")
        if hashlib.sha256(artifact.read_bytes()).hexdigest() != expected[name]:
            raise ValueError("MiniCPM extracted parameter SHA256 mismatch")
    return value


class MiniCPMVisualFeatures(nn.Module):
    """Standalone exported MiniCPM vision tower AND trained merger (no LLM)."""

    def __init__(self, config, tower, merger, extractor: str):
        super().__init__()
        if not extractor:
            raise ValueError("Vision tower needs pinned revision")
        self.config, self.tower, self.merger = config, tower.eval(), merger.eval()
        self.extractor = extractor
        for p in self.parameters():
            p.requires_grad_(False)

    @classmethod
    def from_local_export(cls, directory: str, *, extractor: str):
        verify_minicpm_export(directory)
        from safetensors.torch import load_file
        from transformers import MiniCPMV4_7Config
        from transformers.models.minicpmv4_7.modeling_minicpmv4_7 import (
            MiniCPMV4_7Merger,
            MiniCPMV4_7VisionModel,
        )
        path = Path(directory)
        config = MiniCPMV4_7Config.from_pretrained(path, local_files_only=True)
        tower = MiniCPMV4_7VisionModel(config.vision_config)
        merger = MiniCPMV4_7Merger(config)
        tower.load_state_dict(load_file(path/"vision_tower.safetensors"),strict=True)
        merger.load_state_dict(load_file(path/"merger.safetensors"),strict=True)
        return cls(config,tower,merger,extractor)

    @torch.no_grad()
    def forward(self, pixel_values: Tensor, target_sizes: Tensor, *,
                downsample_mode: str = "16x", frame_times=None):
        if downsample_mode not in ("16x","4x"):
            raise ValueError("Unknown MiniCPM downsampling mode")
        if target_sizes.ndim != 2 or target_sizes.shape[-1] != 2:
            raise ValueError("MiniCPM requires actual patch grids")
        if frame_times is None:
            frame_times = [0.0] * len(target_sizes)
        if len(frame_times) != len(target_sizes):
            raise ValueError("Every image/frame needs an actual local timestamp")
        use_vit_merger = downsample_mode != "4x"
        result = self.tower(
            pixel_values.to(self.tower.dtype), target_sizes=target_sizes,
            use_vit_merger=use_vit_merger,
        )
        grid = target_sizes // 2 if use_vit_merger else target_sizes
        patches = self.merger(result.last_hidden_state,grid)
        features = torch.cat(patches,dim=0)
        times = torch.cat(
            (torch.full((x.shape[0],),float(t),device=features.device)
             for x,t in zip(patches,frame_times,strict=True)),dim=0
        )
        return OrderedView(features[None],
                           torch.ones((1,features.shape[0]),dtype=torch.bool,
                                      device=features.device),
                           times, self.extractor)


def export_minicpm_submodules(parent_snapshot: str, destination: str):
    """Extract matching pretrained tower and merger tensors from HF safetensors.

    Stream only selected tensor keys from the parent's sharded files; do not
    allocate or instantiate its language model. Requires a local immutable
    parent snapshot with safetensors and saved config.
    """
    import json
    from shutil import copyfile

    from safetensors import safe_open
    from safetensors.torch import save_file
    src, dst = Path(parent_snapshot),Path(destination)
    config = src/"config.json"
    if not config.is_file():
        raise FileNotFoundError("MiniCPM parent config.json missing")
    index=src/"model.safetensors.index.json"
    if index.exists():
        names=sorted(set(json.loads(index.read_text())["weight_map"].values()))
    else:
        names=["model.safetensors"]
    grouped={"vision_tower": {}, "merger": {}}
    for name in names:
        with safe_open(src/name,framework="pt",device="cpu") as handle:
            for key in handle.keys():
                for part in grouped:
                    prefix="model."+part+"."
                    if key.startswith(prefix):
                        grouped[part][key[len(prefix):]]=handle.get_tensor(key)
    if any(not value for value in grouped.values()):
        raise ValueError("Parent snapshot lacks trained vision tower/merger tensors")
    dst.mkdir(parents=True,exist_ok=True)
    for key,weights in grouped.items():
        save_file(weights,dst/(key+".safetensors"))
    copyfile(config,dst/"config.json")
    return {k:len(v) for k,v in grouped.items()}
