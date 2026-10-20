#!/usr/bin/env python3
"""Offline frozen view/codec extraction. Preparation is a separate explicit act.

No downloads, mutable model aliases, or training are performed here.
Each generated asset carries the exact requested preprocessing and codec name.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from megatoken.data.media_assets import TensorAssetStore
from megatoken.omni.perception import (
    WhistleFeatures, PocketAcousticFeatures, MiniCPMVisualFeatures,
)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",required=True)
    p.add_argument("--input",required=True,help="Local .npy waveform or preprocessed ViT input .pt")
    p.add_argument("--mode",required=True,choices=("whistle","pocket","minicpm"))
    p.add_argument("--model",help="Optional explicit path; must match the pinned pack lock")
    p.add_argument("--manifest",required=True,help="Complete SHA256 pack lock")
    p.add_argument("--extractor",required=True)
    p.add_argument("--width",type=int,default=0)
    p.add_argument("--sample-rate",type=int,default=0)
    p.add_argument("--profile",choices=("16x","4x"),default="16x")
    p.add_argument("--available-samples",type=int,default=None)
    p.add_argument("--output",required=True)
    a=p.parse_args()
    from megatoken.omni.loader import read_locked_manifest
    base, packs = read_locked_manifest(a.manifest)
    pack_name = {"whistle": "whistle", "pocket": "pocket", "minicpm": "minicpm"}[a.mode]
    if pack_name not in packs:
        raise ValueError(f"Perceptual donor {pack_name} absent from lockfile")
    item = packs[pack_name]
    reference = {"whistle": item.get("weights"), "pocket": item.get("config"),
                 "minicpm": item.get("snapshot")}[a.mode]
    if not reference:
        raise ValueError("Pinned perceptual donor has no checkpoint location")
    model_path = (base / reference).resolve()
    if a.model and Path(a.model).resolve() != model_path:
        raise ValueError("Extraction model cannot differ from verified pack lock")
    a.model = str(model_path)
    store=TensorAssetStore(a.root)
    if a.mode=="whistle":
        encoder=WhistleFeatures(a.model,feature_width=a.width,extractor=a.extractor)
        view=encoder(np.load(a.input),sample_rate=a.sample_rate,
                     available_samples=a.available_samples)
    elif a.mode=="pocket":
        from pocket_tts import TTSModel
        if not Path(a.model).is_file():
            raise FileNotFoundError("Require local Pocket YAML, not a moving model alias")
        donor=TTSModel.load_model(config=a.model)
        view=PocketAcousticFeatures(donor.mimi,a.extractor)(
            torch.from_numpy(np.load(a.input)).float()[None],
            sample_rate=a.sample_rate,available_samples=a.available_samples,
        )
    else:
        bundle=torch.load(a.input,map_location="cpu",weights_only=True)
        encoder=MiniCPMVisualFeatures.from_local_export(a.model,extractor=a.extractor)
        view=encoder(bundle["pixel_values"],bundle["target_sizes"],
                     downsample_mode=a.profile,frame_times=bundle.get("frame_times"))
    asset=store.put(view.features[0].detach().cpu().numpy(),
                    modality="audio" if a.mode in ("whistle","pocket") else "image",
                    kind="features",extractor=a.extractor,
                    sample_rate=a.sample_rate if a.mode != "minicpm" else 0,
                    valid_count=int(view.valid.sum()),
                    normalization="frozen-ordered-output-v1")
    Path(a.output).write_text(json.dumps({
        "asset":__import__("dataclasses").asdict(asset),
        "timestamps_seconds":view.timestamps.cpu().tolist(),
        "view_profile":a.profile if a.mode=="minicpm" else a.mode,
    },sort_keys=True,indent=2),encoding="utf-8")

if __name__=="__main__":
    main()
