#!/usr/bin/env python3
"""Offline pinned Wav2Vec2/SigLIP2 feature caching into media asset descriptors.

This helper is opt-in and runs entirely outside the model/training process.
It never adds another recurrent transformer or hidden KV state. Donor weights
must exist locally unless --allow-download is explicitly provided.
Install optional dependencies: pip install transformers pillow
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch

from megatoken.data.media_assets import MediaRecord, TensorAssetStore


def _vision_features(model, processor, raw, device):
    from PIL import Image

    frames = raw[None] if raw.ndim == 3 else raw
    output = []
    with torch.inference_mode():
        for frame in frames:
            pixels = np.clip(frame.transpose(1, 2, 0), 0, 1)
            image = Image.fromarray(np.rint(pixels * 255).astype(np.uint8), "RGB")
            arguments = processor(images=image, return_tensors="pt")
            arguments = {key: val.to(device) for key, val in arguments.items()
                         if isinstance(val, torch.Tensor)}
            features = model(**arguments).last_hidden_state
            if features.ndim != 3 or features.shape[0] != 1:
                raise ValueError("Vision donor must return ordered patch features")
            output.append(features[0].float().cpu().numpy())
    return np.concatenate(output, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Prepared dataset containing media/")
    parser.add_argument("--record", required=True, help="Canonical input MediaRecord JSON file")
    parser.add_argument("--output", required=True, help="New canonical MediaRecord JSON file")
    parser.add_argument("--model", required=True, help="Pinned Hugging Face model repository ID")
    parser.add_argument("--revision", required=True, help="Immutable 40-character commit SHA")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()

    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        raise ValueError("Feature donor revision must be an immutable full 40-digit git SHA")
    store = TensorAssetStore(args.root)
    original = MediaRecord.from_bytes(Path(args.record).read_bytes())
    if original.source.kind != "raw":
        raise ValueError("Offline extraction requires native media input, not existing features")
    modality = original.modality
    if modality == "audio" and args.model != "facebook/wav2vec2-base-960h":
        raise ValueError("Initial speech adapter only supports the reviewed Wav2Vec2 donor")
    if modality in {"image", "video"} and args.model != "google/siglip2-base-patch16-224":
        raise ValueError("Initial visual adapter only supports the reviewed SigLIP2 donor")

    import transformers

    device = torch.device(args.device)
    local_only = not args.allow_download
    common = dict(revision=args.revision, local_files_only=local_only, trust_remote_code=False)
    if modality == "audio":
        model = transformers.Wav2Vec2Model.from_pretrained(args.model, **common)
        model.eval().to(device)
        raw = store.load(original.source)
        with torch.inference_mode():
            inputs = torch.from_numpy(raw.copy()).to(device)[None]
            features = model(input_values=inputs).last_hidden_state[0].float().cpu().numpy()
    else:
        model = transformers.Siglip2VisionModel.from_pretrained(args.model, **common)
        processor = transformers.AutoImageProcessor.from_pretrained(args.model, **common)
        model.eval().to(device)
        features = _vision_features(model, processor, store.load(original.source), device)
    if features.ndim != 2 or not np.isfinite(features).all():
        raise ValueError("Donor output is not an ordered finite feature sequence")
    extractor = f"hf:{args.model}@{args.revision}"
    cached = store.put(
        features, modality=modality, kind="features", extractor=extractor,
        sample_rate=original.source.sample_rate,
        fps_num=original.source.fps_num, fps_den=original.source.fps_den,
    )
    result = MediaRecord(modality, cached, original.target)
    path = Path(args.output)
    if path.exists():
        raise FileExistsError("Feature export must use a new immutable output path")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(result.to_bytes())
    print(json.dumps({
        "extractor": extractor,
        "feature_dim": int(features.shape[-1]),
        "feature_tokens": int(features.shape[0]),
        "sha256": cached.sha256, "media_record": str(path),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
