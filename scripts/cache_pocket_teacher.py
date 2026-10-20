#!/usr/bin/env python3
"""Optional offline Pocket teacher acquisition, not a training/data run by itself.

Generates an *actual matched* waveform and exact pre-Mimi acoustic trajectory
from a locally locked Pocket checkpoint, using a separately approved voice
reference. Store the result outside model input. Requires explicit execution.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from megatoken.data.media_assets import TensorAssetStore
from megatoken.omni.loader import read_locked_manifest
from megatoken.omni.teacher import capture_pocket_teacher


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--voice-waveform", required=True,
                        help="Approved separate 24kHz mono voice reference .npy")
    parser.add_argument("--text-file", required=True,
                        help="Verified spoken form, teacher side only")
    parser.add_argument("--root", required=True)
    parser.add_argument("--source-family", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if not args.source_family or not args.version:
        raise ValueError("Teacher targets require source ancestry and version")
    base, packs = read_locked_manifest(args.manifest)
    if "pocket" not in packs:
        raise ValueError("Native speech teacher must be a verified Pocket checkpoint")
    item = packs["pocket"]
    from pocket_tts import TTSModel
    teacher = TTSModel.load_model(
        config=str(base / item["config"]),
        checkpoint=(str(base / item["checkpoint"]) if item.get("checkpoint") else None),
        quantize=False,
    ).eval()
    reference = np.load(args.voice_waveform, allow_pickle=False).astype(np.float32)
    if reference.ndim != 1 or not 1 <= len(reference) <= 30 * 24000:
        raise ValueError("Voice reference must be mono 24kHz and at most 30 seconds")
    if not np.isfinite(reference).all():
        raise ValueError("Voice reference contains invalid samples")
    text = Path(args.text_file).read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError("Teacher requires a checked spoken-form transcript")
    torch.manual_seed(args.seed)
    voice_state = teacher.get_state_for_audio_prompt(
        torch.from_numpy(reference.copy())[None],
    )
    traced = capture_pocket_teacher(
        teacher, voice_state=voice_state, spoken_text=text,
    )
    store = TensorAssetStore(args.root)
    identity = hashlib.sha256(
        (json.dumps(item["files"], sort_keys=True) + item["revision"]).encode()
    ).hexdigest()[:16]
    common = dict(
        modality="audio", source_family=args.source_family,
        artifact_version=args.version, sample_rate=24000,
    )
    voice_ref = store.put(
        reference, kind="raw", normalization="voice-reference-mono-fp32", **common,
    )
    speech = store.put(
        traced.waveform.flatten().numpy(), kind="raw",
        normalization="teacher-waveform-mono-fp32", **common,
    )
    acoustic = store.put(
        traced.mimi_latents.numpy(), kind="codec_latents",
        extractor="pocket-mimi-teacher-" + identity,
        valid_count=traced.mimi_latents.shape[0],
        normalization="pocket-mimi-continuous-predecoder", **common,
    )
    result = {
        "schema_version": 1,
        "teacher": "pocket",
        "teacher_revision": item["revision"],
        "voice_reference": asdict(voice_ref),
        "generated_waveform": asdict(speech),
        "matched_predecoder_latents": asdict(acoustic),
        "reference_provenance": "external_evidence",
        "request_provenance": "teacher_only_spoken_form",
        "teacher_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "render_seed": args.seed,
        "eos_flags": traced.eos_flags,
        "source_family": args.source_family,
        "artifact_version": args.version,
    }
    dest = Path(args.output)
    if dest.exists():
        raise FileExistsError("Teacher target manifest is immutable; choose a new version")
    dest.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(dest)


if __name__ == "__main__":
    main()
