#!/usr/bin/env python3
"""Independently transcribe actual decoded speech and test wrong-caption controls.

The evaluator is version-pinned before loading, never used as a training input,
and calibrated on archived publisher recordings with human transcripts.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

from megatoken.io import sha256_file, write_json
from megatoken.omni.evaluation import character_error_rate, word_error_rate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--caption", type=Path, required=True)
    parser.add_argument(
        "--calibration",
        type=Path,
        required=True,
        help="Preserved acquisition.json with human transcripts",
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model", default="openai/whisper-base")
    parser.add_argument("--revision")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=False)
    from megatoken.hub import install

    install()
    import transformers
    from huggingface_hub import HfApi
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    revision = args.revision or HfApi().model_info(args.model).sha
    if not revision or len(revision) != 40:
        raise ValueError("An immutable recognizer revision is required")
    write_json(
        args.root / "evaluator.json",
        {
            "repository": args.model,
            "revision": revision,
            "transformers_version": transformers.__version__,
            "torch_version": torch.__version__,
            "rate_conversion": "scipy polyphase resampling from the actual WAV clock to 16000 Hz",
            "normalization": "lowercase whitespace words / lowercase characters; no target prompt",
            "language_policy": "English for generated speech; publisher source language for calibration",
            "reference": "https://huggingface.co/openai/whisper-base",
        },
    )
    processor = WhisperProcessor.from_pretrained(args.model, revision=revision)
    model = (
        WhisperForConditionalGeneration.from_pretrained(args.model, revision=revision)
        .to(args.device)
        .eval()
    )

    @torch.inference_mode()
    def recognize(path, language):
        samples, rate = sf.read(path, dtype="float32", always_2d=True)
        if samples.shape[1] != 1 or not np.isfinite(samples).all():
            raise ValueError("Evaluation requires a finite original mono stream")
        gcd = math.gcd(rate, 16000)
        samples = resample_poly(samples[:, 0], 16000 // gcd, rate // gcd).astype(np.float32)
        # Explicitly bound this short-clip verifier; never silently crop.
        if len(samples) > 30 * 16000:
            raise ValueError(
                "Short-clip verifier needs explicit chunking for recordings over 30 seconds"
            )
        features = processor(samples, sampling_rate=16000, return_tensors="pt")
        generated = model.generate(
            features.input_features.to(args.device),
            attention_mask=features.attention_mask.to(args.device)
            if "attention_mask" in features
            else None,
            language=language,
            task="transcribe",
            max_new_tokens=256,
        )
        return processor.batch_decode(generated, skip_special_tokens=True)[0].strip()

    calibration = []
    for source in json.loads(args.calibration.read_text())["sources"][:4]:
        text = recognize(source["decoded_path"], "fr")
        reference = source["publisher_transcript"]
        correct = word_error_rate(reference, text)
        wrong = word_error_rate("A deliberately unrelated caption about orbital mechanics.", text)
        calibration.append(
            {
                "audio_sha256": sha256_file(source["decoded_path"]),
                "family_id": source["family_id"],
                "transcript": text,
                "human_reference": reference,
                "word_error_rate": correct,
                "mismatched_caption_word_error_rate": wrong,
                "negative_control_separated": wrong > correct,
            }
        )
    actual = recognize(args.audio, "en")
    caption_bytes = args.caption.read_bytes()
    try:
        caption = caption_bytes.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        caption = None
    write_json(
        args.root / "report.json",
        {
            "format": "independent-decoded-speech/v1",
            "audio_sha256": sha256_file(args.audio),
            "caption_sha256": sha256_file(args.caption),
            "recognized_generated_speech": actual,
            "generated_caption_utf8": caption,
            "generated_caption_is_valid_text": caption is not None,
            "generated_speech_caption_word_error_rate": None
            if caption is None
            else word_error_rate(caption, actual),
            "generated_speech_caption_character_error_rate": None
            if caption is None
            else character_error_rate(caption, actual),
            "calibration_on_human_transcripts": calibration,
            "all_negative_controls_separated": all(
                c["negative_control_separated"] for c in calibration
            ),
            "human_review": "pending; evaluator output is not a human review",
            "quality_pass": False,
        },
    )


if __name__ == "__main__":
    main()
