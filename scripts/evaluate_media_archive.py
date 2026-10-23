#!/usr/bin/env python3
"""Independently score archived decoded speech, text, and human-reference controls."""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from megatoken.io import sha256_file, write_json
from megatoken.omni.evaluation import character_error_rate, word_error_rate
from megatoken.omni.recognition import IndependentWhisper


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--calibration-source", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--sample-rate", type=int, default=24000)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    args.root.mkdir(parents=True, exist_ok=False)
    from megatoken.hub import install

    install()
    recognizer = IndependentWhisper(device=args.device)
    write_json(args.root / "evaluator.json", recognizer.metadata)
    calibration = []
    for source in json.loads(args.calibration_source.read_text())["sources"][:4]:
        samples, rate = sf.read(source["target_audio"], dtype="float32")
        recognized = recognizer.transcribe(samples, sample_rate=rate)
        reference = source["publisher_translation"]
        correct = word_error_rate(reference, recognized)
        wrong = word_error_rate(
            "a deliberately unrelated statement about orbital mechanics", recognized
        )
        calibration.append(
            {
                "source_sha256": sha256_file(source["target_audio"]),
                "reference": reference,
                "recognized": recognized,
                "word_error": correct,
                "wrong_caption_word_error": wrong,
                "negative_control_separated": wrong > correct,
            }
        )
    results = []
    for path in sorted(args.archive.rglob("report.json")):
        report = json.loads(path.read_text())
        if report.get("format") != "decoded-quality/v1":
            continue
        payloads = {}
        for tape_path in path.parent.glob("selected-*.json"):
            tape = json.loads(tape_path.read_text())
            payloads.update(
                (identity, base64.b64decode(payload, validate=True))
                for identity, payload in zip(tape["event_ids"], tape["payloads"], strict=True)
            )
        for relation in report["declared_relations"]:
            if relation["kind"] != "same_spoken_content":
                continue
            left, right = relation["left"], relation["right"]
            assets = report["sampled_assets"]
            if assets.get(right, {}).get("modality") == "audio":
                speech, text = right, left
            elif assets.get(left, {}).get("modality") == "audio":
                speech, text = left, right
            else:
                results.append(
                    {"episode_id": report["episode_id"], "error": "missing decoded speech"}
                )
                continue
            # Asset paths are relative to their path's evaluator root, which may
            # itself be nested under a joint/sequential comparison directory.
            relative = Path(assets[speech]["path"])
            root = path.parent.parent
            audio_path = (root / relative).resolve()
            if (
                not audio_path.is_relative_to(root.resolve())
                or sha256_file(audio_path) != assets[speech]["sha256"]
            ):
                raise ValueError("Archived decoded media path or checksum is invalid")
            samples = np.load(audio_path, allow_pickle=False)
            recognized = recognizer.transcribe(samples, sample_rate=args.sample_rate)
            raw = payloads.get(text)
            try:
                caption = None if raw is None else raw.decode("utf-8", errors="strict")
            except UnicodeDecodeError:
                caption = None
            reference = report["references_post_generation_only"].get(text)
            result = {
                "episode_id": report["episode_id"],
                "family_id": report["family_id"],
                "speech_sha256": assets[speech]["sha256"],
                "recognized": recognized,
                "generated_caption": caption,
                "caption_valid_utf8": caption is not None,
                "speech_caption_word_error": None
                if caption is None
                else word_error_rate(caption, recognized),
                "speech_caption_character_error": None
                if caption is None
                else character_error_rate(caption, recognized),
                "human_translation_reference": reference,
                "speech_reference_word_error": None
                if reference is None
                else word_error_rate(reference, recognized),
            }
            write_json(args.root / f"sample-{len(results)}.json", result)
            results.append(result)
    if not results:
        raise ValueError("Archive contains no actual sampled speech/text relation")
    write_json(
        args.root / "report.json",
        {
            "format": "independent-archive-quality/v1",
            "sampled_pairs": results,
            "calibration_on_publisher_recordings": calibration,
            "definitions": "word/character edits per reference word/character; zero is exact agreement; invalid text and absent samples fail coverage",
            "semantic_adequacy_established": False,
            "human_review": "pending",
            "quality_pass": False,
        },
    )


if __name__ == "__main__":
    main()
