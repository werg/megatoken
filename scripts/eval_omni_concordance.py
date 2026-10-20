#!/usr/bin/env python3
"""Evaluate sampled multi-track media/edited artifacts, never gold target peers.

Input manifest declares relative decoded output tensors or generated text, and
relations that were actually requested. Independent verification is optional
but missing evaluations are explicitly unmeasured. No model is run here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from megatoken.omni.bundles import Relation
from megatoken.omni.concordance import SampledEvent, score_sampled_relations


def evaluate_manifest(path: str, *, asr_path: str | None = None):
    source = Path(path).resolve()
    manifest = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not isinstance(manifest.get("events"), list):
        raise ValueError("Evaluation needs explicit sampled event records")
    outputs = {}
    for description in manifest["events"]:
        if not isinstance(description, dict):
            raise ValueError("Sampled event descriptors must be objects")
        name, modality = description["id"], description["modality"]
        if name in outputs:
            raise ValueError("Duplicate sampled event identity")
        if modality == "text":
            if set(description) & {"path"}:
                raise ValueError("Text output must be literal generated text")
            value = description["text"]
        else:
            relative = Path(description["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Sampled media must be an in-manifest .npy path")
            actual = (source.parent / relative).resolve()
            if not actual.is_relative_to(source.parent) or actual.suffix != ".npy":
                raise ValueError("Invalid sampled output array path")
            value = np.load(actual, allow_pickle=False)
            digest = hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
            if description.get("sha256") != digest:
                raise ValueError("Sampled decoded array digest differs from manifest")
        mask = None
        if description.get("change_mask"):
            region = Path(description["change_mask"])
            if region.is_absolute() or ".." in region.parts:
                raise ValueError("Invalid requested edit-region path")
            mask = np.load(source.parent / region, allow_pickle=False).astype(bool)
        outputs[name] = SampledEvent(
            name, modality, value,
            detected_events=tuple(
                (event["name"], float(event["time"]))
                for event in description.get("detected_events", ())
            ),
            changed_region=mask,
            provenance=description.get("provenance", "sampled"),
        )
    relations = tuple(Relation(**r) for r in manifest.get("relations", ()))
    speech_oracle = None
    if asr_path:
        asr = json.loads(Path(asr_path).read_text(encoding="utf-8"))
        if not asr.get("verifier") or not asr.get("revision"):
            raise ValueError("Independent ASR results require a named pinned verifier")
        transcripts = asr["by_audio_sha256"]
        def speech_oracle(audio):
            digest = hashlib.sha256(np.ascontiguousarray(audio).tobytes()).hexdigest()
            if digest not in transcripts:
                raise ValueError("ASR verification does not match sampled audio bytes")
            return transcripts[digest]
    report = score_sampled_relations(
        relations, outputs, speech_recognizer=speech_oracle,
    )
    return {
        "schema_version": 1,
        "evaluated_pairs": report.evaluated_pairs,
        "measured": report.measured,
        "unmeasured": report.unmeasured,
        "output_ids": sorted(outputs),
        "source_family": manifest.get("source_family", ""),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--asr", default=None, help="Independent ASR results keyed by waveform digest")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = evaluate_manifest(args.manifest, asr_path=args.asr)
    Path(args.output).write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
