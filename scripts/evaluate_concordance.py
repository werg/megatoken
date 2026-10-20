#!/usr/bin/env python3
"""Score ACTUAL free-rendered, held-out concordant results from an immutable JSONL.

Does not ask a teacher for gold text or render teacher targets through R.
Each example must name its source-family split and recorded renderer sampler
settings. Predictions may be sampled by ServingSession or other production
harnesses; this script scores only verified tensors/strings already emitted.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from megatoken.training.concordance_eval import (
    SampledEvalLedger,
    audit_source_prefix,
    clock_sync_ms,
    compare_endpoints,
    masked_reconstruction,
    preserved_region_error,
    transcript_metrics,
)


def _tensor(path: str, root: Path) -> np.ndarray:
    candidate = (root / path).resolve()
    if not candidate.is_relative_to(root.resolve()):
        raise ValueError("Verifier paths must stay within the allowed evaluation root")
    if candidate.suffix != ".npy":
        raise ValueError("Offline evaluator accepts only explicit pinned .npy arrays")
    result = np.load(candidate, allow_pickle=False)
    if result.size == 0 or not np.issubdtype(result.dtype, np.number):
        raise ValueError("Verifier array must contain numeric physical output values")
    return result


def one_case(case: dict, root: Path) -> dict[str, float]:
    task = case["task_family"]
    metrics: dict[str, float] = {}
    if "text_reference" in case:
        metrics.update(transcript_metrics(case["text_reference"], case["text_prediction"]))
    if "media_prediction" in case and "media_target" in case:
        pred = _tensor(case["media_prediction"], root)
        target = _tensor(case["media_target"], root)
        mask = _tensor(case["valid_mask"], root) if "valid_mask" in case else None
        metrics.update(masked_reconstruction(pred, target, mask))
    if "first_endpoint" in case:
        clip = _tensor(case["media_prediction"], root)
        first = _tensor(case["first_endpoint"], root)
        last = _tensor(case["last_endpoint"], root)
        metrics.update(compare_endpoints(
            clip, first, last, displayed_indices=case.get("displayed_indices"),
        ))
    if "editable_mask" in case:
        edited = _tensor(case["media_prediction"], root)
        baseline = _tensor(case["pre_edit_media"], root)
        mask = _tensor(case["editable_mask"], root)
        metrics["preserved_region_mae"] = preserved_region_error(baseline, edited, mask)
    if "audio_onsets_ms" in case:
        metrics.update(clock_sync_ms(case["audio_onsets_ms"],
                                      case["video_onsets_ms"],
                                      offset_ms=case.get("av_offset_ms", 0)))
    if "available_until_ms" in case:
        audit_source_prefix(case["available_until_ms"], case["evidence_end_ms"])
        metrics["no_future_prefix_violations"] = 0.0
    if not metrics:
        raise ValueError(f"No supported independent physical/text verifier for {task}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", required=True,
                        help="JSONL produced by actual free rollouts, not teacher reconstructions")
    parser.add_argument("--media-root", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    root = Path(args.media_root).resolve()
    samples = Path(args.samples)
    target = Path(args.report)
    if target.exists():
        raise FileExistsError("Do not overwrite an existing evaluation report")
    ledger = SampledEvalLedger()
    for lineno, line in enumerate(samples.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            case = json.loads(line)
            ledger.add(
                task_id=case["task_family"],
                source_family=case["source_family"],
                split=case["split"],
                metrics=one_case(case, root),
                evidence={
                    "renderer_seed": case["renderer_seed"],
                    "sampler_steps": case["sampler_steps"],
                    "pack_id": case.get("pack_id"),
                    "codec_id": case.get("codec_id"),
                    "sampling_mode": case.get("sampling_mode", "sampled"),
                    "source_row": case.get("source_row", None),
                },
            )
        except (TypeError, KeyError, ValueError) as err:
            raise ValueError(f"Cannot score sample line {lineno}: {err}") from err
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(ledger.report(), indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
