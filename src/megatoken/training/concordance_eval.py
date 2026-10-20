"""Measured, held-out sampled-output evaluation for concordant multimodal tasks.

Verifiers are independent from the trained teacher. Nothing in this module
edits session controls, training targets or a selected output's raw Z.
Returned metrics are observations, never claims of learned benchmark quality.
"""
from __future__ import annotations

import itertools
import math
import unicodedata
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np


def text_units(value: str, *, words: bool) -> tuple[str, ...]:
    if not isinstance(value, str):
        raise TypeError("Metric inputs must decode to Unicode strings")
    value = unicodedata.normalize("NFC", value).casefold()
    if words:
        return tuple(value.split())
    return tuple(ch for ch in value if not ch.isspace())


def edit_distance(source: Sequence[str], target: Sequence[str]) -> int:
    """Dynamic-programming Levenshtein over real words/codepoints; O(m) memory."""
    prev = list(range(len(target) + 1))
    for row, left in enumerate(source, 1):
        current = [row]
        for col, right in enumerate(target, 1):
            current.append(min(
                prev[col] + 1,
                current[col-1] + 1,
                prev[col-1] + (left != right),
            ))
        prev = current
    return prev[-1]


def transcript_metrics(reference: str, hypothesis: str) -> dict[str, float]:
    word_ref = text_units(reference, words=True)
    word_hyp = text_units(hypothesis, words=True)
    char_ref = text_units(reference, words=False)
    char_hyp = text_units(hypothesis, words=False)
    if not word_ref or not char_ref:
        raise ValueError("WER/CER requires nonempty verified reference transcript")
    return {
        "wer": edit_distance(word_ref, word_hyp) / len(word_ref),
        "cer": edit_distance(char_ref, char_hyp) / len(char_ref),
        "exact": float(reference == hypothesis),
    }


def masked_reconstruction(prediction: np.ndarray, target: np.ndarray,
                          valid_mask: np.ndarray | None = None) -> dict[str, float]:
    if prediction.shape != target.shape or not prediction.size:
        raise ValueError("Decoded physical output geometry must match a held-out target")
    pred = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    if not np.isfinite(pred).all() or not np.isfinite(truth).all():
        raise ValueError("Evaluator refuses nonfinite decoded samples")
    mask = (np.ones_like(truth, dtype=bool) if valid_mask is None else
            np.broadcast_to(valid_mask, truth.shape).astype(bool))
    if not mask.any():
        raise ValueError("No valid nonpadding pixels or samples to evaluate")
    error = (pred[mask] - truth[mask]) ** 2
    mse = float(error.mean())
    return {
        "mse": mse,
        "mae": float(np.abs(pred[mask] - truth[mask]).mean()),
        "psnr_db": -10.0 * math.log10(max(mse, 1e-12)),
        "valid_count": int(mask.sum()),
    }


def compare_endpoints(clip: np.ndarray, first: np.ndarray, last: np.ndarray,
                      *, displayed_indices: Sequence[int] | None = None,
                      mask: np.ndarray | None = None) -> dict[str, float]:
    """Compare true visible frames, not padded video latent indices."""
    if clip.ndim != 4 or len(clip) < 2 or clip.shape[1] != 3:
        raise ValueError("Endpoints require displayed [T,C,H,W] video")
    if displayed_indices is not None and (
        len(displayed_indices) != len(clip) or
        any(a >= b for a, b in itertools.pairwise(displayed_indices))
    ):
        raise ValueError("Displayed time sampling must be strictly increasing")
    a = masked_reconstruction(clip[0], first, mask)
    b = masked_reconstruction(clip[-1], last, mask)
    return {
        "first_endpoint_mse": a["mse"],
        "last_endpoint_mse": b["mse"],
        "endpoint_max_mse": max(a["mse"], b["mse"]),
        "display_frames": len(clip),
    }


def preserved_region_error(before: np.ndarray, edited: np.ndarray,
                           editable_mask: np.ndarray) -> float:
    """Pixel/latent-coordinate preservation excludes only declared editable region."""
    if before.shape != edited.shape:
        raise ValueError("Edit verifier requires aligned before/after grids")
    retained = ~np.broadcast_to(np.asarray(editable_mask, dtype=bool), before.shape)
    if not retained.any():
        raise ValueError("Preservation score requires an unedited reference region")
    return float(np.abs(before[retained].astype(np.float32)
                        - edited[retained].astype(np.float32)).mean())


def clock_sync_ms(audio_onsets_ms: Sequence[int], video_onsets_ms: Sequence[int],
                  *, offset_ms: int = 0) -> dict[str, float]:
    """Absolute A/V event-clock error; no utterance text masquerades as timestamps."""
    if not audio_onsets_ms or len(audio_onsets_ms) != len(video_onsets_ms):
        raise ValueError("A/V sync needs paired verified audible/visible events")
    if any(type(x) is not int or x < 0 for x in
           (*audio_onsets_ms, *video_onsets_ms)):
        raise ValueError("A/V event timestamps must be nonnegative integer milliseconds")
    differences = [
        abs((a + offset_ms) - v)
        for a, v in zip(audio_onsets_ms, video_onsets_ms, strict=True)
    ]
    return {
        "mean_sync_error_ms": float(np.mean(differences)),
        "worst_sync_error_ms": float(max(differences)),
        "aligned_events": len(differences),
    }


@dataclass(frozen=True, slots=True)
class VerifierIdentity:
    verifier_id: str
    weights_sha256: str
    policy: str
    independence_note: str

    def __post_init__(self) -> None:
        if (not self.verifier_id or not self.policy or
                not self.independence_note or len(self.weights_sha256) != 64):
            raise ValueError("Every external verifier needs pinned independent provenance")


@dataclass
class SampledEvalLedger:
    """Aggregate observed metrics by task and by source-family held-out split."""
    _records: list[dict[str, object]] = field(default_factory=list)
    source_families: set[str] = field(default_factory=set)
    exemplar_limit: int = 1000

    def add(self, *, task_id: str, source_family: str, split: str,
            metrics: Mapping[str, float | int], evidence: Mapping[str, object]) -> None:
        if split not in {"validation", "test"}:
            raise ValueError("Held-out sampled quality evaluation cannot use training sources")
        if not source_family or not task_id:
            raise ValueError("Evaluation must keep a source family and task identity")
        if len(set(metrics)) != len(metrics) or not metrics:
            raise ValueError("Evaluation record needs measured metric names")
        for name, value in metrics.items():
            if (not isinstance(value, (int, float)) or isinstance(value, bool)
                    or not math.isfinite(float(value)) or not name):
                raise ValueError("Evaluation metric must be finite and numeric")
        if "renderer_seed" not in evidence or "sampler_steps" not in evidence:
            raise ValueError("Sampled evaluation needs exact free sampler provenance")
        self.source_families.add(source_family)
        self._records.append({
            "task_id": task_id, "source_family": source_family,
            "split": split, "metrics": dict(metrics), "evidence": dict(evidence),
        })

    def report(self) -> dict[str, object]:
        grouped: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for record in self._records:
            for name, value in record["metrics"].items():
                grouped[record["task_id"]][name].append(float(value))
        return {
            "type": "measured-free-sampled-output",
            "n_examples": len(self._records),
            "source_families": len(self.source_families),
            "tasks": {
                task: {
                    name: {"mean": float(np.mean(values)), "n": len(values),
                           "p90": float(np.percentile(values, 90))}
                    for name, values in families.items()
                }
                for task, families in grouped.items()
            },
            "examples": self._records[:self.exemplar_limit],
        }


def audit_source_prefix(available_until_ms: int, evidence_end_ms: int,
                        *, verifier_observed_ms: int | None = None) -> None:
    if (available_until_ms < 0 or evidence_end_ms < 0
            or evidence_end_ms > available_until_ms):
        raise ValueError("Model observed audio/video after the declared available prefix")
    if verifier_observed_ms is not None and verifier_observed_ms > available_until_ms:
        raise ValueError("Verifier saw future content while defining a causal control")


def supervised_speech_metrics(
    generated_waveform: np.ndarray, verified_text: str,
    independent_asr: Callable[[np.ndarray], str], *,
    voice_target_embedding: np.ndarray | None = None,
    independent_voice_embedder: Callable[[np.ndarray], np.ndarray] | None = None,
) -> dict[str, float]:
    """Transcript agreement from an independent ASR, optional speaker similarity."""
    if generated_waveform.ndim != 1 or not np.isfinite(generated_waveform).all():
        raise ValueError("Generated speech requires finite mono waveform")
    result = transcript_metrics(verified_text, independent_asr(generated_waveform))
    if voice_target_embedding is not None:
        if independent_voice_embedder is None:
            raise ValueError("Voice similarity requires a separate pinned verifier")
        predicted = independent_voice_embedder(generated_waveform)
        if predicted.shape != voice_target_embedding.shape:
            raise ValueError("Voice verifier embedding dimension changed")
        denom = np.linalg.norm(predicted) * np.linalg.norm(voice_target_embedding)
        if denom == 0:
            raise ValueError("Empty voice verification embedding")
        result["voice_similarity"] = float(np.vdot(predicted, voice_target_embedding) / denom)
    return result
