"""Scoring concordance on *sampled, decoded* peer artifacts.

No labels, transcripts, target videos or teacher prefixes enter P through this
module. This is a post-realization evaluator, not an implied differentiable loss.
Unavailable independent verifiers are explicitly reported as unmeasured.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np

from megatoken.omni.bundles import Relation
from megatoken.omni.evaluation import (
    MetricAccumulator,
    character_error_rate,
    edit_preservation_rmse,
    endpoint_rmse,
    timed_event_matches,
    word_error_rate,
)


@dataclass(frozen=True)
class SampledEvent:
    event_id: str
    modality: str
    decoded: str | np.ndarray
    # Realized sound/video events must come from an output-side detector or
    # independent human annotations, not target annotations/prompt text.
    detected_events: tuple[tuple[str, float], ...] = ()
    # For edits, only the actual requested changed region is exempted.
    changed_region: np.ndarray | None = None
    provenance: str = "sampled"

    def __post_init__(self):
        if self.provenance != "sampled":
            raise ValueError("Concordance evaluation requires sampled outputs")
        if self.modality not in {"text", "audio", "image", "video"}:
            raise ValueError("Unknown decoded event modality")
        if self.modality == "text":
            if not isinstance(self.decoded, str):
                raise ValueError("Text event must contain realized words")
        elif not isinstance(self.decoded, np.ndarray) or not np.isfinite(self.decoded).all():
            raise ValueError("Media event must contain finite decoded array")


@dataclass(frozen=True)
class ConcordanceReport:
    measured: Mapping[str, dict[str, float | int]]
    unmeasured: tuple[str, ...]
    evaluated_pairs: int


def score_sampled_relations(
    relations: tuple[Relation, ...],
    outputs: Mapping[str, SampledEvent],
    *,
    speech_recognizer: Callable[[np.ndarray], str] | None = None,
    pair_verifiers: Mapping[str, Callable[[SampledEvent, SampledEvent], float]] | None = None,
    timing_tolerance_seconds: float = 0.12,
) -> ConcordanceReport:
    """Score true sampled peers, never their withheld ground-truth counterparts.

    Optional third-party verifiers must return finite scores in [0,1]. Their
    provenance, language/normalization policy and version belong in the
    evaluation manifest. Missing verifiers do NOT grant a perfect score.
    """
    metrics = MetricAccumulator()
    missing = []
    verifiers = {} if pair_verifiers is None else dict(pair_verifiers)
    for relation in relations:
        label = f"{relation.kind}:{relation.left}:{relation.right}"
        if relation.left not in outputs or relation.right not in outputs:
            missing.append(label + ":missing_output")
            continue
        left, right = outputs[relation.left], outputs[relation.right]
        kind = relation.kind
        if kind in {"first_frame_of", "last_frame_of"}:
            if left.modality != "image" or right.modality != "video":
                raise ValueError("Endpoint relation needs image -> video direction")
            first, last = endpoint_rmse(
                np.asarray(right.decoded),
                np.asarray(left.decoded),
                np.asarray(left.decoded),
            )
            metrics.record(kind + "_rmse", first if kind == "first_frame_of" else last)
        elif kind == "synchronized_with":
            if not left.detected_events or not right.detected_events:
                missing.append(label + ":no_output_event_detection")
                continue
            matched, count = timed_event_matches(
                left.detected_events,
                right.detected_events,
                tolerance_seconds=timing_tolerance_seconds,
            )
            if count:
                metrics.record("av_event_timing_recall", matched, count)
        elif kind == "same_spoken_content":
            if speech_recognizer is None:
                missing.append(label + ":speech_recognizer_required")
                continue
            if left.modality == "text" and right.modality == "audio":
                transcript, audio = left.decoded, right.decoded
            elif right.modality == "text" and left.modality == "audio":
                transcript, audio = right.decoded, left.decoded
            else:
                raise ValueError("Spoken-content relation needs text and speech")
            recognized = speech_recognizer(np.asarray(audio))
            if not isinstance(recognized, str):
                raise ValueError("Speech recognizer must return a string")
            metrics.record("sampled_speech_text_wer", word_error_rate(transcript, recognized))
            metrics.record("sampled_speech_text_cer", character_error_rate(transcript, recognized))
        elif kind in {"edit_of", "preserves"}:
            if (
                left.modality != right.modality
                or not isinstance(left.decoded, np.ndarray)
                or not isinstance(right.decoded, np.ndarray)
            ):
                missing.append(label + ":decoded_edit_geometry_required")
                continue
            if left.changed_region is None:
                missing.append(label + ":requested_change_mask_required")
                continue
            preservation = edit_preservation_rmse(
                np.asarray(right.decoded),
                np.asarray(left.decoded),
                left.changed_region,
            )
            metrics.record("edit_unchanged_rmse", preservation)
        elif kind in {"translation_of", "describes"}:
            if kind not in verifiers:
                missing.append(label + ":independent_verifier_required")
                continue
            value = verifiers[kind](left, right)
            if not isinstance(value, (int, float)) or not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Verifier score must be finite and within [0,1]")
            metrics.record("sampled_" + kind + "_score", float(value))
        else:
            missing.append(label + ":unsupported_relation")
    measured = metrics.report()
    return ConcordanceReport(measured, tuple(missing), len(relations))
