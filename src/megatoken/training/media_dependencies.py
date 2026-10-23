"""Startup checks for pretrained media decoders used by training objectives."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from megatoken.data.media_assets import MEDIA


def required_media_modalities(
    dataset: Any,
    scored_output_modalities: Iterable[str],
    *,
    autoencode_items: int,
) -> tuple[str, ...]:
    """Find realizer modalities used by scored media outputs or reconstruction.

    Media features used as observations do not need a decoder. The stage's
    encode-and-reconstruct objective is different: it samples each available
    media feature modality, so those modalities need their decoder too.
    """
    required = set(scored_output_modalities)
    if not required <= set(MEDIA):
        raise ValueError("Prepared scored media summary contains an unknown modality")
    if autoencode_items:
        required.update(
            modality for modality in MEDIA if dataset.feature_count("train", modality) > 0
        )
    return tuple(sorted(required))


def validate_media_realizers(model: Any, modalities: Iterable[str]) -> None:
    """Fail before optimizer creation if an active objective lacks its pack."""
    missing = []
    for modality in modalities:
        try:
            model.media_realizers.resident(modality)
        except (ValueError, KeyError):
            missing.append(modality)
    if missing:
        joined = ", ".join(missing)
        raise ValueError(
            f"Active training objectives need installed media packs for: {joined}. "
            "Set packs to the lock file that declares those modalities."
        )
