"""Resolve available media/voice/style references into causal E observations.

A SHA256 reference in a control payload is only provenance. Real timbre,
speaker identity, spatial detail and edit-base semantics require an actual
observed event selected into the shared history BEFORE future P queries.
No source feature/codec latent may be read directly by the current R.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from megatoken.controls import ControlReference
from megatoken.data.media_assets import MediaRecord, TensorAsset, TensorAssetStore
from megatoken.model.multitrack import ObservedEvent


def _ordered_view(store: TensorAssetStore, asset: TensorAsset, device):
    if asset.kind not in ("features", "codec_latents") or not asset.extractor:
        raise ValueError("Reference feature view must be pinned and ordered")
    value = torch.from_numpy(store.load(asset)).to(device=device, dtype=torch.float32)[None]
    if asset.kind == "codec_latents":
        if asset.modality == "image":
            value = value.flatten(2).transpose(1, 2)
            sites_per_time = value.shape[1]
        elif asset.modality == "video":
            value = value.permute(0, 1, 3, 4, 2).reshape(
                value.shape[0], -1, value.shape[2],
            )
            sites_per_time = asset.shape[-2] * asset.shape[-1]
        else:
            sites_per_time = 1
    else:
        sites_per_time = 1
    valid = (asset.valid_count or asset.shape[0]) * sites_per_time
    mask = (
        torch.arange(value.shape[1], device=device)[None] < valid
    )
    return (asset.extractor, value, mask)


@dataclass(frozen=True)
class ResolvedReference:
    reference: ControlReference
    event: ObservedEvent


def resolve_reference(
    store: TensorAssetStore, ref: ControlReference, record: MediaRecord, *,
    track: str, device: torch.device,
) -> ResolvedReference:
    """Return one available causal media event with fully verified feature bytes."""
    if ref.sha256 != record.source.sha256:
        raise ValueError("Reference digest is not the pinned observed media source")
    if not ref.asset_id or not track or not ref.purpose:
        raise ValueError("Reference event needs stable identity and purpose")
    expected = record.modality
    if not (
        ref.media_type.startswith(expected + "/")
        or ref.media_type == "application/vnd.megatoken.media-reference+json"
    ):
        raise ValueError("Reference declared media MIME disagrees with observed artifact")
    source = record.source
    extras = [_ordered_view(store, asset, device) for asset in record.source_views]
    if source.kind in ("features", "codec_latents"):
        primary = _ordered_view(store, source, device)
        identity, payload, mask = primary
        event = ObservedEvent(
            track=track, modality=expected,
            payload=payload, extractor=identity, valid_mask=mask,
            source_views=tuple(extras),
        )
    elif extras:
        # For raw 24 kHz / dense visual references with declared frozen
        # views, do not silently feed raw samples to the 16 kHz debug stem.
        identity, payload, mask = extras[0]
        event = ObservedEvent(
            track=track, modality=expected, payload=payload,
            extractor=identity, valid_mask=mask,
            source_views=tuple(extras[1:]),
        )
    else:
        waveform = torch.from_numpy(store.load(source)).to(
            device=device, dtype=torch.float32,
        )[None]
        event = ObservedEvent(
            track=track, modality=expected, payload=waveform,
            sample_rate=source.sample_rate or 16000,
        )
    return ResolvedReference(ref, event)
