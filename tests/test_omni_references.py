"""Voice/reference descriptors are not content until E observes real verified bytes."""
import numpy as np
import pytest
import torch

from megatoken.controls import ControlReference
from megatoken.data.media_assets import MediaRecord, TensorAssetStore
from megatoken.omni.bundles import TaskView
from megatoken.omni.references import resolve_reference


def test_frozen_speaker_reference_resolves_causally_and_masks_padding(tmp_path):
    store = TensorAssetStore(tmp_path)
    source = store.put(
        np.ones((6, 8), dtype=np.float32),
        modality="audio", kind="features",
        extractor="pocket-mimi-reference-embedding-v1",
        sample_rate=24000, valid_count=4,
    )
    reference = ControlReference(
        "voice-approval-1", source.sha256, "audio/wav",
        "speaker_timbre", "external_evidence",
    )
    resolved = resolve_reference(
        store, reference, MediaRecord("audio", source),
        track="approved_narrator", device=torch.device("cpu"),
    )
    event = resolved.event
    assert event.modality == "audio"
    assert event.extractor == "pocket-mimi-reference-embedding-v1"
    assert event.payload.shape == (1, 6, 8)
    assert event.valid_mask.tolist() == [[True, True, True, True, False, False]]
    assert event.source_views == ()


def test_reference_rejects_mismatched_asset_or_future_target(tmp_path):
    store = TensorAssetStore(tmp_path)
    source = store.put(
        np.ones((6, 8), dtype=np.float32), modality="audio",
        kind="features", extractor="pocket-mimi-reference-embedding-v1",
        sample_rate=24000,
    )
    with pytest.raises(ValueError, match="digest"):
        resolve_reference(
            store,
            ControlReference("ref", "0" * 64, "audio/wav", "style", "external_evidence"),
            MediaRecord("audio", source), track="speaker",
            device=torch.device("cpu"),
        )
    with pytest.raises(ValueError, match="available observed inputs"):
        TaskView(
            "verbalization",
            inputs=("words",), outputs=("speech",),
            requests={"words": "Read text", "speech": "Say text"},
            reference_event_ids=("speech",),
        )
