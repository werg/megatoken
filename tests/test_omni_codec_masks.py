"""Prepared raw/media/codec interface masks preserve valid coordinates."""
from __future__ import annotations

import numpy as np
import torch

from megatoken.data.media_assets import MediaRecord, TensorAssetStore
from megatoken.model.media import MediaOutputSpec
from megatoken.training.meridians import MediaPlan
from megatoken.omni.flow import flow_matching_loss


def test_pocket_mask_broadcasts_only_valid_acoustic_frames(tmp_path):
    store = TensorAssetStore(tmp_path)
    source = store.put(np.zeros(320, dtype=np.float32), modality="audio", sample_rate=16000)
    latent = store.put(
        np.ones((5, 4), dtype=np.float32), modality="audio", kind="codec_latents",
        extractor="pinned-pocket-mimi", sample_rate=24000, valid_count=3,
    )
    spec = MediaOutputSpec(
        "audio", renderer="pocket", sample_rate=24000,
        sample_count=3 * 1920, latent_shape=(5, 4),
        shape_provenance="teacher_only",
    )
    plan = MediaPlan(MediaRecord("audio", source, latent), spec, "bootstrap", 32, 8)
    resolved = plan.resolve(store, torch.device("cpu"))
    assert resolved.target.shape == (1, 5, 4)
    assert resolved.target_valid_mask.shape == (1, 5, 1)
    numerator, count = flow_matching_loss(
        torch.zeros_like(resolved.target), resolved.target,
        resolved.target_valid_mask,
    )
    assert count.item() == 3 * 4
    assert numerator.item() == 12


def test_video_latents_are_channel_first_after_storage_conversion(tmp_path):
    store = TensorAssetStore(tmp_path)
    source = store.put(
        np.zeros((2, 3, 8, 8), dtype=np.float32),
        modality="video", fps_num=8,
    )
    latent = store.put(
        np.ones((3, 4, 2, 2), dtype=np.float32),
        modality="video", kind="codec_latents",
        extractor="pinned-ltx2-vae", fps_num=8, valid_count=2,
    )
    spec = MediaOutputSpec(
        "video", renderer="sana_video", frames=17, height=64, width=64,
        latent_shape=(4, 3, 2, 2), fps_num=8,
        shape_provenance="teacher_only",
    )
    resolved = MediaPlan(
        MediaRecord("video", source, latent), spec, "recurrent", 32, 8
    ).resolve(store, torch.device("cpu"))
    assert resolved.target.shape == (1, 4, 3, 2, 2)
    assert resolved.target_valid_mask.shape == (1, 1, 3, 1, 1)
    numerator, count = flow_matching_loss(
        torch.zeros_like(resolved.target), resolved.target, resolved.target_valid_mask,
    )
    assert count.item() == 4 * 2 * 2 * 2
    assert numerator.item() == count.item()
