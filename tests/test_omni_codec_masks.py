"""Padded codec frames never supervise a realizer: mask semantics end to end."""

from __future__ import annotations

import pytest
import torch

from megatoken.data.media_assets import TensorAssetStore
from megatoken.model.media import MediaOutputSpec
from megatoken.omni.realizers import masked_square_error
from megatoken.training.meridians import MediaPlan
from tests.media_support import FakeRealizer, prepared_record


def resolve(tmp_path, modality, valid):
    store = TensorAssetStore(tmp_path)
    record = prepared_record(store, modality, valid_frames=valid)
    spec = MediaOutputSpec.from_geometry(record.output_geometry)
    plan = MediaPlan(record, spec, "bootstrap", 32, 8)
    return plan.resolve(store, torch.device("cpu"))


def test_audio_mask_marks_only_the_leading_valid_frames(tmp_path):
    resolved = resolve(tmp_path, "audio", valid=3)
    assert resolved.target.shape == (1, 6, 4)
    assert resolved.target_mask.tolist() == [[True, True, True, False, False, False]]
    zeros = torch.zeros_like(resolved.target)
    loss = masked_square_error(zeros, resolved.target, resolved.target_mask)
    torch.testing.assert_close(loss, resolved.target[:, :3].square().mean())
    # The mean is over real coordinates, not diluted by padding.
    assert loss != masked_square_error(zeros, resolved.target, None)


def test_video_mask_is_over_time_major_frames(tmp_path):
    resolved = resolve(tmp_path, "video", valid=2)
    assert resolved.target.shape == (1, 3, 4, 2, 2)  # [B, F, C, h, w]
    assert resolved.target_mask.tolist() == [[True, True, False]]
    loss = masked_square_error(
        torch.zeros_like(resolved.target), resolved.target, resolved.target_mask
    )
    torch.testing.assert_close(loss, resolved.target[:, :2].square().mean())


def test_a_complete_target_has_no_mask(tmp_path):
    assert resolve(tmp_path, "image", valid=0).target_mask is None
    assert resolve(tmp_path, "audio", valid=0).target_mask is None


def test_masked_square_error_ignores_padding_values():
    target = torch.zeros(2, 4, 3)
    mask = torch.tensor([[1, 1, 0, 0], [1, 0, 0, 0]], dtype=torch.bool)
    prediction = torch.ones(2, 4, 3)
    prediction[:, 2:] = 1e6
    prediction[1, 1] = 1e6
    assert masked_square_error(prediction, target, mask).item() == 1.0


def test_masked_square_error_broadcasts_a_time_mask_over_latent_dims():
    prediction, target = torch.ones(1, 3, 4, 2, 2), torch.zeros(1, 3, 4, 2, 2)
    mask = torch.tensor([[1, 0, 1]])  # [B, F], any integer mask; trailing dims are filled in
    assert masked_square_error(prediction, target, mask).item() == 1.0


def test_masked_square_error_reduces_in_fp32_and_guards_empty_masks():
    half = torch.full((1, 3, 2), 3.0, dtype=torch.bfloat16)
    assert masked_square_error(half, torch.zeros_like(half), None).dtype == torch.float32
    empty = torch.zeros(1, 3, dtype=torch.bool)
    value = masked_square_error(torch.ones(1, 3, 2), torch.zeros(1, 3, 2), empty)
    assert value.item() == 0.0 and torch.isfinite(value)


def test_masked_square_error_rejects_geometry_mismatch():
    with pytest.raises(ValueError, match="geometry"):
        masked_square_error(torch.zeros(1, 3, 4), torch.zeros(1, 4, 4), None)


def test_padded_frames_do_not_move_a_realizer_objective():
    realizer = FakeRealizer("audio", 32, 8)
    raw = torch.randn(1, 32, 8)
    target = torch.randn(1, 6, 4)
    mask = torch.tensor([[1, 1, 0, 0, 0, 0]], dtype=torch.bool)
    padded = target.clone()
    padded[:, 2:] = 1e3
    a = realizer.loss(raw, target, mask)
    b = realizer.loss(raw, padded, mask)
    torch.testing.assert_close(a, b)
