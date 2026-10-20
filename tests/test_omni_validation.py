import pytest
import torch
from torch import nn
from megatoken.omni.validation import check_pretrained_student, assert_content_only_invariance

class MockDonor(nn.Module):
    def __init__(self):
        super().__init__()
        self.mimi = nn.Linear(4,4)
        self.adapter = nn.Linear(8,4)
        for p in self.mimi.parameters():
            p.requires_grad_(False)
    def native_loss(self, raw, target, *, valid_mask=None):
        predicted = self.adapter(raw.mean(1))[:,None,:].expand_as(target)
        squared = (predicted-target).square()
        if valid_mask is not None:
            squared = squared * valid_mask
            return squared.sum(),valid_mask.expand_as(squared).sum()
        return squared.sum(),torch.tensor(squared.numel(),device=squared.device)

def test_donor_preflight_requires_frozen_codec_and_adapter_gradient():
    model=MockDonor()
    result=check_pretrained_student("pocket",model,torch.randn(1,32,8),torch.randn(1,4,4))
    assert result.pack=="pocket"
    assert result.trainable_parameters==36
    next(model.mimi.parameters()).requires_grad_(True)
    with pytest.raises(ValueError,match="frozen"):
        check_pretrained_student("pocket",model,torch.randn(1,32,8),torch.randn(1,4,4))

def test_preflight_detects_missing_facets_and_bad_mask():
    model=MockDonor()
    with pytest.raises(ValueError,match="32"):
        check_pretrained_student("pocket",model,torch.randn(1,16,8),torch.randn(1,4,4))

class PureRenderer:
    def realize(self,raw,spec,noise):
        return raw.mean(1)*spec

def test_renderer_content_only_is_replayable():
    result=assert_content_only_invariance(PureRenderer(),torch.randn(1,32,8),2.)
    assert result.shape==(1,8)
