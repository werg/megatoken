"""Optimizer-owner accounting distinguishes frozen packs from trainable adapters."""
from types import SimpleNamespace

import torch
from torch import nn

from megatoken.omni.accounting import resident_footprint


class ToyOmni(nn.Module):
    def __init__(self):
        super().__init__()
        self.core = nn.Linear(5, 5)
        self.media_realizers = nn.Module()
        self.media_realizers.pretrained = nn.ModuleDict()
        student = nn.Module()
        student.condition = nn.Linear(5, 5)
        student.vae = nn.Linear(8, 8)
        for p in student.vae.parameters():
            p.requires_grad_(False)
        self.media_realizers.pretrained["sana"] = student


def test_resident_count_is_actual_owner_not_donor_marketing_label():
    model = ToyOmni()
    footprint = resident_footprint(model)
    counts = footprint["categories"]
    assert counts["adapter:sana"]["parameters"] == 30
    assert counts["frozen:sana"]["parameters"] == 72
    assert counts["shared_core_trainable"]["parameters"] == 30
    assert footprint["fp32_optimizer_master_bytes_upper_bound"] == 240
    assert footprint["runtime_workspace_bytes"] == "unmeasured"
