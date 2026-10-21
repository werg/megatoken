"""Exact independently sampled KV/LoRA no-evidence interventions."""

import torch

from megatoken.memory.context import admit_kb
from megatoken.memory.semantic import (
    AdmittedGroup,
    Constituent,
    independently_drop_memory,
)
from megatoken.model.config import ModelConfig
from megatoken.model.core import FacetedCore
from megatoken.model.multitrack import MultiTrackCore


def _snapshot():
    core = FacetedCore(
        ModelConfig.tiny(
            semantic_dim=8,
            semantic_value_dim=16,
            max_kb_positions=2,
        )
    )
    tracks = MultiTrackCore(core)
    initial = tracks.initial_state(1)
    semantic = torch.randn(1, 16, requires_grad=True)
    group = AdmittedGroup(
        "point-a",
        (
            Constituent(
                "a",
                semantic,
                torch.ones(1),
                torch.ones(1) * -10,
                admitted_at=0,
                version="r1",
            ),
        ),
        anchor=0,
        revision="r1",
        compatible_group="team/a",
    )
    return core, initial, tracks.materialize_memory(admit_kb(initial, group))


def test_memory_interventions_use_exact_kv_mask_and_zero_lora_gate():
    core, initial, snapshot = _snapshot()
    assert snapshot.features.mask.any()
    assert snapshot.strengths.abs().sum() > 0
    dropped = independently_drop_memory(snapshot, kv_rate=1, lora_rate=1)
    assert not dropped.features.mask.any()
    assert torch.count_nonzero(dropped.strengths) == 0
    x = torch.randn(1, 4, core.config.facet_dim)
    torch.testing.assert_close(
        core.memory_adapters.apply(
            x,
            module="p",
            stage=0,
            track_ids=None,
            snapshot=dropped,
        ),
        x,
        rtol=0,
        atol=0,
    )
    assert initial.history.slots == 0


def test_kv_and_lora_dropout_are_independent_and_disabled_rate_is_identity():
    _, _, snapshot = _snapshot()
    assert independently_drop_memory(snapshot, kv_rate=0, lora_rate=0) is snapshot
    kv_only = independently_drop_memory(snapshot, kv_rate=1, lora_rate=0)
    assert not kv_only.features.mask.any()
    torch.testing.assert_close(kv_only.strengths, snapshot.strengths)
    lora_only = independently_drop_memory(snapshot, kv_rate=0, lora_rate=1)
    torch.testing.assert_close(lora_only.features.mask, snapshot.features.mask)
    assert not lora_only.strengths.any()
