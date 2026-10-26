from types import SimpleNamespace

import torch
from torch import nn

from megatoken.omni.accounting import resident_footprint
from megatoken.omni.realizers import Realizer


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.core = nn.Linear(3, 2, bias=False)
        self.alias = self.core
        self.media_realizers = nn.Module()
        self.media_realizers.packs = nn.ModuleDict()
        pack = nn.Module()
        pack.condition = nn.Linear(2, 2, bias=False)
        pack.donor = nn.Parameter(torch.ones(4), requires_grad=False)
        pack.register_buffer("lookup", torch.ones(2, dtype=torch.float16))
        self.media_realizers.packs["sana"] = pack


def test_accounting_deduplicates_aliases_and_classifies_frozen_donor_and_adapter():
    report = resident_footprint(Model())
    groups = {(g["owner"], g["dtype"]): g for g in report["logical_tensors"]}
    assert groups[("shared_core_trainable", "torch.float32")]["elements"] == 6
    assert groups[("trainable_adapter:sana", "torch.float32")]["elements"] == 4
    assert groups[("frozen_pretrained:sana", "torch.float32")]["elements"] == 4
    assert groups[("frozen_pretrained:sana", "torch.float16")]["elements"] == 2
    assert report["fp32_optimizer_master_bytes_upper_bound"] == 40
    assert report["adam_optimizer_total_bytes_upper_bound"] == 120
    assert report["runtime_workspace_bytes"] == "unmeasured"
    assert report["gpu_fit"] == "not_claimed"


class PocketHolder(Realizer):
    _donor = "pocket"

    def __init__(self, pocket):
        super().__init__()
        self._hold(pocket=pocket, flow_lm=pocket.flow_lm, mimi=pocket.mimi)
        self.condition = nn.Linear(2, 2, bias=False)


class PipelineHolder(Realizer):
    _donor = "pipe"

    def __init__(self, pipe):
        super().__init__()
        self._hold(pipe=pipe, transformer=pipe.transformer, vae=pipe.vae)
        self.condition = nn.Linear(2, 2, bias=False)


def _physical_owner_bytes(report, owner):
    return sum(row["storage_bytes"] for row in report["physical_storage"] if row["owner"] == owner)


def test_realizer_hold_counts_pocket_donor_outside_tree_and_deduplicates_aliases():
    donor = nn.Module()
    donor.flow_lm = nn.Linear(3, 2, bias=False)
    donor.mimi = donor.flow_lm
    holder = PocketHolder(donor)
    holder.condition.weight.data.fill_(1)
    report = resident_footprint(holder)
    donor_logical = [
        row for row in report["logical_tensors"] if row["owner"] == "frozen_pretrained"
    ]
    assert sum(row["elements"] for row in donor_logical) == 6
    assert _physical_owner_bytes(report, "frozen_pretrained") == 24
    assert report["trainable_parameter_tensors"] == 1


def test_pipeline_components_are_counted_without_arbitrary_pipeline_walk():
    transformer, vae = nn.Linear(3, 2, bias=False), nn.Linear(2, 1, bias=False)
    pipe = SimpleNamespace(
        components={"transformer": transformer, "vae": vae},
        transformer=transformer,
        vae=vae,
        unrelated=nn.Linear(99, 99, bias=False),
    )
    report = resident_footprint(PipelineHolder(pipe))
    donor_logical = [
        row for row in report["logical_tensors"] if row["owner"] == "frozen_pretrained"
    ]
    assert sum(row["elements"] for row in donor_logical) == 8
    assert _physical_owner_bytes(report, "frozen_pretrained") == (6 + 2) * 4


def test_pipeline_donor_trainability_is_reported_and_cross_dtype_storage_is_mixed():
    weights = nn.Linear(3, 2, bias=False)
    vae = nn.Linear(2, 1, bias=False)
    pipeline_only = nn.Linear(2, 1, bias=False)
    pipeline_only.weight.requires_grad_(True)
    pipe = SimpleNamespace(
        components={"transformer": weights, "vae": vae, "extra": pipeline_only},
        transformer=weights,
        vae=vae,
    )
    report = resident_footprint(PipelineHolder(pipe))
    assert any(
        row["owner"] == "trainable_pretrained" and row["parameter_tensors"] == 1
        for row in report["logical_tensors"]
    )

    class ReinterpretedStorage(nn.Module):
        def __init__(self):
            super().__init__()
            value = torch.arange(4, dtype=torch.float32)
            self.parameter = nn.Parameter(value)
            self.register_buffer("bytes", value.view(torch.uint8))

    mixed = resident_footprint(ReinterpretedStorage())
    assert any(
        row["dtype"] == "mixed" and row["dtypes"] == ["torch.float32", "torch.uint8"]
        for row in mixed["physical_storage"]
    )


def test_nested_realizer_donors_keep_pack_buckets_and_shared_alias_is_counted_once():
    common = nn.Parameter(torch.ones(2), requires_grad=False)
    pocket_root = nn.Module()
    pocket_root.flow_lm = nn.Linear(3, 2, bias=False)
    pocket_root.flow_lm.weight = common
    pocket_root.mimi = pocket_root.flow_lm
    pocket = PocketHolder(pocket_root)
    sana_transformer = nn.Linear(2, 2, bias=False)
    sana_transformer.weight = common
    sana_vae = nn.Linear(2, 1, bias=False)
    sana_pipe = SimpleNamespace(
        components={"transformer": sana_transformer, "vae": sana_vae},
        transformer=sana_transformer,
        vae=sana_vae,
    )
    sana = PipelineHolder(sana_pipe)

    class Wrapped(nn.Module):
        def __init__(self):
            super().__init__()
            self.media_realizers = nn.Module()
            self.media_realizers.packs = nn.ModuleDict({"pocket": pocket, "sana": sana})

    report = resident_footprint(Wrapped())
    donor_groups = {
        row["owner"]: row
        for row in report["logical_tensors"]
        if row["owner"].startswith("frozen_pretrained:")
    }
    assert donor_groups["frozen_pretrained:pocket"]["parameter_tensors"] == 1
    assert donor_groups["frozen_pretrained:sana"]["parameter_tensors"] == 1
    assert report["unique_storage_count"] == 4
    assert (
        sum(
            row["storage_bytes"]
            for row in report["physical_storage"]
            if row["owner"] == "shared_storage_alias"
        )
        == 8
    )


def test_shared_storage_preserves_logical_parameter_counts_and_meta_tensors():
    class Views(nn.Module):
        def __init__(self):
            super().__init__()
            backing = torch.arange(8, dtype=torch.float32)
            self.frozen_view = nn.Parameter(backing[:4], requires_grad=False)
            self.trainable_view = nn.Parameter(backing[2:6], requires_grad=True)
            self.unbuilt = nn.Parameter(torch.empty(5, device="meta"))

    report = resident_footprint(Views())
    assert report["trainable_parameter_tensors"] == 2
    assert report["fp32_optimizer_master_bytes_upper_bound"] == 36
    assert report["unique_storage_count"] == 1
    assert any(
        row["owner"] == "shared_core_trainable" and row["elements"] == 4
        for row in report["logical_tensors"]
    )
    assert any(
        row["owner"] == "shared_core_trainable" and row["unmaterialized_tensors"] == 1
        for row in report["logical_tensors"]
    )
    assert any(row["owner"] == "shared_storage_alias" for row in report["physical_storage"])
