"""Auditable core/adapters/frozen donor residency and optimizer footprint.

Actual bytes are reported by dtype and owner, not inferred from donor model
names. Activation/KV, resident CPU, checkpoint, and optimizer workspaces remain
separate from on-device parameter bytes.
"""
from __future__ import annotations

from collections import defaultdict

import torch
from torch import nn


def resident_footprint(model: nn.Module) -> dict:
    categories = defaultdict(lambda: {"parameters": 0, "parameter_bytes": 0, "tensors": 0})
    seen = set()
    for name, param in model.named_parameters(remove_duplicate=False):
        if id(param) in seen:
            continue
        seen.add(id(param))
        if name.startswith("media_realizers.pretrained."):
            parts = name.split(".")
            pack = parts[2] if len(parts) > 2 else "unknown"
            group = f"{'adapter' if param.requires_grad else 'frozen'}:{pack}"
        elif "media_realizers" in name:
            group = "native_debug_realizers"
        else:
            group = "shared_core_trainable" if param.requires_grad else "shared_core_frozen"
        categories[group]["parameters"] += param.numel()
        categories[group]["parameter_bytes"] += param.numel() * param.element_size()
        categories[group]["tensors"] += 1
    master_bytes = sum(
        p.numel() * 4 for p in model.parameters() if p.requires_grad
    )
    total_bytes = sum(v["parameter_bytes"] for v in categories.values())
    return {
        "categories": dict(sorted(categories.items())),
        "unique_parameter_tensors": len(seen),
        "all_parameter_bytes": total_bytes,
        "fp32_optimizer_master_bytes_upper_bound": master_bytes,
        "runtime_workspace_bytes": "unmeasured",
        "kv_history_bytes": "separately metered by live event geometry",
    }
