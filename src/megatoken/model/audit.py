"""Audit actual parameter ownership; declared budgets never substitute for this."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any

from torch import nn


@dataclass(frozen=True)
class ParameterOwner:
    name: str
    shape: tuple[int, ...]
    numel: int
    dtype: str
    trainable: bool
    aliases: tuple[str, ...]


def parameter_owners(module: nn.Module) -> list[ParameterOwner]:
    """Count real tensors once, report aliases, and reject overlapping owners.

    A meta-device construction is sufficient for a shape audit. On instantiated
    models, different Parameter objects sharing a storage are checked as well.
    """
    entries: dict[int, tuple[str, nn.Parameter]] = {}
    aliases: dict[int, list[str]] = defaultdict(list)
    storages: dict[tuple[str, int], tuple[int, str]] = {}
    for name, parameter in module.named_parameters(remove_duplicate=False):
        identity = id(parameter)
        if identity in entries:
            aliases[identity].append(name)
            continue
        if parameter.device.type != "meta":
            key = (str(parameter.device), parameter.untyped_storage().data_ptr())
            if key in storages and storages[key][0] != identity:
                raise ValueError(
                    f"Separate parameter owners overlap storage: {storages[key][1]} and {name}"
                )
            storages[key] = (identity, name)
        entries[identity] = (name, parameter)
    return [
        ParameterOwner(
            name, tuple(p.shape), p.numel(), str(p.dtype), p.requires_grad, tuple(aliases[identity])
        )
        for identity, (name, p) in entries.items()
    ]


def audit_parameters(module: nn.Module) -> dict[str, Any]:
    owners = parameter_owners(module)
    return {
        "total_parameters": sum(owner.numel for owner in owners),
        "trainable_parameters": sum(owner.numel for owner in owners if owner.trainable),
        "unique_parameter_owners": len(owners),
        "aliases": sum(len(owner.aliases) for owner in owners),
        "owners": [asdict(owner) for owner in owners],
        "claim": "Actual instantiated shape/ownership audit; not a throughput or quality measurement",
    }
