"""Versioned inventory of instantiated training tensors and optimizer bounds."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

import torch
from torch import nn


def _owner(name: str, trainable: bool, donor: bool = False) -> str:
    if name.startswith("media_realizers.packs."):
        pack = name.split(".")[2]
        if donor:
            return f"trainable_pretrained:{pack}" if trainable else f"frozen_pretrained:{pack}"
        if ".condition." in name:
            return f"trainable_adapter:{pack}" if trainable else f"frozen_adapter:{pack}"
        return f"trainable_realizer:{pack}" if trainable else f"frozen_pretrained:{pack}"
    if donor:
        return "trainable_pretrained" if trainable else "frozen_pretrained"
    return "shared_core_trainable" if trainable else "shared_core_frozen"


def _donor_tensors(realizer: nn.Module) -> Iterable[tuple[str, torch.Tensor]]:
    """Enumerate only the explicit frozen root named by each Realizer._donor.

    Pocket exposes a module root with shared flow/Mimi submodules. Diffusers
    exposes its explicitly held pipeline and its `components` mapping. Neither
    path walks arbitrary objects or Python attributes.
    """
    donor_name = getattr(type(realizer), "_donor", None)
    if not isinstance(donor_name, str):
        return
    root = getattr(realizer, donor_name, None)
    modules: list[tuple[str, nn.Module]] = []
    if isinstance(root, nn.Module):
        modules.append((donor_name, root))
    else:
        components = getattr(root, "components", None)
        if isinstance(components, Mapping):
            modules.extend(
                (f"{donor_name}.{name}", component)
                for name, component in components.items()
                if isinstance(component, nn.Module)
            )
    seen: set[int] = set()
    for prefix, module in modules:
        for name, tensor in (
            *module.named_parameters(remove_duplicate=False),
            *module.named_buffers(remove_duplicate=False),
        ):
            if id(tensor) in seen:
                continue
            seen.add(id(tensor))
            yield f"{prefix}.{name}", tensor


def resident_footprint(model: nn.Module, optimizer: Any | None = None) -> dict[str, Any]:
    """Count logical tensors and unique storage by device, dtype, and owner."""
    logical: dict[tuple[str, str, str], dict[str, int]] = defaultdict(
        lambda: {
            "parameter_tensors": 0,
            "buffer_tensors": 0,
            "elements": 0,
            "unmaterialized_tensors": 0,
        }
    )
    physical: dict[tuple[str, str, str], dict[str, Any]] = defaultdict(
        lambda: {"storage_count": 0, "storage_bytes": 0, "dtypes": set()}
    )
    storage_entries: dict[tuple[str, int, int], list[tuple[str, str, str]]] = defaultdict(list)
    seen_tensor_ids: set[int] = set()
    parameters: list[nn.Parameter] = []
    entries: list[tuple[str, torch.Tensor, bool, bool]] = []
    alias_entries: list[tuple[str, torch.Tensor, bool, bool]] = []
    for name, tensor in (
        *model.named_parameters(remove_duplicate=False),
        *model.named_buffers(remove_duplicate=False),
    ):
        is_parameter = isinstance(tensor, nn.Parameter)
        if id(tensor) in seen_tensor_ids:
            alias_entries.append((name, tensor, is_parameter, False))
            continue
        seen_tensor_ids.add(id(tensor))
        entries.append((name, tensor, is_parameter, False))
        if is_parameter:
            parameters.append(tensor)
    # Frozen donors live outside `model` registration. Read only the roots
    # deliberately held by each realizer; identity dedup covers root aliases.
    for module_name, module in model.named_modules():
        if not hasattr(type(module), "_donor"):
            continue
        prefix = f"{module_name}." if module_name else ""
        for donor_path, tensor in _donor_tensors(module):
            if id(tensor) in seen_tensor_ids:
                alias_entries.append(
                    (f"{prefix}{donor_path}", tensor, isinstance(tensor, nn.Parameter), True)
                )
                continue
            seen_tensor_ids.add(id(tensor))
            entries.append(
                (f"{prefix}{donor_path}", tensor, isinstance(tensor, nn.Parameter), True)
            )

    for name, tensor, is_parameter, donor in entries:
        trainable = bool(is_parameter and tensor.requires_grad)
        key = (str(tensor.device), str(tensor.dtype), _owner(name, trainable, donor))
        item = logical[key]
        item["parameter_tensors" if is_parameter else "buffer_tensors"] += 1
        item["elements"] += tensor.numel()
        if tensor.device.type == "meta":
            item["unmaterialized_tensors"] += 1
            continue
        storage = tensor.untyped_storage()
        storage_key = (str(tensor.device), storage.data_ptr(), storage.nbytes())
        storage_entries[storage_key].append(key)

    # Preserve alternate ownership labels for an aliased logical tensor when
    # deciding who shares its one physical storage allocation.
    for name, tensor, is_parameter, donor in alias_entries:
        if tensor.device.type == "meta":
            continue
        trainable = bool(is_parameter and tensor.requires_grad)
        key = (str(tensor.device), str(tensor.dtype), _owner(name, trainable, donor))
        storage = tensor.untyped_storage()
        storage_key = (str(tensor.device), storage.data_ptr(), storage.nbytes())
        storage_entries[storage_key].append(key)

    for storage_key, owners in storage_entries.items():
        device = storage_key[0]
        owner_keys = set(owners)
        # Attribute cross-owner aliases explicitly instead of crediting the
        # same bytes to whichever tensor happened to be visited first.
        dtypes = sorted({key[1] for key in owner_keys})
        dtype = dtypes[0] if len(dtypes) == 1 else "mixed"
        owner = next(iter(owner_keys))[2] if len(owner_keys) == 1 else "shared_storage_alias"
        group = physical[(device, dtype, owner)]
        group["storage_count"] += 1
        group["storage_bytes"] += storage_key[2]
        group["dtypes"].update(dtypes)

    trainable = [p for p in parameters if p.requires_grad]
    trainable_logical_count = sum(
        item["parameter_tensors"]
        for (device, dtype, owner), item in logical.items()
        if owner.startswith("trainable_pretrained")
        or owner.startswith("shared_core_trainable")
        or owner.startswith("trainable_adapter:")
        or owner.startswith("trainable_realizer:")
    )
    master_upper = sum(p.numel() * 4 for p in trainable)
    adam_upper = sum(p.numel() * 12 for p in trainable)
    masters = getattr(optimizer, "masters", None) if optimizer is not None else None
    if isinstance(masters, dict):
        # MasterAdamW stores FP32 model parameters directly as their own
        # masters. Count only additional allocations, deduplicated by physical
        # storage against both the model and other master views.
        model_storage_keys = set(storage_entries)
        master_storage_keys: set[tuple[str, int, int]] = set()
        for tensor in masters.values():
            if not isinstance(tensor, torch.Tensor) or tensor.device.type == "meta":
                continue
            storage = tensor.untyped_storage()
            key = (str(tensor.device), storage.data_ptr(), storage.nbytes())
            if key not in model_storage_keys:
                master_storage_keys.add(key)
        actual_master = sum(key[2] for key in master_storage_keys)
    else:
        actual_master = "unmeasured"
    state_bytes: int | str = "unmeasured"
    if optimizer is not None:
        supported = False
        measured = 0
        for inner_name in ("optimizer", "muon_optimizer"):
            inner = getattr(optimizer, inner_name, None)
            if inner is None:
                continue
            supported = True
            for state in inner.state.values():
                measured += sum(
                    value.numel() * value.element_size()
                    for value in state.values()
                    if isinstance(value, torch.Tensor)
                )
        if supported:
            state_bytes = measured
    return {
        "schema": "megatoken.training.tensor-accounting/1",
        "logical_tensors": [
            {"device": device, "dtype": dtype, "owner": owner, **counts}
            for (device, dtype, owner), counts in sorted(logical.items())
        ],
        "physical_storage": [
            {
                "device": device,
                "dtype": dtype,
                "dtypes": sorted(counts["dtypes"]),
                "owner": owner,
                "storage_count": counts["storage_count"],
                "storage_bytes": counts["storage_bytes"],
            }
            for (device, dtype, owner), counts in sorted(physical.items())
        ],
        "unique_storage_count": len(storage_entries),
        "trainable_parameter_tensors": trainable_logical_count,
        "optimizer_parameter_tensors": len(trainable),
        "fp32_optimizer_master_bytes_upper_bound": master_upper,
        "adam_optimizer_total_bytes_upper_bound": adam_upper,
        "optimizer_master_bytes_actual": actual_master,
        "optimizer_state_bytes_actual": state_bytes,
        "runtime_workspace_bytes": "unmeasured",
        "gpu_fit": "not_claimed",
    }
