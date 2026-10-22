"""Fragment planning and numerics shared by independent DiLoCo workers and syncer.

Only trainable FP32 master parameters participate in outer optimization. A stable
flat offset is used for very large packed-expert tensors; parameter ownership and
serving dtypes never change. Wire data uses safetensors, not pickle.
"""

from __future__ import annotations

import math
import os
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from megatoken.io import content_hash, fsync_directory


@dataclass(frozen=True)
class Slice:
    name: str
    offset: int
    length: int


@dataclass(frozen=True)
class Fragment:
    index: int
    slices: tuple[Slice, ...]

    @property
    def length(self) -> int:
        return sum(part.length for part in self.slices)

    def to_dict(self) -> dict[str, Any]:
        return {"index": self.index, "slices": [asdict(item) for item in self.slices]}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Fragment:
        return cls(
            index=int(value["index"]),
            slices=tuple(Slice(**item) for item in value["slices"]),
        )


def training_identity(config: Any) -> str:
    """Objective identity excludes local paths, worker IDs and logging cadence."""
    fields = (
        "phase",
        "model",
        "codec",
        "precision",
        "seed",
        "model_overrides",
        "expert_backend",
        "representation_mode",
        "generated_capacity",
        "microbatch_episodes",
        "accumulate_scored",
        "accumulate_ingested",
        "max_accumulation_windows",
        "max_history_tokens",
        "max_window_records",
        "scored_decisions",
        "max_record_bytes",
        "effort",
        "use_stage_mixture",
        "use_effort_mixture",
        "loss_weights",
        "grad_clip",
        "weight_decay",
        "inner_optimizer",
        "gradient_checkpointing",
        "budget_meter",
        "accumulate_events",
        "router_balance_mode",
        "trainable",
        "text_anchor_weight",
    )
    identity: dict[str, Any] = {name: getattr(config, name) for name in fields}
    # Pinned inputs are identified by content, not by per-sandbox paths.
    from megatoken.io import sha256_file

    for name in ("packs", "text_anchors"):
        path = getattr(config, name)
        if path:
            identity[name] = sha256_file(Path(path))
    return content_hash(identity)


def parameter_identity(masters: Mapping[str, torch.Tensor]) -> str:
    return content_hash({name: list(p.shape) for name, p in sorted(masters.items())})


def plan_fragments(
    masters: Mapping[str, torch.Tensor], *, target_bytes: int = 64 << 20
) -> tuple[Fragment, ...]:
    """Deterministic, balanced byte-sized fragments, including huge expert tensors."""
    if target_bytes < 1024 or target_bytes % 4:
        raise ValueError("fragment size must be >=1024 bytes and a multiple of four")
    capacity = target_bytes // 4
    result: list[Fragment] = []
    current: list[Slice] = []
    remaining = capacity

    def emit() -> None:
        nonlocal current, remaining
        if current:
            result.append(Fragment(len(result), tuple(current)))
            current = []
            remaining = capacity

    for name, param in sorted(masters.items()):
        if not torch.is_floating_point(param):
            raise ValueError(f"Nonfloating master parameter {name}")
        start = 0
        while start < param.numel():
            count = min(remaining, param.numel() - start)
            current.append(Slice(name, start, count))
            start += count
            remaining -= count
            if not remaining:
                emit()
    emit()
    if not result:
        raise ValueError("A DiLoCo run needs trainable parameters")
    return tuple(result)


@torch.no_grad()
def pack_fragment(masters: Mapping[str, torch.Tensor], fragment: Fragment) -> torch.Tensor:
    """Take a consistent snapshot only at a completed optimizer-step boundary.

    Device slices land in one pinned FP32 host buffer via non-blocking copies
    and a single synchronization (pageable per-slice copies are >20x slower).
    """
    on_device = any(masters[part.name].device.type == "cuda" for part in fragment.slices)
    packed = torch.empty(fragment.length, dtype=torch.float32, pin_memory=on_device)
    cursor = 0
    for part in fragment.slices:
        source = masters[part.name].detach().reshape(-1)[part.offset : part.offset + part.length]
        packed[cursor : cursor + part.length].copy_(source, non_blocking=on_device)
        cursor += part.length
    if on_device:
        torch.cuda.synchronize()
    return packed


@torch.no_grad()
def install_fragment(optimizer: Any, fragment: Fragment, values: torch.Tensor) -> None:
    """Overwrite masters and served tensors; preserve *inner* optimizer moments."""
    if values.ndim != 1 or values.numel() != fragment.length or not values.isfinite().all():
        raise ValueError("Invalid synchronized parameter fragment")
    cursor = 0
    touched: set[str] = set()
    for part in fragment.slices:
        master = optimizer.masters[part.name]
        master.detach().reshape(-1)[part.offset : part.offset + part.length].copy_(
            values[cursor : cursor + part.length].to(master.device, master.dtype)
        )
        cursor += part.length
        touched.add(part.name)
    for name in touched:
        master = optimizer.masters[name]
        served = optimizer.model_parameters[name]
        if master is not served:
            served.copy_(master)


def merge_pseudogradients(
    deltas: list[torch.Tensor],
    tokens: list[int],
    steps: list[int],
    staleness: list[int],
) -> torch.Tensor:
    """Paper's token quantity x tokens-per-step quality, with stale down-weighting.

    Pseudogradient is baseline minus worker weights. Normalization is over
    contributors, not configured worker count (failed workers have no vote).
    """
    if not deltas or not (len(deltas) == len(tokens) == len(steps) == len(staleness)):
        raise ValueError("Malformed outer gradient quorum")
    if any(t <= 0 or s <= 0 or age < 0 for t, s, age in zip(tokens, steps, staleness, strict=True)):
        raise ValueError("Outer gradients require positive counts and nonnegative age")
    if any(d.shape != deltas[0].shape or not torch.isfinite(d).all() for d in deltas):
        raise ValueError("Invalid outer gradient tensor")
    weights = [
        (float(t) * (float(t) / float(s))) / (1.0 + float(age))
        for t, s, age in zip(tokens, steps, staleness, strict=True)
    ]
    if any(not math.isfinite(value) or value <= 0 for value in weights):
        raise ValueError("Invalid merge weight")
    total = sum(weights)
    return sum(
        (d.float() * (w / total) for d, w in zip(deltas, weights, strict=True)),
        torch.zeros_like(deltas[0], dtype=torch.float32),
    )


def outer_nesterov(
    theta: torch.Tensor,
    velocity: torch.Tensor,
    pseudo_gradient: torch.Tensor,
    *,
    lr: float,
    momentum: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not (lr > 0 and 0 <= momentum < 1):
        raise ValueError("Invalid outer Nesterov hyperparameters")
    velocity = velocity * momentum + pseudo_gradient
    next_theta = theta - lr * (pseudo_gradient + momentum * velocity)
    if not torch.isfinite(next_theta).all():
        raise ValueError("Outer update generated nonfinite parameters")
    return next_theta, velocity


def save_tensors(
    path: str | Path, tensors: dict[str, torch.Tensor], metadata: dict[str, str] | None = None
) -> None:
    """Atomically commit a checksummed-by-format safetensors file with fsync."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, filename = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(fd)
    try:
        save_file(
            {k: v.detach().cpu().contiguous() for k, v in tensors.items()},
            filename,
            metadata=metadata,
        )
        with open(filename, "rb") as file:
            os.fsync(file.fileno())
        os.replace(filename, path)
        fsync_directory(path.parent)
    finally:
        Path(filename).unlink(missing_ok=True)


def load_tensors(path: str | Path) -> dict[str, torch.Tensor]:
    return load_file(str(path), device="cpu")
