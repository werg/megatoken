"""A computation ledger with separate transcript, policy and media meters."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class Counters:
    D: int = 0
    I: int = 0  # noqa: E741 - the specification's actual-ingestion meter
    P: int = 0
    G: int = 0  # Actual selected physical megatokens, including media.
    audio_samples: int = 0
    video_frames: int = 0
    image_events: int = 0
    media_events: int = 0
    source_bytes: int = 0
    record_ingestions: int = 0
    reconstruction_tokens: int = 0
    reconstruction_records: int = 0
    facet_encodes: int = 0
    raw_facet_replays: int = 0
    control_encodes: int = 0
    scored_windows: int = 0
    writer_reader_windows: int = 0

    def add(self, other: Counters) -> None:
        for key, value in asdict(other).items():
            setattr(self, key, getattr(self, key) + value)
        self.validate()

    def validate(self) -> None:
        if min(asdict(self).values()) < 0:
            raise ValueError("Computation counters cannot be negative")
        if self.I != self.D + self.reconstruction_tokens:
            raise ValueError("I must equal D plus no-loss reconstruction tokens")
        if self.reconstruction_records > self.record_ingestions:
            raise ValueError("Reconstruction records exceed actual ingestions")


class JsonlLogger:
    def __init__(self, path: str | Path, enabled: bool = True) -> None:
        self.path = Path(path)
        self.enabled = enabled
        self.started = time.perf_counter()
        if enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: str, **fields: Any) -> dict[str, Any]:
        record = {"event": event, "elapsed_seconds": time.perf_counter() - self.started, **fields}
        if self.enabled:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
            print(json.dumps(record, sort_keys=True, allow_nan=False), flush=True)
        return record


def rates(counters: Counters, elapsed: float) -> dict[str, float]:
    seconds = max(elapsed, 1e-9)
    return {
        f"{key}_per_second": value / seconds
        for key, value in asdict(counters).items()
        if key
        in {
            "D",
            "I",
            "P",
            "G",
            "media_events",
            "source_bytes",
            "record_ingestions",
            "reconstruction_tokens",
        }
    }


def materialize_scalars(values: dict[str, Any]) -> dict[str, float]:
    """Copy a transaction's detached diagnostic scalars to the host together.

    Calling ``float(cuda_tensor)`` at every loss or generation decision inserts
    host synchronization into the model path. Accumulate scalar tensors on their
    device, then call this once at the logging/collective boundary instead.
    """
    import torch

    tensors = {name: value for name, value in values.items() if isinstance(value, torch.Tensor)}
    output = {name: float(value) for name, value in values.items() if name not in tensors}
    if tensors:
        if any(value.numel() != 1 for value in tensors.values()):
            raise ValueError("Diagnostic metrics must be scalar tensors")
        device = next(
            (value.device for value in tensors.values() if value.device.type != "cpu"),
            next(iter(tensors.values())).device,
        )
        packed = torch.stack(
            [
                value.detach().reshape(()).to(device=device, dtype=torch.float64)
                for value in tensors.values()
            ]
        )
        output.update(zip(tensors, packed.cpu().tolist(), strict=True))
    return output


def dispatch_summary(metrics: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """The same expert utilization report for production and benchmarks."""
    result = {}
    for family, values in metrics.items():
        counts = values["routed_rows"].double()
        total = counts.sum().clamp_min(1)
        probability = counts / total
        entropy = -(probability * probability.clamp_min(1e-30).log()).sum()
        result[family] = {
            **{name: value.cpu().tolist() for name, value in values.items()},
            "total_rows": int(counts.sum().item()),
            "active_experts": int((counts > 0).sum().item()),
            "load_entropy": float(entropy.item()),
            "normalized_load_entropy": float(entropy.item() / math.log(len(counts)))
            if len(counts) > 1
            else 0.0,
            "padding_ratio": float(values["padded_rows"].item() / total.item()),
        }
    return result
