"""Checksummed, atomic checkpoints containing all restart-critical state."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import torch

from megatoken.io import fsync_directory, sha256_file, write_json

FORMAT_VERSION = 10


def stage_checkpoint(root: str | Path, name: str) -> Path:
    """Validate ``name`` and create the private staging directory for it.

    A root has one writer at a time, so any staging directory still lying there is the
    debris of a write that crashed; it is removed first.
    """
    root = Path(root)
    if not name or Path(name).name != name or name.startswith("."):
        raise ValueError("Checkpoint name must be a simple visible filename")
    root.mkdir(parents=True, exist_ok=True)
    if (root / name).exists():
        raise FileExistsError(f"Refusing to replace committed checkpoint {root / name}")
    for stale in root.glob(".checkpoint-*"):
        shutil.rmtree(stale, ignore_errors=True)
    return Path(tempfile.mkdtemp(prefix=".checkpoint-", dir=root))


def commit_checkpoint(
    root: str | Path,
    name: str,
    temporary: Path,
    payload: dict[str, Any],
    metadata: dict[str, Any],
    *,
    extra: dict[str, Any] | None = None,
    pinned: bool = False,
    keep: int = 3,
) -> Path:
    """Write weights next to already staged attachments, then publish atomically."""
    root = Path(root)
    destination = root / name
    try:
        weights = temporary / "state.pt"
        with weights.open("wb") as handle:
            torch.save({**payload, "format_version": FORMAT_VERSION}, handle)
            handle.flush()
            os.fsync(handle.fileno())
        extra = extra or {}
        if set(extra).intersection(metadata):
            raise ValueError("Checkpoint attachment metadata conflicts with existing keys")
        complete = {
            **metadata,
            **extra,
            "format_version": FORMAT_VERSION,
            "state_sha256": sha256_file(weights),
            "pinned": pinned,
            "checkpoint_name": name,
        }
        write_json(temporary / "metadata.json", complete)
        fsync_directory(temporary)
        os.rename(temporary, destination)
        fsync_directory(root)
        write_json(
            root / "latest.json",
            {"checkpoint": name, "metadata_sha256": sha256_file(destination / "metadata.json")},
        )
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    prune_checkpoints(root, keep=keep)
    return destination


def transducer_attachment(state: Callable[[], dict]) -> Callable[[Path], dict[str, Any]]:
    """Stage the transducer bank (``{"bank": snapshot, ...}``) beside a checkpoint.

    ``state`` returns private host tensors, so they are written without another copy.
    """

    def attach(stage: Path) -> dict[str, Any]:
        path = stage / "transducer.pt"
        with path.open("wb") as handle:
            torch.save(state(), handle)
            handle.flush()
            os.fsync(handle.fileno())
        return {"transducer_checkpoint": {"file": path.name, "sha256": sha256_file(path)}}

    return attach


def save_checkpoint(
    root: str | Path,
    name: str,
    payload: dict[str, Any],
    metadata: dict[str, Any],
    *,
    pinned: bool = False,
    keep: int = 3,
    attachments: Callable[[Path], dict[str, Any]] | None = None,
) -> Path:
    temporary = stage_checkpoint(root, name)
    try:
        # Stage externally owned snapshots alongside the neural optimizer BEFORE
        # publication, so latest.json can never point to half a neural/bank pair.
        extra = attachments(temporary) if attachments is not None else {}
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return commit_checkpoint(
        root, name, temporary, payload, metadata, extra=extra, pinned=pinned, keep=keep
    )


def snapshot_to_cpu(value: Any) -> Any:
    """Private host copy of a state tree, taken at a step boundary.

    Device tensors are copied into pinned host memory without blocking and
    synchronized once: pageable per-tensor copies are over 20x slower. Tensors
    sharing a storage view stay shared, so the serialized size matches a direct
    ``torch.save`` of the live state.
    """
    memo: dict[tuple, torch.Tensor] = {}
    copied = _snapshot(value, memo)
    if any(key[1].type == "cuda" for key in memo):
        torch.cuda.synchronize()
    return copied


def _snapshot(value: Any, memo: dict[tuple, torch.Tensor]) -> Any:
    if isinstance(value, torch.Tensor):
        tensor = value.detach()
        key = (
            tensor.untyped_storage().data_ptr(),
            tensor.device,
            tensor.dtype,
            tensor.storage_offset(),
            tuple(tensor.shape),
            tuple(tensor.stride()),
        )
        if key not in memo:
            if tensor.device.type == "cuda":
                host = torch.empty(tensor.shape, dtype=tensor.dtype, pin_memory=True)
                memo[key] = host.copy_(tensor, non_blocking=True)
            else:
                memo[key] = tensor.clone()
        return memo[key]
    if isinstance(value, dict):
        return type(value)((k, _snapshot(v, memo)) for k, v in value.items())
    if isinstance(value, list):
        return [_snapshot(v, memo) for v in value]
    if isinstance(value, tuple):
        return tuple(_snapshot(v, memo) for v in value)
    return value


class CheckpointWriter:
    """Serializes at most one checkpoint in the background while training continues.

    Callers snapshot state to host memory at the step boundary, then hand the
    write to this lane. A failed write surfaces at the next ``wait``.
    """

    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="checkpoint")
        self._future: Future | None = None

    def wait(self) -> None:
        future, self._future = self._future, None
        if future is not None:
            future.result()

    def submit(self, function: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        self.wait()
        self._future = self._executor.submit(function, *args, **kwargs)

    def close(self) -> None:
        try:
            self.wait()
        finally:
            self._executor.shutdown(wait=True)


def resolve_checkpoint(path: str | Path) -> Path:
    source = Path(path)
    if (source / "state.pt").is_file():
        return source
    if source.name == "latest.json" or (source / "latest.json").is_file():
        index = source if source.name == "latest.json" else source / "latest.json"
        value = json.loads(index.read_text())
        name = value["checkpoint"]
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError("Invalid checkpoint pointer")
        resolved = index.parent / name
        if sha256_file(resolved / "metadata.json") != value["metadata_sha256"]:
            raise ValueError("Checkpoint metadata checksum mismatch")
        return resolved
    raise FileNotFoundError(f"No committed checkpoint found at {source}")


def load_checkpoint(
    path: str | Path, *, map_location: str | torch.device = "cpu"
) -> tuple[dict[str, Any], dict[str, Any]]:
    directory = resolve_checkpoint(path)
    metadata = json.loads((directory / "metadata.json").read_text())
    if metadata.get("format_version") != FORMAT_VERSION:
        raise ValueError(
            "Unsupported checkpoint format: v10 carries the transducer bank; "
            "earlier checkpoints are unsupported"
        )
    state_path = directory / "state.pt"
    if sha256_file(state_path) != metadata["state_sha256"]:
        raise ValueError(f"Checkpoint payload checksum mismatch at {directory}")
    transducer = metadata["transducer_checkpoint"]
    bank_path = directory / transducer["file"]
    if bank_path.name != transducer["file"] or (
        not bank_path.is_file() or sha256_file(bank_path) != transducer["sha256"]
    ):
        raise ValueError("Checkpoint transducer bank missing or checksum mismatch")
    payload = torch.load(state_path, map_location=map_location, weights_only=True)
    if payload.get("format_version") != FORMAT_VERSION:
        raise ValueError("Unsupported payload format")
    return payload, metadata


def prune_checkpoints(root: str | Path, keep: int = 3) -> None:
    if keep < 1:
        raise ValueError("At least one periodic checkpoint must be retained")
    root = Path(root)
    periodic = []
    for metadata_path in root.glob("*/metadata.json"):
        try:
            metadata = json.loads(metadata_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        directory = metadata_path.parent
        if (
            metadata.get("format_version") == FORMAT_VERSION
            and metadata.get("checkpoint_name") == directory.name
            and not metadata.get("pinned", True)
        ):
            periodic.append((metadata.get("global_step", 0), directory))
    latest = None
    if (root / "latest.json").exists():
        latest = json.loads((root / "latest.json").read_text()).get("checkpoint")
    for _, directory in sorted(periodic, reverse=True)[keep:]:
        if directory.name != latest:
            shutil.rmtree(directory)
