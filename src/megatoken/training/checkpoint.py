"""Checksummed, atomic checkpoints containing all restart-critical state."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable

import torch

from megatoken.io import fsync_directory, sha256_file, write_json

FORMAT_VERSION = 8


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
    root = Path(root)
    if not name or Path(name).name != name or name.startswith("."):
        raise ValueError("Checkpoint name must be a simple visible filename")
    root.mkdir(parents=True, exist_ok=True)
    destination = root / name
    if destination.exists():
        raise FileExistsError(f"Refusing to replace committed checkpoint {destination}")
    temporary = Path(tempfile.mkdtemp(prefix=".checkpoint-", dir=root))
    try:
        weights = temporary / "state.pt"
        with weights.open("wb") as handle:
            torch.save({**payload, "format_version": FORMAT_VERSION}, handle)
            handle.flush()
            os.fsync(handle.fileno())
        # Stage externally owned snapshots alongside the neural optimizer BEFORE
        # publication, so latest.json can never point to half a neural/KB pair.
        extra = attachments(temporary) if attachments is not None else {}
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
            "Unsupported checkpoint format: v8 requires direct raw-facet realization "
            "and the joint multitrack core; earlier weights are unsupported"
        )
    state_path = directory / "state.pt"
    if sha256_file(state_path) != metadata["state_sha256"]:
        raise ValueError(f"Checkpoint payload checksum mismatch at {directory}")
    kb = metadata.get("memory_checkpoint")
    if kb is not None:
        memory_path = directory / kb["kb_file"]
        if memory_path.name != kb["kb_file"] or (
            not memory_path.is_file() or sha256_file(memory_path) != kb["kb_sha256"]
        ):
            raise ValueError("Checkpoint memory artifact missing or checksum mismatch")
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
