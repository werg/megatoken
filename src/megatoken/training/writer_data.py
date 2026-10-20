"""Immutable, provenance-pinned writer-training shard contracts.

The preparation pipeline supplies actual frozen native E(source) blocks and
original optimized assignment targets. No answer vectors or optimized target
values are concatenated to writer inputs. The shard is checked before any
training update; missing receipts are not silently accepted.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Iterator

import torch
from torch import Tensor

from megatoken.memory.writers import AssignmentTargets
from megatoken.training.hindsight import PairWriterBatch, SourceWriterBatch


@dataclass(frozen=True)
class WriterShard:
    path: str
    sha256: str
    source_footprint_digest: str
    assignment_digest: str

    def __post_init__(self) -> None:
        if not self.path or any(not x or x.startswith("PIN_") for x in (
            self.sha256, self.source_footprint_digest, self.assignment_digest,
        )):
            raise ValueError("Writer shard requires pinned bytes and original source identity")
        if len(self.sha256) != 64 or any(c not in "0123456789abcdef" for c in self.sha256):
            raise ValueError("Writer shard SHA256 must be a lowercase hexadecimal digest")


@dataclass(frozen=True)
class WriterDataset:
    schema: str
    kind: str
    recipe_fingerprint: str
    semantic_space_fingerprint: str
    model_checkpoint_id: str
    free_bank_sha256: str
    source_context_policy: str
    mode: str
    layout: dict[str, int]
    shards: tuple[WriterShard, ...]

    def __post_init__(self) -> None:
        if self.schema != "megatoken-writer-dataset-v1":
            raise ValueError("Unsupported source-writer dataset schema")
        if self.kind not in {"source", "pair"}:
            raise ValueError("Writer dataset kind must be source or pair")
        if self.mode not in {"suffix", "full"}:
            raise ValueError("Writer dataset must declare the physical source view")
        if self.kind == "pair" and self.mode != "full":
            raise ValueError("Pair reproduction requires permitted full source footprints")
        for name in (
            "recipe_fingerprint", "semantic_space_fingerprint", "model_checkpoint_id",
            "free_bank_sha256", "source_context_policy",
        ):
            value = getattr(self, name)
            if not value or value.startswith("PIN_"):
                raise ValueError(f"Writer dataset must pin {name}")
        if not self.shards:
            raise ValueError("Writer dataset has no audited shards")
        required = (
            {"facets", "facet_dim", "semantic_dim", "value_dim", "max_blocks"}
            if self.kind == "source" else {"semantic_dim", "value_dim", "statistics_dim"}
        )
        if not required.issubset(self.layout) or any(
            not isinstance(self.layout[k], int) or self.layout[k] <= 0
            for k in required
        ):
            raise ValueError("Writer shard layout must pin strictly positive dimensions")

    @classmethod
    def load(cls, path: str | Path) -> WriterDataset:
        obj = json.loads(Path(path).read_text())
        obj["shards"] = tuple(WriterShard(**x) for x in obj["shards"])
        return cls(**obj)


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _shard_tensors(root: Path, ref: WriterShard) -> dict[str, Tensor]:
    from safetensors.torch import load_file

    path = (root / ref.path).resolve()
    # Do not allow a manifest to reach arbitrary external paths or mutate its
    # source train/eval trees via absolute or traversing references.
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError("Writer shard must exist inside the manifest directory")
    if file_sha256(path) != ref.sha256:
        raise RuntimeError(f"Writer shard checksum differs: {ref.path}")
    result = load_file(str(path), device="cpu")
    if not result or any(not bool(torch.isfinite(t).all()) for t in result.values()
                         if t.is_floating_point()):
        raise ValueError("Writer shard contains nonfinite or missing tensors")
    return result


def _required(data: dict[str, Tensor], names: tuple[str, ...]) -> tuple[Tensor, ...]:
    missing = set(names) - data.keys()
    if missing:
        raise ValueError(f"Writer shard is missing required fields: {sorted(missing)}")
    return tuple(data[name] for name in names)


def source_batches(
    manifest_path: str | Path, manifest: WriterDataset, *,
    device: torch.device, local_gap: int = 2,
) -> Iterator[SourceWriterBatch]:
    if manifest.kind != "source" or (manifest.mode == "suffix" and local_gap != 2):
        raise ValueError("Source writer requires its declared local j >= i+2 view")
    root = Path(manifest_path).resolve().parent
    for ref in manifest.shards:
        data = _shard_tensors(root, ref)
        names = (
            "encoded", "valid_blocks", "slots", "key", "value",
            "strength", "lifetime_logit", "distance", "valid",
        )
        (encoded, valid_blocks, slots, key, value, strength, lifetime,
         distance, valid) = _required(data, names)
        n, blocks, facets, width = encoded.shape
        layout = manifest.layout
        if (
            facets != layout["facets"] or width != layout["facet_dim"]
            or blocks > layout["max_blocks"]
            or key.shape[-1] != layout["semantic_dim"]
            or value.shape[-1] != layout["value_dim"]
            or slots.dtype != torch.long
        ):
            raise ValueError("Writer shard does not match pinned native semantic layout")
        target = AssignmentTargets(
            slots=slots.to(device), key=key.to(device), value=value.to(device),
            strength=strength.to(device), lifetime_logit=lifetime.to(device),
            distance=distance.to(device), valid=valid.to(device).bool(),
        )
        yield SourceWriterBatch(
            encoded.detach().to(device),
            valid_blocks.to(device).bool(),
            target, manifest.mode, local_gap,
        )


def pair_batches(
    manifest_path: str | Path, manifest: WriterDataset, *, device: torch.device,
) -> Iterator[PairWriterBatch]:
    if manifest.kind != "pair":
        raise ValueError("Expected a prepared pair-writer dataset")
    root = Path(manifest_path).resolve().parent
    for ref in manifest.shards:
        data = _shard_tensors(root, ref)
        fields = (
            "left_key", "right_key", "left_value", "right_value",
            "statistics", "target_value", "target_strength",
            "target_lifetime", "target_dependence", "valid", "distance",
        )
        values = _required(data, fields)
        key_dim = manifest.layout["semantic_dim"]
        value_dim = manifest.layout["value_dim"]
        stats_dim = manifest.layout["statistics_dim"]
        if (
            values[0].shape[-1] != key_dim or values[1].shape[-1] != key_dim
            or any(values[i].shape[-1] != value_dim for i in (2, 3, 5))
            or values[4].shape[-1] != stats_dim
            or values[8].shape[-1] != 2
        ):
            raise ValueError("Pair writer shard has incompatible key/value/edge geometry")
        yield PairWriterBatch(*(
            t.detach().to(device).bool() if i == 9 else t.detach().to(device)
            for i, t in enumerate(values)
        ))
