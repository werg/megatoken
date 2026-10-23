"""Pinned codec assets and offline native-table warm starts.

The shared curriculum selects the production donors. Diagnostic extraction can read a
selected donor embedding tensor, builds fixed 128D byte-key descriptors, and never
introduces a donor tokenizer or backbone into the served graph. Multiple selected
donors retain separate maps and coverage rather than mixing descriptor spaces.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import tempfile
import urllib.request
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from megatoken.generation.codec import (
    LING_REPOSITORY,
    LING_REVISION,
    LING_TOKENIZER_SHA256,
    LING_VOCAB_SIZE,
)
from megatoken.io import atomic_binary_writer, fsync_directory, sha256_file, write_json
from megatoken.loading import load_codec, load_model_config


def fetch_codec(output: str | Path, *, local_tokenizer: str | Path | None = None) -> dict[str, Any]:
    """Download one pinned tokenizer file, verify it, then publish atomically."""
    destination = Path(output)
    if destination.exists():
        codec = load_codec(destination, vocab_size=LING_VOCAB_SIZE)
        metadata = json.loads((destination / "codec.json").read_text())
        if metadata.get("tokenizer_sha256") != LING_TOKENIZER_SHA256:
            raise ValueError("Existing codec directory is not the pinned default asset")
        return {**metadata, "codec_fingerprint": codec.fingerprint}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".codec-", dir=destination.parent))
    try:
        tokenizer = temporary / "tokenizer.json"
        if local_tokenizer:
            shutil.copyfile(local_tokenizer, tokenizer)
        else:
            url = f"https://huggingface.co/{LING_REPOSITORY}/resolve/{LING_REVISION}/tokenizer.json"
            with (
                urllib.request.urlopen(url, timeout=60) as response,
                atomic_binary_writer(tokenizer) as handle,
            ):
                shutil.copyfileobj(response, handle)
        if sha256_file(tokenizer) != LING_TOKENIZER_SHA256:
            raise ValueError("Downloaded tokenizer does not match its pinned SHA256")
        codec = load_codec(tokenizer, vocab_size=LING_VOCAB_SIZE)
        metadata = {
            "format_version": 1,
            "repository": LING_REPOSITORY,
            "revision": LING_REVISION,
            "tokenizer_file": "tokenizer.json",
            "tokenizer_sha256": LING_TOKENIZER_SHA256,
            "vocab_size": codec.vocab_size,
            "lexical_rows": len(codec.valid_token_ids),
            "codec_fingerprint": codec.fingerprint,
            "normalization": "disabled; exact raw bytes",
        }
        write_json(temporary / "codec.json", metadata)
        fsync_directory(temporary)
        os.rename(temporary, destination)
        fsync_directory(destination.parent)
        return metadata
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def extract_engram(
    embedding_file: str | Path | None,
    tensor_name: str | None,
    donor_codec: str | None,
    output: str | Path,
    *,
    donor_profile: str | None = None,
    donor_root: str | Path | None = None,
    model: str = "32x512",
    seed: int = 17,
    chunk_rows: int = 2048,
    support_target: int = 32,
) -> dict[str, Any]:
    """Aggregate deterministic random projections over exact unigram/bigram keys.

    Each distinct donor lexical token containing a key contributes one descriptor
    to that key. Support is vocabulary support, not an invented corpus frequency.
    Per-token centering/RMS and a seeded 128D Gaussian projection are fixed offline.
    Hash-only longer n-grams receive no imported features. The learned 128→table
    map is small, removable, and installed before optimizer construction.
    """
    import torch
    from safetensors import safe_open

    if chunk_rows < 1 or support_target < 1:
        raise ValueError("Chunk rows and support target must be positive")
    if donor_profile:
        if embedding_file is not None or tensor_name is not None or donor_codec is not None:
            raise ValueError(
                "Use either a pinned donor profile or explicit embedding/tokenizer files"
            )
        if donor_root is None:
            raise ValueError("A donor root is required with a pinned donor profile")
        from megatoken.donors import donor_profile_metadata, load_donor_profile

        asset = load_donor_profile(donor_profile, donor_root)
        embedding_file = asset.embedding_file
        tensor_name = asset.profile.tensor_name
        codec = asset.vocabulary
        provenance = donor_profile_metadata(asset)
        if asset.embedding_file.stat().st_size != asset.profile.shard_size:
            raise ValueError("Pinned donor embedding shard has the wrong size")
        donor_digest = sha256_file(asset.embedding_file)
        if donor_digest != asset.profile.shard_sha256:
            raise ValueError("Pinned donor embedding shard SHA256 mismatch")
    else:
        if embedding_file is None or tensor_name is None or donor_codec is None:
            raise ValueError("Explicit extraction requires embedding, tensor name, and donor codec")
        codec = load_codec(donor_codec)
        provenance = {}
        donor_digest = None
    config = load_model_config(model)
    destination = Path(output)
    if destination.exists():
        raise FileExistsError("Engram assets are immutable; choose a new output directory")
    destination.parent.mkdir(parents=True, exist_ok=True)
    sums = torch.zeros(config.exact_rows, 128, dtype=torch.float64)
    counts = torch.zeros(config.exact_rows, dtype=torch.int64)
    with safe_open(str(embedding_file), framework="pt", device="cpu") as donor:
        if tensor_name not in donor.keys():
            raise ValueError(f"Donor tensor {tensor_name!r} was not found")
        weights = donor.get_slice(tensor_name)
        shape = weights.get_shape()
        if len(shape) != 2 or shape[0] != codec.vocab_size:
            raise ValueError(
                "Donor tensor rows must exactly match the raw tokenizer vocabulary; "
                "hashed n-gram tables and padded rows are not tokenizer embeddings"
            )
        if donor_profile and shape[1] != asset.profile.hidden_size:
            raise ValueError("Donor embedding width differs from its pinned profile")
        generator = torch.Generator().manual_seed(seed)
        projection = torch.randn(shape[1], 128, generator=generator) / math.sqrt(shape[1])
        valid = set(codec.valid_token_ids)
        for start in range(0, codec.vocab_size, chunk_rows):
            rows = weights[start : min(codec.vocab_size, start + chunk_rows)].float()
            rows = rows - rows.mean(-1, keepdim=True)
            rows = rows / rows.square().mean(-1, keepdim=True).clamp_min(1e-8).sqrt()
            descriptors = rows @ projection
            if not bool(torch.isfinite(descriptors).all()):
                raise ValueError("Non-finite donor embedding features")
            for offset, descriptor in enumerate(descriptors):
                index = start + offset
                if index not in valid:
                    continue
                key = codec.token_bytes(index)
                native = set(key)
                if config.exact_bigrams:
                    native.update(
                        256 + 256 * left + right for left, right in zip(key, key[1:], strict=False)
                    )
                native_rows = torch.tensor(sorted(native), dtype=torch.long)
                sums[native_rows] += descriptor.double()
                counts[native_rows] += 1
    covered = counts > 0
    rows = covered.nonzero().flatten()
    descriptors = (sums[covered] / counts[covered, None]).float()
    support = (counts[covered].float().log1p() / math.log1p(support_target)).clamp_max(1)
    temporary = Path(tempfile.mkdtemp(prefix=".engram-", dir=destination.parent))
    try:
        with atomic_binary_writer(temporary / "descriptors.npz") as handle:
            np.savez(
                handle,
                rows=rows.numpy(),
                descriptors=descriptors.numpy(),
                coverage=np.ones(len(rows), dtype=np.float32),
                support=support.numpy(),
            )
        metadata = {
            "format_version": 1,
            "algorithm": "exact-byte-vocabulary-projection-v1",
            "descriptor_sha256": sha256_file(temporary / "descriptors.npz"),
            "donor_sha256": donor_digest or sha256_file(embedding_file),
            "donor_tensor": tensor_name,
            "donor_codec_fingerprint": codec.fingerprint,
            "donor_shape": shape,
            "seed": seed,
            "support_target": support_target,
            "exact_rows": config.exact_rows,
            "ngram_rows": config.ngram_rows,
            "row_space": "exact",
            "covered_rows": len(rows),
            "descriptor_dim": 128,
            "support_unit": "distinct lexical donor tokens",
            **provenance,
        }
        write_json(temporary / "engram.json", metadata)
        fsync_directory(temporary)
        os.rename(temporary, destination)
        fsync_directory(destination.parent)
        return metadata
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def install_engram_asset(model: Any, path: str | Path) -> None:
    import torch

    source = Path(path)
    metadata = json.loads((source / "engram.json").read_text())
    if metadata["format_version"] != 1 or metadata["exact_rows"] != model.core.config.exact_rows:
        raise ValueError("Engram asset does not match the native key space")
    row_space = metadata.get("row_space", "exact")
    if row_space not in {"exact", "native_hashed"}:
        raise ValueError("Engram asset declares an unsupported row space")
    expected_rows = (
        metadata.get("ngram_rows")
        if row_space == "native_hashed"
        else metadata.get("ngram_rows", model.core.config.ngram_rows)
    )
    if expected_rows != model.core.config.ngram_rows:
        raise ValueError("Engram asset does not match the native table size")
    if row_space == "native_hashed":
        if metadata.get("algorithm") != "native-context-to-byte-hash-v1":
            raise ValueError("Native hashed asset does not declare a supported extraction method")
        if metadata.get("hash_algorithm") != "rolling-257-order-seeded-v1":
            raise ValueError("Native hashed asset does not match the current hash function")
        if metadata.get("ngram_orders") != list(model.core.config.ngram_orders):
            raise ValueError("Native hashed asset n-gram orders differ from the model")
        if metadata.get("exact_bigrams") != model.core.config.exact_bigrams:
            raise ValueError("Native hashed asset exact-byte layout differs from the model")
    if sha256_file(source / "descriptors.npz") != metadata["descriptor_sha256"]:
        raise ValueError("Engram descriptor checksum mismatch")
    with np.load(source / "descriptors.npz", allow_pickle=False) as archive:
        tensors = {
            name: torch.from_numpy(archive[name].copy())
            for name in ("rows", "descriptors", "coverage", "support")
        }
    model.core.frontend.ngrams.install_import(**tensors, allow_hashed=row_space == "native_hashed")


def install_configured_imports(model, config, *, resumed=False, phase_transition=False):
    """Honor requested donors after restore; preserve existing maps and forbid duplicates."""
    import torch

    requested = (
        [config.engram_import] if config.engram_import else list(config.engram_imports or [])
    )
    table = model.core.frontend.ngrams
    restored = ([table] if table.import_map is not None else []) + list(table.import_branches)
    if len(restored) > len(requested):
        raise ValueError("Checkpoint carries unrequested donor imports; bake them explicitly first")
    if resumed and not phase_transition and len(restored) != len(requested):
        raise ValueError(
            "Exact resume cannot introduce missing donor imports; use a phase transition"
        )
    for branch, path in zip(restored, requested, strict=False):
        source = Path(path)
        metadata = json.loads((source / "engram.json").read_text())
        if sha256_file(source / "descriptors.npz") != metadata["descriptor_sha256"]:
            raise ValueError("Restored donor descriptor checksum mismatch")
        with np.load(source / "descriptors.npz", allow_pickle=False) as archive:
            rows = torch.from_numpy(archive["rows"].copy()).long()
            order = rows.argsort()
            descriptors = torch.from_numpy(archive["descriptors"].copy()).float()[order]
            weights = torch.from_numpy((archive["coverage"] * archive["support"]).copy()).float()[
                order
            ]
        if (
            not torch.equal(branch.import_rows.cpu(), rows[order])
            or not torch.equal(branch.descriptors.cpu(), descriptors)
            or not torch.equal(branch.import_weights.cpu(), weights)
        ):
            raise ValueError(
                "Configured donor differs from the restored import; cannot reorder or replace it"
            )
    owners_before = {name for name, _ in model.named_parameters()}
    for path in requested[len(restored) :]:
        install_engram_asset(model, path)
    return tuple(
        name.rsplit(".", 1)[0] + "."
        for name, _ in model.named_parameters()
        if name not in owners_before
    )


def bake_checkpoint(
    checkpoint: str | Path,
    output: str | Path,
    *,
    codec_path: str = "byte",
    packs: str | Path | None = None,
) -> dict[str, Any]:
    """Create a new parity-checked checkpoint with import assets removed."""
    import torch

    from megatoken.io import sha256_file
    from megatoken.loading import build_model
    from megatoken.model.config import ModelConfig
    from megatoken.training.checkpoint import load_checkpoint, resolve_checkpoint, save_checkpoint
    from megatoken.training.optimizer import MasterAdamW

    payload, metadata = load_checkpoint(checkpoint)
    config = ModelConfig.from_dict(payload["model_config"])
    codec = load_codec(codec_path, vocab_size=config.vocab_size)
    if codec.fingerprint != payload["codec_fingerprint"]:
        raise ValueError("Bake codec differs from the checkpoint")
    packs = packs or payload["train_config"].get("packs")
    expected_packs = payload["packs_fingerprint"]
    actual_packs = None if packs is None else sha256_file(packs)
    if actual_packs != expected_packs:
        raise ValueError("Bake realizer packs differ from the checkpoint")
    model = build_model(config, codec, precision=payload["train_config"]["precision"], packs=packs)
    model.load_state_dict(payload["model"], strict=True)
    table = model.core.frontend.ngrams
    optimizer = None
    if payload.get("export_only"):
        masters = payload.get("master_weights")
        if not isinstance(masters, dict):
            raise ValueError("DiLoCo export lacks global FP32 master weights")
        master_name = "core.frontend.ngrams.embedding.weight"
        if master_name not in masters:
            raise ValueError("DiLoCo export lacks the native table master")
        if masters[master_name].dtype != torch.float32:
            raise ValueError("DiLoCo native table master is not FP32")
        report = asdict(table.bake(master_weights=masters[master_name]))
        masters = {
            name: value
            for name, value in masters.items()
            if name != "core.frontend.ngrams.import_map.weight"
            and not re.fullmatch(
                r"core\.frontend\.ngrams\.import_branches\.\d+\.import_map\.weight",
                name,
            )
        }
        # An export is an inference snapshot and next phase seed. It has no
        # local optimizer to restore; retain the coordinator's FP32 masters.
        saved = {
            **payload,
            "model": model.state_dict(),
            "master_weights": masters,
            "engram_bake": report,
        }
    else:
        optimizer = MasterAdamW(
            model,
            1e-5,
            payload["train_config"].get("weight_decay", 0.1),
            inner_optimizer=payload["train_config"].get("inner_optimizer", "adamw"),
        )
        try:
            optimizer.load_state_dict(payload["optimizer"])
            report = asdict(table.bake(optimizer))
            saved = {
                **payload,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "engram_bake": report,
            }
        except BaseException:
            optimizer.close()
            raise
    bank = metadata.pop("transducer_checkpoint")

    def carry_bank(stage: Path) -> dict[str, Any]:
        shutil.copyfile(resolve_checkpoint(checkpoint) / bank["file"], stage / bank["file"])
        return {"transducer_checkpoint": bank}

    try:
        path = save_checkpoint(
            output,
            f"{payload['train_config']['phase']}-{payload['global_step']:09d}-baked",
            saved,
            {**metadata, "engram_bake": report},
            pinned=True,
            attachments=carry_bank,
        )
        return {"checkpoint": str(path), **report}
    finally:
        if optimizer is not None:
            optimizer.close()
