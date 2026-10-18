"""Pinned codec assets and optional, offline native-table warm starts.

Cold native initialization is the default. The optional extractor reads only a
selected local donor embedding tensor, builds fixed 128D byte-key descriptors,
and never introduces a donor tokenizer or backbone into the served graph.
"""

from __future__ import annotations

import json
import math
import os
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
    embedding_file: str | Path,
    tensor_name: str,
    donor_codec: str,
    output: str | Path,
    *,
    model: str = "recurrent32",
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
    config = load_model_config(model)
    codec = load_codec(donor_codec)
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
        if len(shape) != 2 or shape[0] < codec.vocab_size:
            raise ValueError("Donor embedding rows must cover the declared raw donor codec")
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
            "donor_sha256": sha256_file(embedding_file),
            "donor_tensor": tensor_name,
            "donor_codec_fingerprint": codec.fingerprint,
            "donor_shape": shape,
            "seed": seed,
            "support_target": support_target,
            "exact_rows": config.exact_rows,
            "covered_rows": len(rows),
            "descriptor_dim": 128,
            "support_unit": "distinct lexical donor tokens",
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
    if sha256_file(source / "descriptors.npz") != metadata["descriptor_sha256"]:
        raise ValueError("Engram descriptor checksum mismatch")
    with np.load(source / "descriptors.npz", allow_pickle=False) as archive:
        tensors = {
            name: torch.from_numpy(archive[name].copy())
            for name in ("rows", "descriptors", "coverage", "support")
        }
    model.core.frontend.ngrams.install_import(**tensors)


def bake_checkpoint(
    checkpoint: str | Path, output: str | Path, *, codec_path: str = "byte"
) -> dict[str, Any]:
    """Create a new parity-checked checkpoint with import assets removed."""
    from megatoken.loading import build_model
    from megatoken.model.config import ModelConfig
    from megatoken.training.checkpoint import load_checkpoint, save_checkpoint
    from megatoken.training.optimizer import MasterAdamW

    payload, metadata = load_checkpoint(checkpoint)
    config = ModelConfig.from_dict(payload["model_config"])
    codec = load_codec(codec_path, vocab_size=config.vocab_size)
    if codec.fingerprint != payload["codec_fingerprint"]:
        raise ValueError("Bake codec differs from the checkpoint")
    model = build_model(config, codec, precision=payload["train_config"]["precision"])
    model.load_state_dict(payload["model"], strict=True)
    optimizer = MasterAdamW(model, 1e-5, payload["train_config"].get("weight_decay", 0.1))
    try:
        optimizer.load_state_dict(payload["optimizer"])
        report = asdict(model.core.frontend.ngrams.bake(optimizer))
        saved = {
            **payload,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "engram_bake": report,
        }
        path = save_checkpoint(
            output,
            f"{payload['train_config']['phase']}-{payload['global_step']:09d}-baked",
            saved,
            {**metadata, "engram_bake": report},
            pinned=True,
        )
        return {"checkpoint": str(path), **report}
    finally:
        optimizer.close()
