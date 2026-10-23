"""Sparse approximate transfer from pinned native donor n-gram tables."""

from __future__ import annotations

import hashlib
import math
import os
import random
import shutil
import tempfile
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tokenizers import Tokenizer

from megatoken.io import atomic_binary_writer, fsync_directory, sha256_file, write_json
from megatoken.loading import load_model_config
from megatoken.safetensors_ranges import LocalSafetensorsRowReader, SafetensorsRangeReader


def extract_native_engram(
    donor_profile: str,
    donor_root: str | Path,
    corpus_roots: Sequence[str | Path],
    output: str | Path,
    *,
    model: str,
    max_contexts: int,
    seed: int,
    split: str = "train",
    cache_dir: str | Path | None = None,
    source_mode: str = "ranges",
    disk_reserve_bytes: int = 4 << 30,
    batch_contexts: int = 256,
    max_gap_rows: int = 0,
) -> dict[str, Any]:
    """Project selected native lookup results onto Megatoken hashed byte rows.

    This is an explicitly approximate transfer. It reproduces the donor's pinned
    hash-address function, queries only rows selected by actual corpus token
    contexts, concatenates every layer/order/head component before one donor-local
    projection, and aggregates each result on the current byte suffix's native
    hashed rows. It does not copy donor row IDs or invent prefix history.
    """
    if max_contexts < 1 or batch_contexts < 1 or max_gap_rows < 0:
        raise ValueError("Context, batch and range-gap limits must be positive")
    if source_mode not in {"ranges", "local"}:
        raise ValueError("Native table source mode must be ranges or local")
    if disk_reserve_bytes < 0:
        raise ValueError("Native shard disk reserve cannot be negative")
    if not corpus_roots:
        raise ValueError("Native donor extraction requires prepared corpus roots")
    from megatoken.data import PreparedCorpus

    addressor = _load_addressor(donor_profile, donor_root)
    vocabulary = addressor.profile.vocabulary
    tokenizer_path = Path(donor_root) / "tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    model_config = load_model_config(model)
    destination = Path(output)
    if destination.exists():
        raise FileExistsError("Engram assets are immutable; choose a new output directory")
    destination.parent.mkdir(parents=True, exist_ok=True)
    cache_root = Path(cache_dir) if cache_dir else destination.parent / ".native-row-cache"
    cache_root.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    selected: list[tuple[tuple[int, ...], bytes]] = []
    eligible = 0
    max_context = addressor.max_context_tokens
    with PreparedCorpus(corpus_roots, split=split) as corpus:
        corpus_identity = corpus.identity
        for episode in corpus.iter_episodes():
            for record in episode.records:
                if record.media_type != "text/plain; charset=utf-8":
                    continue
                try:
                    text = record.payload.decode("utf-8", errors="strict")
                except UnicodeDecodeError:
                    continue
                encoding = tokenizer.encode(text, add_special_tokens=False)
                for context, suffix in _original_text_contexts(
                    text,
                    encoding.ids,
                    encoding.offsets,
                    max_context,
                    vocabulary,
                ):
                    eligible += 1
                    candidate = (context, suffix)
                    if len(selected) < max_contexts:
                        selected.append(candidate)
                    else:
                        replace_index = rng.randrange(eligible)
                        if replace_index < max_contexts:
                            selected[replace_index] = candidate
    if not selected:
        raise ValueError("The selected corpus split has no complete native donor contexts")

    refs_by_context = [addressor.address(context) for context, _ in selected]
    ref_template = refs_by_context[0]
    if not ref_template or any(len(refs) != len(ref_template) for refs in refs_by_context):
        raise ValueError("Native donor address components are not a fixed ordered layout")
    component_index = tuple(ref.component_index for ref in ref_template)
    if component_index != tuple(sorted(component_index)) or len(set(component_index)) != len(
        component_index
    ):
        raise ValueError("Native donor components are not uniquely ordered")
    width = sum(ref.width for ref in ref_template)
    if any(
        any(ref.component_index != component_index[index] for index, ref in enumerate(refs))
        or sum(ref.width for ref in refs) != width
        for refs in refs_by_context
    ):
        raise ValueError("Native donor component layout changed across selected contexts")
    shards: dict[str, tuple[str, int]] = {}
    for ref in ref_template:
        pin = (ref.shard_sha256, ref.shard_size)
        previous = shards.setdefault(ref.shard_name, pin)
        if previous != pin:
            raise ValueError("Native donor shard pins disagree across row references")
    for refs in refs_by_context:
        for ref in refs:
            pin = (ref.shard_sha256, ref.shard_size)
            previous = shards.setdefault(ref.shard_name, pin)
            if previous != pin:
                raise ValueError("Native donor shard pins disagree across row references")
    _verify_remote_shards(
        addressor.profile.profile.repository,
        addressor.profile.profile.revision,
        shards,
    )
    local_shards: dict[str, Path] = {}
    local_download_bytes = 0
    if source_mode == "local":
        local_shards, local_download_bytes = _download_native_shards(
            addressor.profile.profile.repository,
            addressor.profile.profile.revision,
            shards,
            cache_root / "hf-shards",
            disk_reserve_bytes=disk_reserve_bytes,
        )
    projection_seed = seed ^ int.from_bytes(
        hashlib.sha256(donor_profile.encode("utf-8")).digest()[:8], "little"
    )
    projection_generator = torch.Generator().manual_seed(projection_seed)
    projection = torch.randn(width, 128, generator=projection_generator) / math.sqrt(width)

    reader_cache: dict[tuple[str, str], SafetensorsRangeReader] = {}
    sums: dict[int, np.ndarray] = {}
    counts: Counter[int] = Counter()
    collisions: dict[int, set[bytes]] = {}
    donor_row_contexts: dict[tuple[int, int], tuple[bytes, bool]] = {}
    query_components = 0
    for start in range(0, len(selected), batch_contexts):
        batch_refs = refs_by_context[start : start + batch_contexts]
        requests: dict[tuple[str, str], set[int]] = {}
        for refs in batch_refs:
            for ref in refs:
                requests.setdefault((ref.shard_name, ref.tensor_name), set()).add(ref.row)
                if ref.scale_tensor_name:
                    requests.setdefault((ref.shard_name, ref.scale_tensor_name), set()).add(
                        int(ref.scale_row)
                    )
        fetched: dict[tuple[str, str], dict[int, torch.Tensor]] = {}
        for (shard, tensor_name), rows in requests.items():
            first_ref = next(
                ref
                for refs in batch_refs
                for ref in refs
                if ref.shard_name == shard
                and (ref.tensor_name == tensor_name or ref.scale_tensor_name == tensor_name)
            )
            shard_hash = first_ref.shard_sha256
            shard_size = first_ref.shard_size
            key = (shard, tensor_name)
            reader = reader_cache.get(key)
            if reader is None:
                url = (
                    f"https://huggingface.co/{addressor.profile.profile.repository}/resolve/"
                    f"{addressor.profile.profile.revision}/{shard}"
                )
                if source_mode == "local":
                    reader = LocalSafetensorsRowReader(str(local_shards[shard]), tensor_name)
                else:
                    reader = SafetensorsRangeReader(
                        url,
                        tensor_name,
                        file_size=shard_size,
                        expected_sha256=shard_hash,
                        headers=None,
                        row_cache=str(cache_root / f"{shard_hash}.sqlite"),
                        max_gap_rows=max_gap_rows,
                    )
                reader_cache[key] = reader
            fetched[key] = reader.read_rows(sorted(rows))

        for refs, (context_ids, suffix) in zip(
            batch_refs, selected[start : start + len(batch_refs)], strict=True
        ):
            components: list[torch.Tensor] = []
            context_digests: dict[int, bytes] = {}
            for ref in refs:
                donor_key = (ref.component_index, ref.global_row)
                context_digest = context_digests.get(ref.order)
                if context_digest is None:
                    lookup_key = addressor.lookup_key(context_ids, ref.order)
                    context_digest = hashlib.sha256(
                        b"native-lookup-key-v1\0"
                        + b"".join(int(token).to_bytes(4, "little") for token in lookup_key)
                    ).digest()
                    context_digests[ref.order] = context_digest
                prior = donor_row_contexts.get(donor_key)
                if prior is None:
                    donor_row_contexts[donor_key] = (context_digest, False)
                elif not prior[1] and prior[0] != context_digest:
                    donor_row_contexts[donor_key] = (prior[0], True)
                raw = fetched[(ref.shard_name, ref.tensor_name)][ref.row].float().reshape(-1)
                if ref.scale_tensor_name:
                    scales = (
                        fetched[(ref.shard_name, ref.scale_tensor_name)][int(ref.scale_row)]
                        .float()
                        .reshape(-1)
                    )
                    if ref.block_size is None or raw.numel() != scales.numel() * ref.block_size:
                        raise ValueError("Native donor block-scale layout is inconsistent")
                    raw = raw.reshape(-1, ref.block_size) * scales.repeat_interleave(
                        ref.block_size
                    ).reshape(-1, ref.block_size)
                    raw = raw.reshape(-1)
                components.append(raw)
            native_vector = torch.cat(components)
            # Dequantize each source component first, then normalize its combined
            # donor-local lookup vector to keep the seeded projection well-scaled.
            native_vector = native_vector / native_vector.square().mean().sqrt().clamp_min(1e-8)
            descriptor = native_vector @ projection
            if not bool(torch.isfinite(descriptor).all()):
                raise ValueError("Native donor lookup produced non-finite descriptors")
            query_components += len(refs)
            for row, key in _megatoken_hashed_suffixes(suffix, model_config):
                if row not in sums:
                    sums[row] = np.zeros(128, dtype=np.float64)
                    collisions[row] = set()
                sums[row] += descriptor.numpy().astype(np.float64)
                counts[row] += 1
                collisions[row].add(hashlib.sha256(key).digest())

    for reader in reader_cache.values():
        reader.close()
    rows = np.asarray(sorted(sums), dtype=np.int64)
    descriptors = np.stack([sums[int(row)] / counts[int(row)] for row in rows]).astype(np.float32)
    support = np.asarray(
        [min(1.0, math.log1p(counts[int(row)]) / math.log1p(max_contexts)) for row in rows],
        dtype=np.float32,
    )
    temporary = Path(tempfile.mkdtemp(prefix=".native-engram-", dir=destination.parent))
    try:
        with (temporary / "descriptors.npz").open("wb") as handle:
            np.savez_compressed(
                handle,
                rows=rows,
                descriptors=descriptors,
                coverage=np.ones(len(rows), dtype=np.float32),
                support=support,
            )
            handle.flush()
            os.fsync(handle.fileno())
        profile = addressor.profile.profile
        license_path = Path(donor_root) / "LICENSE"
        if not license_path.is_file():
            raise ValueError("Pinned donor license file is missing from the metadata snapshot")
        license_bytes = license_path.read_bytes()
        with atomic_binary_writer(temporary / "DONOR-LICENSE") as handle:
            handle.write(license_bytes)
        profile_metadata = {
            "donor_kind": f"native_{addressor.native_memory}",
            "repository": profile.repository,
            "revision": profile.revision,
            "license_id": profile.license_id,
            "tokenizer_sha256": profile.tokenizer_sha256,
            "config_sha256": profile.config_sha256,
            "index_sha256": profile.index_sha256,
        }
        metadata = {
            "format_version": 1,
            "algorithm": "native-context-to-byte-hash-v1",
            "row_space": "native_hashed",
            "descriptor_sha256": sha256_file(temporary / "descriptors.npz"),
            "donor_profile": donor_profile,
            "donor_native_memory": addressor.native_memory,
            "source_mode": source_mode,
            "donor_metadata": profile_metadata,
            "donor_license_sha256": hashlib.sha256(license_bytes).hexdigest(),
            "donor_tokenizer_fingerprint": addressor.profile.vocabulary.fingerprint,
            "donor_native_mapping_sha256": addressor.compressed_token_map_sha256,
            "native_reference": _reference(donor_profile),
            "native_table_shards": sorted(
                {
                    (ref.shard_name, ref.shard_sha256, ref.shard_size)
                    for refs in refs_by_context
                    for ref in refs
                }
            ),
            "corpus_fingerprint": corpus_identity,
            "corpus_split": split,
            "eligible_contexts": eligible,
            "selected_contexts": len(selected),
            "selection": "seeded-reservoir-over-real-token-contexts-v1",
            "seed": seed,
            "projection_seed": projection_seed,
            "max_contexts": max_contexts,
            "max_context_tokens": max_context,
            "native_components_per_context": len(ref_template),
            "native_component_order": [
                {
                    "layer": ref.layer_id,
                    "order": ref.order,
                    "head": ref.head,
                    "component": ref.component_index,
                }
                for ref in ref_template
            ],
            "query_components": query_components,
            "covered_rows": len(rows),
            "hashed_collisions": sum(max(0, len(keys) - 1) for keys in collisions.values()),
            "donor_table_collision_rows": sum(
                collided for _, collided in donor_row_contexts.values()
            ),
            "distinct_donor_component_rows": len(donor_row_contexts),
            "descriptor_dim": 128,
            "exact_rows": model_config.exact_rows,
            "ngram_rows": model_config.ngram_rows,
            "ngram_orders": list(model_config.ngram_orders),
            "exact_bigrams": model_config.exact_bigrams,
            "hash_algorithm": "rolling-257-order-seeded-v1",
            "projection": "one seeded donor-local projection over ordered dequantized native heads/layers",
            "range_requests": sum(reader.requests for reader in reader_cache.values()),
            "range_bytes_received": sum(reader.bytes_received for reader in reader_cache.values()),
            "range_cache": "pinned-shard-and-tensor keyed SQLite rows",
            "range_max_gap_rows": max_gap_rows,
            "local_shard_count": len(local_shards),
            "local_download_bytes": local_download_bytes,
            "local_disk_reserve_bytes": disk_reserve_bytes,
        }
        write_json(temporary / "engram.json", metadata)
        fsync_directory(temporary)
        os.rename(temporary, destination)
        fsync_directory(destination.parent)
        return metadata
    finally:
        if temporary.exists():
            import shutil

            shutil.rmtree(temporary)


def _megatoken_hashed_suffixes(suffix: bytes, config) -> list[tuple[int, bytes]]:
    hash_rows = config.ngram_rows - config.exact_rows
    found = []
    for order in config.ngram_orders:
        if order <= 2 or len(suffix) < order:
            continue
        key = suffix[-order:]
        value = order
        for byte in key:
            value = (value * 257 + byte + 1) % hash_rows
        found.append((config.exact_rows + value, key))
    return found


def _original_text_contexts(text, token_ids, offsets, width, vocabulary):
    """Yield complete token windows paired with their exact original UTF-8 bytes.

    Tokenizer normalizers can change spelling, and byte fallback can assign the
    same character span to several tokens. Hash addresses use token IDs, while
    Megatoken keys must use the original source bytes; windows cutting through a
    shared character span are therefore omitted.
    """
    if len(token_ids) != len(offsets):
        raise ValueError("Donor tokenizer returned mismatched IDs and source offsets")
    for end in range(width - 1, len(token_ids)):
        first = end - width + 1
        token_window = tuple(int(value) for value in token_ids[first : end + 1])
        spans = offsets[first : end + 1]
        if any(not 0 <= start < stop <= len(text) for start, stop in spans):
            continue
        if first and offsets[first - 1][1] > spans[0][0]:
            continue
        if end + 1 < len(offsets) and offsets[end + 1][0] < spans[-1][1]:
            continue
        if any(
            spans[i][0] > spans[i + 1][0] or spans[i][1] < spans[i + 1][0]
            for i in range(len(spans) - 1)
        ):
            continue
        if any(not vocabulary.pieces[token_id] for token_id in token_window):
            continue
        raw_suffix = text[spans[0][0] : spans[-1][1]].encode("utf-8")
        if raw_suffix:
            yield token_window, raw_suffix


def _load_addressor(name: str, root: str | Path):
    from megatoken.native_donor_hash import load_native_donor

    return load_native_donor(name, root)


def _verify_remote_shards(repository: str, revision: str, shards: dict[str, tuple[str, int]]):
    """Bind expected byte ranges to Hugging Face LFS metadata without fetching weights."""
    from huggingface_hub import HfApi

    items = HfApi(token=False).get_paths_info(
        repository,
        list(sorted(shards)),
        revision=revision,
        repo_type="model",
    )
    found = {item.path: item for item in items}
    if set(found) != set(shards):
        raise ValueError("Pinned Hub revision did not report every native donor table shard")
    for name, (expected_sha, expected_size) in shards.items():
        lfs = getattr(found[name], "lfs", None)
        if (
            lfs is None
            or getattr(lfs, "sha256", None) != expected_sha
            or getattr(found[name], "size", None) != expected_size
        ):
            raise ValueError(f"Native donor shard pin differs from official metadata: {name}")


def _download_native_shards(
    repository: str,
    revision: str,
    shards: dict[str, tuple[str, int]],
    cache_dir: str | Path,
    *,
    disk_reserve_bytes: int,
) -> tuple[dict[str, Path], int]:
    """Download only the referenced pinned table shards in cloud-local HF cache."""
    from huggingface_hub import hf_hub_download, try_to_load_from_cache

    destination = Path(cache_dir)
    destination.mkdir(parents=True, exist_ok=True)
    cached: dict[str, Path] = {}
    missing_bytes = 0
    for shard, (_, size) in shards.items():
        existing = try_to_load_from_cache(
            repo_id=repository,
            filename=shard,
            revision=revision,
            repo_type="model",
            cache_dir=str(destination),
        )
        if isinstance(existing, str) and Path(existing).is_file():
            if Path(existing).stat().st_size == size:
                cached[shard] = Path(existing)
                continue
        missing_bytes += size
    free_bytes = shutil.disk_usage(destination).free
    required = missing_bytes + disk_reserve_bytes
    if free_bytes < required:
        raise ValueError(
            "Insufficient cloud cache space for selected native donor shards: "
            f"need {required} bytes ({missing_bytes} shard bytes plus "
            f"{disk_reserve_bytes} reserve), have {free_bytes}"
        )

    downloaded_bytes = 0
    for shard, (expected_sha, expected_size) in shards.items():
        path = cached.get(shard)
        if path is None:
            path = Path(
                hf_hub_download(
                    repo_id=repository,
                    filename=shard,
                    revision=revision,
                    repo_type="model",
                    cache_dir=str(destination),
                    token=False,
                )
            )
            downloaded_bytes += expected_size
        if not path.is_file() or path.stat().st_size != expected_size:
            raise ValueError(f"Downloaded native donor shard has the wrong size: {shard}")
        if sha256_file(path) != expected_sha:
            raise ValueError(f"Downloaded native donor shard SHA256 mismatch: {shard}")
        cached[shard] = path
    return cached, downloaded_bytes


def _reference(name: str) -> str:
    from megatoken.native_donor_hash import DEEPSEEK_REFERENCE, QWEN_REFERENCE

    return DEEPSEEK_REFERENCE if name == "deepseek-v4.1-flash-1017" else QWEN_REFERENCE
