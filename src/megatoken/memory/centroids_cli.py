"""Deterministic, bounded spherical centroid fitting from pinned SQLite source keys.

Offline data-prep utility; never changes indexed source keys or trains an online
teacher. A fixed reservoir bounds RAM for large sources. Its output .npy is
immutable and must be content-hash pinned in the live training recipe.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from megatoken.io import sha256_file
from megatoken.memory.recipe import RetrievalRecipe, open_bank_for_recipe


def _normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x.astype(np.float64), axis=-1, keepdims=True)
    if not np.isfinite(norms).all() or (norms < 1e-8).any():
        raise ValueError("IVF address matrix contains a zero or nonfinite row")
    return (x / norms.astype(np.float32)).astype("<f4", copy=False)


def fit_centroids(
    recipe: RetrievalRecipe, *, bank_path: Path, output: Path,
    cells: int, max_samples: int = 32768, iterations: int = 12,
    seed: int = 12345,
) -> dict:
    if output.exists() or output.with_suffix(output.suffix + ".manifest.json").exists():
        raise FileExistsError("Never overwrite pinned semantic-space centroids")
    if not output.name.endswith(".npy") or (
        cells < 1 or max_samples < cells or iterations < 1
    ):
        raise ValueError("Centroid sampling/iterations and file layout must be bounded")
    rng = np.random.default_rng(seed)
    samples = []
    seen = 0
    with open_bank_for_recipe(recipe, bank_path, readonly=True) as bank:
        for row in bank.db.execute(
            "SELECT key FROM records WHERE collection='source' AND active=1 ORDER BY id"
        ):
            address = np.frombuffer(row[0], dtype="<f4")
            if address.shape != (recipe.space.dimension,) or not np.isfinite(address).all():
                raise ValueError("Source catalog contains malformed semantic keys")
            seen += 1
            if len(samples) < max_samples:
                samples.append(address.copy())
            else:
                slot = int(rng.integers(0, seen))
                if slot < max_samples:
                    samples[slot] = address.copy()
        if seen < cells:
            raise ValueError("Cannot fit more IVF cells than pinned source records")
        bank_identity = bank.database_id
        bank_key_revision = bank.revisions().key
    matrix = _normalize(np.stack(samples))
    centers = matrix[rng.choice(len(matrix), size=cells, replace=False)].copy()
    # Assignments are streamed in bounded matrix batches; no N-by-K cache.
    for _ in range(iterations):
        totals = np.zeros_like(centers)
        counts = np.zeros(cells, dtype=np.int64)
        for start in range(0, len(matrix), 2048):
            block = matrix[start:start + 2048]
            closest = np.argmax(block @ centers.T, axis=1)
            np.add.at(totals, closest, block)
            np.add.at(counts, closest, 1)
        occupied = counts > 0
        if np.any(occupied):
            centers[occupied] = _normalize(totals[occupied])
        # Empty clusters do not remain zero. Deterministic seeded reseeding
        # preserves exactly the requested number of spherical partitions.
        if np.any(~occupied):
            refill = rng.choice(len(matrix), size=int((~occupied).sum()), replace=False)
            centers[~occupied] = matrix[refill]
    centers = _normalize(centers)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=output.parent, prefix=".ivf-centroids-", suffix=".npy", delete=False,
    ) as staging:
        staged_path = Path(staging.name)
        np.save(staging, centers, allow_pickle=False)
    try:
        digest = sha256_file(staged_path)
        os.replace(staged_path, output)
        receipt = {
            "schema": "megatoken-ivf-centroids-v1",
            "semantic_space": recipe.space.fingerprint,
            "recipe": recipe.fingerprint,
            "database_id": bank_identity,
            "bank_key_generation": bank_key_revision,
            "source_records": seen,
            "sample_count": len(samples),
            "cells": cells,
            "dimension": recipe.space.dimension,
            "iterations": iterations,
            "seed": seed,
            "centroid_sha256": digest,
            "normalized": True,
        }
        output.with_suffix(output.suffix + ".manifest.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n"
        )
        return receipt
    finally:
        staged_path.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Fit bounded normalized IVF centroids")
    p.add_argument("--recipe", required=True, type=Path)
    p.add_argument("--bank", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--cells", required=True, type=int)
    p.add_argument("--max-samples", type=int, default=32768)
    p.add_argument("--iterations", type=int, default=12)
    p.add_argument("--seed", type=int, default=12345)
    args = p.parse_args(argv)
    print(json.dumps(fit_centroids(
        RetrievalRecipe.load(args.recipe),
        bank_path=args.bank, output=args.output, cells=args.cells,
        max_samples=args.max_samples, iterations=args.iterations,
        seed=args.seed,
    ), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
