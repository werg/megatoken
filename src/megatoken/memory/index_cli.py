"""Pinned SQLite-IVF posting publisher for source and learned memory collections.

Use after source-catalog creation, before training and at centroid-revision
boundaries. The training owner updates touched key postings transactionally
and refreshes newly allocated record postings at its growth barrier.

Publication also calibrates the probe count: the smallest ``nprobe`` whose IVF
top-k agrees with exact top-k on held-out query vectors at ``target_recall``.
Every reader takes it from the bank's publication receipt
(``SQLiteDiskIVF.published``); nothing else chooses a probe count.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from megatoken.io import sha256_file
from megatoken.memory.contracts import Collection
from megatoken.memory.disk_ivf import SQLiteDiskIVF
from megatoken.memory.recipe import RetrievalRecipe, open_bank_for_recipe

MAX_CALIBRATION_KEYS = 262_144


def calibrate_nprobe(
    keys: np.ndarray,
    centroids: np.ndarray,
    queries: np.ndarray,
    *,
    k: int = 10,
    target_recall: float = 0.97,
) -> tuple[int, dict[int, float]]:
    """Smallest probe count whose IVF top-k overlaps exact top-k at ``target_recall``.

    Rows of every argument are unit vectors; ``keys`` may be a uniform sample.
    Returns the choice and the measured overlap of every probe count tried.
    """
    if not 0 < target_recall <= 1 or len(queries) < 1 or len(keys) <= k:
        raise ValueError("Calibration needs queries, more keys than k and a recall in (0, 1]")
    assignment = np.argmax(keys @ centroids.T, axis=1)
    cells = len(centroids)
    ladder = sorted({n for n in (1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128) if n < cells})
    ladder.append(cells)
    hits = dict.fromkeys(ladder, 0.0)
    # Bounded memory: one block of query rows against the key sample at a time.
    for start in range(0, len(queries), 128):
        block = queries[start : start + 128]
        scores = block @ keys.T
        exact = np.argpartition(-scores, k, axis=1)[:, :k]
        order = np.argsort(-(block @ centroids.T), axis=1, kind="stable")
        for i in range(len(block)):
            reference = set(exact[i].tolist())
            for nprobe in ladder:
                candidates = np.flatnonzero(np.isin(assignment, order[i, :nprobe]))
                top = candidates[np.argsort(-scores[i, candidates], kind="stable")[:k]]
                hits[nprobe] += len(reference.intersection(top.tolist())) / k
    curve = {nprobe: hits[nprobe] / len(queries) for nprobe in ladder}
    for nprobe in ladder:
        if curve[nprobe] >= target_recall:
            return nprobe, curve
    return cells, curve


def _key_sample(bank, dimension: int, limit: int, seed: int = 12345) -> np.ndarray:
    rng = np.random.default_rng(seed)
    sample: list[np.ndarray] = []
    seen = 0
    for (blob,) in bank.db.execute(
        "SELECT key FROM records WHERE collection='source' AND active=1 ORDER BY id"
    ):
        seen += 1
        if len(sample) < limit:
            sample.append(np.frombuffer(blob, dtype="<f4"))
        elif (slot := int(rng.integers(0, seen))) < limit:
            sample[slot] = np.frombuffer(blob, dtype="<f4")
    if not sample:
        raise ValueError("Cannot calibrate an IVF index without source keys")
    keys = np.stack(sample).reshape(len(sample), dimension)
    return keys / np.linalg.norm(keys, axis=1, keepdims=True)


def publish_index(
    recipe: RetrievalRecipe,
    *,
    bank_path: Path,
    centroid_path: Path,
    centroid_sha256: str,
    queries: np.ndarray,
    target_recall: float = 0.97,
    k: int = 10,
) -> dict:
    if not centroid_sha256 or sha256_file(centroid_path) != centroid_sha256:
        raise ValueError("IVF centroids must match a pinned immutable SHA256")
    centroids = np.load(centroid_path, allow_pickle=False)
    queries = np.asarray(queries, dtype=np.float32)
    queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    with open_bank_for_recipe(recipe, bank_path) as bank:
        nprobe, curve = calibrate_nprobe(
            _key_sample(bank, recipe.space.dimension, MAX_CALIBRATION_KEYS),
            centroids / np.linalg.norm(centroids, axis=1, keepdims=True),
            queries,
            k=k,
            target_recall=target_recall,
        )
        index = SQLiteDiskIVF(bank, centroids, nprobe=nprobe)
        scopes = [
            row[0]
            for row in bank.db.execute(
                "SELECT DISTINCT scope FROM records WHERE active=1 ORDER BY scope"
            ).fetchall()
        ]
        results = {}
        with bank.transaction():
            for scope in scopes:
                for kind in (Collection.SOURCE, Collection.MEMORY):
                    count = index.rebuild(scope=scope, collection=kind)
                    results[f"{scope}:{kind.value}"] = count
        receipt = {
            "schema": "megatoken-disk-ivf-index-v1",
            "recipe": recipe.fingerprint,
            "space": recipe.space.fingerprint,
            "database_id": bank.database_id,
            "centroid_file_sha256": centroid_sha256,
            "centroid_vectors_sha256": index.identity.centroid_sha256,
            "nprobe": nprobe,
            "calibration": {
                "k": k,
                "queries": len(queries),
                "target_recall": target_recall,
                "overlap": {str(n): v for n, v in curve.items()},
            },
            "cells": index.identity.cells,
            "scopes": scopes,
            "postings": results,
            "revisions": asdict(bank.revisions()),
        }
        bank.put_growth_state("ivf_publication", receipt)
        receipt["published_revisions"] = asdict(bank.revisions())
        return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build canonical typed SQLite-IVF postings")
    parser.add_argument("--recipe", required=True, type=Path)
    parser.add_argument("--bank", required=True, type=Path)
    parser.add_argument("--centroids", required=True, type=Path)
    parser.add_argument("--sha256", required=True)
    parser.add_argument(
        "--queries",
        required=True,
        nargs="+",
        type=Path,
        help="Held-out query sidecars whose teacher vectors calibrate the probe count",
    )
    parser.add_argument("--target-recall", type=float, default=0.97)
    args = parser.parse_args(argv)
    from megatoken.data.retrieval import read_sidecar

    queries = np.stack(
        [
            np.asarray(annotation.teacher_target, dtype=np.float32)
            for path in args.queries
            for annotation in read_sidecar(path)
        ]
    )
    report = publish_index(
        RetrievalRecipe.load(args.recipe),
        bank_path=args.bank,
        centroid_path=args.centroids,
        centroid_sha256=args.sha256,
        queries=queries,
        target_recall=args.target_recall,
    )
    args.bank.with_suffix(args.bank.suffix + ".index.json").write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
