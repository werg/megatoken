"""Pinned SQLite-IVF posting publisher for source and learned memory collections.

Use after source-catalog creation, before training and at centroid-revision
boundaries. The training owner updates touched key postings transactionally
and refreshes newly allocated record postings at its growth barrier.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from megatoken.io import sha256_file
from megatoken.memory.contracts import Collection
from megatoken.memory.disk_ivf import SQLiteDiskIVF
from megatoken.memory.recipe import RetrievalRecipe, open_bank_for_recipe


def publish_index(
    recipe: RetrievalRecipe, *, bank_path: Path,
    centroid_path: Path, centroid_sha256: str,
    nprobe: int = 8,
) -> dict:
    if not centroid_sha256 or sha256_file(centroid_path) != centroid_sha256:
        raise ValueError("IVF centroids must match a pinned immutable SHA256")
    centroids = np.load(centroid_path, allow_pickle=False)
    with open_bank_for_recipe(recipe, bank_path) as bank:
        index = SQLiteDiskIVF(bank, centroids, nprobe=nprobe)
        scopes = [
            row[0] for row in bank.db.execute(
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
            "nprobe": nprobe, "cells": index.identity.cells,
            "scopes": scopes, "postings": results,
            "revisions": asdict(bank.revisions()),
        }
        bank.put_growth_state("ivf_publication", receipt)
        return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build canonical typed SQLite-IVF postings")
    parser.add_argument("--recipe", required=True, type=Path)
    parser.add_argument("--bank", required=True, type=Path)
    parser.add_argument("--centroids", required=True, type=Path)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--nprobe", type=int, default=8)
    args = parser.parse_args(argv)
    report = publish_index(
        RetrievalRecipe.load(args.recipe), bank_path=args.bank,
        centroid_path=args.centroids, centroid_sha256=args.sha256,
        nprobe=args.nprobe,
    )
    args.bank.with_suffix(args.bank.suffix + ".index.json").write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
