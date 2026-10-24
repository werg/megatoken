#!/usr/bin/env python3
"""Republish retained source episodes after causal duplicate filtering changes.

Never changes content, timestamps, targets, original families or frozen splits.
All tensors are copied and checked; the original release remains immutable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from megatoken.data import DedupConfig, DedupStore, Episode
from megatoken.data.media_assets import TensorAssetStore
from megatoken.data.sources import TermsReview
from megatoken.data.storage import export_prepared
from megatoken.io import content_hash, sha256_file
from megatoken.loading import load_codec


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=Path, required=True)
    p.add_argument("--assets", type=Path, required=True)
    p.add_argument("--receipt", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--codec", default="byte")
    args = p.parse_args()
    if args.root.exists():
        p.error("Choose a new immutable release root")
    original = json.loads(args.receipt.read_text())
    TermsReview(**original["terms"]).require_approval()
    args.root.mkdir(parents=True)
    codec = load_codec(args.codec)
    code = Path(__file__).resolve().parents[1] / "src/megatoken/data"
    receipt = {
        "operation": "causal-view republication",
        "original_receipt": original,
        "episodes_sha256": sha256_file(args.episodes),
        "original_receipt_sha256": sha256_file(args.receipt),
        "dedup_sha256": sha256_file(code / "dedup.py"),
        "causal_identity_sha256": sha256_file(code / "causal_identity.py"),
        "content_and_frozen_splits_changed": False,
    }
    with DedupStore(
        args.root / "dedup.sqlite", DedupConfig(train_fraction=1, validation_fraction=0)
    ) as store:
        with args.episodes.open() as handle, store.transaction():
            for line in handle:
                if line.strip():
                    store.add(Episode.from_dict(json.loads(line)), codec)
        store.freeze()
        result = export_prepared(
            store,
            args.root,
            codec=codec,
            assets=TensorAssetStore(args.assets),
            source_receipts=[receipt],
            pipeline_fingerprint=content_hash(receipt),
        )
    print(json.dumps({"fingerprint": result["fingerprint"], "counts": result["counts"]}))


if __name__ == "__main__":
    main()
