#!/usr/bin/env python3
"""Collect executed coordinator/child teaching and export a native prepared corpus.

Guest children must already pass the pinned source/native producer gates; those
gates are rerun here. Fixed initialized weights only ingest observations. No
model actions are sampled, no native guest effects are repeated, and no optimizer
is created. Coordinator policy actions use the actual serving lifecycle executor.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from megatoken.data.chunking import ChunkingConfig
from megatoken.data.dedup import DedupConfig, DedupStore
from megatoken.data.semantic_corpus import source_verifier_for_checkout
from megatoken.data.semantic_lifecycle import collect_semantic_lifecycle
from megatoken.data.storage import export_prepared
from megatoken.generation.codec import LING_TOKENIZER_SHA256, ByteCodec
from megatoken.generation.policy import FacetedGenerator
from megatoken.io import write_json
from megatoken.model import FacetedCore, ModelConfig
from megatoken.model.multitrack import MultiTrackGenerator
from megatoken.multitrack_runtime import MultiTrackRuntime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tracer-root", type=Path, required=True)
    parser.add_argument("--producer-revision", required=True)
    parser.add_argument("--consumer-revision", required=True)
    parser.add_argument("--tokenizer-json", type=Path, required=True)
    parser.add_argument("--max-positions", type=int, default=1024)
    parser.add_argument("--max-history", type=int, default=1024)
    parser.add_argument("--max-active", type=int, default=1)
    options = parser.parse_args()
    codec = ByteCodec.from_tokenizer_json(
        options.tokenizer_json, expected_sha256=LING_TOKENIZER_SHA256
    )
    verify = source_verifier_for_checkout(options.tracer_root, options.producer_revision)
    torch.set_num_threads(1)
    torch.manual_seed(17)
    model = FacetedGenerator(
        FacetedCore(
            ModelConfig.tiny(
                vocab_size=codec.vocab_size,
                max_history_chunks=options.max_history,
                max_capacity=options.max_positions,
            )
        ),
        codec,
        training_heads=False,
    ).eval()
    chunking = ChunkingConfig(hard_max_tokens=options.max_positions)
    runtime = MultiTrackRuntime(
        MultiTrackGenerator.of(model), chunking=chunking, context_archive_events=options.max_history
    )
    try:
        with torch.inference_mode():
            episode, receipt = collect_semantic_lifecycle(
                [json.loads(path.read_text()) for path in options.input],
                runtime=runtime,
                codec=codec,
                producer_revision=options.producer_revision,
                consumer_revision=options.consumer_revision,
                source_verifier=verify,
                max_history=options.max_history,
                max_positions=options.max_positions,
                max_active=options.max_active,
                representation="compact",
            )
    finally:
        runtime.close()
    options.output.mkdir(parents=True, exist_ok=True)
    write_json(options.output / "lifecycle-episode.json", episode.to_dict())
    with DedupStore(
        options.output / "lifecycle-dedup.sqlite", DedupConfig(split_seed="semantic-family/4")
    ) as store:
        if store.add(episode, codec) != "accepted_candidate":
            raise ValueError("Lifecycle collection failed whole-family native admission")
        store.freeze()
        manifest = export_prepared(
            store,
            options.output,
            codec=codec,
            source_receipts=[receipt],
            pipeline_fingerprint=receipt["episodeDigest"],
            chunking=chunking,
            sidecars={"lifecycle-admission.json": receipt},
        )
    print(
        json.dumps(
            {"manifestFingerprint": manifest["fingerprint"], "receipt": receipt},
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
