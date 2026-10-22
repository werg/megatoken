#!/usr/bin/env python3
"""Build the ``long_document_reading`` stream: answer from a document read compressed.

Each task becomes one episode: the user's question, the reference documents
joined into one document record that is read compressed (EmbeddingGemma2
hidden states, data/compression.py), and the scored assistant answer. The
answer must come from the document, so the stream teaches the model to use
compressed text, which the stage plan mixes into base pretraining and later
stages.

Tasks come from ``--tasks`` (JSONL rows ``{"id", "question", "documents": [str],
"answer"}``) and/or straight from HotpotQA (distractor) and MuSiQue-Ans with the
loaders of prepare_retrieval_corpus.py; for those, every paragraph of a
question (supporting and distractor) is part of its document.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

from megatoken.data import DedupConfig, DedupStore, Episode, Lineage, PhysicalRecord
from megatoken.data.compression import CompressionPolicy, DocumentCompressor
from megatoken.data.media_assets import TensorAssetStore
from megatoken.data.schema import sequential_meridians, stable_id
from megatoken.data.storage import export_prepared
from megatoken.loading import load_codec
from megatoken.omni.embeddinggemma2 import EmbeddingGemma2
from megatoken.teacher import PINNED_TEACHER

STREAM = "long_document_reading"


def qa_tasks(hotpot: Path | None, musique: Path | None, count: int, seed: int) -> list[dict]:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from prepare_retrieval_corpus import load_hotpot, load_musique

    rows = []
    if hotpot is not None:
        rows += load_hotpot(hotpot, "train")
    if musique is not None:
        rows += load_musique(musique, "train")
    random.Random(seed).shuffle(rows)
    return [
        {
            "id": row["qid"],
            "question": row["question"],
            "documents": [p["payload"].decode() for p in row["paragraphs"]],
            "answer": row["answer"],
        }
        for row in rows[:count]
    ]


def reading_episode(task: dict, compressor: DocumentCompressor) -> Episode:
    question = task["question"].encode()
    document = "\n\n".join(task["documents"]).encode()
    answer = task["answer"].encode()
    episode_id = stable_id("long-document-reading-v1", task["id"], question, document)
    logical = stable_id("logical-v1", episode_id, "document")
    records = (
        PhysicalRecord(f"{episode_id}-question", "user", question, provenance="natural"),
        *compressor.records(document, logical_id=logical, role="document", provenance="natural"),
        PhysicalRecord(
            f"{episode_id}-answer",
            "assistant",
            answer,
            turn_index=1,
            policy_mask=True,
            provenance="natural",
        ),
    )
    family = hashlib.sha256(document).hexdigest()
    return sequential_meridians(
        Episode(episode_id, records, Lineage(STREAM, "v1", family), stream=STREAM)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Prepared dataset root")
    parser.add_argument("--codec", required=True)
    parser.add_argument("--packs", required=True, help="Locked pack manifest (EmbeddingGemma2)")
    parser.add_argument("--tasks", type=Path, help="JSONL {id, question, documents, answer}")
    parser.add_argument("--hotpot", type=Path, help="HotpotQA distractor parquet dir")
    parser.add_argument("--musique", type=Path, help="MuSiQue-Ans JSONL dir")
    parser.add_argument("--count", type=int, default=25000, help="QA tasks to draw")
    parser.add_argument("--seed", type=int, default=20261022)
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    args = parser.parse_args()
    if (args.root / "manifest.json").exists():
        parser.error("Prepared manifest already exists; prepared data is immutable")
    tasks = []
    if args.tasks is not None:
        tasks += [json.loads(line) for line in args.tasks.read_text().splitlines() if line]
    if args.hotpot is not None or args.musique is not None:
        tasks += qa_tasks(args.hotpot, args.musique, args.count, args.seed)
    if not tasks:
        parser.error("Give --tasks and/or --hotpot/--musique")
    teacher = EmbeddingGemma2.from_manifest(args.packs)
    if teacher.extractor != PINNED_TEACHER:
        raise ValueError("The pack manifest's teacher is not the pinned teacher")
    # Every document of this stream is read compressed, whatever its length.
    compressor = DocumentCompressor(
        teacher,
        TensorAssetStore(args.root),
        CompressionPolicy(min_bytes=1, roles=frozenset({"document"})),
    )
    codec = load_codec(args.codec)
    dedup = DedupStore(
        args.root / "preparation.sqlite",
        DedupConfig(
            train_fraction=1 - args.validation_fraction,
            validation_fraction=args.validation_fraction,
        ),
    )
    try:
        with dedup.transaction():
            for task in tasks:
                dedup.add(reading_episode(task, compressor), codec)
        dedup.freeze()
        export_prepared(
            dedup,
            args.root,
            codec=codec,
            source_receipts=[{"name": STREAM, "tasks": len(tasks), "teacher": PINNED_TEACHER}],
            pipeline_fingerprint="long-document-reading-v1",
        )
    finally:
        dedup.close()
    print(json.dumps({STREAM: len(tasks)}))


if __name__ == "__main__":
    main()
