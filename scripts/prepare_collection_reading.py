#!/usr/bin/env python3
"""Build the ``collection_reading`` stream: answer from items found in a collection.

Like ``long_document_reading``, but the supporting items are not in the context.
They sit, with every other task's items, in a source collection next to the
prepared root (``collections/<split>``, built from that split's tasks only), and the
episode holds just the question and the scored answer. The only way to answer is to
retrieve, so the stream teaches the query head (``megatoken.training.retrieval``) and
the use of observed items. The episode's annotations name the collection, the
teacher's embedding of the question, and the items that answer.

Tasks come from ``--tasks`` (JSONL rows ``{"id", "question", "answer", "documents":
[str | {"image": path}], "supporting": [index, ...]}``, an optional ``"request"``
overriding what is searched for), from HotpotQA (distractor) and MuSiQue-Ans with the
loaders in qa_corpora.py (their supporting paragraphs are the positives, every other
paragraph of every task a distractor), and from ``--captions`` (JSONL rows
``{"id", "image": path, "captions": [str, str, ...]}``: the first caption is the request
for the image, the second the answer, and every other image a distractor).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import tempfile
from pathlib import Path

import numpy as np

from megatoken.data import DedupConfig, DedupStore, Episode, Lineage, PhysicalRecord, Supervision
from megatoken.data.media_assets import TensorAssetStore
from megatoken.data.open_media import frozen_split
from megatoken.data.schema import sequential_meridians
from megatoken.data.storage import export_prepared
from megatoken.io import canonical_json
from megatoken.loading import load_codec
from megatoken.retrieval.annotation import KEY, RetrievalAnnotation
from megatoken.retrieval.collection import (
    DEFAULT_THRESHOLD,
    TEXT_LIMIT,
    SourceCollection,
    text_item_id,
)
from megatoken.retrieval.ingest import DOCUMENT_PROMPT, QUERY_PROMPT
from megatoken.teacher import PINNED_TEACHER

STREAM = "collection_reading"
MAX_REQUEST_BYTES = 1024
EMBED_BATCH = 32


def qa_tasks(hotpot: Path | None, musique: Path | None, count: int, seed: int) -> list[dict]:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from qa_corpora import load_hotpot, load_musique

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
            "answer": row["answer"],
            "documents": [p["payload"].decode() for p in row["paragraphs"]],
            "supporting": [
                i for i, p in enumerate(row["paragraphs"]) if p["content_id"] in set(row["gold"])
            ],
        }
        for row in rows[:count]
    ]


def caption_tasks(rows: list[dict]) -> list[dict]:
    return [
        {
            "id": row["id"],
            "request": row["captions"][0],
            "question": f"Describe the image that matches this caption: {row['captions'][0]}",
            "answer": row["captions"][1],
            "documents": [{"image": row["image"]}],
            "supporting": [0],
        }
        for row in rows
    ]


def embed_texts(teacher, texts: list[str], prompt: str) -> np.ndarray:
    """The teacher's unit embedding of every text, in order."""
    out = []
    for start in range(0, len(texts), EMBED_BATCH):
        batch = [("text", text) for text in texts[start : start + EMBED_BATCH]]
        out += [o.embedding.numpy() for o in teacher.encode_batch(batch, prompt=prompt)]
    return np.stack(out)


def check(task: dict) -> None:
    request = task.get("request", task["question"])
    if len(request.encode()) > MAX_REQUEST_BYTES or not task["supporting"]:
        raise ValueError(f"Task {task['id']}: a short request and a supporting item are required")
    if not set(task["supporting"]) <= set(range(len(task["documents"]))):
        raise ValueError(f"Task {task['id']}: supporting indexes outside its documents")


def fill(collection: SourceCollection, group: list[dict], preparer) -> dict[str, list[str]]:
    """Add every item of the tasks to the collection; the identity of each task's supporting items."""
    texts = sorted(
        {d for t in group for d in t["documents"] if isinstance(d, str)}, key=lambda d: d.encode()
    )
    if any(len(text.encode()) > TEXT_LIMIT for text in texts):
        raise ValueError(f"Passages are at most {TEXT_LIMIT} bytes; split longer documents")
    for text, key in zip(texts, embed_texts(preparer.teacher, texts, DOCUMENT_PROMPT), strict=True):
        collection.add_text(text.encode(), key, origin=f"text:{text_item_id(text.encode())[:16]}")
    identities = {}
    for task in group:
        ids = []
        for document in task["documents"]:
            if isinstance(document, str):
                ids.append(text_item_id(document.encode()))
                continue
            record = preparer.media("image", document["image"], target=False)
            collection.add_record(record, preparer.store, origin=str(document["image"]))
            ids.append(record.source.source_family)
        identities[task["id"]] = [ids[i] for i in task["supporting"]]
    return identities


def build(
    tasks: list[dict],
    preparer,
    root: Path,
    codec,
    *,
    validation_fraction: float = 0.05,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict:
    for task in tasks:
        check(task)
    store = TensorAssetStore(root)
    requests = [t.get("request", t["question"]) for t in tasks]
    queries = embed_texts(preparer.teacher, requests, QUERY_PROMPT)
    split_of = {t["id"]: frozen_split(t["id"]) for t in tasks}
    train = [q for t, q in zip(tasks, queries, strict=True) if split_of[t["id"]] == "train"]
    center = np.mean(train if train else queries, axis=0)
    supporting, receipts = {}, {}
    for split in ("train", "validation"):
        group = [t for t in tasks if split_of[t["id"]] == split]
        if not group:
            continue
        path = root / "collections" / split
        collection = SourceCollection.create(path, name=f"{STREAM}-{split}", teacher=PINNED_TEACHER)
        supporting.update(fill(collection, group, preparer))
        collection.reindex()
        collection.set_threshold(threshold)
        collection.set_query_center(center)
        receipts[split] = {"items": collection.count, "identity": collection.identity()}
        collection.close()
    episodes = []
    for task, request, query in zip(tasks, requests, queries, strict=True):
        split = split_of[task["id"]]
        asset = store.put(
            query,
            modality="text",
            kind="embedding",
            extractor=PINNED_TEACHER,
            source_family=hashlib.sha256(request.encode()).hexdigest(),
        )
        annotation = RetrievalAnnotation(
            (f"collections/{split}",), request, asset, tuple(supporting[task["id"]])
        )
        episode_id = f"collection-reading-v1-{hashlib.sha256(task['id'].encode()).hexdigest()[:24]}"
        records = (
            PhysicalRecord(f"{episode_id}-question", "user", task["question"].encode()),
            PhysicalRecord(
                f"{episode_id}-answer",
                "assistant",
                task["answer"].encode(),
                turn_index=1,
                policy_mask=True,
                provenance="natural",
            ),
        )
        episodes.append(
            sequential_meridians(
                Episode(
                    episode_id,
                    records,
                    Lineage(STREAM, "v1", task["id"], frozen_split=split),
                    stream=STREAM,
                    supervision=Supervision(
                        annotations_json=canonical_json({KEY: annotation.to_json()}).decode()
                    ),
                )
            )
        )
    dedup = DedupStore(
        root / "preparation.sqlite",
        DedupConfig(
            train_fraction=1 - validation_fraction, validation_fraction=validation_fraction
        ),
    )
    try:
        with dedup.transaction():
            for episode in episodes:
                dedup.add(episode, codec)
        dedup.freeze()
        export_prepared(
            dedup,
            root,
            codec=codec,
            source_receipts=[
                {
                    "name": STREAM,
                    "tasks": len(tasks),
                    "teacher": PINNED_TEACHER,
                    "collections": receipts,
                }
            ],
            pipeline_fingerprint="collection-reading-v1",
        )
    finally:
        dedup.close()
    return {STREAM: len(tasks), "collections": receipts}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Prepared dataset root")
    parser.add_argument("--codec", required=True)
    parser.add_argument("--packs", required=True, help="Locked pack manifest (EmbeddingGemma2)")
    parser.add_argument("--tasks", type=Path, help="JSONL {id, question, answer, documents, ...}")
    parser.add_argument("--captions", type=Path, help="JSONL {id, image, captions}")
    parser.add_argument("--hotpot", type=Path, help="HotpotQA distractor parquet dir")
    parser.add_argument("--musique", type=Path, help="MuSiQue-Ans JSONL dir")
    parser.add_argument("--count", type=int, default=25000, help="QA tasks to draw")
    parser.add_argument("--seed", type=int, default=20261022)
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if (args.root / "manifest.json").exists():
        parser.error("Prepared manifest already exists; prepared data is immutable")

    def rows(path: Path) -> list[dict]:
        return [json.loads(line) for line in path.read_text("utf-8").split("\n") if line.strip()]

    tasks = []
    if args.tasks is not None:
        tasks += rows(args.tasks)
    if args.captions is not None:
        tasks += caption_tasks(rows(args.captions))
    if args.hotpot is not None or args.musique is not None:
        tasks += qa_tasks(args.hotpot, args.musique, args.count, args.seed)
    if not tasks:
        parser.error("Give --tasks, --captions and/or --hotpot/--musique")
    from megatoken.data.media_prepare import MediaPreparer

    codec = load_codec(args.codec)
    # The scratch root holds the teacher features of images before they are copied
    # into their collection.
    with tempfile.TemporaryDirectory() as scratch:
        preparer = MediaPreparer(args.packs, scratch, device=args.device)
        if preparer.teacher.extractor != PINNED_TEACHER:
            raise ValueError("The pack manifest's teacher is not the pinned teacher")
        report = build(
            tasks, preparer, args.root, codec, validation_fraction=args.validation_fraction
        )
    print(json.dumps(report))


if __name__ == "__main__":
    main()
