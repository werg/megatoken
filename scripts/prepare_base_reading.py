#!/usr/bin/env python3
"""Build the ``base_reading`` stream: answer from items that are in a knowledge base, not the context.

Each task becomes one episode of two turns: the user's question (a finished question is a
lookup boundary, so a memory lookup falls between question and answer) and the scored
answer. The reference items are not in the episode's records: they are declared
for a base (``knowledge_items``), which the trainer fills with its current ingest heads
and mounts, so the only way to answer is to find them (data/knowledge.py,
training/reading_base.py). Three kinds of task, from the same data the reading and media
streams use:

  documents   a question over a document (HotpotQA / MuSiQue paragraphs or ``--tasks``
              JSONL rows {"id", "question", "documents", "answer"}); the items are the
              document's compressed pieces, chained in reading order
  images      a DOCCI photo; the answer is its description, the question names the
              description's first sentence so the right photo can be found
  speech      a LibriSpeech clip; the answer is its transcript, the question quotes
              its first words

Documents need only the teacher (``--packs``); photos and speech also need ``--inputs`` laid
out as for ``prepare_open_media.py``. Video clips are not drawn yet; ``knowledge_episode``
takes any prepared item.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from megatoken.data import DedupConfig, DedupStore, Episode, Lineage
from megatoken.data.compression import CompressionPolicy, DocumentCompressor
from megatoken.data.knowledge import STREAM, knowledge_episode
from megatoken.data.media_assets import MediaRecord, TensorAssetStore
from megatoken.data.open_media import SOURCES, first_sentence, frozen_split
from megatoken.data.schema import stable_id
from megatoken.data.storage import export_prepared
from megatoken.loading import load_codec
from megatoken.teacher import PINNED_TEACHER

sys.path.insert(0, str(Path(__file__).resolve().parent))


def document_episode(task: Mapping[str, Any], compressor: DocumentCompressor) -> Episode:
    """A question over a document whose compressed pieces are the base's items."""
    document = "\n\n".join(task["documents"]).encode()
    episode_id = stable_id("base-reading-document-v1", task["id"], task["question"], document)
    items = [
        (descriptor, "document", part, f"{task['id']}#{part}")
        for part, (_, _, descriptor) in enumerate(compressor.descriptors(document))
    ]
    family = hashlib.sha256(document).hexdigest()
    return knowledge_episode(
        episode_id, task["question"], task["answer"], items, Lineage(STREAM, "v1", family)
    )


def _media_episode(
    source_id: str,
    clip: Mapping[str, Any],
    record: MediaRecord,
    label: str,
    question: str,
    answer: str,
    cue: str,
) -> Episode:
    family = f"{source_id}:{clip['id']}"
    episode_id = stable_id("base-reading-media-v1", family, question, cue)
    lineage = Lineage(
        source_id,
        SOURCES[source_id].revision,
        family,
        frozen_split=frozen_split(family),
        source_uri=f"hf://datasets/{SOURCES[source_id].repository}",
    )
    return knowledge_episode(
        episode_id, question, answer, [(record, "", 0, label)], lineage, cue=cue
    )


def image_episode(media: Any, clip: Mapping[str, Any]) -> Episode:
    """A DOCCI photo in the base; the question quotes the description's first sentence."""
    record = media.media("image", clip["image"], target=False)
    return _media_episode(
        "docci",
        clip,
        record,
        Path(clip["image"]).name,
        "Describe the photo from the knowledge base that this sentence belongs to.",
        clip["description"],
        first_sentence(clip["description"]),
    )


def speech_episode(media: Any, clip: Mapping[str, Any]) -> Episode:
    """A LibriSpeech clip in the base; the question quotes its first words."""
    record = media.media("audio", clip["audio"], target=False)
    start = " ".join(clip["text"].split()[:5])
    return _media_episode(
        "librispeech_asr",
        clip,
        record,
        Path(clip["audio"]).name,
        "Transcribe the recording from the knowledge base that starts with these words.",
        clip["text"],
        start,
    )


def tasks(args: argparse.Namespace) -> list[dict]:
    rows = []
    if args.tasks is not None:
        rows += [
            json.loads(line)
            for line in args.tasks.read_text(encoding="utf-8").split("\n")
            if line.strip()
        ]
    if args.hotpot is not None or args.musique is not None:
        from prepare_long_document_reading import qa_tasks

        rows += qa_tasks(args.hotpot, args.musique, args.count, args.seed)
    return rows


def episodes(args: argparse.Namespace, root: Path) -> Iterator[Episode]:
    documents = tasks(args)
    if documents:
        from megatoken.omni.embeddinggemma2 import EmbeddingGemma2

        teacher = EmbeddingGemma2.from_manifest(args.packs)
        if teacher.extractor != PINNED_TEACHER:
            raise ValueError("The pack manifest's teacher is not the pinned teacher")
        compressor = DocumentCompressor(
            teacher, TensorAssetStore(root), CompressionPolicy(min_bytes=1)
        )
        for task in documents:
            yield document_episode(task, compressor)
    if args.docci or args.librispeech:
        # The readers this branch called (read_librispeech, and read_docci with its old
        # signature) were replaced by the source/shard readers of data/open_media.py, which
        # this script does not construct. Say so instead of failing on an import.
        raise SystemExit(
            "--docci and --librispeech are not wired to the current open-media readers; "
            "prepare those clips with scripts/prepare_open_media.py"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Prepared dataset root (new)")
    parser.add_argument("--codec", required=True)
    parser.add_argument("--packs", required=True, help="Locked pack manifest (EmbeddingGemma2)")
    parser.add_argument("--tasks", type=Path, help="JSONL {id, question, documents, answer}")
    parser.add_argument("--hotpot", type=Path, help="HotpotQA distractor parquet dir")
    parser.add_argument("--musique", type=Path, help="MuSiQue-Ans JSONL dir")
    parser.add_argument("--count", type=int, default=10000, help="QA tasks to draw")
    parser.add_argument("--inputs", type=Path, help="Open-media inputs, as for prepare_open_media")
    parser.add_argument("--work", type=Path, help="Scratch for unpacked media")
    parser.add_argument("--docci", type=int, default=0, help="Photos to draw")
    parser.add_argument("--librispeech", type=int, default=0, help="Speech clips to draw")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20261022)
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    args = parser.parse_args()
    if (args.root / "manifest.json").exists():
        parser.error("Prepared manifest already exists; prepared data is immutable")
    if (args.docci or args.librispeech) and not (args.inputs and args.work):
        parser.error("Photos and speech need --inputs and --work")
    codec = load_codec(args.codec)
    dedup = DedupStore(
        args.root / "preparation.sqlite",
        DedupConfig(
            train_fraction=1 - args.validation_fraction,
            validation_fraction=args.validation_fraction,
        ),
    )
    count = 0
    try:
        with dedup.transaction():
            for episode in episodes(args, args.root):
                dedup.add(episode, codec)
                count += 1
        if not count:
            parser.error("Give --tasks, --hotpot/--musique, --docci or --librispeech")
        dedup.freeze()
        export_prepared(
            dedup,
            args.root,
            codec=codec,
            source_receipts=[{"name": STREAM, "episodes": count, "teacher": PINNED_TEACHER}],
            pipeline_fingerprint="base-reading-v1",
        )
    finally:
        dedup.close()
    print(json.dumps({STREAM: count}))


if __name__ == "__main__":
    main()
