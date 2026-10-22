"""Items for knowledge bases: prepared records, files, and the reading-from-a-base episodes.

An **item** is one physical record already turned into teacher features by the
existing preparation path: a compressed text piece, an image, a speech clip or a
video clip (a ``MediaRecord``). A knowledge base is written from items (see
``megatoken.kb``). Three places produce them:

- a prepared corpus: every observed (not generated) feature record of an
  episode, plus the items an episode declares for a base instead of its context;
- plain files, prepared on the spot with the same teacher (``prepare_files``);
- the reading-from-a-base stream, whose episodes hold only the questions and
  answers and declare their reference items in ``knowledge_items``.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .compression import CompressionPolicy, DocumentCompressor
from .media_assets import MEDIA_RECORD_TYPE, MediaRecord, TensorAssetStore
from .schema import (
    Episode,
    Lineage,
    PhysicalRecord,
    Supervision,
    canonical_json,
    sequential_meridians,
)

STREAM = "base_reading"
# The first assistant turn of a base-reading episode. A completed assistant turn
# is a lookup boundary, so the lookup that finds the items falls after the
# question and before the answer.
LOOKUP_TURN = "Let me check the knowledge base."

SUFFIXES = {
    "image": {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"},
    "audio": {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".opus"},
    "video": {".mp4", ".mov", ".mkv", ".webm", ".avi"},
}


@dataclass(frozen=True)
class KnowledgeItem:
    """One prepared item and where it is filed."""

    record: MediaRecord
    assets: TensorAssetStore
    metadata: Mapping[
        str, object
    ]  # fields of ``metadata_payload``: filed where such a session starts
    source: str  # a label for the item's origin: file name, document id
    document: str = ""  # parts of one document share this...
    part: int = 0  # ...and are numbered from 0 in reading order

    @property
    def modality(self) -> str:
        """The event modality the item is observed as."""
        return "compressed_text" if self.record.modality == "text" else self.record.modality


def declared_items(episode: Episode) -> list[dict[str, Any]]:
    """The reference items an episode declares for a base instead of its context."""
    rows = json.loads(episode.supervision.annotations_json).get("knowledge_items", [])
    if not isinstance(rows, list):
        raise ValueError("knowledge_items must be a list")
    return rows


def declared_knowledge(
    episode: Episode, assets: TensorAssetStore, metadata: Mapping[str, object]
) -> list[KnowledgeItem]:
    """The items an episode declares for a base (see ``knowledge_episode``)."""
    return [
        KnowledgeItem(
            MediaRecord.from_bytes(row["descriptor"].encode("utf-8")),
            assets,
            metadata,
            f"{episode.episode_id}:{row['source']}",
            f"{episode.episode_id}:{row['document']}" if row["document"] else "",
            int(row["part"]),
        )
        for row in declared_items(episode)
    ]


def episode_items(
    episode: Episode, assets: TensorAssetStore, metadata: Mapping[str, object]
) -> list[KnowledgeItem]:
    """Every item of an episode: its observed feature records and its declared items.

    Generated records (the episode's own targets) are never items. The parts of
    one compressed document share its logical record's identity.
    """
    observed = [
        KnowledgeItem(
            MediaRecord.from_bytes(record.payload),
            assets,
            metadata,
            f"{episode.episode_id}:{record.record_id}",
            f"{episode.episode_id}:{record.logical_id or record.record_id}",
            record.part_index,
        )
        for record in episode.records
        if record.media_type == MEDIA_RECORD_TYPE and not record.policy_mask
    ]
    return observed + declared_knowledge(episode, assets, metadata)


def knowledge_annotation(items: Sequence[tuple[MediaRecord, str, int, str]]) -> str:
    """``annotations_json`` declaring (descriptor, document, part, source) items for a base."""
    return canonical_json(
        {
            "knowledge_items": [
                {
                    "descriptor": record.to_bytes().decode("utf-8"),
                    "document": document,
                    "part": part,
                    "source": source,
                }
                for record, document, part, source in items
            ]
        }
    )


def knowledge_episode(
    episode_id: str,
    question: str,
    answer: str,
    items: Sequence[tuple[MediaRecord, str, int, str]],
    lineage: Lineage,
    *,
    cue: str = "",
) -> Episode:
    """A reading-from-a-base episode: the question, a lookup turn, then the answer.

    The reference items are not in the context: they are declared for a base
    (``knowledge_items``), which the trainer fills and mounts, so the model can
    answer only by finding them. ``cue`` (optional) is added to the question to say
    which item is meant when the question alone does not (a media clip's first words).
    """
    if not items:
        raise ValueError("A reading-from-a-base episode needs at least one item")
    asked = f"{question}\n{cue}" if cue else question
    records = (
        PhysicalRecord(f"{episode_id}-question", "user", asked.encode(), provenance="natural"),
        PhysicalRecord(
            f"{episode_id}-lookup",
            "assistant",
            LOOKUP_TURN.encode(),
            turn_index=1,
            policy_mask=True,
            provenance="natural",
        ),
        PhysicalRecord(
            f"{episode_id}-answer",
            "assistant",
            answer.encode(),
            turn_index=2,
            policy_mask=True,
            provenance="natural",
        ),
    )
    episode = Episode(
        episode_id,
        records,
        lineage,
        stream=STREAM,
        supervision=Supervision(annotations_json=knowledge_annotation(items)),
    )
    return sequential_meridians(episode)


def episode_metadata(episode: Episode) -> dict[str, object]:
    """The metadata prefix fields a session on this episode can know before its first event.

    Declared session tracks only: the modalities later events happen to use are future.
    """
    modalities = sorted({track.default_modality for track in episode.tracks})
    return {"task": episode.stream, "modality": modalities, "version": "v10"}


# --------------------------------------------------------------------- files


class FilePreparer(Protocol):
    """What preparing files needs: ``MediaPreparer`` (the existing media preparation path)."""

    teacher: Any
    store: TensorAssetStore

    def media(self, modality: str, path: str | Path, *, target: bool = True) -> MediaRecord: ...


def file_modality(path: Path) -> str:
    suffix = path.suffix.lower()
    for modality, suffixes in SUFFIXES.items():
        if suffix in suffixes:
            return modality
    return "text"


def prepare_files(
    paths: Iterable[str | Path],
    preparer: FilePreparer,
    *,
    metadata: Mapping[str, object] | None = None,
) -> list[KnowledgeItem]:
    """Turn files into items with the existing media preparation path.

    A text file becomes the consecutive pieces of one compressed document; an
    image, speech or video file becomes one item. Each item is filed under its
    modality and, unless ``metadata`` says otherwise, its file's name as topic.
    """
    items = []
    for raw in paths:
        path = Path(raw)
        modality = file_modality(path)
        filed = {"modality": ["text" if modality == "text" else modality], "topic": path.stem}
        filed.update(metadata or {})
        if modality == "text":
            compressor = DocumentCompressor(
                preparer.teacher, preparer.store, CompressionPolicy(min_bytes=1)
            )
            for part, (_, _, record) in enumerate(compressor.descriptors(path.read_bytes())):
                items.append(
                    KnowledgeItem(record, preparer.store, filed, path.name, str(path), part)
                )
        else:
            record = preparer.media(modality, path, target=False)
            items.append(KnowledgeItem(record, preparer.store, filed, path.name))
    return items
