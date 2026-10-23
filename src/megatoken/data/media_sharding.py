"""Split the media sources into balanced shards that separate GPU sandboxes prepare.

Text sharding (``sharding.py``) divides the *rows* of a manifest; media is divided the same
way, by the files each source reads: a shard is a set of ``{source_id: {files, stripe,
options}}`` entries that ``scripts/prepare_open_media.py --shard-spec`` reads. Every machine
computes the same plan from the manifests and the committed file listing
(``configs/data/files.media.json``: each source's files at its pinned revision with their
stored sizes), so shards need no coordinator.

The work of a file is estimated in GPU seconds: the clips it holds (the source's inventory,
spread over its files by stored size) times a cost per clip. The per-clip costs are
assumptions, used for balance and for the cost table in docs/data.md until a single shard has
been timed; a wrong constant makes shards unequal, never wrong.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .open_media import SOURCES, TASKS, OpenSource
from .sharding import SPLIT_ABOVE, Unit, assign

# Readers whose sources divide by file (a linked source by its archives, unless one stream).
FILE_READERS = ("parquet", "webdataset", "archive", "linked")

# Estimated GPU seconds per clip by task: the teacher reads each media event once and the codec
# encodes it once, whatever the number of episodes built from it. Images 20 a second, speech 40
# clips a second (10-20 s each), video clips 2 a second, an edit pair is two media events.
CLIP_SECONDS = {
    "captioned_image": 0.05,
    "picture": 0.05,
    "image_questions": 0.05,
    "image_edit": 0.1,
    "image_edit_chain": 0.05,  # per image: a session holds its original and each result
    "transcription": 0.025,
    "translation_text": 0.025,
    "translation_pair": 0.05,
    "verbalization": 0.025,
    "sound": 0.03,
    "captioned_audio": 0.03,
    "audio_questions": 0.03,
    "captioned_video": 0.5,
    "video_questions": 1.0,
    "av_caption": 0.5,
    "av_scenes": 0.5,
    "video": 0.5,
    "video_edit": 1.0,
    "video_edit_chain": 0.5,  # per video, as above
}
DEFAULT_CLIP_SECONDS = 0.1


def clip_seconds(source: OpenSource) -> float:
    seconds = source.options.get(
        "clip_seconds", CLIP_SECONDS.get(source.task, DEFAULT_CLIP_SECONDS)
    )
    if source.task.endswith("_chain"):
        items = source.inventory.get("items") or 1
        seconds *= 1 + source.inventory.get("turns", items) / items
    return seconds


def source_items(source: OpenSource) -> float:
    """The clips a source offers (its inventory; the number is only used for balance)."""
    return float(source.inventory.get("items") or 1)


def list_files(source: OpenSource) -> list[tuple[str, int]]:
    """``(file, stored bytes)`` for the files the source's ``include`` patterns match at its
    pinned revision (Hub metadata only)."""
    from huggingface_hub import HfFileSystem

    filesystem = HfFileSystem()
    prefix = f"datasets/{source.repository}@{source.revision}/"
    found: dict[str, int] = {}
    # A linked source's tables are small and found when it runs; its archives are what divides.
    patterns = source.options["linked"]["archives"] if source.reader == "linked" else source.include
    for pattern in patterns:
        for item in filesystem.glob(prefix + pattern, detail=True).values():
            if item["type"] == "file":
                found[item["name"].removeprefix(prefix)] = int(item["size"])
    return sorted(found.items())


def read_listing(
    sources: Mapping[str, OpenSource], path: str | Path | None
) -> dict[str, list[tuple[str, int]]]:
    """Each file-based source's listing, from ``path`` when it exists, else from the Hub (and
    written to ``path``: commit it so every machine plans from the same lists). A source that
    the file lacks is listed from the Hub and added to the file."""
    stored: dict[str, list[tuple[str, int]]] = {}
    if path is not None and Path(path).exists():
        raw = json.loads(Path(path).read_text())
        stored = {key: [(name, int(size)) for name, size in value] for key, value in raw.items()}
    wanted = {
        source_id: source
        for source_id, source in sources.items()
        if source.reader in FILE_READERS and source_id not in stored
    }
    if wanted:
        found = {source_id: list_files(source) for source_id, source in wanted.items()}
        stored.update(found)
        if path is not None:
            # One source per line: the file is large, and a diff shows which source changed.
            lines = (
                f"{json.dumps(key)}: {json.dumps(stored[key], separators=(',', ':'))}"
                for key in sorted(stored)
            )
            Path(path).write_text("{\n" + ",\n".join(lines) + "\n}\n")
    return stored


def units_of(
    source: OpenSource, listing: Sequence[tuple[str, int]] | None, target: float, count: int
) -> list[Unit]:
    """The parts one source divides into, for shards of about ``target`` GPU seconds each."""
    seconds = source_items(source) * clip_seconds(source)
    if source.reader == "fleurs":
        # One unit per language; the shard reads its languages' files only.
        languages = source.options["languages"]
        return [Unit(source.source_id, seconds / len(languages), (name,)) for name in languages]
    one_stream = source.reader == "linked" and source.options["linked"].get("multipart")
    if not listing or source.reader not in FILE_READERS or one_stream:
        if seconds > SPLIT_ABOVE * target:
            # Whole-source readers (one archive cut in parts, tables of rows): every shard reads
            # it all and keeps its share of the records.
            stripes = min(count, max(2, math.ceil(seconds / (SPLIT_ABOVE * target))))
            names = tuple(name for name, _ in listing) if listing else None
            return [
                Unit(source.source_id, seconds / stripes, names, (i, stripes))
                for i in range(stripes)
            ]
        return [Unit(source.source_id, seconds)]
    total = sum(size for _, size in listing) or 1
    costs = [seconds * max(size, 1) / total for _, size in listing]
    if max(costs) <= SPLIT_ABOVE * target:
        return [
            Unit(source.source_id, cost, (name,))
            for (name, _), cost in zip(listing, costs, strict=True)
        ]
    # A file bigger than half a shard: every shard that takes a stripe opens all the files.
    stripes = min(count, max(2, math.ceil(seconds / (SPLIT_ABOVE * target))))
    names = tuple(name for name, _ in listing)
    return [Unit(source.source_id, seconds / stripes, names, (i, stripes)) for i in range(stripes)]


def plan_media_shards(
    sources: Mapping[str, OpenSource],
    count: int,
    listings: Mapping[str, Sequence[tuple[str, int]]],
) -> list[dict[str, dict[str, Any]]]:
    """``count`` shard specs that together read every source once, balanced by estimated work.

    A spec maps ``source_id`` to ``{"files": [...]}`` (a source read by file), ``{"stripe":
    [i, n], "files": [...]}`` (every n-th clip of all its files), ``{"options": {"languages": [...]}}``
    (FLEURS), or ``{}`` (the whole source).
    """
    if count < 1:
        raise ValueError("Shard count must be positive")
    total = sum(source_items(s) * clip_seconds(s) for s in sources.values())
    target = total / count if total else 1.0
    units = [
        unit
        for source in sources.values()
        for unit in units_of(source, listings.get(source.source_id), target, count)
    ]
    plan = []
    for part in assign(units, count):
        spec: dict[str, dict[str, Any]] = {}
        for unit in part:
            entry = spec.setdefault(unit.source_id, {})
            source = sources[unit.source_id]
            if source.reader == "fleurs":
                entry.setdefault("options", {}).setdefault("languages", []).extend(unit.files or ())
            elif unit.stripe is not None:
                entry["stripe"] = list(unit.stripe)
                if unit.files:
                    entry["files"] = list(unit.files)  # all of them: no search at run time
            elif unit.files is not None:
                entry.setdefault("files", []).extend(unit.files)
        for source_id, entry in spec.items():
            if entry.get("files"):
                # The files' stored sizes travel with the shard, so reading them needs no
                # Hub call to find out.
                sizes = dict(listings.get(source_id, ()))
                entry["sizes"] = {name: sizes[name] for name in entry["files"] if name in sizes}
        plan.append(spec)
    return plan


def shard_seconds(
    sources: Mapping[str, OpenSource],
    spec: Mapping[str, Mapping[str, Any]],
    listings: Mapping[str, Sequence[tuple[str, int]]],
) -> float:
    """Estimated GPU seconds of one shard (the same model the plan balances)."""
    seconds = 0.0
    for source_id, entry in spec.items():
        source = sources[source_id]
        whole = source_items(source) * clip_seconds(source)
        sizes = dict(listings.get(source_id, ()))
        if "stripe" in entry:
            seconds += whole / entry["stripe"][1]
        elif entry.get("files") and sizes:
            seconds += (
                whole * sum(sizes.get(f, 0) for f in entry["files"]) / (sum(sizes.values()) or 1)
            )
        elif source.reader == "fleurs" and "options" in entry:
            seconds += whole * len(entry["options"]["languages"]) / len(source.options["languages"])
        else:
            seconds += whole
    return seconds


def all_sources(include: Sequence[str] | None = None) -> dict[str, OpenSource]:
    """The manifest's sources (all of them, or the named ones), each checked to be buildable."""
    chosen = {source_id: SOURCES[source_id] for source_id in (include if include else SOURCES)}
    for source in chosen.values():
        source.terms.require_approval()
        if source.task not in TASKS:
            raise ValueError(f"{source.source_id}: unknown task {source.task}")
    return chosen
