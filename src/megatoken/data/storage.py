"""Checksummed JSONL shards and logarithmic-time indexed episode access."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from bisect import bisect_right
from collections import OrderedDict
from collections.abc import Iterator, Mapping, Sequence
from itertools import accumulate
from pathlib import Path
from typing import Any, BinaryIO, NamedTuple

from megatoken.io import (
    content_hash,
    fsync_directory,
    sha256_file,
    write_json,
)

from .chunking import ChunkingConfig, ChunkingStats
from .dedup import DedupStore
from .media_assets import MEDIA, MEDIA_RECORD_TYPE, MediaRecord, TensorAssetStore
from .schema import Codec, Episode

DATASET_VERSION = 3


def _scored_media_outputs(episode: Episode) -> dict[str, int]:
    """Count media targets that are actually selected by training meridians."""
    records = {record.record_id: record for record in episode.records}
    counts: dict[str, int] = {}
    for meridian in episode.meridians:
        for event in meridian.events:
            record = records[event.record_id]
            if not record.policy_mask or event.modality not in MEDIA:
                continue
            if record.media_type != MEDIA_RECORD_TYPE:
                continue
            media = MediaRecord.from_bytes(record.payload)
            if media.modality == event.modality and media.target is not None:
                counts[event.modality] = counts.get(event.modality, 0) + 1
    return counts


def export_prepared(
    store: DedupStore,
    output: str | Path,
    *,
    codec: Codec,
    source_receipts: list[dict[str, Any]],
    pipeline_fingerprint: str,
    shard_bytes: int = 128 << 20,
    chunking: ChunkingConfig | None = None,
    sidecars: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The manifest is the commit marker; an interrupted export is safely rerun.

    ``sidecars`` maps file names to JSON reports written beside the shards
    before the manifest, so a manifest never exists without them.
    """
    if shard_bytes < 1:
        raise ValueError("shard_bytes must be positive")
    chunking = chunking or ChunkingConfig()
    chunking_stats = ChunkingStats()
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    temporary_index = root / "index.building.sqlite"
    temporary_index.unlink(missing_ok=True)
    connection = sqlite3.connect(temporary_index)
    connection.executescript("""
        CREATE TABLE episodes(entry_id INTEGER PRIMARY KEY,episode_id TEXT NOT NULL UNIQUE,
            shard TEXT NOT NULL,offset INTEGER NOT NULL,length INTEGER NOT NULL,sha256 TEXT NOT NULL,
            split TEXT NOT NULL,family_id TEXT NOT NULL,stream TEXT NOT NULL,source_id TEXT NOT NULL,
            tokens INTEGER NOT NULL,scored_positions INTEGER NOT NULL,outcome TEXT NOT NULL,
            verified_success INTEGER NOT NULL);
        CREATE TABLE positions(scope TEXT NOT NULL,position INTEGER NOT NULL,entry_id INTEGER NOT NULL,
            PRIMARY KEY(scope,position));
        CREATE INDEX by_split_stream ON episodes(split,stream,entry_id);
        CREATE INDEX by_source ON episodes(source_id,entry_id);
        CREATE TABLE features(scope TEXT NOT NULL,position INTEGER NOT NULL,source TEXT NOT NULL,
            descriptor BLOB NOT NULL,PRIMARY KEY(scope,position));
        CREATE TABLE feature_sources(split TEXT NOT NULL,source TEXT NOT NULL,PRIMARY KEY(split,source));
    """)
    shards: list[dict[str, Any]] = []
    scope_counts: dict[str, int] = {}
    feature_counts: dict[str, int] = {}
    scored_media_counts: dict[str, int] = {}
    shard_index = 0
    shard_handle: BinaryIO | None = None
    shard_temp: Path | None = None
    shard_name = ""
    shard_size = 0
    shard_count = 0
    transcript_counts = {
        "executed_transcript_U": 0,
        "simulated_transcript_U": 0,
        "verified_success_U": 0,
        "incomplete_transcript_U": 0,
    }

    def close_shard() -> None:
        nonlocal shard_handle, shard_temp, shard_size, shard_count
        if shard_handle is None or shard_temp is None:
            return
        shard_handle.flush()
        os.fsync(shard_handle.fileno())
        shard_handle.close()
        os.replace(shard_temp, root / shard_name)
        shards.append(
            {
                "path": shard_name,
                "sha256": sha256_file(root / shard_name),
                "bytes": shard_size,
                "episodes": shard_count,
            }
        )
        shard_handle = None
        shard_temp = None

    try:
        for entry_id, row in enumerate(store.accepted_rows()):
            episode = Episode.from_dict(json.loads(row["episode_json"]))
            for modality, count in _scored_media_outputs(episode).items():
                key = f"{row['split']}|{row['stream']}|{modality}"
                scored_media_counts[key] = scored_media_counts.get(key, 0) + count
            # Token counts were measured when the episode was admitted.
            shapes = json.loads(row["record_shape"])
            for record, (tokens, positions) in zip(episode.records, shapes, strict=True):
                chunking_stats.observe(
                    positions=positions,
                    reference_tokens=tokens,
                    byte_count=len(record.payload),
                    boundary=record.chunk_boundary,
                    forced=record.forced_chunk_split,
                    soft_max=chunking.soft_max_tokens,
                    languages=record.chunk_languages,
                    diagnostics=record.chunk_diagnostics,
                )
            if any(
                record.provenance == "executed" and record.role in {"tool", "environment"}
                for record in episode.records
            ):
                transcript_counts["executed_transcript_U"] += row["token_count"]
            if any(
                record.provenance == "simulated" and record.role in {"tool", "environment"}
                for record in episode.records
            ):
                transcript_counts["simulated_transcript_U"] += row["token_count"]
            if episode.supervision.verified_success:
                transcript_counts["verified_success_U"] += row["token_count"]
            if episode.supervision.outcome == "incomplete":
                transcript_counts["incomplete_transcript_U"] += row["token_count"]
            line = row["episode_json"].encode("utf-8") + b"\n"
            if shard_handle is None or (shard_size and shard_size + len(line) > shard_bytes):
                close_shard()
                shard_name = f"episodes-{shard_index:06d}.jsonl"
                shard_index += 1
                shard_temp = root / (shard_name + ".partial")
                shard_handle = shard_temp.open("wb")
                shard_size = shard_count = 0
            offset = shard_size
            shard_handle.write(line)
            shard_size += len(line)
            shard_count += 1
            connection.execute(
                "INSERT INTO episodes VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    entry_id,
                    episode.episode_id,
                    shard_name,
                    offset,
                    len(line),
                    hashlib.sha256(line).hexdigest(),
                    row["split"],
                    row["resolved_family"],
                    row["stream"],
                    row["source_id"],
                    row["token_count"],
                    row["scored_positions"],
                    row["outcome"],
                    row["verified_success"],
                ),
            )
            scopes = (
                "*",
                f"split:{row['split']}",
                f"stream:{row['stream']}",
                f"split:{row['split']}|stream:{row['stream']}",
            )
            for scope in scopes:
                position = scope_counts.get(scope, 0)
                connection.execute(
                    "INSERT INTO positions VALUES(?,?,?)", (scope, position, entry_id)
                )
                scope_counts[scope] = position + 1
            # Every distinct prepared feature item (teacher features of media or
            # compressed text) per split, for the encode-and-reconstruct step.
            for record in episode.records:
                if record.media_type != MEDIA_RECORD_TYPE:
                    continue
                descriptor = MediaRecord.from_bytes(record.payload)
                if connection.execute(
                    "INSERT OR IGNORE INTO feature_sources VALUES(?,?)",
                    (row["split"], descriptor.source.sha256),
                ).rowcount:
                    scope = f"split:{row['split']}|modality:{descriptor.modality}"
                    position = feature_counts.get(scope, 0)
                    connection.execute(
                        "INSERT INTO features VALUES(?,?,?,?)",
                        (scope, position, descriptor.source.sha256, record.payload),
                    )
                    feature_counts[scope] = position + 1
            if entry_id % 256 == 0:
                connection.commit()
        close_shard()
        connection.commit()
        connection.execute("VACUUM")
        connection.close()
        with temporary_index.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary_index, root / "index.sqlite")
        fsync_directory(root)
        census = store.census()
        census.update(transcript_counts)
        parser_review = json.loads(store.get_meta("parser_review_summary") or "{}")
        census["parser_review_rows"] = parser_review.get("total", 0)
        with sqlite3.connect(f"file:{root / 'index.sqlite'}?mode=ro", uri=True) as index:
            feature_extractors = sorted(
                {
                    MediaRecord.from_bytes(bytes(row[0])).source.extractor
                    for scope in feature_counts
                    if (
                        row := index.execute(
                            "SELECT descriptor FROM features WHERE scope=? AND position=0",
                            (scope,),
                        ).fetchone()
                    )
                }
            )
        manifest = {
            "schema_version": DATASET_VERSION,
            "codec_fingerprint": codec.fingerprint,
            "pipeline_fingerprint": pipeline_fingerprint,
            "chunking": chunking.identity(codec),
            "chunking_stats": chunking_stats.to_dict(),
            "parser_review": parser_review,
            "sources": source_receipts,
            "dedup": json.loads(store.get_meta("dedup_config") or "{}"),
            "census": census,
            "shards": shards,
            "index": {"path": "index.sqlite", "sha256": sha256_file(root / "index.sqlite")},
            "counts": scope_counts,
            "feature_counts": feature_counts,
            "scored_media_counts": scored_media_counts,
            "feature_extractors": feature_extractors,
            "complete": True,
        }
        manifest["fingerprint"] = content_hash(manifest)
        for name, report in (sidecars or {}).items():
            write_json(root / name, report)
        write_json(root / "manifest.json", manifest)
        return manifest
    except BaseException:
        if shard_handle is not None:
            shard_handle.close()
        connection.close()
        raise


SPLITS = ("train", "validation", "test")

# Files whose checksum this process already verified, so the many readers of one
# root (one per stream and rank) do not each re-hash gigabytes. The size and
# modification time are part of the key: a rewritten file is hashed again.
_VERIFIED_FILES: set[tuple[str, str, int, int]] = set()


def _verify_checksum(path: Path, expected: str, message: str) -> None:
    status = path.stat()
    key = (str(path), expected, status.st_size, status.st_mtime_ns)
    if key in _VERIFIED_FILES:
        return
    if sha256_file(path) != expected:
        raise ValueError(message)
    _VERIFIED_FILES.add(key)


class FeatureItem(NamedTuple):
    """A prepared feature descriptor with the asset store of the root that holds its tensors."""

    record: MediaRecord
    assets: TensorAssetStore


class PreparedDataset:
    """A read-only, process-local reader of one prepared root."""

    def __init__(
        self,
        root: str | Path,
        split: str | None = "train",
        stream: str | None = None,
        verify_checksums: bool = True,
    ):
        self.root = Path(root).resolve()
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        if self.manifest.get("schema_version") != DATASET_VERSION or not self.manifest.get(
            "complete"
        ):
            raise ValueError("Incomplete/unsupported prepared dataset")
        expected_fingerprint = self.manifest.get("fingerprint")
        unsigned = {key: value for key, value in self.manifest.items() if key != "fingerprint"}
        if content_hash(unsigned) != expected_fingerprint:
            raise ValueError("Prepared manifest fingerprint is invalid")
        if split is not None and split not in SPLITS:
            raise ValueError(f"Unknown split {split!r}; choose from {', '.join(SPLITS)}")
        self.split, self.stream = split, stream
        self.assets = TensorAssetStore(self.root)
        self.verify_checksums = verify_checksums
        self.scope = (
            f"split:{split}|stream:{stream}"
            if split and stream
            else f"split:{split}"
            if split
            else f"stream:{stream}"
            if stream
            else "*"
        )
        self._length = self.manifest["counts"].get(self.scope, 0)
        index = self._safe_path(self.manifest["index"]["path"])
        if verify_checksums:
            _verify_checksum(
                index, self.manifest["index"]["sha256"], "Prepared episode index checksum mismatch"
            )
        self.connection = sqlite3.connect(f"file:{index}?mode=ro&immutable=1", uri=True)
        self.connection.row_factory = sqlite3.Row
        self._shard_hashes = {item["path"]: item["sha256"] for item in self.manifest["shards"]}
        self._handles: OrderedDict[str, BinaryIO] = OrderedDict()

    def _safe_path(self, name: str) -> Path:
        path = (self.root / name).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("A prepared dataset path escapes its root")
        return path

    def __len__(self) -> int:
        return self._length

    def feature_count(self, split: str, modality: str) -> int:
        """Distinct prepared feature items of ``modality`` (``text`` = compressed text)."""
        return self.manifest["feature_counts"].get(f"split:{split}|modality:{modality}", 0)

    def scored_media_modalities(self, streams: Sequence[str]) -> set[str]:
        """Media decoder modalities used by selected scored outputs.

        New roots carry this compact summary in their verified manifest. Older
        roots derive it once from their actual episode records and cache the
        result outside the immutable prepared root.
        """
        if self.split is None:
            raise ValueError("Scored media outputs require a selected data split")
        selected = set(streams)
        if "scored_media_counts" in self.manifest:
            summary = self.manifest["scored_media_counts"]
            if not isinstance(summary, dict):
                raise ValueError("Prepared scored-media summary is malformed")
            result = set()
            for key, count in summary.items():
                try:
                    split, remainder = key.split("|", 1)
                    stream, modality = remainder.rsplit("|", 1)
                except (AttributeError, ValueError):
                    raise ValueError("Prepared scored-media summary is malformed") from None
                if (
                    split not in SPLITS
                    or not stream
                    or modality not in MEDIA
                    or type(count) is not int
                    or count <= 0
                ):
                    raise ValueError("Prepared scored-media summary is malformed")
                if (
                    split == self.split
                    and stream in selected
                ):
                    result.add(modality)
            return result
        result = set()
        if not any(self.feature_count(self.split, modality) for modality in MEDIA):
            return result
        for stream in sorted(selected):
            result.update(self._legacy_scored_media_modalities(stream))
        return result

    def _legacy_scored_media_modalities(self, stream: str) -> set[str]:
        cache_name = content_hash(
            {"fingerprint": self.manifest["fingerprint"], "split": self.split, "stream": stream}
        )
        cache_root = Path("/tmp/megatoken-prepared-media-summary")
        cache_root.mkdir(parents=True, exist_ok=True)
        cache_path = cache_root / f"{cache_name}.json"
        lock_path = cache_root / f"{cache_name}.lock"
        import fcntl

        with lock_path.open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                try:
                    cached = json.loads(cache_path.read_text())
                    modalities = cached["modalities"]
                    if (
                        cached.get("fingerprint") == self.manifest["fingerprint"]
                        and cached.get("split") == self.split
                        and cached.get("stream") == stream
                        and isinstance(modalities, list)
                        and set(modalities) <= set(MEDIA)
                    ):
                        return set(modalities)
                except (OSError, ValueError, KeyError, TypeError):
                    pass
                modalities: set[str] = set()
                with PreparedDataset(
                    self.root, self.split, stream, verify_checksums=self.verify_checksums
                ) as scoped:
                    for episode in scoped.iter_episodes():
                        modalities.update(_scored_media_outputs(episode))
                write_json(
                    cache_path,
                    {
                        "fingerprint": self.manifest["fingerprint"],
                        "split": self.split,
                        "stream": stream,
                        "modalities": sorted(modalities),
                    },
                )
                return modalities
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def feature_item(self, split: str, modality: str, position: int) -> FeatureItem:
        row = self.connection.execute(
            "SELECT descriptor FROM features WHERE scope=? AND position=?",
            (f"split:{split}|modality:{modality}", position),
        ).fetchone()
        if row is None:
            raise IndexError(position)
        return FeatureItem(MediaRecord.from_bytes(bytes(row["descriptor"])), self.assets)

    def feature_extractors(self) -> set[str]:
        """Teachers that produced this root's feature tensors (one item per split and modality)."""
        if "feature_extractors" in self.manifest:
            return set(self.manifest["feature_extractors"])
        extractors = set()
        for scope in self.manifest["feature_counts"]:
            row = self.connection.execute(
                "SELECT descriptor FROM features WHERE scope=? AND position=0", (scope,)
            ).fetchone()
            extractors.add(MediaRecord.from_bytes(bytes(row["descriptor"])).source.extractor)
        return extractors

    def index_entry(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        row = self.connection.execute(
            "SELECT e.* FROM positions p JOIN episodes e USING(entry_id) WHERE p.scope=? AND p.position=?",
            (self.scope, index),
        ).fetchone()
        if row is None:
            raise ValueError("Prepared index positions are inconsistent")
        return dict(row)

    def _read(self, row: MappingLike) -> Episode:
        name = row["shard"]
        if name not in self._shard_hashes:
            raise ValueError("Index references an uncommitted shard")
        path = self._safe_path(name)
        if self.verify_checksums:
            _verify_checksum(path, self._shard_hashes[name], f"Shard checksum mismatch: {name}")
        handle = self._handles.pop(name, None)
        if handle is None:
            handle = path.open("rb")
        self._handles[name] = handle
        if len(self._handles) > 8:
            _, old = self._handles.popitem(last=False)
            old.close()
        handle.seek(row["offset"])
        line = handle.read(row["length"])
        if self.verify_checksums and hashlib.sha256(line).hexdigest() != row["sha256"]:
            raise ValueError("Episode payload checksum mismatch")
        episode = Episode.from_dict(json.loads(line))
        if episode.episode_id != row["episode_id"]:
            raise ValueError("Index/payload episode identity mismatch")
        return episode

    def __getitem__(self, index: int) -> Episode:
        return self._read(self.index_entry(index))

    def iter_episodes(
        self, start: int = 0, rank: int = 0, world_size: int = 1
    ) -> Iterator[Episode]:
        if world_size <= 0 or not 0 <= rank < world_size or start < 0:
            raise ValueError("Invalid rank/world size/cursor")
        first = start + (rank - start) % world_size
        for index in range(first, len(self), world_size):
            yield self[index]

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()
        self.connection.close()

    def __enter__(self) -> PreparedDataset:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


class PreparedCorpus:
    """Several prepared roots read as one: per-scope concatenation in a canonical order.

    Each preparation script writes its own immutable root with its own media
    assets, so an episode is always fetched together with the asset store of
    the root it came from. Roots are ordered by manifest fingerprint, never by
    the order or paths they were listed in, so every rank and DiLoCo worker
    (whatever its mount points) sees the same concatenation and shards it
    identically. A scope that spans several roots (the same stream prepared in
    pieces) is one stream whose episodes are drawn in proportion to size.
    Roots must agree on codec, chunking policy and teacher; roots with nothing
    in the scope are dropped from it.
    """

    def __init__(
        self,
        roots: Sequence[str | Path],
        split: str | None = "train",
        stream: str | None = None,
        verify_checksums: bool = True,
    ):
        if not roots:
            raise ValueError("A prepared corpus needs at least one root")
        opened: list[PreparedDataset] = []
        try:
            for root in roots:
                opened.append(PreparedDataset(root, split, stream, verify_checksums))
            opened.sort(key=lambda part: part.manifest["fingerprint"])
            self.extractor = self._validate(opened)
        except BaseException:
            for part in opened:
                part.close()
            raise
        self.split, self.stream = split, stream
        self.scope = opened[0].scope
        self.parts = tuple(part for part in opened if len(part))
        for part in opened:
            if not len(part):
                part.close()
        self.all_roots = tuple(part.root for part in opened)
        self._starts = [0, *accumulate(len(part) for part in self.parts)]
        self.codec_fingerprint = opened[0].manifest["codec_fingerprint"]
        self.chunking = opened[0].manifest["chunking"]
        self.identity = content_hash([part.manifest["fingerprint"] for part in opened])

    @staticmethod
    def _validate(parts: list[PreparedDataset]) -> str | None:
        """Reject incompatible roots; the one teacher behind their features, if any."""
        first = parts[0]
        if len({part.manifest["fingerprint"] for part in parts}) != len(parts):
            raise ValueError("The same prepared dataset is listed twice")
        extractors: dict[str, Path] = {}
        for part in parts:
            for key, what in (("codec_fingerprint", "codec"), ("chunking", "chunking policy")):
                if part.manifest[key] != first.manifest[key]:
                    raise ValueError(
                        f"Prepared datasets {first.root} and {part.root} use different {what}s"
                    )
            for extractor in part.feature_extractors():
                extractors.setdefault(extractor, part.root)
        if len(extractors) > 1:
            raise ValueError(
                "Prepared datasets carry features from different teachers: "
                + ", ".join(f"{name} ({root})" for name, root in sorted(extractors.items()))
            )
        return next(iter(extractors), None)

    def __len__(self) -> int:
        return self._starts[-1]

    def fetch(self, index: int) -> tuple[Episode, TensorAssetStore]:
        """Episode ``index`` of the concatenation and the assets of its own root."""
        if not 0 <= index < len(self):
            raise IndexError(index)
        part = bisect_right(self._starts, index) - 1
        return self.parts[part][index - self._starts[part]], self.parts[part].assets

    def iter_episodes(self) -> Iterator[Episode]:
        for part in self.parts:
            yield from part.iter_episodes()

    def feature_count(self, split: str, modality: str) -> int:
        return sum(part.feature_count(split, modality) for part in self.parts)

    def scored_media_modalities(self, streams: Sequence[str]) -> set[str]:
        """Union selected output modalities across the canonical prepared roots."""
        result: set[str] = set()
        for part in self.parts:
            result.update(part.scored_media_modalities(streams))
        return result

    def feature_item(self, split: str, modality: str, position: int) -> FeatureItem:
        if position < 0:
            raise IndexError(position)
        for part in self.parts:
            count = part.feature_count(split, modality)
            if position < count:
                return part.feature_item(split, modality, position)
            position -= count
        raise IndexError(position)

    @property
    def stores(self) -> tuple[TensorAssetStore, ...]:
        """The asset store of every root, including roots without data in this scope."""
        return tuple(TensorAssetStore(root) for root in self.all_roots)

    def close(self) -> None:
        for part in self.parts:
            part.close()

    def __enter__(self) -> PreparedCorpus:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


MappingLike = dict[str, Any] | sqlite3.Row
