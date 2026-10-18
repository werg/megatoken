"""Checksummed JSONL shards and logarithmic-time indexed episode access."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from collections import OrderedDict
from collections.abc import Iterator
from pathlib import Path
from typing import Any, BinaryIO

from megatoken.io import (
    content_hash,
    fsync_directory,
    sha256_file,
    write_json,
)

from .chunking import ChunkingConfig, ChunkingStats
from .dedup import DedupStore
from .schema import Codec, Episode, encode_record

DATASET_VERSION = 2


def export_prepared(
    store: DedupStore,
    output: str | Path,
    *,
    codec: Codec,
    source_receipts: list[dict[str, Any]],
    pipeline_fingerprint: str,
    shard_bytes: int = 128 << 20,
    chunking: ChunkingConfig | None = None,
) -> dict[str, Any]:
    """The manifest is the commit marker; an interrupted export is safely rerun."""
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
        CREATE TABLE shared_prefixes(hash TEXT PRIMARY KEY,tokens INTEGER NOT NULL,occurrences INTEGER NOT NULL);
    """)
    shards: list[dict[str, Any]] = []
    scope_counts: dict[str, int] = {}
    shard_index = 0
    shard_handle: BinaryIO | None = None
    shard_temp: Path | None = None
    shard_name = ""
    shard_size = 0
    shard_count = 0
    shared_prefix_tokens = 0
    shared_prefix_bytes = 0
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
            record_token_counts = []
            for record in episode.records:
                encoded = encode_record(record, codec)
                record_token_counts.append(encoded.tokens)
                chunking_stats.observe(
                    positions=encoded.positions,
                    reference_tokens=encoded.tokens,
                    byte_count=len(record.payload),
                    boundary=record.chunk_boundary,
                    forced=record.forced_chunk_split,
                    soft_max=chunking.soft_max_tokens,
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
            # Prefix identity includes roles, bytes, and their preceding prefix.
            # It reports shared causal context without dropping it from episodes.
            prefix_digest = hashlib.sha256()
            for record, token_count in zip(episode.records, record_token_counts, strict=True):
                if record.policy_mask:
                    break
                prefix_digest.update(record.role.encode() + b"\0")
                prefix_digest.update(len(record.payload).to_bytes(8, "big") + record.payload)
                key = prefix_digest.hexdigest()
                previous = connection.execute(
                    "SELECT tokens FROM shared_prefixes WHERE hash=?", (key,)
                ).fetchone()
                if previous:
                    shared_prefix_tokens += token_count
                    shared_prefix_bytes += len(record.payload)
                connection.execute(
                    "INSERT INTO shared_prefixes VALUES(?,?,1) ON CONFLICT(hash) DO UPDATE SET occurrences=occurrences+1",
                    (key, token_count),
                )
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
        census["duplicate_prefix_bytes"] = shared_prefix_bytes
        census["duplicate_prefix_tokens"] = shared_prefix_tokens
        manifest = {
            "schema_version": DATASET_VERSION,
            "codec_fingerprint": codec.fingerprint,
            "pipeline_fingerprint": pipeline_fingerprint,
            "chunking": chunking.identity(codec),
            "chunking_stats": chunking_stats.to_dict(),
            "sources": source_receipts,
            "dedup": json.loads(store.get_meta("dedup_config") or "{}"),
            "census": census,
            "shards": shards,
            "index": {"path": "index.sqlite", "sha256": sha256_file(root / "index.sqlite")},
            "counts": scope_counts,
            "complete": True,
        }
        manifest["fingerprint"] = content_hash(manifest)
        write_json(root / "manifest.json", manifest)
        return manifest
    except BaseException:
        if shard_handle is not None:
            shard_handle.close()
        connection.close()
        raise


class PreparedDataset:
    """A read-only, process-local reader suitable for rank/worker partitioning."""

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
        self.split, self.stream = split, stream
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
        if verify_checksums and sha256_file(index) != self.manifest["index"]["sha256"]:
            raise ValueError("Prepared episode index checksum mismatch")
        self.connection = sqlite3.connect(f"file:{index}?mode=ro&immutable=1", uri=True)
        self.connection.row_factory = sqlite3.Row
        self._shard_hashes = {item["path"]: item["sha256"] for item in self.manifest["shards"]}
        self._verified: set[str] = set()
        self._handles: OrderedDict[str, BinaryIO] = OrderedDict()

    def _safe_path(self, name: str) -> Path:
        path = (self.root / name).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("A prepared dataset path escapes its root")
        return path

    def __len__(self) -> int:
        return self._length

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
        if self.verify_checksums and name not in self._verified:
            if sha256_file(path) != self._shard_hashes[name]:
                raise ValueError(f"Shard checksum mismatch: {name}")
            self._verified.add(name)
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

    def iter_index(self) -> Iterator[dict[str, Any]]:
        for row in self.connection.execute(
            "SELECT e.*,p.position FROM positions p JOIN episodes e USING(entry_id) WHERE p.scope=? ORDER BY p.position",
            (self.scope,),
        ):
            yield dict(row)

    def iter_episodes(
        self, start: int = 0, rank: int = 0, world_size: int = 1
    ) -> Iterator[Episode]:
        if world_size <= 0 or not 0 <= rank < world_size or start < 0:
            raise ValueError("Invalid rank/world size/cursor")
        first = start + (rank - start) % world_size
        for index in range(first, len(self), world_size):
            yield self[index]

    def verify(self) -> None:
        for name, expected in self._shard_hashes.items():
            if sha256_file(self._safe_path(name)) != expected:
                raise ValueError(f"Shard checksum mismatch: {name}")
            self._verified.add(name)

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()
        self.connection.close()

    def __enter__(self) -> PreparedDataset:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


MappingLike = dict[str, Any] | sqlite3.Row
