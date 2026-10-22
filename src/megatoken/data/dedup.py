"""Disk-backed deduplication, ancestor union, and frozen family splits.

The full corpus remains in SQLite while ancestor components are resolved. No
training shard is released until the component graph and benchmark quarantine
are final. Near dedup uses bounded bottom-k word-shingle sketches and exact
sketch Jaccard checks; its approximation is declared in the output manifest.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import os
import sqlite3
import struct
from collections.abc import Iterable, Iterator, Sequence
from contextlib import closing, contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .schema import Codec, EncodedRecord, Episode, canonical_json, encode_record


def _utc_timestamp(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("Source dates and evaluation cutoffs must be ISO-8601 strings")
    try:
        instant = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("Source dates and evaluation cutoffs must be ISO-8601 strings") from exc
    if instant.tzinfo is None:
        instant = instant.replace(tzinfo=UTC)
    return instant.astimezone(UTC).isoformat(timespec="microseconds")


@dataclass(frozen=True, slots=True)
class DedupConfig:
    near_threshold: float = 0.94
    shingle_words: int = 5
    sketch_size: int = 256
    anchor_count: int = 8
    max_candidates: int = 4096
    min_near_words: int = 40
    max_descendants_per_parent: int = 5
    train_fraction: float = 0.98
    validation_fraction: float = 0.01
    split_seed: str = "fmt-v13-family-split-v1"
    evaluation_after: str | None = None

    def __post_init__(self) -> None:
        if not 0 < self.near_threshold <= 1:
            raise ValueError("near_threshold must lie in (0, 1]")
        if min(self.shingle_words, self.sketch_size, self.anchor_count, self.max_candidates) <= 0:
            raise ValueError("Sketch parameters must be positive")
        if self.anchor_count > self.sketch_size:
            raise ValueError("Cannot use more anchors than sketch values")
        # Tolerance: 0.9 + 0.1 must be accepted although it exceeds 1.0 in floating point.
        if (
            not 0 <= self.train_fraction <= 1
            or not 0 <= self.validation_fraction
            or self.train_fraction + self.validation_fraction > 1 + 1e-9
        ):
            raise ValueError("Invalid train/validation/test proportions")
        if self.min_near_words < 0 or self.max_descendants_per_parent < 1:
            raise ValueError("Near minimum/descendant cap must be nonnegative/positive")
        if self.evaluation_after is not None:
            object.__setattr__(self, "evaluation_after", _utc_timestamp(self.evaluation_after))


def _word_hashes(episode: Episode, config: DedupConfig) -> tuple[tuple[int, ...], int]:
    """The ``sketch_size`` smallest distinct shingle hashes of the episode, and its word count."""
    hashes: set[int] = set()
    words = 0
    window = config.shingle_words
    # Whitespace/case normalization is retrieval only. The retained bytes never
    # change, and exact matching uses the original role-aware transcript hash.
    for group in episode.iter_logical_records():
        # Do not invent shingles across action/observation boundaries.
        tokens = b"".join(record.payload for record in group).lower().split()
        words += len(tokens)
        for start in range(len(tokens) - window + 1):
            hashes.add(
                int.from_bytes(
                    hashlib.blake2b(
                        b"\0".join(tokens[start : start + window]),
                        digest_size=8,
                        person=b"fmt-shingle-v1",
                    ).digest(),
                    "little",
                )
            )
    return tuple(sorted(heapq.nsmallest(config.sketch_size, hashes))), words


def _pack(values: tuple[int, ...]) -> bytes:
    return struct.pack(f"<{len(values)}Q", *values)


def _unpack(value: bytes) -> set[int]:
    return set(struct.unpack(f"<{len(value) // 8}Q", value))


def _unpack_sorted(value: bytes) -> tuple[int, ...]:
    return struct.unpack(f"<{len(value) // 8}Q", value)


class DedupStore:
    def __init__(self, path: str | Path, config: DedupConfig | None = None):
        config = config or DedupConfig()
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS families(key TEXT PRIMARY KEY, parent TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS splits(family_id TEXT PRIMARY KEY, split TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS benchmarks(key TEXT PRIMARY KEY, kind TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS episodes(
                ordinal INTEGER PRIMARY KEY AUTOINCREMENT, episode_id TEXT NOT NULL UNIQUE,
                content_sha256 TEXT NOT NULL, family_key TEXT NOT NULL, resolved_family TEXT,
                stream TEXT NOT NULL, source_id TEXT NOT NULL, episode_json TEXT NOT NULL,
                sketch BLOB NOT NULL, word_count INTEGER NOT NULL, token_count INTEGER NOT NULL,
                scored_positions INTEGER NOT NULL, assistant_tokens INTEGER NOT NULL,
                observation_tokens INTEGER NOT NULL, executed_tokens INTEGER NOT NULL,
                outcome TEXT NOT NULL, verified_success INTEGER NOT NULL, created_at TEXT,
                duplicate_of TEXT, quarantine_reason TEXT, split TEXT, variant_key TEXT NOT NULL,
                inherited_split TEXT, record_shape TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS episode_content ON episodes(content_sha256);
            CREATE INDEX IF NOT EXISTS episode_family ON episodes(family_key);
            CREATE INDEX IF NOT EXISTS episode_resolved_family ON episodes(resolved_family);
            CREATE TABLE IF NOT EXISTS anchors(anchor TEXT NOT NULL, episode_id TEXT NOT NULL,
                                               PRIMARY KEY(anchor, episode_id));
            CREATE TABLE IF NOT EXISTS source_rows(source_key TEXT NOT NULL, row_index INTEGER NOT NULL,
                status TEXT NOT NULL, detail TEXT, PRIMARY KEY(source_key, row_index));
            CREATE TABLE IF NOT EXISTS source_progress(source_key TEXT PRIMARY KEY, next_row INTEGER NOT NULL,
                                                        complete INTEGER NOT NULL DEFAULT 0);
        """)
        prior = self.get_meta("dedup_config")
        current = canonical_json(asdict(config))
        if prior is not None and prior != current:
            raise ValueError("Dedup/split configuration changed; use a new preparation directory")
        self.set_meta("dedup_config", current)
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> DedupStore:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    @contextmanager
    def transaction(self) -> Iterator[None]:
        with self.connection:
            yield

    @contextmanager
    def row_savepoint(self) -> Iterator[None]:
        """One source row's writes, all or nothing; the caller commits batches of rows.

        Nothing here is durable until ``connection.commit()``. A crash loses the
        uncommitted rows together with their cursor advance, so they are simply
        read again on resume.
        """
        if not self.connection.in_transaction:
            self.connection.execute("BEGIN")
        self.connection.execute("SAVEPOINT row")
        try:
            yield
        except BaseException:
            self.connection.execute("ROLLBACK TO row")
            self.connection.execute("RELEASE row")
            raise
        self.connection.execute("RELEASE row")

    def get_meta(self, key: str) -> str | None:
        row = self.connection.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row[0] if row else None

    def set_meta(self, key: str, value: str) -> None:
        self.connection.execute(
            "INSERT INTO meta VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )

    @property
    def frozen(self) -> bool:
        return self.get_meta("families_frozen") == "true"

    def _find(self, key: str) -> str:
        path = []
        current = key
        while True:
            row = self.connection.execute(
                "SELECT parent FROM families WHERE key=?", (current,)
            ).fetchone()
            if row is None:
                if self.frozen:
                    raise ValueError(f"Unknown ancestor after family splits were frozen: {key}")
                self.connection.execute("INSERT INTO families VALUES(?,?)", (current, current))
                root = current
                break
            if row[0] == current:
                root = current
                break
            path.append(current)
            current = row[0]
        for node in path:
            self.connection.execute("UPDATE families SET parent=? WHERE key=?", (root, node))
        return root

    def register_family(self, keys: Iterable[str]) -> str:
        """Register ancestors before any synthetic descendants are generated."""
        keys = tuple(sorted(set(keys)))
        if not keys:
            raise ValueError("At least one ancestor family is required")
        roots = sorted({self._find(key) for key in keys})
        if self.frozen and len(roots) != 1:
            raise ValueError("Cannot merge separately assigned families after splitting")
        root = roots[0]
        for other in roots[1:]:
            self.connection.execute("UPDATE families SET parent=? WHERE key=?", (root, other))
        return root

    def register_episode_lineage(self, episode: Episode) -> str:
        """Known benchmark ancestry survives later quality rejection of a row."""
        root = self.register_family(episode.lineage.family_keys)
        if (
            episode.supervision.benchmark_tags
            or self.connection.execute(
                "SELECT 1 FROM benchmarks WHERE key=? AND kind='content'", (episode.content_sha256,)
            ).fetchone()
        ):
            self.connection.execute("INSERT OR IGNORE INTO benchmarks VALUES(?, 'family')", (root,))
        return root

    def quarantine_benchmarks(
        self, *, family_keys: Iterable[str] = (), content_sha256: Iterable[str] = ()
    ) -> None:
        if self.frozen:
            raise ValueError("Register the benchmark snapshot before freezing families")
        for key in family_keys:
            self.connection.execute("INSERT OR IGNORE INTO benchmarks VALUES(?, 'family')", (key,))
            self._find(key)
        for digest in content_sha256:
            self.connection.execute(
                "INSERT OR IGNORE INTO benchmarks VALUES(?, 'content')", (digest,)
            )
        self.connection.commit()

    def source_cursor(self, key: str) -> tuple[int, bool]:
        row = self.connection.execute(
            "SELECT next_row,complete FROM source_progress WHERE source_key=?", (key,)
        ).fetchone()
        return (row[0], bool(row[1])) if row else (0, False)

    def mark_source(
        self, key: str, row: int, *, status: str, detail: str = "", complete: bool = False
    ) -> None:
        self.connection.execute(
            "INSERT OR IGNORE INTO source_rows VALUES(?,?,?,?)", (key, row, status, detail[:2048])
        )
        self.connection.execute(
            "INSERT INTO source_progress VALUES(?,?,?) ON CONFLICT(source_key) DO UPDATE SET next_row=excluded.next_row,complete=excluded.complete",
            (key, row + 1, int(complete)),
        )

    def _anchors(self, sketch: tuple[int, ...]) -> list[str]:
        """The smallest sketch values that still discriminate between documents.

        A value that more than ``max_candidates`` earlier documents already
        carry (a licence header, a system prompt) says nothing about which of
        them is a near duplicate, so it is skipped and the next-smallest value
        takes its place.
        """
        anchors: list[str] = []
        for value in sketch:
            anchor = f"{value:016x}"
            posting = self.connection.execute(
                "SELECT COUNT(*) FROM (SELECT 1 FROM anchors WHERE anchor=? LIMIT ?)",
                (anchor, self.config.max_candidates + 1),
            ).fetchone()[0]
            if posting <= self.config.max_candidates:
                anchors.append(anchor)
                if len(anchors) == self.config.anchor_count:
                    break
        return anchors

    def add(
        self, episode: Episode, codec: Codec, encoded: Sequence[EncodedRecord] | None = None
    ) -> str:
        """Register one episode; a caller that already encoded it passes ``encoded``."""
        if self.frozen:
            raise ValueError(
                "Base ingestion is closed after splits freeze; export descendants to a new preparation run with inherited frozen splits"
            )
        episode_json = canonical_json(episode.to_dict())
        prior_episode = self.connection.execute(
            "SELECT episode_json FROM episodes WHERE episode_id=?", (episode.episode_id,)
        ).fetchone()
        if prior_episode:
            if prior_episode[0] != episode_json:
                raise ValueError(
                    "An immutable episode ID was reused with changed bytes, masks, or metadata"
                )
            return "already_present"
        if encoded is None:
            encoded = tuple(encode_record(record, codec) for record in episode.records)
        root = self.register_episode_lineage(episode)
        sketch, word_count = _word_hashes(episode, self.config)
        variant = (
            f"task:{episode.lineage.task_id}"
            if episode.lineage.task_id
            else episode.lineage.family_id
        )
        if variant.startswith("repository:"):
            variant = episode.lineage.raw_sha256 or episode.episode_id
        return self._admit(
            episode_id=episode.episode_id,
            content_sha256=episode.content_sha256,
            root=root,
            stream=episode.stream,
            source_id=episode.lineage.source_id,
            episode_json=episode_json,
            sketch=sketch,
            word_count=word_count,
            token_count=sum(record.tokens for record in encoded),
            scored_positions=sum(record.P for record in encoded),
            assistant_tokens=sum(
                record.tokens
                for record in encoded
                if record.record.role == "assistant" and record.record.policy_mask
            ),
            observation_tokens=sum(
                record.tokens for record in encoded if record.record.role in {"tool", "environment"}
            ),
            executed_tokens=sum(
                record.tokens for record in encoded if record.record.provenance == "executed"
            ),
            outcome=episode.supervision.outcome,
            verified_success=int(episode.supervision.verified_success),
            created_at=(
                _utc_timestamp(episode.lineage.created_at) if episode.lineage.created_at else None
            ),
            variant_key=variant,
            inherited_split=episode.lineage.frozen_split,
            record_shape=json.dumps([[record.tokens, record.positions] for record in encoded]),
            tagged=bool(episode.supervision.benchmark_tags),
        )

    def _admit(self, *, root: str, sketch: tuple[int, ...], tagged: bool, **row: Any) -> str:
        """Compare a measured episode with those already held, then record it.

        Exact duplicates (same transcript) and near duplicates (sketch similarity) join
        the earlier episode's family and are kept only as duplicates; ``tagged`` marks an
        episode known to come from a benchmark.
        """
        word_count = row["word_count"]
        duplicate = self.connection.execute(
            "SELECT episode_id,family_key FROM episodes WHERE content_sha256=? ORDER BY ordinal LIMIT 1",
            (row["content_sha256"],),
        ).fetchone()
        duplicate_of = duplicate[0] if duplicate else None
        anchors = [] if duplicate else self._anchors(sketch)
        if duplicate:
            root = self.register_family((root, duplicate[1]))
        elif len(sketch) >= self.config.shingle_words and word_count >= self.config.min_near_words:
            placeholders = ",".join("?" for _ in anchors)
            # A near duplicate shares most of its anchors, so the candidates
            # sharing the most of them are compared; the rest cannot beat them.
            candidates = self.connection.execute(
                f"SELECT episode_id FROM anchors WHERE anchor IN ({placeholders}) "
                "GROUP BY episode_id ORDER BY COUNT(*) DESC, episode_id LIMIT ?",
                (*anchors, self.config.max_candidates),
            ).fetchall()
            own = set(sketch)
            for candidate in candidates:
                other = self.connection.execute(
                    "SELECT sketch,family_key,word_count FROM episodes WHERE episode_id=?",
                    (candidate[0],),
                ).fetchone()
                if (
                    min(word_count, other[2]) / max(word_count, other[2], 1)
                    < self.config.near_threshold
                ):
                    continue
                values = _unpack(other[0])
                similarity = len(own & values) / max(1, len(own | values))
                if similarity >= self.config.near_threshold:
                    root = self.register_family((root, other[1]))
                    if duplicate_of is None:
                        duplicate_of = candidate[0]
        reason = (
            "benchmark_contamination"
            if tagged
            or self.connection.execute(
                "SELECT 1 FROM benchmarks WHERE key=? AND kind='content'", (row["content_sha256"],)
            ).fetchone()
            else None
        )
        self.connection.execute(
            """INSERT INTO episodes(
            episode_id,content_sha256,family_key,stream,source_id,episode_json,sketch,word_count,
            token_count,scored_positions,assistant_tokens,observation_tokens,executed_tokens,
            outcome,verified_success,created_at,duplicate_of,quarantine_reason,variant_key,inherited_split,
            record_shape)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                row["episode_id"],
                row["content_sha256"],
                root,
                row["stream"],
                row["source_id"],
                row["episode_json"],
                _pack(sketch),
                word_count,
                row["token_count"],
                row["scored_positions"],
                row["assistant_tokens"],
                row["observation_tokens"],
                row["executed_tokens"],
                row["outcome"],
                row["verified_success"],
                row["created_at"],
                duplicate_of,
                reason,
                row["variant_key"],
                row["inherited_split"],
                row["record_shape"],
            ),
        )
        if not duplicate_of:
            self.connection.executemany(
                "INSERT OR IGNORE INTO anchors VALUES(?,?)",
                ((anchor, row["episode_id"]) for anchor in anchors),
            )
        return "quarantine" if reason else "duplicate" if duplicate_of else "accepted_candidate"

    def write_digest(self, path: str | Path) -> int:
        """Write what ``reconcile`` needs to compare this store with the other shards' stores.

        Every episode this store holds as an original (not a duplicate of another one it
        holds) with its sketch and family, the family links, and the benchmark keys; the
        episode text stays here. Returns the number of episodes written.
        """
        self.connection.commit()
        target = Path(path)
        partial = target.with_suffix(".partial")
        partial.unlink(missing_ok=True)
        self.connection.execute("ATTACH DATABASE ? AS digest", (str(partial),))
        try:
            columns = (
                "ordinal,episode_id,content_sha256,family_key,stream,source_id,sketch,word_count,"
                "token_count,scored_positions,assistant_tokens,observation_tokens,executed_tokens,"
                "outcome,verified_success,created_at,quarantine_reason,variant_key,inherited_split,"
                "record_shape"
            )
            self.connection.execute(
                f"CREATE TABLE digest.episodes AS SELECT {columns} FROM main.episodes "
                "WHERE duplicate_of IS NULL ORDER BY ordinal"
            )
            self.connection.execute(
                "CREATE TABLE digest.families AS SELECT key,parent FROM main.families WHERE key<>parent"
            )
            self.connection.execute(
                "CREATE TABLE digest.benchmarks AS SELECT * FROM main.benchmarks"
            )
            self.connection.execute(
                "CREATE TABLE digest.meta AS SELECT * FROM main.meta WHERE key='dedup_config'"
            )
            count = self.connection.execute("SELECT COUNT(*) FROM digest.episodes").fetchone()[0]
            self.connection.commit()
        finally:
            self.connection.execute("DETACH DATABASE digest")
        with partial.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(partial, target)
        return count

    def apply_decisions(self, path: str | Path) -> None:
        """Adopt ``reconcile``'s verdicts for this shard instead of freezing it alone.

        Each original episode takes its global duplicate status, quarantine reason, split
        and family; episodes that were duplicates inside this shard stay excluded.
        """
        if self.frozen:
            raise ValueError("Verdicts must be applied before the families freeze")
        self.connection.commit()
        self.connection.execute("ATTACH DATABASE ? AS verdicts", (str(path),))
        try:
            missing = self.connection.execute(
                "SELECT COUNT(*) FROM main.episodes WHERE duplicate_of IS NULL AND episode_id NOT IN "
                "(SELECT episode_id FROM verdicts.decisions)"
            ).fetchone()[0]
            if missing:
                raise ValueError(
                    f"{missing} episodes have no verdict; they were not part of the reconciliation"
                )
            self.connection.execute(
                "UPDATE main.episodes SET duplicate_of=d.duplicate_of, "
                "quarantine_reason=d.quarantine_reason, split=d.split, "
                "resolved_family=d.resolved_family FROM verdicts.decisions AS d "
                "WHERE main.episodes.episode_id=d.episode_id"
            )
            self.set_meta("families_frozen", "true")
            self.connection.commit()
        finally:
            self.connection.execute("DETACH DATABASE verdicts")

    def _absorb(self, shard: int, row: sqlite3.Row) -> None:
        """Admit one original episode of another shard's digest (see ``reconcile``)."""
        tagged = row["quarantine_reason"] == "benchmark_contamination"
        root = self.register_family((row["family_key"],))
        if (
            tagged
            or self.connection.execute(
                "SELECT 1 FROM benchmarks WHERE key=? AND kind='content'", (row["content_sha256"],)
            ).fetchone()
        ):
            self.connection.execute("INSERT OR IGNORE INTO benchmarks VALUES(?, 'family')", (root,))
        name = f"{shard}:{row['episode_id']}"
        self._admit(
            episode_id=name,
            root=root,
            sketch=tuple(_unpack_sorted(row["sketch"])),
            tagged=tagged,
            episode_json="",
            **{
                key: row[key]
                for key in (
                    "content_sha256",
                    "stream",
                    "source_id",
                    "word_count",
                    "token_count",
                    "scored_positions",
                    "assistant_tokens",
                    "observation_tokens",
                    "executed_tokens",
                    "outcome",
                    "verified_success",
                    "created_at",
                    "variant_key",
                    "inherited_split",
                    "record_shape",
                )
            },
        )
        self.connection.execute(
            "INSERT INTO origins VALUES(?,?,?)", (name, shard, row["episode_id"])
        )

    def freeze(self) -> None:
        """Resolve complete ancestor components, then assign and propagate splits."""
        if self.frozen:
            return
        with self.connection:
            benchmark_roots = {
                self._find(row[0])
                for row in self.connection.execute("SELECT key FROM benchmarks WHERE kind='family'")
            }
            for row in self.connection.execute(
                "SELECT ordinal,family_key FROM episodes ORDER BY ordinal"
            ):
                root = self._find(row[1])
                self.connection.execute(
                    "UPDATE episodes SET resolved_family=? WHERE ordinal=?", (root, row[0])
                )
            benchmark_roots.update(
                row[0]
                for row in self.connection.execute(
                    "SELECT DISTINCT resolved_family FROM episodes WHERE quarantine_reason='benchmark_contamination'"
                )
            )
            timed_roots: set[str] = set()
            if self.config.evaluation_after:
                # Both source timestamps and cutoffs were normalized to UTC.
                timed_roots = {
                    row[0]
                    for row in self.connection.execute(
                        "SELECT DISTINCT resolved_family FROM episodes WHERE created_at >= ?",
                        (self.config.evaluation_after,),
                    )
                }
            for row in self.connection.execute(
                "SELECT key FROM families WHERE parent=key ORDER BY key"
            ):
                root = row[0]
                inherited = {
                    item[0]
                    for item in self.connection.execute(
                        "SELECT DISTINCT inherited_split FROM episodes WHERE resolved_family=? AND inherited_split IS NOT NULL",
                        (root,),
                    )
                }
                if root in benchmark_roots:
                    split = "quarantine"
                elif len(inherited) > 1:
                    # New cross-source joins cannot move a previously held-out
                    # descendant into training. Quarantine the entire conflict.
                    split = "quarantine"
                elif inherited:
                    split = next(iter(inherited))
                elif root in timed_roots:
                    split = "test"
                else:
                    value = (
                        int.from_bytes(
                            hashlib.sha256(f"{self.config.split_seed}\0{root}".encode()).digest()[
                                :8
                            ],
                            "big",
                        )
                        / 2**64
                    )
                    split = (
                        "train"
                        if value < self.config.train_fraction
                        else "validation"
                        if value < self.config.train_fraction + self.config.validation_fraction
                        else "test"
                    )
                self.connection.execute("INSERT OR REPLACE INTO splits VALUES(?,?)", (root, split))
                self.connection.execute(
                    "UPDATE episodes SET split=? WHERE resolved_family=?", (split, root)
                )
                if split == "quarantine":
                    self.connection.execute(
                        "UPDATE episodes SET quarantine_reason='benchmark_family' WHERE resolved_family=?",
                        (root,),
                    )
            # SQL window functions avoid keeping all variant/task counts in RAM.
            self.connection.execute(
                """UPDATE episodes SET quarantine_reason='parent_variant_cap'
                WHERE ordinal IN (SELECT ordinal FROM (
                    SELECT ordinal, ROW_NUMBER() OVER (PARTITION BY variant_key ORDER BY ordinal) AS n
                    FROM episodes WHERE duplicate_of IS NULL AND quarantine_reason IS NULL
                ) WHERE n > ?)""",
                (self.config.max_descendants_per_parent,),
            )
            self.set_meta("families_frozen", "true")

    def split_of(self, family_key: str) -> str:
        if not self.frozen:
            raise ValueError("Freeze source/parent families before creating descendants")
        root = self._find(family_key)
        row = self.connection.execute(
            "SELECT split FROM splits WHERE family_id=?", (root,)
        ).fetchone()
        if row is None:
            raise ValueError("Family has no frozen split")
        return row[0]

    def accepted_rows(self) -> Iterator[sqlite3.Row]:
        if not self.frozen:
            raise ValueError("Family splits must be frozen before shards are written")
        yield from self.connection.execute(
            "SELECT * FROM episodes WHERE duplicate_of IS NULL AND quarantine_reason IS NULL ORDER BY ordinal"
        )

    def census(self) -> dict[str, Any]:
        rows = self.connection.execute("""SELECT stream,source_id,split,COUNT(*) AS episodes,
            SUM(token_count) AS U,SUM(scored_positions) AS free_position_candidates,
            SUM(assistant_tokens) AS eligible_assistant_tokens,SUM(observation_tokens) AS tool_observation_tokens,
            SUM(executed_tokens) AS executed_observation_tokens,COUNT(DISTINCT resolved_family) AS families
            FROM episodes WHERE duplicate_of IS NULL AND quarantine_reason IS NULL
            GROUP BY stream,source_id,split ORDER BY stream,source_id,split""").fetchall()
        accepted = [dict(row) for row in rows]
        total = sum(row["U"] for row in accepted)
        return {
            "U": total,
            "free_position_candidates": sum(row["free_position_candidates"] for row in accepted),
            "accepted_episodes": sum(row["episodes"] for row in accepted),
            "independent_families": self.connection.execute(
                "SELECT COUNT(DISTINCT resolved_family) FROM episodes WHERE duplicate_of IS NULL AND quarantine_reason IS NULL"
            ).fetchone()[0],
            "duplicates": self.connection.execute(
                "SELECT COUNT(*) FROM episodes WHERE duplicate_of IS NOT NULL"
            ).fetchone()[0],
            "quarantined": self.connection.execute(
                "SELECT COUNT(*) FROM episodes WHERE quarantine_reason IS NOT NULL"
            ).fetchone()[0],
            "rejected_rows": self.connection.execute(
                "SELECT COUNT(*) FROM source_rows WHERE status='rejected'"
            ).fetchone()[0],
            "filtered_rows": self.connection.execute(
                "SELECT COUNT(*) FROM source_rows WHERE status='filtered'"
            ).fetchone()[0],
            "strata": accepted,
            "U_definition": "Raw-codec tokens in accepted deduplicated complete transcripts, including shared context.",
            "free_position_candidates_definition": "The static masked target inventory: positions that could carry loss.",
        }


# Reconciliation commits after this many replayed episodes.
COMMIT_EVERY = 4096


def reconcile(digests: Sequence[str | Path], output: str | Path) -> dict[str, Any]:
    """Join separately ingested shards into one set of verdicts, as if one store had read them all.

    Each shard has already removed its own duplicates. This replays every shard's
    originals, shard by shard in the order given, through the same exact-duplicate and
    near-duplicate comparison, joins families that share a key or a near duplicate across
    shards, and freezes splits, benchmark quarantine and per-parent variant caps once over
    the union. Shard ``i`` gets ``decisions-<i>.sqlite`` in ``output`` for
    ``DedupStore.apply_decisions``. Only digests are read: no episode text moves.
    """
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    paths = [Path(item) for item in digests]
    if not paths:
        raise ValueError("Reconciliation needs at least one shard digest")
    configs = set()
    for path in paths:
        with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as digest:
            configs.add(
                digest.execute("SELECT value FROM meta WHERE key='dedup_config'").fetchone()[0]
            )
    if len(configs) != 1:
        raise ValueError("Shards were prepared with different dedup/split configurations")
    config = DedupConfig(**json.loads(configs.pop()))
    joined = root / "reconciled.sqlite"
    for suffix in ("", "-wal", "-shm"):
        Path(f"{joined}{suffix}").unlink(missing_ok=True)
    columns = (
        "episode_id,content_sha256,family_key,stream,source_id,sketch,word_count,token_count,"
        "scored_positions,assistant_tokens,observation_tokens,executed_tokens,outcome,"
        "verified_success,created_at,quarantine_reason,variant_key,inherited_split,record_shape"
    )
    with DedupStore(joined, config) as store:
        store.connection.execute(
            "CREATE TABLE origins(episode_id TEXT PRIMARY KEY, shard INTEGER NOT NULL, local_id TEXT NOT NULL)"
        )
        for shard, path in enumerate(paths):
            with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as digest:
                digest.row_factory = sqlite3.Row
                store.connection.executemany(
                    "INSERT OR IGNORE INTO benchmarks VALUES(?,?)",
                    digest.execute("SELECT key,kind FROM benchmarks"),
                )
                for key, parent in digest.execute("SELECT key,parent FROM families"):
                    store.register_family((key, parent))
                for count, row in enumerate(
                    digest.execute(f"SELECT {columns} FROM episodes ORDER BY ordinal")
                ):
                    store._absorb(shard, row)
                    if count % COMMIT_EVERY == 0:
                        store.connection.commit()
            store.connection.commit()
        store.freeze()
        counts = {
            "shards": len(paths),
            "episodes": store.connection.execute("SELECT COUNT(*) FROM episodes").fetchone()[0],
            "cross_shard_duplicates": store.connection.execute(
                "SELECT COUNT(*) FROM episodes WHERE duplicate_of IS NOT NULL"
            ).fetchone()[0],
            "quarantined": store.connection.execute(
                "SELECT COUNT(*) FROM episodes WHERE quarantine_reason IS NOT NULL"
            ).fetchone()[0],
            "accepted": store.connection.execute(
                "SELECT COUNT(*) FROM episodes WHERE duplicate_of IS NULL AND quarantine_reason IS NULL"
            ).fetchone()[0],
        }
        for shard in range(len(paths)):
            target = root / f"decisions-{shard:04d}.sqlite"
            target.unlink(missing_ok=True)
            store.connection.execute("ATTACH DATABASE ? AS verdicts", (str(target),))
            try:
                store.connection.execute(
                    "CREATE TABLE verdicts.decisions(episode_id TEXT PRIMARY KEY, duplicate_of TEXT, "
                    "quarantine_reason TEXT, split TEXT, resolved_family TEXT)"
                )
                store.connection.execute(
                    "INSERT INTO verdicts.decisions SELECT o.local_id,e.duplicate_of,"
                    "e.quarantine_reason,e.split,e.resolved_family FROM main.episodes e "
                    "JOIN main.origins o USING(episode_id) WHERE o.shard=?",
                    (shard,),
                )
                store.connection.commit()
            finally:
                store.connection.execute("DETACH DATABASE verdicts")
    return counts
