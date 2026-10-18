"""Checksummed external-memory joins for released variants with missing parents."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from megatoken.io import sha256_file

from .schema import canonical_json


class LineageJoin:
    """Join raw row IDs to original ancestors without changing transcript fields.

    JSONL rows: {"row_id": "...", "family_id": "...", "ancestor_ids": [...],
                 "repository": "owner/repo", "task_id": "..."}.
    Conflicting mappings are rejected. The join is pinned independently from the
    public source; ``seed_source``/variant UUIDs alone are not parent identity.
    """

    def __init__(self, path: str | Path, sha256: str, database: str | Path):
        source = Path(path)
        if not sha256 or sha256_file(source) != sha256:
            raise ValueError("Lineage-join checksum is absent or changed")
        self.connection = sqlite3.connect(database)
        self.connection.executescript(
            "CREATE TABLE IF NOT EXISTS parent_rows(row_id TEXT PRIMARY KEY, mapping TEXT NOT NULL); CREATE TABLE IF NOT EXISTS join_meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);"
        )
        previous = self.connection.execute(
            "SELECT value FROM join_meta WHERE key='sha256'"
        ).fetchone()
        if previous:
            if previous[0] != sha256:
                raise ValueError("A lineage join changed; use a new preparation output")
            return
        # A crash before the fingerprint commit leaves no accepted partial join.
        with self.connection:
            self.connection.execute("DELETE FROM parent_rows")
            with source.open("rb") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    value = json.loads(line)
                    row_id = str(value.pop("row_id"))
                    if not value.get("family_id"):
                        raise ValueError("Every lineage join row requires family_id")
                    if set(value) - {"family_id", "ancestor_ids", "repository", "task_id"}:
                        raise ValueError("A lineage join may contain only lineage fields")
                    encoded = canonical_json(value)
                    old = self.connection.execute(
                        "SELECT mapping FROM parent_rows WHERE row_id=?", (row_id,)
                    ).fetchone()
                    if old and old[0] != encoded:
                        raise ValueError(f"Conflicting source-parent mapping: {row_id}")
                    self.connection.execute(
                        "INSERT OR IGNORE INTO parent_rows VALUES(?,?)", (row_id, encoded)
                    )
            self.connection.execute("INSERT INTO join_meta VALUES('sha256',?)", (sha256,))

    def apply(self, row: Mapping[str, Any], key_path: str) -> dict[str, Any]:
        current: Any = row
        for key in key_path.split("."):
            if isinstance(current, str):
                current = json.loads(current)
            if not isinstance(current, Mapping) or key not in current:
                raise ValueError(f"Missing source row key for lineage join: {key_path}")
            current = current[key]
        result = self.connection.execute(
            "SELECT mapping FROM parent_rows WHERE row_id=?", (str(current),)
        ).fetchone()
        if result is None:
            raise ValueError(f"No original-parent mapping for source row {current}")
        mapping = json.loads(result[0])
        for key, value in mapping.items():
            if key in row and row[key] != value:
                raise ValueError(f"Source and joined lineage conflict at {key}")
        return {**row, **mapping}

    def close(self) -> None:
        self.connection.close()
