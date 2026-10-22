"""Split a source manifest into balanced shards for parallel preparation.

    python scripts/shard_sources.py configs/data/sources.agentic.json out.json --shard 2/8
    python scripts/shard_sources.py configs/data/sources.agentic.json out.json --scale 8 --shard 2/8
    python scripts/shard_sources.py configs/data/sources.agentic.json out.json --full --shard 2/64 \
        --listing configs/data/files.agentic.json

The manifest's own row caps are a first real run; ``--scale`` multiplies them and keeps the
leading rows identical. ``--full`` drops every cap, so each source offers all its rows, and
splits the sources by file (or by row stripe, for a source whose files are too big to split)
using each source's file list at its pinned revision. ``--listing`` names a JSON file of
those lists: it is read if it exists and otherwise built from the Hugging Face Hub (metadata
only) and written there, so a fleet of machines can share one committed copy.

Every machine computes the same plan from the same inputs and keeps its own part, so
shards need no coordinator. Prepare each shard into its own root with
``megatoken prepare --ingest-only`` and join them with ``megatoken reconcile``
(docs/data.md, "Preparing at full volume").
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from megatoken.data.sharding import list_source_files, plan_shards


def listings_for(manifest: dict, path: Path | None) -> dict[str, list[tuple[str, int]]]:
    """Each source's ``(file, stored bytes)`` list, from ``path`` when it exists."""
    if path is not None and path.exists():
        stored = json.loads(path.read_text())
        return {key: [tuple(item) for item in value] for key, value in stored.items()}
    sources = manifest["sources"]

    def one(source: dict) -> tuple[str, list[tuple[str, int]]]:
        if source["format"] == "hf":
            return source["source_id"], list_source_files(source)
        return source["source_id"], [
            (item["path"], item.get("size_bytes") or Path(item["path"]).stat().st_size)
            for item in source["files"]
        ]

    with ThreadPoolExecutor(8) as pool:
        found = dict(pool.map(one, sources))
    if path is not None:
        path.write_text(json.dumps(found, indent=1, sort_keys=True))
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--full", action="store_true", help="drop every row cap")
    parser.add_argument("--listing", type=Path, help="JSON file of each source's files and sizes")
    parser.add_argument("--shard", default="0/1", help="i/n: keep the i-th of n balanced shards")
    args = parser.parse_args()
    index, count = (int(part) for part in args.shard.split("/"))
    if not 0 <= index < count:
        raise SystemExit("Shard index must be in [0, count)")
    manifest = json.loads(args.manifest.read_text())
    listings = listings_for(manifest, args.listing) if args.full else None
    shard = plan_shards(manifest, count, listings, scale=args.scale, full=args.full)[index]
    args.output.write_text(json.dumps(shard, indent=1))
    rows = [source["max_rows"] for source in shard["sources"]]
    detail = "all rows" if args.full else f"{sum(rows)} rows at scale {args.scale}"
    print(f"shard {index}/{count}: {len(shard['sources'])} sources, {detail}")


if __name__ == "__main__":
    main()
