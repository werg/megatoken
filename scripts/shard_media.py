"""List the files of the media sources and split them into balanced shards.

    python scripts/shard_media.py --listing configs/data/files.media.json
    python scripts/shard_media.py --listing configs/data/files.media.json --shards 64
    python scripts/shard_media.py --listing configs/data/files.media.json --shards 64 --shard 3 --output s.json

``--listing`` names a JSON file of each file-based source's files at its pinned revision with
their stored sizes (Hub metadata only): it is read if it exists and otherwise built and written
there, so a fleet of sandboxes can share one committed copy and plan identical shards.
``--shards N`` prints the estimated GPU hours of every shard (``data/media_sharding.py``;
the per-clip costs are assumptions until a shard is timed); with ``--shard I --output FILE`` it
writes the spec that ``scripts/prepare_open_media.py --shard-spec`` reads. ``--sources`` limits
the plan to some source ids.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from megatoken.data.media_sharding import (
    all_sources,
    plan_media_shards,
    read_listing,
    shard_seconds,
)


def main() -> None:
    from megatoken.hub import install

    install()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--listing", type=Path, required=True)
    parser.add_argument("--shards", type=int)
    parser.add_argument("--shard", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--sources", nargs="*", help="source ids (default: every media source)")
    args = parser.parse_args()
    sources = all_sources(args.sources)
    listings = read_listing(sources, args.listing)
    if not args.shards:
        files = sum(len(value) for value in listings.values())
        print(f"{len(listings)} sources, {files} files listed in {args.listing}")
        return
    plan = plan_media_shards(sources, args.shards, listings)
    hours = [shard_seconds(sources, spec, listings) / 3600 for spec in plan]
    print(
        f"{args.shards} shards: estimated GPU hours min {min(hours):.2f}, "
        f"max {max(hours):.2f}, total {sum(hours):.0f}"
    )
    if args.shard is not None:
        if not args.output:
            parser.error("--shard needs --output")
        args.output.write_text(json.dumps(plan[args.shard], indent=1))
        print(f"shard {args.shard}: {len(plan[args.shard])} sources -> {args.output}")


if __name__ == "__main__":
    main()
