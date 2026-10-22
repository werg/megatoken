"""Scale a manifest's row caps and split it into balanced shards for parallel preparation.

    python scripts/scale_source_caps.py configs/data/sources.agentic.json out.json \
        --scale 8 --shard 2/8

The manifest's own caps are a first real run. A larger run only raises how many rows are
read from each pinned revision; the leading rows stay the same. Preparation is one
single-threaded process per manifest (about 25 KB of source text per second on long agent
traces), so a run is split across processes: ``--shard i/n`` keeps the sources that the
longest-job-first assignment gives shard ``i``, balanced by estimated text, and each shard is
prepared into its own root.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Rough source text per row in KiB, by stream; only the balance between shards uses it.
ROW_KIB = {
    "executed_trajectories": 120,
    "verified_trajectories": 120,
    "verified_successful_multiturn": 120,
    "mixed_quality_episodes": 120,
    "synthetic_reasoning": 30,
    "tool_use": 15,
    "environment_trajectories": 8,
    "chat": 5,
    "typed_exact_instructions": 3,
    "recovery_clarification": 2,
}


# Share of rows read that pass a source's row filter (measured on the first rows of each); a
# rejected row costs a read but no preparation, so only the passing share counts as work.
PASS_RATE = {
    "swe-agent-nebius": 0.17,
    "swegym-sampled": 0.08,
    "rebench-openhands": 0.48,
    "swe-smith-tool": 0.43,
    "swe-smith-xml": 0.43,
    "swe-smith-ticks": 0.43,
    "orchard-swe-verified": 0.7,
    "s-verified-rebench": 0.48,
    "s-verified-swe-smith": 0.43,
    "s-verified-orchard": 0.7,
}


def cost(source: dict) -> float:
    """Estimated KiB of source text to prepare for one source."""
    rate = PASS_RATE.get(source["source_id"], 1.0)
    return source["max_rows"] * rate * ROW_KIB.get(source["stream"], 10)


def scale(manifest: dict, factor: float) -> dict:
    if factor <= 0:
        raise ValueError("The scale must be positive")
    if any(source.get("max_rows") is None for source in manifest["sources"]):
        raise ValueError("Every source needs a max_rows cap before it can be scaled")
    sources = [
        {**source, "max_rows": max(1, round(source["max_rows"] * factor))}
        for source in manifest["sources"]
    ]
    return {**manifest, "sources": sources}


def shard(manifest: dict, index: int, count: int) -> dict:
    """The sources of one shard, assigned longest job first to the lightest shard."""
    if not 0 <= index < count:
        raise ValueError("Shard index must be in [0, count)")
    loads, chosen = [0.0] * count, [[] for _ in range(count)]
    for source in sorted(manifest["sources"], key=cost, reverse=True):
        lightest = loads.index(min(loads))
        loads[lightest] += cost(source)
        chosen[lightest].append(source["source_id"])
    keep = set(chosen[index])
    return {**manifest, "sources": [s for s in manifest["sources"] if s["source_id"] in keep]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--shard", default="0/1", help="i/n: keep the i-th of n balanced shards")
    args = parser.parse_args()
    index, count = (int(part) for part in args.shard.split("/"))
    result = shard(scale(json.loads(args.manifest.read_text()), args.scale), index, count)
    args.output.write_text(json.dumps(result, indent=1))
    rows = sum(source["max_rows"] for source in result["sources"])
    print(
        f"shard {index}/{count}: {len(result['sources'])} sources, {rows} rows at scale {args.scale}"
    )


if __name__ == "__main__":
    main()
