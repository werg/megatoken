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
    "recovery_clarification": 30,
    "long_verified_dependencies": 120,
    "issue_pr_sequences": 40,
    "issue_pr_diffs": 40,
    "code_evolution": 30,
    "connected_repositories": 200,
    "long_natural_code_replay": 200,
    "build_logs": 100,
    "long_context": 700,
    "natural_document_replay": 700,
    "grounded_artifacts": 80,
    "long_context_fim": 40,
    "synthetic_code_rewrite": 12,
    "tested_artifacts": 4,
}

# Sources whose rows are far from their stream's typical size.
SOURCE_ROW_KIB = {
    "when2call-sft": 2,
    "gharchive-threads": 2,
    "pr-traces-10k": 190,
    "adp-go-browse-wa": 48,
    "adp-nnetnav-wa": 170,
    "adp-mind2web": 460,
    "agentrefine-gpt4o": 3,
    "commit-chronicle": 4,
    "commitpackft": 3,
    "swe-fixer-train-110k": 7,
    "pie-perf-slow-to-fast": 1.5,
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
    "ldep-rebench-openhands": 0.2,
    "ldep-scale-swe-v4-pro": 0.4,
    "ldep-coderforge": 0.3,
    "pr-resolve-issue-to-patch": 0.69,
    "pr-resolve-comment-to-fix": 0.31,
    "code-world-traces-affected": 0.33,
    "s-tested-opencodeinstruct": 0.39,
    "repository-v2-long": 0.5,
    "repository-v2-long-replay": 0.5,
    "pg19-books": 0.5,
    "pre-1929-books": 0.5,
}


def cost(source: dict) -> float:
    """Estimated KiB of source text to prepare for one source."""
    rate = PASS_RATE.get(source["source_id"], 1.0)
    row_kib = SOURCE_ROW_KIB.get(source["source_id"], ROW_KIB.get(source["stream"], 10))
    return source["max_rows"] * rate * row_kib


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
