"""Split a source manifest into balanced shards that can be prepared on separate machines.

A shard is an ordinary manifest: the same sources, each reading a share of its files (or,
for a source too big to split by file, every n-th row). Shards are prepared independently
and then reconciled (``DedupStore.reconcile``), so a duplicate or a family that straddles
two shards is still found and kept in one split.

The plan depends only on the manifest and on each source's file list at its pinned
revision, so every machine computes the same plan and keeps its own part: no coordinator
and no shared state.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

# Rough source text per row in KiB, by stream; used for sources capped at a row count.
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
DEFAULT_ROW_KIB = 10

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

# Stored bytes expand to roughly this many bytes of row text once decompressed.
EXPANSION = {".parquet": 3.0, ".gz": 4.0, ".zst": 4.0, ".json": 1.0, ".jsonl": 1.0}

# A unit larger than this share of a shard's work is split, so no shard is a single unit.
SPLIT_ABOVE = 0.5


@dataclass(frozen=True)
class Unit:
    """Work a shard takes whole: some of a source's files, or one stripe of all of them."""

    source_id: str
    cost: float  # KiB of row text to prepare
    files: tuple[str, ...] | None = None  # None: the source as its manifest names it
    stripe: tuple[int, int] | None = None


def row_cost(source: Mapping[str, Any]) -> float:
    """KiB of row text to prepare for a source capped at ``max_rows``."""
    rate = PASS_RATE.get(source["source_id"], 1.0)
    return source["max_rows"] * rate * ROW_KIB.get(source["stream"], DEFAULT_ROW_KIB)


def file_cost(name: str, size: int, source: Mapping[str, Any]) -> float:
    suffix = name.rsplit(".", 1)[-1]
    rate = PASS_RATE.get(source["source_id"], 1.0)
    return size / 1024 * EXPANSION.get(f".{suffix}", 1.0) * rate


def source_cost(source: Mapping[str, Any], listing: Sequence[tuple[str, int]] | None) -> float:
    """KiB of row text a whole source costs to prepare."""
    if source.get("max_rows") is not None or not listing:
        return row_cost(source) if source.get("max_rows") is not None else 0.0
    return sum(file_cost(name, size, source) for name, size in listing)


def units_of(
    source: Mapping[str, Any],
    listing: Sequence[tuple[str, int]] | None,
    target: float,
    count: int,
) -> list[Unit]:
    """The parts one source can be divided into for shards of about ``target`` KiB each."""
    if source.get("max_rows") is not None or not listing:
        # A capped source reads its leading rows; those cannot be divided by file.
        if source.get("max_rows") is None:
            raise ValueError(
                f"Source {source['source_id']} is uncapped but has no file listing to split"
            )
        return [Unit(source["source_id"], row_cost(source))]
    costs = [file_cost(name, size, source) for name, size in listing]
    if max(costs) <= SPLIT_ABOVE * target:
        return [
            Unit(source["source_id"], cost, (name,))
            for (name, _), cost in zip(listing, costs, strict=True)
        ]
    # Some file is too big for one shard: every shard reads all files and keeps its stripe.
    total = sum(costs)
    stripes = min(count, max(2, math.ceil(total / (SPLIT_ABOVE * target))))
    names = tuple(name for name, _ in listing)
    return [
        Unit(source["source_id"], total / stripes, names, (index, stripes))
        for index in range(stripes)
    ]


def assign(units: Sequence[Unit], count: int) -> list[list[Unit]]:
    """Longest unit first onto the lightest shard; stripes of one source go to different shards."""
    loads, shards = [0.0] * count, [[] for _ in range(count)]
    for unit in sorted(
        units, key=lambda u: (-u.cost, u.source_id, u.stripe or (0, 0), u.files or ())
    ):
        taken = {
            i for i, shard in enumerate(shards) if any(u.source_id == unit.source_id for u in shard)
        }
        # A source's pieces share one spec per shard unless they are stripes, which may not.
        eligible = [i for i in range(count) if unit.stripe is None or i not in taken] or [
            i for i in range(count)
        ]
        chosen = min(eligible, key=lambda i: (loads[i], i))
        loads[chosen] += unit.cost
        shards[chosen].append(unit)
    return shards


def plan_shards(
    manifest: Mapping[str, Any],
    count: int,
    listings: Mapping[str, Sequence[tuple[str, int]]] | None = None,
    *,
    scale: float = 1.0,
    full: bool = False,
) -> list[dict[str, Any]]:
    """``count`` manifests that together read exactly the sources' rows, once each.

    ``full`` drops every row cap, so each source offers all its rows; ``listings`` then
    maps a source id to its ``(file, stored bytes)`` list. Otherwise caps are multiplied
    by ``scale`` and a capped source goes to one shard whole.
    """
    if count < 1 or scale <= 0:
        raise ValueError("Shard count and scale must be positive")
    sources = [dict(source) for source in manifest["sources"]]
    if full:
        for source in sources:
            source["max_rows"] = None
    else:
        if any(source.get("max_rows") is None for source in sources):
            raise ValueError("Every source needs a max_rows cap before it can be scaled")
        for source in sources:
            source["max_rows"] = max(1, round(source["max_rows"] * scale))
    listings = listings or {}
    by_id = {source["source_id"]: source for source in sources}
    total = sum(source_cost(source, listings.get(source["source_id"])) for source in sources)
    target = total / count if total else 1.0
    units: list[Unit] = []
    for source in sources:
        units.extend(units_of(source, listings.get(source["source_id"]), target, count))
    shards = []
    for part in assign(units, count):
        specs: dict[str, dict[str, Any]] = {}
        files: dict[str, list[str]] = {}
        for unit in part:
            spec = specs.setdefault(unit.source_id, dict(by_id[unit.source_id]))
            if unit.files is not None:
                files.setdefault(unit.source_id, []).extend(unit.files)
            if unit.stripe is not None:
                spec["stripe"] = list(unit.stripe)
        for source_id, names in files.items():
            specs[source_id]["files"] = [{"path": name} for name in names]
        shards.append(
            {
                **manifest,
                "sources": [specs[s["source_id"]] for s in sources if s["source_id"] in specs],
            }
        )
    return shards


def list_source_files(source: Mapping[str, Any]) -> list[tuple[str, int]]:
    """The files (as ``hf://`` URIs at the pinned revision) and stored sizes a source reads.

    Resolved by ``datasets`` itself, so the list is exactly what streaming the source
    would open, and only metadata is fetched.
    """
    from datasets import load_dataset_builder
    from huggingface_hub import HfApi

    repository = source["uri"].removeprefix("hf://datasets/")
    keywords: dict[str, Any] = {
        "path": repository,
        "name": source.get("subset"),
        "revision": source["revision"],
    }
    if source.get("files"):
        keywords["data_files"] = {source["split"]: [item["path"] for item in source["files"]]}
    uris = list(load_dataset_builder(**keywords).config.data_files[source["split"]])
    prefix = f"hf://datasets/{repository}@{source['revision']}/"
    paths = [uri.removeprefix(prefix) for uri in uris]
    info = HfApi().get_paths_info(
        repository, paths, repo_type="dataset", revision=source["revision"]
    )
    sizes = {item.path: item.size for item in info}
    return [(uri, sizes[path]) for uri, path in zip(uris, paths, strict=True)]
