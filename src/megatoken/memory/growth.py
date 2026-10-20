"""Deterministic query-density allocation and incremental source-window pair admission."""
from __future__ import annotations

import math
from dataclasses import dataclass
from hashlib import sha256

import numpy as np

from .contracts import Collection, Footprint, Visibility
from .sqlite import SQLiteKB


@dataclass(frozen=True)
class CatalogChunk:
    content_id: str
    key: np.ndarray
    scope: str
    region: str
    footprint: Footprint
    family: str
    group: str = ""

    def __post_init__(self):
        if not self.content_id or not self.region or not self.family:
            raise ValueError("Allocation requires stable content/region/family")


@dataclass(frozen=True)
class GrowthPolicy:
    value_dim: int
    seed: int
    dense_fraction: float = 0.75
    max_points: int = 100000
    max_pairs: int = 100000
    min_pair_windows: int = 2
    max_neighbors: int = 16
    partition_id: str = "catalog-v1"

    def __post_init__(self):
        if not 0 < self.dense_fraction <= 1 or min(
            self.value_dim, self.max_points, self.max_neighbors, self.min_pair_windows
        ) < 1 or self.max_pairs < 0:
            raise ValueError("Invalid bounded KB growth policy")


class QueryDemand:
    def __init__(self, store: SQLiteKB, policy: GrowthPolicy):
        self.store, self.policy = store, policy

    def observe(self, *, event_id: str, region: str, family: str, step: int) -> bool:
        if not event_id or not region or not family or step < 0:
            raise ValueError("Demand requires deduplicated causal query identity")
        return self.store.record_demand(event_id, self.policy.partition_id, region,
                                        family, step)

    def counts(self, *, after_step: int = 0) -> dict[str, int]:
        rows = self.store.db.execute(
            """SELECT region, COUNT(DISTINCT family) AS n FROM demand_events
               WHERE partition_id=? AND step>=? GROUP BY region""",
            (self.policy.partition_id, after_step),
        ).fetchall()
        return {row[0]: int(row[1]) for row in rows}


def _seed(seed: int, identity: str) -> int:
    return int.from_bytes(sha256(f"{seed}/{identity}".encode()).digest()[:8], "little")


def _free_value(policy: GrowthPolicy, identity: str) -> np.ndarray:
    rng = np.random.default_rng(_seed(policy.seed, identity))
    return (rng.standard_normal(policy.value_dim) * 0.01).astype("<f4")


class GrowthAllocator:
    """All publication occurs in the bank's SINGLE owned transaction."""
    def __init__(self, store: SQLiteKB, policy: GrowthPolicy):
        if store.value_dim != policy.value_dim:
            raise ValueError("Growth must use canonical semantic-value geometry")
        self.store, self.policy = store, policy
        self.demand = QueryDemand(store, policy)

    def grow(self, catalog: list[CatalogChunk], *, visibility: Visibility, step: int,
             budget: int) -> tuple[int, ...]:
        if budget < 0 or step < 0:
            raise ValueError("Growth budget/step must be nonnegative")
        if not self.store.growth_state("readiness", {}).get("coherence", False) or not (
            self.store.growth_state("readiness", {}).get("embeddings", False)
        ):
            raise RuntimeError("Free memory cannot grow before coherence AND embedding gates")
        existing = self.store.scan(Collection.MEMORY, visibility.scope)
        assigned = {x.source_assignment for x in existing}
        available = [x for x in catalog if x.scope == visibility.scope
                     and x.content_id not in assigned
                     and visibility.permits(x.footprint)
                     and not self.store.footprint_revoked(x.footprint)]
        counts = self.demand.counts()
        allocated: dict[str, int] = {}
        for row in existing:
            region = next((c.region for c in catalog if c.content_id == row.source_assignment),
                          "")
            allocated[region] = allocated.get(region, 0) + 1
        by_region: dict[str, list[CatalogChunk]] = {}
        for chunk in available:
            by_region.setdefault(chunk.region, []).append(chunk)
        for values in by_region.values():
            values.sort(key=lambda x: x.content_id)
        quota = min(budget, max(0, self.policy.max_points - len(existing)))
        main_quota = min(quota, max(1, math.ceil(quota * self.policy.dense_fraction))) if quota else 0
        selected: list[CatalogChunk] = []
        added: dict[str, int] = {}
        # Re-rank after EVERY choice. Otherwise one record per region wins the
        # density pass and the remaining budget devolves into lexical ordering,
        # regardless of where unmet query demand is concentrated.
        for _ in range(main_quota):
            live = [region for region, chunks in by_region.items() if chunks]
            if not live:
                break
            region = min(
                live,
                key=lambda name: (
                    -counts.get(name, 0) / (1 + allocated.get(name, 0) + added.get(name, 0)),
                    name,
                ),
            )
            selected.append(by_region[region].pop(0))
            added[region] = added.get(region, 0) + 1
        # An independently budgeted exploration tail samples real permitted
        # catalog chunks from the least represented regions, never synthetic
        # source-less records. It is not a graph bonus or RAG coverage signal.
        for _ in range(quota - len(selected)):
            live = [region for region, chunks in by_region.items() if chunks]
            if not live:
                break
            region = min(
                live,
                key=lambda name: (
                    allocated.get(name, 0) + added.get(name, 0), name,
                ),
            )
            selected.append(by_region[region].pop(0))
            added[region] = added.get(region, 0) + 1
        with self.store.transaction():
            results = []
            for c in selected:
                results.append(self.store.insert_memory(
                    scope=c.scope, footprint=c.footprint, key=c.key,
                    value=_free_value(self.policy, "point:" + c.content_id),
                    assignment=c.content_id, birth_step=step, group=c.group,
                ))
            self.store.put_growth_state("last_growth", {
                "step": step, "ids": results, "partition": self.policy.partition_id,
            })
        return tuple(results)

    def admit_pairs(self, windows: dict[str, tuple[str, ...]], *,
                    scope: str, visibility: Visibility) -> tuple[int, ...]:
        """Count each source window once irrespective of allocation order."""
        from .contracts import Collection
        memories = tuple(
            row for row in self.store.scan(Collection.MEMORY, scope)
            if row.active and visibility.permits(row.footprint)
            and not self.store.footprint_revoked(row.footprint)
        )
        assignment = {x.source_assignment: x for x in memories}
        previous = self.store.relations(scope)
        # UNIQUE(left,right) spans retired rows too: never re-propose them.
        existing = {(x.left, x.right)
                    for x in self.store.relations(scope, include_inactive=True)}
        degrees: dict[int, int] = {}
        for old in previous:
            degrees[old.left] = degrees.get(old.left, 0) + 1
            degrees[old.right] = degrees.get(old.right, 0) + 1
        counts: dict[tuple[str, str], set[str]] = {}
        for window_id, content_ids in windows.items():
            distinct = sorted(set(content_ids))
            for i, a in enumerate(distinct):
                for b in distinct[i+1:]:
                    counts.setdefault((a, b), set()).add(window_id)
        proposals = []
        budget = max(0, self.policy.max_pairs - len(previous))
        for a, b in sorted(counts, key=lambda key: (-len(counts[key]), key)):
            if len(counts[a,b]) < self.policy.min_pair_windows:
                continue
            if a not in assignment or b not in assignment:
                continue
            left, right = sorted((assignment[a], assignment[b]), key=lambda x: x.id)
            if (left.id, right.id) in existing:
                continue
            if degrees.get(left.id, 0) >= self.policy.max_neighbors or (
                degrees.get(right.id, 0) >= self.policy.max_neighbors
            ):
                continue
            joint = Footprint(tuple(dict.fromkeys(
                (*left.footprint.ranges, *right.footprint.ranges)
            )))
            if not visibility.permits(joint) or self.store.footprint_revoked(joint):
                continue
            if len(proposals) >= budget:
                break
            proposals.append((left, right, joint, len(counts[a,b])))
            degrees[left.id] = degrees.get(left.id, 0) + 1
            degrees[right.id] = degrees.get(right.id, 0) + 1
        with self.store.transaction():
            ids = []
            for left, right, support, n in proposals:
                prior = min(1.0, n / (n + 2.0))
                ids.append(self.store.insert_relation(
                    scope=scope, left=left.id, right=right.id, prior=prior,
                    value=_free_value(self.policy, f"pair:{left.id}:{right.id}"),
                    dependence_logits=(-2., -2.), footprint=support,
                ))
        return tuple(ids)
