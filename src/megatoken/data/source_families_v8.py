"""Independently check current source-family relations and resource accounting.

The producer's fresh source recheck derives lexical fingerprints and normalized
cores from the original JavaScript. This checker binds every original byte hash
and rebuilds the relation graph, clone comparisons, holdouts, and split choices.
It does not mistake a self-consistent registry digest for separation evidence.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter, defaultdict

from .semantic_json import (
    _js_string_length,
    _js_string_units,
    _js_utf8,
    stable_algorithm_hash,
)

_HASH = re.compile(r"[a-f0-9]{64}\Z")
_SHINGLE = re.compile(r"[a-f0-9]{24}\Z")
_BUDGET = ("maxComparisons", "maxCandidateVisits", "maxIndexEntries")
_HOLDOUT = ("families", "sourceDigests", "coreDigests")
_SPLITS = ("train", "validation", "test")


def _require(condition, message):
    if not condition:
        raise ValueError("V8 source-family registry: " + message)


def _exact(value, fields, label):
    _require(type(value) is dict and set(value) == set(fields), "malformed " + label)


def _safe_int(value, minimum=0):
    return type(value) is int and minimum <= value <= 2**53 - 1


def _hash(value):
    return type(value) is str and _HASH.fullmatch(value) is not None


def _strings(value, label):
    _require(type(value) is list and all(type(item) is str and item for item in value), label)
    _require(value == sorted(set(value), key=_js_string_units), "noncanonical " + label)
    return value


def _fingerprint(value):
    _exact(value, ("schema", "tokenCount", "exactHash", "alphaHash", "shingles"), "fingerprint")
    _require(value["schema"] == "owned-source-fingerprint/1", "unknown fingerprint")
    _require(_safe_int(value["tokenCount"], 1) and value["tokenCount"] <= 50000, "token count")
    _require(_hash(value["exactHash"]) and _hash(value["alphaHash"]), "fingerprint hashes")
    shingles = _strings(value["shingles"], "shingles")
    _require(
        len(shingles) <= 50000 and all(_SHINGLE.fullmatch(item) for item in shingles),
        "shingle bounds",
    )


class _Components:
    def __init__(self, size):
        self.parents = list(range(size))

    def root(self, index):
        while self.parents[index] != index:
            self.parents[index] = self.parents[self.parents[index]]
            index = self.parents[index]
        return index

    def join(self, left, right):
        left, right = self.root(left), self.root(right)
        self.parents[max(left, right)] = min(left, right)


def _clone_relations(rows, threshold, budget):
    """Complete Jaccard prefix filtering, with independently counted work."""
    groups = _Components(len(rows))
    identities = ({}, {})
    for index, row in enumerate(rows):
        fingerprint = row["sourceFingerprint"]
        for table, key in zip(
            identities,
            (
                fingerprint["exactHash"],
                fingerprint["alphaHash"] if fingerprint["tokenCount"] >= 20 else None,
            ),
            strict=True,
        ):
            if key is not None:
                if key in table:
                    groups.join(index, table[key])
                else:
                    table[key] = index
    frequencies = Counter(
        shingle for row in rows for shingle in row["sourceFingerprint"]["shingles"]
    )
    postings = defaultdict(list)
    costs = {"comparisons": 0, "candidateVisits": 0, "indexEntries": 0}

    def spend(resource, maximum):
        if costs[resource] >= budget[maximum]:
            _require(
                False,
                f"clone resource budget exhausted: {resource}; {maximum}={budget[maximum]}; "
                + ", ".join(f"{name}={value}" for name, value in costs.items()),
            )
        costs[resource] += 1

    near_pairs = 0
    for index, row in enumerate(rows):
        shingles = row["sourceFingerprint"]["shingles"]
        size = len(shingles)
        if not size:
            continue
        # Find the first overlap accepted by the same rounded division as the
        # final score. Multiplication followed by ceil can round a boundary
        # upward and omit a pair that the score comparison accepts.
        low, high = 0, size
        while low < high:
            middle = (low + high) // 2
            if middle / size >= threshold:
                high = middle
            else:
                low = middle + 1
        prefix = sorted(shingles, key=lambda item: (frequencies[item], item))[: size - low + 1]
        candidates = set()
        for shingle in prefix:
            for other in postings[shingle]:
                spend("candidateVisits", "maxCandidateVisits")
                other_size = len(rows[other]["sourceFingerprint"]["shingles"])
                if min(size, other_size) / max(size, other_size) >= threshold:
                    candidates.add(other)
        left = set(shingles)
        for other in candidates:
            spend("comparisons", "maxComparisons")
            right = set(rows[other]["sourceFingerprint"]["shingles"])
            intersection = len(left & right)
            if intersection / (len(left) + len(right) - intersection or 1) >= threshold:
                near_pairs += 1
                groups.join(index, other)
        for shingle in prefix:
            spend("indexEntries", "maxIndexEntries")
            postings[shingle].append(index)
    return groups, near_pairs, costs


def verify_region_family_registry(registry):
    """Validate the complete graph; source-derived claims also require fresh JS replay."""
    _exact(
        registry,
        (
            "schema",
            "policy",
            "traceFormat",
            "coreSchema",
            "seed",
            "holdouts",
            "nearThreshold",
            "cloneBudget",
            "sources",
            "components",
            "diagnostics",
            "digest",
        ),
        "registry",
    )
    _require(
        (registry["schema"], registry["policy"], registry["traceFormat"], registry["coreSchema"])
        == (
            "semantic-region-family-registry/1",
            "semantic-region-family-components/1",
            "semantic-trace/8",
            "semantic-algorithm/2",
        ),
        "unsupported current contract",
    )
    body = {key: value for key, value in registry.items() if key != "digest"}
    _require(registry["digest"] == stable_algorithm_hash(body), "digest differs")
    seed, threshold, budget = registry["seed"], registry["nearThreshold"], registry["cloneBudget"]
    _require(type(seed) is int and abs(seed) <= 2**53 - 1, "split seed")
    _require(
        type(threshold) in (int, float) and math.isfinite(threshold) and 0.8 <= threshold <= 1,
        "clone threshold",
    )
    _exact(budget, _BUDGET, "clone budget")
    _require(all(_safe_int(value) for value in budget.values()), "clone budget bounds")
    holdouts = registry["holdouts"]
    _exact(holdouts, _HOLDOUT, "holdouts")
    for key in _HOLDOUT:
        _strings(holdouts[key], key + " holdouts")
        if key != "families":
            _require(all(_hash(value) for value in holdouts[key]), "holdout hashes")
    sources = registry["sources"]
    _require(
        type(sources) is list and len(sources) > 0, "complete nonempty source inventory required"
    )
    for row in sources:
        _exact(
            row,
            (
                "id",
                "family",
                "ancestors",
                "source",
                "entry",
                "sourceDigest",
                "coreDigest",
                "coreStatus",
                "sourceFingerprint",
                "sourceFingerprintStatus",
            ),
            "source member",
        )
        for field in ("id", "family", "entry"):
            _require(
                type(row[field]) is str and 0 < _js_string_length(row[field]) <= 256,
                "source " + field,
            )
        _require(
            type(row["source"]) is str
            and row["source"]
            and len(_js_utf8(row["source"])) <= 2000000,
            "original source",
        )
        _require(
            row["sourceDigest"] == hashlib.sha256(_js_utf8(row["source"])).hexdigest(),
            "original source bytes changed",
        )
        ancestors = _strings(row["ancestors"], "ancestors")
        _require(
            len(ancestors) <= 1024
            and row["family"] not in ancestors
            and all(_js_string_length(value) <= 256 for value in ancestors),
            "source ancestry",
        )
        _require(
            (row["coreStatus"] == "algorithm" and _hash(row["coreDigest"]))
            or (row["coreStatus"] in ("detailed", "source-refused") and row["coreDigest"] is None),
            "normalized core status",
        )
        if row["sourceFingerprint"] is None:
            _require(
                row["sourceFingerprintStatus"] in ("tokenization-refused", "empty-token-stream"),
                "unassessed fingerprint status",
            )
        else:
            _require(row["sourceFingerprintStatus"] == "tokenized", "assessed fingerprint status")
            _fingerprint(row["sourceFingerprint"])
    ids = [row["id"] for row in sources]
    _require(ids == sorted(set(ids), key=_js_string_units), "duplicate or unordered source IDs")
    groups = _Components(len(sources))
    edges = {}
    for index, row in enumerate(sources):
        keys = ["family:" + value for value in (row["family"], *row["ancestors"])]
        keys.append("source:" + row["sourceDigest"])
        if row["coreDigest"] is not None:
            keys.append("core:" + row["coreDigest"])
        for key in keys:
            if key in edges:
                groups.join(index, edges[key])
            else:
                edges[key] = index
    representative_by_digest = {
        row["sourceDigest"]: row for row in sources if row["sourceFingerprint"] is not None
    }
    representatives = sorted(
        representative_by_digest.values(), key=lambda row: _js_string_units(row["id"])
    )
    clones, near_pairs, costs = _clone_relations(representatives, threshold, budget)
    positions = {row["id"]: index for index, row in enumerate(sources)}
    for index, row in enumerate(representatives):
        groups.join(positions[row["id"]], positions[representatives[clones.root(index)]["id"]])
    unassessed = [row["id"] for row in sources if row["sourceFingerprint"] is None]
    for identity in unassessed[1:]:
        groups.join(positions[unassessed[0]], positions[identity])
    known = {
        "families": {value for row in sources for value in (row["family"], *row["ancestors"])},
        "sourceDigests": {row["sourceDigest"] for row in sources},
        "coreDigests": {row["coreDigest"] for row in sources if row["coreDigest"] is not None},
    }
    for key in _HOLDOUT:
        _require(set(holdouts[key]) <= known[key], "unknown " + key + " holdout")
    grouped = defaultdict(list)
    for index, row in enumerate(sources):
        grouped[groups.root(index)].append(row)
    components = []
    for rows in grouped.values():
        identity = "semantic-region-component:" + stable_algorithm_hash(
            [{key: value for key, value in row.items() if key != "source"} for row in rows]
        )
        members = sorted(
            {value for row in rows for value in (row["family"], *row["ancestors"])},
            key=_js_string_units,
        )
        matched = {
            "families": [value for value in members if value in holdouts["families"]],
            "sourceDigests": sorted(
                {
                    row["sourceDigest"]
                    for row in rows
                    if row["sourceDigest"] in holdouts["sourceDigests"]
                }
            ),
            "coreDigests": sorted(
                {
                    row["coreDigest"]
                    for row in rows
                    if row["coreDigest"] is not None
                    and row["coreDigest"] in holdouts["coreDigests"]
                }
            ),
        }
        holdout = matched if any(matched.values()) else None
        bucket = int(hashlib.sha256(_js_utf8(str(seed) + ":" + identity)).hexdigest()[:8], 16) % 100
        split = (
            "test"
            if holdout
            else "train"
            if bucket < 80
            else "validation"
            if bucket < 90
            else "test"
        )
        components.append(
            {
                "id": identity,
                "members": members,
                "sourceIds": [row["id"] for row in rows],
                "split": split,
                "holdout": holdout,
            }
        )
    components.sort(key=lambda row: row["id"])
    _require(
        registry["components"] == components,
        "source relations, component membership, split or holdout changed",
    )
    diagnostics = {
        "sourceMembers": len(sources),
        "exactSourceVariants": len({row["sourceDigest"] + ":" + row["entry"] for row in sources}),
        "declaredFamilies": len({row["family"] for row in sources}),
        "normalizedCores": len(known["coreDigests"]),
        "withoutNormalizedCore": sum(row["coreDigest"] is None for row in sources),
        "nearPairs": near_pairs,
        "cloneAudit": {
            "algorithm": "complete-jaccard-prefix",
            "complete": not unassessed,
            "nearThreshold": threshold,
            "budget": budget,
            "costs": costs,
            "assessedSources": len(sources) - len(unassessed),
            "comparedSourceVariants": len(representatives),
            "unassessedSources": unassessed,
            "unassessedPolicy": "same-component",
        },
        "connectedComponents": len(components),
        "splitComponents": {
            split: sum(row["split"] == split for row in components) for split in _SPLITS
        },
    }
    _require(
        stable_algorithm_hash(registry["diagnostics"]) == stable_algorithm_hash(diagnostics),
        "clone accounting or coverage changed",
    )
    return {
        "valid": True,
        "digest": registry["digest"],
        "components": len(components),
        "nearPairs": near_pairs,
    }
