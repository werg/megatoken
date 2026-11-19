"""Rebuild frozen v7 semantic components and enforce exact split membership.

Python independently verifies source bytes, relations, component identities,
holdouts and split assignments. Source-to-core lowering is separately rechecked
by the pinned producer; its fresh source receipt binds the entire registry.
"""

from __future__ import annotations

import hashlib
import re

from .algorithm_v7 import stable_algorithm_hash

HASH = re.compile(r"[0-9a-f]{64}\Z")
SPLITS = ("train", "validation", "test")


def _sort(values):
    return sorted(set(values), key=lambda value: value.encode("utf-16-be", "surrogatepass"))


def _text(value, limit=256):
    return isinstance(value, str) and 0 < len(value.encode("utf-16-le", "surrogatepass")) // 2 <= limit


def _exact(row, fields, label):
    if not isinstance(row, dict) or set(row) != set(fields):
        raise ValueError(f"V7 {label} has missing or unknown fields")


def verify_family_registry_v7(registry):
    _exact(registry, {"schema", "policy", "seed", "holdouts", "sources", "components", "diagnostics", "digest"}, "family registry")
    if registry["schema"] != "semantic-family-registry/1" or registry["policy"] != "semantic-family-components/1":
        raise ValueError("V7 family registry contract differs")
    seed = registry["seed"]
    if type(seed) is not int or abs(seed) > 9007199254740991:
        raise ValueError("V7 split seed must be an exact safe integer")
    body = {key: value for key, value in registry.items() if key != "digest"}
    if stable_algorithm_hash(body) != registry["digest"]:
        raise ValueError("V7 family registry digest changed")
    holdouts = registry["holdouts"]
    _exact(holdouts, {"families", "sourceDigests", "coreDigests"}, "holdouts")
    for key, values in holdouts.items():
        if not isinstance(values, list) or any(not _text(value, 1024) for value in values) or values != _sort(values):
            raise ValueError("V7 explicit holdouts must be sorted and unique")
        if key != "families" and any(not HASH.fullmatch(value) for value in values):
            raise ValueError("V7 source/core holdout digest differs")
    sources = registry["sources"]
    if not isinstance(sources, list) or not 1 <= len(sources) <= 20000:
        raise ValueError("V7 registry requires its complete bounded source inventory")
    for row in sources:
        _exact(row, {"id", "family", "ancestors", "entry", "source", "sourceDigest", "coreDigest"}, "source member")
        if any(not _text(row[key]) for key in ("id", "family", "entry")):
            raise ValueError("V7 source member identity differs")
        if not isinstance(row["source"], str) or not 1 <= len(row["source"].encode("utf-8")) <= 2000000:
            raise ValueError("V7 source bytes are missing or over budget")
        if hashlib.sha256(row["source"].encode("utf-8")).hexdigest() != row["sourceDigest"]:
            raise ValueError("V7 source digest differs from exact source bytes")
        if row["coreDigest"] is not None and (not isinstance(row["coreDigest"], str) or not HASH.fullmatch(row["coreDigest"])):
            raise ValueError("V7 normalized core identity differs")
        if not isinstance(row["ancestors"], list) or any(not _text(value) for value in row["ancestors"]) or row["ancestors"] != _sort(row["ancestors"]):
            raise ValueError("V7 complete family ancestry differs")
    if [row["id"] for row in sources] != _sort(row["id"] for row in sources):
        raise ValueError("V7 sources must have distinct sorted identities")
    parent = list(range(len(sources)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    related = {}
    for index, row in enumerate(sources):
        keys = ["family:" + value for value in _sort([row["family"], *row["ancestors"]])]
        keys.append("source:" + row["sourceDigest"])
        if row["coreDigest"] is not None:
            keys.append("core:" + row["coreDigest"])
        for key in keys:
            if key in related:
                left, right = find(index), find(related[key])
                parent[max(left, right)] = min(left, right)
            else:
                related[key] = index
    grouped = {}
    for index, row in enumerate(sources):
        grouped.setdefault(find(index), []).append(row)
    components = []
    for rows in grouped.values():
        member_rows = [{key: value for key, value in row.items() if key != "source"} for row in rows]
        members = _sort(value for row in rows for value in [row["family"], *row["ancestors"]])
        identity = "semantic-component:" + stable_algorithm_hash(member_rows)
        forced_families = [value for value in members if value in holdouts["families"]]
        forced_sources = _sort(row["sourceDigest"] for row in rows if row["sourceDigest"] in holdouts["sourceDigests"])
        forced_cores = _sort(row["coreDigest"] for row in rows if row["coreDigest"] is not None and row["coreDigest"] in holdouts["coreDigests"])
        forced = next(({"kind": kind, "keys": values} for kind, values in (
            ("family", forced_families), ("exact-source", forced_sources), ("normalized-core", forced_cores)
        ) if values), None)
        bucket = int(hashlib.sha256(f"{seed}:{identity}".encode()).hexdigest()[:8], 16) % 100
        split = "test" if forced else "train" if bucket < 80 else "validation" if bucket < 90 else "test"
        components.append({"id": identity, "members": members, "sourceIds": _sort(row["id"] for row in rows), "split": split, "holdout": forced})
    components.sort(key=lambda row: row["id"].encode("utf-16-be"))
    if components != registry["components"]:
        raise ValueError("V7 related sources, normalized cores, split or holdout changed")
    known = {
        "families": {value for row in sources for value in [row["family"], *row["ancestors"]]},
        "sourceDigests": {row["sourceDigest"] for row in sources},
        "coreDigests": {row["coreDigest"] for row in sources if row["coreDigest"] is not None},
    }
    if any(set(values) - known[key] for key, values in holdouts.items()):
        raise ValueError("V7 explicit holdout names an unknown registry member")
    diagnostics = {"sourceVariants": len(sources), "declaredFamilies": len({row["family"] for row in sources}),
                   "normalizedCores": len(known["coreDigests"]), "connectedComponents": len(components),
                   "splitComponents": {split: sum(row["split"] == split for row in components) for split in SPLITS}}
    if registry["diagnostics"] != diagnostics:
        raise ValueError("V7 registry diagnostics differ from rebuilt components")
    return registry["digest"]


def verify_split_identity_v7(document, *, source_receipt=None):
    side = document["sidecar"]
    identity = side.get("splitIdentity")
    _exact(identity, {"schema", "policy", "registryDigest", "seed", "sourceId", "family", "sourceDigest", "coreDigest", "componentId", "members", "split", "holdout", "registry"}, "split identity")
    registry = identity["registry"]
    digest = verify_family_registry_v7(registry)
    member = next((row for row in registry["sources"] if row["id"] == identity["sourceId"]), None)
    if member is None:
        raise ValueError("V7 dataset source is absent from the frozen registry")
    component = next(row for row in registry["components"] if member["id"] in row["sourceIds"])
    expected = {"schema": "semantic-split-identity/2", "policy": registry["policy"], "registryDigest": digest,
                "seed": registry["seed"], "sourceId": member["id"], "family": member["family"],
                "sourceDigest": member["sourceDigest"], "coreDigest": member["coreDigest"],
                "componentId": component["id"], "members": component["members"], "split": component["split"],
                "holdout": component["holdout"], "registry": registry}
    if identity != expected:
        raise ValueError("V7 source identity or immutable split was relabeled")
    private = side.get("sourceArtifact", {}).get("sidecar", {})
    reference = private.get("referenceInput", {})
    if reference.get("source") != member["source"] or reference.get("entry") != member["entry"] or side.get("sourceFamily") != member["family"] or private.get("sourceFamily") != member["family"]:
        raise ValueError("V7 registry membership differs from exact source evidence")
    if document["public"]["representation"] == "algorithm" and side.get("coreDigest") != member["coreDigest"]:
        raise ValueError("V7 algorithm core differs from its registered semantic component")
    binding = {"schema": "semantic-dataset-binding/1", "representation": document["public"]["representation"],
               "episodeDigest": stable_algorithm_hash(document["public"]), "contextsDigest": stable_algorithm_hash(document["contexts"]),
               "registryDigest": digest, "componentId": component["id"], "split": component["split"],
               "sourceId": member["id"], "sourceDigest": member["sourceDigest"], "coreDigest": member["coreDigest"]}
    if side.get("datasetBinding") != binding or side.get("split") != component["split"] or side.get("sourceComponent") != component["id"] or side.get("familyRegistryDigest") != digest:
        raise ValueError("V7 dataset binding differs from exact public/context/source bytes")
    if source_receipt is not None and source_receipt.get("registryDigest") != digest:
        raise ValueError("Fresh source-to-core admission did not cover the entire family registry")
    return identity
