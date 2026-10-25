"""Strict manifest checks for canonical semantic recipe preparation."""

from __future__ import annotations

import hashlib
from collections import Counter
from typing import Any

RECIPE_SCHEMAS = {
    "training-v4": "semantic-source-recipe/4",
    "benchmark-v1": "semantic-source-benchmark/1",
}
SPLITS = ("train", "validation", "test")
ROLES = ("common-support", "effectful", "lifecycle-child")


def _family_split(family: str) -> str:
    bucket = int(hashlib.sha256(f"17:{family}".encode()).hexdigest()[:8], 16) % 100
    return "train" if bucket < 80 else "validation" if bucket < 90 else "test"


def validate_recipe_manifest(
    manifest: dict[str, Any], *, recipe_name: str, representation: str, producer_revision: str
) -> dict[str, Any]:
    """Validate a source/native admitted corpus and return its family-level recipe summary."""
    if recipe_name not in RECIPE_SCHEMAS or representation not in {"explicit", "compact"}:
        raise ValueError("Unsupported canonical recipe or representation")
    recipe = manifest.get("recipe")
    if (
        manifest.get("schema") != "semantic-corpus-manifest/4"
        or manifest.get("producerRevision") != producer_revision
        or manifest.get("representation") != representation
        or not isinstance(recipe, dict)
        or recipe.get("name") != recipe_name
        or recipe.get("schema") != RECIPE_SCHEMAS[recipe_name]
        or recipe.get("retainEveryAuthoredFamily") is not True
        or not isinstance(manifest.get("cases"), list)
        or not manifest["cases"]
        or any(case.get("status") != "admitted" for case in manifest["cases"])
    ):
        raise ValueError(f"Complete source/native admitted {recipe_name} manifest required")

    families = recipe.get("families")
    if (
        not isinstance(families, list)
        or not families
        or any(not isinstance(family, str) or not family for family in families)
        or len(set(families)) != len(families)
    ):
        raise ValueError("Recipe family inventory is missing or duplicated")
    family_set = set(families)
    family_splits = {family: _family_split(family) for family in families}
    split_counts = {
        split: sum(value == split for value in family_splits.values()) for split in SPLITS
    }
    actual_split_counts = {split: split_counts[split] for split in SPLITS}

    seen_ids, role_families = set(), {role: set() for role in ROLES}
    role_rows = Counter()
    family_role_rows = Counter()
    for case in manifest["cases"]:
        if (
            not isinstance(case, dict)
            or not isinstance(case.get("id"), str)
            or case["id"] in seen_ids
            or case.get("family") not in family_set
            or case.get("split") != family_splits[case["family"]]
            or case.get("role") not in role_families
            or not isinstance(case.get("path"), str)
            or not case["path"]
        ):
            raise ValueError("Recipe case identity, split, role or path is invalid")
        seen_ids.add(case["id"])
        role_families[case["role"]].add(case["family"])
        role_rows[case["role"]] += 1
        family_role_rows[(case["family"], case["role"])] += 1

    if recipe_name == "training-v4":
        if set(role_families["common-support"]) != family_set or not role_families["effectful"]:
            raise ValueError("training-v4 role coverage is incomplete")
        if recipe.get("splitSeed") != 17 or recipe.get("splitCounts") not in (
            None,
            actual_split_counts,
        ):
            raise ValueError("training-v4 split metadata differs from the fixed family hash")
        return {
            "recipe": recipe_name,
            "familyCount": len(families),
            "splitCounts": actual_split_counts,
            "familyRoles": {role: len(values) for role, values in role_families.items()},
        }

    algorithms = recipe.get("algorithms")
    if not isinstance(algorithms, list) or len(algorithms) != len(families):
        raise ValueError("benchmark-v1 algorithm catalogue differs from its family inventory")
    algorithm_families, algorithm_names, ast_digests = [], [], []
    for algorithm in algorithms:
        if not isinstance(algorithm, dict):
            raise ValueError("Malformed benchmark-v1 algorithm declaration")
        family = algorithm.get("family")
        ast_digest = algorithm.get("normalizedAstDigest")
        if (
            family not in family_set
            or algorithm.get("split") != family_splits[family]
            or not isinstance(algorithm.get("name"), str)
            or algorithm.get("ancestor") != "source-algorithm-v1:" + algorithm["name"]
            or not isinstance(algorithm.get("body"), str)
            or not isinstance(ast_digest, str)
            or len(ast_digest) != 64
            or any(character not in "0123456789abcdef" for character in ast_digest)
        ):
            raise ValueError("Benchmark algorithm split, ancestry or AST identity is invalid")
        algorithm_families.append(family)
        algorithm_names.append(algorithm["name"])
        ast_digests.append(ast_digest)
    if (
        set(algorithm_families) != family_set
        or len(set(algorithm_families)) != len(families)
        or len(set(algorithm_names)) != len(families)
    ):
        raise ValueError("Benchmark algorithm families are duplicated or incomplete")
    if len(set(ast_digests)) != len(ast_digests):
        raise ValueError("Benchmark catalogue contains normalized-AST duplicates")
    if (
        recipe.get("normalizedAstDuplicateCount") != 0
        or recipe.get("normalizedAstAncestorDuplicateCount") != 0
    ):
        raise ValueError("Benchmark catalogue duplicates a new or training-v4 source algorithm")
    if recipe.get("splitCounts") != actual_split_counts:
        raise ValueError("Benchmark family split counts differ from the seed-17 hash")
    coverage = recipe.get("coverage")
    if (
        not isinstance(coverage, dict)
        or coverage.get("familyCount") != len(families)
        or coverage.get("splitCounts") != actual_split_counts
        or coverage.get("familyRoles")
        != {role: len(values) for role, values in role_families.items()}
        or set(role_families["common-support"]) != family_set
        or set(role_families["effectful"]) != family_set
    ):
        raise ValueError("Benchmark family and role coverage metadata differs from emitted cases")
    if role_rows["common-support"] != len(families) or role_rows["effectful"] != 2 * len(families):
        raise ValueError("Benchmark must retain one pure and two effectful cases per family")
    if any(
        family_role_rows[(family, "common-support")] != 1
        or family_role_rows[(family, "effectful")] != 2
        for family in families
    ):
        raise ValueError(
            "Benchmark family must retain exactly one pure and two effectful primary cases"
        )

    lifecycle = recipe.get("lifecycleInputs")
    if not isinstance(lifecycle, list) or len(lifecycle) != 2:
        raise ValueError(
            "Benchmark must declare one certified lifecycle child per train/validation split"
        )
    lifecycle_splits = set()
    for child in lifecycle:
        if (
            not isinstance(child, dict)
            or child.get("split") not in {"train", "validation"}
            or child.get("family") not in family_set
            or child.get("split") != family_splits[child["family"]]
            or not isinstance(child.get("path"), str)
            or not child["path"]
        ):
            raise ValueError("Benchmark lifecycle child changed its family or frozen split")
        lifecycle_splits.add(child["split"])
    if lifecycle_splits != {"train", "validation"}:
        raise ValueError("Benchmark lifecycle children must retain train and validation inputs")
    if role_rows["lifecycle-child"] != len(lifecycle):
        raise ValueError("Benchmark lifecycle manifest and emitted child cases differ")
    emitted_children = {
        (case["id"], case["family"], case["split"], case["path"])
        for case in manifest["cases"]
        if case["role"] == "lifecycle-child"
    }
    declared_children = {
        (child["id"], child["family"], child["split"], child["path"]) for child in lifecycle
    }
    if emitted_children != declared_children:
        raise ValueError("Benchmark lifecycle declarations differ from emitted child cases")
    return {
        "recipe": recipe_name,
        "familyCount": len(families),
        "splitCounts": actual_split_counts,
        "familyRoles": {role: len(values) for role, values in role_families.items()},
        "algorithmCount": len(algorithms),
        "normalizedAstDuplicateCount": len(ast_digests) - len(set(ast_digests)),
        "normalizedAstAncestorDuplicateCount": recipe["normalizedAstAncestorDuplicateCount"],
        "primaryCaseCounts": {
            "common-support": role_rows["common-support"],
            "effectful": role_rows["effectful"],
        },
        "lifecycleInputCount": len(lifecycle),
    }


def validate_paired_recipe_manifests(left: dict[str, Any], right: dict[str, Any]) -> None:
    """Require explicit/compact manifests to retain identical families, splits and roles."""
    if left["recipe"]["name"] != right["recipe"]["name"]:
        raise ValueError("Explicit and compact recipe names differ")
    recipe = left["recipe"]["name"]
    if recipe == "training-v4":
        if left["recipe"].get("templateDigest") != right["recipe"].get("templateDigest"):
            raise ValueError("Explicit and compact training-v4 template declarations differ")
    elif recipe == "benchmark-v1":
        left_algorithms = left["recipe"].get("algorithms")
        right_algorithms = right["recipe"].get("algorithms")
        if left_algorithms != right_algorithms:
            raise ValueError("Explicit and compact benchmark algorithm catalogues differ")
    else:
        raise ValueError("Unsupported paired semantic recipe")

    def case_keys(manifest):
        return {
            (case["id"], case["family"], case["split"], case["role"]) for case in manifest["cases"]
        }

    if case_keys(left) != case_keys(right):
        raise ValueError("Explicit and compact recipe families, splits or roles are not paired")


def benchmark_recipe_preparation_summary(manifest_summary, prepared_counts):
    """Separate benchmark lineage/count evidence from unchanged v4 dataset manifests."""
    if manifest_summary.get("recipe") != "benchmark-v1":
        raise ValueError("Benchmark preparation summary requires benchmark-v1 metadata")
    return {
        "schema": "semantic-benchmark-preparation/1",
        **manifest_summary,
        "preparedCaseCounts": dict(sorted(prepared_counts.items())),
        "optimizerSteps": 0,
        "learnedSuccess": None,
    }
