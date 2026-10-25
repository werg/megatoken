import copy
import hashlib

import pytest

from megatoken.data.semantic_recipe_preparation import (
    benchmark_recipe_preparation_summary,
    validate_recipe_manifest,
)


def _split(family):
    bucket = int(hashlib.sha256(f"17:{family}".encode()).hexdigest()[:8], 16) % 100
    return "train" if bucket < 80 else "validation" if bucket < 90 else "test"


def _benchmark_pair():
    names = [f"algorithm-{index:02d}" for index in range(42)]
    families = [f"semantic-benchmark-v1:{name}" for name in names]
    splits = {family: _split(family) for family in families}
    lifecycle_families = {
        split: next(family for family in families if splits[family] == split)
        for split in ("train", "validation")
    }
    algorithms = [
        {
            "name": name,
            "body": f"let index=0;while(index<{index + 1})index=index+1;return index;",
            "family": families[index],
            "ancestor": "source-algorithm-v1:" + name,
            "split": splits[families[index]],
            "normalizedAstDigest": hashlib.sha256(name.encode()).hexdigest(),
        }
        for index, name in enumerate(names)
    ]
    split_counts = {
        split: sum(value == split for value in splits.values())
        for split in ("train", "validation", "test")
    }
    rows = []
    for family in families:
        split = splits[family]
        rows.append(
            {
                "id": family + "-pure",
                "family": family,
                "split": split,
                "role": "common-support",
                "status": "admitted",
                "path": family + "-pure.json",
            }
        )
        for suffix in ("callbacks", "repeated"):
            rows.append(
                {
                    "id": family + "-" + suffix,
                    "family": family,
                    "split": split,
                    "role": "effectful",
                    "status": "admitted",
                    "path": family + "-" + suffix + ".json",
                }
            )
    lifecycle = []
    for split, family in lifecycle_families.items():
        item = {
            "id": family + "-child",
            "family": family,
            "split": split,
            "path": family + "-child.json",
        }
        lifecycle.append(item)
        rows.append({**item, "role": "lifecycle-child", "status": "admitted"})
    recipe = {
        "schema": "semantic-source-benchmark/1",
        "name": "benchmark-v1",
        "algorithms": algorithms,
        "families": families,
        "splitCounts": split_counts,
        "normalizedAstDuplicateCount": 0,
        "normalizedAstAncestorDuplicateCount": 0,
        "retainEveryAuthoredFamily": True,
        "lifecycleInputs": lifecycle,
        "coverage": {
            "familyCount": len(families),
            "splitCounts": split_counts,
            "familyRoles": {
                "common-support": len(families),
                "effectful": len(families),
                "lifecycle-child": len(lifecycle),
            },
        },
    }
    manifest = {
        "schema": "semantic-corpus-manifest/4",
        "producerRevision": "a" * 40,
        "catalogDigest": "b" * 64,
        "representation": "compact",
        "recipe": recipe,
        "cases": rows,
    }
    return manifest, split_counts


def test_benchmark_preparation_requires_all_families_splits_roles_and_unique_ast_lineage():
    compact, expected_splits = _benchmark_pair()
    summary = validate_recipe_manifest(
        compact, recipe_name="benchmark-v1", representation="compact", producer_revision="a" * 40
    )
    assert summary["recipe"] == "benchmark-v1"
    assert summary["familyCount"] == summary["algorithmCount"] == 42
    assert summary["splitCounts"] == expected_splits
    assert summary["primaryCaseCounts"] == {"common-support": 42, "effectful": 84}
    assert summary["lifecycleInputCount"] == 2
    validate_recipe_manifest(
        compact, recipe_name="benchmark-v1", representation="compact", producer_revision="a" * 40
    )
    sidecar = benchmark_recipe_preparation_summary(summary, {"common-support/compact": 42})
    assert sidecar["schema"] == "semantic-benchmark-preparation/1"
    assert sidecar["preparedCaseCounts"] == {"common-support/compact": 42}
    assert sidecar["optimizerSteps"] == 0 and sidecar["learnedSuccess"] is None

    duplicate_ast = copy.deepcopy(compact)
    duplicate_ast["recipe"]["algorithms"][1]["normalizedAstDigest"] = duplicate_ast["recipe"][
        "algorithms"
    ][0]["normalizedAstDigest"]
    with pytest.raises(ValueError, match="normalized-AST duplicates"):
        validate_recipe_manifest(
            duplicate_ast,
            recipe_name="benchmark-v1",
            representation="compact",
            producer_revision="a" * 40,
        )


def test_training_v4_remains_the_default_compatible_manifest_branch():
    family = "semantic-recipe:fixture"
    split = _split(family)
    manifest = {
        "schema": "semantic-corpus-manifest/4",
        "producerRevision": "c" * 40,
        "representation": "compact",
        "recipe": {
            "schema": "semantic-source-recipe/4",
            "name": "training-v4",
            "families": [family],
            "splitSeed": 17,
            "retainEveryAuthoredFamily": True,
            "templateDigest": "d" * 64,
        },
        "cases": [
            {
                "id": "pure",
                "family": family,
                "split": split,
                "role": "common-support",
                "status": "admitted",
                "path": "pure.json",
            },
            {
                "id": "callbacks",
                "family": family,
                "split": split,
                "role": "effectful",
                "status": "admitted",
                "path": "callbacks.json",
            },
        ],
    }
    result = validate_recipe_manifest(
        manifest, recipe_name="training-v4", representation="compact", producer_revision="c" * 40
    )
    assert result["familyCount"] == 1


def test_benchmark_preparation_rejects_split_leakage_or_unadmitted_case():
    compact, _ = _benchmark_pair()
    crossed = copy.deepcopy(compact)
    crossed["cases"][0]["split"] = "test" if crossed["cases"][0]["split"] != "test" else "train"
    with pytest.raises(ValueError, match="split"):
        validate_recipe_manifest(
            crossed,
            recipe_name="benchmark-v1",
            representation="compact",
            producer_revision="a" * 40,
        )
    unadmitted = copy.deepcopy(compact)
    unadmitted["cases"][0]["status"] = "locally_verified"
    with pytest.raises(ValueError, match="admitted"):
        validate_recipe_manifest(
            unadmitted,
            recipe_name="benchmark-v1",
            representation="compact",
            producer_revision="a" * 40,
        )


def test_benchmark_primary_role_counts_are_checked_per_family_not_only_globally():
    compact, _ = _benchmark_pair()
    same_split_families = next(
        (left, right)
        for left in compact["recipe"]["families"]
        for right in compact["recipe"]["families"]
        if left < right and _split(left) == _split(right)
    )
    left, right = same_split_families
    shifted = copy.deepcopy(compact)
    source_row = next(
        case for case in shifted["cases"] if case["family"] == left and case["role"] == "effectful"
    )
    source_row["family"] = right
    # The global role total remains 84, all IDs and splits remain valid, but the
    # family counts become 1+3 and must be rejected.
    assert sum(case["role"] == "effectful" for case in shifted["cases"]) == 84
    with pytest.raises(ValueError, match="exactly one pure and two effectful"):
        validate_recipe_manifest(
            shifted,
            recipe_name="benchmark-v1",
            representation="compact",
            producer_revision="a" * 40,
        )

def test_obsolete_recipe_representation_is_refused():
    compact, _ = _benchmark_pair()
    obsolete = copy.deepcopy(compact)
    obsolete["representation"] = "explicit"
    with pytest.raises(ValueError, match="Complete source/native admitted"):
        validate_recipe_manifest(
            obsolete, recipe_name="benchmark-v1", representation="compact",
            producer_revision="a" * 40,
        )
