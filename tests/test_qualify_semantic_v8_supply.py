"""Pure controlled-fixture checks for the V8 SE supply qualifier.

These tests validate inventory and census refusal boundaries only. They do not
use the pinned tokenizer, prepared production corpus, model, or learner and are
not evidence that real production supply passed.
"""

import copy
import json

import pytest

from megatoken.data.semantic_json import stable_algorithm_hash
from scripts.qualify_semantic_v8_supply import (
    EXPECTED_SE_BEHAVIORS,
    SPLITS,
    read_expected_source_ids,
    validate_behavior_census,
    validate_observed_population,
    validate_registry_population,
)


def frozen_registry(counts=(56, 4, 4)):
    source_ids = [f"source-{index:02d}" for index in range(sum(counts))]
    splits = [split for split, count in zip(SPLITS, counts, strict=True) for _ in range(count)]
    registry = {
        "schema": "semantic-region-family-registry/1",
        "policy": "semantic-region-family-components/1",
        "traceFormat": "semantic-trace/8",
        "coreSchema": "semantic-algorithm/2",
        "seed": 17,
        "holdouts": [],
        "sources": [{"id": source_id} for source_id in source_ids],
        "components": [
            {"id": f"component-{index:02d}", "sourceIds": [source_id], "split": splits[index]}
            for index, source_id in enumerate(source_ids)
        ],
        "diagnostics": {},
    }
    registry["digest"] = stable_algorithm_hash(registry)
    return registry, source_ids, dict(zip(source_ids, splits, strict=True))


def test_expected_inventory_is_exact_and_registry_bound(tmp_path):
    registry, source_ids, split_by_id = frozen_registry()
    inventory = tmp_path / "source-ids.json"
    inventory.write_text(
        json.dumps(
            {
                "schema": "semantic-v8-source-population/1",
                "registryDigest": registry["digest"],
                "sourceIds": source_ids,
            }
        )
    )
    expected = read_expected_source_ids(inventory, registry["digest"])
    assert expected == tuple(sorted(source_ids))
    assert (
        validate_registry_population(
            registry, registry_digest=registry["digest"], expected_source_ids=expected
        )
        == split_by_id
    )

    with pytest.raises(ValueError, match="another registry"):
        read_expected_source_ids(inventory, "0" * 64)
    inventory.write_text(json.dumps(source_ids[:-1]))
    shorter = read_expected_source_ids(inventory, registry["digest"])
    with pytest.raises(ValueError, match="every expected source"):
        validate_registry_population(
            registry, registry_digest=registry["digest"], expected_source_ids=shorter
        )
    inventory.write_text(json.dumps(source_ids[:-1] + [source_ids[0]]))
    with pytest.raises(ValueError, match="unique nonempty"):
        read_expected_source_ids(inventory, registry["digest"])
    inventory.write_text(
        json.dumps(
            {
                "schema": "semantic-v8-source-population/1",
                "registryDigest": registry["digest"],
                "sourceIds": source_ids,
                "extra": "not declared",
            }
        )
    )
    with pytest.raises(ValueError, match="JSON array or bound population object"):
        read_expected_source_ids(inventory, registry["digest"])


def test_registry_and_actual_population_reject_missing_duplicate_and_moved_sources():
    registry, source_ids, split_by_id = frozen_registry()
    expected = tuple(sorted(source_ids))
    assert set(split_by_id.values()) == set(SPLITS)

    missing_registry_source = copy.deepcopy(registry)
    missing_registry_source["sources"].pop()
    missing_registry_source["digest"] = stable_algorithm_hash(
        {key: value for key, value in missing_registry_source.items() if key != "digest"}
    )
    with pytest.raises(ValueError, match="every expected source"):
        validate_registry_population(
            missing_registry_source,
            registry_digest=missing_registry_source["digest"],
            expected_source_ids=expected,
        )

    changed_test_population = copy.deepcopy(registry)
    changed_test_population["components"][-1]["split"] = "train"
    changed_test_population["digest"] = stable_algorithm_hash(
        {key: value for key, value in changed_test_population.items() if key != "digest"}
    )
    # A newly observed assignment is valid; old prepared episodes must still
    # match the precise assignment recorded for their run.
    changed_splits = validate_registry_population(
        changed_test_population,
        registry_digest=changed_test_population["digest"],
        expected_source_ids=expected,
    )
    with pytest.raises(ValueError, match="split"):
        validate_observed_population(
            [{"sourceId": identity, "split": split} for identity, split in split_by_id.items()],
            changed_splits,
        )

    observed = [{"sourceId": source_id, "split": split} for source_id, split in split_by_id.items()]
    actual = validate_observed_population(observed, split_by_id)
    assert {split: len(ids) for split, ids in actual.items()} == {
        "train": 56,
        "validation": 4,
        "test": 4,
    }
    with pytest.raises(ValueError, match="missing frozen source IDs"):
        validate_observed_population(observed[:-1], split_by_id)
    with pytest.raises(ValueError, match="missing frozen source IDs"):
        validate_observed_population(
            [row for row in observed if row["split"] != "validation"], split_by_id
        )
    with pytest.raises(ValueError, match="duplicate source ID"):
        validate_observed_population([*observed, observed[0]], split_by_id)
    changed_split = [*observed]
    changed_split[0] = {**changed_split[0], "split": "validation"}
    with pytest.raises(ValueError, match="frozen registry split"):
        validate_observed_population(changed_split, split_by_id)


def behavior_census():
    census = {}
    for split in ("train", "validation"):
        census[split] = {
            "episodes": 4,
            "behaviors": {
                behavior: {"sourceIds": [f"{split}-{behavior}"], "familyCount": 1}
                for behavior in EXPECTED_SE_BEHAVIORS
            },
        }
    census["test"] = {"episodes": 4, "behaviors": {}}
    return census


def test_behavior_gate_requires_all_eight_actual_census_entries_in_both_splits():
    census = behavior_census()
    validate_behavior_census(census)
    assert len(EXPECTED_SE_BEHAVIORS) == 8
    del census["validation"]["behaviors"]["semantic_child"]
    with pytest.raises(ValueError, match="validation behavior census.*semantic_child"):
        validate_behavior_census(census)

    census = behavior_census()
    census["train"]["episodes"] = 0
    with pytest.raises(ValueError, match="no train episodes"):
        validate_behavior_census(census)

    census = behavior_census()
    census["validation"]["behaviors"]["semantic_callback"]["familyCount"] = 0
    with pytest.raises(ValueError, match="validation behavior census.*semantic_callback"):
        validate_behavior_census(census)


@pytest.mark.parametrize("counts", [(1, 1, 1), (52, 8, 4), (80, 12, 7)])
def test_supply_uses_complete_observed_source_membership(counts):
    registry, identities, expected = frozen_registry(counts)
    assert (
        validate_registry_population(
            registry, registry_digest=registry["digest"], expected_source_ids=identities
        )
        == expected
    )


def test_large_supply_population_preserves_exact_assignment_and_refusal_checks():
    registry, identities, expected = frozen_registry((17000, 2000, 1000))
    assert (
        validate_registry_population(
            registry, registry_digest=registry["digest"], expected_source_ids=identities
        )
        == expected
    )
    for replacement in ("unknown-source", identities[0]):
        changed = copy.deepcopy(registry)
        changed["components"][-1]["sourceIds"] = [replacement]
        changed["digest"] = stable_algorithm_hash(
            {key: value for key, value in changed.items() if key != "digest"}
        )
        with pytest.raises(ValueError, match="duplicates or invents a source assignment"):
            validate_registry_population(
                changed, registry_digest=changed["digest"], expected_source_ids=identities
            )


@pytest.mark.parametrize("invalid_id", [None, "", [], {}, 7])
def test_supply_refuses_malformed_source_ids_before_membership_indexing(invalid_id):
    registry, identities, _expected = frozen_registry()
    registry["sources"][0]["id"] = invalid_id
    registry["digest"] = stable_algorithm_hash(
        {key: value for key, value in registry.items() if key != "digest"}
    )
    with pytest.raises(ValueError, match="full expected source inventory"):
        validate_registry_population(
            registry, registry_digest=registry["digest"], expected_source_ids=identities
        )
