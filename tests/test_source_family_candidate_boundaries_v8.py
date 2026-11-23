"""Boundary and refusal checks for complete source-family candidate discovery."""

import math

import pytest

from megatoken.data.source_families_v8 import _clone_relations

BUDGET = {
    "maxComparisons": 1000000,
    "maxCandidateVisits": 10000000,
    "maxIndexEntries": 2000000,
}


def fingerprint_row(identity, shingles):
    return {
        "sourceFingerprint": {
            "tokenCount": 80,
            "exactHash": f"{identity:064x}",
            "alphaHash": f"{identity + 100000:064x}",
            "shingles": [f"{item:024x}" for item in sorted(shingles)],
        }
    }


@pytest.mark.parametrize("reverse", [False, True])
def test_threshold_equality_subset_is_discovered_in_both_orders(reverse):
    rows = [fingerprint_row(0, range(29)), fingerprint_row(1, range(35))]
    if reverse:
        rows.reverse()
    # The final Jaccard division accepts this exact threshold. Multiplying it
    # back by 35 rounds upward, so ceil(threshold*35) would wrongly require 30.
    threshold = 29 / 35
    assert threshold * 35 > 29
    groups, pairs, _costs = _clone_relations(rows, threshold, BUDGET)
    assert pairs == 1
    assert groups.root(0) == groups.root(1)


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("direction", [-math.inf, math.inf])
def test_threshold_neighbors_retain_the_exact_final_comparison(reverse, direction):
    rows = [fingerprint_row(0, range(29)), fingerprint_row(1, range(35))]
    if reverse:
        rows.reverse()
    threshold = math.nextafter(29 / 35, direction)
    groups, pairs, _costs = _clone_relations(rows, threshold, BUDGET)
    expected = int(29 / 35 >= threshold)
    assert pairs == expected
    assert (groups.root(0) == groups.root(1)) == bool(expected)


@pytest.mark.parametrize("threshold", [0.8, 29 / 35, 0.9, 0.92, 1.0])
def test_boundary_population_matches_exhaustive_pairs_and_components(threshold):
    rows = [fingerprint_row(index, range(size)) for index, size in enumerate(range(25, 45))]
    # Keep one genuine disjoint component; matching near-pair counts alone
    # would not detect an accidental connection to every source.
    rows.append(fingerprint_row(len(rows), range(100, 135)))
    parents = list(range(len(rows)))

    def root(index):
        while parents[index] != index:
            index = parents[index]
        return index

    expected_pairs = 0
    for index, row in enumerate(rows):
        left = set(row["sourceFingerprint"]["shingles"])
        for other in range(index):
            right = set(rows[other]["sourceFingerprint"]["shingles"])
            if len(left & right) / len(left | right) >= threshold:
                expected_pairs += 1
                parents[root(index)] = root(other)
    groups, pairs, _costs = _clone_relations(rows, threshold, BUDGET)
    assert pairs == expected_pairs
    for index in range(len(rows)):
        for other in range(index):
            assert (groups.root(index) == groups.root(other)) == (root(index) == root(other))


@pytest.mark.parametrize(
    ("maximum", "resource", "expected_costs"),
    [
        (
            "maxComparisons",
            "comparisons",
            {"comparisons": 0, "candidateVisits": 3, "indexEntries": 3},
        ),
        (
            "maxCandidateVisits",
            "candidateVisits",
            {"comparisons": 0, "candidateVisits": 0, "indexEntries": 3},
        ),
        (
            "maxIndexEntries",
            "indexEntries",
            {"comparisons": 0, "candidateVisits": 0, "indexEntries": 0},
        ),
    ],
)
def test_budget_refusal_reports_completed_work_and_exhausted_limit(
    maximum, resource, expected_costs
):
    rows = [fingerprint_row(0, range(20)), fingerprint_row(1, range(20))]
    with pytest.raises(ValueError) as caught:
        _clone_relations(rows, 0.9, {**BUDGET, maximum: 0})
    message = str(caught.value)
    assert f"clone resource budget exhausted: {resource}" in message
    assert f"{maximum}=0" in message
    for name, value in expected_costs.items():
        assert f"{name}={value}" in message
