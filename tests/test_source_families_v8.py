"""Current producer fingerprints feed an independently rebuilt native split graph."""

import copy
import json
import random
import unittest
from pathlib import Path

from megatoken.data.semantic_json import stable_algorithm_hash
from megatoken.data.source_families_v8 import _clone_relations, verify_region_family_registry


class SourceFamilyRegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry = json.loads((Path(__file__).parent / "fixtures/current-source-families-v8.json").read_text())

    def test_current_javascript_registry_rebuilds_with_uncertainty_and_exact_costs(self):
        result = verify_region_family_registry(self.registry)
        self.assertTrue(result["valid"])
        self.assertEqual(result["components"], 3)
        self.assertEqual(result["nearPairs"], 1)
        audit = self.registry["diagnostics"]["cloneAudit"]
        self.assertFalse(audit["complete"])
        self.assertEqual(audit["unassessedSources"], ["A😀", "A\ue000"])
        self.assertEqual(audit["comparedSourceVariants"], 3)

    def test_rehashed_tampering_does_not_establish_membership_or_clone_evidence(self):
        mutations = [
            lambda r: r["sources"][2].update(source="different original bytes"),
            lambda r: r["sources"].append(copy.deepcopy(r["sources"][-1])),
            lambda r: r["components"][0].update(split="validation"),
            lambda r: r["components"][0]["sourceIds"].pop(),
            lambda r: r["diagnostics"]["cloneAudit"]["costs"].update(comparisons=99),
            lambda r: r["diagnostics"]["cloneAudit"].update(complete=True),
            lambda r: r["diagnostics"]["cloneAudit"].update(unassessedSources=[]),
            lambda r: r["cloneBudget"].update(maxComparisons=0),
            lambda r: r["holdouts"]["families"].append("absent"),
            lambda r: r["sources"][2]["sourceFingerprint"]["shingles"].reverse(),
        ]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                changed = copy.deepcopy(self.registry)
                mutate(changed)
                changed["digest"] = stable_algorithm_hash({key: value for key, value in changed.items() if key != "digest"})
                with self.assertRaises(ValueError):
                    verify_region_family_registry(changed)

    def test_native_complete_filter_agrees_with_exhaustive_pairs_and_components(self):
        rng = random.Random(18271)
        rows = []
        for group in range(8):
            for variant in range(7):
                shingles = {group * 100 + item for item in range(24) if rng.randrange(20)}
                shingles.update((1000 + len(rows), 2000 + rng.randrange(4)))
                rows.append({"sourceFingerprint": {"tokenCount": 80,
                    "exactHash": f"{len(rows):064x}", "alphaHash": f"{len(rows) + 100000:064x}",
                    "shingles": [f"{item:024x}" for item in sorted(shingles)]}})
        budget = {"maxComparisons": 1000000, "maxCandidateVisits": 10000000, "maxIndexEntries": 2000000}
        for threshold in (0.8, 0.9, 0.92, 1):
            parents = list(range(len(rows)))

            def root(index):
                while parents[index] != index:
                    index = parents[index]
                return index

            pairs = 0
            for index, row in enumerate(rows):
                left = set(row["sourceFingerprint"]["shingles"])
                for other in range(index):
                    right = set(rows[other]["sourceFingerprint"]["shingles"])
                    if len(left & right) / len(left | right) >= threshold:
                        pairs += 1
                        parents[root(index)] = root(other)
            groups, actual_pairs, _costs = _clone_relations(rows, threshold, budget)
            self.assertEqual(actual_pairs, pairs)
            for index in range(len(rows)):
                for other in range(index):
                    self.assertEqual(groups.root(index) == groups.root(other), root(index) == root(other))

    def test_common_postings_are_complete_and_budget_failure_never_returns_a_partial_graph(self):
        rows = [{"sourceFingerprint": {"tokenCount": 80, "exactHash": f"{index:064x}",
            "alphaHash": f"{index + 100000:064x}",
            "shingles": [f"{item:024x}" for item in (*range(20), 1000 + index)]}}
            for index in range(140)]
        budget = {"maxComparisons": 1000000, "maxCandidateVisits": 10000000, "maxIndexEntries": 2000000}
        groups, pairs, _costs = _clone_relations(rows, 0.9, budget)
        self.assertEqual(pairs, 140 * 139 // 2)
        self.assertEqual(len({groups.root(index) for index in range(140)}), 1)
        with self.assertRaisesRegex(ValueError, "resource budget exhausted"):
            _clone_relations(rows, 0.9, {**budget, "maxComparisons": 0})


if __name__ == "__main__":
    unittest.main()
