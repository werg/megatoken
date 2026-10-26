import copy

import pytest

from megatoken.data.host_operations import CONTRACT_DIGEST
from megatoken.generation.codec import ByteCodec
from megatoken.training.semantic_analysis import analyze_semantic_evaluation
from megatoken.training.semantic_evaluation import (
    HARD_GATES,
    _current_execution_identity,
    _digest,
    compare_experiments,
    evaluate_cohort,
    freeze_cohort,
)


def _cohort():
    cases = []
    for case_id, family, source in (
        ("gcd-a", "family-gcd", "function run(a,b){while(b){let r=a%b;a=b;b=r;}return a;}"),
        ("gcd-b", "family-gcd", "function run(a,b){while(b){let r=a%b;a=b;b=r;}return a;}"),
        ("sum-a", "family-array", "function run(a){let s=0;for(let x of a)s+=x;return s;}"),
    ):
        request = {"program": {"source": source}}
        expected = {
            "completion": {"kind": "return", "value": ["num", 1]},
            "state": {"roots": [["num", 1]]},
            "effects": [],
            "callbacks": [],
            "cleanup": {"pendingEffects": 0, "pendingJobs": 0, "openHandlers": 0},
        }
        cases.append(
            {
                "id": case_id,
                "family": family,
                "split": "test",
                "request": request,
                "expected": expected,
                "referenceReceipt": {
                    "status": "agreed",
                    "requestDigest": _digest(request),
                    "outcomeDigest": _digest(expected),
                },
            }
        )
    return freeze_cohort(
        cases, name="effectful-tracks-events", thresholds={"wholeTaskSuccess": 0.9, "traceValid": 1}
    )


def _report(cohort):
    rows = []
    for index, case in enumerate(cohort["cases"]):
        rows.append(
            {
                "caseId": case["id"],
                "family": case["family"],
                "wholeTaskSuccess": index != 1,
                "traceValid": True,
                "firstSemanticInconsistency": None
                if index != 1
                else {"record": 4, "reason": "wrong value"},
                "matches": {
                    key: index != 1
                    for key in ("completion", "state", "effects", "callbacks", "cleanup")
                },
                "hardGates": dict.fromkeys(HARD_GATES, 0),
                "costs": {
                    "computeSeconds": 0.1 + index,
                    "peakMemoryBytes": 1000 + index,
                    "dispatches": 1,
                    "queueDepth": 0,
                    "leakedResources": 0,
                },
                "latencySeconds": 0.2 + index,
                "costScope": {"computeSeconds": "case-wall-time", "peakMemoryBytes": "process-rss"},
                "failure": None if index != 1 else "outcome-or-hard-gate",
                "failureEvidence": None if index != 1 else {"category": "semantic", "record": 4},
            }
        )
    return {
        "schema": "semantic-evaluation-report/5",
        "cohortDigest": cohort["digest"],
        "cohort": cohort["name"],
        "origin": "controlled",
        "checkpointIdentity": None,
        "seed": 17,
        "representation": "compact",
        "evaluationSplits": ["test"],
        "catalogDigest": CONTRACT_DIGEST,
        **_current_execution_identity(),
        "learnedSuccess": None,
        "summary": {
            "tasks": len(rows),
            "correctTasks": sum(row["wholeTaskSuccess"] for row in rows),
            "wholeTaskSuccess": sum(row["wholeTaskSuccess"] for row in rows) / len(rows),
        },
        "cases": rows,
        "sourceLengthBands": [
            {"name": "0-63", "minBytes": 0, "maxBytes": 63},
            {"name": "64+", "minBytes": 64, "maxBytes": None},
        ],
        "behaviorMetadata": {"gcd-a": ["loop", "integer"], "sum-a": ["array-reduction"]},
    }


def test_analysis_is_family_primary_and_retains_exact_failure_receipts():
    cohort = _cohort()
    report = _report(cohort)
    analysis = analyze_semantic_evaluation(cohort, report)
    assert analysis["schema"] == "semantic-evaluation-analysis/1"
    assert analysis["summary"]["cases"] == 3
    assert analysis["summary"]["families"] == 2
    assert analysis["summary"]["successfulCases"] == 2
    assert analysis["summary"]["successfulFamilies"] == 1
    gcd = next(item for item in analysis["familySummaries"] if item["family"] == "family-gcd")
    assert gcd["caseCount"] == 2 and not gcd["familySuccess"]
    assert gcd["behaviors"] == ["integer", "loop"]
    assert gcd["firstSemanticInconsistency"] == [
        {"caseId": "gcd-b", "receipt": {"record": 4, "reason": "wrong value"}}
    ]
    assert gcd["failureEvidence"] == [
        {"caseId": "gcd-b", "evidence": {"category": "semantic", "record": 4}}
    ]
    assert analysis["costBreakdownByScope"][0]["latencySeconds"]["p50"] == 1.2
    assert analysis["cases"][0]["sourceLengthBand"] in {"0-63", "64+"}
    assert analysis["analysisLabelsForInference"] is False


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda c, r: r.update(cohortDigest="0" * 64), "does not match"),
        (lambda c, r: r["cases"].append(copy.deepcopy(r["cases"][0])), "exactly match"),
        (lambda c, r: r["cases"][0].update(family="wrong-family"), "family differs"),
        (
            lambda c, r: r["cases"][0]["costs"].update(computeSeconds=float("nan")),
            "finite nonnegative",
        ),
        (lambda c, r: r.update(learnedSuccess=1.0), "cannot claim learnedSuccess"),
    ],
)
def test_analysis_rejects_unmatched_or_unmeasured_reports(mutation, match):
    cohort = _cohort()
    report = _report(cohort)
    mutation(cohort, report)
    with pytest.raises(ValueError, match=match):
        analyze_semantic_evaluation(cohort, report)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda report: report.update(schema="semantic-evaluation-report/4"),
        lambda report: report.pop("semanticExecutionSchemaDigest"),
        lambda report: report.update(semanticExecutionSchemaDigest="0" * 64),
        lambda report: report.update(semanticExecutionFixtureDigest="0" * 64),
        lambda report: report.update(hostCatalogDigest="0" * 64),
        lambda report: report.update(catalogDigest="0" * 64),
    ],
)
def test_analysis_rejects_old_or_unbound_semantic_evaluation_reports(mutation):
    cohort = _cohort()
    report = _report(cohort)
    mutation(report)
    with pytest.raises(ValueError, match="semantic-evaluation-report/5|stale or missing"):
        analyze_semantic_evaluation(cohort, report)


def test_current_reports_compare_at_registered_seeds_and_reject_stale_identity():
    cohort = _cohort()
    reports, budgets = [], []
    for seed in (17, 29, 43):
        report = _report(cohort)
        report["seed"] = seed
        reports.append(report)
        budgets.append(
            {
                "schema": "semantic-experiment-budget/4",
                "seed": seed,
                "representation": "compact",
                "checkpointIdentity": None,
                "cohortDigest": cohort["digest"],
                "regime": "fixed-examples",
                "examples": 10,
                "generatedTokens": 100,
                "observedTokens": 200,
                "computeSeconds": 3.0,
            }
        )
    compared = compare_experiments(reports, budgets)
    assert compared["schema"] == "semantic-experiment-comparison/5"
    invalid = copy.deepcopy(reports)
    invalid[0]["semanticExecutionFixtureDigest"] = "0" * 64
    with pytest.raises(ValueError, match="stale or missing"):
        compare_experiments(invalid, budgets)


def test_analysis_marks_missing_behaviors_unknown_and_requires_model_identity():
    cohort = _cohort()
    report = _report(cohort)
    report.pop("behaviorMetadata")
    analysis = analyze_semantic_evaluation(cohort, report)
    assert all(
        family["behaviorMetadataStatus"] == "unknown" for family in analysis["familySummaries"]
    )
    report["origin"] = "fixed-weights"
    with pytest.raises(ValueError, match="checkpoint identity"):
        analyze_semantic_evaluation(cohort, report)


def test_analysis_keeps_real_runner_exception_as_failed_attempt_with_unknown_costs():
    cohort = _cohort()

    def runner(_request):
        raise RuntimeError("runner unavailable")

    report = evaluate_cohort(
        cohort, runner, codec=ByteCodec.byte_fallback(), seed=17, origin="controlled"
    )
    raw = report["cases"][0]
    assert "matches" not in raw and "hardGates" in raw and raw["hardGates"] == {}
    assert "firstSemanticInconsistency" not in raw
    analysis = analyze_semantic_evaluation(cohort, report)
    row = analysis["cases"][0]
    assert row["reportedWholeTaskSuccess"] is False
    assert row["wholeTaskSuccess"] is False and not row["successMeasurementComplete"]
    assert row["matches"] == {
        key: None for key in ("completion", "state", "effects", "callbacks", "cleanup")
    }
    assert row["hardGates"] == dict.fromkeys(HARD_GATES)
    assert row["costs"] == dict.fromkeys(
        ("computeSeconds", "peakMemoryBytes", "dispatches", "queueDepth", "leakedResources")
    )
    assert row["semanticConsistencyStatus"] == "unknown"
    assert "runner unavailable" in row["failure"]
    assert analysis["summary"]["successEvidenceCoverage"] == 0
    assert analysis["summary"]["caseSuccessRate"] == 0
    assert analysis["summary"]["successfulFamilies"] == 0
    assert analysis["summary"]["familySuccessRate"] == 0
    assert analysis["summary"]["hardGates"][HARD_GATES[0]]["missingCases"] == 3
    assert analysis["summary"]["hardGates"][HARD_GATES[0]]["measuredCases"] == 0
    assert analysis["costBreakdownByScope"][0]["costs"]["computeSeconds"]["count"] == 0
    assert analysis["costBreakdownByScope"][0]["costs"]["computeSeconds"]["missingCases"] == 3
    invalid = copy.deepcopy(report)
    invalid["cases"][0]["wholeTaskSuccess"] = True
    invalid["summary"]["correctTasks"] = 1
    with pytest.raises(ValueError, match="positive task success"):
        analyze_semantic_evaluation(cohort, invalid)
