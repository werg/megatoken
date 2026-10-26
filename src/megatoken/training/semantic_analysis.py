"""Read-only, family-aware summaries of frozen semantic evaluation reports.

This analysis is downstream of inference. It never changes evaluator outputs,
adds labels to requests, or returns data that should be fed to a model.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any

from megatoken.training.semantic_evaluation import HARD_GATES, OUTCOMES, SEEDS, verify_cohort

ANALYSIS_SCHEMA = "semantic-evaluation-analysis/1"
COST_FIELDS = (
    "computeSeconds",
    "peakMemoryBytes",
    "dispatches",
    "queueDepth",
    "leakedResources",
)
DEFAULT_SOURCE_LENGTH_BANDS = (
    {"name": "0-255", "minBytes": 0, "maxBytes": 255},
    {"name": "256-1023", "minBytes": 256, "maxBytes": 1023},
    {"name": "1024-4095", "minBytes": 1024, "maxBytes": 4095},
    {"name": "4096+", "minBytes": 4096, "maxBytes": None},
)


def _number(value, label):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{label} must be a finite nonnegative measurement")
    return float(value)


def _quantile(values, percentile):
    """Empirical nearest-rank quantile; no interpolation or fitted model."""
    ordered = sorted(values)
    if not ordered:
        return None
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[rank - 1]


def _distribution(values):
    values = [float(value) for value in values]
    return {
        "count": len(values),
        "p50": _quantile(values, 0.50),
        "p95": _quantile(values, 0.95),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def _source_bytes(case):
    request = case.get("request", {})
    programs = request.get("programs")
    if programs is None:
        program = request.get("program")
        programs = [program] if isinstance(program, dict) else []
    sources = {program.get("source") for program in programs if isinstance(program, dict)}
    sources.discard(None)
    if not sources or any(not isinstance(source, str) for source in sources):
        return None
    return sum(len(source.encode("utf-8")) for source in sources)


def _validate_bands(bands):
    if not isinstance(bands, list) or not bands:
        raise ValueError("sourceLengthBands must be a nonempty report declaration")
    normalized, expected_min = [], 0
    for index, band in enumerate(bands):
        if not isinstance(band, dict) or set(band) != {"name", "minBytes", "maxBytes"}:
            raise ValueError("Source length bands need name, minBytes and maxBytes")
        name, low, high = band["name"], band["minBytes"], band["maxBytes"]
        if not isinstance(name, str) or not name or type(low) is not int or low != expected_min:
            raise ValueError("Source length bands must be named contiguous integer ranges")
        if high is None:
            if index != len(bands) - 1:
                raise ValueError("Only the last source length band may be open-ended")
            expected_min = None
        else:
            if type(high) is not int or high < low:
                raise ValueError("Invalid source length band upper bound")
            expected_min = high + 1
        normalized.append({"name": name, "minBytes": low, "maxBytes": high})
    if normalized[-1]["maxBytes"] is not None:
        raise ValueError("Final source length band must cover all longer programs")
    return normalized


def _band_for(length, bands):
    if length is None:
        return "unknown"
    for band in bands:
        if length >= band["minBytes"] and (band["maxBytes"] is None or length <= band["maxBytes"]):
            return band["name"]
    raise ValueError("Source length is outside the declared bands")


def _behavior(report, case_id):
    """Use only explicitly supplied behavior metadata; never infer from source."""
    metadata = report.get("behaviorMetadata")
    if metadata is None:
        return None
    if not isinstance(metadata, dict):
        raise ValueError("behaviorMetadata must map case IDs to explicit labels")
    labels = metadata.get(case_id)
    if labels is None:
        return None
    if not isinstance(labels, list) or any(
        not isinstance(label, str) or not label for label in labels
    ):
        raise ValueError("Behavior metadata labels must be nonempty strings")
    return sorted(set(labels))


def analyze_semantic_evaluation(cohort: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    """Validate matched frozen inputs and emit a distinct descriptive report."""
    verify_cohort(cohort)
    if not isinstance(report, dict) or report.get("schema") != "semantic-evaluation-report/4":
        raise ValueError("Expected a semantic-evaluation-report/4 input")
    if report.get("cohortDigest") != cohort["digest"] or report.get("cohort") != cohort["name"]:
        raise ValueError("Evaluation report does not match this frozen cohort")
    if report.get("origin") not in {"controlled", "fixed-weights", "trained"}:
        raise ValueError("Evaluation origin is missing or unsupported")
    if report.get("seed") not in SEEDS:
        raise ValueError("Evaluation seed is not registered")
    if report.get("representation") != "compact":
        raise ValueError("Evaluation representation is missing or unsupported")
    checkpoint = report.get("checkpointIdentity")
    if report["origin"] == "trained" or report["origin"] == "fixed-weights":
        if (
            not isinstance(checkpoint, str)
            or len(checkpoint) != 64
            or any(c not in "0123456789abcdef" for c in checkpoint)
        ):
            raise ValueError("Model-origin report needs a lowercase SHA-256 checkpoint identity")
    elif checkpoint is not None and (
        not isinstance(checkpoint, str)
        or len(checkpoint) != 64
        or any(c not in "0123456789abcdef" for c in checkpoint)
    ):
        raise ValueError("Controlled checkpoint identity must be null or a SHA-256 digest")
    if (
        report["origin"] in {"controlled", "fixed-weights"}
        and report.get("learnedSuccess") is not None
    ):
        raise ValueError("Controlled and fixed-weight evidence cannot claim learnedSuccess")
    if report["origin"] == "trained":
        _number(report.get("learnedSuccess"), "learnedSuccess")

    frozen = cohort["cases"]
    rows = report.get("cases")
    if not isinstance(rows, list) or len(rows) != len(frozen):
        raise ValueError("Evaluation cases must exactly match the frozen cohort inventory")
    expected = {case["id"]: case for case in frozen}
    observed = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("caseId"), str):
            raise ValueError("Evaluation row has no case ID")
        case_id = row["caseId"]
        if case_id in observed or case_id not in expected:
            raise ValueError("Evaluation case IDs are duplicated or do not match the cohort")
        if row.get("family") != expected[case_id]["family"]:
            raise ValueError("Evaluation case family differs from the frozen cohort")
        if type(row.get("wholeTaskSuccess")) is not bool or type(row.get("traceValid")) is not bool:
            raise ValueError("Evaluation success and trace validity must be measured booleans")
        matches = row.get("matches", {})
        if (
            not isinstance(matches, dict)
            or set(matches) - set(OUTCOMES)
            or any(type(value) is not bool for value in matches.values())
        ):
            raise ValueError("Per-outcome comparison results must be measured booleans")
        gates = row.get("hardGates")
        if (
            not isinstance(gates, dict)
            or set(gates) - set(HARD_GATES)
            or any(type(value) is not int or value < 0 for value in gates.values())
        ):
            raise ValueError("Hard-gate values must be measured nonnegative counts")
        costs = row.get("costs")
        if not isinstance(costs, dict) or set(costs) - set(COST_FIELDS):
            raise ValueError("Cost values must use the known measured cost fields")
        for field, value in costs.items():
            _number(value, f"costs.{field}")
        complete = (
            set(matches) == set(OUTCOMES)
            and set(gates) == set(HARD_GATES)
            and set(costs) == set(COST_FIELDS)
        )
        expected_success = (
            (
                all(matches.values())
                and row["traceValid"]
                and not any(gates.values())
                and costs["leakedResources"] == 0
            )
            if complete
            else None
        )
        if not complete and row["wholeTaskSuccess"]:
            raise ValueError(
                "A positive task success cannot be claimed with incomplete measurements"
            )
        if complete and row["wholeTaskSuccess"] != expected_success:
            raise ValueError(
                "Whole-task success contradicts measured matches, trace, gates or cleanup"
            )
        _number(row.get("latencySeconds"), "latencySeconds")
        scope = row.get("costScope")
        if scope is not None and not isinstance(scope, dict):
            raise ValueError("costScope must be an explicit object or null")
        inconsistency_present = "firstSemanticInconsistency" in row
        inconsistency = row.get("firstSemanticInconsistency")
        if inconsistency is not None and not isinstance(inconsistency, dict):
            raise ValueError("firstSemanticInconsistency must retain its exact receipt")
        behavior = _behavior(report, case_id)
        observed[case_id] = {
            **row,
            "reportedWholeTaskSuccess": row["wholeTaskSuccess"],
            "wholeTaskSuccess": row["wholeTaskSuccess"],
            "successMeasurementComplete": complete,
            "matches": {key: matches.get(key) for key in OUTCOMES},
            "outcomeMatchCoverage": sorted(matches),
            "hardGates": {key: gates.get(key) for key in HARD_GATES},
            "hardGateCoverage": sorted(gates),
            "costs": {key: costs.get(key) for key in COST_FIELDS},
            "costCoverage": sorted(costs),
            "firstSemanticInconsistency": inconsistency,
            "semanticConsistencyStatus": "provided" if inconsistency_present else "unknown",
            "sourceLengthBytes": _source_bytes(expected[case_id]),
            "sourceLengthBand": None,
            "behaviorMetadata": behavior,
            "behaviorMetadataStatus": "provided" if behavior is not None else "unknown",
        }

    bands = _validate_bands(report.get("sourceLengthBands", list(DEFAULT_SOURCE_LENGTH_BANDS)))
    for row in observed.values():
        row["sourceLengthBand"] = _band_for(row["sourceLengthBytes"], bands)

    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in observed.values():
        by_family[row["family"]].append(row)
    family_summaries = []
    for family, family_rows in sorted(by_family.items()):
        complete_rows = [row for row in family_rows if row["successMeasurementComplete"]]
        success_count = sum(row["wholeTaskSuccess"] is True for row in family_rows)
        trace_count = sum(row["traceValid"] for row in family_rows)
        explicit_behaviors = sorted(
            {label for row in family_rows for label in (row["behaviorMetadata"] or [])}
        )
        family_summaries.append(
            {
                "family": family,
                "caseCount": len(family_rows),
                "successfulCases": success_count,
                "traceValidCases": trace_count,
                "familySuccess": False
                if any(row["wholeTaskSuccess"] is False for row in family_rows)
                else True
                if len(complete_rows) == len(family_rows)
                else None,
                "successMeasurementComplete": len(complete_rows) == len(family_rows),
                "measuredSuccessCases": len(complete_rows),
                "familyTraceValid": trace_count == len(family_rows),
                "taskSuccessRate": success_count / len(family_rows),
                "traceValidRate": trace_count / len(family_rows),
                "behaviors": explicit_behaviors if explicit_behaviors else None,
                "behaviorMetadataStatus": "provided" if explicit_behaviors else "unknown",
                "failureCounts": dict(
                    Counter(row["failure"] for row in family_rows if row.get("failure"))
                ),
                "hardGates": {
                    key: {
                        "total": sum(row["hardGates"][key] or 0 for row in family_rows),
                        "measuredCases": sum(key in row["hardGateCoverage"] for row in family_rows),
                        "missingCases": sum(
                            key not in row["hardGateCoverage"] for row in family_rows
                        ),
                    }
                    for key in HARD_GATES
                },
                "firstSemanticInconsistency": [
                    {"caseId": row["caseId"], "receipt": row["firstSemanticInconsistency"]}
                    for row in family_rows
                    if row.get("firstSemanticInconsistency") is not None
                ],
                "failureEvidence": [
                    {"caseId": row["caseId"], "evidence": row.get("failureEvidence")}
                    for row in family_rows
                    if row.get("failureEvidence") is not None
                ],
            }
        )

    scopes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in observed.values():
        scope = row.get("costScope")
        scope_key = "unknown" if scope is None else repr(sorted(scope.items()))
        scopes[scope_key].append(row)
    cost_breakdown = []
    for scope_key, scope_rows in sorted(scopes.items()):
        cost_breakdown.append(
            {
                "scope": None if scope_key == "unknown" else dict(scope_rows[0]["costScope"]),
                "caseCount": len(scope_rows),
                "latencySeconds": _distribution([row["latencySeconds"] for row in scope_rows]),
                "costs": {
                    field: {
                        **_distribution(
                            [
                                row["costs"][field]
                                for row in scope_rows
                                if row["costs"][field] is not None
                            ]
                        ),
                        "missingCases": sum(row["costs"][field] is None for row in scope_rows),
                    }
                    for field in COST_FIELDS
                },
            }
        )

    gate_totals = {
        key: {
            "total": sum((row["hardGates"][key] or 0) for row in observed.values()),
            "measuredCases": sum(key in row["hardGateCoverage"] for row in observed.values()),
            "missingCases": sum(key not in row["hardGateCoverage"] for row in observed.values()),
        }
        for key in HARD_GATES
    }
    case_successes = sum(row["wholeTaskSuccess"] is True for row in observed.values())
    measured_success_rows = [row for row in observed.values() if row["successMeasurementComplete"]]
    measured_families = [item for item in family_summaries if item["successMeasurementComplete"]]
    family_successes = sum(item["familySuccess"] is True for item in measured_families)
    failure_counts = Counter(row["failure"] for row in observed.values() if row.get("failure"))
    summary = report.get("summary")
    if not isinstance(summary, dict) or summary.get("tasks") != len(rows):
        raise ValueError("Evaluation summary task count does not match its case rows")
    reported_successes = sum(row["reportedWholeTaskSuccess"] for row in observed.values())
    if summary.get("correctTasks") != reported_successes:
        raise ValueError("Evaluation summary success count does not match measured case rows")
    if report["origin"] == "trained":
        measured = _number(summary.get("wholeTaskSuccess"), "summary.wholeTaskSuccess")
        if measured > 1 or report["learnedSuccess"] != measured:
            raise ValueError("learnedSuccess must equal the trained report's measured task rate")

    return {
        "schema": ANALYSIS_SCHEMA,
        "input": {"cohortDigest": cohort["digest"], "evaluationReportSchema": report["schema"]},
        "evaluation": {
            "cohort": report["cohort"],
            "origin": report["origin"],
            "checkpointIdentity": checkpoint,
            "seed": report["seed"],
            "representation": report["representation"],
            "learnedSuccess": report.get("learnedSuccess"),
        },
        "sourceLengthBands": bands,
        "summary": {
            "families": len(family_summaries),
            "cases": len(rows),
            "successfulCases": case_successes,
            "measuredSuccessCases": len(measured_success_rows),
            "successEvidenceCoverage": len(measured_success_rows) / len(rows),
            "caseSuccessRate": case_successes / len(rows),
            "successfulFamilies": family_successes,
            "measuredFamilies": len(measured_families),
            "familySuccessEvidenceCoverage": len(measured_families) / len(family_summaries),
            "familySuccessRate": family_successes / len(family_summaries),
            "traceValidCases": sum(row["traceValid"] for row in observed.values()),
            "hardGates": gate_totals,
            "failureCounts": dict(failure_counts),
            "latencySeconds": _distribution([row["latencySeconds"] for row in observed.values()]),
        },
        "familySummaries": family_summaries,
        "costBreakdownByScope": cost_breakdown,
        "cases": [observed[case["id"]] for case in frozen],
        "statisticsNote": "Family-level success is primary. Case-level rates describe this frozen cohort only; repeated cases within a family are correlated and do not support population generalization.",
        "analysisLabelsForInference": False,
    }
