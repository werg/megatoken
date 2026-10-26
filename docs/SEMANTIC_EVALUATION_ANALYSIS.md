# Semantic evaluation analysis

`semantic-evaluation-analysis/1` is a read-only summary over one frozen
`semantic-evaluation-cohort/4` and its matching
`semantic-evaluation-report/5`. It verifies the frozen cohort digest, current shared
execution-schema and fixture digests, host catalog identity, seed, origin/checkpoint identity, and an exact one-to-one match of
case IDs and family identities before summarizing measurements. It does not
modify the evaluator's v5 report schema or feed any analysis labels into inference.

Run it after the ordinary evaluator has written its report:

```sh
python scripts/analyze_semantic_evaluation.py \
  --cohort /path/to/frozen-cohort.json \
  --report /path/to/evaluation-report.json \
  --output /path/to/evaluation-analysis.json
```

Family-level success is the primary summary so multiple cases in one family do
not inflate the independent-family count. The report also retains per-case
success, trace validity, hard-gate counts, failure categories, the exact first
semantic inconsistency, and failure evidence. Behavior labels are copied only
when the input evaluation report supplies explicit `behaviorMetadata` keyed by
case ID; absent metadata is `unknown`, and the analyzer never infers behavior
from source text or outcomes.

Every frozen case remains in the attempt denominator. A runner exception is a
measured failed attempt (`wholeTaskSuccess: false`), even when its safety-gate
or cost measurements are unavailable. Missing hard-gate, cost, outcome-match,
or consistency fields are `null` with explicit coverage counts; they are never
filled with zero. An incomplete row cannot claim positive task success. Cost
percentiles use only measured values and report missing-case counts.

Source lengths are measured from the frozen request's unique program sources
and grouped by the byte bands recorded in `sourceLengthBands` in the analysis
report. Latency and each measured cost are reported as empirical nearest-rank
p50/p95/min/max by the runner's declared `costScope`; unknown scope remains
unknown. This gives descriptive measurements for the observed cases, not
population estimates. Cases within a family are correlated, so case-level
rates must not be presented as independent generalization evidence. Controlled
and fixed-weight origins retain `learnedSuccess: null`.

Run the focused checks with:

```sh
pytest -q tests/test_semantic_analysis.py
```
