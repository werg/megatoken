# Optimizer target-exposure controls — 22 November 2026

All logs apply only to the exact candidates below. They include fixture boundaries
and failure output; no skipped control is counted as qualified.

| Candidate / run | Result |
| --- | --- |
| 2e6630f / [44989704524](https://github.com/werg/megatoken/actions/runs/44989704524) | 38 passed, 1 failed, 3 skipped. Real prepared denominators are Mapping objects, not necessarily dicts. The three skips lacked the separately provisioned pinned tokenizer. |
| a18dd3e / [44990229550](https://github.com/werg/megatoken/actions/runs/44990229550) | Workflow failed before creating jobs: runner context was used at job-env scope. Fixed by using the workspace context. |
| b9de0ac / [44990295933](https://github.com/werg/megatoken/actions/runs/44990295933) | Exact tokenizer provisioned; 42 passed, 1 failed, zero skips. Actual update and fresh resume assertions passed; an obsolete string replacement in the forged-state negative test failed to install its exception handler. |
| d737911 / [44990559146](https://github.com/werg/megatoken/actions/runs/44990559146) | Ruff rejected an unused variable in the new dropped-census test before test execution. |
| 910d45b / [44990761001](https://github.com/werg/megatoken/actions/runs/44990761001) | Ruff/formatting and all 57 controls passed, zero skips. Exact tested formatted files are published in this increment. |

The parent-digest control rejects first-step history [3] versus a resumed history
built from [4,4], even when both have monotonic totals. A separately rehashed
forged parent cannot excuse dropping an original scored source. Counts of unique
targets remain separate from repeated scored occurrences and loss denominators.

The complete successful log and earlier executable failure logs are retained
here. Large model/artifact downloads remain in the workflow artifact retention
scope. Full corpus optimization and learned quality require their own runs.
