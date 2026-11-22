# Native bootstrap unification evidence

The two `prior56-*` logs were copied from the evidence bundle in external
candidate commit `56cce3b80423b0cb70c3a096d867561066e3a211`. That candidate
contains the fixes; the logs describe earlier pre-fix behavior, not executions
of commit 56 or this tree. The alias log names earlier published native commit
`b504228288e2935c9e021da23896fc82babc6836`; the response log itself does not
identify its exact source revision. Treat these as preserved historical
reproduction output, not as qualified evidence for the current tree.

`prior56-pre-fix-response-reproduction.log` records a rehashed saved response
whose child parent became `forged-parent` while the postcheckpoint stayed the
same. `prior56-pre-fix-alias-reproduction.log` records the two mixed underscore
and hyphen spellings that passed the guarded raw dispatch before normalization.
The corrected union must independently pass the retained response and all-four
alias controls before it is qualified.

## Local focused native run

The focused command ran `test_semantic_request_journal_v8.py`,
`test_semantic_bootstrap_recovery_v8.py`, and
`test_semantic_commit_recovery_v8.py` with the shared Python 3.12 environment.
It collected 85 test cases: 83 passed and two failed only because test regexes
expected different refusal messages than the implementation emitted. The exact
first run is retained in `focused-raw.log` and `focused-junit.xml`. Rerunning
only those two affected cases after correcting the expected messages passed
both (70 deselected); that run is retained in `corrected-cases-raw.log` and
`corrected-cases-junit.xml`. The first run used the source after moving the
exact supersession assertion ahead of predecessor inspection. Ruff later made
formatting-only changes, and I did not record an interim source hash, so the
first run cannot be mapped to an exact committed tree. Together these runs
cover all 85 cases, but they are not a single all-green run of the three-file
cohort.

Ruff lint and formatting checks pass on the six touched Python files. This is a
local native runtime cohort, not the paired hosted producer/native gate.

The full six-file native target population is 165 tests. The exact collection
list, produced without executing tests, is in `six-file-collection.log`.
