# Unified native bootstrap, close, launch, and router regression

Exact final source commit: `2217b6ba8534b06f0670f4dfecad7d0c3c3f807e`  
Exact source tree before this evidence update: `b02c354e0c5bc804705aafc9b2d791c4f0a925f8`  
The published snapshot is based on native `6f5259fcba8fd68866d628424b3f7df91615972b`, which includes its parent history: `7f6245ce372b52416a281a76f3e1d9b0eb435666`, training-output/tokenizer overlap guard `2c2687271fa7a61dff2485710818f99a5ca66383`, and terminal-close `6ef4794048a4d2ab0e732ccfd61c5cc3e7c3c822`. It adds router receipt validation source from `19d011247e9f1883db632121ca6d8c591abc875d`. The causal child-bootstrap implementation and focused controls were integrated from the earlier `c1dde8a`/`7cf0fdd`/`2843e9f` candidate, with the response/RNG/fence corrections from `2b52dcc8ca8d04dc992d771fc938f9f3d629a475` applied directly in this tree. The terminal-cancel fixture from `6f5259` is retained.

The final implementation uses only `durable-bootstrap-child` / `recover-bootstrap-child`, bound to the exact preceding child-create commit request and receipt. It removes the alternate bootstrap API. The full preserved controls from the alternate implementation now pass the causal create request and all three predecessor fields. The unified tree also retains the exact close/recovery protocol, the training-output/tokenizer overlap guard, and the router receipt consistency checks.

## Discovered bootstrap response gap

On the earlier frozen causal candidate, a valid completed bootstrap result was changed in the journal response to set the child parent to `forged-parent`. After recomputing the stored response and journal hashes while preserving the complete native postcheckpoint, a duplicate `durable_bootstrap_child()` call returned the forged parent. The raw reproduction is retained as `pre-fix-reproduction.log`.

The corrected path binds the full returned child description to the create result (allowing only bootstrap readiness and the permitted active/waiting status change), verifies the live child description and its two observed bootstrap publications, checks the appended bootstrap operation and its exact pre-state RNG, and checks direct bridge fencing before reading a completed slot. The adversarial tests cover altered parent, execution invocation, lifecycle status, random state, and fenced duplicate reads.

## Unified qualification

Command:

```sh
PYTHONPATH=src:. /workspace/scratch/05d92519587f/native-recovery-venv/bin/python -m pytest -q \
  tests/test_semantic_commit_recovery_v8.py \
  tests/test_semantic_request_journal_v8.py \
  tests/test_semantic_bootstrap_recovery_v8.py \
  tests/test_semantic_close_recovery_v8.py \
  tests/test_semantic_launch_v8.py \
  tests/test_semantic_v8_position_receipts.py \
  --junitxml=docs/evidence/native-causal-bootstrap-close-20261122/junit.xml
```

Result on the exact final source tree: 145 tests passed, 0 failed, 0 errors, and 0 skipped in 238.73 seconds. The six suites contributed 42 commit-recovery, 12 request-journal, 13 bootstrap, 7 terminal-close, 16 launch, and 55 router-position receipt cases. Runtime was Python 3.12.14, pytest 9.2.0, and torch 2.14.1+cpu. The repository declares pytest below 9; this local run used pytest 9.2.0 and therefore does not substitute for the locked hosted environment. Complete output and JUnit are in `raw.log` and `junit.xml`. The pre-fixture 145-case proof is preserved separately in `prior-145case-7f-tree.raw.log` and `prior-145case-7f-tree.junit.xml`.

This is a native unit and integration cohort on the exact tree above. It does not claim distributed lease takeover or recovery from unknown remote outcomes, and it is not the paired producer hosted gate. It does not claim a hosted whole-driver result; the paired producer/native hosted gate remains a separate qualification.
