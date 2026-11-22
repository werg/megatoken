# Unified native bootstrap, close, launch, and router regression

Exact source commit used by the full run: `bfe957b2d07daded368edae6a83b199f923fee20`  
Exact source tree used by the full run: `a98960e129e98d2c98e21b6f41a1b5b7a3c02f61`  
This source increment is based on native `cf9f2086e4f668325566c298e4d952d7413b877f`, which preserves router validation/evidence and adds two terminal-close controls. Its history includes router source `19d011247e9f1883db632121ca6d8c591abc875d`, training-output/tokenizer overlap guard `2c2687271fa7a61dff2485710818f99a5ca66383`, and terminal-close work `6ef4794048a4d2ab0e732ccfd61c5cc3e7c3c822`. The causal child-bootstrap implementation and focused controls were integrated from the earlier `c1dde8a`/`7cf0fdd`/`2843e9f` candidate, with response/RNG/fence corrections from `2b52dcc8ca8d04dc992d771fc938f9f3d629a475`.

The final implementation uses only `durable-bootstrap-child` / `recover-bootstrap-child`, bound to the exact preceding child-create commit request and receipt. It removes the alternate bootstrap API. The full preserved controls from the alternate implementation now pass the causal create request and all three predecessor fields. The unified tree also retains the exact close/recovery protocol, the training-output/tokenizer overlap guard, and the router receipt consistency checks.

## Discovered bootstrap response gap

On the earlier frozen causal candidate, a valid completed bootstrap result was changed in the journal response to set the child parent to `forged-parent`. After recomputing the stored response and journal hashes while preserving the complete native postcheckpoint, a duplicate `durable_bootstrap_child()` call returned the forged parent. The raw reproduction is retained as `pre-fix-reproduction.log`.

The corrected path binds the full returned child description to the create result (allowing only bootstrap readiness and the permitted active/waiting status change), verifies the live child description and its two observed bootstrap publications, checks the appended bootstrap operation and its exact pre-state RNG, and checks direct bridge fencing before reading a completed slot. The adversarial tests cover altered parent, execution invocation, lifecycle status, random state, and fenced duplicate reads.

## Discovered raw-operation alias bypass

The previous published candidate blocked the underscore and hyphen spellings independently before dispatch normalization, but accepted the mixed spellings `bootstrap_orchestration-child` and `bootstrap-orchestration_child`. On the real runtime after a durably acknowledged child-create commit, both requests passed guarded JSONL dispatch and appended an unjournaled bootstrap operation. The exact pre-fix output is retained in `bootstrap-alias-pre-fix.log`.

The authorization check now compares the separator-normalized operation name while retaining the original operation string and request bytes for durable supersession. All four spellings are tested against the guarded real adapter; each is refused before native checkpoint mutation, and the completed create request remains the latest journaled slot.

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

Result on the exact source tree above: 151 tests passed, 0 failed, 0 errors, and 0 skipped in 269.15 seconds. The six suites contributed 42 commit-recovery, 12 request-journal, 17 bootstrap (including the four raw-operation spelling controls), 9 terminal-close, 16 launch, and 55 router-position receipt cases. Runtime was Python 3.12.14, pytest 9.2.0, and torch 2.14.1+cpu. The repository declares pytest below 9; this local run used pytest 9.2.0 and therefore does not substitute for the locked hosted environment. Complete output and JUnit are in `raw.log` and `junit.xml`. The prior 145-case proof is preserved separately in `prior-145case-7f-tree.raw.log` and `prior-145case-7f-tree.junit.xml`.

This is a native unit and integration cohort on the exact source tree above. It does not claim distributed lease takeover or recovery from unknown remote outcomes, and it is not the paired producer hosted gate. It does not claim a hosted whole-driver result; the paired producer/native hosted gate remains a separate qualification.
